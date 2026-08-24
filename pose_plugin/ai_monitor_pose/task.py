"""AI Monitor VisionTask 适配器（第 5.8 节）。

唯一动态入口：ai_monitor_pose.task.FallDetectionTask。
构造只解析配置与轻量字段，不加载模型/不 CUDA/不启线程进程/不触达 Worker；
Runtime 在首个有效上下文经 runtime_factory 惰性 acquire（非阻塞、不等待模型 ready）。
run() 非阻塞：poll 结果 +（无 sink 兼容模式）映射 transition；不等待/不执行 YOLO。
"""
from __future__ import annotations

import math
import time
import uuid
from collections import OrderedDict

from vision.tasks import VisionTask

from .config import FallTaskConfig
from .contracts import FrameRequestMetaV1
from .errors import TaskBindingError
from .event_mapper import map_transition_to_vision_event
from .host_protocols import ClockProtocol, EventSinkProtocol, RuntimeFactoryProtocol

_NS_PER_S = 1_000_000_000


class _DefaultClock:
    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def unix_ns(self) -> int:
        return time.time_ns()


class FallDetectionTask(VisionTask):
    name = "fall_detection"

    def __init__(
        self,
        config: dict | None = None,
        full_config: dict | None = None,
        *,
        runtime_factory: RuntimeFactoryProtocol | None = None,
        clock: ClockProtocol | None = None,
        event_sink: EventSinkProtocol | None = None,
        process_factory: object | None = None,
        **_ignored: object,
    ) -> None:
        super().__init__(config or {})
        self._cfg = FallTaskConfig.from_mapping(config or {})
        self.runtime_key = self._cfg.runtime_key
        self.camera_session_id = uuid.uuid4().hex
        self._runtime_factory = runtime_factory
        self._clock: ClockProtocol = clock or _DefaultClock()
        self._event_sink = event_sink
        self._process_factory = process_factory
        self._camera_id: str | None = None
        self._runtime = None
        self._lease = None
        self._closed = False
        self._next_submit_ns = int(self._clock.monotonic_ns())
        # 判重 FIFO：compatibility event_id 有界缓存，超出上限丢弃最旧（防无界增长）
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._seen_limit = 4096
        self._interval_ns = int(_NS_PER_S / max(1, self._cfg.scheduler.target_fps))
        # 姿态叠加缓存:最新一条合法 INFERENCE_RESULT 及其宿主侧接收时刻(单调时钟)
        self._last_result: dict | None = None
        self._last_seen_host_ns: int = 0

    # 构造器不得加载模型 / 启动 Worker；本处仅轻量字段

    def should_run(self, frame_id: int, context) -> bool:
        if self._closed or not self.enabled:
            return False
        if self._camera_id is None:
            self._camera_id = context.camera_id
            self._lease = self._acquire_runtime()
        if context.camera_id != self._camera_id:
            raise TaskBindingError(f"Task 绑定 {self._camera_id}，拒绝 {context.camera_id}")
        runtime = self._lease
        now = self._clock.monotonic_ns()
        due_interval = frame_id % max(1, self.interval) == 0
        due_time = now >= self._next_submit_ns
        has = runtime.has_latest_result_or_health_change(self._camera_id, self.camera_session_id)
        if self._event_sink is None:
            has = has or runtime.has_unseen_compatibility_event(self._camera_id, self.camera_session_id)
        # 缓存仍新鲜(TTL 内)时也要在每个 context 上重挂载 analytics,避免预览帧之间 overlay 闪断
        fresh = self._overlay_fresh()
        return bool(has or fresh or (due_interval and due_time))

    def run(self, frame, context) -> list:
        runtime = self._lease
        bundle = runtime.poll(self._camera_id, self.camera_session_id)
        events: list = []
        # 仅无 sink 的兼容/测试模式，读 durable spool 的可观测副本（不删除、不承担生产投递）
        if self._event_sink is None:
            for tr in getattr(bundle, "compatibility_events", ()) or ():
                if tr.event_id in self._seen:
                    continue
                self._seen[tr.event_id] = None
                if len(self._seen) > self._seen_limit:
                    self._seen.popitem(last=False)  # FIFO：丢弃最旧
                events.append(map_transition_to_vision_event(tr))
        has_new = self._ingest_result(bundle)
        # 提交仍受 target_fps 节流:新结果到来时立即提交,否则按步进节奏,避免 fresh 缓存导致逐帧 offer
        now = self._clock.monotonic_ns()
        if has_new or now >= self._next_submit_ns:
            self._offer(frame, context)
            self._next_submit_ns = now + self._interval_ns
        self._mount_analytics(context)
        return events

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._last_result = None
        self._last_seen_host_ns = 0
        if self._lease is not None:
            if self._camera_id is not None:
                self._lease.unregister_camera(self._camera_id, self.camera_session_id)
            self._lease.release()
            self._lease = None

    def _overlay_fresh(self) -> bool:
        """TTL 内尚有可重挂的姿态叠加缓存。"""
        if self._last_result is None or self._last_seen_host_ns == 0:
            return False
        age_ms = (self._clock.monotonic_ns() - self._last_seen_host_ns) / _NS_PER_S * 1000
        return age_ms < self._cfg.runtime.overlay_ttl_ms

    def _ingest_result(self, bundle) -> bool:
        """接受一条新姿态结果。返回是否出现新结果;仅接受同 session、source 帧严格递增的结果。"""
        result = getattr(bundle, "latest_result", None)
        if not isinstance(result, dict):
            return False
        if result.get("camera_session_id") != self.camera_session_id:
            return False
        src = result.get("source_frame_id")
        prev = (self._last_result or {}).get("source_frame_id")
        if isinstance(src, int) and isinstance(prev, int) and src <= prev:
            return False
        self._last_result = result
        self._last_seen_host_ns = self._clock.monotonic_ns()
        return True

    def _build_tracks(self, result: dict) -> list:
        """把 FallResultV1 的 track 列表映射为 analytics wire 的 track 结构(bbox xyxy→xywh)。"""
        out = []
        for t in result.get("tracks") or []:
            if not isinstance(t, dict):
                continue
            bbox_xywh = None
            bb = t.get("bbox_xyxy")
            if isinstance(bb, (list, tuple)) and len(bb) >= 4:
                try:
                    x1, y1, x2, y2 = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
                    bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
                except (TypeError, ValueError):
                    bbox_xywh = None
            keypoints = []
            for kp in t.get("keypoints_coco17") or []:
                if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                    keypoints.append([float(kp[0]), float(kp[1])])
            out.append({
                "pose_track_id": t.get("pose_track_id"),
                "state": t.get("state"),
                "score": t.get("detection_score", t.get("rule_score")),
                "bbox": bbox_xywh,
                "keypoints": keypoints,
            })
        return out

    def _mount_analytics(self, context) -> None:
        """把最新姿态结果挂到当前 context,供预览叠加;超 TTL 即清除。"""
        result = self._last_result
        if result is None or self._camera_id is None:
            return
        if context.camera_id != self._camera_id or context.camera_id != result.get("camera_id"):
            return
        now = self._clock.monotonic_ns()
        if self._last_seen_host_ns == 0:
            return
        result_age_ms = max(
            0.0, (now - self._last_seen_host_ns) / _NS_PER_S * 1000
        )
        overlay_expires_in_ms = max(
            0, self._cfg.runtime.overlay_ttl_ms - math.ceil(result_age_ms)
        )
        if overlay_expires_in_ms <= 0:
            self._last_result = None
            self._last_seen_host_ns = 0
            return
        health = None
        rt = getattr(self._lease, "runtime", None)
        if rt is not None:
            health = getattr(rt, "state", None)
        if health is None:
            health = getattr(self._lease, "state", None)
        context.analytics["fall_detection"] = {
            "schema_version": 1,
            "camera_session_id": self.camera_session_id,
            "attached_to_frame_id": context.frame_id,
            "source_frame_id": result.get("source_frame_id"),
            "source_width": result.get("source_width"),
            "source_height": result.get("source_height"),
            "coordinate_space": "source_pixels",
            "result_age_ms": round(result_age_ms, 1),
            "overlay_expires_in_ms": overlay_expires_in_ms,
            "worker_end_to_end_ms": result.get("end_to_end_ms"),
            "health": health,
            "tracks": self._build_tracks(result),
        }

    # --- 内部 ---
    def _acquire_runtime(self):
        if self._runtime_factory is None:
            raise RuntimeError("fall_detection 需要 runtime_factory（生产由 PoseRuntimeRegistry 注入）")
        return self._runtime_factory.acquire(
            runtime_key=self.runtime_key,
            config=self._cfg.runtime,  # RuntimeConfig（含 max_frame_width/height 等），PoseRuntime 契约
            event_sink=self._event_sink,
            process_factory=self._process_factory,
            heartbeat_interval_s=self._cfg.worker.heartbeat_interval_s,
            heartbeat_timeout_s=self._cfg.worker.heartbeat_timeout_s,
            mode=self._cfg.mode,
        )

    def _offer(self, frame, context) -> None:
        now = self._clock.monotonic_ns()
        meta = FrameRequestMetaV1(
            schema_version=1,
            request_id=uuid.uuid4().hex,
            camera_id=self._camera_id,
            camera_session_id=self.camera_session_id,
            frame_id=context.frame_id,
            observed_at_unix_ns=self._clock.unix_ns(),
            observed_at_monotonic_ns=now,
            deadline_monotonic_ns=now + self._cfg.runtime.max_result_age_ms * 1_000_000,
            config_revision=self._cfg.config_revision,
        )
        try:
            self._lease.offer_frame(frame, meta)
        except Exception:
            pass
        self._next_submit_ns = now + self._interval_ns
