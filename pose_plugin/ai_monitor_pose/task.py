"""AI Monitor VisionTask 适配器（第 5.8 节）。

唯一动态入口：ai_monitor_pose.task.FallDetectionTask。
构造只解析配置与轻量字段，不加载模型/不 CUDA/不启线程进程/不触达 Worker；
Runtime 在首个有效上下文经 runtime_factory 惰性 acquire（非阻塞、不等待模型 ready）。
run() 非阻塞：poll 结果 +（无 sink 兼容模式）映射 transition；不等待/不执行 YOLO。
"""
from __future__ import annotations

import time
import uuid


def _trace(*parts) -> None:
    try:
        import time as _t
        line = _t.strftime("%H:%M:%S") + " " + " ".join(str(p) for p in parts) + "\n"
        with open(r"D:\ai-monitor-1.1.0\融合实施_work\pose_trace.log", "a") as f:
            f.write(line)
    except Exception:
        pass

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
        self._seen: set[str] = set()
        self._interval_ns = int(_NS_PER_S / max(1, self._cfg.scheduler.target_fps))

    # 构造器不得加载模型 / 启动 Worker；本处仅轻量字段

    def should_run(self, frame_id: int, context) -> bool:
        _trace('T.should_run cam=%s frame=%s enabled=%s closed=%s lease=%s' % (context.camera_id, frame_id, self.enabled, self._closed, self._lease))
        if self._closed or not self.enabled:
            _trace('T.should_run->False disabled')
            return False
        if self._camera_id is None:
            self._camera_id = context.camera_id
            self._lease = self._acquire_runtime()
            _trace('T.should_run acquired lease=%s' % (self._lease,))
        if context.camera_id != self._camera_id:
            raise TaskBindingError(f"Task 绑定 {self._camera_id}，拒绝 {context.camera_id}")
        runtime = self._lease
        now = self._clock.monotonic_ns()
        due_interval = frame_id % max(1, self.interval) == 0
        due_time = now >= self._next_submit_ns
        has = runtime.has_latest_result_or_health_change(self._camera_id, self.camera_session_id)
        if self._event_sink is None:
            has = has or runtime.has_unseen_compatibility_event(self._camera_id, self.camera_session_id)
        _trace('T.should_run->%s due_iv=%s due_t=%s has=%s' % (bool(has or (due_interval and due_time)), due_interval, due_time, has))
        return bool(has or (due_interval and due_time))

    def run(self, frame, context) -> list:
        runtime = self._lease
        bundle = runtime.poll(self._camera_id, self.camera_session_id)
        events: list = []
        # 仅无 sink 的兼容/测试模式，读 durable spool 的可观测副本（不删除、不承担生产投递）
        if self._event_sink is None:
            for tr in getattr(bundle, "compatibility_events", ()) or ():
                if tr.event_id in self._seen:
                    continue
                self._seen.add(tr.event_id)
                events.append(map_transition_to_vision_event(tr))
        self._offer(frame, context)
        return events

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._lease is not None:
            if self._camera_id is not None:
                self._lease.unregister_camera(self._camera_id, self.camera_session_id)
            self._lease.release()
            self._lease = None

    # --- 内部 ---
    def _acquire_runtime(self):
        if self._runtime_factory is None:
            raise RuntimeError("fall_detection 需要 runtime_factory（生产由 PoseRuntimeRegistry 注入）")
        return self._runtime_factory.acquire(
            runtime_key=self.runtime_key,
            config=self._cfg.runtime,  # RuntimeConfig（含 max_frame_width/height 等），PoseRuntime 契约
            event_sink=self._event_sink,
            process_factory=self._process_factory,
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
            _r = self._lease.offer_frame(frame, meta)
            _trace('T.offer->%s cam=%s frame=%s' % (_r, self._camera_id, context.frame_id))
        except Exception as _e:
            _trace('T.offer THREW %s' % repr(_e))
        self._next_submit_ns = now + self._interval_ns
