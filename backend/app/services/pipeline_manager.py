"""
services.pipeline_manager — 摄像头流水线生命周期管理(单例)。

职责:
    - 组装 vision 内核(帧源 / 引擎 / 跟踪器 / 任务)并管理其线程生命周期;
    - 管理 WebSocket 连接与帧/事件推送(线程 → asyncio loop 桥接);
    - 识别事件异步落库(recognition_logs / events,通过 EventBridge);
    - 提供状态与性能指标供 API 查询。

不包含任何推理算法 —— 算法全部在 vision/ 内核与任务插件中。
"""


from __future__ import annotations
import asyncio
import json
import logging
import math
import queue
import threading
import time
from collections import defaultdict
from collections import deque
from typing import Any, Dict, Optional

from fastapi import WebSocket

from vision.camera import OpenCVFrameSource
from vision.config import VisionConfig
from vision.pipeline import VisionPipeline
from vision.tracker import ByteTracker

from .event_ingress import EventIngress
from .gallery import FaceGallery
from .model_manager import EnginePool, get_engine_pool
from .stream_protocol import pack_jpeg_frame
from .task_registry import TaskRegistry
from .stream_subscriber import LatestFrameSender

logger = logging.getLogger(__name__)

_pipeline_manager: "PipelineManager | None" = None

# 全局可靠事件入口(线程安全 submit → 原子入库 + Outbox 广播)。
# 供外部 Task(如跌倒检测)作为 EventSinkProtocol 注入;也承担可靠 fall 事件广播。
_event_ingress: "EventIngress | None" = None


def get_event_ingress() -> EventIngress:
    global _event_ingress
    if _event_ingress is None:
        _event_ingress = EventIngress()
    return _event_ingress


class StreamMetrics:
    """Short-window preview stream counters safe for thread/event-loop use."""

    _RATE_WINDOW_SECONDS = 5.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enqueued = deque()
        self._encoded = deque()
        self._sent = deque()
        self._encode_dropped_frames = 0
        self._subscriber_dropped_frames = 0
        self._encoded_total = 0
        self._jpeg_bytes_total = 0
        self._jpeg_count = 0

    def record_enqueue(self) -> None:
        with self._lock:
            self._enqueued.append(time.monotonic())

    def record_encoded(self, jpeg_bytes: int) -> None:
        with self._lock:
            self._encoded.append(time.monotonic())
            self._encoded_total += 1
            self._jpeg_bytes_total += max(0, int(jpeg_bytes))
            self._jpeg_count += 1

    def record_sent(self) -> None:
        with self._lock:
            self._sent.append(time.monotonic())

    def record_encode_drop(self) -> None:
        with self._lock:
            self._encode_dropped_frames += 1

    def record_subscriber_drops(self, count: int) -> None:
        with self._lock:
            self._subscriber_dropped_frames += max(0, int(count))

    def snapshot(self) -> dict[str, float | int]:
        now = time.monotonic()
        with self._lock:
            cutoff = now - self._RATE_WINDOW_SECONDS
            for samples in (self._enqueued, self._encoded, self._sent):
                while samples and samples[0] < cutoff:
                    samples.popleft()
            window = self._RATE_WINDOW_SECONDS
            return {
                "preview_enqueue_fps": round(len(self._enqueued) / window, 2),
                "encoded_fps": round(len(self._encoded) / window, 2),
                "sent_fps": round(len(self._sent) / window, 2),
                "encoded_frames": self._encoded_total,
                "encode_dropped_frames": self._encode_dropped_frames,
                "subscriber_dropped_frames": self._subscriber_dropped_frames,
                "avg_jpeg_bytes": round(self._jpeg_bytes_total / self._jpeg_count)
                if self._jpeg_count
                else 0,
            }


def _scale_persons_for_preview(
    persons: list[dict], scale_x: float, scale_y: float
) -> list[dict]:
    """Scale track boxes to match the resized JPEG preview dimensions."""
    scaled_persons = []
    for person in persons:
        scaled = dict(person)
        bbox = person.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, width, height = bbox
            scaled["bbox"] = [
                float(x) * scale_x,
                float(y) * scale_y,
                float(width) * scale_x,
                float(height) * scale_y,
            ]
        scaled_persons.append(scaled)
    return scaled_persons


def _project_analytics_to_preview(
    fall: dict | None, preview_width: int, preview_height: int
) -> dict:
    """把 Task 写入的 source-pixels fall analytics 投影为 preview-pixels wire 消息。

    - fall 为 Task 写入 ``context.analytics["fall_detection"]`` 的结构;
    - preview_width/height 必须 > 0(后端实际整数编码尺寸);
    - bbox/keypoints 按 ``scale = preview/source`` 缩放,score 不变;
    - 有效坐标夹紧到 [0, preview-1],NaN/Inf/坏尺寸拒绝;
    - 输出内层 ``fall_detection`` 与外层 preview 字段,pose 状态保持 wire 小写。
    """
    if not isinstance(fall, dict):
        raise ValueError("fall analytics 必须是 dict")
    source_width = fall.get("source_width")
    source_height = fall.get("source_height")
    if not isinstance(source_width, int) or source_width <= 0:
        raise ValueError(f"source_width 非法: {source_width!r}")
    if not isinstance(source_height, int) or source_height <= 0:
        raise ValueError(f"source_height 非法: {source_height!r}")
    if not isinstance(preview_width, int) or preview_width <= 0:
        raise ValueError(f"preview_width 非法: {preview_width!r}")
    if not isinstance(preview_height, int) or preview_height <= 0:
        raise ValueError(f"preview_height 非法: {preview_height!r}")

    scale_x = preview_width / source_width
    scale_y = preview_height / source_height
    if not (math.isfinite(scale_x) and scale_x > 0):
        raise ValueError(f"scale_x 非法: {scale_x!r}")
    if not (math.isfinite(scale_y) and scale_y > 0):
        raise ValueError(f"scale_y 非法: {scale_y!r}")

    camera_session_id = fall.get("camera_session_id")
    source_frame_id = fall.get("source_frame_id")
    preview_frame_id = fall.get("attached_to_frame_id")

    def _scale_coord(v: float, scale: float, limit: int) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError("坐标非有限") from None
        if not math.isfinite(f):
            raise ValueError("坐标非有限(nan/inf)")
        return min(max(f * scale, 0.0), float(limit - 1))

    tracks_out = []
    for t in fall.get("tracks") or []:
        bbox = t.get("bbox")
        _b = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            _b = [
                _scale_coord(bbox[0], scale_x, preview_width),
                _scale_coord(bbox[1], scale_y, preview_height),
                _scale_coord(bbox[2], scale_x, preview_width),
                _scale_coord(bbox[3], scale_y, preview_height),
            ]
        kps_out = []
        for kp in t.get("keypoints") or []:
            if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                kps_out.append(
                    [_scale_coord(kp[0], scale_x, preview_width),
                     _scale_coord(kp[1], scale_y, preview_height)]
                )
        out_t = {
            "pose_track_id": t.get("pose_track_id"),
            "state": t.get("state"),
            "score": t.get("score"),
            "bbox": _b,
            "keypoints": kps_out,
        }
        tracks_out.append(out_t)

    fd = {
        "schema_version": fall.get("schema_version", 1),
        "camera_session_id": camera_session_id,
        "source_frame_id": source_frame_id,
        "preview_width": preview_width,
        "preview_height": preview_height,
        "coordinate_space": "preview_pixels",
        "transform": {
            "kind": "scale_no_letterbox",
            "scale_x": scale_x,
            "scale_y": scale_y,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
        "health": fall.get("health"),
        "result_age_ms": fall.get("result_age_ms"),
        "overlay_expires_in_ms": fall.get("overlay_expires_in_ms"),
        "worker_end_to_end_ms": fall.get("worker_end_to_end_ms"),
        "tracks": tracks_out,
    }
    fd = {k: v for k, v in fd.items() if v is not None}

    return {
        "type": "analytics",
        "schema_version": 1,
        "camera_id": fall.get("camera_id"),
        "camera_session_id": camera_session_id,
        "preview_frame_id": preview_frame_id,
        "fall_detection": fd,
    }


def get_pipeline_manager() -> "PipelineManager":
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager()
    return _pipeline_manager


class PipelineManager:
    """摄像头流水线管理器(进程内单例)。"""

    def __init__(self):
        self._pipelines: Dict[str, VisionPipeline] = {}
        self._subscribers: Dict[str, Dict[WebSocket, LatestFrameSender]] = defaultdict(dict)
        self._event_listeners: set[WebSocket] = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_frames: Dict[str, Any] = {}
        self._stream_settings: Dict[str, dict[str, int]] = {}
        self._stream_metrics: Dict[str, StreamMetrics] = {}
        # 编码线程池:每摄像头一个编码线程(CPU 密集) + 单槽位队列(丢旧帧),
        # 事件循环只做 WS 发送,避免编码阻塞推流/心跳/落库
        self._encode_queues: Dict[str, queue.Queue] = {}
        self._encode_threads: Dict[str, threading.Thread] = {}
        self._encode_stops: Dict[str, threading.Event] = {}
        self._gallery = FaceGallery()
        self._engine_pool: EnginePool = get_engine_pool()
        self._gallery_loaded = False
        self._event_ingress = get_event_ingress()
        self._delivery_modes: Dict[str, str] = {}

    # ── 生命周期挂钩 ──────────────────────────────────────

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        # 绑定可靠事件入口:广播回到本管理的 WS 事件监听者,投递模式按摄像头 fall 配置解析
        ingress = self._event_ingress
        ingress.bind_loop(loop)
        ingress.set_broadcast(self._broadcast_ingress_event)
        ingress.set_mode_resolver(self._resolve_delivery_mode)
        ingress.start()

    async def load_gallery(self) -> int:
        """从 PostgreSQL 拉取全部 embedding 构建内存底库。返回条目数。"""
        from ..deps import AsyncSessionLocal
        from ..repositories.identity_repo import IdentityRepository

        async with AsyncSessionLocal() as db:
            rows = await IdentityRepository(db).all_embeddings()
        self._gallery.rebuild(rows)
        self._gallery_loaded = True
        logger.info("[pipeline-manager] gallery loaded: %s embeddings", len(rows))
        return len(rows)

    async def refresh_gallery(self) -> int:
        """底库变更后调用(注册/删除人脸)。"""
        return await self.load_gallery()

    @property
    def gallery(self) -> FaceGallery:
        return self._gallery

    # ── 摄像头生命周期 ────────────────────────────────────

    async def start_camera(self, camera_id: str, source, config: Dict[str, Any]) -> bool:
        existing = self._pipelines.get(camera_id)
        if existing is not None:
            if existing.is_alive():
                logger.warning("[pipeline-manager] camera %s already running", camera_id)
                return False
            # 线程已退出(如源打开失败)— 清理残留后允许重启
            await self.stop_camera(camera_id)

        if not self._gallery_loaded:
            await self.load_gallery()

        vision_cfg = VisionConfig.from_dict(config.get("vision", {}))
        camera_defaults = config.get("camera_defaults", {})
        stream_cfg = config.get("stream", {})
        from ..config import get_settings

        settings = get_settings()

        # 1. 组装内核(全部依赖注入)
        frame_source = OpenCVFrameSource(
            source=source,
            width=int(camera_defaults.get("width", 640)),
            height=int(camera_defaults.get("height", 480)),
            max_width=int(camera_defaults.get("max_width", 0)),
        )
        engine = self._engine_pool.get(vision_cfg)
        tracker = ByteTracker(vision_cfg.track)

        # 2. 可插拔任务
        fall_cfg = (config.get("tasks", {}) or {}).get("fall_detection", {}) or {}
        with self._lock:
            # 投递模式按摄像头自己的 fall 配置解析;默认 shadow(仅持久化,不实时告警)
            self._delivery_modes[camera_id] = str(fall_cfg.get("mode", "shadow"))
        registry = TaskRegistry(config.get("tasks", {}))
        extra_kwargs = {
            "full_config": config,
            "gallery": self._gallery,
            "tracker": tracker,
            "event_sink": self._event_ingress,  # 线程安全 submit → 原子入库 + Outbox
        }
        if fall_cfg.get("enabled"):
            # 惰性注入姿态 Runtime 工厂 + 生产 process_factory(启用时才引入姿态包;
            # import 不加载模型/不启 Worker;首个有效上下文 acquire 时才拉起真实 GPU 进程)
            from pathlib import Path as _Path

            # fall 配置的持久化路径在 build_camera_config 已解析为绝对路径(相对仓库根);
            # 这里自动创建 var 目录,保证 clone 即跑的便携部署下 journal/spool 可落盘
            rt_cfg = fall_cfg.get("runtime") or {}
            for _p in (rt_cfg.get("worker_journal_path"), rt_cfg.get("event_spool_path")):
                if _p:
                    _Path(str(_p)).parent.mkdir(parents=True, exist_ok=True)

            from ai_monitor_pose.runtime_registry import PoseRuntimeRegistry
            from ai_monitor_pose.worker.launcher import build_worker_process_factory

            extra_kwargs["runtime_factory"] = PoseRuntimeRegistry
            w_cfg = fall_cfg.get("worker") or {}
            worker_py = str(w_cfg.get("python") or "")
            worker_mod = str(w_cfg.get("module") or "ai_monitor_pose.worker")
            pose_root = str(_Path(worker_py).parents[2]) if worker_py else None
            # 闭包捕获 fall_cfg:worker 进程经 WORKER_CONF 构建同源 FallTaskConfig
            extra_kwargs["process_factory"] = build_worker_process_factory(
                worker_py, fall_cfg, module=worker_mod, cwd=pose_root,
            )
        tasks = registry.load(extra_kwargs=extra_kwargs)

        # 3. 组装流水线(帧/事件回调桥接到 asyncio loop)
        loop = self._loop
        push_every = min(30, max(1, int(stream_cfg.get("push_fps", settings.stream_push_fps))))
        last_push: Dict[str, float] = {}

        # 编码线程(单槽位队列,满则丢旧帧 —— 监控只需最新帧)
        enc_q: queue.Queue = queue.Queue(maxsize=1)
        stop_evt = threading.Event()
        self._encode_queues[camera_id] = enc_q
        self._encode_stops[camera_id] = stop_evt
        enc_thread = threading.Thread(
            target=self._encode_loop,
            args=(camera_id, enc_q, stop_evt, loop),
            daemon=True,
            name=f"encode-{camera_id}",
        )
        enc_thread.start()
        self._encode_threads[camera_id] = enc_thread

        def _throttled_push(context) -> None:
            if loop is None or not loop.is_running():
                return
            now = time.time()
            if now - last_push.get(camera_id, 0.0) < 1.0 / push_every:
                return
            last_push[camera_id] = now
            with self._lock:
                has_viewers = bool(self._subscribers.get(camera_id))
                stream_metrics = self._stream_metrics.setdefault(camera_id, StreamMetrics())
            frame_copy = context.frame.copy()
            self._last_frames[camera_id] = frame_copy  # 抓拍用原始帧(与是否有订阅者无关)
            persons = [t.to_dict() for t in context.tracks]
            fall_analytics = context.analytics.get("fall_detection") if isinstance(context.analytics, dict) else None
            if not has_viewers:
                return  # 无订阅者:跳过编码,省 CPU
            item = (frame_copy, persons, context.frame_id, fall_analytics)
            try:
                enc_q.put_nowait(item)
                stream_metrics.record_enqueue()
            except queue.Full:
                # 单槽位:丢弃旧帧,保留最新
                try:
                    enc_q.get_nowait()
                    stream_metrics.record_encode_drop()
                except queue.Empty:
                    pass
                try:
                    enc_q.put_nowait(item)
                    stream_metrics.record_enqueue()
                except queue.Full:
                    stream_metrics.record_encode_drop()

        def _handle_event(evt) -> None:
            if loop is None or not loop.is_running():
                return
            asyncio.run_coroutine_threadsafe(self._on_pipeline_event(evt), loop)

        pipeline = VisionPipeline(
            camera_id=camera_id,
            source=frame_source,
            engine=engine,
            tracker=tracker,
            config=vision_cfg,
            tasks=tasks,
            on_frame=_throttled_push,
            on_event=_handle_event,
        )

        # per-camera 推流参数,编码线程使用
        stream_settings = {
            "max_height": max(0, int(stream_cfg.get("max_height", settings.stream_max_height))),
            "jpeg_quality": min(
                100,
                max(1, int(stream_cfg.get("jpeg_quality", settings.stream_jpeg_quality))),
            ),
            "push_fps": min(30, max(1, push_every)),
        }
        with self._lock:
            self._pipelines[camera_id] = pipeline
            self._stream_settings[camera_id] = stream_settings
            self._stream_metrics.setdefault(camera_id, StreamMetrics())

        pipeline.start()
        # 短暂探测:源立即打开失败的流水线线程会马上退出,据此返回失败
        pipeline.join(timeout=1.0)
        if not pipeline.is_alive():
            await self.stop_camera(camera_id)
            logger.error("[pipeline-manager] camera %s source open failed", camera_id)
            return False
        logger.info("[pipeline-manager] camera %s started (device=%s, pack=%s)",
                    camera_id, engine.device, vision_cfg.model_pack)
        return True

    async def stop_camera(self, camera_id: str) -> bool:
        with self._lock:
            pipeline = self._pipelines.pop(camera_id, None)
            self._stream_settings.pop(camera_id, None)
            self._stream_metrics.pop(camera_id, None)  # 清理流指标,避免启停累积
            self._delivery_modes.pop(camera_id, None)
        self._last_frames.pop(camera_id, None)          # 清理抓拍缓存帧
        # 停止编码线程(唤醒 + join,避免线程泄漏)
        stop_evt = self._encode_stops.pop(camera_id, None)
        enc_thread = self._encode_threads.pop(camera_id, None)
        self._encode_queues.pop(camera_id, None)
        if stop_evt is not None:
            stop_evt.set()
        if enc_thread is not None:
            enc_thread.join(timeout=2.0)
        if pipeline is None:
            return False
        pipeline.close(timeout=5.0)
        logger.info("[pipeline-manager] camera %s stopped", camera_id)
        return True

    def get_status(self, camera_id: str) -> Optional[dict]:
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            return None
        return pipeline.metrics()

    def list_cameras(self) -> list[str]:
        return list(self._pipelines.keys())

    def is_running(self, camera_id: str) -> bool:
        """摄像头流水线是否真正存活(在线程存活的意义上)。"""
        p = self._pipelines.get(camera_id)
        return p is not None and p.is_alive()

    async def shutdown(self) -> None:
        for camera_id in list(self._pipelines.keys()):
            await self.stop_camera(camera_id)
        self._engine_pool.close_all()
        # 停可靠事件入口(有界 drain + 停 dispatcher);幂等
        try:
            await self._event_ingress.close()
        except Exception:  # noqa: BLE001
            logger.exception("[pipeline-manager] event ingress shutdown failed")

    # ── WebSocket 管理 ────────────────────────────────────

    async def register_ws(self, camera_id: str, ws: WebSocket) -> None:
        async def _disconnect() -> None:
            await self.unregister_ws(camera_id, ws)

        with self._lock:
            metrics = self._stream_metrics.setdefault(camera_id, StreamMetrics())
        sender = LatestFrameSender(
            ws,
            on_disconnect=_disconnect,
            on_sent=metrics.record_sent,
        )
        sender.start()
        with self._lock:
            self._subscribers[camera_id][ws] = sender

    async def unregister_ws(self, camera_id: str, ws: WebSocket) -> None:
        with self._lock:
            sender = self._subscribers.get(camera_id, {}).pop(ws, None)
        if sender is not None:
            await sender.close()

    def get_stream_metrics(self, camera_id: str) -> dict:
        with self._lock:
            metrics = self._stream_metrics.get(camera_id)
        return metrics.snapshot() if metrics is not None else StreamMetrics().snapshot()

    def register_event_listener(self, ws: WebSocket) -> None:
        self._event_listeners.add(ws)

    def unregister_event_listener(self, ws: WebSocket) -> None:
        self._event_listeners.discard(ws)

    # ── 帧推送:编码在线程池,发送在事件循环 ────────────────

    def _encode_loop(self, camera_id: str, enc_q: queue.Queue, stop_evt: threading.Event, loop) -> None:
        """编码线程:取最新帧 → 缩放 → JPEG → 桥回事件循环 fanout。

        独立线程使多摄像头编码互不阻塞;单槽位队列保证只处理最新帧。
        """
        import cv2

        from ..config import get_settings

        settings = get_settings()
        while not stop_evt.is_set():
            try:
                item = enc_q.get(timeout=0.3)
            except queue.Empty:
                continue
            frame, persons, frame_id, fall_analytics = item
            try:
                original_height, original_width = frame.shape[:2]
                display_persons = persons
                preview_width = original_width
                preview_height = original_height
                with self._lock:
                    stream_settings = dict(self._stream_settings.get(camera_id, {}))
                push_max_height = stream_settings.get("max_height", settings.stream_max_height)
                if push_max_height > 0 and original_height > push_max_height:
                    scale_y = push_max_height / original_height
                    scale_x = scale_y
                    frame = cv2.resize(
                        frame, (int(original_width * scale_x), int(original_height * scale_y))
                    )
                    preview_width, preview_height = frame.shape[1], frame.shape[0]
                    display_persons = _scale_persons_for_preview(persons, scale_x, scale_y)
                ret, buf = cv2.imencode(
                    ".jpg",
                    frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        stream_settings.get("jpeg_quality", settings.stream_jpeg_quality),
                    ],
                )
                if not ret:
                    continue
                jpeg = buf.tobytes()
                analytics_msg = None
                if fall_analytics is not None:
                    try:
                        analytics_msg = _project_analytics_to_preview(
                            dict(fall_analytics), preview_width, preview_height
                        )
                        analytics_msg["camera_id"] = camera_id
                    except ValueError:  # noqa: BLE001
                        analytics_msg = None  # 坏数据丢弃该 overlay,不影响帧
                with self._lock:
                    stream_metrics = self._stream_metrics.setdefault(camera_id, StreamMetrics())
                stream_metrics.record_encoded(len(jpeg))
            except Exception:  # noqa: BLE001
                continue
            if loop is None or not loop.is_running():
                continue
            try:
                asyncio.run_coroutine_threadsafe(
                    self._fanout_encoded_frame(
                        camera_id, jpeg, display_persons, frame_id, analytics_msg
                    ),
                    loop,
                )
            except Exception:  # noqa: BLE001
                continue

    async def _fanout_encoded_frame(
        self, camera_id: str, jpeg: bytes, persons: list, frame_id: int,
        analytics_msg: dict | None = None,
    ) -> None:
        """事件循环内只做协议封包和非阻塞 fanout,不直接等待 socket。"""
        frame_packet = pack_jpeg_frame(frame_id, jpeg)
        det_msg = json.dumps({"type": "detections", "frame_id": frame_id, "persons": persons})

        with self._lock:
            senders = list(self._subscribers.get(camera_id, {}).values())
            metrics = self._stream_metrics.setdefault(camera_id, StreamMetrics())
        for sender in senders:
            before = sender.dropped_frames
            analytics_json = json.dumps(analytics_msg) if analytics_msg is not None else None
            sender.offer(frame_packet, det_msg, analytics_json)
            metrics.record_subscriber_drops(sender.dropped_frames - before)

    # ── 可靠事件入口(EventIngress)接线 ──────────────────

    def _resolve_delivery_mode(self, camera_id: str) -> str:
        """摄像头 fall 事件的投递模式(alert=广播告警 / shadow=仅持久化)。默认 shadow。"""
        with self._lock:
            return self._delivery_modes.get(camera_id, "shadow")

    async def _broadcast_ingress_event(self, payload: dict) -> None:
        """EventIngress Outbox dispatcher 的广播回调:把可靠 fall 事件推给 WS 监听者。"""
        if not payload or not self._event_listeners:
            return
        msg = json.dumps(payload, ensure_ascii=False)
        dead = []
        for ws in list(self._event_listeners):
            try:
                await ws.send_text(msg)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self._event_listeners.discard(ws)

    # ── 事件处理与落库(EventBridge,在 asyncio loop 内执行)─

    async def _on_pipeline_event(self, evt) -> None:
        """推理事件 → 广播 WS + 按配置落库。"""
        data = evt.to_dict()
        # 1. 广播给订阅者
        await self._broadcast_event(data)

        # 2. 落库
        try:
            if evt.event_type == "recognition":
                await self._persist_recognition(evt)
        except Exception:  # noqa: BLE001
            logger.exception("[event-bridge] persist failed: %s", evt.event_type)

    async def _persist_recognition(self, evt) -> None:
        import uuid as uuid_mod

        payload = evt.payload or {}
        if not payload.get("changed", False):
            return
        from ..deps import AsyncSessionLocal
        from ..models.event import EventType
        from ..repositories.event_repo import EventRepository

        identity_id = payload.get("identity_id") or None
        if identity_id and not isinstance(identity_id, uuid_mod.UUID):
            try:
                identity_id = uuid_mod.UUID(identity_id)
            except (ValueError, AttributeError, TypeError):
                identity_id = None
        async with AsyncSessionLocal() as db:
            repo = EventRepository(db)
            await repo.add_recognition_log(
                camera_id=evt.camera_id,
                identity_id=identity_id,
                track_id=evt.track_id or 0,
                similarity=float(evt.confidence),
                latency_ms=payload.get("latency_ms"),
            )
            await repo.create(
                event_type=EventType.recognition,
                camera_id=evt.camera_id,
                track_id=evt.track_id,
                identity_id=identity_id,
                confidence=float(evt.confidence),
                payload={"name": payload.get("name", "")},
            )

    async def _broadcast_event(self, data: dict) -> None:
        msg = json.dumps({"type": "event", **data}, ensure_ascii=False)
        dead = []
        for ws in list(self._event_listeners):
            try:
                await ws.send_text(msg)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self._event_listeners.discard(ws)

    # ── 抓拍 ──────────────────────────────────────────────

    def get_last_frame(self, camera_id: str):
        return self._last_frames.get(camera_id)

    def snapshot_jpeg(self, camera_id: str, quality: int = 90) -> Optional[bytes]:
        frame = self._last_frames.get(camera_id)
        if frame is None:
            return None
        import cv2

        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ret else None
