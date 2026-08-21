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

from .gallery import FaceGallery
from .model_manager import EnginePool, get_engine_pool
from .stream_protocol import pack_jpeg_frame
from .task_registry import TaskRegistry
from .stream_subscriber import LatestFrameSender

logger = logging.getLogger(__name__)

_pipeline_manager: "PipelineManager | None" = None


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

    # ── 生命周期挂钩 ──────────────────────────────────────

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

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
        registry = TaskRegistry(config.get("tasks", {}))
        tasks = registry.load(extra_kwargs={
            "full_config": config,
            "gallery": self._gallery,
            "tracker": tracker,
        })

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
            if not has_viewers:
                return  # 无订阅者:跳过编码,省 CPU
            item = (frame_copy, persons, context.frame_id)
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
            frame, persons, frame_id = item
            try:
                original_height, original_width = frame.shape[:2]
                display_persons = persons
                with self._lock:
                    stream_settings = dict(self._stream_settings.get(camera_id, {}))
                push_max_height = stream_settings.get("max_height", settings.stream_max_height)
                if push_max_height > 0 and original_height > push_max_height:
                    scale_y = push_max_height / original_height
                    scale_x = scale_y
                    frame = cv2.resize(
                        frame, (int(original_width * scale_x), int(original_height * scale_y))
                    )
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
                with self._lock:
                    stream_metrics = self._stream_metrics.setdefault(camera_id, StreamMetrics())
                stream_metrics.record_encoded(len(jpeg))
            except Exception:  # noqa: BLE001
                continue
            if loop is None or not loop.is_running():
                continue
            try:
                asyncio.run_coroutine_threadsafe(
                    self._fanout_encoded_frame(camera_id, jpeg, display_persons, frame_id), loop
                )
            except Exception:  # noqa: BLE001
                continue

    async def _fanout_encoded_frame(
        self, camera_id: str, jpeg: bytes, persons: list, frame_id: int
    ) -> None:
        """事件循环内只做协议封包和非阻塞 fanout,不直接等待 socket。"""
        frame_packet = pack_jpeg_frame(frame_id, jpeg)
        det_msg = json.dumps({"type": "detections", "frame_id": frame_id, "persons": persons})

        with self._lock:
            senders = list(self._subscribers.get(camera_id, {}).values())
            metrics = self._stream_metrics.setdefault(camera_id, StreamMetrics())
        for sender in senders:
            before = sender.dropped_frames
            sender.offer(frame_packet, det_msg)
            metrics.record_subscriber_drops(sender.dropped_frames - before)

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
