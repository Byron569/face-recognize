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
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Optional

from fastapi import WebSocket

from vision.camera import OpenCVFrameSource
from vision.config import VisionConfig
from vision.pipeline import VisionPipeline
from vision.tracker import ByteTracker

from .gallery import FaceGallery
from .model_manager import EnginePool, get_engine_pool
from .task_registry import TaskRegistry

logger = logging.getLogger(__name__)

_pipeline_manager: "PipelineManager | None" = None


def get_pipeline_manager() -> "PipelineManager":
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager()
    return _pipeline_manager


class PipelineManager:
    """摄像头流水线管理器(进程内单例)。"""

    def __init__(self):
        self._pipelines: Dict[str, VisionPipeline] = {}
        self._ws_connections: Dict[str, set[WebSocket]] = defaultdict(set)
        self._event_listeners: set[WebSocket] = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_frames: Dict[str, Any] = {}
        self._stream_max_height: Dict[str, int] = {}  # per-camera 推流最大高度(0=不缩放)
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
        if camera_id in self._pipelines:
            logger.warning("[pipeline-manager] camera %s already running", camera_id)
            return False

        if not self._gallery_loaded:
            await self.load_gallery()

        vision_cfg = VisionConfig.from_dict(config.get("vision", {}))
        camera_defaults = config.get("camera_defaults", {})
        stream_cfg = config.get("stream", {})

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
        push_every = max(1, int(stream_cfg.get("push_fps", 10)))
        last_push: Dict[str, float] = {}

        def _throttled_push(context) -> None:
            if loop is None or not loop.is_running():
                return
            now = time.time()
            if now - last_push.get(camera_id, 0.0) < 1.0 / push_every:
                return
            last_push[camera_id] = now
            frame_copy = context.frame.copy()
            persons = [t.to_dict() for t in context.tracks]
            asyncio.run_coroutine_threadsafe(
                self._push_frame(camera_id, frame_copy, persons, context.frame_id), loop
            )

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

        # per-camera 推流最大高度(0=不缩放),_push_frame 使用
        from ..config import get_settings

        push_max_height = int(stream_cfg.get("max_height", get_settings().stream_max_height))
        with self._lock:
            self._pipelines[camera_id] = pipeline
            self._stream_max_height[camera_id] = push_max_height

        pipeline.start()
        logger.info("[pipeline-manager] camera %s started (device=%s, pack=%s)",
                    camera_id, engine.device, vision_cfg.model_pack)
        return True

    async def stop_camera(self, camera_id: str) -> bool:
        with self._lock:
            pipeline = self._pipelines.pop(camera_id, None)
            self._stream_max_height.pop(camera_id, None)
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

    async def shutdown(self) -> None:
        for camera_id in list(self._pipelines.keys()):
            await self.stop_camera(camera_id)
        self._engine_pool.close_all()

    # ── WebSocket 管理 ────────────────────────────────────

    def register_ws(self, camera_id: str, ws: WebSocket) -> None:
        with self._lock:
            self._ws_connections[camera_id].add(ws)

    def unregister_ws(self, camera_id: str, ws: WebSocket) -> None:
        with self._lock:
            self._ws_connections[camera_id].discard(ws)

    def register_event_listener(self, ws: WebSocket) -> None:
        self._event_listeners.add(ws)

    def unregister_event_listener(self, ws: WebSocket) -> None:
        self._event_listeners.discard(ws)

    # ── 帧推送(在 asyncio loop 内执行)────────────────────

    async def _push_frame(self, camera_id: str, frame, persons: list, frame_id: int) -> None:
        from ..config import get_settings

        settings = get_settings()
        self._last_frames[camera_id] = frame

        import base64

        import cv2

        h = frame.shape[0]
        push_max_height = self._stream_max_height.get(camera_id, settings.stream_max_height)
        if push_max_height > 0 and h > push_max_height:
            scale = push_max_height / h
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        ret, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.stream_jpeg_quality]
        )
        if not ret:
            return
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        frame_msg = json.dumps(
            {"type": "frame", "data": b64, "timestamp": time.time(), "frame_id": frame_id}
        )
        det_msg = json.dumps(
            {"type": "detections", "frame_id": frame_id, "persons": persons}
        )

        with self._lock:
            ws_set = list(self._ws_connections.get(camera_id, ()))
        dead = []
        for ws in ws_set:
            try:
                await ws.send_text(frame_msg)
                await ws.send_text(det_msg)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.unregister_ws(camera_id, ws)

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
