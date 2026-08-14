"""WebSocket:全局事件通道(所有摄像头的 VisionEvent 统一推送)。"""

from __future__ import annotations
import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services.pipeline_manager import get_pipeline_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/events")
async def event_stream(websocket: WebSocket):
    await websocket.accept()
    mgr = get_pipeline_manager()
    mgr.register_event_listener(websocket)
    logger.info("WS events client connected")
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                try:
                    await websocket.send_text('{"type":"ping"}')
                except Exception:  # noqa: BLE001
                    break
    except WebSocketDisconnect:
        logger.info("WS events client disconnected")
    except Exception:  # noqa: BLE001
        logger.exception("WS events error")
    finally:
        mgr.unregister_event_listener(websocket)
