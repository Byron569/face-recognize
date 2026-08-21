"""WebSocket:摄像头实时二进制 JPEG 预览流 + 检测结果。"""

from __future__ import annotations
import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services.pipeline_manager import get_pipeline_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/cameras/{camera_id}")
async def camera_stream(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    mgr = get_pipeline_manager()
    await mgr.register_ws(camera_id, websocket)
    logger.info("WS client connected: %s", camera_id)
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
        logger.info("WS client disconnected: %s", camera_id)
    except Exception:  # noqa: BLE001
        logger.exception("WS error: %s", camera_id)
    finally:
        await mgr.unregister_ws(camera_id, websocket)
