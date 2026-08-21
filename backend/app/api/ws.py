"""WebSocket:摄像头实时二进制 JPEG 预览流 + 检测结果。"""

from __future__ import annotations
import asyncio
import logging

import json

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_db
from ..services.camera_service import CameraService

from ..services.pipeline_manager import get_pipeline_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/cameras/{camera_id}")
async def camera_stream(
    websocket: WebSocket,
    camera_id: str,
    db: AsyncSession = Depends(get_db),
):
    await websocket.accept()
    camera = await CameraService(db).get(camera_id)
    if not camera:
        await websocket.send_text(json.dumps({"type": "error", "message": "camera not found"}))
        await websocket.close()
        return
    mgr = get_pipeline_manager()
    running = mgr.is_running(camera_id)
    await websocket.send_text(json.dumps({"type": "status", "camera_id": camera_id, "running": running}))
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
