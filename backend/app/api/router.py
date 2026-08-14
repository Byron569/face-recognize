from __future__ import annotations
from fastapi import APIRouter

from . import cameras, events, faces, health, recognition, system, tasks, ws, ws_events

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router, tags=["health"])
api_router.include_router(cameras.router, tags=["cameras"])
api_router.include_router(faces.router, tags=["faces"])
api_router.include_router(events.router, tags=["events"])
api_router.include_router(recognition.router, tags=["recognition"])
api_router.include_router(system.router, tags=["system"])
api_router.include_router(tasks.router, tags=["tasks"])

# WebSocket 路由无 /api 前缀
ws_router = APIRouter()
ws_router.include_router(ws.router)
ws_router.include_router(ws_events.router)

__all__ = ["api_router", "ws_router"]
