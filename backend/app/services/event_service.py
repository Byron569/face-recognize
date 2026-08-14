"""
services.event_service — 事件与识别日志业务(组装 identity 显示名)。
"""


from __future__ import annotations
from datetime import datetime
from typing import Optional

from ..models.event import EventType
from ..repositories.event_repo import EventRepository


class EventService:
    def __init__(self, db):
        self._repo = EventRepository(db)

    async def list_events(
        self,
        page: int = 1,
        page_size: int = 20,
        event_type: Optional[str] = None,
        camera_id: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ):
        et = EventType(event_type) if event_type else None
        events, total = await self._repo.list(page, page_size, et, camera_id, acknowledged, start, end)
        return [self._to_dict(ev) for ev in events], total

    async def get_event(self, event_id: int) -> Optional[dict]:
        ev = await self._repo.get(event_id)
        return self._to_dict(ev) if ev else None

    async def acknowledge(self, event_id: int) -> bool:
        return await self._repo.acknowledge(event_id)

    async def cleanup(self, retention_days: int) -> int:
        from datetime import timedelta, timezone

        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        return await self._repo.cleanup(cutoff)

    # ── 识别日志 ──────────────────────────────────────────

    async def list_recognition_logs(
        self,
        page: int = 1,
        page_size: int = 20,
        camera_id: Optional[str] = None,
        identity_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ):
        logs, total = await self._repo.list_recognition_logs(page, page_size, camera_id, identity_id, start, end)
        return [self._log_to_dict(r) for r in logs], total

    async def get_recognition_log(self, log_id: int) -> Optional[dict]:
        log = await self._repo.get_recognition_log(log_id)
        return self._log_to_dict(log) if log else None

    # ── 序列化 ────────────────────────────────────────────

    @staticmethod
    def _to_dict(ev) -> dict:
        return {
            "id": ev.id,
            "event_type": ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type),
            "camera_id": ev.camera_id,
            "track_id": ev.track_id,
            "identity_id": str(ev.identity_id) if ev.identity_id else None,
            "identity_name": ev.payload.get("name") if ev.payload else None,
            "confidence": ev.confidence,
            "payload": ev.payload or {},
            "snapshot_path": ev.snapshot_path,
            "acknowledged": ev.acknowledged,
            "acknowledged_at": ev.acknowledged_at,
            "created_at": ev.created_at,
        }

    @staticmethod
    def _log_to_dict(log) -> dict:
        return {
            "id": log.id,
            "camera_id": log.camera_id,
            "identity_id": str(log.identity_id) if log.identity_id else None,
            "track_id": log.track_id,
            "similarity": log.similarity,
            "latency_ms": log.latency_ms,
            "created_at": log.created_at,
        }
