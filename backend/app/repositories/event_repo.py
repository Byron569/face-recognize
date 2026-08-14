"""事件与识别日志数据访问。"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.event import Event, EventType, RecognitionLog


class EventRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ── 事件 ─────────────────────────────────────────────

    async def list(
        self,
        page: int = 1,
        page_size: int = 20,
        event_type: Optional[EventType] = None,
        camera_id: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> tuple[list[Event], int]:
        q = select(Event)
        if event_type:
            q = q.where(Event.event_type == event_type)
        if camera_id:
            q = q.where(Event.camera_id == camera_id)
        if acknowledged is not None:
            q = q.where(Event.acknowledged == acknowledged)
        if start:
            q = q.where(Event.created_at >= start)
        if end:
            q = q.where(Event.created_at <= end)
        count_q = select(func.count()).select_from(q.subquery())
        total = (await self.db.execute(count_q)).scalar() or 0
        q = q.order_by(Event.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
        rows = (await self.db.execute(q)).scalars().all()
        return list(rows), total

    async def get(self, event_id: int) -> Optional[Event]:
        return await self.db.get(Event, event_id)

    async def create(
        self,
        event_type: EventType,
        camera_id: str,
        track_id: Optional[int] = None,
        identity_id: Optional[uuid.UUID] = None,
        confidence: float = 0,
        payload: Optional[dict] = None,
    ) -> Event:
        event = Event(
            event_type=event_type,
            camera_id=camera_id,
            track_id=track_id,
            identity_id=identity_id,
            confidence=confidence,
            payload=payload or {},
        )
        self.db.add(event)
        await self.db.commit()
        await self.db.refresh(event)
        return event

    async def acknowledge(self, event_id: int) -> bool:
        result = await self.db.execute(
            update(Event)
            .where(Event.id == event_id)
            .values(acknowledged=True, acknowledged_at=datetime.utcnow())
        )
        await self.db.commit()
        return (result.rowcount or 0) > 0

    async def cleanup(self, cutoff: datetime) -> int:
        r1 = await self.db.execute(delete(Event).where(Event.created_at < cutoff))
        r2 = await self.db.execute(delete(RecognitionLog).where(RecognitionLog.created_at < cutoff))
        await self.db.commit()
        return (r1.rowcount or 0) + (r2.rowcount or 0)

    # ── 识别日志 ──────────────────────────────────────────

    async def add_recognition_log(
        self,
        camera_id: str,
        identity_id: Optional[uuid.UUID],
        track_id: int,
        similarity: float,
        latency_ms: Optional[float] = None,
    ) -> RecognitionLog:
        log = RecognitionLog(
            camera_id=camera_id,
            identity_id=identity_id,
            track_id=track_id,
            similarity=similarity,
            latency_ms=latency_ms,
        )
        self.db.add(log)
        await self.db.commit()
        return log

    async def list_recognition_logs(
        self,
        page: int = 1,
        page_size: int = 20,
        camera_id: Optional[str] = None,
        identity_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> tuple[list[RecognitionLog], int]:
        q = select(RecognitionLog)
        if camera_id:
            q = q.where(RecognitionLog.camera_id == camera_id)
        if identity_id:
            q = q.where(RecognitionLog.identity_id == identity_id)
        if start:
            q = q.where(RecognitionLog.created_at >= start)
        if end:
            q = q.where(RecognitionLog.created_at <= end)
        count_q = select(func.count()).select_from(q.subquery())
        total = (await self.db.execute(count_q)).scalar() or 0
        q = q.order_by(RecognitionLog.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
        rows = (await self.db.execute(q)).scalars().all()
        return list(rows), total

    async def get_recognition_log(self, log_id: int) -> Optional[RecognitionLog]:
        return await self.db.get(RecognitionLog, log_id)
