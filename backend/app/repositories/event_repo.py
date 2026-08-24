"""事件与识别日志数据访问。"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.event import Event, EventOutbox, EventType, RecognitionLog


# 可靠 fall 事件类型;其余类型走旧 best-effort 广播路径
FALL_EVENT_TYPES = {"fall_potential", "fall_detected", "fall_recovered"}
# 固定 PostgreSQL advisory lock 键:防止 outbox 无限增长
OUTBOX_CAPACITY_LOCK = "ai-monitor-event-outbox-capacity-v1"


class IngressError(Exception):
    reason = "INGRESS_ERROR"


class OutboxCapacityError(IngressError):
    reason = "OUTBOX_FULL"


class DedupeMismatchError(IngressError):
    reason = "DEDUPE_CONFLICT"


class IngressRejectedError(IngressError):
    reason = "INGRESS_REJECTED"


class IngressQueueFullError(IngressError):
    reason = "INGRESS_QUEUE_FULL"


class IngestResult:
    __slots__ = ("event_row_id", "inserted")

    def __init__(self, event_row_id: int, inserted: bool):
        self.event_row_id = event_row_id
        self.inserted = inserted


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

    async def delete_many(self, ids: list[int]) -> int:
        """按 id 列表批量删除事件,返回删除条数。"""
        if not ids:
            return 0
        result = await self.db.execute(delete(Event).where(Event.id.in_(ids)))
        await self.db.commit()
        return result.rowcount or 0

    async def delete_filtered(
        self,
        event_type: Optional[EventType] = None,
        camera_id: Optional[str] = None,
        acknowledged: Optional[bool] = None,
    ) -> int:
        """按筛选条件删除全部匹配事件(用于「清空/删除全部」)。"""
        q = delete(Event)
        if event_type:
            q = q.where(Event.event_type == event_type)
        if camera_id:
            q = q.where(Event.camera_id == camera_id)
        if acknowledged is not None:
            q = q.where(Event.acknowledged == acknowledged)
        result = await self.db.execute(q)
        await self.db.commit()
        return result.rowcount or 0

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


# ──────────────────────────────────────────────────────────────
# 通用事务型 Outbox(可靠 fall 事件)
# ──────────────────────────────────────────────────────────────

class EventOutboxRepository:
    """把可靠 fall 事件以单事务写入 events + event_outbox 并管理 outbox。

    events exactly-once 由 dedupe_key 唯一约束 + 原子 ON CONFLICT 保证;
    outbox 提供 at-least-once 广播。事件与 outbox 在同一事务 commit,任何步骤失败整体回滚。
    """

    def __init__(self, db: AsyncSession, *, outbox_pending_capacity: int = 10000):
        self.db = db
        self.capacity = int(outbox_pending_capacity)

    async def _pending_count(self) -> int:
        q = (
            select(func.count())
            .select_from(EventOutbox)
            .where(EventOutbox.delivered_at.is_(None))
        )
        return int((await self.db.execute(q)).scalar() or 0)

    async def ingest(
        self,
        *,
        event_type: str,
        camera_id: str,
        track_id: int | None,
        confidence: float,
        occurred_at: datetime | None,
        source_event_id: str,
        dedupe_key: str,
        incident_id: str | None,
        payload: dict,
        delivery_mode: str,
        outbox_payload: dict | None = None,
    ) -> IngestResult:
        """单事务原子写入。返回 IngestResult,outbox 超容量抛 OutboxCapacityError。"""
        await self.db.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:k))"),
            {"k": OUTBOX_CAPACITY_LOCK},
        )
        pending = await self._pending_count()
        if pending >= self.capacity:
            raise OutboxCapacityError(
                f"pending outbox reached capacity {self.capacity}"
            )

        stmt = (
            pg_insert(Event)
            .values(
                event_type=event_type,
                camera_id=camera_id,
                track_id=track_id,
                confidence=float(confidence or 0),
                occurred_at=occurred_at,
                source_event_id=source_event_id,
                dedupe_key=dedupe_key,
                incident_id=incident_id,
                delivery_mode=delivery_mode,
                payload=payload or {},
            )
            .on_conflict_do_nothing(index_elements=["dedupe_key"])
            .returning(Event.id, Event.source_event_id)
        )
        row = (await self.db.execute(stmt)).first()
        if row is not None:
            event_row_id = int(row.id)
            inserted = True
        else:
            existing = (
                await self.db.execute(
                    select(Event.id, Event.source_event_id).where(
                        Event.dedupe_key == dedupe_key
                    )
                )
            ).first()
            if existing is None:
                raise IngressRejectedError(f"dedupe_key {dedupe_key!r} has no row after conflict")
            if existing.source_event_id != source_event_id:
                raise DedupeMismatchError(
                    f"dedupe_key {dedupe_key!r} already bound to a different source_event_id"
                )
            event_row_id = int(existing.id)
            inserted = False

        if inserted:
            # persist_only:落库即视为已投递,避免影响 outbox 容量
            is_persist_only = delivery_mode == "persist_only"
            envelope = dict(outbox_payload or {})
            envelope["id"] = event_row_id
            out_stmt = (
                pg_insert(EventOutbox)
                .values(
                    event_row_id=event_row_id,
                    dedupe_key=dedupe_key,
                    payload=envelope,
                    delivery_mode=delivery_mode,
                    attempt_count=0,
                    next_attempt_at=datetime.now(timezone.utc),
                    delivered_at=datetime.now(timezone.utc) if is_persist_only else None,
                )
                .on_conflict_do_nothing(index_elements=["dedupe_key"])
            )
            await self.db.execute(out_stmt)

        await self.db.commit()
        await self.db.refresh(await self.db.get(Event, event_row_id))
        return IngestResult(event_row_id, inserted)

    async def fetch_undelivered(
        self, limit: int = 100
    ) -> list[EventOutbox]:
        q = (
            select(EventOutbox)
            .where(
                EventOutbox.delivered_at.is_(None),
                EventOutbox.next_attempt_at <= datetime.now(timezone.utc),
            )
            .order_by(EventOutbox.created_at.asc())
            .limit(limit)
        )
        return list((await self.db.execute(q)).scalars().all())

    async def mark_delivered(self, outbox_id: int, *, error: str | None = None) -> None:
        """标记已投递;error 非空表示投递失败(退避重试)。"""
        from datetime import timedelta

        if error is None:
            values = {
                "attempt_count": EventOutbox.attempt_count + 1,
                "delivered_at": datetime.now(timezone.utc),
            }
        else:
            from ..services.event_ingress import EVENT_RETRY_BASE_SECONDS

            backoff = EVENT_RETRY_BASE_SECONDS * (2 ** min(8, EventOutbox.attempt_count))
            values = {
                "attempt_count": EventOutbox.attempt_count + 1,
                "last_error": str(error)[:2000],
                "next_attempt_at": datetime.now(timezone.utc) + timedelta(seconds=backoff),
            }
        await self.db.execute(
            update(EventOutbox).where(EventOutbox.id == outbox_id).values(**values)
        )
        await self.db.commit()

    async def delivered_count(self) -> int:
        q = select(func.count()).select_from(EventOutbox).where(EventOutbox.delivered_at.is_not(None))
        return int((await self.db.execute(q)).scalar() or 0)

    async def pending_count(self) -> int:
        return await self._pending_count()

    async def prune_delivered(
        self, batch: int = 100, *, retention_hours: float = 24.0, retention_rows: int = 10000
    ) -> int:
        """清理已投递 outbox 审计行;delivered_at IS NULL 的 pending 行永不清理。"""
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
        keep_ids = (
            select(EventOutbox.id)
            .where(EventOutbox.delivered_at.is_not(None))
            .order_by(EventOutbox.delivered_at.desc())
            .limit(max(1, int(retention_rows)))
        )
        prune_ids = (
            select(EventOutbox.id)
            .where(EventOutbox.delivered_at.is_not(None))
            .where(
                (EventOutbox.delivered_at < cutoff)
                | (~EventOutbox.id.in_(keep_ids))
            )
            .limit(max(1, int(batch)))
        )
        result = await self.db.execute(
            delete(EventOutbox).where(EventOutbox.id.in_(prune_ids))
        )
        await self.db.commit()
        return result.rowcount or 0
