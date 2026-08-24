"""阶段8b — 后端测试 3:EventOutboxRepository 可靠落库(需测试库)。

覆盖事务型 outbox 核心不变量:
    - 事件与 outbox 单事务原子写入;
    - 相同 dedupe_key 仅一条(原子 ON CONFLICT);
    - 同一 dedupe_key 绑定不同 source_event_id 抛 DedupeMismatchError;
    - shadow(mode=persist_only)落库即视为已投递,不占 outbox 容量;alert 保持 pending;
    - outbox pending 达容量时抛 OutboxCapacityError,不删旧告警;
    - EventIngress.ingest 返回持久化 ACK(EventSinkAck.persisted=True)。
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.eventdb


def _run(loop, coro):
    """在共享 fall_loop 上串行执行,避免跨 loop 复用 async engine 连接。"""
    return loop.run_until_complete(coro)


def _args(**overrides):
    base = dict(
        event_type="fall_detected",
        camera_id="cam-x",
        track_id=2,
        confidence=0.88,
        occurred_at=datetime.now(timezone.utc),
        source_event_id="evt-100",
        dedupe_key="key-100",
        incident_id="inc-100",
        payload={"event_id": "evt-100", "dedupe_key": "key-100", "score_semantics": "rule"},
        delivery_mode="alert",
        outbox_payload={"type": "event", "event_id": "evt-100", "dedupe_key": "key-100"},
    )
    base.update(overrides)
    return base


def test_ingest_stores_event_and_outbox_atomically(fall_db, fall_loop) -> None:
    async def body(factory):
        from backend.app.models.event import EventOutbox
        from backend.app.repositories.event_repo import Event, EventOutboxRepository
        from sqlalchemy import func, select

        async with factory() as db:
            repo = EventOutboxRepository(db)
            res = await repo.ingest(**_args())
            assert res.inserted is True
            assert res.event_row_id > 0
        async with factory() as db:
            ev = (await db.execute(select(Event))).scalars().first()
            assert ev is not None and ev.dedupe_key == "key-100"
            out = (await db.execute(select(EventOutbox))).scalars().first()
            assert out is not None and out.event_row_id == ev.id

    _run(fall_loop, body(fall_db))


def test_same_dedupe_key_inserts_only_once(fall_db, fall_loop) -> None:
    async def body(factory):
        from backend.app.models.event import EventOutbox
        from backend.app.repositories.event_repo import Event, EventOutboxRepository
        from sqlalchemy import func, select

        async with factory() as db:
            repo = EventOutboxRepository(db)
            first = await repo.ingest(**_args())
            second = await repo.ingest(**_args())
            assert first.inserted is True
            assert second.inserted is False
            assert first.event_row_id == second.event_row_id
        async with factory() as db:
            n_ev = (
                await db.execute(
                    select(func.count()).select_from(Event).where(Event.dedupe_key == "key-100")
                )
            ).scalar()
            n_out = (
                await db.execute(
                    select(func.count())
                    .select_from(EventOutbox)
                    .where(EventOutbox.dedupe_key == "key-100")
                )
            ).scalar()
            assert n_ev == 1 and n_out == 1

    _run(fall_loop, body(fall_db))


def test_same_dedupe_key_different_source_event_id_raises(fall_db, fall_loop) -> None:
    from backend.app.repositories.event_repo import DedupeMismatchError, EventOutboxRepository

    async def body(factory):
        async with factory() as db:
            repo = EventOutboxRepository(db)
            await repo.ingest(**_args())
        async with factory() as db:
            repo = EventOutboxRepository(db)
            with pytest.raises(DedupeMismatchError):
                await repo.ingest(**_args(source_event_id="evt-DIFFERENT"))

    _run(fall_loop, body(fall_db))


def test_shadow_persists_without_pending_outbox(fall_db, fall_loop) -> None:
    from backend.app.repositories.event_repo import EventOutboxRepository
    from sqlalchemy import select

    async def body(factory):
        async with factory() as db:
            repo = EventOutboxRepository(db)
            await repo.ingest(**_args(delivery_mode="persist_only"))
        async with factory() as db:
            from backend.app.models.event import EventOutbox

            out = (await db.execute(select(EventOutbox))).scalars().first()
            assert out is not None
            assert out.delivery_mode == "persist_only"
            assert out.delivered_at is not None  # 落库即视为已投递

    _run(fall_loop, body(fall_db))


def test_alert_outbox_stays_pending_and_fetchable(fall_db, fall_loop) -> None:
    from backend.app.repositories.event_repo import EventOutboxRepository

    async def body(factory):
        async with factory() as db:
            repo = EventOutboxRepository(db)
            await repo.ingest(**_args(delivery_mode="alert", dedupe_key="key-alert"))
        async with factory() as db:
            repo = EventOutboxRepository(db)
            pending = await repo.fetch_undelivered(limit=10)
            assert len(pending) == 1 and pending[0].delivery_mode == "alert"
            assert pending[0].delivered_at is None

    _run(fall_loop, body(fall_db))


def test_outbox_capacity_raises_outbox_capacity_error(fall_db, fall_loop) -> None:
    from backend.app.repositories.event_repo import EventOutboxRepository, OutboxCapacityError

    async def body(factory):
        # 容量 1:先占一个 pending,第二条即超容量
        async with factory() as db:
            repo = EventOutboxRepository(db, outbox_pending_capacity=1)
            await repo.ingest(**_args(delivery_mode="alert", dedupe_key="cap-1"))
        async with factory() as db:
            repo = EventOutboxRepository(db, outbox_pending_capacity=1)
            with pytest.raises(OutboxCapacityError):
                await repo.ingest(**_args(delivery_mode="alert", dedupe_key="cap-2"))

    _run(fall_loop, body(fall_db))


def test_event_ingress_ingest_acks_persisted(fall_db, fall_loop) -> None:
    def make_event():
        payload = {
            "event_id": "evt-500",
            "dedupe_key": "key-500",
            "incident_id": "inc-500",
            "score_semantics": "heuristic_rule_score_not_probability",
        }
        return SimpleNamespace(
            event_type="fall_potential",
            camera_id="cam-y",
            track_id=7,
            confidence=0.6,
            timestamp=1700000000.0,
            payload_json_utf8=json.dumps(payload).encode("utf-8"),
        )

    async def body(session_factory):
        from backend.app.services.event_ingress import EventIngress

        ingress = EventIngress(session_factory=session_factory)
        prepared = ingress._prepare(make_event())
        ack = await ingress.ingest(prepared)
        assert ack.persisted is True
        assert ack.event_id == "evt-500"
        assert ack.database_event_id is not None

    _run(fall_loop, body(fall_db))