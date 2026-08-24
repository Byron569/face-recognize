"""services.event_ingress — 可靠事件入口(线程安全 submit → 原子入库 + Outbox 广播)。

链路(见《融合实施.md》7.2):
    Worker transition -> 父端 SQLite spool -> EventIngress.submit() Future
        -> asyncio ingress queue(容量 delivery.ingress_queue_capacity)
        -> PostgreSQL 单事务:events UPSERT + event_outbox INSERT
        -> Future 返回持久化 ACK
        -> Parent spool 标记 delivered
        -> Outbox dispatcher 广播 WS(alert),persist_only 不广播
        -> outbox 标记 delivered / 失败退避

不变量:
    - submit() 永不阻塞调用线程;队满立即以 INGRESS_QUEUE_FULL 失败(SQLite spool 随后重试);
    - event 只能进 asyncio.Queue,不能删队首;
    - 只有异步 consumer 才创建 DB session 和执行事务;commit 后才 resolve Future;
    - 未绑定 loop / 已关闭时 submit() 立即返回失败 Future,绝不静默 return;
    - 事件与 outbox 在同一事务 commit,任何一步失败整体回滚。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Coroutine, Optional

from ..deps import AsyncSessionLocal
from ..models.event import EventType
from ..repositories.event_repo import (
    FALL_EVENT_TYPES,
    DedupeMismatchError,
    EventOutboxRepository,
    IngressQueueFullError,
    IngressRejectedError,
    OutboxCapacityError,
)

logger = logging.getLogger(__name__)

# 通用网络错误退避基秒(供 Outbox 重试)
EVENT_RETRY_BASE_SECONDS = 1.0


class IngressNotRunning(Exception):
    """submit() 时 ingress 未绑定运行中的 loop 或已关闭。"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _event_type_str(evt) -> str:
    type_str = getattr(evt, "event_type", None)
    if isinstance(type_str, EventType):
        return type_str.value
    return str(type_str or "")


def _mode_delivery(mode: str | None) -> str:
    return "persist_only" if mode == "shadow" else "alert"


class EventIngress:
    """全局可靠事件入口。实现 pose host 的 EventSinkProtocol(submit -> Future[EventSinkAck])。"""

    def __init__(
        self,
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        delivery: Optional[dict] = None,
        session_factory: Any = None,
        broadcast: Optional[Callable[[dict], Coroutine]] = None,
        mode_resolver: Optional[Callable[[str], str]] = None,
        dispatch_interval: float = 1.0,
        prune_interval: float = 120.0,
        prune_batch: int = 100,
    ) -> None:
        delivery = delivery or {}
        self._capacity = max(1, int(delivery.get("ingress_queue_capacity", 1024)))
        self._outbox_pending_capacity = max(
            1, int(delivery.get("outbox_pending_capacity", 10000))
        )
        self._resume_ratio = float(delivery.get("outbox_resume_ratio", 0.5))
        self._retention_hours = float(delivery.get("outbox_delivered_retention_hours", 24.0))
        self._retention_rows = int(delivery.get("outbox_delivered_retention_rows", 10000))
        self._loop = loop
        self._session_factory = session_factory or AsyncSessionLocal
        self._broadcast = broadcast
        self._mode_resolver = mode_resolver or (lambda camera_id: "shadow")
        self._dispatch_interval = max(0.05, float(dispatch_interval))
        self._prune_interval = max(0.05, float(prune_interval))
        self._prune_batch = max(1, int(prune_batch))

        self._queue: Optional[asyncio.Queue] = None
        self._tasks: list[asyncio.Task] = []
        self._closed = False
        self._started = False
        self._lock = threading.Lock()
        # 投递状态指标
        self._metrics = {"ingested": 0, "delivered": 0, "failed": 0, "pruned": 0}

    # ── 生命周期 ──────────────────────────────────────────

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def set_broadcast(self, broadcast: Optional[Callable[[dict], Coroutine]]) -> None:
        """设置 Outbox dispatcher 的广播回调(应把它驱动的 WS 事件推送 back)。"""
        self._broadcast = broadcast

    def set_mode_resolver(self, mode_resolver: Optional[Callable[[str], str]]) -> None:
        """设置按 camera_id 解析投递模式(alert/shadow)的回调。"""
        self._mode_resolver = mode_resolver or (lambda camera_id: "shadow")

    def start(self) -> None:
        """在已绑定的 loop 中用 call_soon_threadsafe 启动后台任务链。"""
        loop = self._loop
        if loop is None or not loop.is_running():
            raise IngressNotRunning("EventIngress must be bound to a running loop before start()")
        self._queue = asyncio.Queue(maxsize=self._capacity)
        if self._closed:
            return
        self._started = True

        def _spawn():
            if not self._started or self._closed:
                return
            self._tasks.append(loop.create_task(self._consumer_loop()))
            self._tasks.append(loop.create_task(self._dispatch_loop()))
            self._tasks.append(loop.create_task(self._prune_loop()))

        loop.call_soon_threadsafe(_spawn)

    async def close(self, deadline_s: float = 5.0) -> None:
        """停止接收 -> 有界 drain ingress -> resolve/失败所有 Future -> 停 dispatcher。"""
        self._closed = True
        self._started = False
        loop = self._loop
        if loop is None or self._queue is None:
            return
        # 发出毒丸终止 consumer;dispatch/prune 随 closed 退出
        await self._queue.put(None)
        tasks = list(self._tasks)
        self._tasks.clear()
        done, _ = await asyncio.wait(tasks, timeout=deadline_s)
        for t in done:
            if t.exception() and not isinstance(t.exception(), (asyncio.CancelledError,)):
                logger.warning("[event-ingress] background task error: %s", t.exception())
        self._queue = None

    # ── EventSink submit(线程安全,非阻塞)────────────────

    def submit(self, event):
        """入队一条事件,返回 Future[ack];队满/异常立即让 Future 失败。"""
        fut: concurrent.futures.Future = concurrent.futures.Future()
        loop = self._loop
        if loop is None or not loop.is_running() or self._closed or self._queue is None:
            fut.set_exception(IngressNotRunning("EventIngress is not accepting events"))
            return fut

        try:
            prepared = self._prepare(event)
        except IngressRejectedError as exc:
            fut.set_exception(exc)
            return fut

        def _enqueue():
            try:
                self._queue.put_nowait((fut, prepared))
            except asyncio.QueueFullError:
                try:
                    fut.set_exception(
                        IngressQueueFullError("ingress queue reached capacity")
                    )
                except Exception:  # noqa: BLE001
                    pass
            except Exception as exc:  # noqa: BLE001
                try:
                    fut.set_exception(exc)
                except Exception:  # noqa: BLE001
                    pass

        loop.call_soon_threadsafe(_enqueue)
        return fut

    # ── 校验与准备 ────────────────────────────────────────

    def _prepare(self, event) -> dict:
        """校验事件并构建持久化所需字段;非法事件抛 IngressRejectedError。"""
        event_type = _event_type_str(event)
        if event_type not in FALL_EVENT_TYPES:
            raise IngressRejectedError(f"event_type {event_type!r} not in reliable fall set")

        raw = getattr(event, "payload_json_utf8", None)
        try:
            payload = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            raise IngressRejectedError("payload_json_utf8 is not valid JSON")

        # 顶层 event_id 与 dedupe_key 必须取自 transition;payload 缺失时拒绝
        event_id = getattr(event, "event_id", "") or str(payload.get("event_id") or "")
        dedupe_key = str(payload.get("dedupe_key") or "")
        if not event_id or not dedupe_key:
            raise IngressRejectedError("reliable fall event requires event_id and dedupe_key")

        payload = self._json_safe(payload)
        camera_id = getattr(event, "camera_id", "") or "unknown"
        delivery_mode = _mode_delivery(self._mode_resolver(camera_id))
        timestamp = float(getattr(event, "timestamp", 0) or 0)
        occurred_at = (
            datetime.fromtimestamp(timestamp, tz=timezone.utc) if timestamp > 0 else None
        )
        envelope = self._envelope(event, payload)
        return {
            "event": event,
            "event_type": event_type,
            "camera_id": camera_id,
            "track_id": getattr(event, "track_id", None),
            "confidence": float(getattr(event, "confidence", 0.0) or 0.0),
            "occurred_at": occurred_at,
            "source_event_id": event_id,
            "dedupe_key": dedupe_key,
            "incident_id": payload.get("incident_id") or None,
            "payload": payload,
            "delivery_mode": delivery_mode,
            "envelope": envelope,
        }

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {k: EventIngress._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [EventIngress._json_safe(v) for v in value]
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, (datetime,)):
            return value.isoformat()
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return None
        return value

    def _envelope(self, event, payload: dict) -> dict:
        """冻结的 WS 顶层 envelope;id 在落库后由 repo 填充。"""
        ts = getattr(event, "timestamp", 0) or 0
        created = _utcnow().isoformat()
        occurred = (
            datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat() if ts else None
        )
        score = payload.get("score_semantics") or "heuristic_rule_score_not_probability"
        return {
            "type": "event",
            "event_id": getattr(event, "event_id", None),
            "dedupe_key": payload.get("dedupe_key"),
            "event_type": _event_type_str(event),
            "camera_id": getattr(event, "camera_id", None),
            "track_id": getattr(event, "track_id", None),
            "confidence": round(float(getattr(event, "confidence", 0.0) or 0.0), 6),
            "occurred_at": occurred,
            "created_at": created,
            "payload": {
                "event_id": getattr(event, "event_id", None),
                "dedupe_key": payload.get("dedupe_key"),
                "score_semantics": score,
                "track_namespace": "pose",
            },
        }

    # ── 后台任务 ──────────────────────────────────────────

    async def _consumer_loop(self) -> None:
        while not self._closed:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            fut, prepared = item
            try:
                ack = await self.ingest(prepared)
                fut.set_result(ack)
                self._metrics["ingested"] += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("[event-ingress] ingest failed event=%s: %s",
                               prepared.get("source_event_id"), exc)
                try:
                    fut.set_exception(exc)
                    self._metrics["failed"] += 1
                except Exception:  # noqa: BLE001
                    pass
            finally:
                self._queue.task_done()

    async def _dispatch_loop(self) -> None:
        while not self._closed:
            try:
                await self.dispatch_once()
            except Exception:  # noqa: BLE001
                logger.exception("[event-ingress] dispatch error")
            await asyncio.sleep(self._dispatch_interval)

    async def _prune_loop(self) -> None:
        while not self._closed:
            try:
                pruned = await self.prune()
                self._metrics["pruned"] += pruned
            except Exception:  # noqa: BLE001
                logger.exception("[event-ingress] prune error")
            await asyncio.sleep(self._prune_interval)

    # ── 可独立调用的核心方法(测试直接驱动)────────────────

    async def ingest(self, prepared: dict) -> Any:
        async with self._session_factory() as db:
            repo = EventOutboxRepository(
                db, outbox_pending_capacity=self._outbox_pending_capacity
            )
            result = await repo.ingest(
                event_type=prepared["event_type"],
                camera_id=prepared["camera_id"],
                track_id=prepared["track_id"],
                confidence=prepared["confidence"],
                occurred_at=prepared["occurred_at"],
                source_event_id=prepared["source_event_id"],
                dedupe_key=prepared["dedupe_key"],
                incident_id=prepared["incident_id"],
                payload=prepared["payload"],
                delivery_mode=prepared["delivery_mode"],
                outbox_payload=prepared["envelope"],
            )
        from ai_monitor_pose.host_protocols import EventSinkAck

        return EventSinkAck(
            event_id=prepared["source_event_id"],
            persisted=True,
            database_event_id=result.event_row_id,
        )

    async def dispatch_once(self, limit: int = 100) -> int:
        """处理一批待广播 outbox(仅 alert)。返回广播成功数。"""
        async with self._session_factory() as db:
            repo = EventOutboxRepository(db, outbox_pending_capacity=self._outbox_pending_capacity)
            rows = await repo.fetch_undelivered(limit=limit)
            delivered = 0
            for row in rows:
                if row.delivery_mode != "alert":
                    continue
                ok = await self._try_broadcast(row.payload)
                await repo.mark_delivered(row.id, error=None if ok else "broadcast failed")
                if ok:
                    delivered += 1
                    self._metrics["delivered"] += 1
        return delivered

    async def _try_broadcast(self, payload: dict) -> bool:
        if self._broadcast is None:
            # 广播未接线:视为投递成功(persist-only 语义或测试环境)
            return True
        try:
            await self._broadcast(payload)
            return True
        except Exception:  # noqa: BLE001
            logger.exception("[event-ingress] WS broadcast failed")
            return False

    async def pending_count(self) -> int:
        async with self._session_factory() as db:
            repo = EventOutboxRepository(db, outbox_pending_capacity=self._outbox_pending_capacity)
            return await repo.pending_count()

    async def delivered_count(self) -> int:
        async with self._session_factory() as db:
            repo = EventOutboxRepository(db, outbox_pending_capacity=self._outbox_pending_capacity)
            return await repo.delivered_count()

    async def prune(self) -> int:
        async with self._session_factory() as db:
            repo = EventOutboxRepository(db, outbox_pending_capacity=self._outbox_pending_capacity)
            return await repo.prune_delivered(
                self._prune_batch,
                retention_hours=self._retention_hours,
                retention_rows=self._retention_rows,
            )

    def metrics(self) -> dict:
        return dict(self._metrics)