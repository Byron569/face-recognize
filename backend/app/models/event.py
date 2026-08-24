"""事件与识别记录模型。

events 表是扩展事件(如未来跌倒检测)的统一落点;
EventType 采用 PostgreSQL 原生 enum 而非字符串,便于约束与索引。
"""

from __future__ import annotations
import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EventType(str, enum.Enum):
    recognition = "recognition"
    fall_detected = "fall_detected"      # 预留:跌倒确认
    fall_potential = "fall_potential"    # 预留:疑似跌倒
    fall_recovered = "fall_recovered"    # 预留:跌倒恢复
    intrusion = "intrusion"              # 预留:闯入
    loitering = "loitering"              # 预留:徘徊


class RecognitionLog(Base):
    """recognition_logs 表:每次识别行为记录。"""

    __tablename__ = "recognition_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    identity_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("identities.id", ondelete="SET NULL"), nullable=True
    )
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    similarity: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class Event(Base):
    """events 表:系统告警/业务事件(含确认状态)。"""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[EventType] = mapped_column(
        SAEnum(EventType, name="event_type", create_type=True), nullable=False, index=True
    )
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    identity_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("identities.id", ondelete="SET NULL"), nullable=True
    )
    confidence: Mapped[float] = mapped_column(Float, default=0)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    snapshot_path: Mapped[str | None] = mapped_column(String(512))
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    # ── 可靠事件幂等字段(跌倒检测等扩展任务)──────────────
    # source_event_id 是 Worker 生成的 transition UUID,不是数据库主键
    incident_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_event_id: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    dedupe_key: Mapped[str | None] = mapped_column(String(160), nullable=True, unique=True)
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    delivery_mode: Mapped[str] = mapped_column(String(16), nullable=False, default="alert")


class EventOutbox(Base):
    """event_outbox 表:通用事务型 outbox。

    at-least-once 广播给 WebSocket;数据库 exactly-once 由 dedupe_key 唯一约束 + 原子
    ON CONFLICT 保证。event_row_id 仅指向 events 自增主键,绝不叫 event_id。
    """

    __tablename__ = "event_outbox"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_row_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("events.id", ondelete="RESTRICT"),
        nullable=False,
    )
    dedupe_key: Mapped[str] = mapped_column(String(160), nullable=False, unique=True)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    delivery_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
