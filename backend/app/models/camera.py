"""摄像头配置模型。"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Camera(Base):
    """cameras 表:摄像头配置。config 为 JSONB 个性化参数(叠加在 profile 之上)。"""

    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), default="")
    source: Mapped[str] = mapped_column(String(512), nullable=False)  # 0/1 或 rtsp://...
    width: Mapped[int] = mapped_column(Integer, default=640)
    height: Mapped[int] = mapped_column(Integer, default=480)
    profile: Mapped[str] = mapped_column(String(32), default="desktop")
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
