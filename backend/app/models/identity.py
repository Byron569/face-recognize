"""人脸底库模型:identities + identity_embeddings。"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, Uuid
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Identity(Base):
    """identities 表:注册人脸身份。"""

    __tablename__ = "identities"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    avatar_path: Mapped[str | None] = mapped_column(String(512))
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    embeddings: Mapped[list["IdentityEmbedding"]] = relationship(
        back_populates="identity", cascade="all, delete-orphan"
    )


class IdentityEmbedding(Base):
    """identity_embeddings 表:每个身份可有多条 512 维特征(多角度注册)。"""

    __tablename__ = "identity_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    identity_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("identities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    embedding = mapped_column(ARRAY(Float), nullable=False)  # float[512],L2 归一化
    quality_score: Mapped[float] = mapped_column(Float, default=0)
    source: Mapped[str] = mapped_column(String(32), default="image")  # image / batch / camera
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    identity: Mapped["Identity"] = relationship(back_populates="embeddings")
