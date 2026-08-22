"""人脸底库数据访问(identities + identity_embeddings)。"""

from __future__ import annotations
import uuid
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.identity import Identity, IdentityEmbedding


@dataclass(frozen=True)
class EmbeddingInput:
    """待写入的一条 embedding(来源 + 质量分)。"""

    embedding: list[float]
    source: str = "image"
    quality_score: float = 0.0


class IdentityRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ── 身份 CRUD ─────────────────────────────────────────

    async def list(self, page: int = 1, page_size: int = 20, search: str = "") -> tuple[list[Identity], int]:
        q = select(Identity).options(selectinload(Identity.embeddings))
        if search:
            q = q.where(Identity.name.ilike(f"%{search}%"))
        count_q = select(func.count(Identity.id))
        if search:
            count_q = count_q.where(Identity.name.ilike(f"%{search}%"))
        total = (await self.db.execute(count_q)).scalar() or 0
        q = q.order_by(Identity.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
        rows = (await self.db.execute(q)).scalars().unique().all()
        return list(rows), total

    async def get(self, identity_id: uuid.UUID) -> Optional[Identity]:
        q = (
            select(Identity)
            .options(selectinload(Identity.embeddings))
            .where(Identity.id == identity_id)
        )
        return (await self.db.execute(q)).scalar_one_or_none()

    async def get_by_name(self, name: str) -> Optional[Identity]:
        q = select(Identity).where(Identity.name == name).limit(1)
        return (await self.db.execute(q)).scalar_one_or_none()

    async def create(self, identity: Identity, embeddings: list[list[float]] | None = None) -> Identity:
        self.db.add(identity)
        await self.db.flush()
        for emb in embeddings or []:
            self.db.add(
                IdentityEmbedding(identity_id=identity.id, embedding=emb, source="image")
            )
        await self.db.commit()
        await self.db.refresh(identity)
        return identity

    async def update(self, identity: Identity, data: dict) -> Identity:
        for key, value in data.items():
            if value is not None and hasattr(identity, key):
                setattr(identity, key, value)
        await self.db.commit()
        await self.db.refresh(identity)
        return identity

    async def delete(self, identity_id: uuid.UUID) -> bool:
        result = await self.db.execute(delete(Identity).where(Identity.id == identity_id))
        await self.db.commit()
        return (result.rowcount or 0) > 0

    # ── embedding 管理 ────────────────────────────────────

    async def add_embedding(self, identity_id: uuid.UUID, embedding: list[float], source: str = "image") -> bool:
        identity = await self.db.get(Identity, identity_id)
        if identity is None:
            return False
        self.db.add(IdentityEmbedding(identity_id=identity_id, embedding=embedding, source=source))
        await self.db.commit()
        return True

    async def delete_embedding(self, embedding_id: uuid.UUID) -> bool:
        result = await self.db.execute(
            delete(IdentityEmbedding).where(IdentityEmbedding.id == embedding_id)
        )
        await self.db.commit()
        return (result.rowcount or 0) > 0

    # ── 原子批量写入(摄像头/视频注册用)────────────────────
    @staticmethod
    def _embedding_rows(
        identity_id: uuid.UUID, items: list[EmbeddingInput]
    ) -> list[IdentityEmbedding]:
        return [
            IdentityEmbedding(
                identity_id=identity_id,
                embedding=list(item.embedding),
                source=item.source,
                quality_score=item.quality_score,
            )
            for item in items
        ]

    async def create_with_embeddings(
        self, identity: Identity, items: list[EmbeddingInput]
    ) -> Identity:
        """原子创建身份 + 全部 embedding(单事务);items 为空时 rollback 并抛错。"""
        if not items:
            raise ValueError("create_with_embeddings requires at least one embedding")
        try:
            self.db.add(identity)
            await self.db.flush()
            self.db.add_all(self._embedding_rows(identity.id, items))
            await self.db.commit()
            await self.db.refresh(identity)
            return identity
        except Exception:
            await self.db.rollback()
            raise

    async def add_many_embeddings(
        self, identity_id: uuid.UUID, items: list[EmbeddingInput]
    ) -> bool:
        """原子批量追加 embedding;身份不存在返回 False;单事务,异常 rollback。"""
        if not items:
            return True
        existing = await self.db.get(Identity, identity_id)
        if existing is None:
            return False
        try:
            self.db.add_all(self._embedding_rows(identity_id, items))
            await self.db.commit()
            return True
        except Exception:
            await self.db.rollback()
            raise

    async def all_embeddings(self) -> list[tuple[uuid.UUID, str, list[float]]]:
        """拉取全部 (identity_id, name, embedding),供内存底库快照与向量化检索。"""
        q = (
            select(IdentityEmbedding.identity_id, Identity.name, IdentityEmbedding.embedding)
            .join(Identity, Identity.id == IdentityEmbedding.identity_id)
        )
        rows = (await self.db.execute(q)).all()
        return [(r[0], r[1], list(r[2])) for r in rows]
