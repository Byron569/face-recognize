"""
services.face_service — 人脸库业务。

注册/追加特征:用共享引擎(EnginePool)提取 embedding —— 不再每次请求重载模型;
底库变更后自动刷新内存快照(FaceGallery),识别热路径即时生效;
搜索走向量化 numpy 比对(FaceGallery)。
"""


from __future__ import annotations
import logging
import uuid
from typing import Any, List, Optional, Tuple

import numpy as np

from ..config import resolve_project_path
from ..models.identity import Identity
from ..repositories.identity_repo import IdentityRepository
from ..services.gallery import FaceGallery
from ..services.model_manager import EnginePool, get_engine_pool
from vision.config import VisionConfig

logger = logging.getLogger(__name__)


class FaceService:
    def __init__(self, db, gallery: FaceGallery, engine_pool: EnginePool):
        self.db = db
        self._repo = IdentityRepository(db)
        self._gallery = gallery
        self._engine_pool = engine_pool

    # ── 引擎(共享,不重复加载)─────────────────────────────

    def _engine_for(self, profile: str = "desktop"):
        # 用 build_camera_config 加载:models_root 等相对路径会解析为绝对路径
        from ..config import build_camera_config

        cfg = VisionConfig.from_dict(build_camera_config(profile).get("vision", {}))
        return self._engine_pool.get(cfg)

    async def extract_embedding(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """从图片提取最大脸的 512-d 归一化 embedding(带注册质量筛选)。

        只接受:检测置信度 ≥ min_det_score 且人脸框最短边 ≥ min_face_size 的人脸,
        防止低质量 embedding(小脸/模糊/低置信度)污染底库。
        """
        engine = self._engine_for()
        faces = engine.detect(image_bgr)
        if not faces:
            return None
        reg = self._registration_cfg()
        min_size = int(reg.get("min_face_size", 48))
        min_score = float(reg.get("min_det_score", 0.0))
        candidates = [
            f for f in faces
            if f.det_score >= min_score and min(f.width, f.height) >= min_size
        ]
        if not candidates:
            return None
        largest = max(candidates, key=lambda f: f.width * f.height)
        return largest.embedding

    def _registration_cfg(self) -> dict:
        """注册质量筛选配置(来自 default/profile 级联的 vision.registration 节)。"""
        from ..config import build_camera_config

        vision_cfg = build_camera_config("desktop").get("vision", {}) or {}
        return vision_cfg.get("registration", {}) or {}

    # ── 身份 CRUD ─────────────────────────────────────────

    async def list_identities(self, page: int = 1, page_size: int = 20, search: str = ""):
        return await self._repo.list(page, page_size, search)

    async def get_identity(self, identity_id: uuid.UUID) -> Optional[Identity]:
        return await self._repo.get(identity_id)

    async def create_identity(self, name: str, embedding: list[float], source: str = "image", notes: str = "") -> Identity:
        identity = Identity(name=name, notes=notes)
        identity = await self._repo.create(identity, embeddings=[embedding])
        await self._refresh_gallery()
        return identity

    async def update_identity(self, identity_id: uuid.UUID, data: dict) -> Optional[Identity]:
        identity = await self._repo.get(identity_id)
        if identity is None:
            return None
        return await self._repo.update(identity, data)

    async def delete_identity(self, identity_id: uuid.UUID) -> bool:
        ok = await self._repo.delete(identity_id)
        if ok:
            await self._refresh_gallery()
        return ok

    async def add_embedding(self, identity_id: uuid.UUID, embedding: list[float], source: str = "image") -> bool:
        ok = await self._repo.add_embedding(identity_id, embedding, source)
        if ok:
            await self._refresh_gallery()
        return ok

    async def delete_embedding(self, embedding_id: uuid.UUID) -> bool:
        ok = await self._repo.delete_embedding(embedding_id)
        if ok:
            await self._refresh_gallery()
        return ok

    # ── 检索 ──────────────────────────────────────────────

    async def search_by_embedding(
        self, embedding: list[float], threshold: float = 0.4
    ) -> Tuple[Optional[str], Optional[str], float]:
        """返回 (identity_id(str), name, similarity)。"""
        hit = self._gallery.search(np.asarray(embedding, dtype=np.float32), threshold)
        if hit is None:
            return None, None, 0.0
        return str(hit[0]), hit[1], hit[2]

    async def import_pickle(self, pickle_path: str) -> int:
        """一次性导入旧版 pickle 底库(兼容旧数据迁移)。"""
        import os
        import pickle

        path = resolve_project_path(pickle_path)
        if not os.path.exists(path):
            return 0
        with open(path, "rb") as f:
            content = f.read()
        if b"\n" in content:
            _, payload = content.split(b"\n", 1)
        else:
            payload = content
        records = pickle.loads(payload)
        count = 0
        for rec in records:
            name = rec.get("name")
            emb = rec.get("embedding")
            if not name or emb is None:
                continue
            existing = await self._repo.get_by_name(name)
            emb_list = np.asarray(emb, dtype=np.float32).ravel().tolist()
            if existing is not None:
                await self._repo.add_embedding(existing.id, emb_list, source="import")
            else:
                await self._repo.create(Identity(name=name), embeddings=[emb_list])
            count += 1
        if count:
            await self._refresh_gallery()
        return count

    # ── 内部 ──────────────────────────────────────────────

    async def _refresh_gallery(self) -> None:
        rows = await self._repo.all_embeddings()
        self._gallery.rebuild(rows)
        logger.info("[face-service] gallery refreshed: %s embeddings", len(rows))
