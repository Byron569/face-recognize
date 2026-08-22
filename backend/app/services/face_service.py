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
from ..repositories.identity_repo import EmbeddingInput
from vision.config import VisionConfig

logger = logging.getLogger(__name__)


class _VideoRegistrationError(ValueError):
    """摄像头/视频注册业务错误(API 层映射 HTTP 状态码)。"""


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

    async def detect_faces(self, image_bgr: np.ndarray) -> list[dict]:
        """预览检测:返回全部人脸框/置信度/关键点/姿态比率(不入库,GPU 推理)。

        老字段 bbox/det_score 保持不变;kps/yaw_ratio/pitch_ratio 为增量,
        供摄像头实时注册的引导判定使用(PCF/alignment 一致)。
        """
        engine = self._engine_for()
        faces = engine.detect(image_bgr)
        from .video_registration import compute_pose_ratios

        out = []
        for f in faces:
            item = {
                "bbox": [float(v) for v in f.bbox],
                "det_score": float(f.det_score),
            }
            if f.kps:
                item["kps"] = [[float(k[0]), float(k[1])] for k in f.kps]
                yaw, pitch = compute_pose_ratios(f.kps)
                item["yaw_ratio"] = yaw
                item["pitch_ratio"] = pitch
            out.append(item)
        return out

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
        from pathlib import Path

        # 解析后必须位于 {project_root}/face_db/ 内,防止任意路径读取
        root = Path(resolve_project_path("face_db")).resolve()
        p = Path(resolve_project_path(pickle_path)).resolve()
        if not p.is_relative_to(root):
            raise ValueError(f"路径必须在 face_db/ 目录内: {pickle_path}")
        if not os.path.exists(p):
            return 0
        with open(p, "rb") as f:
            content = f.read()
        if b"\n" in content:
            _, payload = content.split(b"\n", 1)
        else:
            payload = content
        try:
            records = pickle.loads(payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[face-service] pickle 解析失败: %s", pickle_path)
            raise ValueError("pickle 文件解析失败,格式不合法") from exc
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

    async def refresh_gallery(self) -> None:
        """底库变更后刷新内存快照(公开入口,供 API 层调用)。"""
        await self._refresh_gallery()

    # ── 摄像头/视频实时注册(state 无关: analyze 不写库,commit 复验后原子入库) ──

    def _video_cfg(self) -> dict:
        from ..config import build_camera_config

        registration = (
            build_camera_config("desktop").get("vision", {}).get("registration", {}) or {}
        )
        return registration.get("video", {}) or {}

    async def analyze_registration_frames(
        self, frames: list[tuple[np.ndarray, str, int, str]]
    ) -> dict:
        """批量质量分析(不写库)。

        frames: [(image_bgr, frame_id, timestamp_ms, pose), ...]
        返回 {"sampled", "accepted", "rejected", "recommended_frame_ids"}
        """
        cfg = self._video_cfg()
        max_analyze = max(1, int(cfg.get("max_analyze_frames", 30)))
        if len(frames) > max_analyze:
            raise _VideoRegistrationError("too_many_frames")
        from .video_registration import analyze_face_result, select_diverse_candidates

        engine = self._engine_for()
        accepted: list = []
        rejected: list = []
        for image_bgr, frame_id, ts, pose in frames:
            faces = engine.detect(image_bgr)
            from .video_registration import CandidateFrame

            r = analyze_face_result(faces, image_bgr, frame_id, ts, pose, cfg)
            if isinstance(r, CandidateFrame):
                accepted.append(r)
            else:
                rejected.append({
                    "frame_id": r.frame_id,
                    "timestamp_ms": r.timestamp_ms,
                    "accepted": False,
                    "reason": r.reason,
                })
        recommended = select_diverse_candidates(
            accepted,
            target_per_pose=max(1, int(cfg.get("target_per_pose", 2))),
            duplicate_similarity=float(cfg.get("duplicate_similarity", 0.94)),
        )
        recommended_ids = [c.frame_id for c in recommended]
        return {
            "sampled": len(frames),
            "accepted": accepted,
            "rejected": rejected,
            "recommended_frame_ids": recommended_ids,
        }

    async def commit_registration_frames(
        self,
        *,
        mode: str,
        name: str | None,
        notes: str | None,
        identity_id: uuid.UUID | None,
        frames: list[tuple[np.ndarray, str, int, str]],
    ) -> dict:
        """复验 + 原子入库。

        重新分析(不信任前端)→ accepted < min_submit_frames 抛错;
        超 max_submit_frames 截断→ create/append 原子入库→ gallery 刷新恰一次。
        """
        cfg = self._video_cfg()
        min_submit = max(1, int(cfg.get("min_submit_frames", 3)))
        max_submit = max(min_submit, int(cfg.get("max_submit_frames", 8)))
        from .video_registration import analyze_face_result, CandidateFrame

        engine = self._engine_for()
        accepted: list[CandidateFrame] = []
        for image_bgr, frame_id, ts, pose in frames:
            faces = engine.detect(image_bgr)
            r = analyze_face_result(faces, image_bgr, frame_id, ts, pose, cfg)
            if isinstance(r, CandidateFrame):
                accepted.append(r)
            if len(accepted) >= max_submit:
                break
        if len(accepted) < min_submit:
            raise _VideoRegistrationError("too_few_frames")
        selected = accepted[:max_submit]

        items = [
            EmbeddingInput(
                embedding=c.embedding,
                source="camera",
                quality_score=c.quality_score,
            )
            for c in selected
        ]
        if mode == "create":
            if not name:
                raise _VideoRegistrationError("name_required")
            identity = Identity(name=name.strip(), notes=notes or "")
            identity = await self._repo.create_with_embeddings(identity, items)
            await self.refresh_gallery()
            return {
                "mode": "create",
                "identity_id": identity.id,
                "embedding_count_added": len(items),
            }
        else:  # append
            identity = await self._repo.get(identity_id)
            if identity is None:
                raise _VideoRegistrationError("identity_not_found")
            await self._repo.add_many_embeddings(identity_id, items)
            await self.refresh_gallery()
            return {
                "mode": "append",
                "identity_id": identity_id,
                "embedding_count_added": len(items),
            }
