"""
services.gallery — 内存人脸底库快照。

把 PostgreSQL 中的全部 embedding 加载为 numpy 矩阵,
识别比对 = 一次矩阵点积(O(N×512)),避免逐行扫描;
底库变更(注册/删除)时由 FaceService 调用 refresh() 刷新。

这是识别热路径上的关键结构:同步、无锁竞争热点(pipeline 线程只读),
未来替换 pgvector 只需实现同接口的 provider。
"""


from __future__ import annotations
import threading
from typing import List, Optional, Tuple

import numpy as np


class FaceGallery:
    """内存底库快照(线程安全读写)。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._matrix: Optional[np.ndarray] = None      # (N, 512) float32
        self._names: List[str] = []
        self._identity_ids: List[str] = []

    def rebuild(self, rows: List[Tuple[str, str, List[float]]]) -> None:
        """rows: [(identity_id, name, embedding(512)), ...](identity_id 统一转 str)"""
        ids = [str(r[0]) for r in rows]
        names = [r[1] for r in rows]
        matrix = (
            np.stack([np.asarray(r[2], dtype=np.float32) for r in rows], axis=0)
            if rows
            else np.zeros((0, 512), dtype=np.float32)
        )
        with self._lock:
            self._matrix = matrix
            self._names = names
            self._identity_ids = ids

    def search(self, query: np.ndarray, threshold: float) -> Optional[Tuple[str, str, float]]:
        """向量化检索。返回 (identity_id, name, similarity) 或 None。"""
        with self._lock:
            matrix, names, ids = self._matrix, self._names, self._identity_ids
        if matrix is None or len(matrix) == 0:
            return None
        scores = matrix @ query.ravel()                # 已归一化 → 点积即余弦
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score < threshold:
            return None
        return ids[best_idx], names[best_idx], best_score

    @property
    def size(self) -> int:
        with self._lock:
            return 0 if self._matrix is None else len(self._matrix)
