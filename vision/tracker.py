"""
vision.tracker — 轻量 IoU 跟踪器(内核中唯一自研算法)。

职责单一: 把人脸检测结果关联成稳定 track_id。
不做识别、不做身份逻辑 —— identity 由识别任务写回 TrackResult。

设计:
    - 贪心 IoU 匹配(逐次取最大 IoU,≥ 阈值才关联)
    - hits/min_hits 确认机制(防止单帧误检产生 ID)
    - lost/max_lost 删除机制
    - 每帧输出全部活跃 track 的 TrackResult 快照
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .config import TrackConfig
from .events import FaceResult, TrackResult

logger = logging.getLogger(__name__)


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _Track:
    __slots__ = (
        "track_id", "bbox", "score", "hits", "lost",
        "confirmed", "identity", "similarity", "embedding",
    )

    def __init__(self, track_id: int, bbox: Tuple[float, float, float, float], score: float):
        self.track_id = track_id
        self.bbox = bbox
        self.score = score
        self.hits = 1
        self.lost = 0
        self.confirmed = False
        self.identity = "Unknown"
        self.similarity = 0.0
        self.embedding = None


class IoUTracker:
    """轻量 IoU 跟踪器(参数由 TrackConfig 注入)。"""

    def __init__(self, config: Optional[TrackConfig] = None):
        self._cfg = config or TrackConfig()
        self._tracks: Dict[int, _Track] = {}
        self._next_tid = 1

    # ── 主入口 ────────────────────────────────────────────

    def update(self, detections: List[FaceResult], frame_id: int) -> List[TrackResult]:
        """用本帧检测结果更新跟踪状态,返回活跃 track 快照列表。"""
        if not detections:
            for t in self._tracks.values():
                t.lost += 1
            self._prune()
            return self.snapshot()

        matched_tids: set = set()
        unmatched_dets: List[int] = list(range(len(detections)))

        if self._tracks:
            # 贪心 IoU 匹配
            while unmatched_dets:
                best_iou, best_tid, best_di = 0.0, None, None
                for tid, t in self._tracks.items():
                    if tid in matched_tids:
                        continue
                    for di in unmatched_dets:
                        iou = _iou(t.bbox, detections[di].bbox)
                        if iou > best_iou:
                            best_iou, best_tid, best_di = iou, tid, di
                if best_tid is None or best_iou < self._cfg.iou_threshold:
                    break
                det = detections[best_di]
                t = self._tracks[best_tid]
                t.bbox = det.bbox
                t.score = det.det_score
                t.hits += 1
                t.lost = 0
                if t.hits >= self._cfg.min_hits:
                    t.confirmed = True
                if det.embedding is not None:
                    t.embedding = det.embedding
                matched_tids.add(best_tid)
                unmatched_dets.remove(best_di)

        # 未匹配的 track: lost++
        for tid, t in self._tracks.items():
            if tid not in matched_tids:
                t.lost += 1

        # 未匹配的检测:新建 track(受 max_tracks 约束)
        for di in unmatched_dets:
            if len(self._tracks) >= self._cfg.max_tracks:
                break
            det = detections[di]
            tid = self._next_tid
            self._next_tid += 1
            t = _Track(tid, det.bbox, det.det_score)
            if det.embedding is not None:
                t.embedding = det.embedding
            self._tracks[tid] = t

        self._prune()
        return self.snapshot()

    def _prune(self) -> None:
        stale = [tid for tid, t in self._tracks.items() if t.lost > self._cfg.max_lost]
        for tid in stale:
            del self._tracks[tid]

    # ── 快照与身份写回 ────────────────────────────────────

    def snapshot(self) -> List[TrackResult]:
        """输出全部活跃 track 的快照(confirmed 仅作稳定性标记,不做过滤)。"""
        out = []
        for tid, t in self._tracks.items():
            out.append(
                TrackResult(
                    track_id=tid,
                    bbox=t.bbox,
                    score=t.score,
                    hits=t.hits,
                    confirmed=t.confirmed,
                    identity=t.identity,
                    similarity=t.similarity,
                    embedding=t.embedding,
                )
            )
        return out

    def get(self, track_id: int) -> Optional[_Track]:
        return self._tracks.get(track_id)

    def set_identity(self, track_id: int, identity: str, similarity: float) -> bool:
        """识别任务把结果写回 track。"""
        t = self._tracks.get(track_id)
        if t is None:
            return False
        t.identity = identity
        t.similarity = similarity
        return True

    @property
    def active_count(self) -> int:
        return len(self._tracks)
