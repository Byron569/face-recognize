"""
vision.tracker — ByteTrack 多目标跟踪器。

移植自开源算法 ByteTrack(IFzhang/ByteTrack, MIT License):
    https://github.com/ifzhang/ByteTrack
核心思想:
    1. 卡尔曼滤波对每个 track 的运动状态做预测;
    2. 两阶段数据关联 —— 高置信度检测先与全部 track 做 IoU 匹配,
       未被匹配的低置信度检测再与"已确认"track 做二次匹配(ByteTrack 的关键,
       保留遮挡/模糊等低分框,降低 ID switch);
    3. lost 缓冲复活 + 超时删除。

对外接口与原 IoUTracker 完全一致,业务层(pipeline / 识别任务)零改动:
    update(detections, frame_id) -> List[TrackResult]
    snapshot() / set_identity() / active_count

数据格式: bbox 统一 (x1, y1, x2, y2);内部跟踪用 tlwh + 卡尔曼 8 维状态。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TrackConfig
from .events import FaceResult, TrackResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 卡尔曼滤波(ByteTrack 标准 8 维状态)
#   状态: [cx, cy, aspect_ratio, h, vx, vy, v_aspect, vh]
#   观测: [cx, cy, aspect_ratio, h](由 tlwh 转换而来)
# ──────────────────────────────────────────────────────────────

class KalmanFilter:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        """新轨迹初始化:位置=观测,速度=0,协方差按尺度给先验。"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        projected_mean, projected_cov = self.project(mean, covariance)
        # 卡尔曼增益 K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹
        kalman_gain = np.linalg.multi_dot(
            (covariance, self._update_mat.T, np.linalg.inv(projected_cov))
        )
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


# ──────────────────────────────────────────────────────────────
# 关联工具
# ──────────────────────────────────────────────────────────────

def _bbox_to_tlwh(bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=float)


def _tlwh_to_tlbr(tlwh) -> Tuple[float, float, float, float]:
    x, y, w, h = tlwh
    return (x, y, x + w, y + h)


def _iou_tlbr(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_distance(tracks: List["STrack"], dets: List[Tuple[np.ndarray, float, object]]) -> np.ndarray:
    """成本矩阵 = 1 - IoU(track 预测框, 检测框)。"""
    if not tracks or not dets:
        return np.zeros((len(tracks), len(dets)), dtype=float)
    rows = [_tlwh_to_tlbr(t.pred_tlwh) for t in tracks]
    cols = [_tlwh_to_tlbr(d[0]) for d in dets]
    cost = np.zeros((len(rows), len(cols)), dtype=float)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            cost[i, j] = 1.0 - _iou_tlbr(r, c)
    return cost


def _linear_assignment(cost: np.ndarray, thresh: float):
    """匈牙利匹配(成本 ≤ thresh 才关联),返回 matches / 未匹配行 / 未匹配列。"""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    matches, u_rows, u_cols = [], [], []
    cost = np.asarray(cost, dtype=float)
    rows, cols = linear_sum_assignment(cost)
    matched_rows, matched_cols = set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] > thresh:
            continue
        matches.append((r, c))
        matched_rows.add(r)
        matched_cols.add(c)
    u_rows = [r for r in range(cost.shape[0]) if r not in matched_rows]
    u_cols = [c for c in range(cost.shape[1]) if c not in matched_cols]
    return matches, u_rows, u_cols


# ──────────────────────────────────────────────────────────────
# 轨迹状态
# ──────────────────────────────────────────────────────────────

class _TrackState:
    NEW = 0          # 新建未确认
    TRACKED = 1      # 已确认
    LOST = 2         # 丢失(缓冲期内可复活)
    REMOVED = 3      # 已删除


class STrack:
    """单条轨迹:卡尔曼状态 + 匹配计数 + 识别身份写回。"""

    __slots__ = (
        "track_id", "mean", "covariance", "score", "state", "tracklet_len",
        "frame_id", "start_frame", "identity", "similarity", "embedding",
        "_tlwh", "_kf", "_confirm_hits", "_ever_confirmed",
    )

    def __init__(self, tlwh: np.ndarray, score: float, track_id: int, confirm_hits: int = 3):
        self.track_id = track_id
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.score = score
        self.state = _TrackState.NEW
        self.tracklet_len = 0
        self.frame_id = 0
        self.start_frame = 0
        self.identity = "Unknown"
        self.similarity = 0.0
        self.embedding = None
        self._confirm_hits = max(1, confirm_hits)
        self._ever_confirmed = False
        self._kf = KalmanFilter()
        self.mean, self.covariance = None, None

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is not None:
            ret = self.mean[:4].copy()
            ret[2] *= ret[3]  # aspect_ratio * h = w
            ret[:2] -= ret[2:] / 2  # cx - w/2, cy - h/2
            return ret
        return self._tlwh.copy()

    @property
    def pred_tlwh(self) -> np.ndarray:
        """预测位置(用于关联),未初始化时返回观测框。"""
        return self.tlwh

    def predict(self) -> None:
        if self.mean is None:
            self.mean, self.covariance = self._kf.initiate(self._xyah_of(self._tlwh))
        else:
            self.mean, self.covariance = self._kf.predict(self.mean, self.covariance)

    def activate(self, frame_id: int) -> None:
        self.state = _TrackState.TRACKED
        self._ever_confirmed = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_tlwh: np.ndarray, score: float, frame_id: int) -> None:
        """已确认过的 lost 轨迹被重新关联 → 复活为 TRACKED。"""
        self._update_state(new_tlwh, score)
        self.tracklet_len = 0
        self.state = _TrackState.TRACKED
        self._ever_confirmed = True
        self.frame_id = frame_id

    def update(self, new_tlwh: np.ndarray, score: float, frame_id: int) -> None:
        self._update_state(new_tlwh, score)
        self.tracklet_len += 1
        self.frame_id = frame_id
        if not self._ever_confirmed and self.tracklet_len >= self._confirm_hits:
            self.activate(frame_id)

    def mark_lost(self) -> None:
        self.state = _TrackState.LOST

    def _update_state(self, new_tlwh: np.ndarray, score: float) -> None:
        self.score = score
        self._tlwh = np.asarray(new_tlwh, dtype=float)
        measurement = self._xyah_of(self._tlwh)
        if self.mean is None:
            self.mean, self.covariance = self._kf.initiate(measurement)
        else:
            self.mean, self.covariance = self._kf.update(self.mean, self.covariance, measurement)

    @staticmethod
    def _xyah_of(tlwh: np.ndarray) -> np.ndarray:
        """tlwh → [cx, cy, aspect, h](卡尔曼观测)。"""
        x, y, w, h = tlwh
        return np.asarray([x + w / 2, y + h / 2, w / max(h, 1e-6), h], dtype=float)

    @property
    def is_confirmed(self) -> bool:
        return self.state == _TrackState.TRACKED

    def to_result(self) -> TrackResult:
        return TrackResult(
            track_id=self.track_id,
            bbox=_tlwh_to_tlbr(self.tlwh),
            score=float(self.score),
            hits=self.tracklet_len,
            confirmed=self.is_confirmed,
            identity=self.identity,
            similarity=self.similarity,
            embedding=self.embedding,
        )


# ──────────────────────────────────────────────────────────────
# ByteTrack 主跟踪器
# ──────────────────────────────────────────────────────────────

class ByteTracker:
    """ByteTrack 人脸跟踪器(参数由 TrackConfig 注入,接口与原 IoUTracker 对齐)。"""

    def __init__(self, config: Optional[TrackConfig] = None):
        self._cfg = config or TrackConfig()
        self._tracked: List[STrack] = []   # NEW + TRACKED
        self._lost: List[STrack] = []      # LOST(缓冲期内)
        self._removed: List[STrack] = []
        self._frame_id = 0
        self._next_tid = 1
        # 丢失保留时长(帧)= frame_rate / 30 * track_buffer(ByteTrack 官方语义)
        self._max_time_lost = max(1, int(self._cfg.frame_rate / 30.0 * self._cfg.track_buffer))
        self._confirm_hits = max(1, self._cfg.min_hits)

    # ── 主入口 ────────────────────────────────────────────

    def update(self, detections: List[FaceResult], frame_id: int) -> List[TrackResult]:
        self._frame_id += 1

        # 1. 预测全部存活轨迹
        for t in self._tracked + self._lost:
            t.predict()

        # 2. 拆分高/低置信度检测
        dets_high: List[Tuple[np.ndarray, float, object]] = []
        dets_low: List[Tuple[np.ndarray, float, object]] = []
        for d in detections:
            tlwh = _bbox_to_tlwh(d.bbox)
            if d.det_score >= self._cfg.track_thresh:
                dets_high.append((tlwh, d.det_score, d.embedding))
            elif d.det_score >= self._cfg.low_thresh:
                dets_low.append((tlwh, d.det_score, d.embedding))
            # 低于 low_thresh 直接丢弃(噪声)

        # 3. 第一次关联:高分检测 × 全部轨迹(含 lost,用于复活)
        strack_pool = self._tracked + self._lost
        cost = _iou_distance(strack_pool, dets_high)
        matches, u_track, u_det_high = _linear_assignment(cost, thresh=self._cfg.match_thresh)

        for it, idet in matches:
            track = strack_pool[it]
            tlwh, score, emb = dets_high[idet]
            if emb is not None:
                track.embedding = emb
            if track.state in (_TrackState.TRACKED, _TrackState.NEW):
                track.update(tlwh, score, self._frame_id)  # update 内部按 min_hits 确认
            else:  # LOST
                if track._ever_confirmed:
                    track.re_activate(tlwh, score, self._frame_id)
                else:
                    # 从未确认(降频检测下 NEW 曾短暂丢失)→ 继续累计确认
                    track.update(tlwh, score, self._frame_id)

        # 4. 第二次关联:低分检测 × 第一次未匹配的"已确认"轨迹
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == _TrackState.TRACKED]
        if dets_low:
            cost2 = _iou_distance(r_tracked, dets_low)
            matches2, _, _ = _linear_assignment(cost2, thresh=0.5)
            for it, idet in matches2:
                track = r_tracked[it]
                tlwh, score, emb = dets_low[idet]
                if emb is not None:
                    track.embedding = emb
                track.update(tlwh, score, self._frame_id)

        # 5. 第一次未匹配的轨迹 → 进入 lost(降频检测下 NEW 也保留确认机会,超时删除)
        for i in u_track:
            track = strack_pool[i]
            if track.state != _TrackState.LOST:
                track.mark_lost()

        # 6. 第一次未匹配的高分检测 → 新建轨迹(受 max_tracks 约束)
        for i in u_det_high:
            if self._active_count() >= self._cfg.max_tracks:
                break
            tlwh, score, emb = dets_high[i]
            track = STrack(tlwh, score, self._next_tid, confirm_hits=self._confirm_hits)
            self._next_tid += 1
            if emb is not None:
                track.embedding = emb
            track.tracklet_len = 1
            if track.tracklet_len >= self._confirm_hits:
                track.activate(self._frame_id)
            self._tracked.append(track)

        # 7. 重组列表:lost 归位 / 复活回 tracked / 超时删除(不在遍历时修改原列表)
        new_tracked: List[STrack] = []
        new_lost: List[STrack] = []
        for t in self._tracked:
            if t.state == _TrackState.LOST:
                new_lost.append(t)
            elif t.state != _TrackState.REMOVED:
                new_tracked.append(t)
        for t in self._lost:
            if t.state == _TrackState.TRACKED:
                new_tracked.append(t)  # 本帧被复活
            elif t.state == _TrackState.REMOVED:
                continue
            elif self._frame_id - t.frame_id >= self._max_time_lost:
                t.state = _TrackState.REMOVED
            else:
                new_lost.append(t)
        self._tracked = new_tracked
        self._lost = new_lost

        return self.snapshot()

    def _active_count(self) -> int:
        return len(self._tracked) + len(self._lost)

    # ── 快照与身份写回 ────────────────────────────────────

    def snapshot(self) -> List[TrackResult]:
        out: List[TrackResult] = []
        for t in self._tracked:
            if t.state in (_TrackState.NEW, _TrackState.TRACKED):
                out.append(t.to_result())
        for t in self._lost:
            out.append(t.to_result())
        return out

    def get(self, track_id: int) -> Optional[STrack]:
        for t in self._tracked + self._lost:
            if t.track_id == track_id:
                return t
        return None

    def set_identity(self, track_id: int, identity: str, similarity: float) -> bool:
        t = self.get(track_id)
        if t is None:
            return False
        t.identity = identity
        t.similarity = similarity
        return True

    @property
    def active_count(self) -> int:
        return self._active_count()


# 向后兼容别名: 原内核导出名 IoUTracker 现在指向 ByteTracker。
IoUTracker = ByteTracker
