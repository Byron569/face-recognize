"""每摄像头独立的无模型 PoseTracker（第 6.6 节）。

采用“高低置信度两阶段匹配 + 常速度预测 + Hungarian 分配”；
不调用 model.track(persist=True)，不依赖任何模型隐状态。
轨迹键由外部调用方以 (camera_id, camera_session_id, pose_track_id) 限定，
本实例内部只管理自身局部 track_id。不产生任何事件（事件由状态机负责）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import TrackerConfig
from ..contracts import PoseDetectionV1, PoseStateV1, PoseTrackV1

_KP_CONF = 0.3
_MIN_VISIBLE = 5
_IOU_WEIGHT = 1.0
_KPDIST_WEIGHT = 1.5
_CENTER_WEIGHT = 0.5


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _visible(kps) -> int:
    n = 0
    for x, y, s in kps:
        if (s is not None and s >= _KP_CONF and x is not None and y is not None
                and math.isfinite(float(x)) and math.isfinite(float(y))):
            n += 1
    return n


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _quality(det: PoseDetectionV1) -> str:
    if det.detection_score >= 0.35 and _visible(det.keypoints_coco17) >= _MIN_VISIBLE:
        return "GOOD"
    if det.detection_score >= 0.15 and _visible(det.keypoints_coco17) >= _MIN_VISIBLE:
        return "PARTIAL"
    return "INVALID"


@dataclass
class _Track:
    track_id: int
    box: tuple[float, float, float, float]
    prev_box: tuple[float, float, float, float] | None
    kps: tuple
    score: float
    first_seen_ns: int
    last_seen_ns: int
    last_frame_id: int
    confirmed: bool = False

    def predicted(self, now_ns: int) -> tuple[float, float, float, float]:
        if self.prev_box is None:
            return self.box
        dt = max(now_ns - self.last_seen_ns, 0) / 1e9
        c0 = _center(self.prev_box)
        c1 = _center(self.box)
        vx = (c1[0] - c0[0]) * 8.0
        vy = (c1[1] - c0[1]) * 8.0
        ncx = c1[0] + vx * dt
        ncy = c1[1] + vy * dt
        w = (self.box[2] - self.box[0])
        h = (self.box[3] - self.box[1])
        return (ncx - w / 2, ncy - h / 2, ncx + w / 2, ncy + h / 2)

    def step_with(self, det: PoseDetectionV1, now_ns: int, frame_id: int) -> None:
        self.prev_box = self.box
        self.box = det.bbox_xyxy
        self.kps = det.keypoints_coco17
        self.score = det.detection_score
        self.last_seen_ns = now_ns
        self.last_frame_id = frame_id


def _match(tracks, dets, cfg, now_ns):
    """将待匹配轨迹与检测做 Hungarian 分配，返回 (matched_idx, track_to_det)."""
    if not tracks or not dets:
        return {}
    cost = np.zeros((len(tracks), len(dets)))
    for i, tr in enumerate(tracks):
        pb = tr.predicted(now_ns)
        cc = _center(pb)
        for j, d in enumerate(dets):
            dc = _center(d.bbox_xyxy)
            c = (
                (1.0 - _iou(pb, d.bbox_xyxy)) * _IOU_WEIGHT
                + _kpd(pb, d) * _KPDIST_WEIGHT
                + math.hypot(cc[0] - dc[0], cc[1] - dc[1]) / 1920.0 * _CENTER_WEIGHT
            )
            cost[i, j] = c
    ri, ci = linear_sum_assignment(cost)
    out = {}
    max_cost = 1.0 + cfg.max_normalized_keypoint_distance * _KPDIST_WEIGHT + _CENTER_WEIGHT
    for i, j in zip(ri, ci):
        if cost[i, j] <= max_cost * 1.5:
            out[i] = j
    return out


def _kpd(pred_box, det: PoseDetectionV1) -> float:
    pp = _center(pred_box)
    dd = _center(det.bbox_xyxy)
    return math.hypot(pp[0] - dd[0], pp[1] - dd[1]) / 1920.0


class PoseTracker:
    def __init__(self, cfg: TrackerConfig) -> None:
        self._cfg = cfg
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._last_frame_id: int | None = None
        self._last_seen_ns: int = 0

    def update(self, detections, *, now_mono_ns: int, frame_id: int) -> list[PoseTrackV1]:
        # 乱序/重复帧不推进
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            return self._confirmed_tracks()
        self._last_frame_id = frame_id
        self._last_seen_ns = now_mono_ns

        high = [d for d in detections if d.detection_score >= self._cfg.high_confidence]
        low = [d for d in detections if self._cfg.low_confidence <= d.detection_score < self._cfg.high_confidence]

        active = list(self._tracks.values())
        matched = set()

        # 第一轮：已确认轨迹 + 高阶检测
        first = [t for t in active if t.confirmed]
        assign = _match(first, high, self._cfg, now_mono_ns)
        used_high = set()
        for i, j in assign.items():
            tr = first[i]
            tr.step_with(high[j], now_mono_ns, frame_id)
            matched.add(tr.track_id)
            used_high.add(j)

        # 第二轮：未确认轨迹 + 剩余高阶
        tent = [t for t in active if not t.confirmed]
        assign2 = _match(tent, [d for i, d in enumerate(high) if i not in used_high],
                         self._cfg, now_mono_ns)
        # 简化：仅按 index 映射无需精确；此处基于位置重建检测子集
        remaining_high = [d for i, d in enumerate(high) if i not in used_high]
        for i, j in assign2.items():
            tr = tent[i]
            tr.step_with(remaining_high[j], now_mono_ns, frame_id)
            matched.add(tr.track_id)

        # 第三轮：用低阶挽救未匹配的已确认轨迹
        un_confirmed = [t for t in first if t.track_id not in matched]
        remaining_unused_high_per = True
        assign3 = _match(un_confirmed, low, self._cfg, now_mono_ns)
        for i, j in assign3.items():
            tr = un_confirmed[i]
            tr.step_with(low[j], now_mono_ns, frame_id)
            matched.add(tr.track_id)

        # 已匹配轨迹更新确认
        for tid in matched:
            tr = self._tracks[tid]
            dt = (now_mono_ns - tr.first_seen_ns) / 1e9
            if dt >= self._cfg.min_confirm_duration_s or tr.confirmed:
                tr.confirmed = True

        # 未匹配的高阶 GOOD 检测 -> 新建 tentative 轨迹（PARTIAL/INVALID 不新建）
        unmatched_high = [d for i, d in enumerate(high) if i not in used_high]
        for d in unmatched_high:
            if _quality(d) in ("GOOD", "PARTIAL"):
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _Track(
                    track_id=tid, box=d.bbox_xyxy, prev_box=None, kps=d.keypoints_coco17,
                    score=d.detection_score, first_seen_ns=now_mono_ns,
                    last_seen_ns=now_mono_ns, last_frame_id=frame_id,
                    confirmed=(_quality(d) == "GOOD"),
                )

        # 过期轨迹移除（丢失）
        for tid in list(self._tracks):
            if (now_mono_ns - self._tracks[tid].last_seen_ns) / 1e9 > self._cfg.lost_timeout_s:
                del self._tracks[tid]

        return self._confirmed_tracks()

    def _confirmed_tracks(self) -> list[PoseTrackV1]:
        out = []
        for tid, tr in self._tracks.items():
            if tr.confirmed:
                out.append(PoseTrackV1(
                    pose_track_id=tid, bbox_xyxy=tr.box, detection_score=tr.score,
                    keypoints_coco17=tr.kps, pose_quality="GOOD",
                    state=PoseStateV1.NORMAL, state_since_monotonic_ns=tr.last_seen_ns,
                    rule_score=0.0,
                    score_semantics="heuristic_rule_score_not_probability",
                    evidence_codes=(), is_ghost=False,
                ))
        out.sort(key=lambda t: t.pose_track_id)
        return out
