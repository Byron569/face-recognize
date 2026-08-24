"""阶段2 fixture：构造站立/水平/恢复等身体姿态的关键点、bbox 与带证据的时间样本。"""
from __future__ import annotations

from dataclasses import dataclass

_KP_SCORE = 0.9

# COCO-17 索引
NOSE, LSH, RSH, LHIP, RHIP, LKNEE, RKNEE, LANK, RANK = 0, 5, 6, 11, 12, 13, 14, 15, 16


def _kp(x: float, y: float) -> tuple[float, float, float]:
    return (x, y, _KP_SCORE)


def _zero() -> list[tuple[float, float, float]]:
    return [(0.0, 0.0, 0.0)] * 17


def standing_keypoints(h: float = 480.0, cx: float = 320.0) -> tuple[tuple[float, float, float], ...]:
    pts = _zero()
    top = 0.12 * h
    bottom = 0.92 * h
    span = bottom - top
    pts[NOSE] = _kp(cx + 0.0 * span, top)
    pts[LSH] = _kp(cx - 0.10 * span, top + 0.28 * span)
    pts[RSH] = _kp(cx + 0.10 * span, top + 0.28 * span)
    pts[LHIP] = _kp(cx - 0.05 * span, bottom - 0.30 * span)
    pts[RHIP] = _kp(cx + 0.05 * span, bottom - 0.30 * span)
    pts[LKNEE] = _kp(cx - 0.05 * span, bottom - 0.13 * span)
    pts[RKNEE] = _kp(cx + 0.05 * span, bottom - 0.13 * span)
    pts[LANK] = _kp(cx - 0.02 * span, bottom)
    pts[RANK] = _kp(cx + 0.02 * span, bottom)
    return tuple(pts)


def horizontal_keypoints(h: float = 480.0, cx: float = 320.0) -> tuple[tuple[float, float, float], ...]:
    pts = _zero()
    y = 0.70 * h
    pts[NOSE] = _kp(cx + 0.35 * h, y - 0.03 * h)
    pts[LSH] = _kp(cx + 0.15 * h, y)
    pts[RSH] = _kp(cx + 0.25 * h, y)
    pts[LHIP] = _kp(cx - 0.15 * h, y)
    pts[RHIP] = _kp(cx - 0.05 * h, y)
    pts[LKNEE] = _kp(cx - 0.22 * h, y + 0.05 * h)
    pts[RKNEE] = _kp(cx - 0.12 * h, y + 0.05 * h)
    pts[LANK] = _kp(cx - 0.25 * h, y + 0.12 * h)
    pts[RANK] = _kp(cx - 0.15 * h, y + 0.12 * h)
    return tuple(pts)


def bbox_of(kps: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float, float]:
    vis = [k for k in kps if k[2] > 0.5]
    xs = [k[0] for k in vis]
    ys = [k[1] for k in vis]
    return (min(xs), min(ys), max(xs), max(ys))


@dataclass(frozen=True)
class Observation:
    t_sec: float
    frame_id: int
    keypoints: tuple[tuple[float, float, float], ...]
    bbox: tuple[float, float, float, float]
    fall_evidence: bool
    pose_quality: str = "GOOD"


def observation(t: float, frame_id: int, fall: bool = False, *,
                quality: str = "GOOD") -> Observation:
    kps = horizontal_keypoints() if fall else standing_keypoints()
    return Observation(
        t_sec=t, frame_id=frame_id, keypoints=kps, bbox=bbox_of(kps),
        fall_evidence=fall, pose_quality=quality,
    )
