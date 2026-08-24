"""纯姿态特征函数（第 6.7 节）。

只依赖标准库，不导入 OpenCV / Torch / Ultralytics。所有角度/距离/速度均采用明确的
关节语义，速度类特征用真实 dt 并在时间内归一化，尺寸类特征按人体尺度归一化。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..config import AlgorithmConfig

SCORE_SEMANTICS = "heuristic_rule_score_not_probability"

# COCO-17
NOSE, LSH, RSH, LHIP, RHIP, LKNEE, RKNEE = 0, 5, 6, 11, 12, 13, 14


def _vec(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (b[0] - a[0], b[1] - a[1])


def _angle_between(u: tuple[float, float], v: tuple[float, float]) -> float:
    lu = math.hypot(*u)
    lv = math.hypot(*v)
    if lu <= 0.0 or lv <= 0.0:
        return float("nan")
    cosv = (u[0] * v[0] + u[1] * v[1]) / (lu * lv)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def _mid(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def visible(kp: tuple[float, float, float], min_conf: float) -> bool:
    x, y, s = kp
    return (s >= min_conf) and math.isfinite(x) and math.isfinite(y)


def torso_inclination_deg(shoulder_mid, hip_mid) -> float:
    """躯干相对图像竖直方向的倾角，范围 0..90。"""
    v = _vec(hip_mid, shoulder_mid)
    if math.hypot(*v) <= 1e-9:
        return float("nan")
    cosv = (v[0] * 0 + v[1] * -1) / math.hypot(*v)
    cosv = max(-1.0, min(1.0, cosv))
    deg = math.degrees(math.acos(cosv))
    return min(deg, 180.0 - deg)


def hip_angle_deg(shoulder, hip, knee) -> float:
    """髋角：顶点必须为 hip。返回 shoulder-hip-knee 夹角 0..180。"""
    u = _vec(hip, shoulder)
    v = _vec(hip, knee)
    return _angle_between(u, v)


def height_width_ratio(bbox_xyxy) -> float | None:
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    if w <= 0:
        return None
    return h / w


def head_descent_body_heights(cur_head_y: float, standing_head_y: float, body_height_px: float) -> float:
    if body_height_px <= 0:
        return 0.0
    return (cur_head_y - standing_head_y) / body_height_px


def rotation_energy_rad_s(prev_deg: float, cur_deg: float, dt: float) -> float:
    if dt <= 0 or math.isnan(prev_deg) or math.isnan(cur_deg):
        return 0.0
    diff = abs(cur_deg - prev_deg) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return math.radians(diff) / dt


def gravity_factor_body_heights_s2(
    cur_vertical_velocity_px_s: float,
    prev_vertical_velocity_px_s: float,
    dt: float,
    body_height_px: float,
) -> float:
    if dt <= 0 or body_height_px <= 0:
        return 0.0
    return abs(cur_vertical_velocity_px_s - prev_vertical_velocity_px_s) / dt / body_height_px


@dataclass(frozen=True)
class FallEvidence:
    pose_quality: str
    has_reliable_standing: bool
    body_height_px: float
    horizontal_geometry: bool
    torso_horizontal: bool
    extended_hip: bool
    dynamic_descent: bool
    dynamic_rotation: bool
    dynamic_gravity: bool
    fast_dynamic: bool
    rule_score: float
    score_semantics: str = SCORE_SEMANTICS
    evidence_codes: tuple[str, ...] = ()


def compute_evidence(
    *,
    keypoints,
    bbox,
    prev_keypoints=None,
    prev_torso_angle_deg: float | None = None,
    prev_vertical_velocity_px_s: float = 0.0,
    dt: float = 0.0,
    frame_height: float,
    standing_head_y: float | None,
    stable_body_height: float,
    cfg: AlgorithmConfig,
    pose_quality: str = "GOOD",
) -> FallEvidence:
    min_conf = cfg.keypoint_min_confidence
    body_height = stable_body_height if stable_body_height > 0 else (bbox[3] - bbox[1])
    head_y = keypoints[NOSE][1] if visible(keypoints[NOSE], min_conf) else (bbox[1] + bbox[3]) / 2
    head_descent = head_descent_body_heights(
        head_y, standing_head_y if standing_head_y else bbox[1], body_height
    )

    hw = height_width_ratio(bbox)
    horizontal_geometry = hw is not None and hw <= cfg.bbox_height_width_fall_max

    sm = _mid(keypoints[LSH], keypoints[RSH])
    hm = _mid(keypoints[LHIP], keypoints[RHIP])
    incl = torso_inclination_deg(sm, hm)
    torso_horizontal = (not math.isnan(incl)) and incl >= cfg.torso_inclination_from_vertical_min_deg

    hip_vals = []
    for sh, hip, knee in (
        (keypoints[LSH], keypoints[LHIP], keypoints[LKNEE]),
        (keypoints[RSH], keypoints[RHIP], keypoints[RKNEE]),
    ):
        if all(visible(k, min_conf) for k in (sh, hip, knee)):
            hip_vals.append(hip_angle_deg((sh[0], sh[1]), (hip[0], hip[1]), (knee[0], knee[1])))
    extended_hip = any(hv >= cfg.hip_angle_fall_min_deg for hv in hip_vals)

    prev_head_y = None
    if prev_keypoints is not None and visible(prev_keypoints[NOSE], min_conf):
        prev_head_y = prev_keypoints[NOSE][1]
    cur_vy = 0.0
    if prev_head_y is not None and dt > 0:
        cur_vy = (head_y - prev_head_y) / dt

    re = rotation_energy_rad_s(
        prev_torso_angle_deg if prev_torso_angle_deg is not None else 0.0,
        incl if not math.isnan(incl) else 0.0,
        dt,
    )
    gf = gravity_factor_body_heights_s2(cur_vy, prev_vertical_velocity_px_s, dt, body_height)

    dynamic_descent = head_descent >= cfg.head_descent_body_heights_min
    dynamic_rotation = re >= cfg.rotation_energy_min_rad_s
    dynamic_gravity = gf >= cfg.gravity_factor_min_body_heights_s2
    fast_dynamic = (
        re >= cfg.fast_rotation_energy_min_rad_s
        and gf >= cfg.fast_gravity_factor_min_body_heights_s2
    )

    has_reliable_standing = (
        stable_body_height >= frame_height * cfg.minimum_body_height_ratio
    )

    codes: list[str] = []
    if horizontal_geometry:
        codes.append("horizontal_geometry")
    if torso_horizontal:
        codes.append("torso_horizontal")
    if extended_hip:
        codes.append("extended_hip")
    if dynamic_descent:
        codes.append("dynamic_descent")
    if dynamic_rotation:
        codes.append("dynamic_rotation")
    if dynamic_gravity:
        codes.append("dynamic_gravity")

    hits = sum((horizontal_geometry, torso_horizontal, extended_hip, dynamic_descent, dynamic_gravity))
    rule_score = min(1.0, hits / 6.0)

    return FallEvidence(
        pose_quality=pose_quality,
        has_reliable_standing=has_reliable_standing,
        body_height_px=body_height,
        horizontal_geometry=horizontal_geometry,
        torso_horizontal=torso_horizontal,
        extended_hip=extended_hip,
        dynamic_descent=dynamic_descent,
        dynamic_rotation=dynamic_rotation,
        dynamic_gravity=dynamic_gravity,
        fast_dynamic=fast_dynamic,
        rule_score=rule_score,
        evidence_codes=tuple(codes),
    )
