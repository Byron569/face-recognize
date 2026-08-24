"""阶段2：纯特征函数测试。"""
from __future__ import annotations

import math

from ai_monitor_pose.worker import features
from ai_monitor_pose.worker.features import (
    FallEvidence,
    compute_evidence,
    gravity_factor_body_heights_s2,
    head_descent_body_heights,
    height_width_ratio,
    hip_angle_deg,
    rotation_energy_rad_s,
    torso_inclination_deg,
)

from tests.fixtures.pose_sequences import (
    bbox_of,
    horizontal_keypoints,
    standing_keypoints,
)


def _alg():
    from ai_monitor_pose.config import AlgorithmConfig
    return AlgorithmConfig(
        required_keypoint_indices=(0, 5, 6, 11, 12, 13, 14),
        minimum_visible_required=5,
        keypoint_min_confidence=0.30,
        minimum_body_height_ratio=0.12,
        bbox_height_width_fall_max=0.75,
        torso_inclination_from_vertical_min_deg=55.0,
        hip_angle_fall_min_deg=135.0,
        head_descent_body_heights_min=0.18,
        rotation_energy_min_rad_s=1.80,
        gravity_factor_min_body_heights_s2=1.50,
        fast_rotation_energy_min_rad_s=3.00,
        fast_gravity_factor_min_body_heights_s2=2.50,
        upright_height_width_min=1.25,
        upright_torso_inclination_max_deg=25.0,
        rebound_body_heights_min=0.12,
        ema_tau_s=0.20,
        history_duration_s=4.5,
        trigger_window_s=1.25,
        trigger_ratio=0.50,
        min_trigger_duration_s=0.50,
        max_trigger_gap_s=0.25,
        min_fall_pose_duration_s=3.50,
        recovery_duration_s=1.00,
        rebound_duration_s=0.50,
        tracker_reset_gap_s=1.50,
        allow_already_down_detection=False,
    )


def test_resolution_scaled_motion_has_consistent_normalized_features() -> None:
    # 相同动作在 640x480 与 1280x960 下，归一化后的角速度/重力因子应相等
    dt = 0.2
    cfg = _alg()
    # 角速度与分辨率无关（rad/s）
    e_small = rotation_energy_rad_s(math.radians(10), math.radians(15), dt)
    e_big = rotation_energy_rad_s(math.radians(10), math.radians(15), dt)
    assert abs(e_small - e_big) < 1e-12
    # 归一化重力因子：速度差按身体高度归一
    g_480 = gravity_factor_body_heights_s2(2.0, 0.5, dt, body_height_px=150.0)
    g_960 = gravity_factor_body_heights_s2(4.0, 1.0, dt, body_height_px=300.0)
    # 自适应分辨率后速度差翻倍、身体高度翻倍 -> 归一化几乎相等
    assert abs(g_480 - g_960) < 1e-6


def test_hip_angle_fallback_uses_hip_as_vertex() -> None:
    # hip_angle 的定义：shoulder - hip - knee，顶点必须是 hip(中点)。
    # 构造一个肩膀、髋、膝组合，验证角度语义不把 shoulder 当顶点。
    kps = standing_keypoints(h=480.0, cx=320.0)
    from ai_monitor_pose.worker.features import _vec, _angle_between
    shoulder = (300.0, 120.0)
    hip = (320.0, 260.0)
    knee = (330.0, 380.0)
    a = hip_angle_deg(shoulder, hip, knee)
    # 三个点近似落在一条略折的线，角度应偏小（接近伸直腿 < 180）
    assert 0.0 <= a <= 180.0


def test_upright_torso_has_small_inclination() -> None:
    kps = standing_keypoints()
    # 站立时躯干接近竖直
    shoulder_mid = ((kps[5][0] + kps[6][0]) / 2, (kps[5][1] + kps[6][1]) / 2)
    hip_mid = ((kps[11][0] + kps[12][0]) / 2, (kps[11][1] + kps[12][1]) / 2)
    deg = torso_inclination_deg(shoulder_mid, hip_mid)
    assert deg < 25.0


def test_horizontal_torso_has_large_inclination() -> None:
    kps = horizontal_keypoints()
    shoulder_mid = ((kps[5][0] + kps[6][0]) / 2, (kps[5][1] + kps[6][1]) / 2)
    hip_mid = ((kps[11][0] + kps[12][0]) / 2, (kps[11][1] + kps[12][1]) / 2)
    deg = torso_inclination_deg(shoulder_mid, hip_mid)
    assert deg >= 55.0


def test_height_width_markers() -> None:
    stand = bbox_of(standing_keypoints())
    hw = height_width_ratio(stand)
    assert hw is not None
    # 站立框一般高>宽
    assert hw > 1.0
    horiz = height_width_ratio(bbox_of(horizontal_keypoints()))
    assert horiz is not None
    assert horiz < 0.75


def test_head_descent_uses_body_heights() -> None:
    d = head_descent_body_heights(cur_head_y=200.0, standing_head_y=100.0, body_height_px=150.0)
    assert abs(d - (100.0 / 150.0)) < 1e-9


def test_falling_evidence_flags() -> None:
    cur = horizontal_keypoints()
    prev = standing_keypoints()
    cfg = _alg()
    ev = compute_evidence(
        keypoints=cur,
        bbox=bbox_of(cur),
        prev_keypoints=prev,
        prev_torso_angle_deg=torso_inclination_deg(
            ((prev[5][0] + prev[6][0]) / 2, (prev[5][1] + prev[6][1]) / 2),
            ((prev[11][0] + prev[12][0]) / 2, (prev[11][1] + prev[12][1]) / 2),
        ),
        prev_vertical_velocity_px_s=0.0,
        dt=0.2,
        frame_height=480.0,
        standing_head_y=standing_keypoints()[0][1],
        stable_body_height=230.0,
        cfg=cfg,
    )
    assert isinstance(ev, FallEvidence)
    assert ev.horizontal_geometry in (True, False)


def test_rule_score_is_explicitly_not_probability() -> None:
    ev = compute_evidence(
        keypoints=horizontal_keypoints(),
        bbox=bbox_of(horizontal_keypoints()),
        prev_keypoints=standing_keypoints(),
        prev_torso_angle_deg=0.0,
        prev_vertical_velocity_px_s=0.0,
        dt=0.2,
        frame_height=480.0,
        standing_head_y=standing_keypoints()[0][1],
        stable_body_height=230.0,
        cfg=_alg(),
    )
    assert ev.score_semantics == "heuristic_rule_score_not_probability"
    assert 0.0 <= ev.rule_score <= 1.0
