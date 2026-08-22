"""视频/摄像头注册纯函数测试。"""

from __future__ import annotations

import numpy as np

from vision.events import FaceResult
from backend.app.services.video_registration import (
    CandidateFrame,
    RejectedFrame,
    analyze_face_result,
    blur_variance,
    classify_pose,
    compute_pose_ratios,
    select_diverse_candidates,
)

# 5 点关键点顺序: [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角](insightface 输出)
# 用像素量级(生产实际),inter_eye 数十像素,不会被 clamp
KPS_FRONTAL = [(200.0, 180.0), (320.0, 180.0), (260.0, 260.0), (220.0, 340.0), (300.0, 340.0)]
KPS_PROFILE = [(160.0, 180.0), (340.0, 180.0), (300.0, 260.0), (240.0, 340.0), (360.0, 340.0)]


def _cfg(**overrides):
    cfg = {
        "min_det_score": 0.5,
        "min_face_size": 48,
        "min_blur_variance": 55.0,
        "max_yaw_ratio": 0.20,
        "max_pitch_ratio": 0.28,
        "duplicate_similarity": 0.94,
    }
    cfg.update(overrides)
    return cfg


def _image_sharp():
    """锐利图像:随机噪声+强边缘,Laplacian 方差大。"""
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    return img


_MISSING = object()


def _face(bbox=(120, 120, 240, 240), det_score=0.95, kps=_MISSING, embedding=_MISSING):
    if embedding is _MISSING:
        rng = np.random.default_rng(123)
        embedding = rng.standard_normal(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
    if kps is _MISSING:
        kps = KPS_FRONTAL
    return FaceResult(bbox=bbox, det_score=det_score, kps=kps, embedding=embedding)


def test_compute_pose_ratios_symmetric_frontal_face():
    yaw, pitch = compute_pose_ratios(KPS_FRONTAL)
    assert abs(yaw) < 1e-6
    assert abs(pitch) < 1e-6


def test_compute_pose_ratios_profile_has_large_yaw():
    yaw, _ = compute_pose_ratios(KPS_PROFILE)
    assert yaw > 0.20


def test_classify_pose_all_five_regions():
    # 左右约定:yaw>0 = 用户向左转头(left)
    assert classify_pose(0.0, 0.0, 0.2, 0.28) == "frontal"
    assert classify_pose(0.3, 0.0, 0.2, 0.28) == "left"
    assert classify_pose(-0.3, 0.0, 0.2, 0.28) == "right"
    assert classify_pose(0.0, -0.3, 0.2, 0.28) == "up"
    assert classify_pose(0.0, 0.3, 0.2, 0.28) == "down"


def test_compute_pose_ratios_left_turn_is_positive():
    # 用户向左转头 → 画面中鼻尖右移(nose_x > eye_mid_x)→ yaw > 0
    yaw, _ = compute_pose_ratios(KPS_PROFILE)
    assert yaw > 0.20


def test_compute_pose_ratios_right_turn_is_negative():
    # 对称镜像:鼻尖在双眼中点左侧 → 用户向右转头 → yaw < 0(防后人把符号改回去)
    mirrored = [(640.0 - x, y) for (x, y) in KPS_PROFILE]
    yaw, _ = compute_pose_ratios(mirrored)
    assert yaw < -0.20


def test_analyze_rejects_no_face():
    r = analyze_face_result([], _image_sharp(), "f1", 0, "frontal", _cfg())
    assert isinstance(r, RejectedFrame) and r.reason == "no_face"


def test_analyze_rejects_multiple_faces():
    r = analyze_face_result([_face(), _face(bbox=(300, 120, 400, 240))], _image_sharp(), "f1", 0, "frontal", _cfg())
    assert isinstance(r, RejectedFrame) and r.reason == "multiple_faces"


def test_analyze_rejects_low_detection_score():
    r = analyze_face_result([_face(det_score=0.3)], _image_sharp(), "f1", 0, "frontal", _cfg())
    assert isinstance(r, RejectedFrame) and r.reason == "low_detection_score"


def test_analyze_rejects_face_too_small():
    small = _face(bbox=(200, 200, 230, 230))  # 30x30 < 48
    r = analyze_face_result([small], _image_sharp(), "f1", 0, "frontal", _cfg())
    assert isinstance(r, RejectedFrame) and r.reason == "face_too_small"


def test_analyze_rejects_missing_landmarks():
    r = analyze_face_result([_face(kps=None)], _image_sharp(), "f1", 0, "frontal", _cfg())
    assert isinstance(r, RejectedFrame) and r.reason == "missing_landmarks"


def test_analyze_rejects_missing_embedding():
    r = analyze_face_result([_face(embedding=None)], _image_sharp(), "f1", 0, "frontal", _cfg())
    assert isinstance(r, RejectedFrame) and r.reason == "missing_embedding"


def test_analyze_passes_frontal_frame():
    r = analyze_face_result([_face()], _image_sharp(), "f1", 0, "frontal", _cfg())
    assert isinstance(r, CandidateFrame), r
    assert r.pose == "frontal"
    assert r.quality_score > 0


def test_select_diverse_per_pose_buckets_and_dedup():
    rng = np.random.default_rng(5)
    def emb():
        e = rng.standard_normal(512).astype(np.float32)
        return (e / np.linalg.norm(e)).tolist()
    base = emb()
    dup = [v * 1.0 for v in base]  # 完全同向(去重命中)
    left3 = [
        CandidateFrame(f"l{i}", i * 100, "left", (0, 0, 120, 120), 0.9, 0.3, 0.0, 120.0, 0.8 - i * 0.05, base if i == 0 else dup if i == 1 else emb())
        for i in range(3)
    ]
    frontal = [CandidateFrame("fa", 0, "frontal", (0, 0, 120, 120), 0.9, 0.0, 0.0, 130.0, 0.9, emb())]
    selected = select_diverse_candidates(frontal + left3, target_per_pose=2, duplicate_similarity=0.94)
    # frontal 桶优先
    assert selected[0].frame_id == "fa"
    left_ids = [c.frame_id for c in selected if c.pose == "left"]
    assert "l0" in left_ids
    assert "l1" not in left_ids  # 与 l0 去重
    assert len(left_ids) <= 2


def test_select_diverse_never_exceeds_target_per_pose():
    rng = np.random.default_rng(3)
    def emb():
        e = rng.standard_normal(512).astype(np.float32)
        return (e / np.linalg.norm(e)).tolist()
    frames = [CandidateFrame(f"f{i}", i, "right", (0, 0, 100, 100), 0.9, 0.3, 0.0, 110.0, 0.7, emb()) for i in range(6)]
    selected = select_diverse_candidates(frames, target_per_pose=2, duplicate_similarity=0.94)
    assert len(selected) <= 2


def test_blur_variance_of_noise_is_high():
    bv = blur_variance(_image_sharp(), (100, 100, 300, 300))
    assert bv > 55.0


def test_blur_variance_zero_for_out_of_bounds():
    bv = blur_variance(_image_sharp(), (900, 900, 950, 950))
    assert bv == 0.0
