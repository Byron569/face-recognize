"""阶段10（M2）— 跌倒姿态叠加的预览坐标变换与 analytics wire 协议。

只测纯函数:把 Task 写入 ``context.analytics["fall_detection"]`` 的 source-pixels
结果,按后端实际 preview 整数尺寸投影为 preview-pixels 并组装向后兼容的
``type: analytics`` 消息。不启动 Worker / 不连 DB / 不 import Torch。
"""
from __future__ import annotations

import math

import pytest


# 用一个最小合法 source 区块(近似 Task 写入 context.analytics 的结构)
def _source_fall(**overrides):
    d = {
        "schema_version": 1,
        "camera_session_id": "sess-001",
        "attached_to_frame_id": 100,
        "source_frame_id": 98,
        "source_width": 1920,
        "source_height": 1080,
        "coordinate_space": "source_pixels",
        "result_age_ms": 82.4,
        "overlay_expires_in_ms": 1118,
        "worker_end_to_end_ms": 61.7,
        "health": "READY",
        "tracks": [
            {
                "pose_track_id": 3,
                "state": "fallen",
                "score": 0.9,
                "bbox": [100.0, 200.0, 200.0, 500.0],
                "keypoints": [[100.0, 200.0], [110.0, 210.0]],
            }
        ],
    }
    d.update(overrides)
    return d


def _project(fall, preview_w, preview_h):
    """生产实现入口(pipeline_manager 模块级函数),GREEN 时由源码提供。"""
    from backend.app.services.pipeline_manager import _project_analytics_to_preview

    return _project_analytics_to_preview(fall, preview_w, preview_h)


# ── 基本投影与坐标空间 ────────────────────────────────────

def test_projects_to_preview_pixels_and_writes_transform():
    out = _project(_source_fall(), 853, 480)
    assert out["type"] == "analytics"
    assert out["schema_version"] == 1
    assert out["camera_session_id"] == "sess-001"
    assert out["preview_frame_id"] == 100
    fd = out["fall_detection"]
    assert fd["coordinate_space"] == "preview_pixels"
    assert fd["preview_width"] == 853
    assert fd["preview_height"] == 480
    # scale_x 用整数舍入后的实际尺寸之比(1920->853 非相等比例)
    assert math.isclose(fd["transform"]["scale_x"], 853 / 1920, rel_tol=1e-6)
    assert math.isclose(fd["transform"]["scale_y"], 480 / 1080, rel_tol=1e-6)
    assert fd["transform"]["kind"] == "scale_no_letterbox"
    assert fd["transform"]["offset_x"] == 0.0
    assert fd["transform"]["offset_y"] == 0.0


def test_no_resize_yields_identity_transform():
    out = _project(_source_fall(source_width=640, source_height=480), 640, 480)
    fd = out["fall_detection"]
    assert math.isclose(fd["transform"]["scale_x"], 1.0)
    assert math.isclose(fd["transform"]["scale_y"], 1.0)


def test_source_and_preview_frame_ids_stay_distinct():
    fall = _source_fall(source_frame_id=98, attached_to_frame_id=100)
    out = _project(fall, 853, 480)
    fd = out["fall_detection"]
    assert fd["source_frame_id"] == 98
    assert out["preview_frame_id"] == 100
    assert fd["source_frame_id"] != out["preview_frame_id"]


def test_track_bbox_and_keypoints_are_scaled_to_preview():
    out = _project(_source_fall(), 853, 480)
    fd = out["fall_detection"]
    t = fd["tracks"][0]
    sx = 853 / 1920
    sy = 480 / 1080
    # bbox [x, y, w, h]
    assert t["bbox"] == [pytest.approx(100 * sx), pytest.approx(200 * sy),
                          pytest.approx(200 * sx), pytest.approx(500 * sy)]
    assert t["keypoints"][0] == [pytest.approx(100 * sx), pytest.approx(200 * sy)]
    assert t["keypoints"][1] == [pytest.approx(110 * sx), pytest.approx(210 * sy)]
    # score 不变
    assert t["score"] == 0.9
    assert t["state"] == "fallen"


# ── 尺寸 / 校验 ──────────────────────────────────────────

def test_bad_or_nonpositive_preview_size_rejected():
    with pytest.raises(ValueError):
        _project(_source_fall(), 0, 480)
    with pytest.raises(ValueError):
        _project(_source_fall(), 853, -1)
    with pytest.raises(ValueError):
        _project(_source_fall(), 640, 0)


def test_odd_preview_size_uses_exact_integer_dimensions():
    out = _project(_source_fall(source_width=1921, source_height=1081), 641, 361)
    fd = out["fall_detection"]
    assert fd["preview_width"] == 641
    assert fd["preview_height"] == 361
    assert math.isclose(fd["transform"]["scale_x"], 641 / 1921, rel_tol=1e-6)
    assert math.isclose(fd["transform"]["scale_y"], 361 / 1081, rel_tol=1e-6)


def test_track_coordinates_clamped_to_preview_bounds():
    t = [{
        "pose_track_id": 1, "state": "fallen", "score": 0.8,
        "bbox": [-10.0, -5.0, 5000.0, 4000.0],
        "keypoints": [[-100.0, 50.0], [9000.0, 7000.0]],
    }]
    out = _project(_source_fall(tracks=t), 853, 480)
    t0 = out["fall_detection"]["tracks"][0]
    # bbox 四角夹到 [0, preview-1]
    x1 = max(0.0, -10.0 * (853 / 1920))
    y1 = max(0.0, -5.0 * (480 / 1080))
    x2 = min(852.0, 5000.0 * (853 / 1920))
    y2 = min(479.0, 4000.0 * (480 / 1080))
    assert t0["bbox"][0] >= 0.0 and t0["bbox"][0] <= x2
    assert t0["bbox"][1] >= 0.0
    # 只断言 keypoint 也在有效范围
    for kp in t0["keypoints"]:
        assert 0.0 <= kp[0] <= 852.0
        assert 0.0 <= kp[1] <= 479.0


def test_nan_or_infinite_scale_rejected():
    with pytest.raises(ValueError):
        _project(_source_fall(source_width=0), 853, 480)
    with pytest.raises(ValueError):
        _project(
            _source_fall(tracks=[{
                "pose_track_id": 1, "state": "normal", "score": 0.5,
                "bbox": [float("nan"), 0.0, 1.0, 1.0], "keypoints": [],
            }]),
            853, 480,
        )


# ── 健康 / 生命周期延续字段 ───────────────────────────────

def test_preserves_health_and_ttl_and_worker_latency():
    out = _project(_source_fall(), 853, 480)
    fd = out["fall_detection"]
    assert fd["health"] == "READY"
    assert fd["result_age_ms"] == 82.4
    assert fd["overlay_expires_in_ms"] == 1118
    assert fd["worker_end_to_end_ms"] == 61.7


def test_inner_and_outer_session_must_match():
    out = _project(_source_fall(), 853, 480)
    assert out["camera_session_id"] == out["fall_detection"]["camera_session_id"]


# ── 无 analytics 时旧链路不受影响(NEGATIVE) ──────────────

def test_missing_fall_section_raises_value_error():
    with pytest.raises(ValueError):
        _project(None, 853, 480)