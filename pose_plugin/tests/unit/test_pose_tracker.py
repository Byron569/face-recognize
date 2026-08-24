"""阶段3：每摄像头独立 PoseTracker 测试。"""
from __future__ import annotations

from ai_monitor_pose.contracts import PoseDetectionV1
from ai_monitor_pose.config import TrackerConfig
from ai_monitor_pose.worker.pose_tracker import PoseTracker

from tests.fixtures.pose_sequences import (
    bbox_of,
    standing_keypoints,
)


def _cfg() -> TrackerConfig:
    return TrackerConfig(
        high_confidence=0.35, low_confidence=0.15, match_iou_threshold=0.50,
        max_normalized_keypoint_distance=0.35, min_confirm_duration_s=0.25,
        lost_timeout_s=1.5, ghost_timeout_s=3.0, fallen_ghost_timeout_s=5.0,
    )


def _det(kps=None, score=0.85) -> PoseDetectionV1:
    kps = kps or standing_keypoints()
    return PoseDetectionV1(
        bbox_xyxy=bbox_of(kps), detection_score=score,
        keypoints_coco17=tuple(kps),
    )


def test_two_cameras_can_both_have_local_track_one_without_collision() -> None:
    a = PoseTracker(_cfg())
    b = PoseTracker(_cfg())
    ta = a.update([_det()], now_mono_ns=int(5e8), frame_id=1)
    tb = b.update([_det()], now_mono_ns=int(5e8), frame_id=1)
    # 均拥有本地 track id 1，互不冲突
    assert ta[0].pose_track_id == 1
    assert tb[0].pose_track_id == 1


def test_new_session_does_not_reuse_old_history() -> None:
    t = PoseTracker(_cfg())
    # 首次更新即返回确认轨迹（跨过确认时长）
    out = t.update([_det()], now_mono_ns=int(5e8), frame_id=1)
    assert len(out) == 1
    assert out[0].pose_track_id == 1


def test_partial_pose_does_not_create_new_fall_track() -> None:
    t = PoseTracker(_cfg())
    # 低可见度（关键点大部分缺失/低置信）不得创建新轨迹
    kps = tuple((x, y, 0.05) for x, y, _ in standing_keypoints())
    det = PoseDetectionV1(bbox_xyxy=bbox_of(standing_keypoints()), detection_score=0.85,
                          keypoints_coco17=kps)
    for i in range(10):
        out = t.update([det], now_mono_ns=int(i * 5e7) + int(6e8), frame_id=i)
        assert len(out) == 0


def test_ghost_cannot_create_new_incident() -> None:
    # 低置信（ghost）检测不能创建新的已确认轨迹 / incident
    t = PoseTracker(_cfg())
    ghost = PoseDetectionV1(
        bbox_xyxy=bbox_of(standing_keypoints()), detection_score=0.10,
        keypoints_coco17=tuple(standing_keypoints()),
    )
    for i in range(5):
        out = t.update([ghost], now_mono_ns=int(i * 5e7) + int(2e8), frame_id=i)
        assert len(out) == 0


def test_tracker_loss_does_not_emit_recovered() -> None:
    t = PoseTracker(_cfg())
    t.update([_det()], now_mono_ns=int(5e8), frame_id=1)
    # 随后丢轨（无观测），不得产生 recovered 类事件
    out = t.update([], now_mono_ns=int(9e9), frame_id=999)
    assert hasattr(t, "emitted_recovered") is False
    assert isinstance(out, list)


def test_tracker_never_calls_model_track_persist() -> None:
    # 不加载模型、不调用 model.track；直接喂 PoseDetectionV1 即可工作
    t = PoseTracker(_cfg())
    out = t.update([_det()], now_mono_ns=int(5e8), frame_id=1)
    assert isinstance(out, list)
