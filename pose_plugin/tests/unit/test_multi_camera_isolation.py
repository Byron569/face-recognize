"""阶段3：多摄像头状态与调度隔离。"""
from __future__ import annotations

from ai_monitor_pose.config import TrackerConfig
from ai_monitor_pose.contracts import PoseDetectionV1
from ai_monitor_pose.scheduler import FallScheduler
from ai_monitor_pose.worker.pose_tracker import PoseTracker

from tests.fixtures.pose_sequences import bbox_of, standing_keypoints


def _cfg() -> TrackerConfig:
    return TrackerConfig(
        high_confidence=0.35, low_confidence=0.15, match_iou_threshold=0.50,
        max_normalized_keypoint_distance=0.35, min_confirm_duration_s=0.25,
        lost_timeout_s=1.5, ghost_timeout_s=3.0, fallen_ghost_timeout_s=5.0,
    )


def _det():
    return PoseDetectionV1(
        bbox_xyxy=bbox_of(standing_keypoints()), detection_score=0.9,
        keypoints_coco17=tuple(standing_keypoints()),
    )


def test_scheduler_isolates_camera_slots() -> None:
    s = FallScheduler(target_fps=8, batch_size=1)
    s.register_camera("cam-1")
    s.register_camera("cam-2")
    now = int(4e12)
    s.offer("cam-1", 1, 1, now)
    s.offer("cam-2", 1, 1, now)
    # cam-1 的帧不占用/影响 cam-2
    d1 = s.pick(now)
    assert d1 is not None and d1[0] == "cam-1"
    s.complete("cam-1", d1[1], ok=True)
    d2 = s.pick(now)
    assert d2 is not None and d2[0] == "cam-2"
    s.complete("cam-2", d2[1], ok=True)


def test_trackers_fully_isolated_after_update() -> None:
    a = PoseTracker(_cfg())
    b = PoseTracker(_cfg())
    ta = a.update([_det()], now_mono_ns=int(6e8), frame_id=1)
    tb = b.update([_det()], now_mono_ns=int(6e8), frame_id=1)
    assert len(ta) == 1 and len(tb) == 1
    assert ta[0].pose_track_id == tb[0].pose_track_id == 1
    # 各自状态互不影响
    assert ta[0].bbox_xyxy == tb[0].bbox_xyxy
