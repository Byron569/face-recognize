"""vision 内核单元测试(IoU 跟踪器 + 数据模型 + 任务接口 + 配置)。"""

from vision.config import TrackConfig, VisionConfig
from vision.events import FaceResult, PipelineContext, TrackResult, VisionEvent
from vision.tasks import VisionTask
from vision.tracker import IoUTracker, _iou


def _face(x1, y1, x2, y2, score=0.9):
    return FaceResult(bbox=(x1, y1, x2, y2), det_score=score)


# ── IoU ──────────────────────────────────────────────────

def test_iou_full_overlap():
    assert _iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0


def test_iou_no_overlap():
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half():
    assert abs(_iou((0, 0, 100, 100), (50, 0, 150, 100)) - 1 / 3) < 1e-6


# ── 跟踪器 ───────────────────────────────────────────────

def test_tracker_creates_stable_id():
    tracker = IoUTracker(TrackConfig(min_hits=2, max_lost=15))
    snap1 = tracker.update([_face(10, 10, 110, 110)], 1)
    snap2 = tracker.update([_face(12, 12, 112, 112)], 2)
    assert len(snap1) == len(snap2) == 1
    assert snap1[0].track_id == snap2[0].track_id
    assert snap2[0].confirmed
    assert snap2[0].hits == 2


def test_tracker_no_detection_increments_lost():
    tracker = IoUTracker(TrackConfig(max_lost=3))
    tracker.update([_face(10, 10, 100, 100)], 1)
    for i in range(2, 6):  # 4 帧无检测 → 超过 max_lost 删除
        tracker.update([], i)
    assert tracker.active_count == 0


def test_tracker_separates_two_faces():
    tracker = IoUTracker(TrackConfig(min_hits=1))
    snap = tracker.update([_face(0, 0, 50, 50), _face(200, 200, 260, 260)], 1)
    assert len(snap) == 2
    assert snap[0].track_id != snap[1].track_id


def test_tracker_identity_write_back():
    tracker = IoUTracker(TrackConfig(min_hits=1))
    snap = tracker.update([_face(0, 0, 50, 50)], 1)
    assert tracker.set_identity(snap[0].track_id, "Byron", 0.87)
    snap2 = tracker.update([], 2)
    assert snap2[0].identity == "Byron"
    assert abs(snap2[0].similarity - 0.87) < 1e-6


def test_tracker_respects_max_tracks():
    tracker = IoUTracker(TrackConfig(max_tracks=2, min_hits=1))
    faces = [_face(i * 100, 0, i * 100 + 50, 50) for i in range(5)]
    snap = tracker.update(faces, 1)
    assert len(snap) == 2


# ── 数据模型 ─────────────────────────────────────────────

def test_track_result_to_dict_bbox_is_xywh():
    tr = TrackResult(track_id=1, bbox=(10, 20, 110, 120), identity="Byron", similarity=0.9)
    d = tr.to_dict()
    assert d["bbox"] == [10, 20, 100, 100]
    assert d["identity"] == "Byron"
    assert d["confidence"] == 0.9


def test_vision_event_to_dict():
    ev = VisionEvent(event_type="fall_detected", camera_id="cam0", track_id=3, confidence=0.8)
    d = ev.to_dict()
    assert d["event_type"] == "fall_detected"
    assert d["camera_id"] == "cam0"
    assert d["track_id"] == 3


# ── 任务接口 ─────────────────────────────────────────────

class _DummyTask(VisionTask):
    name = "dummy"

    def should_run(self, frame_id, context):
        return frame_id % self.interval == 0

    def run(self, frame, context):
        return [VisionEvent(event_type="dummy_event", track_id=1)]


def test_task_interval_and_enabled():
    task = _DummyTask({"enabled": True, "interval": 3})
    ctx = PipelineContext(camera_id="c0", frame_id=0, frame=None)
    assert task.enabled and task.interval == 3
    assert task.should_run(3, ctx)
    assert not task.should_run(4, ctx)
    events = task.run(None, ctx)
    assert events[0].event_type == "dummy_event"


# ── 配置 ────────────────────────────────────────────────

def test_vision_config_defaults_gpu():
    cfg = VisionConfig.from_dict({})
    assert cfg.device == "cuda"
    assert cfg.det_size == (640, 640)
    assert cfg.track.iou_threshold == 0.3
    assert cfg.recognition.threshold == 0.40


def test_vision_config_from_yaml_shape():
    cfg = VisionConfig.from_dict(
        {
            "device": "cpu",
            "det_size": [320, 320],
            "track": {"max_tracks": 10},
            "recognition": {"threshold": 0.55},
        }
    )
    assert cfg.device == "cpu"
    assert cfg.det_size == (320, 320)
    assert cfg.track.max_tracks == 10
    assert cfg.recognition.threshold == 0.55
