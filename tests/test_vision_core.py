"""vision 内核单元测试(ByteTrack 跟踪器 + 数据模型 + 任务接口 + 配置)。"""

from vision.config import TrackConfig, VisionConfig
from vision.events import FaceResult, PipelineContext, TrackResult, VisionEvent
from vision.tasks import VisionTask
from vision.tracker import ByteTracker, _TrackState


def _face(x1, y1, x2, y2, score=0.9):
    return FaceResult(bbox=(x1, y1, x2, y2), det_score=score)


# ── ByteTrack 跟踪器 ──────────────────────────────────────

def test_tracker_creates_stable_id():
    tracker = ByteTracker(TrackConfig(min_hits=2, track_buffer=30, frame_rate=30))
    snap1 = tracker.update([_face(10, 10, 110, 110)], 1)
    snap2 = tracker.update([_face(12, 12, 112, 112)], 2)
    assert len(snap1) == len(snap2) == 1
    assert snap1[0].track_id == snap2[0].track_id
    assert snap2[0].confirmed
    assert snap2[0].hits == 2


def test_tracker_removes_lost_track_after_buffer():
    tracker = ByteTracker(TrackConfig(min_hits=1, track_buffer=3, frame_rate=30))
    tracker.update([_face(10, 10, 100, 100)], 1)
    for i in range(2, 8):  # 超过 buffer 后删除
        tracker.update([], i)
    assert tracker.active_count == 0


def test_tracker_separates_two_faces():
    tracker = ByteTracker(TrackConfig(min_hits=1))
    snap = tracker.update([_face(0, 0, 50, 50), _face(200, 200, 260, 260)], 1)
    assert len(snap) == 2
    assert snap[0].track_id != snap[1].track_id


def test_tracker_identity_write_back():
    tracker = ByteTracker(TrackConfig(min_hits=1))
    snap = tracker.update([_face(0, 0, 50, 50)], 1)
    assert tracker.set_identity(snap[0].track_id, "Byron", 0.87)
    snap2 = tracker.update([], 2)
    assert snap2[0].identity == "Byron"
    assert abs(snap2[0].similarity - 0.87) < 1e-6


def test_tracker_respects_max_tracks():
    tracker = ByteTracker(TrackConfig(max_tracks=2, min_hits=1))
    faces = [_face(i * 100, 0, i * 100 + 50, 50) for i in range(5)]
    snap = tracker.update(faces, 1)
    assert len(snap) == 2


def test_tracker_second_association_for_low_score():
    """ByteTrack 核心:低置信度检测通过二次关联保持既有 ID(而非新建)。"""
    tracker = ByteTracker(TrackConfig(min_hits=1))
    tracker.update([_face(0, 0, 50, 50, score=0.9)], 1)
    snap = tracker.update([_face(1, 1, 51, 51, score=0.3)], 2)
    assert len(snap) == 1
    assert snap[0].track_id == 1


def test_tracker_rescued_track_not_marked_lost():
    """修复:低分框救回的轨迹当帧不得被标丢失(第 5 步跳过 rescued)。"""
    tracker = ByteTracker(TrackConfig(min_hits=1))
    tracker.update([_face(0, 0, 50, 50, score=0.9)], 1)
    tracker.update([_face(1, 1, 51, 51, score=0.3)], 2)
    t = tracker.get(1)
    assert t is not None
    assert t.state == _TrackState.TRACKED, "rescued track must stay TRACKED, got LOST"


def test_tracker_skip_keeps_hits_and_state():
    """修复:未检测帧只 predict,不判定丢失、不重置 hits。"""
    tracker = ByteTracker(TrackConfig(min_hits=1))
    tracker.update([_face(0, 0, 50, 50)], 1)
    snap = tracker.skip(2)  # 模拟 det_interval 跳过的帧
    assert len(snap) == 1
    assert snap[0].hits == 1, "hits must not change on skip frames"
    t = tracker.get(1)
    assert t.state == _TrackState.TRACKED, "track must not go LOST on skip frames"


def test_tracker_re_activate_after_occlusion():
    """丢失后重新出现,ID 复活(在 track_buffer 内)。"""
    tracker = ByteTracker(TrackConfig(min_hits=1, track_buffer=30, frame_rate=30))
    snap1 = tracker.update([_face(0, 0, 50, 50)], 1)
    tracker.update([], 2)  # 丢失
    snap3 = tracker.update([_face(2, 2, 52, 52)], 3)  # 复活
    assert len(snap3) == 1
    assert snap3[0].track_id == snap1[0].track_id


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
    assert cfg.track.track_thresh == 0.5
    assert cfg.track.max_tracks == 30
    assert cfg.recognition.threshold == 0.40


def test_vision_config_from_yaml_shape():
    cfg = VisionConfig.from_dict(
        {
            "device": "cpu",
            "det_size": [320, 320],
            "track": {"max_tracks": 10, "track_thresh": 0.4},
            "recognition": {"threshold": 0.55},
        }
    )
    assert cfg.device == "cpu"
    assert cfg.det_size == (320, 320)
    assert cfg.track.max_tracks == 10
    assert cfg.track.track_thresh == 0.4
    assert cfg.recognition.threshold == 0.55
