"""人脸识别质量筛选与轨迹多帧稳定聚合的纯逻辑测试。"""

import pytest

from backend.app.tasks.builtin.face_recognition_task import FaceRecognitionTask
from vision.events import PipelineContext, TrackResult


class FakeGallery:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = 0

    def search(self, query, threshold):
        self.calls += 1
        return self.results.pop(0) if self.results else None


class TrackerSpy:
    def __init__(self):
        self.identities = []

    def set_identity(self, track_id, identity, similarity):
        self.identities.append((track_id, identity, similarity))
        return True


def make_track(
    *,
    score=0.90,
    bbox=(0, 0, 100, 100),
    frame_id=1,
    embedding_frame_id=None,
):
    return TrackResult(
        track_id=1,
        bbox=bbox,
        score=score,
        embedding=[0.1] * 512,
        embedding_frame_id=frame_id if embedding_frame_id is None else embedding_frame_id,
    )


def context_for(track, frame_id=None):
    return PipelineContext(
        camera_id="cam-0",
        frame_id=track.embedding_frame_id if frame_id is None else frame_id,
        frame=None,
        tracks=[track],
    )


def make_task(overrides=None):
    gallery = FakeGallery()
    tracker = TrackerSpy()
    task = FaceRecognitionTask(
        config=overrides or {},
        full_config={"vision": {"recognition": {}}},
        gallery=gallery,
        tracker=tracker,
    )
    return task, gallery, tracker


def run_sample(task, frame_id, **kwargs):
    return task.run(
        None,
        context_for(make_track(frame_id=frame_id, **kwargs)),
    )


def test_low_detection_score_is_skipped_without_gallery_search():
    task, gallery, _ = make_task({"quality": {"min_det_score": 0.60}})

    events = run_sample(task, 1, score=0.59)

    assert events == []
    assert gallery.calls == 0
    assert task._states[1].skip_reasons["low_det_score"] == 1


def test_small_face_is_skipped_without_gallery_search():
    task, gallery, _ = make_task({"quality": {"min_face_size": 80}})

    events = run_sample(task, 1, bbox=(0, 0, 60, 100))

    assert events == []
    assert gallery.calls == 0
    assert task._states[1].skip_reasons["face_too_small"] == 1


def test_fewer_than_min_valid_samples_does_not_confirm_identity():
    task, gallery, tracker = make_task({"temporal": {"min_valid_samples": 3}})
    gallery.results = [
        ("id-a", "Alice", 0.70),
        ("id-a", "Alice", 0.71),
    ]

    assert run_sample(task, 1) == []
    assert run_sample(task, 2) == []
    assert tracker.identities == []


def test_top_k_average_is_used_for_event_confidence():
    task, gallery, tracker = make_task(
        {"temporal": {"min_valid_samples": 4, "top_k": 3}}
    )
    gallery.results = [
        ("id-a", "Alice", 0.51),
        ("id-a", "Alice", 0.73),
        ("id-a", "Alice", 0.61),
        ("id-a", "Alice", 0.55),
    ]

    batches = [run_sample(task, i) for i in range(1, 5)]
    event = next(item for batch in batches for item in batch)
    expected = (0.73 + 0.61 + 0.55) / 3

    assert event.confidence == pytest.approx(expected)
    assert event.payload["stable_score"] == pytest.approx(expected)
    assert tracker.identities[-1] == (1, "Alice", pytest.approx(expected))


def test_candidate_identity_samples_are_not_mixed():
    task, gallery, tracker = make_task({"temporal": {"min_valid_samples": 3}})
    gallery.results = [
        ("id-a", "Alice", 0.90),
        ("id-b", "Bob", 0.90),
        ("id-a", "Alice", 0.90),
        ("id-a", "Alice", 0.90),
    ]

    assert run_sample(task, 1) == []
    assert run_sample(task, 2) == []
    assert run_sample(task, 3) == []
    assert tracker.identities == []

    events = run_sample(task, 4)
    assert events[0].payload["name"] == "Alice"
    assert tracker.identities[-1][1] == "Alice"


def test_same_embedding_frame_is_not_sampled_twice():
    task, gallery, _ = make_task({"temporal": {"min_valid_samples": 2}})
    gallery.results = [
        ("id-a", "Alice", 0.80),
        ("id-a", "Alice", 0.81),
    ]

    run_sample(task, 1, embedding_frame_id=7)
    run_sample(task, 2, embedding_frame_id=7)

    assert gallery.calls == 1


def test_stable_average_below_threshold_does_not_confirm():
    task, gallery, tracker = make_task(
        {"threshold": 0.70, "temporal": {"min_valid_samples": 3}}
    )
    gallery.results = [
        ("id-a", "Alice", 0.60),
        ("id-a", "Alice", 0.61),
        ("id-a", "Alice", 0.62),
    ]

    assert all(run_sample(task, i) == [] for i in range(1, 4))
    assert tracker.identities == []


def test_event_confidence_equals_stable_score_and_changed_deduplicates():
    task, gallery, _ = make_task(
        {
            "recognized_cooldown_frames": 0,
            "temporal": {"min_valid_samples": 1},
        }
    )
    gallery.results = [
        ("id-a", "Alice", 0.80),
        ("id-a", "Alice", 0.82),
    ]

    first = run_sample(task, 1)
    second = run_sample(task, 2)

    assert first[0].confidence == pytest.approx(0.80)
    assert first[0].payload["similarity"] == pytest.approx(first[0].confidence)
    assert first[0].payload["changed"] is True
    assert second[0].confidence == pytest.approx(0.82)
    assert second[0].payload["similarity"] == pytest.approx(second[0].confidence)
    assert second[0].payload["changed"] is False


def test_unknown_track_keeps_existing_failure_cooldown():
    task, gallery, _ = make_task(
        {"cooldown_frames": 10, "failed_backoff_frames": 0}
    )
    gallery.results = [None, None]

    run_sample(task, 1)
    run_sample(task, 5)

    assert gallery.calls == 1


def test_confirmed_track_keeps_existing_recognized_cooldown():
    task, gallery, _ = make_task(
        {
            "recognized_cooldown_frames": 10,
            "temporal": {"min_valid_samples": 1},
        }
    )
    gallery.results = [
        ("id-a", "Alice", 0.90),
        ("id-a", "Alice", 0.91),
    ]

    first = run_sample(task, 1)
    run_sample(task, 5)

    assert first[0].payload["name"] == "Alice"
    assert gallery.calls == 1
