"""阶段7：AI Monitor FallDetectionTask 契约（Fake Runtime）。"""
from __future__ import annotations

import time
import uuid

import pytest

from ai_monitor_pose.task import FallDetectionTask
from ai_monitor_pose.errors import TaskBindingError
from tests.fixtures.transitions import make_transition


class FakeClock:
    def __init__(self):
        self._t = int(1e12)
    def monotonic_ns(self): return self._t
    def unix_ns(self): return 0
    def advance(self, ns): self._t += ns


class FakeRuntime:
    def __init__(self, compat=None, has_result=False):
        self.compat = list(compat or [])
        self.has_result = has_result
        self.closed = False
        self.offers = 0
    def has_latest_result_or_health_change(self, c, s): return self.has_result
    def has_unseen_compatibility_event(self, c, s): return bool(self.compat)
    def offer_frame(self, frame, meta): self.offers += 1
    def poll(self, c, s):
        from types import SimpleNamespace
        evs = list(self.compat); self.compat = []
        return SimpleNamespace(latest_result=None, compatibility_events=evs, health=None)
    def unregister_camera(self, c, s): pass
    def release(self): self.closed = True


def _cfg(enabled=True):
    import json
    base = {
        "enabled": enabled, "mode": "shadow",
        "runtime_key": "pose-cuda-0",
        "scheduler": {"target_fps": 8},
        "runtime": {"capacity_manifest_path": "D:/x/capacity.json",
                    "max_result_age_ms": 1500, "overlay_ttl_ms": 1200,
                    "frame_slots_per_camera": 2, "batch_size": 1,
                    "capacity_manifest_sha256": "a" * 64},
        "worker": {"python": "C:/x/python.exe"},
        "model": {"path": "D:/x/m.pt", "sha256_file": "D:/x/m.sha"},
        "gpu": {"required": True, "device": "cuda:0", "precision": "fp16",
                "allow_cpu_fallback": False},
        "algorithm": {
            "rotation_energy_min_rad_s": 1.8, "gravity_factor_min_body_heights_s2": 1.5,
            "fast_rotation_energy_min_rad_s": 3.0, "fast_gravity_factor_min_body_heights_s2": 2.5,
            "bbox_height_width_fall_max": 0.75, "trigger_ratio": 0.5,
            "min_trigger_duration_s": 0.5, "min_fall_pose_duration_s": 3.5,
            "recovery_duration_s": 1.0,
        },
        "cross_camera": {"enabled": False, "max_timestamp_skew_ms": 200},
    }
    return base


class _Factory:
    def __init__(self, runtimes=None):
        self.runtimes = list(runtimes or [])
    def acquire(self, runtime_key, config, event_sink, process_factory=None, **_kwargs):
        return self.runtimes.pop(0)


def _ctx(camera_id="cam-1", frame_id=1):
    from types import SimpleNamespace
    return SimpleNamespace(camera_id=camera_id, frame_id=frame_id, frame=object(), tracks=[])


def test_constructor_does_not_start_worker_or_import_torch() -> None:
    import sys
    assert "torch" not in sys.modules
    t = FallDetectionTask(_cfg())
    assert t.name == "fall_detection"


def test_first_context_binds_camera_and_session() -> None:
    rt = FakeRuntime()
    t = FallDetectionTask(_cfg(), runtime_factory=_Factory([rt]))
    assert t.should_run(1, _ctx(frame_id=1)) is True
    assert t._camera_id == "cam-1"


def test_run_submits_without_waiting() -> None:
    rt = FakeRuntime()
    t = FallDetectionTask(_cfg(), runtime_factory=_Factory([rt]))
    t.should_run(1, _ctx())
    started = time.perf_counter()
    evs = t.run(None, _ctx())
    dt = (time.perf_counter() - started) * 1000
    assert dt < 20
    assert rt.offers == 1
    assert isinstance(evs, list)


def test_same_task_rejects_different_camera() -> None:
    rt = FakeRuntime()
    t = FallDetectionTask(_cfg(), runtime_factory=_Factory([rt]))
    t.should_run(1, _ctx())
    with pytest.raises(TaskBindingError):
        t.should_run(1, _ctx(camera_id="cam-2"))


def test_compatibility_mode_maps_transitions_without_deleting() -> None:
    tr = make_transition(etype="fall_detected")
    rt = FakeRuntime(compat=[tr])
    t = FallDetectionTask(_cfg(), runtime_factory=_Factory([rt]))
    t.should_run(1, _ctx())
    evs = t.run(None, _ctx())
    assert [e.event_type for e in evs] == ["fall_detected"]
    assert evs[0].payload["track_namespace"] == "pose"


def test_close_releases_only_own_camera_and_is_idempotent() -> None:
    rt = FakeRuntime()
    t = FallDetectionTask(_cfg(), runtime_factory=_Factory([rt]))
    t.should_run(1, _ctx())
    t.close()
    assert rt.closed is True
    t.close()  # 幂等
    assert rt.closed is True


# ── 姿态叠加 analytics 生产者 ──────────────────────────────

class ResultRuntime(FakeRuntime):
    def __init__(self, result=None):
        super().__init__()
        self.result = result
        self.state = "READY"

    def poll(self, c, s):
        from types import SimpleNamespace
        return SimpleNamespace(latest_result=self.result, compatibility_events=(), health=self.state)


def _result_dict(session, source_frame_id=98):
    return {
        "camera_id": "cam-1", "camera_session_id": session,
        "source_frame_id": source_frame_id, "source_width": 1920, "source_height": 1080,
        "coordinate_space": "source_pixels", "end_to_end_ms": 61.7,
        "tracks": [{
            "pose_track_id": 3, "state": "normal", "detection_score": 0.9,
            "bbox_xyxy": [100.0, 200.0, 300.0, 700.0],
            "keypoints_coco17": [[100.0, 200.0, 0.9], [110.0, 210.0, 0.8]],
        }],
    }


def _ctx_analytics(camera_id="cam-1", frame_id=100):
    from types import SimpleNamespace
    return SimpleNamespace(camera_id=camera_id, frame_id=frame_id, frame=object(), tracks=[], analytics={})


def test_run_attaches_pose_analytics_from_latest_result() -> None:
    rt = ResultRuntime()
    t = FallDetectionTask(_cfg(), runtime_factory=_Factory([rt]))
    rt.result = _result_dict(t.camera_session_id)
    t.should_run(1, _ctx())
    ctx = _ctx_analytics()
    t.run(None, ctx)
    fd = ctx.analytics["fall_detection"]
    assert fd["camera_session_id"] == t.camera_session_id
    assert fd["attached_to_frame_id"] == 100
    assert fd["source_frame_id"] == 98
    assert fd["source_width"] == 1920 and fd["source_height"] == 1080
    assert fd["coordinate_space"] == "source_pixels"
    assert fd["health"] == "READY"
    assert fd["overlay_expires_in_ms"] > 0
    tr = fd["tracks"][0]
    # bbox 由 xyxy 投影为 xywh
    assert tr["bbox"] == [100.0, 200.0, 200.0, 500.0]
    assert tr["keypoints"][0][:2] == [100.0, 200.0]
    assert tr["state"] == "normal"


def test_analytics_cleared_after_overlay_ttl() -> None:
    clock = FakeClock()
    rt = ResultRuntime()
    t = FallDetectionTask(_cfg(), runtime_factory=_Factory([rt]), clock=clock)
    rt.result = _result_dict(t.camera_session_id)
    t.should_run(1, _ctx())
    ctx = _ctx_analytics()
    t.run(None, ctx)
    assert "fall_detection" in ctx.analytics
    # 越过 overlay_ttl_ms(1200)后重挂:应清除缓存且不再写 analytics
    clock.advance(1_300_000_000)
    rt.result = None  # 无新结果
    ctx2 = _ctx_analytics()
    t.run(None, ctx2)
    assert "fall_detection" not in ctx2.analytics
    assert t._last_result is None
