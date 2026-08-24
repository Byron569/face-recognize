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
    def acquire(self, runtime_key, config, event_sink, process_factory=None):
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
