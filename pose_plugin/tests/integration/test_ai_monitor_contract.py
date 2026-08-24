"""阶段7：多 Task 共享单一 Runtime；close 只释放自身（无需 DB）。"""
from __future__ import annotations

import sys

from ai_monitor_pose.task import FallDetectionTask
from tests.fixtures.transitions import make_transition


class FakeClock:
    def __init__(self): self._t = int(1e12)
    def monotonic_ns(self): return self._t
    def unix_ns(self): return 0


class FakeRuntime:
    def __init__(self):
        self.leases = 0
        self.offers = 0
        self.unreg = []
        self.compat = []
    def has_latest_result_or_health_change(self, c, s): return False
    def has_unseen_compatibility_event(self, c, s): return False
    def offer_frame(self, frame, meta): self.offers += 1
    def poll(self, c, s):
        from types import SimpleNamespace
        evs = list(self.compat); self.compat = []
        return SimpleNamespace(latest_result=None, compatibility_events=evs, health=None)
    def unregister_camera(self, c, s): self.unreg.append((c, s))
    def release(self): self.leases += 1


class SharedFactory:
    """模拟注册表：同一 runtime_key 返回同一 lease。"""
    def __init__(self, rt): self.rt = rt
    def acquire(self, runtime_key, config, event_sink, process_factory=None, **_kwargs): return self.rt


def _cfg():
    return {
        "enabled": True, "mode": "shadow", "runtime_key": "pose-cuda-0",
        "scheduler": {"target_fps": 8},
        "runtime": {"capacity_manifest_path": "D:/x/c.json",
                    "capacity_manifest_sha256": "a" * 64,
                    "max_result_age_ms": 1500, "overlay_ttl_ms": 1200,
                    "frame_slots_per_camera": 2, "batch_size": 1},
        "worker": {"python": "C:/x/p.exe"},
        "model": {"path": "D:/x/m.pt", "sha256_file": "D:/x/m.sha"},
        "gpu": {"required": True, "device": "cuda:0", "precision": "fp16",
                "allow_cpu_fallback": False},
        "algorithm": {"rotation_energy_min_rad_s": 1.8, "gravity_factor_min_body_heights_s2": 1.5,
                      "fast_rotation_energy_min_rad_s": 3.0, "fast_gravity_factor_min_body_heights_s2": 2.5,
                      "bbox_height_width_fall_max": 0.75, "trigger_ratio": 0.5,
                      "min_trigger_duration_s": 0.5, "min_fall_pose_duration_s": 3.5,
                      "recovery_duration_s": 1.0},
        "cross_camera": {"enabled": False, "max_timestamp_skew_ms": 200},
    }


def _ctx(camera, frame_id=1):
    from types import SimpleNamespace
    return SimpleNamespace(camera_id=camera, frame_id=frame_id, frame=object(), tracks=[])


def test_two_tasks_share_one_runtime() -> None:
    rt = FakeRuntime()
    fac = SharedFactory(rt)
    t1 = FallDetectionTask(_cfg(), runtime_factory=fac)
    t2 = FallDetectionTask(_cfg(), runtime_factory=fac)
    t1.should_run(1, _ctx("cam-a"))
    t2.should_run(1, _ctx("cam-b"))
    assert t1._lease is rt and t2._lease is rt  # 同一 Runtime
    t1.run(None, _ctx("cam-a"))
    t2.run(None, _ctx("cam-b"))
    assert rt.offers == 2


def test_close_releases_only_own_camera_and_is_idempotent() -> None:
    rt = FakeRuntime()
    t1 = FallDetectionTask(_cfg(), runtime_factory=SharedFactory(rt))
    t1.should_run(1, _ctx("cam-x"))
    t1.close()
    assert ("cam-x", t1.camera_session_id) in rt.unreg
    assert rt.leases == 1
    t1.close()
    assert rt.leases == 1  # 幂等，不重复 release


def test_two_isolated_tasks_have_unique_sessions() -> None:
    rt = FakeRuntime()
    fac = SharedFactory(rt)
    a = FallDetectionTask(_cfg(), runtime_factory=fac)
    b = FallDetectionTask(_cfg(), runtime_factory=fac)
    assert a.camera_session_id != b.camera_session_id
