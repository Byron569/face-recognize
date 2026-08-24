"""阶段6剩余（集成/生命周期）：真实 fake_worker 子进程，验证多 lease 单 Worker、握手、引用计数停止、单摄不影响他摄。"""
from __future__ import annotations

import time

import pytest

from ai_monitor_pose.config import RuntimeConfig
from ai_monitor_pose.runtime import PoseRuntime, RuntimeOfferOutcome
from ai_monitor_pose.runtime_registry import PoseRuntimeRegistry
from tests.fixtures.fake_worker import make_process_factory


def _cfg(**kw) -> RuntimeConfig:
    defaults = dict(
        max_cameras=4, requested_max_total_fps=32, capacity_manifest_path="C:/cap.json",
        capacity_manifest_sha256="0" * 64, capacity_headroom_ratio=0.75,
        frame_slots_per_camera=2, max_frame_width=1920, max_frame_height=1080,
        max_result_age_ms=1500, overlay_ttl_ms=1200,
        transition_queue_capacity_per_camera=64, transition_queue_resume_ratio=0.5,
        worker_journal_path="", worker_journal_pending_capacity=10000,
        event_spool_path="", event_spool_pending_capacity=10000,
        delivered_retention_hours=24, delivered_retention_rows=1000,
    )
    defaults.update(kw)
    return RuntimeConfig(**defaults)


@pytest.fixture(autouse=True)
def _cleanup():
    PoseRuntimeRegistry._reset()
    yield
    PoseRuntimeRegistry._reset()


def wait_ready(rt: PoseRuntime, s: float = 5.0) -> bool:
    end = time.monotonic() + s
    while time.monotonic() < end:
        if rt.state == "READY":
            return True
        rt.poll()
        time.sleep(0.01)
    return False


def test_multiple_leases_start_exactly_one_worker():
    l1 = PoseRuntimeRegistry.acquire("lk1", _cfg(), None,
                                     process_factory=make_process_factory())
    l2 = PoseRuntimeRegistry.acquire("lk1", _cfg(), None,
                                     process_factory=make_process_factory())
    rt = l1.runtime
    assert l2.runtime is rt
    assert rt.start_count == 1
    assert wait_ready(rt)
    assert rt._children_alive() == 1
    PoseRuntimeRegistry.release(l1)
    PoseRuntimeRegistry.release(l2)


def test_ready_handshake_contains_epoch_pid_and_device_metadata():
    l = PoseRuntimeRegistry.acquire("lk2", _cfg(), None,
                                    process_factory=make_process_factory(cfg={"device_name": "Fake GPU"}))
    rt = l.runtime
    assert wait_ready(rt)
    assert rt.worker_epoch and isinstance(rt.worker_epoch, str)
    assert rt.worker_pid and rt.worker_pid > 0
    assert rt.gpu_device_name == "Fake GPU"
    assert rt.precision == "fp16"
    PoseRuntimeRegistry.release(l)


def test_worker_start_is_lazy_and_nonblocking():
    t0 = time.perf_counter()
    l = PoseRuntimeRegistry.acquire("lk3", _cfg(), None,
                                    process_factory=make_process_factory())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 500  # acquire 不等 CUDA / 不等 READY
    PoseRuntimeRegistry.release(l)


def test_last_lease_close_stops_worker_and_unlinks():
    l = PoseRuntimeRegistry.acquire("lk4", _cfg(), None,
                                    process_factory=make_process_factory())
    rt = l.runtime
    assert wait_ready(rt)
    PoseRuntimeRegistry.release(l)
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        if rt.state == "STOPPED":
            break
        time.sleep(0.01)
    assert rt.state == "STOPPED"
    assert rt._children_alive() == 0


def test_closing_one_camera_keeps_other_camera_alive():
    la = PoseRuntimeRegistry.acquire("lka", _cfg(), None, process_factory=make_process_factory())
    lb = PoseRuntimeRegistry.acquire("lkb", _cfg(), None, process_factory=make_process_factory())
    ra, rb = la.runtime, lb.runtime
    assert wait_ready(ra) and wait_ready(rb)
    PoseRuntimeRegistry.release(la)
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        if ra.state == "STOPPED":
            break
        time.sleep(0.01)
    assert ra.state == "STOPPED"
    assert rb.state == "READY"
    assert rb._children_alive() == 1
    PoseRuntimeRegistry.release(lb)