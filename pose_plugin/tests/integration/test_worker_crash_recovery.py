"""阶段6剩余（集成/崩溃恢复）：fake_worker 崩溃→UNAVAILABLE、非阻塞 offer、old-epoch 丢弃、重启熔断。"""
from __future__ import annotations

import time

import pytest

from ai_monitor_pose.config import RuntimeConfig
from ai_monitor_pose.runtime import RuntimeOfferOutcome
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


def wait_state(rt, states, s=6.0) -> bool:
    end = time.monotonic() + s
    while time.monotonic() < end:
        rt.poll()
        if rt.state in states:
            return True
        time.sleep(0.01)
    return False


def test_worker_crash_marks_unavailable_within_heartbeat_timeout():
    l = PoseRuntimeRegistry.acquire("ck1", _cfg(), None,
                                    process_factory=make_process_factory(crash_after=1),
                                    restart_limit=0)
    rt = l.runtime
    assert wait_state(rt, ("READY",))
    # 第一次 INFER 触发崩溃
    assert rt.offer(camera_id="c", camera_session_id="s", frame_id=1,
                    config_revision="r") == RuntimeOfferOutcome.ACCEPTED
    assert wait_state(rt, ("UNAVAILABLE",))
    assert rt.restart_count == 0  # restart_limit=0 表示不重启
    PoseRuntimeRegistry.release(l)


def test_worker_crash_never_blocks_offer_frame():
    l = PoseRuntimeRegistry.acquire("ck2", _cfg(), None,
                                    process_factory=make_process_factory(crash_after=1),
                                    restart_limit=2)
    rt = l.runtime
    wait_state(rt, ("READY",))
    t0 = time.monotonic()
    rt.offer(camera_id="c", camera_session_id="s", frame_id=1, config_revision="r")
    assert (time.monotonic() - t0) < 0.1
    PoseRuntimeRegistry.release(l)


def test_old_epoch_results_are_discarded_after_restart():
    l = PoseRuntimeRegistry.acquire("ck3", _cfg(), None,
                                    process_factory=make_process_factory(crash_after=1),
                                    restart_limit=3)
    rt = l.runtime
    assert wait_state(rt, ("READY",))
    epoch0 = rt.worker_epoch
    rt.offer(camera_id="c", camera_session_id="s", frame_id=1, config_revision="r")
    # 等待重启完成一轮，epoch 变化
    end = time.monotonic() + 6.0
    while time.monotonic() < end and (rt.worker_epoch == epoch0 or rt.state != "READY"):
        rt.poll()
        time.sleep(0.01)
    assert rt.worker_epoch != epoch0
    # 旧 epoch 结果注入必须被丢弃
    rt._ingest({"message_type": "INFERENCE_RESULT", "worker_epoch": epoch0, "payload": {"request_id": "x"}})
    assert rt.latest_result("c", "s") is None
    PoseRuntimeRegistry.release(l)


def test_restart_limit_opens_circuit_breaker():
    l = PoseRuntimeRegistry.acquire("ck4", _cfg(), None,
                                    process_factory=make_process_factory(crash_after=1),
                                    restart_limit=1)
    rt = l.runtime
    assert wait_state(rt, ("READY",))
    # 反复 offer 驱动每个 READY 子进程在第一次 INFER 崩溃；每轮重启后再次崩溃，直到 rc 超过 restart_limit 熔断
    end = time.monotonic() + 8.0
    while time.monotonic() < end:
        rt.poll()
        if rt.circuit_open:
            break
        rt.offer(camera_id="c", camera_session_id="s", frame_id=1, config_revision="r")
        time.sleep(0.02)
    assert rt.circuit_open
    assert rt.state == "UNAVAILABLE"
    assert rt._children_alive() == 0
    PoseRuntimeRegistry.release(l)