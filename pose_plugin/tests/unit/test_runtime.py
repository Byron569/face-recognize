"""阶段6剩余（单元）：Registry 引用计数 / 全局配置冲突 / sink 冲突 / Runtime 未就绪与关闭语义。

本文件不产生真实子进程；用 StubChild 覆盖 Runtime，用内存 Registry 覆盖 acquire/release。
"""
from __future__ import annotations

import time

import pytest

from ai_monitor_pose.config import RuntimeConfig
from ai_monitor_pose.runtime import PoseRuntime, RuntimeOfferOutcome
from ai_monitor_pose.runtime_registry import (
    PoseRuntimeRegistry,
    RUNTIME_CONFIG_CONFLICT,
    RUNTIME_EVENT_SINK_CONFLICT,
)

_SINK_A = object()
_SINK_B = object()


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


class StubChild:
    """一个永不就绪的假子进程：保持 is_alive=False，写即抛。"""

    pid = -1
    stdin = None
    stdout = None

    def is_alive(self) -> bool:
        return False

    def write(self, env: dict) -> None:
        raise BrokenPipeError("stub child not alive")

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass

    def wait(self, timeout=None) -> int | None:
        return -1

    def close(self) -> None:
        pass


def _stub_factory():
    return StubChild()


def _fresh_registry():
    PoseRuntimeRegistry._reset()


# ---------------- Registry：config 冲突 ----------------

def test_same_runtime_key_rejects_different_global_config():
    _fresh_registry()
    l1 = PoseRuntimeRegistry.acquire("rk1", _cfg(max_cameras=4), _SINK_A,
                                     process_factory=_stub_factory)
    with pytest.raises(RuntimeError) as ei:
        PoseRuntimeRegistry.acquire("rk1", _cfg(max_cameras=8), _SINK_A,
                                    process_factory=_stub_factory)
    assert RUNTIME_CONFIG_CONFLICT in str(ei.value)
    PoseRuntimeRegistry.release(l1)


def test_multiple_leases_return_same_runtime_and_counts_refs():
    _fresh_registry()
    l1 = PoseRuntimeRegistry.acquire("rk2", _cfg(), _SINK_A, process_factory=_stub_factory)
    l2 = PoseRuntimeRegistry.acquire("rk2", _cfg(), _SINK_A, process_factory=_stub_factory)
    assert l1.runtime is l2.runtime
    assert PoseRuntimeRegistry._refcounts["rk2"] == 2
    PoseRuntimeRegistry.release(l1)
    assert PoseRuntimeRegistry._refcounts["rk2"] == 1
    PoseRuntimeRegistry.release(l2)
    _wait_idle("rk2")


def _wait_idle(key: str, s: float = 1.0) -> None:
    end = time.monotonic() + s
    while time.monotonic() < end:
        if PoseRuntimeRegistry._refcounts.get(key, 0) <= 0:
            return
        time.sleep(0.01)


def test_first_acquire_binds_sink_and_different_sink_is_rejected():
    _fresh_registry()
    l1 = PoseRuntimeRegistry.acquire("rk3", _cfg(), _SINK_A, process_factory=_stub_factory)
    with pytest.raises(RuntimeError) as ei:
        PoseRuntimeRegistry.acquire("rk3", _cfg(), _SINK_B, process_factory=_stub_factory)
    assert RUNTIME_EVENT_SINK_CONFLICT in str(ei.value)
    PoseRuntimeRegistry.release(l1)
    _wait_idle("rk3")


def test_release_is_idempotent():
    _fresh_registry()
    l = PoseRuntimeRegistry.acquire("rk4", _cfg(), None, process_factory=_stub_factory)
    PoseRuntimeRegistry.release(l)
    PoseRuntimeRegistry.release(l)
    _wait_idle("rk4")


def test_health_snapshot_shape_for_existing_runtime():
    _fresh_registry()
    l = PoseRuntimeRegistry.acquire("rk5", _cfg(), None, process_factory=_stub_factory)
    snap = PoseRuntimeRegistry.health_snapshot("rk5")
    assert snap.runtime_key == "rk5"
    assert snap.worker.state in ("STARTING", "READY", "UNAVAILABLE", "STOPPING", "STOPPED", "DEGRADED")
    assert getattr(snap.worker, "worker_pid", None) is not None or snap.worker.worker_pid is None
    PoseRuntimeRegistry.release(l)
    _wait_idle("rk5")


# ---------------- Runtime：未就绪 / 关闭 不阻塞 ----------------

def test_offer_before_worker_ready_returns_not_ready_without_blocking():
    rt = PoseRuntime(_cfg(), process_factory=_stub_factory)
    rt.start()
    t0 = time.monotonic()
    out = rt.offer(camera_id="c1", camera_session_id="s1", frame_id=1, config_revision="r")
    assert (time.monotonic() - t0) < 0.05
    assert out == RuntimeOfferOutcome.WORKER_NOT_READY
    rt.stop_and_drain_blocking()

def test_offer_after_close_returns_closed():
    rt = PoseRuntime(_cfg(), process_factory=_stub_factory)
    rt.start()
    rt.stop_and_drain_blocking()
    assert rt.offer(camera_id="c", camera_session_id="s", frame_id=0, config_revision="r") == RuntimeOfferOutcome.CLOSED


# ---------------- 修复2：health_snapshot 真实计数（submitted/analyzed/replaced） ----------------

import threading

from ai_monitor_pose.event_spool import EventSpool
from ai_monitor_pose.host_protocols import EventSinkAck
from ai_monitor_pose.worker.transition_journal import WorkerJournal
from tests.fixtures.transitions import make_transition


class _RecordingChild:
    """可写的假子进程：offer 能成功（记录消息），stdout=None 不启动读线程。"""

    pid = 4242
    stdout = None

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def is_alive(self) -> bool:
        return True

    def write(self, env: dict) -> None:
        self.sent.append(env)

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass

    def wait(self, timeout=None) -> int:
        return 0

    def close(self) -> None:
        pass


def _feed_ready(rt: PoseRuntime) -> None:
    rt._ingest({"message_type": "WORKER_READY", "worker_epoch": "e1",
                "payload": {"worker_epoch": "e1", "worker_instance_id": "w1",
                            "worker_pid": 4242, "device": "cuda:0",
                            "device_name": "Fake GPU", "model_sha256": "x",
                            "precision": "fp16"}})


def _feed_result(rt: PoseRuntime, camera="c1", session="s1", request_id="r1") -> None:
    rt._ingest({"message_type": "INFERENCE_RESULT", "worker_epoch": "e1",
                "payload": {"request_id": request_id, "camera_id": camera,
                            "camera_session_id": session, "status": "ok"}})


def test_health_snapshot_reports_real_totals_per_camera():
    rt = PoseRuntime(_cfg(), process_factory=lambda: _RecordingChild())
    rt.start()
    _feed_ready(rt)
    assert rt.offer("c1", "s1", 1, "r") == RuntimeOfferOutcome.ACCEPTED
    assert rt.offer("c1", "s1", 2, "r") == RuntimeOfferOutcome.ACCEPTED
    _feed_result(rt)                       # analyzed=1
    _feed_result(rt, request_id="r2")     # 覆盖未消费 latest → replaced=1
    snap = rt.health_snapshot("rk")
    (cam,) = snap.cameras
    assert (cam.camera_id, cam.camera_session_id) == ("c1", "s1")
    assert cam.submitted_total == 2
    assert cam.analyzed_total == 2
    assert cam.replaced_total == 1
    rt.stop_and_drain_blocking()


def test_health_snapshot_counters_are_isolated_per_camera():
    rt = PoseRuntime(_cfg(), process_factory=lambda: _RecordingChild())
    rt.start()
    _feed_ready(rt)
    rt.offer("c1", "s1", 1, "r")
    rt.offer("c2", "s2", 1, "r")
    _feed_result(rt, camera="c1")
    snap = rt.health_snapshot()
    by_cam = {(c.camera_id, c.camera_session_id): c for c in snap.cameras}
    assert by_cam[("c1", "s1")].submitted_total == 1
    assert by_cam[("c2", "s2")].submitted_total == 1
    assert by_cam[("c1", "s1")].analyzed_total == 1
    assert by_cam[("c2", "s2")].analyzed_total == 0
    assert by_cam[("c1", "s1")].replaced_total == 0
    rt.stop_and_drain_blocking()


def test_health_snapshot_counters_survive_concurrent_offers():
    rt = PoseRuntime(_cfg(), process_factory=lambda: _RecordingChild())
    rt.start()
    _feed_ready(rt)

    def _bombard(cam: str) -> None:
        for i in range(50):
            rt.offer(cam, "s1", i, "r")

    threads = [threading.Thread(target=_bombard, args=(f"c{k}",)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = rt.health_snapshot()
    assert sum(c.submitted_total for c in snap.cameras) == 200
    rt.stop_and_drain_blocking()


# ---------------- 修复3：_drain_journal 成功才 ack + 毒丸落父端 spool ----------------

import concurrent.futures
import json
import sqlite3
import sys
import types


@pytest.fixture()
def _vision_stub():
    """event_mapper 需要宿主 vision.events；姿态 venv 缺该包，安装最小桩并按测试还原。"""
    keys = ("vision", "vision.events")
    saved = {k: sys.modules.get(k) for k in keys}
    vision = types.ModuleType("vision")
    events = types.ModuleType("vision.events")

    class VisionEvent:
        def __init__(self, *, event_type, camera_id, track_id, confidence, timestamp, payload):
            self.event_type = event_type
            self.camera_id = camera_id
            self.track_id = track_id
            self.confidence = confidence
            self.timestamp = timestamp
            self.payload = payload

    events.VisionEvent = VisionEvent
    vision.events = events
    sys.modules["vision"] = vision
    sys.modules["vision.events"] = events
    yield
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


class _ScriptedSink:
    """按脚本返回已完成 Future 的 event sink。

    脚本项：'fail'（Future 异常）/ 'ok'（event_id 匹配且 persisted）/
    'bad-id'（ACK event_id 不匹配）/ 'unpersisted'（persisted=False）。
    脚本耗尽后重复最后一项。
    """

    def __init__(self, script) -> None:
        self.script = list(script)
        self.i = 0
        self.submitted: list = []

    def submit(self, event):
        self.submitted.append(event)
        fut = concurrent.futures.Future()
        step = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        if step == "fail":
            fut.set_exception(RuntimeError("sink down"))
        elif step == "bad-id":
            fut.set_result(EventSinkAck(event_id="wrong-id", persisted=True))
        elif step == "unpersisted":
            fut.set_result(EventSinkAck(event_id=event.event_id, persisted=False))
        else:
            fut.set_result(EventSinkAck(event_id=event.event_id, persisted=True))
        return fut


def _drain_cfg(tmp_path, **kw) -> RuntimeConfig:
    return _cfg(worker_journal_path=str(tmp_path / "journal.sqlite3"),
                event_spool_path=str(tmp_path / "spool.sqlite3"), **kw)


def _seed_journal(tmp_path, tr) -> str:
    jp = str(tmp_path / "journal.sqlite3")
    j = WorkerJournal(jp, worker_instance_id="w1")
    seq = j.begin_add()
    j.commit(seq, tr)
    j.close()
    return jp


def _journal_attempt_count(jp: str) -> int:
    conn = sqlite3.connect(jp)
    try:
        return int(conn.execute("SELECT attempt_count FROM journal").fetchone()[0])
    finally:
        conn.close()


def _clear_backoff(jp: str) -> None:
    conn = sqlite3.connect(jp)
    try:
        conn.execute("UPDATE journal SET next_attempt_at=0")
        conn.commit()
    finally:
        conn.close()


def _pending_event_ids(jp: str) -> list[str]:
    # 断言“未 ack”以 parent_acked_at 为准；pending() 带退避过滤，不含未到期重试事件
    conn = sqlite3.connect(jp)
    try:
        return [r[0] for r in conn.execute(
            "SELECT event_id FROM journal WHERE parent_acked_at IS NULL").fetchall()]
    finally:
        conn.close()


def test_drain_journal_keeps_pending_when_sink_future_fails(_vision_stub, tmp_path):
    tr = make_transition()
    jp = _seed_journal(tmp_path, tr)
    sink = _ScriptedSink(["fail"])
    rt = PoseRuntime(_drain_cfg(tmp_path), process_factory=_stub_factory, event_sink=sink)
    rt._drain_journal()
    assert _pending_event_ids(jp) == [tr.event_id]   # 未 ack，保留 pending
    assert _journal_attempt_count(jp) == 1           # 失败已计数
    rt.stop_and_drain_blocking()


def test_drain_journal_retries_on_next_round_and_acks_on_success(_vision_stub, tmp_path):
    tr = make_transition()
    jp = _seed_journal(tmp_path, tr)
    sink = _ScriptedSink(["fail", "ok"])
    rt = PoseRuntime(_drain_cfg(tmp_path), process_factory=_stub_factory, event_sink=sink)
    rt._drain_journal()
    assert _pending_event_ids(jp) == [tr.event_id]
    _clear_backoff(jp)
    rt._drain_journal()                              # 下轮重试成功
    assert _pending_event_ids(jp) == []
    rt.stop_and_drain_blocking()


def test_drain_journal_rejects_mismatched_and_unpersisted_ack(_vision_stub, tmp_path):
    for step in ("bad-id", "unpersisted"):
        tr = make_transition()
        sub = tmp_path / step
        sub.mkdir()
        jp = _seed_journal(sub, tr)
        sink = _ScriptedSink([step])
        rt = PoseRuntime(_drain_cfg(sub), process_factory=_stub_factory, event_sink=sink)
        rt._drain_journal()
        assert _pending_event_ids(jp) == [tr.event_id]
        rt.stop_and_drain_blocking()


def test_drain_journal_poison_pill_spooled_and_acked_after_max_failures(_vision_stub, tmp_path):
    tr = make_transition()
    jp = _seed_journal(tmp_path, tr)
    sink = _ScriptedSink(["fail"])
    rt = PoseRuntime(_drain_cfg(tmp_path), process_factory=_stub_factory, event_sink=sink,
                     drain_max_attempts=2)
    rt._drain_journal()                # 失败 #1
    _clear_backoff(jp)
    rt._drain_journal()                # 失败 #2 → 达上限 → 落父端 spool 兜底并 ack
    assert _pending_event_ids(jp) == []
    sp = EventSpool(rt.config.event_spool_path, pending_capacity=100)
    try:
        assert [p["event_id"] for p in sp.pending()] == [tr.event_id]
    finally:
        sp.close()
    rt.stop_and_drain_blocking()


def test_drain_journal_without_sink_acks_immediately(tmp_path):
    tr = make_transition()
    jp = _seed_journal(tmp_path, tr)
    rt = PoseRuntime(_drain_cfg(tmp_path), process_factory=_stub_factory)
    rt._drain_journal()
    assert _pending_event_ids(jp) == []
    rt.stop_and_drain_blocking()


def test_drain_journal_corrupt_payload_is_poison_pilled_to_spool(tmp_path):
    tr = make_transition()
    jp = _seed_journal(tmp_path, tr)
    conn = sqlite3.connect(jp)
    conn.execute("UPDATE journal SET payload_json='not-json'")
    conn.commit()
    conn.close()
    rt = PoseRuntime(_drain_cfg(tmp_path), process_factory=_stub_factory,
                     drain_max_attempts=2)
    rt._drain_journal()
    _clear_backoff(jp)
    rt._drain_journal()
    assert _pending_event_ids(jp) == []
    sp = EventSpool(rt.config.event_spool_path, pending_capacity=100)
    try:
        pend = list(sp.pending())
    finally:
        sp.close()
    assert [p["event_id"] for p in pend] == [tr.event_id]
    payload = json.loads(pend[0]["payload"])
    assert payload.get("poison_pill") is True
    rt.stop_and_drain_blocking()


def test_drain_journal_retry_does_not_duplicate_compatibility_events(_vision_stub, tmp_path):
    tr = make_transition()
    jp = _seed_journal(tmp_path, tr)
    sink = _ScriptedSink(["fail", "fail", "ok"])
    rt = PoseRuntime(_drain_cfg(tmp_path), process_factory=_stub_factory, event_sink=sink)
    for _ in range(3):
        rt._drain_journal()
        _clear_backoff(jp)
    assert _pending_event_ids(jp) == []
    evts = rt._compatibility_by_camera.get((tr.camera_id, tr.camera_session_id), [])
    assert [e.event_id for e in evts] == [tr.event_id]
    rt.stop_and_drain_blocking()
