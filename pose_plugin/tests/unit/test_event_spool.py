
"""阶段6：父端 durable event spool。"""
from __future__ import annotations

import sqlite3
import uuid

from ai_monitor_pose.event_spool import EventSpool
from tests.fixtures.transitions import make_transition


def _spool(tmp_path):
    return EventSpool(str(tmp_path / ("sp-" + uuid.uuid4().hex[:6] + ".sqlite3")),
                      pending_capacity=1000)


def test_transition_is_kept_until_parent_sqlite_commit_ack(tmp_path) -> None:
    sp = _spool(tmp_path)
    ev = make_transition()
    tid = sp.add(ev)
    assert list(sp.pending()) != []
    sp.mark_delivered(ev.event_id)
    assert list(sp.pending()) == []


def test_transition_ack_can_accept_subset_and_rejected_ids_remain_pending(tmp_path) -> None:
    sp = _spool(tmp_path)
    a = make_transition(camera_id="a")
    b = make_transition(camera_id="b", frame=6)
    sp.add(a); sp.add(b)
    sp.mark_delivered(a.event_id)
    pend = [p["event_id"] for p in sp.pending()]
    assert a.event_id not in pend and b.event_id in pend


def test_spool_full_pauses_inference_and_low_watermark_resumes(tmp_path) -> None:
    sp = EventSpool(str(tmp_path / "x.sqlite3"), pending_capacity=5)
    for i in range(5):
        sp.add(make_transition(camera_id=f"c{i}", frame=i))
    assert sp.is_full()
    # 只剩余1个容量，第6个不能入队 -> full
    ok = sp.try_add(make_transition(camera_id="overflow", frame=99))
    assert ok is False
    # ack 到低于 50% 恢复
    for i in range(4):
        try:
            ev = list(sp.pending())[0]
            sp.mark_delivered(ev["event_id"])
        except IndexError:
            break
    assert not sp.is_full()
    assert sp.try_add(make_transition(camera_id="again", frame=100)) is True


def test_prune_never_removes_pending(tmp_path) -> None:
    sp = _spool(tmp_path)
    sp.add(make_transition(camera_id="keep"))
    sp.prune_delivered(retention_hours=999, retention_rows=0)
    assert list(sp.pending()) != []


def test_pending_capacity_ignores_delivered(tmp_path) -> None:
    sp = EventSpool(str(tmp_path / "y.sqlite3"), pending_capacity=3)
    e1 = make_transition(camera_id="d")
    sp.add(e1)
    sp.mark_delivered(e1.event_id)
    assert list(sp.pending()) == []
    for i in range(3):
        sp.add(make_transition(camera_id=f"z{i}", frame=i))
    # 已投递的不计入 pending 容量
    assert sp.pending_count() == 3
