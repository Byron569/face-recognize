
"""阶段6：Worker crash-safe transition journal。"""
from __future__ import annotations

import sqlite3
import uuid

import pytest

from ai_monitor_pose.worker.transition_journal import WorkerJournal
from tests.fixtures.transitions import make_transition


def _jpath(tmp_path):
    return tmp_path / (uuid.uuid4().hex[:6] + ".sqlite3")


def test_journal_commits_before_fsm_state_is_applied(tmp_path) -> None:
    j = WorkerJournal(str(_jpath(tmp_path)), worker_instance_id="w1")
    ev = make_transition()
    seq = j.begin_add()
    j.commit(seq, ev)
    # 已持久化，可查询
    assert len(list(j.pending())) == 1


def test_parent_death_after_journal_commit_replays(tmp_path) -> None:
    j = WorkerJournal(str(_jpath(tmp_path)), worker_instance_id="w1")
    ev = make_transition()
    seq = j.begin_add()
    j.commit(seq, ev)
    # 模拟父端从未 ack
    j2 = WorkerJournal(str(j.path), worker_instance_id="w1")  # 重开
    pend = list(j2.pending())
    assert len(pend) == 1
    assert pend[0]["event_id"] == ev.event_id


def test_ack_marks_delivered(tmp_path) -> None:
    j = WorkerJournal(str(_jpath(tmp_path)), worker_instance_id="w1")
    ev = make_transition()
    seq = j.begin_add(); j.commit(seq, ev)
    j.mark_parent_acked(ev.event_id)
    assert list(j.pending()) == []


def test_journal_corruption_blocks_and_preserves_file(tmp_path) -> None:
    jp = _jpath(tmp_path)
    with open(jp, "wb") as f:
        f.write(b"not a real sqlite" * 4)
    with pytest.raises(Exception):
        WorkerJournal(str(jp), worker_instance_id="w1")
    assert jp.exists()  # 保留原件
