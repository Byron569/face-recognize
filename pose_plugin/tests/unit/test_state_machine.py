"""阶段2：时间制摔倒状态机测试。"""
from __future__ import annotations

from ai_monitor_pose.contracts import PoseStateV1
from ai_monitor_pose.worker.state_machine import (
    FallTrackStateMachine,
    run_sequence,
)

from tests.fixtures.pose_sequences import observation


def _ns(t: float) -> int:
    return int(round(t * 1_000_000_000))


def _fall_samples(n: int, step: float = 0.25):
    return [
        observation(t=i * step, frame_id=i + 1, fall=True)
        for i in range(n)
    ]


def test_upright_pose_stays_normal() -> None:
    samples = [observation(t=i * 0.25, frame_id=i, fall=False) for i in range(10)]
    m = FallTrackStateMachine()
    states = [run_sequence([s], machine=m)[0].state for s in samples]
    assert all(s is PoseStateV1.NORMAL for s in states)


def test_one_horizontal_sample_does_not_confirm_fall() -> None:
    m = FallTrackStateMachine()
    out = run_sequence(_fall_samples(1), machine=m)
    assert out[0].state is PoseStateV1.NORMAL
    assert all(not e for e in out[0].events)


def test_continuous_evidence_becomes_fallen_only_after_configured_seconds() -> None:
    # min_fall_pose_duration_s = 3.5，步长 0.25 -> 第 14 帧(t=3.50) 才 FALLEN
    samples = _fall_samples(15)
    m = FallTrackStateMachine()
    out = run_sequence(samples, machine=m)
    assert out[13].state is not PoseStateV1.FALLEN  # t=3.25
    assert out[14].state is PoseStateV1.FALLEN      # t=3.50


def test_detected_transition_is_emitted_exactly_once() -> None:
    samples = _fall_samples(20)
    m = FallTrackStateMachine()
    out = run_sequence(samples, machine=m)
    detected = [
        e for x in out for e in x.events
        if e.event_type == "fall_detected"
    ]
    assert len(detected) == 1


def test_recovered_transition_is_emitted_exactly_once_after_good_upright_evidence() -> None:
    samples = _fall_samples(16) + [
        observation(t=4.0 + i * 0.25, frame_id=20 + i, fall=False)
        for i in range(10)
    ]
    m = FallTrackStateMachine()
    out = run_sequence(samples, machine=m, recovery_duration_s=1.0)
    recovered = [e for x in out for e in x.events if e.event_type == "fall_recovered"]
    assert len(recovered) == 1
    assert out[-1].state is PoseStateV1.NORMAL


def test_potential_to_normal_does_not_emit_recovered() -> None:
    # 短暂疑似（未达 FALLEN）后恢复正常，不产生 recovered
    samples = _fall_samples(2) + [
        observation(t=1.0 + i * 0.25, frame_id=100 + i, fall=False) for i in range(8)
    ]
    m = FallTrackStateMachine()
    out = run_sequence(samples, machine=m)
    recovered = [e for x in out for e in x.events if e.event_type == "fall_recovered"]
    assert recovered == []


def test_duplicate_frame_does_not_advance_duration() -> None:
    samples = [
        observation(t=0.0, frame_id=1, fall=True),
        observation(t=0.0, frame_id=1, fall=True),      # 重复
        *[observation(t=i * 0.25, frame_id=i + 1, fall=True) for i in range(1, 20)],
    ]
    m = FallTrackStateMachine()
    out = run_sequence(samples, machine=m)
    assert len([e for x in out for e in x.events if e.event_type == "fall_detected"]) == 1


def test_out_of_order_frame_does_not_advance_duration() -> None:
    samples = _fall_samples(20)
    # 乱序重放一帧不额外推进
    samples.insert(5, observation(t=0.5, frame_id=3, fall=True))
    m = FallTrackStateMachine()
    out = run_sequence(samples, machine=m)
    assert out[-1].state is PoseStateV1.FALLEN


def test_large_gap_breaks_continuous_evidence() -> None:
    # 0.251s > max_trigger_gap_s=0.25 -> 连续证据中断，不能累积到 FALLEN
    samples = [observation(t=i * 0.25, frame_id=i + 1, fall=True) for i in range(5)]
    samples.append(observation(t=samples[-1].t_sec + 0.251, frame_id=1000, fall=True))
    samples.extend(observation(t=samples[-1].t_sec + (i + 1) * 0.25, frame_id=2000 + i, fall=True) for i in range(20))
    m = FallTrackStateMachine()
    out = run_sequence(samples, machine=m)
    detected = [e for x in out for e in x.events if e.event_type == "fall_detected"]
    assert detected == []


def test_unavailable_does_not_turn_fallen_into_recovered() -> None:
    samples = _fall_samples(16)
    m = FallTrackStateMachine()
    run_sequence(samples, machine=m)
    before = m.state
    assert before is PoseStateV1.FALLEN
    # 长时间无观测不等于恢复
    m.close_out_of_observation(now_t_sec=samples[-1].t_sec + 999.0)
    assert m.state is PoseStateV1.FALLEN


def test_partial_pose_cannot_open_new_incident() -> None:
    samples = [observation(t=i * 0.25, frame_id=i, fall=True, quality="PARTIAL") for i in range(16)]
    m = FallTrackStateMachine()
    out = run_sequence(samples, machine=m)
    incidents = {e.incident_id for x in out for e in x.events}
    assert incidents == set()


def test_same_local_track_id_in_two_sessions_is_isolated() -> None:
    a = FallTrackStateMachine()
    b = FallTrackStateMachine()
    run_sequence(_fall_samples(16), machine=a)
    run_sequence([observation(t=i * 0.25, frame_id=i, fall=False) for i in range(5)], machine=b)
    assert a.state is PoseStateV1.FALLEN
    assert b.state is PoseStateV1.NORMAL


def test_rule_score_is_explicitly_not_probability() -> None:
    out = run_sequence(_fall_samples(16))
    evs = [e for x in out for e in x.events]
    for e in evs:
        assert e.score_semantics == "heuristic_rule_score_not_probability"
