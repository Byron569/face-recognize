"""时间制摔倒状态机（第 6.7 节）。

长期状态 NORMAL/POTENTIAL/FALLEN；RECOVERED 是转换事件，不是长期状态。
只用单调时间判断持续时间，重复/乱序帧不推进时钟；超过 max_trigger_gap_s 的证据
空洞中断连续性；unavailable/丢轨不触发 recovered。PARTIAL/INVALID 不能开新 incident。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from ..contracts import PoseStateV1
from .features import SCORE_SEMANTICS


@dataclass(frozen=True)
class StateEvent:
    event_type: str
    incident_id: str
    pose_quality: str
    rule_score: float
    score_semantics: str = SCORE_SEMANTICS
    evidence_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepResult:
    state: PoseStateV1
    events: tuple[StateEvent, ...]


def _incident_id() -> str:
    return uuid.uuid4().hex


class FallTrackStateMachine:
    """每个 (camera, session, pose_track_id) 一个实例，状态完全隔离。"""

    def __init__(self) -> None:
        self.state: PoseStateV1 = PoseStateV1.NORMAL
        self._last_frame_id: int | None = None
        self._last_obs_ns: int | None = None
        self.incident_id: str | None = None
        self._emitted_potential = False
        self._emitted_detected = False
        # 连续 GOOD 跌倒证据爆发
        self._burst_count = 0
        self._burst_start_ns: int | None = None
        self._fall_start_ns: int | None = None
        self._recovery_start_ns: int | None = None

    def observe(self, sample, *, max_trigger_gap_s: float, min_fall_pose_duration_s: float,
                recovery_duration_s: float) -> tuple[PoseStateV1, list[StateEvent]]:
        now_ns = (
            int(round(sample.t_sec * 1_000_000_000))
            if hasattr(sample, "t_sec")
            else sample.observed_at_monotonic_ns
        )
        frame_id = sample.frame_id
        events: list[StateEvent] = []

        # 乱序/重复帧不推进
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            return self.state, events
        if self._last_obs_ns is not None and now_ns < self._last_obs_ns:
            return self.state, events

        quality = getattr(sample, "pose_quality", "GOOD")
        evidence_active = bool(getattr(sample, "fall_evidence", False))
        gap = (
            (now_ns - self._last_obs_ns) / 1e9 if self._last_obs_ns is not None else 0.0
        )
        gap_ok = self._last_obs_ns is None or gap <= max_trigger_gap_s

        ok_evidence = evidence_active and quality == "GOOD" and gap_ok
        if ok_evidence:
            if self._burst_count == 0:
                self._burst_start_ns = now_ns
            self._burst_count += 1
        else:
            self._burst_count = 0
            self._burst_start_ns = None

        can_open = self._burst_count >= 2

        if self.state is PoseStateV1.NORMAL:
            if can_open:
                self.incident_id = _incident_id()
                self._fall_start_ns = self._burst_start_ns
                self._emitted_potential = True
                self._recovery_start_ns = None
                self.state = PoseStateV1.POTENTIAL
                events.append(StateEvent(
                    event_type="fall_potential", incident_id=self.incident_id,
                    pose_quality=quality, rule_score=0.5,
                ))

        elif self.state is PoseStateV1.POTENTIAL:
            if ok_evidence and can_open:
                if self._fall_start_ns is None:
                    self._fall_start_ns = self._burst_start_ns
                sustained = (now_ns - self._fall_start_ns) / 1e9 >= min_fall_pose_duration_s
                if sustained and not self._emitted_detected:
                    self.state = PoseStateV1.FALLEN
                    self._emitted_detected = True
                    self.sustained_at_ns = now_ns
                    self._recovery_start_ns = now_ns
                    events.append(StateEvent(
                        event_type="fall_detected", incident_id=self.incident_id,
                        pose_quality=quality, rule_score=0.9,
                    ))
            else:
                # 证据中断或有空洞 -> 取消疑似，不产生 recovered
                self.state = PoseStateV1.NORMAL
                self._fall_start_ns = None
                self._emitted_potential = False
                self.incident_id = None

        elif self.state is PoseStateV1.FALLEN:
            if not evidence_active:
                if self._recovery_start_ns is None:
                    self._recovery_start_ns = now_ns
                if (now_ns - self._recovery_start_ns) / 1e9 >= recovery_duration_s:
                    self.state = PoseStateV1.NORMAL
                    events.append(StateEvent(
                        event_type="fall_recovered", incident_id=self.incident_id,
                        pose_quality=quality, rule_score=0.1,
                    ))
                    self.incident_id = None
                    self._emitted_potential = False
                    self._emitted_detected = False
                    self._fall_start_ns = None
                    self._recovery_start_ns = None
            else:
                self._recovery_start_ns = None

        self._last_frame_id = frame_id
        self._last_obs_ns = now_ns
        return self.state, events

    def close_out_of_observation(self, now_t_sec: float) -> None:
        """长时间无观测。绝不把 FALLEN 当作 recovered。"""
        return None


def run_sequence(samples, machine: FallTrackStateMachine | None = None, *,
                 max_trigger_gap_s: float = 0.25,
                 min_fall_pose_duration_s: float = 3.50,
                 recovery_duration_s: float = 1.0) -> list[StepResult]:
    if machine is None:
        machine = FallTrackStateMachine()
    out: list[StepResult] = []
    for s in samples:
        state, events = machine.observe(
            s, max_trigger_gap_s=max_trigger_gap_s,
            min_fall_pose_duration_s=min_fall_pose_duration_s,
            recovery_duration_s=recovery_duration_s,
        )
        out.append(StepResult(state=state, events=tuple(events)))
    return out
