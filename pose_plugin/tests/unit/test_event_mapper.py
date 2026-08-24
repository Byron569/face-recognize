"""阶段7：event_mapper 与 host_protocols（宿主侧适配层）。"""
from __future__ import annotations

import json

import pytest

from ai_monitor_pose.contracts import PoseStateV1
from ai_monitor_pose.event_mapper import map_transition_to_vision_event
from ai_monitor_pose.host_protocols import FrozenHostEventV1
from tests.fixtures.transitions import make_transition


def _tr(etype="fall_detected"):
    return make_transition(camera_id="cam-1", etype=etype, track=7, frame=5)


def test_maps_three_existing_event_names() -> None:
    for name in ("fall_potential", "fall_detected", "fall_recovered"):
        ev = map_transition_to_vision_event(_tr(name))
        assert ev.event_type == name


def test_payload_is_json_safe_and_declares_pose_namespace() -> None:
    ev = map_transition_to_vision_event(_tr())
    json.dumps(ev.payload, allow_nan=False)  # 不抛 = JSON-safe
    assert ev.payload["track_namespace"] == "pose"
    assert ev.payload["score_semantics"] == "heuristic_rule_score_not_probability"
    assert ev.payload["event_id"]


def test_track_id_is_pose_track_and_confidence_is_rule_score() -> None:
    ev = map_transition_to_vision_event(_tr())
    assert ev.track_id == 7
    assert ev.confidence == pytest.approx(ev.payload["rule_score"])


def test_frozen_host_event_deep_copies_canonical_json() -> None:
    tr = _tr()
    ev = map_transition_to_vision_event(tr)
    frozen = FrozenHostEventV1.from_vision_event(ev)
    assert frozen.event_type == tr.event_type
    assert frozen.event_id == tr.event_id
    # canonical bytes 可解析且 payload 与事件一致
    obj = json.loads(frozen.payload_json_utf8.decode("utf-8"))
    assert obj["event_id"] == tr.event_id
    assert obj["track_namespace"] == "pose"
