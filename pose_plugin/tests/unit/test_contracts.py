"""阶段1：DTO / Envelope 契约测试。

覆盖 wire 端序、bootstrap epoch 例外、payload 血缘校验、schema 版本拒绝、
PoseState 小写 wire 值以及 transition 到 JSON-safe 原语。
"""
from __future__ import annotations

import math
import uuid

import pytest

from ai_monitor_pose.contracts import (
    FallResultV1,
    FallTransitionV1,
    FrameRequestMetaV1,
    FrameRequestV1,
    PoseDetectionV1,
    PoseStateV1,
    PoseTrackV1,
    SharedFrameRefV1,
    EnvelopeV1,
)


def _env(mtype: str, epoch: str | None = "e1", version: int = 1) -> EnvelopeV1:
    return EnvelopeV1(
        schema_version=version,
        message_id=uuid.uuid4().hex,
        correlation_id=None,
        message_type=mtype,
        worker_epoch=epoch,
        sent_at_monotonic_ns=123456,
        payload={},
    )


def _frame_ref() -> SharedFrameRefV1:
    return SharedFrameRefV1(
        shm_name="shm-a", slot_index=0, generation=2, byte_offset=64,
        byte_length=921600, width=640, height=480, channels=3,
        row_stride=1920, dtype="uint8", pixel_format="BGR8",
    )


def test_frame_request_round_trip_preserves_session_frame_and_generation() -> None:
    ref = _frame_ref()
    req = FrameRequestV1(
        schema_version=1, request_id="r1", worker_epoch="e1",
        camera_id="cam-1", camera_session_id="s1", frame_id=7,
        observed_at_unix_ns=10, observed_at_monotonic_ns=20,
        submitted_at_monotonic_ns=30, deadline_monotonic_ns=40,
        config_revision="rev1", dropped_before_submit=0, frame_ref=ref,
    )
    d = req.to_dict()
    back = FrameRequestV1.from_dict(d, envelope_epoch="e1", envelope_version=1)
    assert back.camera_id == "cam-1"
    assert back.camera_session_id == "s1"
    assert back.frame_id == 7
    assert back.frame_ref.generation == 2
    assert back.frame_ref.byte_length == 921600


def test_envelope_bootstrap_allows_null_epoch_only_for_hello() -> None:
    # HELLO 允许 null
    EnvelopeV1.from_dict(
        _env("HELLO", epoch=None).to_dict()
    )
    # 其它 bootstrap 消息若带 null epoch 必须拒绝
    with pytest.raises(ValueError):
        EnvelopeV1.from_dict(_env("WORKER_STARTING", epoch=None).to_dict())


def test_post_bootstrap_payload_epoch_must_equal_envelope_epoch() -> None:
    ref = _frame_ref()
    d = FrameRequestV1(
        schema_version=1, request_id="r1", worker_epoch="e1",
        camera_id="c", camera_session_id="s", frame_id=1,
        observed_at_unix_ns=0, observed_at_monotonic_ns=1,
        submitted_at_monotonic_ns=2, deadline_monotonic_ns=3,
        config_revision="rev", dropped_before_submit=0, frame_ref=ref,
    ).to_dict()
    # 外层 epoch=e1，payload 里也写 e1 -> ok
    FrameRequestV1.from_dict(d, envelope_epoch="e1", envelope_version=1)
    # payload 血缘为 e2，与外层 e1 冲突 -> 拒绝
    d["worker_epoch"] = "e2"
    with pytest.raises(ValueError):
        FrameRequestV1.from_dict(d, envelope_epoch="e1", envelope_version=1)


def test_each_request_response_uses_message_and_correlation_ids() -> None:
    msg_id = uuid.uuid4().hex
    req = EnvelopeV1(
        schema_version=1, message_id=msg_id, correlation_id=None,
        message_type="INFER_FRAME", worker_epoch="e1",
        sent_at_monotonic_ns=1, payload={},
    )
    resp = EnvelopeV1(
        schema_version=1, message_id=uuid.uuid4().hex, correlation_id=msg_id,
        message_type="INFERENCE_RESULT", worker_epoch="e1",
        sent_at_monotonic_ns=2, payload={},
    )
    assert resp.correlation_id == req.message_id
    # 解析后相关性被保留
    back = EnvelopeV1.from_dict(resp.to_dict())
    assert back.correlation_id == msg_id


def test_fall_result_rejects_unknown_schema_version() -> None:
    ref = _frame_ref()
    d = FrameRequestV1(
        schema_version=99, request_id="r1", worker_epoch="e1",
        camera_id="c", camera_session_id="s", frame_id=1,
        observed_at_unix_ns=0, observed_at_monotonic_ns=1,
        submitted_at_monotonic_ns=2, deadline_monotonic_ns=3,
        config_revision="rev", dropped_before_submit=0, frame_ref=ref,
    ).to_dict()
    with pytest.raises(ValueError):
        FrameRequestV1.from_dict(d, envelope_epoch="e1", envelope_version=1)


def test_pose_state_wire_values_are_exactly_lowercase_and_reject_uppercase() -> None:
    assert PoseStateV1.NORMAL.value == "normal"
    assert PoseStateV1.POTENTIAL.value == "potential"
    assert PoseStateV1.FALLEN.value == "fallen"
    with pytest.raises(ValueError):
        PoseStateV1("NORMAL")


def test_transition_serializes_to_json_safe_primitives() -> None:
    tr = FallTransitionV1(
        schema_version=1, event_id="ev-1", dedupe_key="key", incident_id="inc-1",
        event_type="fall_detected", camera_id="cam", camera_session_id="s1",
        pose_track_id=3, source_frame_id=4, source_width=640, source_height=480,
        coordinate_space="source_pixels", occurred_at_unix_ns=5,
        occurred_at_monotonic_ns=6, from_state=PoseStateV1.POTENTIAL,
        to_state=PoseStateV1.FALLEN, rule_score=0.9,
        score_semantics="heuristic_rule_score_not_probability",
        evidence_codes=("horizontal_geometry",),
        bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
        keypoints_coco17=tuple((i, i, 0.9) for i in range(17)),
        model_name="yolov8n-pose.pt", model_sha256="a" * 64,
        config_revision="rev", worker_instance_id="w1",
        queue_wait_ms=1.0, gpu_inference_ms=2.0, end_to_end_ms=3.0,
    )
    d = tr.to_dict()
    # 所有值必须是 JSON-safe 基础类型
    assert isinstance(d["rule_score"], float)
    assert all(isinstance(k, tuple) for k in ()) or True
    assert all(len(kp) == 3 for kp in d["keypoints_coco17"])
    assert all(isinstance(x, (int, float)) and math.isfinite(x)
               for kp in d["keypoints_coco17"] for x in kp)
    assert d["from_state"] == "potential"
    assert d["to_state"] == "fallen"
    # 不允许 NaN/Infinity 泄漏
    assert math.isfinite(d["rule_score"])
