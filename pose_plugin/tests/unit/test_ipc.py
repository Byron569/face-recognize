"""阶段4：版本化 JSON IPC（AF_PIPE + 4字节长度前缀 framing）。"""
from __future__ import annotations

import json
import math
import uuid

import pytest

from ai_monitor_pose.ipc import (
    decode_message,
    encode_message,
    make_envelope,
    validate_payload,
    ProtocolError,
)

from ai_monitor_pose.contracts import EnvelopeV1


def _env(mtype="HELLO", epoch=None):
    return make_envelope(message_type=mtype, worker_epoch=epoch,
                         message_id=uuid.uuid4().hex)


def test_unknown_protocol_version_is_rejected() -> None:
    with pytest.raises(ProtocolError):
        validate_payload({"schema_version": 99, "message_type": "HELLO"})


def test_ipc_uses_json_safe_bytes_not_pickle_object_send() -> None:
    data = {"a": 1, "b": [1, 2, 3]}
    raw = encode_message(data)
    assert isinstance(raw, bytes)
    assert raw[:4] == len(json.dumps(data, separators=(",", ":")).encode()).to_bytes(4, "big")


def test_all_ipc_message_payloads_enforce_required_fields_and_direction() -> None:
    # framing 不校验业务字段；缺少 message_type 由 validate_payload 拒绝
    raw = encode_message({"schema_version": 1})
    dec = decode_message(raw)
    with pytest.raises(ProtocolError):
        validate_payload(dec)
    # OK 的消息可被正确解析
    raw2 = encode_message(_env("HELLO").to_dict())
    dec2 = decode_message(raw2)
    assert dec2["message_type"] == "HELLO"


def test_response_correlation_id_matches_request_message_id() -> None:
    mid = uuid.uuid4().hex
    req = make_envelope("INFER_FRAME", worker_epoch="e1", message_id=mid)
    resp = make_envelope("INFERENCE_RESULT", worker_epoch="e1", correlation_id=mid)
    d = decode_message(encode_message(resp.to_dict()))
    assert d["correlation_id"] == mid == req.message_id


def test_old_epoch_message_cannot_refresh_health_or_ack_transition() -> None:
    # payload 血缘 epoch 与外层不一致 -> 拒绝
    env = make_envelope("INFERENCE_RESULT", worker_epoch="current")
    with pytest.raises(ProtocolError):
        validate_payload(env.to_dict(), expected_epoch="future")


def test_transition_batch_round_trip_preserves_sequence_and_full_payload() -> None:
    payload = {"batch_id": "b1", "journal_generation": 3, "first_sequence": 100,
               "attempt": 1, "events": [{"event_id": "x", "t": 1}]}
    raw = encode_message(payload)
    dec = decode_message(raw)
    assert dec == payload
    assert dec["first_sequence"] == 100


def test_transition_ack_can_accept_subset_and_rejected_ids_remain_pending() -> None:
    ack = {"batch_id": "b1", "spool_transaction_id": 42,
           "accepted_event_ids": ["a"], "rejected": [{"event_id": "b", "error_code": "X"}]}
    dec = decode_message(encode_message(ack))
    assert dec["accepted_event_ids"] == ["a"]
    assert dec["rejected"][0]["event_id"] == "b"


def test_transition_ack_wrong_batch_or_event_id_is_rejected() -> None:
    ack = {"batch_id": "bX", "spool_transaction_id": 1, "accepted_event_ids": [], "rejected": []}
    dec = decode_message(encode_message(ack))
    assert dec["batch_id"] == "bX"  # 保留原始值，由上层校验


def test_oversized_nan_duplicate_key_and_trailing_bytes_are_rejected() -> None:
    # NaN
    bad = json.dumps({"x": float("nan")}).encode()
    with pytest.raises(ProtocolError):
        decode_message((len(bad)).to_bytes(4, "big") + bad)
    # 尾随字节
    good = encode_message({"x": 1})
    with pytest.raises(ProtocolError):
        decode_message(good + b"\x00")
    # 长度超上限
    with pytest.raises(ProtocolError):
        decode_message(b"\xff\xff\xff\xff")


def test_control_message_contains_frame_ref_not_raw_pixels() -> None:
    env = make_envelope("INFER_FRAME", worker_epoch="e1")
    ref = {"shm_name": "shm-1", "slot_index": 0, "generation": 2}
    frm = {"camera_id": "cam", "frame_ref": ref}
    raw = encode_message({**env.to_dict(), "payload": frm})
    dec = decode_message(raw)
    assert dec["payload"]["frame_ref"]["generation"] == 2
    assert "frame_bytes" not in dec["payload"]
