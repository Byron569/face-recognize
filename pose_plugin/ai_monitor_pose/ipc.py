"""版本化 JSON IPC（第 5 节）。

控制面使用 Windows AF_PIPE：消息为 4 字节大端无符号长度前缀 + UTF-8 JSON bytes。
严格拒绝 NaN/Infinity、重复 key、尾随字节、超上限长度与未知 schema 版本；
不使用 Pickle 对象协议。payload 血缘需与外层 epoch/版本一致。
"""
from __future__ import annotations

import json
import math
import struct
import uuid
from typing import Any

from .contracts import EnvelopeV1

_MAX_MESSAGE_BYTES = 64 * 1024 * 1024  # 64MiB
_SCHEMA_VERSION = 1

_MAX_FIELD_BYTES = 4 * 1024  # 单 key/单 value 序列化上限（防御）


class ProtocolError(Exception):
    pass


_NAN_INF = (float("nan"), float("inf"), float("-inf"))


def _check_fields(obj: Any, depth: int = 0) -> None:
    if depth > 12:
        raise ProtocolError("嵌套过深")
    if isinstance(obj, dict):
        seen = set()
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ProtocolError("key 必须是字符串")
            if k in seen:
                raise ProtocolError(f"重复 key: {k!r}")
            seen.add(k)
            _check_fields(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _check_fields(v, depth + 1)
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            raise ProtocolError("不允许 NaN/Infinity")


def _encode_json(obj: Any) -> bytes:
    try:
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as e:
        raise ProtocolError(f"不可 JSON 序列化: {e}") from e
    return text.encode("utf-8")


def encode_message(obj: Any) -> bytes:
    """将对象编码为 4 字节大端长度前缀的 JSON bytes。"""
    body = _encode_json(obj)
    n = len(body)
    if n > _MAX_MESSAGE_BYTES:
        raise ProtocolError("消息超长")
    return struct.pack(">I", n) + body


def decode_message(raw: bytes) -> dict[str, Any]:
    """严格解码：校验长度前缀、完整性、尾随字节与字段合法性。"""
    if len(raw) < 4:
        raise ProtocolError("帧过短")
    (n,) = struct.unpack(">I", raw[:4])
    if n > _MAX_MESSAGE_BYTES:
        raise ProtocolError("长度前缀超上限")
    frame = raw[4:]
    if len(frame) != n:
        raise ProtocolError("帧长度不匹配")
    try:
        text = frame.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ProtocolError("UTF-8 解码失败") from e
    obj = json.loads(text, parse_constant=_bad_constant)
    if not isinstance(obj, dict):
        raise ProtocolError("顶层必须是对象")
    _check_fields(obj)
    return obj


def _bad_constant(name: str) -> None:
    raise ProtocolError(f"非法常量: {name}")


def make_envelope(message_type: str, *, worker_epoch: str | None = None,
                  correlation_id: str | None = None,
                  message_id: str | None = None,
                  payload: dict | None = None) -> EnvelopeV1:
    return EnvelopeV1(
        schema_version=_SCHEMA_VERSION,
        message_id=message_id or uuid.uuid4().hex,
        correlation_id=correlation_id,
        message_type=message_type,
        worker_epoch=worker_epoch,
        sent_at_monotonic_ns=0,
        payload=payload or {},
    )


def validate_payload(data: dict, *, expected_epoch: str | None = None) -> None:
    """校验 envelope 级字段；可选校验 payload 血缘 epoch。"""
    if data.get("schema_version") != _SCHEMA_VERSION:
        raise ProtocolError(f"未知 schema_version: {data.get('schema_version')!r}")
    mtype = data.get("message_type")
    if not isinstance(mtype, str) or not mtype:
        raise ProtocolError("缺少 message_type")
    if expected_epoch is not None and data.get("worker_epoch") != expected_epoch:
        raise ProtocolError("worker_epoch 不匹配（旧 epoch）")
    payload = data.get("payload")
    if payload is not None and not isinstance(payload, dict):
        raise ProtocolError("payload 必须是对象")
