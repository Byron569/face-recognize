"""宿主适配层协议（第 5.7 节）。

只供 AI Monitor client 进程使用；对 VisionEvent 在 TYPE_CHECKING 下导入并使用字符串注解，
Worker import graph 禁止触达本模块。
"""
from __future__ import annotations

import concurrent.futures
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vision.events import VisionEvent

from .contracts import _json_safe


@dataclass(frozen=True, slots=True)
class FrozenHostEventV1:
    event_id: str
    event_type: str
    camera_id: str
    track_id: int | None
    confidence: float | None
    timestamp: float
    payload_json_utf8: bytes

    @classmethod
    def from_vision_event(cls, event: "VisionEvent") -> "FrozenHostEventV1":
        payload = _json_safe(event.payload)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
        return cls(
            event_id=str(payload.get("event_id") or ""),
            event_type=event.event_type,
            camera_id=event.camera_id,
            track_id=event.track_id,
            confidence=event.confidence,
            timestamp=float(event.timestamp),
            payload_json_utf8=canonical,
        )


@dataclass(frozen=True, slots=True)
class EventSinkAck:
    event_id: str
    persisted: bool
    database_event_id: int | None = None


class EventSinkProtocol(Protocol):
    def submit(self, event: FrozenHostEventV1) -> concurrent.futures.Future["EventSinkAck"]: ...


class FallRuntimeHandle(Protocol):
    def has_latest_result_or_health_change(self, camera_id: str, camera_session_id: str) -> bool: ...
    def has_unseen_compatibility_event(self, camera_id: str, camera_session_id: str) -> bool: ...
    def offer_frame(self, frame, request_meta) -> object: ...
    def poll(self, camera_id: str, camera_session_id: str): ...
    def unregister_camera(self, camera_id: str, camera_session_id: str) -> None: ...
    def release(self) -> None: ...


class RuntimeFactoryProtocol(Protocol):
    def acquire(self, runtime_key: str, config, event_sink) -> FallRuntimeHandle: ...


class ClockProtocol(Protocol):
    def monotonic_ns(self) -> int: ...
    def unix_ns(self) -> int: ...
