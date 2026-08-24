"""版本化 DTO 与统一 Envelope（第 5 节）。

隐含约束：本文模块只可被独立 Worker 安全导入；不得 import Torch/Ultralytics/FastAPI/DB。
所有跨进程消息都是 JSON-safe 基础类型，禁止传 NumPy/Tensor/Results 等对象。
"""
from __future__ import annotations

import math
import typing
import uuid
from dataclasses import dataclass, field
from enum import Enum

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class PoseStateV1(str, Enum):
    """Wire 值固定为小写。JSON / WS / TS 只允许 lowercase value。"""

    NORMAL = "normal"
    POTENTIAL = "potential"
    FALLEN = "fallen"


FALL_EVENT_NAMESPACE_UUID = uuid.UUID("6f2a3c4d-9b8a-4f1e-b3c2-1a2b3c4d5e6f")


def _json_safe(value: object) -> JsonValue:
    """递归校验并转换任意值为 JSON-safe 基础类型；NaN/Infinity 一律拒绝。"""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("NaN/Infinity not allowed in JSON payload")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    raise TypeError(f"not a JSON-safe value: {type(value)!r}")


@dataclass(frozen=True, slots=True)
class EnvelopeV1:
    schema_version: int
    message_id: str
    correlation_id: str | None
    message_type: str
    worker_epoch: str | None
    sent_at_monotonic_ns: int
    payload: dict[str, JsonValue]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "message_type": self.message_type,
            "worker_epoch": self.worker_epoch,
            "sent_at_monotonic_ns": self.sent_at_monotonic_ns,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EnvelopeV1":
        if d.get("schema_version") != 1:
            raise ValueError(f"未知 schema_version: {d.get('schema_version')!r}")
        mtype = d.get("message_type")
        epoch = d.get("worker_epoch")
        # bootstrap 例外：只有 HELLO 允许 null epoch
        if epoch is None and mtype != "HELLO":
            raise ValueError(f"非 HELLO 消息不允许 null worker_epoch: {mtype!r}")
        mid = d.get("message_id")
        if not isinstance(mid, str) or not mid:
            raise ValueError("message_id 缺失")
        payload = d.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("payload 必须是对象")
        return cls(
            schema_version=1,
            message_id=mid,
            correlation_id=d.get("correlation_id"),
            message_type=mtype,
            worker_epoch=epoch,
            sent_at_monotonic_ns=int(d.get("sent_at_monotonic_ns", 0)),
            payload=payload,
        )


def _require_epoch_and_version(d: dict, field_: str, envelope_epoch: str | None, envelope_version: int) -> None:
    if envelope_version != 1:
        raise ValueError(f"未知 envelope schema_version: {envelope_version!r}")
    payload_version = d.get("schema_version")
    if payload_version != 1:
        raise ValueError(f"未知 payload schema_version: {payload_version!r}")
    payload_epoch = d.get(field_)
    if payload_epoch != envelope_epoch:
        raise ValueError(f"payload 血缘 {field_} 与 Envelope epoch 不一致")


@dataclass(frozen=True, slots=True)
class SharedFrameRefV1:
    shm_name: str
    slot_index: int
    generation: int
    byte_offset: int
    byte_length: int
    width: int
    height: int
    channels: int
    row_stride: int
    dtype: str
    pixel_format: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "shm_name": self.shm_name, "slot_index": self.slot_index, "generation": self.generation,
            "byte_offset": self.byte_offset, "byte_length": self.byte_length,
            "width": self.width, "height": self.height, "channels": self.channels,
            "row_stride": self.row_stride, "dtype": self.dtype, "pixel_format": self.pixel_format,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SharedFrameRefV1":
        return cls(**{k: d[k] for k in ("shm_name", "slot_index", "generation", "byte_offset",
                                        "byte_length", "width", "height", "channels",
                                        "row_stride", "dtype", "pixel_format")})


@dataclass(frozen=True, slots=True)
class FrameRequestMetaV1:
    schema_version: int
    request_id: str
    camera_id: str
    camera_session_id: str
    frame_id: int
    observed_at_unix_ns: int
    observed_at_monotonic_ns: int
    deadline_monotonic_ns: int
    config_revision: str


@dataclass(frozen=True, slots=True)
class FrameRequestV1:
    schema_version: int
    request_id: str
    worker_epoch: str
    camera_id: str
    camera_session_id: str
    frame_id: int
    observed_at_unix_ns: int
    observed_at_monotonic_ns: int
    submitted_at_monotonic_ns: int
    deadline_monotonic_ns: int
    config_revision: str
    dropped_before_submit: int
    frame_ref: SharedFrameRefV1

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version, "request_id": self.request_id,
            "worker_epoch": self.worker_epoch, "camera_id": self.camera_id,
            "camera_session_id": self.camera_session_id, "frame_id": self.frame_id,
            "observed_at_unix_ns": self.observed_at_unix_ns,
            "observed_at_monotonic_ns": self.observed_at_monotonic_ns,
            "submitted_at_monotonic_ns": self.submitted_at_monotonic_ns,
            "deadline_monotonic_ns": self.deadline_monotonic_ns,
            "config_revision": self.config_revision,
            "dropped_before_submit": self.dropped_before_submit,
            "frame_ref": self.frame_ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict, *, envelope_epoch: str | None, envelope_version: int) -> "FrameRequestV1":
        _require_epoch_and_version(d, "worker_epoch", envelope_epoch, envelope_version)
        fr = d.get("frame_ref") or {}
        return cls(
            schema_version=1, request_id=d["request_id"], worker_epoch=d["worker_epoch"],
            camera_id=d["camera_id"], camera_session_id=d["camera_session_id"],
            frame_id=int(d["frame_id"]),
            observed_at_unix_ns=int(d["observed_at_unix_ns"]),
            observed_at_monotonic_ns=int(d["observed_at_monotonic_ns"]),
            submitted_at_monotonic_ns=int(d["submitted_at_monotonic_ns"]),
            deadline_monotonic_ns=int(d["deadline_monotonic_ns"]),
            config_revision=d["config_revision"],
            dropped_before_submit=int(d.get("dropped_before_submit", 0)),
            frame_ref=SharedFrameRefV1.from_dict(fr),
        )


@dataclass(frozen=True, slots=True)
class PoseDetectionV1:
    bbox_xyxy: tuple[float, float, float, float]
    detection_score: float
    keypoints_coco17: tuple[tuple[float, float, float], ...]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "detection_score": self.detection_score,
            "keypoints_coco17": [list(k) for k in self.keypoints_coco17],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PoseDetectionV1":
        return cls(
            bbox_xyxy=tuple(float(x) for x in d["bbox_xyxy"]),
            detection_score=float(d["detection_score"]),
            keypoints_coco17=tuple(tuple(float(x) for x in k) for k in d["keypoints_coco17"]),
        )


@dataclass(frozen=True, slots=True)
class PoseTrackV1:
    pose_track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    detection_score: float
    keypoints_coco17: tuple[tuple[float, float, float], ...]
    pose_quality: str
    state: PoseStateV1
    state_since_monotonic_ns: int
    rule_score: float
    score_semantics: str
    evidence_codes: tuple[str, ...]
    is_ghost: bool

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "pose_track_id": self.pose_track_id, "bbox_xyxy": list(self.bbox_xyxy),
            "detection_score": self.detection_score,
            "keypoints_coco17": [list(k) for k in self.keypoints_coco17],
            "pose_quality": self.pose_quality, "state": self.state.value,
            "state_since_monotonic_ns": self.state_since_monotonic_ns,
            "rule_score": self.rule_score, "score_semantics": self.score_semantics,
            "evidence_codes": list(self.evidence_codes), "is_ghost": self.is_ghost,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PoseTrackV1":
        return cls(
            pose_track_id=int(d["pose_track_id"]),
            bbox_xyxy=tuple(float(x) for x in d["bbox_xyxy"]),
            detection_score=float(d["detection_score"]),
            keypoints_coco17=tuple(tuple(float(x) for x in k) for k in d["keypoints_coco17"]),
            pose_quality=d["pose_quality"], state=PoseStateV1(d["state"]),
            state_since_monotonic_ns=int(d["state_since_monotonic_ns"]),
            rule_score=float(d["rule_score"]), score_semantics=d["score_semantics"],
            evidence_codes=tuple(d["evidence_codes"]), is_ghost=bool(d["is_ghost"]),
        )


@dataclass(frozen=True, slots=True)
class FallResultV1:
    schema_version: int
    request_id: str
    worker_epoch: str
    worker_instance_id: str
    camera_id: str
    camera_session_id: str
    source_frame_id: int
    source_width: int
    source_height: int
    coordinate_space: str
    observed_at_unix_ns: int
    observed_at_monotonic_ns: int
    completed_at_monotonic_ns: int
    status: str
    config_revision: str
    model_name: str
    model_sha256: str
    device: str
    precision: str
    queue_wait_ms: float
    frame_copy_ms: float
    gpu_inference_ms: float
    postprocess_ms: float
    end_to_end_ms: float
    tracks: tuple[PoseTrackV1, ...]
    transition_event_ids: tuple[str, ...]
    error_code: str | None
    error_message: str | None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version, "request_id": self.request_id,
            "worker_epoch": self.worker_epoch, "worker_instance_id": self.worker_instance_id,
            "camera_id": self.camera_id, "camera_session_id": self.camera_session_id,
            "source_frame_id": self.source_frame_id, "source_width": self.source_width,
            "source_height": self.source_height, "coordinate_space": self.coordinate_space,
            "observed_at_unix_ns": self.observed_at_unix_ns,
            "observed_at_monotonic_ns": self.observed_at_monotonic_ns,
            "completed_at_monotonic_ns": self.completed_at_monotonic_ns,
            "status": self.status, "config_revision": self.config_revision,
            "model_name": self.model_name, "model_sha256": self.model_sha256,
            "device": self.device, "precision": self.precision,
            "queue_wait_ms": self.queue_wait_ms, "frame_copy_ms": self.frame_copy_ms,
            "gpu_inference_ms": self.gpu_inference_ms, "postprocess_ms": self.postprocess_ms,
            "end_to_end_ms": self.end_to_end_ms,
            "tracks": [t.to_dict() for t in self.tracks],
            "transition_event_ids": list(self.transition_event_ids),
            "error_code": self.error_code, "error_message": self.error_message,
        }


@dataclass(frozen=True, slots=True)
class FallTransitionV1:
    schema_version: int
    event_id: str
    dedupe_key: str
    incident_id: str
    event_type: str
    camera_id: str
    camera_session_id: str
    pose_track_id: int
    source_frame_id: int
    source_width: int
    source_height: int
    coordinate_space: str
    occurred_at_unix_ns: int
    occurred_at_monotonic_ns: int
    from_state: PoseStateV1
    to_state: PoseStateV1
    rule_score: float
    score_semantics: str
    evidence_codes: tuple[str, ...]
    bbox_xyxy: tuple[float, float, float, float]
    keypoints_coco17: tuple[tuple[float, float, float], ...]
    model_name: str
    model_sha256: str
    config_revision: str
    worker_instance_id: str
    queue_wait_ms: float
    gpu_inference_ms: float
    end_to_end_ms: float

    def to_dict(self) -> dict[str, JsonValue]:
        return _json_safe({
            "schema_version": self.schema_version, "event_id": self.event_id,
            "dedupe_key": self.dedupe_key, "incident_id": self.incident_id,
            "event_type": self.event_type, "camera_id": self.camera_id,
            "camera_session_id": self.camera_session_id, "pose_track_id": self.pose_track_id,
            "source_frame_id": self.source_frame_id, "source_width": self.source_width,
            "source_height": self.source_height, "coordinate_space": self.coordinate_space,
            "occurred_at_unix_ns": self.occurred_at_unix_ns,
            "occurred_at_monotonic_ns": self.occurred_at_monotonic_ns,
            "from_state": self.from_state.value, "to_state": self.to_state.value,
            "rule_score": self.rule_score, "score_semantics": self.score_semantics,
            "evidence_codes": list(self.evidence_codes), "bbox_xyxy": list(self.bbox_xyxy),
            "keypoints_coco17": [list(k) for k in self.keypoints_coco17],
            "model_name": self.model_name, "model_sha256": self.model_sha256,
            "config_revision": self.config_revision, "worker_instance_id": self.worker_instance_id,
            "queue_wait_ms": self.queue_wait_ms, "gpu_inference_ms": self.gpu_inference_ms,
            "end_to_end_ms": self.end_to_end_ms,
        })

    @classmethod
    def from_dict(cls, d: dict) -> "FallTransitionV1":
        return cls(
            schema_version=int(d["schema_version"]), event_id=d["event_id"],
            dedupe_key=d["dedupe_key"], incident_id=d["incident_id"],
            event_type=d["event_type"], camera_id=d["camera_id"],
            camera_session_id=d["camera_session_id"], pose_track_id=int(d["pose_track_id"]),
            source_frame_id=int(d["source_frame_id"]), source_width=int(d["source_width"]),
            source_height=int(d["source_height"]), coordinate_space=d["coordinate_space"],
            occurred_at_unix_ns=int(d["occurred_at_unix_ns"]),
            occurred_at_monotonic_ns=int(d["occurred_at_monotonic_ns"]),
            from_state=PoseStateV1(d["from_state"]), to_state=PoseStateV1(d["to_state"]),
            rule_score=float(d["rule_score"]), score_semantics=d["score_semantics"],
            evidence_codes=tuple(d["evidence_codes"]),
            bbox_xyxy=tuple(float(x) for x in d["bbox_xyxy"]),
            keypoints_coco17=tuple(tuple(float(x) for x in k) for k in d["keypoints_coco17"]),
            model_name=d["model_name"], model_sha256=d["model_sha256"],
            config_revision=d["config_revision"], worker_instance_id=d["worker_instance_id"],
            queue_wait_ms=float(d["queue_wait_ms"]), gpu_inference_ms=float(d["gpu_inference_ms"]),
            end_to_end_ms=float(d["end_to_end_ms"]),
        )
