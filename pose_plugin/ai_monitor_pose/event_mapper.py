"""FallTransitionV1 -> AI Monitor VisionEvent（第 5.9 节）。

唯一负责 transition->VisionEvent 的宿主适配模块。payload 在返回前递归校验并转为
JSON-safe 类型；pose track_id 用于 VisionEvent.track_id，payload 标注 pose 命名空间，
与 face track 命名空间绝不混同。本模块可能 import VisionEvent（宿主侧）。
"""
from __future__ import annotations

from .contracts import FallTransitionV1
from .contracts import _json_safe


def map_transition_to_vision_event(transition: FallTransitionV1):
    """把一条已确认的可靠 transition 映射为 AI Monitor 原生 VisionEvent。"""
    from vision.events import VisionEvent

    payload = _json_safe({
        "schema_version": 1,
        "event_id": transition.event_id,
        "dedupe_key": transition.dedupe_key,
        "incident_id": transition.incident_id,
        "track_namespace": "pose",
        "camera_session_id": transition.camera_session_id,
        "source_frame_id": transition.source_frame_id,
        "source_width": transition.source_width,
        "source_height": transition.source_height,
        "coordinate_space": transition.coordinate_space,
        "from_state": transition.from_state.value,
        "to_state": transition.to_state.value,
        "rule_score": transition.rule_score,
        "score_semantics": transition.score_semantics,
        "evidence_codes": list(transition.evidence_codes),
        "bbox_xyxy": list(transition.bbox_xyxy),
        "keypoints_coco17": [list(k) for k in transition.keypoints_coco17],
        "model_name": transition.model_name,
        "model_sha256": transition.model_sha256,
        "config_revision": transition.config_revision,
        "worker_instance_id": transition.worker_instance_id,
        "timing": {
            "queue_wait_ms": transition.queue_wait_ms,
            "gpu_inference_ms": transition.gpu_inference_ms,
            "end_to_end_ms": transition.end_to_end_ms,
        },
    })
    return VisionEvent(
        event_type=transition.event_type,
        camera_id=transition.camera_id,
        track_id=transition.pose_track_id,
        confidence=transition.rule_score,
        timestamp=transition.occurred_at_unix_ns / 1_000_000_000,
        payload=payload,
    )


def map_transition_event_type(transition: FallTransitionV1) -> str:
    return transition.event_type
