
from ai_monitor_pose.contracts import FallTransitionV1, PoseStateV1


def make_transition(camera_id="cam-1", session="s1", track=1, etype="fall_detected",
                    frame=5, incident="in-1") -> FallTransitionV1:
    return FallTransitionV1(
        schema_version=1, event_id="ev-" + etype + "-" + camera_id,
        dedupe_key="key-" + etype, incident_id=incident, event_type=etype,
        camera_id=camera_id, camera_session_id=session, pose_track_id=track,
        source_frame_id=frame, source_width=640, source_height=480,
        coordinate_space="source_pixels", occurred_at_unix_ns=1,
        occurred_at_monotonic_ns=2, from_state=PoseStateV1.NORMAL,
        to_state=PoseStateV1.FALLEN, rule_score=0.9,
        score_semantics="heuristic_rule_score_not_probability",
        evidence_codes=(), bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
        keypoints_coco17=tuple((0.0, 0.0, 0.5) for _ in range(17)),
        model_name="m", model_sha256="m", config_revision="r",
        worker_instance_id="w0", queue_wait_ms=0.0, gpu_inference_ms=0.0,
        end_to_end_ms=0.0,
    )
