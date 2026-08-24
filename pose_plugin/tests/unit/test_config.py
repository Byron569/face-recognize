"""阶段1：强类型配置契约测试（RED → GREEN）。

仅在构造期验证：路径绝对性、设备/精度/下载开关、规则阈值合法性、config_revision 敏感性，
以及 cross_camera 分组约束。模型文件真实性与 SHA-256 真值校验属于阶段5 Worker bootstrap。
"""
from __future__ import annotations

import pytest

from ai_monitor_pose.config import FallTaskConfig

PLUGIN = "D:/ai-monitor-1.1.0/pose_plugin"
ABS = f"{PLUGIN}/models/yolov8n-pose.pt"
SHA = f"{PLUGIN}/models/yolov8n-pose.pt.sha256"
CFG_MANIFEST = f"{PLUGIN}/models/capacity-cuda0.json"


def _valid_mapping() -> dict:
    return {
        "enabled": False,
        "mode": "shadow",
        "class_path": "ai_monitor_pose.task.FallDetectionTask",
        "interval": 1,
        "runtime_key": "pose-cuda-0",
        "worker": {
            "python": ABS,
            "module": "ai_monitor_pose.worker",
            "pipe_prefix": "ai-monitor-fall-v1",
            "startup_timeout_s": 45,
            "heartbeat_interval_s": 1,
            "heartbeat_timeout_s": 3,
            "inference_timeout_s": 5,
            "graceful_shutdown_timeout_s": 5,
            "zero_lease_worker_drain_timeout_s": 5,
            "idle_retention_s": 60,
        },
        "gpu": {
            "required": True,
            "device": "cuda:0",
            "expected_device_name_regex": "(?i)RTX 4060",
            "precision": "fp16",
            "allow_cpu_fallback": False,
            "max_gpu_memory_mb": 3072,
            "max_vram_fraction": 0.80,
            "minimum_free_vram_mb": 1024,
        },
        "model": {
            "path": ABS,
            "sha256_file": SHA,
            "allow_download": False,
            "imgsz": 640,
            "confidence": 0.35,
            "iou": 0.45,
            "max_detections": 50,
        },
        "runtime": {
            "max_cameras": 4,
            "requested_max_total_fps": 32,
            "capacity_manifest_path": CFG_MANIFEST,
            "capacity_manifest_sha256": None,
            "capacity_headroom_ratio": 0.75,
            "frame_slots_per_camera": 2,
            "max_frame_width": 1920,
            "max_frame_height": 1080,
            "max_result_age_ms": 1500,
            "overlay_ttl_ms": 1200,
            "transition_queue_capacity_per_camera": 64,
            "transition_queue_resume_ratio": 0.50,
            "worker_journal_path": f"{PLUGIN}/var/worker-transition-journal.sqlite3",
            "worker_journal_pending_capacity": 10000,
            "event_spool_path": f"{PLUGIN}/var/fall-event-spool.sqlite3",
            "event_spool_pending_capacity": 10000,
            "delivered_retention_hours": 24,
            "delivered_retention_rows": 1000,
        },
        "delivery": {
            "ingress_queue_capacity": 1024,
            "outbox_pending_capacity": 10000,
            "outbox_resume_ratio": 0.50,
            "outbox_delivered_retention_hours": 24,
            "outbox_delivered_retention_rows": 10000,
        },
        "scheduler": {
            "target_fps": 8,
            "latest_only": True,
            "batch_size": 1,
            "fair_policy": "service_debt_round_robin",
            "not_ready_retry_ms": 500,
        },
        "tracker": {
            "high_confidence": 0.35,
            "low_confidence": 0.15,
            "match_iou_threshold": 0.50,
            "max_normalized_keypoint_distance": 0.35,
            "min_confirm_duration_s": 0.25,
            "lost_timeout_s": 1.5,
            "ghost_timeout_s": 3.0,
            "fallen_ghost_timeout_s": 5.0,
        },
        "algorithm": {
            "required_keypoint_indices": [0, 5, 6, 11, 12, 13, 14],
            "minimum_visible_required": 5,
            "keypoint_min_confidence": 0.30,
            "minimum_body_height_ratio": 0.12,
            "bbox_height_width_fall_max": 0.75,
            "torso_inclination_from_vertical_min_deg": 55.0,
            "hip_angle_fall_min_deg": 135.0,
            "head_descent_body_heights_min": 0.18,
            "rotation_energy_min_rad_s": 1.80,
            "gravity_factor_min_body_heights_s2": 1.50,
            "fast_rotation_energy_min_rad_s": 3.00,
            "fast_gravity_factor_min_body_heights_s2": 2.50,
            "upright_height_width_min": 1.25,
            "upright_torso_inclination_max_deg": 25.0,
            "rebound_body_heights_min": 0.12,
            "ema_tau_s": 0.20,
            "history_duration_s": 4.5,
            "trigger_window_s": 1.25,
            "trigger_ratio": 0.50,
            "min_trigger_duration_s": 0.50,
            "max_trigger_gap_s": 0.25,
            "min_fall_pose_duration_s": 3.50,
            "recovery_duration_s": 1.00,
            "rebound_duration_s": 0.50,
            "tracker_reset_gap_s": 1.50,
            "allow_already_down_detection": False,
        },
        "cross_camera": {
            "enabled": False,
            "camera_group_id": None,
            "view_id": None,
            "max_timestamp_skew_ms": 200,
        },
    }


def _set(root: dict, keypath: list[str], value) -> None:
    node = root
    for k in keypath[:-1]:
        node = node[k]
    node[keypath[-1]] = value


def test_config_rejects_cpu_device() -> None:
    m = _valid_mapping()
    _set(m, ["gpu", "device"], "cpu")
    with pytest.raises(ValueError):
        FallTaskConfig.from_mapping(m)


def test_config_rejects_auto_device() -> None:
    m = _valid_mapping()
    _set(m, ["gpu", "device"], "auto")
    with pytest.raises(ValueError):
        FallTaskConfig.from_mapping(m)


def test_config_rejects_allow_cpu_fallback_true() -> None:
    m = _valid_mapping()
    _set(m, ["gpu", "allow_cpu_fallback"], True)
    with pytest.raises(ValueError):
        FallTaskConfig.from_mapping(m)


def test_config_rejects_relative_model_path() -> None:
    m = _valid_mapping()
    _set(m, ["model", "path"], "models/yolov8n-pose.pt")
    with pytest.raises(ValueError):
        FallTaskConfig.from_mapping(m)


def test_task_config_accepts_absolute_model_path_without_touching_filesystem() -> None:
    cfg = FallTaskConfig.from_mapping(_valid_mapping())
    assert cfg.model.path == ABS
    # 构造期不允许触达文件系统或访问模型。
    assert cfg.model.path  # 仅路径字段，不做真实 IO


def test_task_config_rejects_relative_model_and_sha256_paths() -> None:
    for key in ("path", "sha256_file"):
        m = _valid_mapping()
        _set(m, ["model", key], "a/relative.pt")
        with pytest.raises(ValueError):
            FallTaskConfig.from_mapping(m)


def test_disabled_config_allows_null_capacity_digest_but_enabled_rejects_it() -> None:
    cfg = FallTaskConfig.from_mapping(_valid_mapping())  # enabled=false, digest=None -> ok
    assert cfg.runtime.capacity_manifest_sha256 is None

    m = _valid_mapping()
    m["enabled"] = True
    with pytest.raises(ValueError):
        FallTaskConfig.from_mapping(m)


def test_capacity_manifest_path_must_be_absolute_and_digest_lowercase_sha256() -> None:
    m = _valid_mapping()
    _set(m, ["runtime", "capacity_manifest_path"], "relative/capacity.json")
    with pytest.raises(ValueError):
        FallTaskConfig.from_mapping(m)

    m2 = _valid_mapping()
    m2["enabled"] = True
    _set(m2, ["runtime", "capacity_manifest_sha256"], "ABCDEF0123456789" * 4)  # 大写 -> 拒绝
    with pytest.raises(ValueError):
        FallTaskConfig.from_mapping(m2)

    m3 = _valid_mapping()
    m3["enabled"] = True
    _set(m3, ["runtime", "capacity_manifest_sha256"], "a" * 63)  # 长度不足
    with pytest.raises(ValueError):
        FallTaskConfig.from_mapping(m3)


def test_capacity_digest_change_changes_config_revision_without_file_io() -> None:
    a = FallTaskConfig.from_mapping(_valid_mapping())
    b = _valid_mapping()
    b["enabled"] = True
    _set(b, ["runtime", "capacity_manifest_sha256"], "b" * 64)
    b_cfg = FallTaskConfig.from_mapping(b)
    assert a.config_revision != b_cfg.config_revision


def test_algorithm_rule_thresholds_are_complete_finite_and_in_valid_ranges() -> None:
    import math
    from dataclasses import fields as dc_fields
    cfg = FallTaskConfig.from_mapping(_valid_mapping())
    for name, value in ((f.name, getattr(cfg.algorithm, f.name)) for f in dc_fields(cfg.algorithm)):
        if name.startswith("_") or value is None:
            continue
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        assert math.isfinite(value), name
    assert 0.0 < cfg.algorithm.bbox_height_width_fall_max < 1.0
    assert 0.0 < cfg.algorithm.trigger_ratio < 1.0
    # 时间门槛必须为正
    assert cfg.algorithm.min_trigger_duration_s > 0
    assert cfg.algorithm.min_fall_pose_duration_s > 0
    assert cfg.algorithm.recovery_duration_s > 0


def test_algorithm_threshold_change_changes_config_revision() -> None:
    a = FallTaskConfig.from_mapping(_valid_mapping())
    m = _valid_mapping()
    _set(m, ["algorithm", "rotation_energy_min_rad_s"], 2.0)
    b = FallTaskConfig.from_mapping(m)
    assert a.config_revision != b.config_revision


def test_cross_camera_group_and_view_are_required_only_when_enabled() -> None:
    m = _valid_mapping()
    m["cross_camera"]["enabled"] = True
    m["cross_camera"]["camera_group_id"] = None
    m["cross_camera"]["view_id"] = None
    with pytest.raises(ValueError):
        FallTaskConfig.from_mapping(m)

    m2 = _valid_mapping()
    m2["cross_camera"]["enabled"] = True
    m2["cross_camera"]["camera_group_id"] = "g1"
    m2["cross_camera"]["view_id"] = None
    with pytest.raises(ValueError):
        FallTaskConfig.from_mapping(m2)

    m3 = _valid_mapping()
    m3["cross_camera"]["enabled"] = True
    m3["cross_camera"]["camera_group_id"] = "g1"
    m3["cross_camera"]["view_id"] = "v1"
    cfg = FallTaskConfig.from_mapping(m3)
    assert cfg.cross_camera.camera_group_id == "g1"
