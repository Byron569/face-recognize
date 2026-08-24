"""强类型不可变配置（第 4 节）。

构造期只做纯字段校验（绝对路径 / 设备 / 精度 / 下载开关 / 规则阈值 / revision），
不访问模型文件、不计算哈希、不启动 Runtime。模型存在性与 SHA-256 真值校验在阶段5 Worker bootstrap。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from typing import Any

from .errors import ConfigError

_CUDA_INDEX_RE = re.compile(r"^cuda:(\d+)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_abs(p: str) -> bool:
    return isinstance(p, str) and (p.startswith(("/", "\\")) or re.match(r"^[a-zA-Z]:[\\/]", p) is not None)


def _require_abs(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _is_abs(value):
        raise ConfigError(f"{what} 必须是绝对路径: {value!r}")
    return value


def _require_false(value: Any, what: str) -> bool:
    if value not in (False, True) or value is True:
        raise ConfigError(f"{what} 必须为 False: {value!r}")
    return False


def _require_true(value: Any, what: str) -> bool:
    if value is not True:
        raise ConfigError(f"{what} 必须为 True: {value!r}")
    return True


@dataclass(frozen=True, slots=True)
class GpuConfig:
    required: bool
    device: str
    expected_device_name_regex: str
    precision: str
    allow_cpu_fallback: bool
    max_gpu_memory_mb: int
    max_vram_fraction: float
    minimum_free_vram_mb: int


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    python: str
    module: str
    pipe_prefix: str
    startup_timeout_s: float
    heartbeat_interval_s: float
    heartbeat_timeout_s: float
    inference_timeout_s: float
    graceful_shutdown_timeout_s: float
    zero_lease_worker_drain_timeout_s: float
    idle_retention_s: float


@dataclass(frozen=True, slots=True)
class ModelConfig:
    path: str
    sha256_file: str
    allow_download: bool
    imgsz: int
    confidence: float
    iou: float
    max_detections: int


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    max_cameras: int
    requested_max_total_fps: int
    capacity_manifest_path: str
    capacity_manifest_sha256: str | None
    capacity_headroom_ratio: float
    frame_slots_per_camera: int
    max_frame_width: int
    max_frame_height: int
    max_result_age_ms: int
    overlay_ttl_ms: int
    transition_queue_capacity_per_camera: int
    transition_queue_resume_ratio: float
    worker_journal_path: str
    worker_journal_pending_capacity: int
    event_spool_path: str
    event_spool_pending_capacity: int
    delivered_retention_hours: int
    delivered_retention_rows: int


@dataclass(frozen=True, slots=True)
class DeliveryConfig:
    ingress_queue_capacity: int
    outbox_pending_capacity: int
    outbox_resume_ratio: float
    outbox_delivered_retention_hours: int
    outbox_delivered_retention_rows: int


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    target_fps: int
    latest_only: bool
    batch_size: int
    fair_policy: str
    not_ready_retry_ms: int


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    high_confidence: float
    low_confidence: float
    match_iou_threshold: float
    max_normalized_keypoint_distance: float
    min_confirm_duration_s: float
    lost_timeout_s: float
    ghost_timeout_s: float
    fallen_ghost_timeout_s: float


@dataclass(frozen=True, slots=True)
class AlgorithmConfig:
    required_keypoint_indices: tuple[int, ...]
    minimum_visible_required: int
    keypoint_min_confidence: float
    minimum_body_height_ratio: float
    bbox_height_width_fall_max: float
    torso_inclination_from_vertical_min_deg: float
    hip_angle_fall_min_deg: float
    head_descent_body_heights_min: float
    rotation_energy_min_rad_s: float
    gravity_factor_min_body_heights_s2: float
    fast_rotation_energy_min_rad_s: float
    fast_gravity_factor_min_body_heights_s2: float
    upright_height_width_min: float
    upright_torso_inclination_max_deg: float
    rebound_body_heights_min: float
    ema_tau_s: float
    history_duration_s: float
    trigger_window_s: float
    trigger_ratio: float
    min_trigger_duration_s: float
    max_trigger_gap_s: float
    min_fall_pose_duration_s: float
    recovery_duration_s: float
    rebound_duration_s: float
    tracker_reset_gap_s: float
    allow_already_down_detection: bool


@dataclass(frozen=True, slots=True)
class CrossCameraConfig:
    enabled: bool
    camera_group_id: str | None
    view_id: str | None
    max_timestamp_skew_ms: int


def _sub(data: Mapping, key: str, builder: Any) -> Any:
    return builder(data.get(key) or {})


def _positive_int(data: Mapping, key: str, default: int | None = None) -> int:
    v = data.get(key, default)  # type: ignore[arg-type]
    if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
        raise ConfigError(f"{key} 必须为正整数: {v!r}")
    return v


def _ratio(data: Mapping, key: str, default: float | None = None) -> float:
    v = data.get(key, default)  # type: ignore[arg-type]
    try:
        vf = float(v)
    except (TypeError, ValueError):
        raise ConfigError(f"{key} 必须为数值: {v!r}") from None
    if not math.isfinite(vf) or not (0.0 < vf < 1.0):
        raise ConfigError(f"{key} 必须在 (0,1): {v!r}")
    return vf


@dataclass(frozen=True, slots=True)
class FallTaskConfig:
    enabled: bool
    mode: str
    class_path: str
    interval: int
    runtime_key: str
    config_revision: str
    worker: WorkerConfig
    gpu: GpuConfig
    model: ModelConfig
    runtime: RuntimeConfig
    delivery: DeliveryConfig
    scheduler: SchedulerConfig
    tracker: TrackerConfig
    algorithm: AlgorithmConfig
    cross_camera: CrossCameraConfig

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> "FallTaskConfig":
        d = dict(raw or {})
        enabled = bool(d.get("enabled", False))
        mode = str(d.get("mode", "shadow"))
        if mode not in ("shadow", "alert"):
            raise ConfigError(f"mode 只允许 shadow/alert: {mode!r}")
        if mode == "alert" and not enabled:
            raise ConfigError("alert mode 要求 enabled=true")

        gpu_d = d.get("gpu") or {}
        if _require_true(gpu_d.get("required"), "gpu.required") is False:  # pragma: no cover
            raise ConfigError("gpu.required 必须为 true")
        device = gpu_d.get("device")
        if isinstance(device, str):
            m = _CUDA_INDEX_RE.match(device)
            if not m:
                raise ConfigError(f"gpu.device 只接受 cuda:N: {device!r}")
        else:
            raise ConfigError(f"gpu.device 必须是字符串: {device!r}")
        precision = gpu_d.get("precision")
        if precision not in ("fp16", "fp32"):
            raise ConfigError(f"precision 只允许 fp16/fp32: {precision!r}")
        allow_cpu = gpu_d.get("allow_cpu_fallback", False)
        if allow_cpu not in (False,) or bool(allow_cpu):
            raise ConfigError("allow_cpu_fallback 必须为 false (不允许 CPU fallback)")
        gpu = GpuConfig(
            required=True,
            device=device,
            expected_device_name_regex=str(gpu_d.get("expected_device_name_regex", "")),
            precision=precision,
            allow_cpu_fallback=False,
            max_gpu_memory_mb=_positive_int(gpu_d, "max_gpu_memory_mb", 3072),
            max_vram_fraction=_ratio(
                gpu_d, "max_vram_fraction", 0.80
            )
            if "max_vram_fraction" in gpu_d
            else 0.80,
            minimum_free_vram_mb=_positive_int(gpu_d, "minimum_free_vram_mb", 1024),
        )

        w_d = d.get("worker") or {}
        worker = WorkerConfig(
            python=_require_abs(w_d.get("python"), "worker.python"),
            module=str(w_d.get("module", "ai_monitor_pose.worker")),
            pipe_prefix=str(w_d.get("pipe_prefix", "ai-monitor-fall-v1")),
            startup_timeout_s=float(w_d.get("startup_timeout_s", 45)),
            heartbeat_interval_s=float(w_d.get("heartbeat_interval_s", 1)),
            heartbeat_timeout_s=float(w_d.get("heartbeat_timeout_s", 3)),
            inference_timeout_s=float(w_d.get("inference_timeout_s", 5)),
            graceful_shutdown_timeout_s=float(w_d.get("graceful_shutdown_timeout_s", 5)),
            zero_lease_worker_drain_timeout_s=float(w_d.get("zero_lease_worker_drain_timeout_s", 5)),
            idle_retention_s=float(w_d.get("idle_retention_s", 60)),
        )

        m_d = d.get("model") or {}
        md = ModelConfig(
            path=_require_abs(m_d.get("path"), "model.path"),
            sha256_file=_require_abs(m_d.get("sha256_file"), "model.sha256_file"),
            allow_download=(
                _require_false(m_d.get("allow_download", False), "model.allow_download")
            ),
            imgsz=int(m_d.get("imgsz", 640)),
            confidence=float(m_d.get("confidence", 0.35)),
            iou=float(m_d.get("iou", 0.45)),
            max_detections=int(m_d.get("max_detections", 50)),
        )

        r_d = d.get("runtime") or {}
        cap_path = _require_abs(r_d.get("capacity_manifest_path"), "runtime.capacity_manifest_path")
        cap_digest = r_d.get("capacity_manifest_sha256")
        if cap_digest is not None:
            if not isinstance(cap_digest, str) or not _SHA256_RE.match(cap_digest):
                raise ConfigError("capacity_manifest_sha256 必须为 64 位小写 sha256 或 null")
        if enabled and cap_digest is None:
            raise ConfigError("enabled=true 时 capacity_manifest_sha256 不能为 null")
        frame_slots = int(r_d.get("frame_slots_per_camera", 2))
        batch = int(d.get("scheduler", {}).get("batch_size", 1))
        if frame_slots != 2:
            raise ConfigError("frame_slots_per_camera 第一版固定为 2")
        if batch != 1:
            raise ConfigError("scheduler.batch_size 第一版固定为 1")
        overlay_ttl = _positive_int(r_d, "overlay_ttl_ms", 1200)
        max_age = _positive_int(r_d, "max_result_age_ms", 1500)
        if overlay_ttl > max_age:
            raise ConfigError("overlay_ttl_ms 必须 <= max_result_age_ms")
        rt = RuntimeConfig(
            max_cameras=_positive_int(r_d, "max_cameras", 4),
            requested_max_total_fps=_positive_int(r_d, "requested_max_total_fps", 32),
            capacity_manifest_path=cap_path,
            capacity_manifest_sha256=cap_digest,
            capacity_headroom_ratio=_ratio(r_d, "capacity_headroom_ratio", 0.75),
            frame_slots_per_camera=frame_slots,
            max_frame_width=_positive_int(r_d, "max_frame_width", 1920),
            max_frame_height=_positive_int(r_d, "max_frame_height", 1080),
            max_result_age_ms=max_age,
            overlay_ttl_ms=overlay_ttl,
            transition_queue_capacity_per_camera=_positive_int(r_d, "transition_queue_capacity_per_camera", 64),
            transition_queue_resume_ratio=_ratio(r_d, "transition_queue_resume_ratio", 0.50),
            worker_journal_path=str(r_d.get("worker_journal_path", "")),
            worker_journal_pending_capacity=_positive_int(r_d, "worker_journal_pending_capacity", 10000),
            event_spool_path=str(r_d.get("event_spool_path", "")),
            event_spool_pending_capacity=_positive_int(r_d, "event_spool_pending_capacity", 10000),
            delivered_retention_hours=_positive_int(r_d, "delivered_retention_hours", 24),
            delivered_retention_rows=_positive_int(r_d, "delivered_retention_rows", 1000),
        )

        dl_d = d.get("delivery") or {}
        delivery = DeliveryConfig(
            ingress_queue_capacity=_positive_int(dl_d, "ingress_queue_capacity", 1024),
            outbox_pending_capacity=_positive_int(dl_d, "outbox_pending_capacity", 10000),
            outbox_resume_ratio=_ratio(dl_d, "outbox_resume_ratio", 0.50),
            outbox_delivered_retention_hours=_positive_int(dl_d, "outbox_delivered_retention_hours", 24),
            outbox_delivered_retention_rows=_positive_int(dl_d, "outbox_delivered_retention_rows", 10000),
        )

        s_d = d.get("scheduler") or {}
        scheduler = SchedulerConfig(
            target_fps=_positive_int(s_d, "target_fps", 8),
            latest_only=bool(s_d.get("latest_only", True)),
            batch_size=1,
            fair_policy=str(s_d.get("fair_policy", "service_debt_round_robin")),
            not_ready_retry_ms=_positive_int(s_d, "not_ready_retry_ms", 500),
        )

        t_d = d.get("tracker") or {}
        tracker = TrackerConfig(
            high_confidence=float(t_d.get("high_confidence", 0.35)),
            low_confidence=float(t_d.get("low_confidence", 0.15)),
            match_iou_threshold=float(t_d.get("match_iou_threshold", 0.50)),
            max_normalized_keypoint_distance=float(t_d.get("max_normalized_keypoint_distance", 0.35)),
            min_confirm_duration_s=float(t_d.get("min_confirm_duration_s", 0.25)),
            lost_timeout_s=float(t_d.get("lost_timeout_s", 1.5)),
            ghost_timeout_s=float(t_d.get("ghost_timeout_s", 3.0)),
            fallen_ghost_timeout_s=float(t_d.get("fallen_ghost_timeout_s", 5.0)),
        )

        a_d = d.get("algorithm") or {}
        algorithm = AlgorithmConfig(
            required_keypoint_indices=tuple(
                int(x) for x in a_d.get("required_keypoint_indices", [0, 5, 6, 11, 12, 13, 14])
            ),
            minimum_visible_required=int(a_d.get("minimum_visible_required", 5)),
            keypoint_min_confidence=float(a_d.get("keypoint_min_confidence", 0.30)),
            minimum_body_height_ratio=float(a_d.get("minimum_body_height_ratio", 0.12)),
            bbox_height_width_fall_max=float(a_d.get("bbox_height_width_fall_max", 0.75)),
            torso_inclination_from_vertical_min_deg=float(a_d.get("torso_inclination_from_vertical_min_deg", 55.0)),
            hip_angle_fall_min_deg=float(a_d.get("hip_angle_fall_min_deg", 135.0)),
            head_descent_body_heights_min=float(a_d.get("head_descent_body_heights_min", 0.18)),
            rotation_energy_min_rad_s=float(a_d.get("rotation_energy_min_rad_s", 1.80)),
            gravity_factor_min_body_heights_s2=float(a_d.get("gravity_factor_min_body_heights_s2", 1.50)),
            fast_rotation_energy_min_rad_s=float(a_d.get("fast_rotation_energy_min_rad_s", 3.00)),
            fast_gravity_factor_min_body_heights_s2=float(a_d.get("fast_gravity_factor_min_body_heights_s2", 2.50)),
            upright_height_width_min=float(a_d.get("upright_height_width_min", 1.25)),
            upright_torso_inclination_max_deg=float(a_d.get("upright_torso_inclination_max_deg", 25.0)),
            rebound_body_heights_min=float(a_d.get("rebound_body_heights_min", 0.12)),
            ema_tau_s=float(a_d.get("ema_tau_s", 0.20)),
            history_duration_s=float(a_d.get("history_duration_s", 4.5)),
            trigger_window_s=float(a_d.get("trigger_window_s", 1.25)),
            trigger_ratio=float(a_d.get("trigger_ratio", 0.50)),
            min_trigger_duration_s=float(a_d.get("min_trigger_duration_s", 0.50)),
            max_trigger_gap_s=float(a_d.get("max_trigger_gap_s", 0.25)),
            min_fall_pose_duration_s=float(a_d.get("min_fall_pose_duration_s", 3.50)),
            recovery_duration_s=float(a_d.get("recovery_duration_s", 1.00)),
            rebound_duration_s=float(a_d.get("rebound_duration_s", 0.50)),
            tracker_reset_gap_s=float(a_d.get("tracker_reset_gap_s", 1.50)),
            allow_already_down_detection=bool(a_d.get("allow_already_down_detection", False)),
        )
        _validate_algorithm(algorithm)

        x_d = d.get("cross_camera") or {}
        xc_enabled = bool(x_d.get("enabled", False))
        grp = x_d.get("camera_group_id")
        vid = x_d.get("view_id")
        if xc_enabled and (not grp or not vid):
            raise ConfigError("cross_camera.enabled=true 时必须同时提供 camera_group_id 与 view_id")
        cross = CrossCameraConfig(
            enabled=xc_enabled,
            camera_group_id=(str(grp) if grp else None),
            view_id=(str(vid) if vid else None),
            max_timestamp_skew_ms=_positive_int(x_d, "max_timestamp_skew_ms", 200),
        )

        candidate = FallTaskConfig(
            enabled=enabled,
            mode=mode,
            class_path=str(d.get("class_path", "ai_monitor_pose.task.FallDetectionTask")),
            interval=int(d.get("interval", 1)),
            runtime_key=str(d.get("runtime_key", "pose-cuda-0")),
            config_revision="",
            worker=worker,
            gpu=gpu,
            model=md,
            runtime=rt,
            delivery=delivery,
            scheduler=scheduler,
            tracker=tracker,
            algorithm=algorithm,
            cross_camera=cross,
        )
        rev = candidate._compute_config_revision()
        return replace(candidate, config_revision=rev)

    def _compute_config_revision(self) -> str:
        d = asdict(self)
        d.pop("config_revision", None)
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_algorithm(a: AlgorithmConfig) -> None:
    algos = {f.name: getattr(a, f.name) for f in fields(a)}
    for name, v in algos.items():
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
            raise ConfigError(f"algorithm.{name} 必须为有限数值")
    if not (0.0 < algos["bbox_height_width_fall_max"] < 1.0):
        raise ConfigError("bbox_height_width_fall_max 必须在 (0,1)")
    if not (0.0 < algos["trigger_ratio"] < 1.0):
        raise ConfigError("trigger_ratio 必须在 (0,1)")
    if algos["min_trigger_duration_s"] <= 0:
        raise ConfigError("min_trigger_duration_s 必须为正")
    if algos["min_fall_pose_duration_s"] <= 0:
        raise ConfigError("min_fall_pose_duration_s 必须为正")
    if algos["recovery_duration_s"] <= 0:
        raise ConfigError("recovery_duration_s 必须为正")
