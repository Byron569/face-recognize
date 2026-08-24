"""健康 DTO 与工具（第 6.9 节）。

只可被父进程侧（client/registry/runtime）安全导入；不得 import Torch/模型/Worker service。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

HEALTH_STATES = ("STARTING", "READY", "DEGRADED", "UNAVAILABLE", "STOPPING", "STOPPED")

STARTING = "STARTING"
READY = "READY"
DEGRADED = "DEGRADED"
UNAVAILABLE = "UNAVAILABLE"
STOPPING = "STOPPING"
STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class WorkerHealthV1:
    schema_version: int
    state: str
    error_code: str | None
    error_message: str | None
    worker_epoch: str | None
    worker_pid: int | None
    cuda_device: str | None
    cuda_device_name: str | None
    model_sha256: str | None
    last_heartbeat_monotonic_ns: int | None
    restart_count: int


@dataclass(frozen=True, slots=True)
class CameraFallHealthV1:
    camera_id: str
    camera_session_id: str
    state: str
    submitted_total: int
    analyzed_total: int
    replaced_total: int
    stale_total: int
    effective_fps: float
    latest_result_age_ms: float | None
    transition_queue_depth: int
    open_incidents: int


@dataclass(frozen=True, slots=True)
class FallRuntimeHealthSnapshotV1:
    schema_version: int
    enabled: bool
    mode: str | None
    runtime_key: str | None
    worker: WorkerHealthV1
    gpu_metrics: dict[str, Any]
    model_metadata: dict[str, Any]
    delivery_metrics: dict[str, Any]
    cameras: tuple[CameraFallHealthV1, ...]