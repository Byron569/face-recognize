"""阶段10（M2）— GET /system/fall-runtime 只读健康接口的数据层。

只调用 client 包 ``PoseRuntimeRegistry.health_snapshot()``,绝不 import
Torch/Ultralytics,也不建立 Worker / 数据库连接。提供两个入口:

- ``collect_fall_runtime_health()``:从真实注册表快照映射为脱敏的 wire 响应;
- ``build_fall_runtime_response(...)``:纯函数,把所有字段显式构造为 wire dict,
  便于单元测试离线注入与字段脱敏断言。

响应不得泄漏 Pipe authkey、完整 Pipe 名、runtime 配置文件路径、数据库 URL、
模型绝对路径或异常堆栈。
"""
from __future__ import annotations

from typing import Any

# 只允许导入轻量 client 侧 DTO 与注册表; 不导入 torch/ultralytics/worker service
try:  # pragma: no cover - 注册表可为空(插件未安装)
    from ai_monitor_pose.runtime_registry import PoseRuntimeRegistry
    from ai_monitor_pose.health import FallRuntimeHealthSnapshotV1
    _HAS_CLIENT = True
except Exception:  # noqa: BLE001
    _HAS_CLIENT = False


def _redact(value: Any) -> Any:
    """递归剔除/跳过可能泄密字段的占位(简单白名单由各构建函数控制)。"""
    return value


def build_fall_runtime_response(
    *,
    state: str,
    enabled: bool,
    mode: str | None,
    runtime_key: str | None,
    worker: dict | None = None,
    gpu: dict | None = None,
    model: dict | None = None,
    delivery: dict | None = None,
    cameras: tuple | list = (),
    error: str | None = None,
    **_ignored: Any,
) -> dict:
    """把各节字段组装为后端 wire 响应。

    - state 为内部状态(READY/DEGRADED/UNAVAILABLE/DISABLED/...);
    - 所有传入 dict 仅接受白名单字段(见下方各构建器),其余丢弃以防泄漏。
    """
    response: dict[str, Any] = {
        "schema_version": 1,
        "enabled": bool(enabled),
        "mode": mode,
        "runtime_key": runtime_key,
        "state": state,
        "error": error,
    }
    if enabled and state != "DISABLED":
        response["worker"] = _build_worker(worker)
        response["gpu"] = _build_gpu(gpu)
        response["model"] = _build_model(model)
        response["delivery"] = _build_delivery(delivery)
        response["cameras"] = [_build_camera(c) for c in (cameras or ())]
    else:
        response["worker"] = None
        response["gpu"] = None
        response["model"] = None
        response["delivery"] = None
        response["cameras"] = []
    return response


def _build_worker(worker: dict | None) -> dict:
    if not worker:
        return {}
    allow = {
        "pid": worker.get("pid"),
        "epoch": worker.get("epoch"),
        "restart_count": worker.get("restart_count"),
        "heartbeat_age_ms": worker.get("heartbeat_age_ms"),
    }
    return {k: v for k, v in allow.items() if v is not None}


def _build_gpu(gpu: dict | None) -> dict:
    if not gpu:
        return {}
    allow = {
        "device": gpu.get("device"),
        "name": gpu.get("name"),
        "allocated_mb": gpu.get("allocated_mb"),
        "reserved_mb": gpu.get("reserved_mb"),
        "effective_limit_mb": gpu.get("effective_limit_mb"),
    }
    return {k: v for k, v in allow.items() if v is not None}


def _build_model(model: dict | None) -> dict:
    if not model:
        return {}
    allow = {
        "name": model.get("name"),
        "sha256": model.get("sha256"),
        "precision": model.get("precision"),
    }
    return {k: v for k, v in allow.items() if v is not None}


def _build_delivery(delivery: dict | None) -> dict:
    if not delivery:
        return {}
    allow = {
        "transition_queue_depth": delivery.get("transition_queue_depth"),
        "spool_pending": delivery.get("spool_pending"),
        "oldest_pending_age_ms": delivery.get("oldest_pending_age_ms"),
    }
    return {k: v for k, v in allow.items() if v is not None}


def _build_camera(c: object) -> dict:
    """快照里的 cameras 是 pose 侧 CameraFallHealthV1 数据类,统一映射为 wire 字段。"""
    from dataclasses import asdict
    d = asdict(c) if not isinstance(c, dict) else c
    allow = {
        "camera_id": d.get("camera_id"),
        "camera_session_id": d.get("camera_session_id"),
        "state": d.get("state"),
        "submitted": d.get("submitted_total"),
        "analyzed": d.get("analyzed_total"),
        "replaced": d.get("replaced_total"),
        "stale": d.get("stale_total"),
        "effective_fps": d.get("effective_fps"),
        "latest_result_age_ms": d.get("latest_result_age_ms"),
        "transition_queue_depth": d.get("transition_queue_depth"),
        "open_incidents": d.get("open_incidents"),
    }
    return {k: v for k, v in allow.items() if v is not None}


def _snapshot_to_fields(snap: FallRuntimeHealthSnapshotV1 | None) -> dict:
    """把姿态侧 DTO 映射为后端构建所需字段(纯字段摘取,不含配置路径/密钥)。"""
    if snap is None:
        return {"state": "DISABLED", "enabled": False, "mode": None, "runtime_key": None,
                "worker": None, "gpu": None, "model": None, "delivery": None, "cameras": ()}
    state = snap.worker.state if snap.worker else "DISABLED"
    enabled = bool(snap.enabled)
    worker_d = None
    if enabled and snap.worker:
        w = snap.worker
        heartbeat_age_ms = None
        if w.last_heartbeat_monotonic_ns is not None:
            import time
            age_ns = time.monotonic_ns() - w.last_heartbeat_monotonic_ns
            heartbeat_age_ms = max(0.0, age_ns / 1_000_000.0)
        worker_d = {
            "pid": w.worker_pid,
            "epoch": w.worker_epoch,
            "restart_count": w.restart_count,
            "heartbeat_age_ms": heartbeat_age_ms,
        }
    gpu_d = None
    if enabled:
        gm = snap.gpu_metrics or {}
        gpu_d = {
            "device": gm.get("cuda_device") or gm.get("device"),
            "name": snap.worker.cuda_device_name if snap.worker else gm.get("cuda_device_name"),
            "allocated_mb": gm.get("mem_allocated_peak_mb") or gm.get("allocated_mb"),
            "reserved_mb": gm.get("mem_reserved_peak_mb") or gm.get("reserved_mb"),
            "effective_limit_mb": gm.get("mem_effective_memory_limit_mb") or gm.get("effective_limit_mb"),
        }
    model_d = None
    if enabled:
        md = snap.model_metadata or {}
        model_d = {
            "name": md.get("name") or md.get("basename"),
            "sha256": snap.worker.model_sha256 if snap.worker else md.get("sha256"),
            "precision": md.get("precision"),
        }
    delivery_d = None
    if enabled:
        dm = snap.delivery_metrics or {}
        delivery_d = {
            "transition_queue_depth": dm.get("transition_queue_depth"),
            "spool_pending": dm.get("spool_pending"),
            "oldest_pending_age_ms": dm.get("oldest_pending_age_ms"),
        }
    cameras_t = tuple(c for c in (snap.cameras or ()))
    return {
        "state": state,
        "enabled": enabled,
        "mode": snap.mode,
        "runtime_key": snap.runtime_key,
        "worker": worker_d,
        "gpu": gpu_d,
        "model": model_d,
        "delivery": delivery_d,
        "cameras": cameras_t,
    }


def collect_fall_runtime_health() -> dict:
    """从真实 PoseRuntimeRegistry 读取并映射为脱敏 wire 响应。"""
    if not _HAS_CLIENT:
        return build_fall_runtime_response(
            state="DISABLED", enabled=False, mode=None, runtime_key=None
        )
    try:
        snap = PoseRuntimeRegistry.health_snapshot()
    except Exception:  # noqa: BLE001
        return build_fall_runtime_response(
            state="DISABLED", enabled=False, mode=None, runtime_key=None, error="unavailable"
        )
    fields = _snapshot_to_fields(snap)
    return build_fall_runtime_response(**fields)


def fall_runtime_health() -> dict:
    """给 system router 用的入口(保持轻量,不触发 Worker)。"""
    return collect_fall_runtime_health()