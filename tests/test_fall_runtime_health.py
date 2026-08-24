"""阶段10（M2）— GET /system/fall-runtime 只读健康接口。

只测纯映射/脱敏函数,构造注入姿态侧 snapshot 与旁路 health endpoint 的"观测不改
状态"语义。不 import Torch/Ultralytics,不启动 Worker,不连 DB。
"""
from __future__ import annotations

import json

import pytest


# 姿态侧 snapshot 字段与后端 fall_runtime_health 的映射由实现提供
def _build_response(worker=None, gpu=None, model=None, delivery=None, cameras=(), **top):
    """生产实现入口:GREEN 时由源码提供。"""
    from backend.app.services.fall_runtime_health import build_fall_runtime_response

    return build_fall_runtime_response(
        worker=worker or {}, gpu=gpu or {}, model=model or {},
        delivery=delivery or {}, cameras=cameras, **top,
    )


def test_basic_ready_payload_layout():
    out = _build_response(
        state="READY",
        enabled=True,
        mode="shadow",
        runtime_key="pose-cuda-0",
        worker={
            "pid": 4321, "epoch": "epoch-A", "restart_count": 0,
            "heartbeat_age_ms": 120.0,
        },
        gpu={"device": "cuda:0", "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
             "allocated_mb": 712, "reserved_mb": 864, "effective_limit_mb": 3072},
        model={"name": "yolov8n-pose.pt", "sha256": "a" * 64, "precision": "fp16"},
        delivery={"transition_queue_depth": 0, "spool_pending": 0, "oldest_pending_age_ms": None},
    )
    assert out["schema_version"] == 1
    assert out["state"] == "READY"
    assert out["enabled"] is True
    assert out["mode"] == "shadow"
    assert out["runtime_key"] == "pose-cuda-0"
    assert out["error"] is None
    assert out["worker"]["pid"] == 4321
    assert out["worker"]["epoch"] == "epoch-A"
    assert out["gpu"]["allocated_mb"] == 712
    assert out["model"]["precision"] == "fp16"
    assert out["delivery"]["spool_pending"] == 0
    assert out["cameras"] == []


def test_disabled_returns_structured_disabled_not_error():
    out = _build_response(state="DISABLED", enabled=False, mode=None, runtime_key="pose-cuda-0")
    assert out["state"] == "DISABLED"
    assert out["enabled"] is False
    assert out["worker"] is None
    assert out["gpu"] is None


def test_sensitive_fields_are_redacted():
    out = _build_response(
        state="READY", enabled=True, mode="shadow", runtime_key="pose-cuda-0",
        worker={"pid": 4321, "epoch": "e", "restart_count": 0, "heartbeat_age_ms": 1.0},
        gpu={"device": "cuda:0", "name": "RTX 4060"},
        leaked_secret="token",
        exception_trace="Traceback ...",
    )
    blob = json.dumps(out)
    assert "token" not in blob
    assert "Traceback" not in blob
    # 绝对路径 / authkey 也不得泄漏
    assert "C:/Users" not in blob
    assert "authkey" not in blob.lower()


def test_old_heartbeat_and_incidents_surface_metrics():
    out = _build_response(
        state="DEGRADED", enabled=True, mode="shadow", runtime_key="k",
        worker={"pid": 1, "epoch": "e", "restart_count": 2, "heartbeat_age_ms": 5000.0},
        cameras=[
            {"camera_id": "cam-1", "camera_session_id": "s", "state": "DEGRADED",
             "effective_fps": 14.5, "latest_result_age_ms": 1800.0,
             "transition_queue_depth": 2, "open_incidents": 1},
        ],
    )
    assert out["state"] == "DEGRADED"
    assert out["worker"]["heartbeat_age_ms"] == 5000.0
    assert out["cameras"][0]["effective_fps"] == 14.5
    assert out["cameras"][0]["open_incidents"] == 1


def test_empty_snapshot_yields_disabled_without_worker():
    out = _build_response(state="DISABLED", enabled=False, mode=None, runtime_key=None)
    assert out["enabled"] is False
    assert out["runtime_key"] is None


def test_no_heavy_imports():
    """确保 fall_runtime_health 未 import torch/ultralytics。"""
    import sys

    from backend.app.services import fall_runtime_health as mod
    mod_src = " ".join(mod.__dict__.keys())
    for banned in ("torch", "ultralytics", "ai_monitor_pose.worker", "cv2"):
        assert banned not in mod_src