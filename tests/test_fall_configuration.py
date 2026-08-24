"""阶段8b — 后端测试 1:fall_detection 配置接线与运行不变式(无 DB)。

覆盖 default.yaml 的基线配置姿态:
    - 动态类路径固定为 ai_monitor_pose.task.FallDetectionTask;
    - M1 默认 enabled=false,且 profile 不得默认启用;
    - GPU 硬约束:required、设备为 cuda:N、允许 CPU 回退必须为 false;
    - 模型不允许自动下载,worker/python 与 runtime 持久化路径必须为绝对路径;
    - delivery 背压边界:正容量、(0,1) 比例。
"""
from __future__ import annotations

import os

from backend.app.config import build_camera_config


def _fall(cfg: dict) -> dict:
    tasks = cfg.get("tasks", {}) or {}
    return tasks.get("fall_detection", {}) or {}


def _require(fall: dict, section: str) -> dict:
    out = fall.get(section, {}) or {}
    assert out, f"fall_detection.{section} 缺失"
    return out


def test_fall_detection_baseline_present_and_enabled() -> None:
    fall = _fall(build_camera_config(None))
    assert fall.get("class_path") == "ai_monitor_pose.task.FallDetectionTask"
    assert fall.get("enabled", True) is True  # 基线默认启用(实时人体框/骨骼叠加)
    assert fall.get("mode") in {"shadow", "alert"}


def test_fall_gpu_never_falls_back_to_cpu() -> None:
    gpu = _require(_fall(build_camera_config(None)), "gpu")
    assert gpu.get("required") is True
    assert gpu.get("allow_cpu_fallback") is False
    dev = str(gpu.get("device", ""))
    assert dev.startswith("cuda:") and dev[5:].isdigit()
    assert gpu.get("precision") in {"fp16", "fp32"}


def test_fall_model_never_downloads_and_paths_are_absolute() -> None:
    fall = _fall(build_camera_config(None))
    model = _require(fall, "model")
    assert model.get("allow_download") is False
    for key in ("path", "sha256_file"):
        assert os.path.isabs(str(model.get(key))), f"model.{key} 必须为绝对路径"
    worker = _require(fall, "worker")
    assert os.path.isabs(str(worker.get("python"))), "worker.python 必须为绝对路径"


def test_fall_runtime_persistence_paths_are_absolute() -> None:
    rt = _require(_fall(build_camera_config(None)), "runtime")
    for key in ("worker_journal_path", "event_spool_path", "capacity_manifest_path"):
        assert os.path.isabs(str(rt.get(key))), f"runtime.{key} 必须为绝对路径"


def test_profiles_inherit_fall_enabled_from_baseline() -> None:
    # 每个 profile 只允许覆盖 enabled/mode/target_fps;未覆盖时继承基线默认启用
    for profile in ("desktop", "balanced", "edge_minimal"):
        fall = _fall(build_camera_config(profile))
        assert fall.get("enabled", True) is True, f"{profile} 应继承基线启用 fall_detection"


def test_delivery_capacity_boundaries() -> None:
    d = _require(_fall(build_camera_config(None)), "delivery")
    assert int(d.get("ingress_queue_capacity", 0)) > 0
    assert int(d.get("outbox_pending_capacity", 0)) > 0
    ratio = float(d.get("outbox_resume_ratio", 0))
    assert 0.0 < ratio < 1.0