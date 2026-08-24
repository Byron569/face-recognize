"""阶段7：AI Monitor 侧 Task 契约（TaskRegistry 按目标路径加载）。"""
from __future__ import annotations

import sys

from backend.app.services.task_registry import instantiate_task, TaskRegistry

from ai_monitor_pose.task import FallDetectionTask
from vision.tasks import VisionTask


def _cfg() -> dict:
    return {
        "enabled": True,
        "mode": "shadow",
        "class_path": "ai_monitor_pose.task.FallDetectionTask",
        "runtime_key": "pose-cuda-0",
        "scheduler": {"target_fps": 8},
        "runtime": {"capacity_manifest_path": "D:/x/capacity.json",
                    "capacity_manifest_sha256": "a" * 64,
                    "max_result_age_ms": 1500, "overlay_ttl_ms": 1200,
                    "frame_slots_per_camera": 2, "batch_size": 1},
        "worker": {"python": "C:/x/python.exe"},
        "model": {"path": "D:/x/m.pt", "sha256_file": "D:/x/m.sha"},
        "gpu": {"required": True, "device": "cuda:0", "precision": "fp16",
                "allow_cpu_fallback": False},
        "algorithm": {
            "rotation_energy_min_rad_s": 1.8, "gravity_factor_min_body_heights_s2": 1.5,
            "fast_rotation_energy_min_rad_s": 3.0, "fast_gravity_factor_min_body_heights_s2": 2.5,
            "bbox_height_width_fall_max": 0.75, "trigger_ratio": 0.5,
            "min_trigger_duration_s": 0.5, "min_fall_pose_duration_s": 3.5,
            "recovery_duration_s": 1.0,
        },
        "cross_camera": {"enabled": False, "max_timestamp_skew_ms": 200},
    }


def test_fall_task_is_vision_task() -> None:
    assert issubclass(FallDetectionTask, VisionTask)


def test_existing_task_registry_loads_exact_class_path() -> None:
    reg = TaskRegistry({"fall_detection": _cfg()})
    tasks = reg.load()
    assert len(tasks) == 1
    assert isinstance(tasks[0], FallDetectionTask)
    assert tasks[0].name == "fall_detection"


def test_instantiate_task_accepts_extra_kwargs() -> None:
    task = instantiate_task("ai_monitor_pose.task.FallDetectionTask", _cfg(),
                            extra_kwargs={"event_sink": None, "junk_ignore": 1})
    assert isinstance(task, FallDetectionTask)


def test_import_does_not_pull_torch_or_ultralytics() -> None:
    assert "torch" not in sys.modules
    assert "ultralytics" not in sys.modules