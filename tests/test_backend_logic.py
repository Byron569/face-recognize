"""backend 纯逻辑单元测试(不依赖数据库/GPU):配置级联 + 底库快照 + 任务注册。"""

import numpy as np
import pytest

from backend.app.config import build_camera_config, load_profile_config, _deep_merge


def _root():
    import os

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── 配置级联 ────────────────────────────────────────────

def test_deep_merge_nested_override():
    base = {"vision": {"device": "cuda", "det_interval": 2}, "tasks": {}}
    override = {"vision": {"det_interval": 5}}
    out = _deep_merge(base, override)
    assert out["vision"]["device"] == "cuda"
    assert out["vision"]["det_interval"] == 5


def test_profile_cascade_vision_device():
    merged = load_profile_config("desktop")
    assert merged["vision"]["device"] == "cuda"
    assert merged["vision"]["det_interval"] == 2


def test_camera_extra_overrides_profile():
    cfg = build_camera_config("desktop", {"vision": {"det_interval": 9}})
    assert cfg["vision"]["det_interval"] == 9
    assert cfg["vision"]["device"] == "cuda"


def test_camera_stream_config_overrides_all_default_preview_values():
    cfg = build_camera_config(
        "desktop",
        {"stream": {"max_height": 720, "jpeg_quality": 68, "push_fps": 20}},
    )
    assert cfg["stream"] == {
        "max_height": 720,
        "jpeg_quality": 68,
        "push_fps": 20,
    }


# ── 内存底库快照 ────────────────────────────────────────

def test_gallery_search():
    from backend.app.services.gallery import FaceGallery

    gallery = FaceGallery()
    emb_a = np.array([1.0, 0.0] * 256, dtype=np.float32)
    emb_a /= np.linalg.norm(emb_a)
    emb_b = np.array([0.0, 1.0] * 256, dtype=np.float32)
    emb_b /= np.linalg.norm(emb_b)
    gallery.rebuild(
        [("id-a", "Alice", emb_a.tolist()), ("id-b", "Bob", emb_b.tolist())]
    )
    hit = gallery.search(emb_a, threshold=0.4)
    assert hit is not None and hit[1] == "Alice"
    assert hit[2] > 0.99
    assert gallery.size == 2


def test_gallery_empty_and_threshold():
    from backend.app.services.gallery import FaceGallery

    gallery = FaceGallery()
    assert gallery.search(np.zeros(512, dtype=np.float32), 0.4) is None
    emb = np.eye(512)[0].astype(np.float32)
    gallery.rebuild([("id-a", "Alice", emb.tolist())])
    assert gallery.search(np.eye(512)[1].astype(np.float32), 0.9) is None


# ── 任务注册表 ──────────────────────────────────────────

def test_task_registry_instantiate_via_class_path():
    from backend.app.services.task_registry import TaskRegistry, instantiate_task

    registry = TaskRegistry(
        {
            "face_recognition": {
                "enabled": True,
                "class_path": "app.tasks.builtin.face_recognition_task.FaceRecognitionTask",
            },
            "fall_detection": {"enabled": False, "class_path": None},
        }
    )
    import sys

    sys.path.insert(0, os.path.join(_root(), "backend"))
    tasks = registry.load(extra_kwargs={"full_config": {}, "gallery": None, "tracker": None})
    assert len(tasks) == 1
    assert tasks[0].name == "face_recognition"
    registered = registry.registered
    assert any(t["name"] == "fall_detection" and not t["enabled"] for t in registered)


def test_task_registry_skips_disabled_and_bad_paths():
    from backend.app.services.task_registry import TaskRegistry

    registry = TaskRegistry(
        {
            "a": {"enabled": False, "class_path": "x.y.Z"},
            "b": {"enabled": True, "class_path": None},
            "c": {"enabled": True, "class_path": "no.such.module.Task"},
        }
    )
    assert registry.load(extra_kwargs={}) == []


# ── 识别任务调度(纯逻辑,无 GPU/DB)────────────────────────

def test_recognition_task_cooldown_logic():
    from backend.app.tasks.builtin.face_recognition_task import (
        FaceRecognitionTask,
        _TrackRecState,
    )

    task = FaceRecognitionTask(config={}, full_config={}, gallery=None, tracker=None)
    st = _TrackRecState()
    st.last_attempt_frame = 0
    rec_cfg = task._rec_cfg
    # 冷却期内不应识别(在 _maybe_recognize 中判断,这里直接验证冷却计算)
    effective = rec_cfg.cooldown_frames + st.fail_count * rec_cfg.failed_backoff_frames
    assert 100 - st.last_attempt_frame < effective  # frame 100 时仍在冷却
    st.fail_count = 3
    effective = rec_cfg.cooldown_frames + st.fail_count * rec_cfg.failed_backoff_frames
    assert effective == 300 + 3 * 90


def test_recognition_task_limits_attempts_even_when_unknown():
    """max_per_frame 限流:识别为 Unknown(也执行了底库比对)同样计入次数。"""
    from backend.app.tasks.builtin.face_recognition_task import FaceRecognitionTask
    from vision.events import PipelineContext, TrackResult

    class FakeGallery:
        def __init__(self):
            self.calls = 0

        def search(self, query, threshold):
            self.calls += 1
            return None  # 全部 Unknown

    g = FakeGallery()
    task = FaceRecognitionTask(config={"max_per_frame": 1}, full_config={}, gallery=g, tracker=None)
    ctx = PipelineContext(
        camera_id="c0",
        frame_id=10,
        frame=None,
        tracks=[
            TrackResult(track_id=1, bbox=(0, 0, 10, 10), embedding=[0.1] * 512),
            TrackResult(track_id=2, bbox=(20, 20, 30, 30), embedding=[0.2] * 512),
        ],
    )
    task.run(None, ctx)
    assert g.calls == 1, f"expected 1 gallery search (rate-limited), got {g.calls}"


import os  # noqa: E402
