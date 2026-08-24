"""阶段8b — 后端测试 2:EventIngress 接线与事件入口(无 DB)。

覆盖:
    - 单例入口 get_event_ingress;
    - 非 fall 事件 / 缺 event_id·dedupe_key 的事件在 submit 的 _prepare 阶段即被拒绝;
    - 未绑定 loop 时 submit 立即返回 IngressNotRunning,绝不静默;
    - 投递模式解析(默认 shadow,可按摄像头覆盖);
    - TaskRegistry 将 event_sink 与 runtime_factory 注入 FallDetectionTask;
    - PipelineManager.set_event_loop 绑定/启动 ingress,广播回调安全、shutdown 幂等。
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest


# ── 辅助 ──────────────────────────────────────────────

def _fall_event(**overrides) -> SimpleNamespace:
    payload = {
        "event_id": "evt-1",
        "dedupe_key": "dk-1",
        "incident_id": "inc-1",
        "score_semantics": "heuristic_rule_score_not_probability",
    }
    payload.update(overrides.pop("_payload", {}))
    base = dict(
        event_type="fall_detected",
        camera_id="cam-1",
        track_id=3,
        confidence=0.9,
        timestamp=1700000000.0,
        payload_json_utf8=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _fresh_ingress(**kw):
    from backend.app.services.event_ingress import EventIngress

    return EventIngress(**kw)


def _min_fall_cfg() -> dict:
    return {
        "enabled": True,
        "mode": "shadow",
        "class_path": "ai_monitor_pose.task.FallDetectionTask",
        "runtime_key": "pose-cuda-0",
        "scheduler": {"target_fps": 8},
        "runtime": {
            "capacity_manifest_path": "D:/x/capacity.json",
            "capacity_manifest_sha256": "a" * 64,
            "max_result_age_ms": 1500,
            "overlay_ttl_ms": 1200,
            "frame_slots_per_camera": 2,
            "batch_size": 1,
        },
        "worker": {"python": "C:/x/python.exe"},
        "model": {"path": "D:/x/m.pt", "sha256_file": "D:/x/m.sha"},
        "gpu": {
            "required": True,
            "device": "cuda:0",
            "precision": "fp16",
            "allow_cpu_fallback": False,
        },
        "algorithm": {
            "rotation_energy_min_rad_s": 1.8,
            "gravity_factor_min_body_heights_s2": 1.5,
            "fast_rotation_energy_min_rad_s": 3.0,
            "fast_gravity_factor_min_body_heights_s2": 2.5,
            "bbox_height_width_fall_max": 0.75,
            "trigger_ratio": 0.5,
            "min_trigger_duration_s": 0.5,
            "min_fall_pose_duration_s": 3.5,
            "recovery_duration_s": 1.0,
        },
        "cross_camera": {"enabled": False, "max_timestamp_skew_ms": 200},
    }


# ── 单例 ─────────────────────────────────────────────

def test_get_event_ingress_returns_singleton() -> None:
    from backend.app.services.pipeline_manager import get_event_ingress

    assert get_event_ingress() is get_event_ingress()


# ── _prepare 拒绝路径(不依赖 DB/loop)──────────────────

def test_prepare_rejects_non_fall_event_type() -> None:
    """非 fall 事件在 _prepare 阶段即被拒绝(IngressRejectedError)。"""
    from backend.app.services.event_ingress import IngressRejectedError

    ingress = _fresh_ingress()
    with pytest.raises(IngressRejectedError):
        ingress._prepare(_fall_event(event_type="recognition"))


def test_prepare_rejects_missing_event_id_and_dedupe_key() -> None:
    from backend.app.services.event_ingress import IngressRejectedError

    ingress = _fresh_ingress()
    # payload 仅 incident_id,无 event_id / dedupe_key → 必须拒绝
    bare = _fall_event(
        payload_json_utf8=json.dumps({"incident_id": "inc-1"}).encode("utf-8")
    )
    with pytest.raises(IngressRejectedError):
        ingress._prepare(bare)


def test_prepare_rejects_payload_bad_json() -> None:
    from backend.app.services.event_ingress import IngressRejectedError

    ingress = _fresh_ingress()
    bad = _fall_event(payload_json_utf8=b"not json")
    with pytest.raises(IngressRejectedError):
        ingress._prepare(bad)


def test_submit_when_not_started_fails_with_ingress_not_running() -> None:
    from backend.app.services.event_ingress import IngressNotRunning

    # 未绑定 loop: 有效 fall 事件直接失败,不静默
    fut = _fresh_ingress().submit(_fall_event())
    with pytest.raises(IngressNotRunning):
        fut.result(timeout=1)


# ── 投递模式解析 ─────────────────────────────────────

def test_mode_resolver_defaults_to_shadow() -> None:
    ingress = _fresh_ingress()
    assert ingress._mode_resolver("any-cam") == "shadow"


def test_mode_resolver_respects_custom() -> None:
    ingress = _fresh_ingress(mode_resolver=lambda cid: "alert" if cid == "a" else "shadow")
    assert ingress._mode_resolver("a") == "alert"
    assert ingress._mode_resolver("b") == "shadow"


# ── 注入到 FallDetectionTask ──────────────────────────

def test_task_registry_injects_event_sink_and_runtime_factory() -> None:
    from ai_monitor_pose.runtime_registry import PoseRuntimeRegistry
    from ai_monitor_pose.task import FallDetectionTask
    from backend.app.services.task_registry import TaskRegistry

    sink = _fresh_ingress()
    tasks = TaskRegistry({"fall_detection": _min_fall_cfg()}).load(
        extra_kwargs={"event_sink": sink, "runtime_factory": PoseRuntimeRegistry}
    )
    assert len(tasks) == 1
    task = tasks[0]
    assert isinstance(task, FallDetectionTask)
    assert task._event_sink is sink
    assert task._runtime_factory is PoseRuntimeRegistry


# ── PipelineManager 触发 ingress ─────────────────────

def test_pipeline_manager_wires_ingress_and_shutdown_idempotent() -> None:
    from backend.app.services.pipeline_manager import PipelineManager

    mgr = PipelineManager()

    async def _exercise() -> None:
        mgr.set_event_loop(asyncio.get_running_loop())
        assert mgr._event_ingress is not None
        # 默认 shadow;摄像头覆盖为 alert
        assert mgr._resolve_delivery_mode("unknown-cam") == "shadow"
        mgr._delivery_modes["camA"] = "alert"
        assert mgr._resolve_delivery_mode("camA") == "alert"
        # 无监听者时广播安全
        await mgr._broadcast_ingress_event({"type": "event", "event_id": "e1"})
        # shutdown 幂等(无摄像头 + 关 ingress)
        await mgr.shutdown()
        await mgr.shutdown()

    asyncio.run(_exercise())