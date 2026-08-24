"""阶段6 worker.service 离线单测：mock 引擎 + mock 帧读取，验证 pipe 协议面与整条推理流水线。

不依赖 GPU / 数据库 / 运行中的后端。协议消息形状对齐 tests/fixtures/fake_worker.py 黄金样本。
"""
from __future__ import annotations

import os

import numpy as np

from ai_monitor_pose.config import FallTaskConfig
from ai_monitor_pose.worker.service import WorkerService

from tests.fixtures.pose_sequences import bbox_of, horizontal_keypoints, standing_keypoints


def _make_config() -> FallTaskConfig:
    """构造满足 config 校验的最简绝对路径集；enabled=False 走 shadow 不强制 manifest sha。"""
    import tempfile
    base = tempfile.mkdtemp()
    return FallTaskConfig.from_mapping({
        "enabled": False,
        "mode": "shadow",
        "gpu": {"required": True, "device": "cuda:0", "precision": "fp16",
                "allow_cpu_fallback": False},
        "worker": {"python": "C:/py.exe", "module": "ai_monitor_pose.worker"},
        "model": {"path": os.path.join(base, "yolov8n-pose.pt"),
                  "sha256_file": os.path.join(base, "yolov8n-pose.pt.sha256"),
                  "allow_download": False,
                  "imgsz": 640, "confidence": 0.35, "iou": 0.45, "max_detections": 50},
        "runtime": {"capacity_manifest_path": os.path.join(base, "capacity-cuda0.json")},
    })


class _FakeRaw:
    def __init__(self, kps) -> None:
        box = bbox_of(kps)
        self.boxes = type("B", (), {"data": np.array(
            [[box[0], box[1], box[2], box[3], 0.90]], dtype=np.float32)})()
        self.keypoints = type("K", (), {"data": np.array(
            [list(k) for k in kps], dtype=np.float32).reshape(1, 17, 3)})()


class FakeEngine:
    def __init__(self, detect: str = "standing") -> None:
        self.detect = detect
        self.calls: list[dict] = []

    def predict(self, *, source, device, half, imgsz, conf, iou, max_det, verbose, workers=0):
        self.calls.append({"device": device, "half": half, "imgsz": imgsz,
                           "conf": conf, "iou": iou, "max_det": max_det})
        kps = horizontal_keypoints() if self.detect == "horizontal" else standing_keypoints()
        return [_FakeRaw(kps)]


class RecSink:
    """记录所有 finalize 过渡的过渡 sink；对外暴露兼容 _submitted_id 的协议。"""

    def __init__(self) -> None:
        self.items = []
        self._submitted_id = []

    def submit(self, transition) -> None:
        self.items.append(transition)
        self._submitted_id.append(transition.event_id)

    def close(self) -> None:
        return None


def _service(detect: str = "standing", *, frame_reader=None, sink=None) -> tuple[WorkerService, RecSink]:
    s = RecSink() if sink is None else sink
    svc = WorkerService(
        _make_config(),
        engine=FakeEngine(detect=detect),
        frame_reader=frame_reader or (lambda payload: np.zeros((480, 640, 3), dtype=np.uint8)),
        sink=s,
        device_name="Mock GPU",
    )
    return svc, s


def _hello() -> dict:
    return {"message_type": "HELLO", "correlation_id": "c1",
            "payload": {"parent_pid": 1234, "supported_schema_versions": [1]}}


def _infer_msg(frame_id: int, now_mono: int, now_unix: int, *, camera: str = "cam-1",
               session: str = "s1") -> dict:
    return {
        "message_type": "INFER_FRAME", "correlation_id": f"corr-{frame_id}",
        "worker_epoch": "eph",
        "payload": {"request_id": f"req-{frame_id}", "camera_id": camera,
                    "camera_session_id": session, "frame_id": frame_id,
                    "config_revision": "r", "observed_at_monotonic_ns": now_mono,
                    "observed_at_unix_ns": now_unix},
    }


# ---------------------------------------------------------------- 协议面
def test_hello_emits_starting_then_ready_with_golden_fields() -> None:
    svc, _ = _service()
    msgs = svc.handle_message(_hello())
    assert [m["message_type"] for m in msgs] == ["WORKER_STARTING", "WORKER_READY"]
    start = msgs[0]["payload"]
    assert start["selected_schema_version"] == 1
    assert start["worker_pid"] == os.getpid()
    assert start["parent_pid"] == 1234
    ready = msgs[1]["payload"]
    for field in ("worker_epoch", "worker_instance_id", "worker_pid", "device",
                  "device_name", "model_sha256", "precision"):
        assert field in ready, field
    assert ready["device"] == "cuda:0"
    assert ready["precision"] == "fp16"
    assert ready["device_name"] == "Mock GPU"


def test_ping_round_trip_pong() -> None:
    svc, _ = _service()
    out = svc.handle_message({
        "message_type": "PING", "correlation_id": "p1",
        "payload": {"ping_id": "pid-1", "parent_sent_ns": 100},
    })[0]
    assert out["message_type"] == "PONG"
    p = out["payload"]
    assert p["ping_id"] == "pid-1"
    assert p["parent_sent_ns"] == 100
    assert p["worker_received_ns"]
    assert p["worker_sent_ns"]


def test_register_and_unregister_camera() -> None:
    svc, _ = _service()
    reg = svc.handle_message({
        "message_type": "REGISTER_CAMERA", "correlation_id": "r1",
        "payload": {"camera_id": "cam-1", "camera_session_id": "s1", "camera_config_revision": "r"},
    })[0]
    assert reg["message_type"] == "CAMERA_REGISTERED"
    unreg = svc.handle_message({
        "message_type": "UNREGISTER_CAMERA", "correlation_id": "u1",
        "payload": {"camera_id": "cam-1", "camera_session_id": "s1"},
    })[0]
    assert unreg["message_type"] == "CAMERA_UNREGISTERED"
    assert unreg["payload"]["pending_transition_count"] == 0


def test_shutdown_emits_stopped() -> None:
    svc, _ = _service()
    out = svc.handle_message({
        "message_type": "SHUTDOWN", "correlation_id": "s1", "payload": {"reason": "down"},
    })[0]
    assert out["message_type"] == "STOPPED"
    assert out["payload"]["exit_code_intent"] == 0


# ---------------------------------------------------------------- 推理
def test_infer_frame_produces_result_with_tracks() -> None:
    svc, sink = _service()
    out = svc.handle_message(_infer_msg(1, now_mono=int(1e9), now_unix=int(2e9)))[0]
    assert out["message_type"] == "INFERENCE_RESULT"
    p = out["payload"]
    assert p["status"] == "ok"
    assert p["request_id"] == "req-1"
    assert p["source_frame_id"] == 1
    assert isinstance(p["tracks"], list)
    assert p["transition_event_ids"] == []  # 站立帧不应产生 transition
    assert sink.items == []
    # 引擎以显式 cuda 索引 + half 被调用
    assert svc._engine.calls[-1]["device"] == 0
    assert svc._engine.calls[-1]["half"] is True


def test_infer_rejects_when_no_frame_available() -> None:
    svc, _ = _service(frame_reader=lambda payload: None)
    out = svc.handle_message(_infer_msg(3, now_mono=int(1e9), now_unix=int(2e9)))[0]
    assert out["message_type"] == "FRAME_REJECTED"
    p = out["payload"]
    assert p["request_id"] == "req-3"
    assert p["retryable"] is True
    assert p["error_code"] == "TORN_FRAME"


def test_fall_flow_emits_fall_potential_and_fallen_transitions() -> None:
    svc, sink = _service(detect="horizontal")
    base_mono = int(1e9)
    base_unix = int(2e9)
    # 0.25s 步进；t 到达 3.5s(第 14 帧) 应确认 fallen
    for i in range(20):
        mono = base_mono + i * 250_000_000
        unix = base_unix + i * 250_000_000
        msgs = svc.handle_message(_infer_msg(i + 1, now_mono=mono, now_unix=unix))
        result = msgs[0]
        assert result["message_type"] == "INFERENCE_RESULT"
        assert result["payload"]["tracks"], f"第 {i} 帧应有确认轨迹"

    kinds = {tr.event_type for tr in sink.items}
    assert "fall_potential" in kinds, kinds
    assert "fall_detected" in kinds, kinds
    # 过渡语义标注为规则分而非概率
    for tr in sink.items:
        assert tr.score_semantics == "heuristic_rule_score_not_probability"