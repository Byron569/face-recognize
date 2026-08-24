"""阶段6 worker.service 离线单测：mock 引擎 + mock 帧读取，验证 pipe 协议面与整条推理流水线。

不依赖 GPU / 数据库 / 运行中的后端。协议消息形状对齐 tests/fixtures/fake_worker.py 黄金样本。
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import sys
import time
import types

import numpy as np

from ai_monitor_pose.config import FallTaskConfig
from ai_monitor_pose.worker import service as _service_mod
from ai_monitor_pose.worker.service import WorkerService, _resolve_model_file

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


class SeqEngine:
    """按调用次序依次返回关键点序列的假引擎（用于逐帧构造运动过程）。"""

    def __init__(self, seq) -> None:
        self.seq = list(seq)
        self.i = 0

    def predict(self, *, source, device, half, imgsz, conf, iou, max_det, verbose, workers=0):
        kps = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return [_FakeRaw(kps)]


def _shift_kps(kps, dy: float):
    """整体向下平移 dy 像素（图像 y 轴向下为正）。"""
    return tuple((x, y + dy, s) for (x, y, s) in kps)


def _bent_keypoints():
    """躯干约 27° 倾斜的弯腰姿态：非直立（>25°）、非水平躺倒（<55°）且 bbox 高宽比远大于 0.75。"""
    pts = [(0.0, 0.0, 0.0)] * 17
    pts[0] = (340.0, 250.0, 0.9)   # nose
    pts[5] = (360.0, 300.0, 0.9)   # LSH
    pts[6] = (380.0, 300.0, 0.9)   # RSH
    pts[11] = (310.0, 400.0, 0.9)  # LHIP
    pts[12] = (330.0, 400.0, 0.9)  # RHIP
    pts[13] = (310.0, 480.0, 0.9)  # LKNEE
    pts[14] = (330.0, 480.0, 0.9)  # RKNEE
    pts[15] = (305.0, 540.0, 0.9)  # LANK
    pts[16] = (325.0, 540.0, 0.9)  # RANK
    return tuple(pts)


class RecSink:
    """记录所有 finalize 过渡的过渡 sink；对外暴露兼容 _submitted_id/_submitted_count 的协议。"""

    def __init__(self) -> None:
        self.items = []
        self._submitted_id = []
        self._submitted_count = 0

    def submit(self, transition) -> None:
        self.items.append(transition)
        self._submitted_id.append(transition.event_id)
        self._submitted_count += 1

    def close(self) -> None:
        return None


def _service(detect: str = "standing", *, frame_reader=None, sink=None,
             engine=None) -> tuple[WorkerService, RecSink]:
    s = RecSink() if sink is None else sink
    svc = WorkerService(
        _make_config(),
        engine=engine if engine is not None else FakeEngine(detect=detect),
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


# ---------------------------------------------------------------- 算法语义修复
def test_prev_vertical_velocity_backfills_real_px_s() -> None:
    # 两帧：nose 下移 25px、dt=0.5s -> 轨迹状态回填真实垂直速度 50 px/s，
    # 而不是把 0~1 的 rule_score 当速度存回。
    drop_px = 25.0
    dt_ns = 500_000_000
    engine = SeqEngine([standing_keypoints(), _shift_kps(standing_keypoints(), drop_px)])
    svc, _ = _service(engine=engine)
    base_mono = int(1e9)
    base_unix = int(2e9)
    svc.handle_message(_infer_msg(1, now_mono=base_mono, now_unix=base_unix))
    svc.handle_message(
        _infer_msg(2, now_mono=base_mono + dt_ns, now_unix=base_unix + dt_ns))
    st = svc._track_states[("cam-1", "s1", 1)]
    expected_px_s = drop_px / (dt_ns / 1e9)
    # FakeRaw 把关键点转成 float32，容差需覆盖 float32 舍入（~1e-6 相对误差）
    assert abs(st.prev_vertical_velocity_px_s - expected_px_s) < 1e-3
    # 速度量纲必须是 px/s（远超 rule_score 的 0~1 区间），排除旧的语义污染
    assert st.prev_vertical_velocity_px_s > 1.0


def test_stable_body_height_ema_converges_when_upright_no_evidence() -> None:
    # 帧1 站立（bbox 高 384）首帧初始化 -> 帧2 直立无证据且人走近（bbox 高 464）：
    # stable_body_height 以 alpha=0.2 EMA 收敛到 384*0.8 + 464*0.2 = 400
    engine = SeqEngine([standing_keypoints(h=480.0), standing_keypoints(h=580.0)])
    svc, _ = _service(engine=engine)
    base_mono = int(1e9)
    base_unix = int(2e9)
    dt_ns = 250_000_000
    svc.handle_message(_infer_msg(1, now_mono=base_mono, now_unix=base_unix))
    svc.handle_message(
        _infer_msg(2, now_mono=base_mono + dt_ns, now_unix=base_unix + dt_ns))
    st = svc._track_states[("cam-1", "s1", 1)]
    h1 = 0.80 * 480.0
    h2 = 0.80 * 580.0
    assert abs(st.stable_body_height - (h1 * 0.8 + h2 * 0.2)) < 1e-3
    # 收敛方向：向新 bbox 高靠拢但不超过
    assert h1 < st.stable_body_height < h2


def test_stable_body_height_frozen_during_fall_evidence() -> None:
    # 帧1 水平躺倒（首帧初始化 72px）-> 帧2 直立站立但 fast_dynamic 触发 fall_evidence
    # （0.125s 内躯干 90°->0° 且头部速度巨大）：直立不足以放行，evidence 帧必须冻结基准。
    engine = SeqEngine([
        horizontal_keypoints(),
        standing_keypoints(),
        standing_keypoints(),
    ])
    svc, _ = _service(engine=engine)
    base_mono = int(1e9)
    base_unix = int(2e9)
    svc.handle_message(_infer_msg(1, now_mono=base_mono, now_unix=base_unix))
    svc.handle_message(
        _infer_msg(2, now_mono=base_mono + 125_000_000,
                   now_unix=base_unix + 125_000_000))
    st = svc._track_states[("cam-1", "s1", 1)]
    lb = bbox_of(horizontal_keypoints())
    lying_h = lb[3] - lb[1]  # 水平躺倒 bbox 高 = 393.6 - 321.6 = 72
    assert abs(lying_h - 72.0) < 1e-3
    assert abs(st.stable_body_height - lying_h) < 1e-3  # evidence 帧不更新
    # 帧3 恢复直立无证据 -> EMA 恢复更新
    svc.handle_message(
        _infer_msg(3, now_mono=base_mono + 375_000_000,
                   now_unix=base_unix + 375_000_000))
    standing_h = 0.80 * 480.0
    assert abs(st.stable_body_height - (lying_h * 0.8 + standing_h * 0.2)) < 1e-3


def test_standing_head_y_kept_while_bent_not_upright() -> None:
    # 弯腰帧（躯干 ~27°，非直立且无跌倒证据）：头部基准必须保持"最后站立头位"，
    # 不得追弯腰后的头位，否则头部下降量被弯腰过程稀释。
    engine = SeqEngine([standing_keypoints(), _bent_keypoints()])
    svc, _ = _service(engine=engine)
    base_mono = int(1e9)
    base_unix = int(2e9)
    svc.handle_message(_infer_msg(1, now_mono=base_mono, now_unix=base_unix))
    svc.handle_message(
        _infer_msg(2, now_mono=base_mono + 250_000_000,
                   now_unix=base_unix + 250_000_000))
    st = svc._track_states[("cam-1", "s1", 1)]
    stand_nose_y = standing_keypoints()[0][1]
    assert abs(st.standing_head_y - stand_nose_y) < 1e-3
    # 非直立帧同样不更新身高基准
    assert abs(st.stable_body_height - 0.80 * 480.0) < 1e-3


# ---------------------------------------------------------------- 指标埋点
def test_infer_metrics_queue_wait_frame_copy_end_to_end() -> None:
    # 构造父端 offer 时刻为 200ms 前：queue_wait/frame_copy/end_to_end 必须做真，
    # 且端到端覆盖排队+读帧+推理+后处理（>= 推理段耗时）。
    svc, _ = _service()
    past_unix = time.time_ns() - 200_000_000
    out = svc.handle_message(_infer_msg(1, now_mono=int(1e9), now_unix=past_unix))[0]
    assert out["message_type"] == "INFERENCE_RESULT"
    p = out["payload"]
    assert p["queue_wait_ms"] > 0.0
    assert p["frame_copy_ms"] >= 0.0
    assert p["end_to_end_ms"] >= p["gpu_inference_ms"]
    assert p["end_to_end_ms"] >= p["queue_wait_ms"]


# ---------------------------------------------------------------- TensorRT engine 优选
def test_resolve_model_file_pt_only(tmp_path) -> None:
    # 只有 .pt（engine 不存在）→ 原样返回 .pt
    pt = tmp_path / "yolov8n-pose.pt"
    pt.write_bytes(b"pt-weights")
    assert _resolve_model_file(str(pt)) == (str(pt), False)


def test_resolve_model_file_engine_mtime_newer_or_equal_wins(tmp_path) -> None:
    pt = tmp_path / "yolov8n-pose.pt"
    eng = tmp_path / "yolov8n-pose.engine"
    pt.write_bytes(b"pt-weights")
    eng.write_bytes(b"engine-bytes")
    # mtime 相等 → engine 可用（>= 语义，防时间戳精度抖动误判）
    os.utime(pt, (2000, 2000))
    os.utime(eng, (2000, 2000))
    assert _resolve_model_file(str(pt)) == (str(eng), True)
    # engine 严格更新 → engine 可用
    os.utime(eng, (3000, 3000))
    assert _resolve_model_file(str(pt)) == (str(eng), True)


def test_resolve_model_file_stale_engine_falls_back_to_pt(tmp_path) -> None:
    # engine 比 .pt 旧（.pt 更新后 engine 陈旧）→ 回退 .pt
    pt = tmp_path / "yolov8n-pose.pt"
    eng = tmp_path / "yolov8n-pose.engine"
    pt.write_bytes(b"pt-weights")
    eng.write_bytes(b"engine-bytes")
    os.utime(eng, (1000, 1000))
    os.utime(pt, (2000, 2000))
    assert _resolve_model_file(str(pt)) == (str(pt), False)


def test_resolve_model_file_engine_stat_error_falls_back_to_pt(tmp_path, monkeypatch) -> None:
    # engine 路径 stat 抛 OSError（如权限拒绝）→ 按不存在处理回退 .pt，不向外抛
    pt = tmp_path / "yolov8n-pose.pt"
    eng = tmp_path / "yolov8n-pose.engine"
    pt.write_bytes(b"pt-weights")
    eng.write_bytes(b"engine-bytes")
    real_stat = pathlib.Path.stat

    def _stat_raises_for_engine(self, **kwargs):
        if self.suffix == ".engine":
            raise PermissionError(13, "permission denied")
        return real_stat(self, **kwargs)

    monkeypatch.setattr(pathlib.Path, "stat", _stat_raises_for_engine)
    assert _resolve_model_file(str(pt)) == (str(pt), False)


def test_resolve_model_file_missing_pt_falls_back(tmp_path) -> None:
    # .pt 缺失（pt.stat() 抛 FileNotFoundError, 属 OSError）→ 回退 .pt 路径
    eng = tmp_path / "yolov8n-pose.engine"
    eng.write_bytes(b"engine-bytes")
    pt = tmp_path / "yolov8n-pose.pt"
    assert _resolve_model_file(str(pt)) == (str(pt), False)


class _FakeTrtTorchCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_name(index: int) -> str:
        return "Mock TensorRT GPU"


def _fake_torch_module():
    """绕过 torch 导入门槛：assert_gpu_ready / check_device_index 均可通过的假 torch。"""
    mod = types.ModuleType("torch")
    mod.version = types.SimpleNamespace(cuda="12.4-fake")
    mod.cuda = _FakeTrtTorchCuda
    return mod


class _FakeLoadableModel:
    """记录 .to/.half/predict 调用的假模型（YOLO 构造产物即模型本身）。"""

    def __init__(self, path: str) -> None:
        self.loaded_path = path
        self.to_devices: list[str] = []
        self.half_calls = 0
        self.predict_calls: list[dict] = []

    def to(self, device):
        self.to_devices.append(device)
        return self

    def half(self):
        self.half_calls += 1
        return self

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return []


def _trt_config(model_path: str, sha_path: str) -> FallTaskConfig:
    """model.path 指向 tmp_path 真实 .pt 的合法配置（engine 接线测试用）。"""
    import tempfile
    base = tempfile.mkdtemp()
    return FallTaskConfig.from_mapping({
        "enabled": False,
        "mode": "shadow",
        "gpu": {"required": True, "device": "cuda:0", "precision": "fp16",
                "allow_cpu_fallback": False},
        "worker": {"python": "C:/py.exe", "module": "ai_monitor_pose.worker"},
        "model": {"path": model_path, "sha256_file": sha_path,
                  "allow_download": False,
                  "imgsz": 640, "confidence": 0.35, "iou": 0.45, "max_detections": 50},
        "runtime": {"capacity_manifest_path": os.path.join(base, "capacity-cuda0.json")},
    })


def _make_model_pair(tmp_path):
    """构造真实 .pt + 匹配的 sha256 sidecar；返回 (pt_path, digest)。"""
    pt = tmp_path / "yolov8n-pose.pt"
    pt.write_bytes(b"fake-pt-weights")
    digest = hashlib.sha256(pt.read_bytes()).hexdigest()
    (tmp_path / "yolov8n-pose.pt.sha256").write_text(digest, encoding="utf-8")
    return str(pt), digest


def test_ensure_engine_load_failure_falls_back_to_pt(tmp_path, monkeypatch, capsys) -> None:
    # engine 文件存在但加载抛异常（内容非法）：必须回退 .pt 完整加载路径
    # （.to/.half/预热照旧），_engine_is_trt 保持 False，绝不阻断启动。
    pt_path, digest = _make_model_pair(tmp_path)
    eng = tmp_path / "yolov8n-pose.engine"
    eng.write_bytes(b"garbage-not-a-real-engine")
    load_calls: list[str] = []

    class FakeYOLO(_FakeLoadableModel):
        def __init__(self, path):
            if str(path).endswith(".engine"):
                load_calls.append(str(path))
                raise RuntimeError("invalid engine payload")
            super().__init__(str(path))
            load_calls.append(str(path))

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setattr(_service_mod, "_resolve_model_file", lambda p: (str(eng), True))

    svc = WorkerService(
        _trt_config(pt_path, str(pt_path) + ".sha256"),
        sink=RecSink(),
        device_name="Mock GPU",
    )
    model = svc._ensure_engine()

    assert load_calls == [str(eng), pt_path]  # 先尝试 engine，失败后回退 .pt
    assert model is svc._engine
    assert svc._engine_is_trt is False
    assert svc._engine.loaded_path == pt_path
    assert svc._engine.to_devices == ["cuda:0"]  # .pt 路径 .to 照旧
    assert svc._engine.half_calls == 1           # precision=fp16 → .half() 照旧
    assert svc._engine.predict_calls             # 预热 predict 已执行
    assert svc._engine.predict_calls[0]["half"] is True  # .pt 语义预热
    assert svc.model_sha256 == digest            # 血缘 sha 仍是 .pt 的
    err = capsys.readouterr().err
    assert "engine load failed" in err and "fallback to pt" in err


def test_ensure_engine_prefers_trt_engine_and_skips_to_half(tmp_path, monkeypatch, capsys) -> None:
    # engine 存在且新于 .pt（真实 _resolve_model_file 选中）：优先加载 engine，
    # 跳过 .to/.half，预热 half=False；.pt 的 sha 仍是血缘凭证。
    pt_path, digest = _make_model_pair(tmp_path)
    eng = tmp_path / "yolov8n-pose.engine"
    eng.write_bytes(b"fake-trt-engine")
    os.utime(pt_path, (1000, 1000))
    os.utime(eng, (2000, 2000))
    load_calls: list[str] = []

    class FakeYOLO(_FakeLoadableModel):
        def __init__(self, path):
            load_calls.append(str(path))
            super().__init__(str(path))

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    svc = WorkerService(
        _trt_config(pt_path, str(pt_path) + ".sha256"),
        sink=RecSink(),
        device_name="Mock GPU",
    )
    model = svc._ensure_engine()

    assert load_calls == [str(eng)]  # 只加载 engine，.pt 未加载
    assert model is svc._engine
    assert svc._engine_is_trt is True
    assert svc._engine.loaded_path == str(eng)
    assert svc._engine.to_devices == []  # TensorRT 引擎跳过 .to
    assert svc._engine.half_calls == 0   # 精度已烘焙，跳过 .half
    warm = svc._engine.predict_calls[0]  # 预热 predict 已执行
    assert warm["half"] is False         # engine 分支预热不传 half
    assert warm["imgsz"] == 640
    assert warm["device"] == 0
    assert svc.model_sha256 == digest    # 血缘 sha 仍是 .pt 的
    assert svc.device_name == "Mock TensorRT GPU"
    assert "TensorRT engine: yolov8n-pose.engine" in capsys.readouterr().err


def test_infer_half_suppressed_when_engine_is_trt() -> None:
    # run_inference 的 half 实参：trt 后端时不再传 half=True（引擎已 FP16）；
    # 非 trt 后端保持原语义（fp16 → half=True）。
    svc, _ = _service()
    svc._engine_is_trt = True
    svc.handle_message(_infer_msg(1, now_mono=int(1e9), now_unix=int(2e9)))
    assert svc._engine.calls[-1]["half"] is False

    svc2, _ = _service()
    svc2.handle_message(_infer_msg(2, now_mono=int(1e9), now_unix=int(2e9)))
    assert svc2._engine.calls[-1]["half"] is True