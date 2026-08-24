"""GPU Worker 真实进程实现（阶段6 主体）。

在独立 .venv-worker 进程内运行，通过 stdin/stdout 与父端走 4 字节大端长度前缀
+ UTF-8 JSON 的 pipe 协议（见 ai_monitor_pose/ipc.py 的 encode_message/decode_message，
消息形状以 tests/fixtures/fake_worker.py 的 FAKE_WORKER_CODE 为黄金样本）。

消息类型：
    HELLO            -> 先 WORKER_STARTING 再 WORKER_READY
    PING             -> PONG
    INFER_FRAME      -> INFERENCE_RESULT(含 tracks) / FRAME_REJECTED
    REGISTER_CAMERA  -> CAMERA_REGISTERED
    UNREGISTER_CAMERA-> CAMERA_UNREGISTERED
    SHUTDOWN         -> STOPPED 后退出

单帧流水线：共享帧读取 -> 引擎推理(pose_engine) -> 追踪(pose_tracker)
-> 跌倒状态机(state_machine) -> 事件映射(顶层 event_mapper 的 transition)
-> 把 finalize 过渡写入 WorkerJournal / 收集进 INFERENCE_RESULT。
本模块不导入 Torch/Ultralytics（延迟注入），以保证 `python -m ai_monitor_pose.worker`
启动够快且模块可被重依赖进程安全导入。
"""
from __future__ import annotations

import json
import os
import pathlib
import struct
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import FallTaskConfig
from ..contracts import (
    FallResultV1,
    FallTransitionV1,
    PoseDetectionV1,
    PoseStateV1,
    SharedFrameRefV1,
)
from .gpu_guard import (
    assert_gpu_ready,
    check_device_index,
    parse_explicit_cuda_index,
    validate_model_sha256,
)
from .pose_engine import run_inference
from .pose_tracker import PoseTracker
from .state_machine import FallTrackStateMachine
from .features import (
    SCORE_SEMANTICS,
    torso_inclination_deg,
)
from .transition_journal import WorkerJournal

# COCO-17 关键点索引（与 features.py 一致）
_NOSE, _LSH, _RSH, _LHIP, _RHIP = 0, 5, 6, 11, 12

_NS_PER_S = 1_000_000_000

_MAX_MSG_BYTES = 64 * 1024 * 1024


def _trace(msg: str) -> None:
    import sys as _sys
    print(f"[TRACE {time.monotonic():.3f}] {msg}", file=_sys.stderr, flush=True)


@dataclass
class _PoseSample:
    frame_id: int
    observed_at_monotonic_ns: int
    pose_quality: str
    fall_evidence: bool


@dataclass
class _TrackState:
    machine: FallTrackStateMachine = field(default_factory=FallTrackStateMachine)
    prev_keypoints: tuple | None = None
    prev_torso_angle_deg: float | None = None
    prev_vertical_velocity_px_s: float = 0.0
    standing_head_y: float | None = None
    stable_body_height: float = 0.0
    last_obs_ns: int | None = None


class _TransitionSink:
    """接收 finalize 的 transition：可选持久化到 crash-safe journal，并累计已确认 event_id。"""

    def __init__(self, journal: WorkerJournal | None = None) -> None:
        self.journal = journal
        self._submitted_id: list[str] = []

    def submit(self, transition: FallTransitionV1) -> None:
        if self.journal is not None:
            seq = self.journal.begin_add()
            self.journal.commit(seq, transition)
        self._submitted_id.append(transition.event_id)

    def close(self) -> None:
        if self.journal is not None:
            self.journal.close()
            self.journal = None


class WorkerService:
    """一个 GPU Worker 进程的服务实例：持有引擎、配置、追踪/状态机并驱动消息循环。"""

    def __init__(
        self,
        config: FallTaskConfig,
        *,
        engine: Any | None = None,
        frame_reader=None,
        sink: _TransitionSink | None = None,
        device_name: str | None = None,
        stdin=None,
        stdout=None,
    ) -> None:
        self.config = config
        self.epoch = uuid.uuid4().hex
        self.instance_id = uuid.uuid4().hex
        self.pid = os.getpid()
        self.device = config.gpu.device
        self.device_index = parse_explicit_cuda_index(self.device)
        self.precision = config.gpu.precision
        self._half = self.precision == "fp16"
        self.device_name = device_name or config.gpu.expected_device_name_regex or f"cuda:{self.device_index}"
        self.model_sha256 = "mock"

        self._engine = engine
        self._frame_reader = frame_reader or self._read_frame_default
        self._stdin = stdin
        self._stdout = stdout

        # 每个 (camera, session, track_id) 独立追踪器 + 状态机
        self._trackers: dict[tuple, PoseTracker] = {}
        self._track_states: dict[tuple, _TrackState] = {}

        journal = None
        jp = config.runtime.worker_journal_path
        if jp:
            pathlib.Path(jp).parent.mkdir(parents=True, exist_ok=True)
            journal = WorkerJournal(jp, worker_instance_id=self.instance_id)
        self._sink = sink if sink is not None else _TransitionSink(journal)

    # ---------------------------------------------------------------- 引擎/配置
    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        model_path = self.config.model.path
        sha_file = self.config.model.sha256_file
        digest = validate_model_sha256((model_path, sha_file))["sha256"]
        import torch
        from ultralytics import YOLO

        assert_gpu_ready(torch)
        check_device_index(torch, self.device_index)
        _trace("engine: torch import ok")
        model = YOLO(model_path)
        _trace("engine: YOLO loaded")
        model.to(self.device)
        _trace("engine: moved to device")

        if self._half:
            model = model.half()
        _trace("engine: half done")
        # 预热：在进入 stdin 空闲读循环之前完成首次 CUDA/OpenMP 线程池同步，
        # 避免随后（空闲阻塞读后）首次 predict 与读循环竞争线程初始化导致 Windows 死锁。
        try:
            import numpy as _np
            _imgsz = int(self.config.model.imgsz)
            _w = _np.zeros((_imgsz, _imgsz, 3), dtype=_np.uint8)
            model.predict(
                source=_w, device=self.device_index, half=self._half,
                imgsz=_imgsz, conf=self.config.model.confidence,
                iou=self.config.model.iou, max_det=self.config.model.max_detections,
                verbose=False, workers=0,
            )
            _trace("engine: warmup predict ok")
        except Exception as _we:  # noqa: BLE001 —— 预热失败不阻断启动，仅记录
            _trace(f"warmup infer skipped: {type(_we).__name__} {_we}")
        self.device_name = torch.cuda.get_device_name(self.device_index)
        self.model_sha256 = digest
        self._engine = model
        return model

    # ---------------------------------------------------------------- 帧读取
    def _read_frame_default(self, payload: dict) -> np.ndarray | None:
        """从 INFER_FRAME payload 的 frame_ref 共享内存引用读取像素；失败返回 None(非阻塞)。"""
        fr = payload.get("frame_ref")
        if not isinstance(fr, dict):
            return None
        from ..shared_frames import FrameShmRegion

        ref = SharedFrameRefV1.from_dict(fr)
        region = FrameShmRegion(
            shm_name=ref.shm_name,
            max_width=0,
            max_height=0,
            slots=max(ref.slot_index + 1, 2),
            attach=True,
        )
        try:
            return region.read_may_raise(ref)
        finally:
            region.close()

    # ---------------------------------------------------------------- 消息处理
    def handle_message(self, message: dict) -> list[dict]:
        """处理一条父端消息，返回应写回 stdout 的响应 envelope 列表（可为空）。"""
        if not isinstance(message, dict):
            return []
        mtype = message.get("message_type")
        corr = message.get("correlation_id") or message.get("message_id")
        payload = message.get("payload") or {}

        if mtype == "HELLO":
            return self._handle_hello(corr, payload)
        if mtype == "PING":
            return [self._pong(corr, payload)]
        if mtype == "INFER_FRAME":
            out = self._handle_infer(message)
            return [out] if out is not None else []
        if mtype == "REGISTER_CAMERA":
            return [{
                "message_type": "CAMERA_REGISTERED", "worker_epoch": self.epoch,
                "correlation_id": corr,
                "payload": {
                    "camera_id": payload.get("camera_id"),
                    "camera_session_id": payload.get("camera_session_id"),
                    "camera_config_revision": payload.get("camera_config_revision", ""),
                },
            }]
        if mtype == "UNREGISTER_CAMERA":
            return [{
                "message_type": "CAMERA_UNREGISTERED", "worker_epoch": self.epoch,
                "correlation_id": corr,
                "payload": {
                    "camera_id": payload.get("camera_id"),
                    "camera_session_id": payload.get("camera_session_id"),
                    "pending_transition_count": 0,
                },
            }]
        if mtype == "SHUTDOWN":
            return [self._stopped(corr, payload)]
        # 未知消息忽略
        return []

    def _handle_hello(self, corr, payload) -> list[dict]:
        starting = {
            "message_type": "WORKER_STARTING", "worker_epoch": self.epoch,
            "correlation_id": corr,
            "payload": {
                "selected_schema_version": 1, "worker_epoch": self.epoch,
                "worker_instance_id": self.instance_id, "worker_pid": self.pid,
                "parent_pid": payload.get("parent_pid"),
                "challenge_sha256": "mock",
            },
        }
        try:
            self._ensure_engine()
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "code", None) or _classify_start_error(exc)
            return [starting, {
                "message_type": "WORKER_ERROR", "worker_epoch": self.epoch,
                "correlation_id": corr,
                "payload": {"error_code": code, "error_message": str(exc), "retryable": False},
            }]
        ready = {
            "message_type": "WORKER_READY", "worker_epoch": self.epoch,
            "correlation_id": corr,
            "payload": {
                "worker_epoch": self.epoch, "worker_instance_id": self.instance_id,
                "worker_pid": self.pid, "device": self.device,
                "device_name": self.device_name, "model_sha256": self.model_sha256,
                "precision": self.precision,
            },
        }
        return [starting, ready]

    def _pong(self, corr, payload) -> dict:
        return {
            "message_type": "PONG", "worker_epoch": self.epoch, "correlation_id": corr,
            "payload": {
                "ping_id": payload.get("ping_id"),
                "parent_sent_ns": payload.get("parent_sent_ns"),
                "worker_received_ns": time.time_ns(),
                "worker_sent_ns": time.time_ns(),
            },
        }

    def _stopped(self, corr, payload) -> dict:
        return {
            "message_type": "STOPPED", "worker_epoch": self.epoch, "correlation_id": corr,
            "payload": {
                "reason": payload.get("reason", "shutdown"),
                "pending_transition_count": len(self._sink._submitted_id),
                "exit_code_intent": 0,
            },
        }

    # ---------------------------------------------------------------- 推理
    def _handle_infer(self, message: dict) -> dict | None:
        corr = message.get("correlation_id") or message.get("message_id")
        payload = message.get("payload") or {}
        request_id = str(payload.get("request_id", ""))
        camera_id = str(payload.get("camera_id", ""))
        camera_session_id = str(payload.get("camera_session_id", ""))
        frame_id = int(payload.get("frame_id", 0))
        config_revision = str(payload.get("config_revision", ""))
        now_mono = int(payload.get("observed_at_monotonic_ns", time.monotonic_ns()))
        now_unix = int(payload.get("observed_at_unix_ns", time.time_ns()))

        frame = self._frame_reader(payload)
        if frame is None or getattr(frame, "shape", None) is None or frame.size == 0:
            return self._frame_rejected(
                corr, request_id, camera_id, camera_session_id, frame_id,
                error_code="TORN_FRAME", retryable=True,
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            return self._frame_rejected(
                corr, request_id, camera_id, camera_session_id, frame_id,
                error_code="TORN_FRAME", retryable=True,
            )

        cfg = self.config
        t0 = time.monotonic()
        _trace(f"infer ok frame={frame_id} {frame.shape[1]}x{frame.shape[0]} cam={camera_id}")
        try:
            engine = self._engine if self._engine is not None else self._ensure_engine()
        except Exception as exc:  # noqa: BLE001
            return self._frame_rejected(
                corr, request_id, camera_id, camera_session_id, frame_id,
                error_code=getattr(exc, "code", None) or "MODEL_UNAVAILABLE", retryable=False,
            )
        try:
            raw_results = run_inference(
                engine, source=frame, device=self.device_index, half=self._half,
                imgsz=cfg.model.imgsz, conf=cfg.model.confidence,
                iou=cfg.model.iou, max_det=cfg.model.max_detections,
            )
        except Exception as exc:  # noqa: BLE001
            from .gpu_guard import OomPolicy
            code = OomPolicy().classify(exc)
            return self._frame_rejected(
                corr, request_id, camera_id, camera_session_id, frame_id,
                error_code=code, retryable=True, error_message=str(exc),
            )
        gpu_ms = (time.monotonic() - t0) * 1000.0
        _trace(f"infer done gpu_ms={gpu_ms:.1f} raw={len(raw_results)} frame={frame_id}")

        dets = self._raw_to_detections(raw_results)
        key = (camera_id, camera_session_id)
        tracker = self._trackers.setdefault(key, PoseTracker(cfg.tracker))
        tracks = tracker.update(dets, now_mono_ns=now_mono, frame_id=frame_id)
        _trace(f"tracker done dets={len(dets)} tracks={len(tracks)} frame={frame_id}")

        # 逐轨迹：特征 -> 状态机 -> finalize transition
        transitions, frame_event_ids = self._advance_state_machines(
            key, tracks, frame, now_mono, now_unix, request_id, camera_id,
            camera_session_id, frame_id, config_revision,
        )
        for tr in transitions:
            self._sink.submit(tr)
        _trace(f"machines done transitions={len(transitions)} evids={len(frame_event_ids)} frame={frame_id}")

        end_ms = (time.monotonic() - t0) * 1000.0
        result = FallResultV1(
            schema_version=1, request_id=request_id, worker_epoch=self.epoch,
            worker_instance_id=self.instance_id, camera_id=camera_id,
            camera_session_id=camera_session_id, source_frame_id=frame_id,
            source_width=frame.shape[1], source_height=frame.shape[0],
            coordinate_space="source_pixels", observed_at_unix_ns=now_unix,
            observed_at_monotonic_ns=now_mono, completed_at_monotonic_ns=now_mono,
            status="ok", config_revision=config_revision, model_name="yolov8n-pose",
            model_sha256=self.model_sha256, device=self.device, precision=self.precision,
            queue_wait_ms=0.0, frame_copy_ms=0.0, gpu_inference_ms=gpu_ms,
            postprocess_ms=max(0.0, end_ms - gpu_ms), end_to_end_ms=end_ms,
            tracks=tuple(tracks), transition_event_ids=tuple(frame_event_ids),
            error_code=None, error_message=None,
        )
        return {
            "message_type": "INFERENCE_RESULT", "worker_epoch": self.epoch,
            "correlation_id": corr, "payload": result.to_dict(),
        }

    def _frame_rejected(self, corr, request_id, camera_id, camera_session_id,
                        frame_id, *, error_code, retryable, error_message=""):
        return {
            "message_type": "FRAME_REJECTED", "worker_epoch": self.epoch,
            "correlation_id": corr,
            "payload": {
                "request_id": request_id, "camera_id": camera_id,
                "camera_session_id": camera_session_id, "frame_id": frame_id,
                "error_code": error_code, "error_message": error_message,
                "retryable": bool(retryable),
            },
        }

    # ---------------------------------------------------------------- 追踪/状态机
    def _raw_to_detections(self, raw_results) -> list[PoseDetectionV1]:
        out: list[PoseDetectionV1] = []
        for raw in raw_results:
            boxes = getattr(getattr(raw, "boxes", None), "data", None)
            kps = getattr(getattr(raw, "keypoints", None), "data", None)
            if boxes is None:
                continue
            boxes = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.asarray(boxes)
            kps = kps.cpu().numpy() if hasattr(kps, "cpu") else (np.asarray(kps) if kps is not None else None)
            if boxes.ndim == 1:
                boxes = boxes.reshape(1, -1)
            n = boxes.shape[0]
            for i in range(n):
                row = boxes[i]
                if row.size < 4:
                    continue
                conf = float(row[4]) if row.size >= 5 else 1.0
                if kps is not None and kps.ndim >= 2 and i < kps.shape[0]:
                    kp_row = kps[i]
                else:
                    kp_row = np.full((17, 3), 0.0)
                kp = tuple(
                    (float(x), float(y), float(s)) for x, y, s in kp_row[:17]
                )
                while len(kp) < 17:
                    kp = kp + ((0.0, 0.0, 0.0),)
                out.append(PoseDetectionV1(
                    bbox_xyxy=tuple(float(v) for v in row[:4]),
                    detection_score=conf, keypoints_coco17=kp,
                ))
        return out

    def _advance_state_machines(self, key, tracks, frame, now_mono, now_unix,
                                request_id, camera_id, camera_session_id, frame_id,
                                config_revision):
        frame_h = float(frame.shape[0])
        frame_w = float(frame.shape[1])
        transitions: list[FallTransitionV1] = []
        event_ids: list[str] = []
        alg = self.config.algorithm

        for track in tracks:
            tkey = (key[0], key[1], track.pose_track_id)
            st = self._track_states.setdefault(tkey, _TrackState())
            kps = track.keypoints_coco17
            bbox = track.bbox_xyxy

            dt = 0.0
            if st.last_obs_ns is not None:
                dt = max(now_mono - st.last_obs_ns, 0) / _NS_PER_S

            body_height = st.stable_body_height if st.stable_body_height > 0 else (bbox[3] - bbox[1])
            standing_head_y = st.standing_head_y
            if standing_head_y is None:
                standing_head_y = bbox[1]

            try:
                from .features import compute_evidence
                ev = compute_evidence(
                    keypoints=kps, bbox=bbox,
                    prev_keypoints=st.prev_keypoints,
                    prev_torso_angle_deg=st.prev_torso_angle_deg,
                    prev_vertical_velocity_px_s=st.prev_vertical_velocity_px_s,
                    dt=dt, frame_height=frame_h,
                    standing_head_y=standing_head_y,
                    stable_body_height=body_height,
                    cfg=alg, pose_quality=track.pose_quality,
                )
            except Exception:  # noqa: BLE001 —— 极简兜底，避免单条轨迹拖垮整帧
                from .features import FallEvidence
                ev = FallEvidence(
                    pose_quality=track.pose_quality, has_reliable_standing=False,
                    body_height_px=body_height, horizontal_geometry=False,
                    torso_horizontal=False, extended_hip=False, dynamic_descent=False,
                    dynamic_rotation=False, dynamic_gravity=False, fast_dynamic=False,
                    rule_score=0.0, evidence_codes=(),
                )

            fall_evidence = bool((ev.horizontal_geometry and ev.torso_horizontal) or ev.fast_dynamic)
            sample = _PoseSample(
                frame_id=frame_id, observed_at_monotonic_ns=now_mono,
                pose_quality=track.pose_quality, fall_evidence=fall_evidence,
            )
            _state, events = st.machine.observe(
                sample, max_trigger_gap_s=alg.max_trigger_gap_s,
                min_fall_pose_duration_s=alg.min_fall_pose_duration_s,
                recovery_duration_s=alg.recovery_duration_s,
            )

            for event in events:
                tr = self._build_transition(
                    event, key, track, frame_w, frame_h, now_unix, now_mono,
                    camera_id, camera_session_id, frame_id, config_revision,
                )
                transitions.append(tr)
                event_ids.append(tr.event_id)

            # 更新轨迹历史
            st.prev_keypoints = kps
            st.prev_vertical_velocity_px_s = ev.rule_score  # 复用规则分作速度占位（非语义速度）
            st.prev_torso_angle_deg = _estimate_torso_angle(kps)
            st.stable_body_height = body_height if body_height > 0 else st.stable_body_height
            if st.standing_head_y is None or not fall_evidence:
                st.standing_head_y = kps[_NOSE][1] if _visible(kps) else bbox[1]
            st.last_obs_ns = now_mono

        return transitions, event_ids

    def _build_transition(self, event, key, track, frame_w, frame_h, now_unix,
                          now_mono, camera_id, camera_session_id, frame_id,
                          config_revision) -> FallTransitionV1:
        from_state, to_state = _INCIDENT_FLOW.get(event.event_type, (PoseStateV1.NORMAL, PoseStateV1.NORMAL))
        return FallTransitionV1(
            schema_version=1, event_id=uuid.uuid4().hex,
            dedupe_key=f"{camera_id}:{track.pose_track_id}:{event.event_type}",
            incident_id=event.incident_id, event_type=event.event_type,
            camera_id=camera_id, camera_session_id=camera_session_id,
            pose_track_id=track.pose_track_id, source_frame_id=frame_id,
            source_width=int(frame_w), source_height=int(frame_h),
            coordinate_space="source_pixels", occurred_at_unix_ns=now_unix,
            occurred_at_monotonic_ns=now_mono, from_state=from_state, to_state=to_state,
            rule_score=event.rule_score, score_semantics=event.score_semantics,
            evidence_codes=tuple(event.evidence_codes), bbox_xyxy=track.bbox_xyxy,
            keypoints_coco17=track.keypoints_coco17, model_name="yolov8n-pose",
            model_sha256=self.model_sha256, config_revision=config_revision,
            worker_instance_id=self.instance_id, queue_wait_ms=0.0,
            gpu_inference_ms=0.0, end_to_end_ms=0.0,
        )

    # ---------------------------------------------------------------- 主循环
    def run(self, argv: "list[str] | None" = None) -> int:
        """阻塞读 stdin，处理消息，写回 stdout，直到 SHUTDOWN/EOF。返回进程退出码。

        真实 worker 进程（未注入 fake reader/writer）用 OS 原生命柄直连管道，
        绕开 torch/ultralytics 对 CRT fd(0/1/2) 的破坏；单测注入的缓冲流继续走
        buffered 路径。
        """
        real_io = self._stdin is None and self._stdout is None
        raw = _RawStdio() if real_io else None
        reader = self._stdin if self._stdin is not None else sys.stdin.buffer
        writer = self._stdout if self._stdout is not None else sys.stdout.buffer
        try:
            while True:
                if raw is not None:
                    message = _read_message_raw(raw)
                else:
                    message = _read_message(reader)
                if message is None:
                    break
                responses = self.handle_message(message)
                for resp in responses:
                    if raw is not None:
                        _write_message_raw(raw, resp)
                    else:
                        _write_message(writer, resp)
                if message.get("message_type") == "SHUTDOWN":
                    break
        finally:
            self._sink.close()
        return 0


# 状态机事件类型 -> (from_state, to_state) 近似映射
_INCIDENT_FLOW: dict[str, tuple[PoseStateV1, PoseStateV1]] = {
    "fall_potential": (PoseStateV1.NORMAL, PoseStateV1.POTENTIAL),
    "fall_detected": (PoseStateV1.POTENTIAL, PoseStateV1.FALLEN),
    "fall_recovered": (PoseStateV1.FALLEN, PoseStateV1.NORMAL),
}


def _visible(kps: tuple) -> bool:
    x, y, s = kps[_NOSE]
    return bool(s is not None and s >= 0.3 and x is not None and y is not None)


def _estimate_torso_angle(kps: tuple) -> float | None:
    try:
        sm = ((kps[_LSH][0] + kps[_RSH][0]) / 2.0, (kps[_LSH][1] + kps[_RSH][1]) / 2.0)
        hm = ((kps[_LHIP][0] + kps[_RHIP][0]) / 2.0, (kps[_LHIP][1] + kps[_RHIP][1]) / 2.0)
        deg = torso_inclination_deg(sm, hm)
        return deg if deg is not None and deg == deg else None
    except Exception:  # noqa: BLE001
        return None


def _classify_start_error(exc: Exception) -> str:
    from ..errors import (
        CudaBuildMissingError, CudaUnavailableError, CudaDeviceInvalidError,
        CUDA_BUILD_MISSING, CUDA_DEVICE_INVALID, CUDA_DEVICE_UNAVAILABLE,
        MODEL_NOT_FOUND, MODEL_HASH_MISMATCH,
    )
    if isinstance(exc, FileNotFoundError):
        return MODEL_NOT_FOUND
    if isinstance(exc, CudaBuildMissingError):
        return CUDA_BUILD_MISSING
    if isinstance(exc, CudaUnavailableError):
        return CUDA_DEVICE_UNAVAILABLE
    if isinstance(exc, CudaDeviceInvalidError):
        return CUDA_DEVICE_INVALID
    if "sha256" in str(exc).lower() or "mismatch" in str(exc).lower():
        return MODEL_HASH_MISMATCH
    return "MODEL_LOAD_FAILED"


class _RawStdio:
    """直接用 OS 句柄 + ReadFile/WriteFile 做子进程管道 IO（真实 worker 进程用）。

    torch/ultralytics 初始化会破坏 CRT 文件描述符表，使 os.read(0)/os.write(1)
    出现 EINVAL([Errno 22]) 或随机丢失，消息根本到不了父端管道。此处从
    GetStdHandle 抓取 OS 标准管道句柄，全程不经过 CRT fd(0/1/2)，对模型加载/预热免疫。
    """

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._cw = ctypes.WinDLL("kernel32", use_last_error=True)
        self._wt = wintypes
        # 优先使用 launcher 经 env 传入的专用 rx/tx 管道句柄（AI_MONITOR_POSE_RX_HANDLE /
        # TX_HANDLE）。这些句柄不在 CRT fd(0/1/2) 表内，ultralytics predict 只破坏标准
        # stdin/stdout/stderr，不会动它们，从而规避 err=232/句柄失效。无 env 时回退
        # 到 GetStdHandle（单测/直跑仍可用）。
        rx = os.environ.get("AI_MONITOR_POSE_RX_HANDLE")
        tx = os.environ.get("AI_MONITOR_POSE_TX_HANDLE")
        try:
            self.hin = int(rx) if rx else int(self._cw.GetStdHandle(-10))
        except Exception:  # noqa: BLE001
            self.hin = int(self._cw.GetStdHandle(-10))
        try:
            self.hout = int(tx) if tx else int(self._cw.GetStdHandle(-11))
        except Exception:  # noqa: BLE001
            self.hout = int(self._cw.GetStdHandle(-11))

    def readn(self, n: int):
        import ctypes

        out = b""
        while len(out) < n:
            buf = ctypes.create_string_buffer(n - len(out))
            got = self._wt.DWORD(0)
            if not self._cw.ReadFile(self.hin, buf, n - len(out), ctypes.byref(got), None):
                return None
            gotn = got.value
            if gotn == 0:
                return None
            out += buf.raw[:gotn]
        return out

    def writeall(self, data: bytes) -> bool:
        import ctypes

        _trace("RAWW hout=%d len=%d" % (int(self.hout), len(data)))
        buf = ctypes.create_string_buffer(data, len(data))
        addr = ctypes.addressof(buf)
        offset = 0
        while offset < len(data):
            got = self._wt.DWORD(0)
            ok = self._cw.WriteFile(
                self.hout, ctypes.c_void_p(addr + offset),
                len(data) - offset, ctypes.byref(got), None,
            )
            if not ok:
                _trace("RAWW FAIL hout=%d err=%d partial=%d" % (
                    int(self.hout), ctypes.get_last_error(), offset))
                return False
            n = got.value
            _trace("RAWW wrote chunk=%d total=%d" % (n, offset + n))
            if n == 0:
                _trace("RAWW ZERO hout=%d" % int(self.hout))
                return False
            offset += n
        return True


def _read_message(reader) -> dict | None:
    """buffered 路径（单测/注入流）：4 字节大端长度前缀 + UTF-8 JSON。"""
    hdr = reader.read(4)
    if not hdr or len(hdr) < 4:
        return None
    (n,) = struct.unpack(">I", hdr)
    n = int(n)
    if n > _MAX_MSG_BYTES:
        return None
    data = reader.read(n)
    if data is None or len(data) < n:
        return None
    return json.loads(data.decode("utf-8"))


def _read_message_raw(raw) -> dict | None:
    """原生命柄读：绕过 CRT fd 表。"""
    hdr = raw.readn(4)
    if not hdr or len(hdr) < 4:
        return None
    (n,) = struct.unpack(">I", hdr)
    n = int(n)
    if n > _MAX_MSG_BYTES:
        return None
    data = raw.readn(n)
    if data is None:
        return None
    return json.loads(data.decode("utf-8"))


def _write_message(writer, obj: dict) -> None:
    """buffered 路径（单测/注入流）。"""
    body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    writer.write(struct.pack(">I", len(body)) + body)
    writer.flush()


def _write_message_raw(raw, obj: dict) -> None:
    """原生命柄写：绕过 CRT fd 表。整个帧一次 WriteFile 推入管道。"""
    body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    raw.writeall(struct.pack(">I", len(body)) + body)


# ---------------------------------------------------------------- 配置加载/入口
def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent.parent


def _load_config() -> FallTaskConfig:
    """从 WORKER_CONF（JSON 字符串或文件路径）构建 FallTaskConfig；缺省补全必需绝对路径。"""
    raw_text = os.environ.get("WORKER_CONF", "") or os.environ.get("WORKER_CONFIG", "")
    data: dict = {}
    if raw_text:
        p = raw_text.strip()
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            data = json.loads(p)
    if not isinstance(data, dict):
        data = {}
    root = _project_root()

    data.setdefault("mode", os.environ.get("WORKER_MODE", "shadow"))
    data.setdefault("enabled", os.environ.get("WORKER_ENABLED", "False") == "True")
    data.setdefault("runtime_key", os.environ.get("WORKER_RUNTIME_KEY", "pose-cuda-0"))

    data.setdefault("gpu", {
        "required": True, "device": os.environ.get("WORKER_DEVICE", "cuda:0"),
        "precision": os.environ.get("WORKER_PRECISION", "fp16"),
        "allow_cpu_fallback": False,
    })
    gpu = data["gpu"]
    gpu.setdefault("required", True)
    gpu.setdefault("device", os.environ.get("WORKER_DEVICE", "cuda:0"))
    gpu.setdefault("precision", os.environ.get("WORKER_PRECISION", "fp16"))
    gpu["allow_cpu_fallback"] = False

    data.setdefault("worker", {"python": os.environ.get("WORKER_PYTHON") or sys.executable})
    w = data["worker"]
    w.setdefault("python", os.environ.get("WORKER_PYTHON") or sys.executable)
    w.setdefault("module", "ai_monitor_pose.worker")

    model_path = os.environ.get(
        "WORKER_MODEL", str(root / "models" / "yolov8n-pose.pt")
    )
    data.setdefault("model", {
        "path": model_path,
        "sha256_file": os.environ.get(
            "WORKER_MODEL_SHA", str(pathlib.Path(model_path)) + ".sha256"
        ),
        "allow_download": False,
    })
    m = data["model"]
    m.setdefault("path", model_path)
    m.setdefault("sha256_file", str(pathlib.Path(m["path"])) + ".sha256")
    m.setdefault("allow_download", False)

    data.setdefault("runtime", {
        "capacity_manifest_path": os.environ.get(
            "WORKER_CAPACITY_MANIFEST",
            str(root / "models" / "capacity-cuda0.json"),
        ),
    })
    rt = data["runtime"]
    rt.setdefault("capacity_manifest_path", os.environ.get(
        "WORKER_CAPACITY_MANIFEST", str(root / "models" / "capacity-cuda0.json"),
    ))
    rt.setdefault("worker_journal_path", os.environ.get("WORKER_JOURNAL", ""))

    return FallTaskConfig.from_mapping(data)


def run(argv: "list[str] | None" = None) -> int:
    """真实 Worker 入口：加载配置、构造服务、进入消息循环。"""
    config = _load_config()
    svc = WorkerService(config)
    return svc.run(argv)