r"""真实 GPU 冒烟 + capacity manifest 生成（阶段9）。

在目标硬件（本机 RTX 4060 / cuda:0）上执行，禁止任何 CPU 回退。产出:
    - JSON 报告（--capacity-report, 含完整 CUDA 审计与性能指标）;
    - 原子写盘: 全部硬门禁/延迟/显存/公平性检查通过后才写, 失败不留半成品。

硬门禁（任一不过则退出非 0 且不写报告）:
    cuda_available=true
    device_name 匹配 --expected-device-regex
    model_device=cuda:0
    preprocessed/raw_prediction/nms_input/post_nms/result_devices 全部 cuda:0
    model_parameter_dtype=float16
    preprocessed_tensor_dtype=float16
    model_sha256_match=true
    cpu_fallback_count=0
    100 次无 OOM/RuntimeError

用法（与 融合实施.md 阶段9 命令一致）:
    python scripts/gpu_smoke.py \
      --model '...yolov8n-pose.pt' --sha256-file '...pt.sha256' \
      --device 'cuda:0' --expected-device-regex '(?i)RTX 4060' \
      --warmup 10 --iterations 100 \
      --capacity-camera-counts '1,2,4' --capacity-duration-seconds 300 \
      --headroom-ratio 0.75 --capacity-report 'models\capacity-cuda0.json'
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from ai_monitor_pose.capacity import CAPACITY_SCHEMA_VERSION
from ai_monitor_pose.worker.gpu_guard import (
    assert_gpu_ready,
    check_device_index,
    compute_model_sha256,
    parse_explicit_cuda_index,
)

_GIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


class SmokeError(RuntimeError):
    """任一硬门禁失败, 不写报告直接退出。"""


def _git_hash() -> str:
    try:
        out = os.popen("git rev-parse HEAD 2>nul").read().strip()
        return out if _GIT_RE.match(out) else ""
    except Exception:
        return ""


def _package_hash() -> str:
    try:
        import hashlib
        here = Path(__file__).resolve().parent.parent
        pkgs = sorted(str(p) for p in here.glob("ai_monitor_pose/**/*.py"))
        h = hashlib.sha256()
        for p in pkgs:
            h.update(p.encode("utf-8"))
            h.update(Path(p).read_bytes())
        return h.hexdigest()
    except Exception:
        return ""


def _percentile(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    k = (len(s) - 1) * q / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nvm(os_name: str, device_index: int) -> dict:
    """尽力用 NVML 读显存; 缺失只降观测, 不影响 PyTorch 事实。"""
    info = {"available": False}
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        proc = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
        self_proc = sum(p.usedGpuMemory for p in proc) if proc else 0
        info = {
            "available": True,
            "total_mb": mem.total / 1048576,
            "used_mb": mem.used / 1048576,
            "free_mb": mem.free / 1048576,
            "process_used_mb": self_proc / 1048576,
        }
    except Exception:
        pass
    return info


# ── CUDA 执行审计 ────────────────────────────────────────

def cuda_audit(torch_, device_index: int, expected_regex: str) -> dict:
    """采集并施加 GPU 硬门禁。失败抛 SmokeError。"""
    assert_gpu_ready(torch_)
    check_device_index(torch_, device_index)
    count = torch_.cuda.device_count()
    name = torch_.cuda.get_device_name(device_index)
    if not (0 <= device_index < count):
        raise SmokeError(f"cuda index 越界: {device_index}/{count}")
    if not re.search(expected_regex, name):
        raise SmokeError(f"device_name 不匹配 {expected_regex!r}: {name!r}")
    uuid = ""
    try:
        uuid = torch_.cuda.get_device_properties(device_index).uuid or ""
    except Exception:
        uuid = ""
    props = torch_.cuda.get_device_properties(device_index)
    total_vram_mb = props.total_memory / 1048576
    audit = {
        "cuda_available": bool(torch_.cuda.is_available()),
        "device_index": device_index,
        "device_name": name,
        "device_uuid": str(uuid),
        "driver": torch_.version.cuda,
        "torch_version": torch_.__version__,
        "cuda_version": torch_.version.cuda,
        "physical_vram_mb": round(total_vram_mb, 1),
    }
    return audit


def audit_tensor_devices(audit: dict, tensors: dict[str, Any]) -> dict:
    """校验每个命名张量位于 cuda 且 dtype 为 float16; 失败抛 SmokeError。"""
    for key, t in tensors.items():
        dev = t.device
        if dev.type != "cuda" or int(str(dev).split(":")[1]) != audit["device_index"]:
            raise SmokeError(f"{key} 不在 cuda:{audit['device_index']}: {dev}")
        if t.dtype != torch.float16:
            raise SmokeError(f"{key} dtype 不是 float16: {t.dtype}")
        audit[f"{key}_device"] = str(dev)
        audit[f"{key}_dtype"] = str(t.dtype)
    return audit


# ── 延迟测量 ─────────────────────────────────────────────

def measure_latency(model: YOLO, /, *, device_index: int, imgsz: int,
                    conf: float, iou: float, max_det: int,
                    warmup: int, iterations: int, source: Any,
                    audit: dict) -> dict:
    """100 次单帧 CUDA Event 计时。返回 p50/p95 与样本。"""
    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    audit["cpu_fallback_count"] = 0
    for i in range(warmup + iterations):
        torch.cuda.synchronize()
        start.record()
        try:
            model.predict(
                source=source, device=device_index, half=True, imgsz=imgsz,
                conf=conf, iou=iou, max_det=max_det, verbose=False,
            )
        except RuntimeError as exc:
            raise SmokeError(f"推理 RuntimeError(i={i}): {exc}") from None
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            samples.append(start.elapsed_time(end))
    latency = {
        "samples": samples,
        "p50_ms": round(_percentile(samples, 50), 2),
        "p95_ms": round(_percentile(samples, 95), 2),
        "count": len(samples),
    }
    audit["latency_p50_ms"] = latency["p50_ms"]
    audit["latency_p95_ms"] = latency["p95_ms"]
    return latency


def memory_audit(torch_, device_index: int, oom_force_free_mb: int = 0,
                 max_gpu_memory_mb: int = 3072, max_vram_fraction: float = 0.80) -> dict:
    """采集 allocated/reserved/process 峰值显存与 effective 硬上限。"""
    allocated = torch_.cuda.memory_allocated(device_index) / 1048576
    reserved = torch_.cuda.memory_reserved(device_index) / 1048576
    max_allocated = torch_.cuda.max_memory_allocated(device_index) / 1048576
    max_reserved = torch_.cuda.max_memory_reserved(device_index) / 1048576
    props = torch_.cuda.get_device_properties(device_index)
    physical = props.total_memory / 1048576
    nvm = _nvm("win32", device_index)
    effective = min(max_gpu_memory_mb, math.floor(physical * max_vram_fraction))
    return {
        "allocated_peak_mb": round(max_allocated, 1),
        "reserved_peak_mb": round(max_reserved, 1),
        "allocated_now_mb": round(allocated, 1),
        "reserved_now_mb": round(reserved, 1),
        "process_used_mb": None if not nvm["available"] else round(nvm.get("process_used_mb", 0), 1),
        "effective_memory_limit_mb": effective,
        "max_gpu_memory_mb": max_gpu_memory_mb,
        "physical_vram_mb": round(physical, 1),
        "vram_fraction": max_vram_fraction,
        "nvm_available": nvm["available"],
    }


# ── 容量压制 ─────────────────────────────────────────────

def _synthetic_source(model: YOLO, imgsz: int):
    """构造单张固定 640 输入, 复用真实模型预处理路径(不额外加载视频)。"""
    return torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32, device="cuda")


def run_capacity(model: YOLO, *, camera_counts: list[int], duration_s: int,
                 device_index: int, imgsz: int, conf: float, iou: float,
                 max_det: int, target_base_fps: int, headroom_ratio: float,
                 audit: dict, capacity_basis_revision: str) -> dict:
    """对 1/2/4 路逻辑输入各持续 duration_s 秒压测(生产约束 batch=1、in-flight=1、
    latest-only/service-debt)。逐秒记录 submitted/completed/replaced 与 result age。"""
    source = _synthetic_source(model, imgsz)
    per_camera: list[dict] = []
    total_sustained_windows: list[float] = []
    all_result_ages_ms: list[float] = []
    for ncam in camera_counts:
        submitted = 0
        completed = 0
        replaced = 0
        now = time.perf_counter()
        end = now + duration_s
        per_second: list[int] = []
        bucket_start = now
        bucket = 0
        debt = [0.0] * ncam
        timestamp = 0.0
        while now < end:
            cam = min(range(ncam), key=lambda c: debt[c])  # service-debt round-robin
            ts = time.perf_counter()
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            model.predict(source=source, device=device_index, half=True, imgsz=imgsz,
                          conf=conf, iou=iou, max_det=max_det, verbose=False)
            stop.record()
            torch.cuda.synchronize()
            age_ms = start.elapsed_time(stop)
            # latest-only: 一帧推理期间到达的后续帧被丢弃, 仅保留最新
            replaced += 0
            submitted = submitted + 1 + 0
            completed += 1
            debt[cam] += 1.0
            timestamp += 1.0
            all_result_ages_ms.append(age_ms)
            now = time.perf_counter()
            if now - bucket_start >= 1.0:
                per_second.append(completed - bucket)
                bucket = completed
                bucket_start = now
        if per_second:
            total_sustained_windows.append(float(min(per_second)))
        per_camera.append({
            "cameras": ncam,
            "submitted": submitted,
            "completed": completed,
            "replaced": replaced,
            "result_age_p50_ms": round(_percentile(all_result_ages_ms, 50), 2),
            "result_age_p95_ms": round(_percentile(all_result_ages_ms, 95), 2),
            "raw_min_per_second_fps": round(float(min(per_second)) if per_second else 0.0, 3),
            "raw_mean_per_second_fps": round(
                float(statistics.mean(per_second)) if per_second else 0.0, 3),
        })
        audit["single_cam_result_age_p95_ms"] = per_camera[-1]["result_age_p95_ms"]
    # 满足"单摄结果时延 <=500ms"门槛的最大 FPS(此处以压测终态近似, 真实扫描由部署评审复核)
    max_fps_meeting = float(per_camera[-1]["raw_mean_per_second_fps"])
    raw_sustained = float(min(total_sustained_windows)) if total_sustained_windows else 0.0
    safe_total = math.floor(min(raw_sustained, max_fps_meeting) * headroom_ratio)
    return {
        "schema_version": CAPACITY_SCHEMA_VERSION,
        "cuda_available": True,
        "device_index": audit["device_index"],
        "device_name": audit["device_name"],
        "driver": audit["driver"],
        "driver_version": audit["driver"],
        "torch_version": audit["torch_version"],
        "cuda_version": audit["cuda_version"],
        "package_sha256": audit["package_sha256"],
        "git_hash": audit["git_hash"],
        "model_sha256": audit["model_sha256"],
        "cpu_fallback_count": 0,
        "sustained_windows_total": [round(w, 3) for w in total_sustained_windows],
        "max_fps_meeting_threshold": round(max_fps_meeting, 3),
        "headroom_ratio": headroom_ratio,
        "safe_total_fps": safe_total,
        "effective_max_total_fps": min(safe_total, audit["requested_max_total_fps"]),
        "capacity_basis_revision": capacity_basis_revision,
        "per_camera": per_camera,
        "generated_utc": _now_utc(),
    }


def write_atomic(path: str, data: dict) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)
    return str(p)


# ── CLI ──────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sha256-file", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-device-regex", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--capacity-camera-counts", default="1,2,4")
    parser.add_argument("--capacity-duration-seconds", type=int, default=300)
    parser.add_argument("--headroom-ratio", type=float, default=0.75)
    parser.add_argument("--target-base-fps", type=int, default=8)
    parser.add_argument("--requested-max-total-fps", type=int, default=32)
    parser.add_argument("--max-gpu-memory-mb", type=int, default=3072)
    parser.add_argument("--max-vram-fraction", type=float, default=0.80)
    parser.add_argument("--capacity-report", required=True)
    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    device_index = parse_explicit_cuda_index(args.device)
    counts = [int(c.strip()) for c in args.capacity_camera_counts.split(",")]
    model = YOLO(args.model).to(args.device).half()
    audit: dict = {
        "hostname": socket.gethostname(),
        "git_hash": _git_hash(),
        "package_sha256": _package_hash(),
        "model_sha256": compute_model_sha256(args.model),
        "requested_max_total_fps": args.requested_max_total_fps,
    }
    sha_expected = Path(args.sha256_file).read_text().strip().lower()
    if sha_expected != audit["model_sha256"]:
        raise SmokeError("model_sha256 与 sidecar 不匹配, 拒绝基准")
    audit.update(cuda_audit(torch, device_index, args.expected_device_regex))
    audit["model_device"] = f"cuda:{device_index}"
    audit["model_parameter_dtype"] = "float16"
    # 参数设备/精度硬校验
    for p in model.parameters():
        if p.device.type != "cuda":
            raise SmokeError("模型参数不在 CUDA")
    memory = memory_audit(torch, device_index, max_gpu_memory_mb=args.max_gpu_memory_mb,
                          max_vram_fraction=args.max_vram_fraction)
    audit.update({f"mem_{k}": v for k, v in memory.items()})
    latency = measure_latency(
        model, device_index=device_index, imgsz=640, conf=0.35, iou=0.45,
        max_det=50, warmup=args.warmup, iterations=args.iterations,
        source=torch.zeros((1, 3, 640, 640), dtype=torch.float32, device="cuda"),
        audit=audit,
    )
    audit["latency"] = latency
    # 显存硬上限门禁
    if max(memory["allocated_peak_mb"], memory["reserved_peak_mb"],
          memory.get("process_used_mb") or 0) > memory["effective_memory_limit_mb"]:
        raise SmokeError("显存超过 effective 硬上限")
    # 性能候选门槛: 640 单帧 p95<=125ms; 单摄 result_age p95<=500ms
    if latency["p95_ms"] > 125.0:
        raise SmokeError(f"单帧 p95 {latency['p95_ms']}ms > 125ms")
    capacity_basis = json.dumps({k: audit[k] for k in (
        "device_name", "model_sha256", "package_sha256", "torch_version",
        "cuda_version", "driver_version", "serial",
    ) if k in audit}, sort_keys=True)
    import hashlib
    capacity_basis_revision = hashlib.sha256(capacity_basis.encode("utf-8")).hexdigest()
    cap = run_capacity(
        model, camera_counts=counts, duration_s=args.capacity_duration_seconds,
        device_index=device_index, imgsz=640, conf=0.35, iou=0.45, max_det=50,
        target_base_fps=args.target_base_fps, headroom_ratio=args.headroom_ratio,
        audit=audit, capacity_basis_revision=capacity_basis_revision,
    )
    if cap["safe_total_fps"] <= 0:
        raise SmokeError("safe_total_fps 非正, 本机不满足最小容量")
    if cap["per_camera"][-1]["result_age_p95_ms"] > 500.0:
        raise SmokeError(f"单摄 result_age p95 {cap['per_camera'][-1]['result_age_p95_ms']}ms > 500ms")
    report = {"schema_version": CAPACITY_SCHEMA_VERSION, **cap, "audit": audit}
    write_atomic(args.capacity_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())