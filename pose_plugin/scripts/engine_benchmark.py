r"""TensorRT engine 与 .pt 同机基准对比（延迟 / 真实帧精度 sanity / 显存峰值）。

对 models/ 下每个存在的模型（yolov8n-pose.pt / .engine / .alt.engine）执行:
  a) 合成延迟: np.zeros((640,640,3),uint8) 预热 5 次 + 计时 50 次 predict
     （生产参数 imgsz=640 / conf=0.35 / iou=0.45 / max_det=50 / workers=0,
      device=0; .pt 传 half=True, .engine 不传 half——TRT 精度已烘焙）。
      engine 加载后首个 predict 是懒初始化, 预热必须真实跑 predict;
      计时用 time.perf_counter, 每帧 torch.cuda.synchronize() 避免异步失真。
      报告 avg/P50/P95 ms。
  b) 真实帧精度 sanity: 均匀抽 fixtures/queda.mp4 每 57 帧 1 张共 20 帧,
     统计 conf>=0.35 检出 person 数、以及 7 个必需关键点(idx 0,5,6,11,12,13,14,
     kp conf>=0.30)全可见的目标数; engine 相对 .pt 基准报告:
     帧级检出数一致率(|检出数差|<=1 的帧占比) 与 关键点可见目标数平均差。
  c) 显存: 模型加载前 torch.cuda.reset_peak_memory_stats(), 全部推理后读
     max_memory_allocated()（仅统计 torch allocator 侧, 不含 TRT 自管显存）。

JSON 结果原子写入 pose_plugin/var/engine_benchmark_result.json。

用法（worker venv, 禁止额外安装）:
    .venv-worker/Scripts/python.exe scripts/engine_benchmark.py
    .venv-worker/Scripts/python.exe scripts/engine_benchmark.py \
        --models models/yolov8n-pose.pt,models/yolov8n-pose.engine
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── 生产参数（configs/default.yaml: fall_detection.model / algorithm）────
PROD_IMGSZ = 640
PROD_CONF = 0.35
PROD_IOU = 0.45
PROD_MAX_DET = 50
REQUIRED_KP_INDICES: tuple[int, ...] = (0, 5, 6, 11, 12, 13, 14)
KP_MIN_CONF = 0.30
VIDEO_STRIDE = 57          # queda.mp4 1152 帧 / 20 帧 ≈ 57
VIDEO_MAX_FRAMES = 20

POSE_PLUGIN_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = POSE_PLUGIN_DIR / "fixtures" / "queda.mp4"
DEFAULT_JSON_OUT = POSE_PLUGIN_DIR / "var" / "engine_benchmark_result.json"


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


def _cuda_sync(device: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


# ── 模型发现 ─────────────────────────────────────────────

def resolve_models(models_arg: str | None) -> list[Path]:
    """显式列表优先（逗号分隔）; 否则自动探测 models/ 下 .pt/.engine
    （*.engine 同时覆盖 .alt.engine）, .pt 排最前作为对比基准。"""
    if models_arg:
        paths = [Path(p.strip()) for p in models_arg.split(",") if p.strip()]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise SystemExit(f"指定的模型不存在: {missing}")
        if not paths:
            raise SystemExit("--models 为空")
        return paths
    models_dir = POSE_PLUGIN_DIR / "models"
    found = sorted(models_dir.glob("*.pt")) + sorted(models_dir.glob("*.engine"))
    if not found:
        raise SystemExit(f"models/ 下未发现任何 .pt/.engine: {models_dir}")
    return found


# ── 真实帧采样 ───────────────────────────────────────────

def sample_frames(video_path: Path) -> list[tuple[int, np.ndarray]]:
    """grab/retrieve 均匀抽帧: 索引 0, 57, 114, ... 共 VIDEO_MAX_FRAMES 张。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video_path}")
    frames: list[tuple[int, np.ndarray]] = []
    idx = 0
    while len(frames) < VIDEO_MAX_FRAMES:
        if not cap.grab():
            break
        if idx % VIDEO_STRIDE == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                frames.append((idx, frame))
        idx += 1
    cap.release()
    if not frames:
        raise SystemExit(f"未能从视频抽取任何帧: {video_path}")
    return frames


# ── 结果抽取 ─────────────────────────────────────────────

def extract_counts(pred: Any) -> tuple[int, int]:
    """返回 (conf>=PROD_CONF 检出数, 必需关键点全可见目标数)。"""
    boxes = getattr(pred, "boxes", None)
    data = getattr(boxes, "data", None) if boxes is not None else None
    if data is None or len(data) == 0:
        return 0, 0
    arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
    persons = int((arr[:, 4] >= PROD_CONF).sum())
    kp_complete = 0
    kps = getattr(pred, "keypoints", None)
    kdata = getattr(kps, "data", None) if kps is not None else None
    if kdata is not None:
        karr = kdata.cpu().numpy() if hasattr(kdata, "cpu") else np.asarray(kdata)
        if karr.ndim == 3 and karr.shape[0] == arr.shape[0] and karr.shape[1] > max(REQUIRED_KP_INDICES):
            for row in karr:
                if all(row[i, 2] >= KP_MIN_CONF for i in REQUIRED_KP_INDICES):
                    kp_complete += 1
    return persons, kp_complete


def predict_kwargs(device: int, is_engine: bool) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "imgsz": PROD_IMGSZ, "conf": PROD_CONF, "iou": PROD_IOU,
        "max_det": PROD_MAX_DET, "verbose": False, "workers": 0,
        "device": device,
    }
    if not is_engine:  # TRT 精度已烘焙进 engine, 不传 half
        kw["half"] = True
    return kw


# ── 单模型基准 ───────────────────────────────────────────

def run_model(path: Path, args: argparse.Namespace,
              frames: list[tuple[int, np.ndarray]]) -> dict:
    is_engine = path.suffix != ".pt"
    entry: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "type": "engine" if is_engine else "pt",
        "size_bytes": path.stat().st_size,
        "error": None,
    }
    kw = predict_kwargs(args.device, is_engine)
    model = None
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(args.device)  # 加载前清零峰值
        model = YOLO(str(path), task="pose")
        # a) 合成延迟: 预热必须真实 predict（engine 首次推理懒初始化）
        synthetic = np.zeros((PROD_IMGSZ, PROD_IMGSZ, 3), dtype=np.uint8)
        samples: list[float] = []
        for i in range(args.warmup + args.iterations):
            _cuda_sync(args.device)
            t0 = time.perf_counter()
            model.predict(source=synthetic, **kw)
            _cuda_sync(args.device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if i >= args.warmup:
                samples.append(dt_ms)
        entry["latency_ms"] = {
            "avg": round(statistics.mean(samples), 3),
            "p50": round(_percentile(samples, 50), 3),
            "p95": round(_percentile(samples, 95), 3),
            "min": round(min(samples), 3),
            "max": round(max(samples), 3),
            "std": round(statistics.pstdev(samples), 3),
            "warmup": args.warmup,
            "count": len(samples),
            "samples": [round(x, 3) for x in samples],
        }
        # b) 真实帧精度 sanity
        per_frame: list[dict[str, Any]] = []
        for idx, frame in frames:
            preds = model.predict(source=frame, **kw)
            persons, kp_ok = extract_counts(preds[0] if preds else None)
            per_frame.append({"frame_index": idx, "persons": persons, "kp_complete": kp_ok})
        entry["real_frames"] = {
            "video": str(Path(args.video)),
            "stride": VIDEO_STRIDE,
            "per_frame": per_frame,
            "mean_persons": round(statistics.mean(p["persons"] for p in per_frame), 3),
            "mean_kp_complete": round(statistics.mean(p["kp_complete"] for p in per_frame), 3),
        }
        # c) 显存峰值（加载 + 全部推理）
        if torch.cuda.is_available():
            entry["vram_peak_mb"] = round(
                torch.cuda.max_memory_allocated(args.device) / 1024 ** 2, 1)
    except Exception as exc:  # 半成品 engine 等不应中断其余模型
        import traceback
        tb = traceback.format_exc().strip().splitlines()
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["error_traceback_tail"] = " | ".join(tb[-4:])
        print(f"[{path.name}] 失败: {entry['error']}\n  {entry['error_traceback_tail']}")
    finally:
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return entry


# ── engine vs .pt 对比 ───────────────────────────────────

def build_comparison(entry: dict, baseline: dict) -> dict:
    cmp: dict[str, Any] = {"model": entry["name"], "baseline": baseline["name"],
                           "available": False}
    e_pf = (entry.get("real_frames") or {}).get("per_frame") or []
    b_pf = (baseline.get("real_frames") or {}).get("per_frame") or []
    if not e_pf or not b_pf or len(e_pf) != len(b_pf):
        cmp["reason"] = "per-frame 数据缺失（某侧推理失败）"
        return cmp
    n = len(e_pf)
    within1 = sum(1 for e, b in zip(e_pf, b_pf) if abs(e["persons"] - b["persons"]) <= 1)
    exact = sum(1 for e, b in zip(e_pf, b_pf) if e["persons"] == b["persons"])
    kp_diffs = [e["kp_complete"] - b["kp_complete"] for e, b in zip(e_pf, b_pf)]
    cmp.update({
        "available": True,
        "frames_compared": n,
        "detection_consistency_within1_rate": round(within1 / n, 4),
        "detection_exact_match_rate": round(exact / n, 4),
        "kp_complete_mean_diff": round(statistics.mean(kp_diffs), 3),
        "kp_complete_mean_abs_diff": round(statistics.mean(abs(d) for d in kp_diffs), 3),
    })
    e_lat = (entry.get("latency_ms") or {}).get("avg")
    b_lat = (baseline.get("latency_ms") or {}).get("avg")
    if e_lat and b_lat:
        cmp["latency_speedup_vs_baseline"] = round(b_lat / e_lat, 3)
    return cmp


# ── 环境与输出 ───────────────────────────────────────────

def environment(device: int) -> dict:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_index": device,
    }
    try:
        import ultralytics
        env["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    try:
        import tensorrt
        env["tensorrt"] = tensorrt.__version__
    except Exception:
        env["tensorrt"] = "unavailable"
    if torch.cuda.is_available():
        env["device_name"] = torch.cuda.get_device_name(device)
        env["torch_cuda"] = torch.version.cuda
    return env


def print_summary(results: list[dict], comparisons: list[dict]) -> None:
    print()
    print("=" * 96)
    print("延迟 / 显存（合成 zeros 640 输入, 生产参数 imgsz=640 conf=0.35 iou=0.45 max_det=50）")
    print("=" * 96)
    print(f"{'模型':<30}{'类型':<8}{'avg(ms)':>10}{'P50(ms)':>10}{'P95(ms)':>10}{'显存峰值MB':>12}")
    for r in results:
        lat = r.get("latency_ms") or {}
        vram = r.get("vram_peak_mb")
        if r.get("error"):
            print(f"{r['name']:<30}{r['type']:<8}{'ERROR':>10}  {r['error'][:60]}")
            continue
        print(f"{r['name']:<30}{r['type']:<8}"
              f"{lat.get('avg', 0):>10.2f}{lat.get('p50', 0):>10.2f}"
              f"{lat.get('p95', 0):>10.2f}"
              f"{(f'{vram:.1f}' if vram is not None else 'n/a'):>12}")
    if comparisons:
        print()
        print("=" * 96)
        print("engine vs .pt 基准一致性（真实帧 sanity, 20 帧）")
        print("=" * 96)
        print(f"{'模型':<30}{'检出一致率(|差|≤1)':>20}{'精确一致率':>12}"
              f"{'关键点可见数平均差':>18}{'平均|差|':>10}{'加速比':>8}")
        for c in comparisons:
            if not c.get("available"):
                print(f"{c['model']:<30}  不可用: {c.get('reason', '')}")
                continue
            w1 = c["detection_consistency_within1_rate"]
            ex = c["detection_exact_match_rate"]
            n = c["frames_compared"]
            sp = c.get("latency_speedup_vs_baseline")
            print(f"{c['model']:<30}"
                  f"{f'{int(round(w1 * n))}/{n} ({w1 * 100:.1f}%)':>20}"
                  f"{f'{ex * 100:.1f}%':>12}"
                  f"{c['kp_complete_mean_diff']:>+18.3f}"
                  f"{c['kp_complete_mean_abs_diff']:>10.3f}"
                  f"{(f'{sp:.2f}x' if sp else 'n/a'):>8}")


def write_atomic(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    import os
    os.replace(tmp, path)
    return str(path)


# ── CLI ──────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=None,
                        help="逗号分隔的模型路径列表; 默认自动探测 models/ 下 .pt/.engine")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    return parser


def main(argv: "list[str] | None" = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用, 本基准要求 GPU（生产 gpu.required=true）")
    torch.cuda.init()  # reset_peak_memory_stats 无 lazy_init, 必须先显式初始化 CUDA 上下文
    model_paths = resolve_models(args.models)
    print(f"待基准模型: {[p.name for p in model_paths]}")
    print(f"GPU: {torch.cuda.get_device_name(args.device)}")
    frames = sample_frames(Path(args.video))
    print(f"真实帧 sanity: {args.video} 抽取 {len(frames)} 帧 (stride={VIDEO_STRIDE})")

    results = [run_model(p, args, frames) for p in model_paths]
    baseline = next((r for r in results if r["type"] == "pt" and not r.get("error")), None)
    comparisons: list[dict] = []
    if baseline:
        for r in results:
            if r is baseline:
                continue
            if r["type"] == "engine" and not r.get("error"):
                comparisons.append(build_comparison(r, baseline))
    else:
        print("警告: 无可用 .pt 基准, 跳过一致性对比")

    payload = {
        "schema_version": 1,
        "generated_utc": _now_utc(),
        "environment": environment(args.device),
        "params": {
            "imgsz": PROD_IMGSZ, "conf": PROD_CONF, "iou": PROD_IOU,
            "max_det": PROD_MAX_DET, "device": args.device,
            "warmup": args.warmup, "iterations": args.iterations,
            "half": "pt=True / engine 不传(TRT 精度已烘焙)",
            "required_kp_indices": list(REQUIRED_KP_INDICES),
            "kp_min_conf": KP_MIN_CONF,
        },
        "video": {"path": str(Path(args.video)), "sampled_frames": len(frames),
                  "stride": VIDEO_STRIDE},
        "models": results,
        "comparisons": comparisons,
    }
    out = write_atomic(Path(args.json_out), payload)
    print_summary(results, comparisons)
    print(f"\nJSON 结果: {out}")
    return 0 if all(not r.get("error") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
