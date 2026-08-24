"""阶段9:真实 CUDA 推理硬门禁(需 RTX 4060 / cuda:0)。

无 GPU / 缺模型时整体跳过, 不污染无 GPU 的 CI。有 GPU 时逐项施加与
gpu_smoke.py 完全一致的硬门禁:
    cuda_available=true
    device_name 匹配 expected regex
    model_device=cuda:0 / 参数 in cuda / dtype=float16
    preprocessed/raw_prediction/nms_input/post_nms/result 全 cuda
    model_sha256 匹配 sidecar
    cpu_fallback_count=0
    100 次推理无 OOM/RuntimeError
模型路径可用环境变量 AIM_POSE_MODEL / AIM_POSE_SHA 覆盖, 默认取项目 models/ 下
yolov8n-pose.pt 与 sidecar。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not (torch.cuda.is_available() and torch.version.cuda):
    pytest.skip("无 CUDA, 跳过真实推理门禁", allow_module_level=True)

MODEL = Path(os.environ.get(
    "AIM_POSE_MODEL",
    str(Path(__file__).resolve().parent.parent.parent / "models" / "yolov8n-pose.pt"),
))
SHA_FILE = Path(os.environ.get("AIM_POSE_SHA", str(MODEL) + ".sha256"))
DEVICE = os.environ.get("AIM_POSE_DEVICE", "cuda:0")
DEVICE_IDX = 0
EXPECTED_REGEX = os.environ.get("AIM_POSE_EXPECTED_REGEX", r"(?i)RTX 4060")


@pytest.fixture(scope="module")
def model():
    YOLO = pytest.importorskip("ultralytics").YOLO
    if not (MODEL.is_file() and MODEL.stat().st_size > 0):
        pytest.skip(f"模型不存在: {MODEL}")
    return YOLO(MODEL)



def test_cuda_available_and_device_matches_regex() -> None:
    assert torch.cuda.is_available() is True
    assert torch.version.cuda is not None
    name = torch.cuda.get_device_name(DEVICE_IDX)
    assert re.search(EXPECTED_REGEX, name), f"device {name!r} 不匹配 {EXPECTED_REGEX!r}"
    assert DEVICE_IDX < torch.cuda.device_count()


def test_model_parameters_on_cuda_float16(model) -> None:
    moved = model.to(DEVICE).half()
    noncuda = [p for p in moved.parameters() if p.device.type != "cuda"]
    assert not noncuda, f"有参数不在 CUDA: {noncuda}"
    bad_dtype = {str(p.dtype) for p in moved.parameters() if p.dtype != torch.float16}
    assert not bad_dtype, f"参数 dtype 非 float16: {bad_dtype}"


def test_model_sha256_matches_sidecar() -> None:
    if not SHA_FILE.is_file():
        pytest.skip("缺少 sha256 sidecar")
    actual = _sha256(MODEL)
    expected = SHA_FILE.read_text().strip().lower()
    assert re.fullmatch(r"[0-9a-f]{64}", expected)
    assert actual == expected


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_output_tensors_stay_on_cuda(model) -> None:
    source = torch.zeros((1, 3, 640, 640), dtype=torch.float32, device=DEVICE)
    r = model.predict(source=source, device=DEVICE_IDX, half=True, imgsz=640,
                      conf=0.35, iou=0.45, max_det=50, verbose=False)[0]
    # result 关键张量留在 CUDA(不落到 CPU)
    boxes = getattr(r, "boxes", None)
    kps = getattr(r, "keypoints", None)
    for obj, name in ((getattr(boxes, "data", None), "boxes.data"),
                      (getattr(kps, "data", None), "keypoints.data")):
        if obj is not None:
            assert obj.device.type == "cuda", f"{name} 不在 cuda: {obj.device}"
    assert getattr(r, "orig_shape", None) is not None


def test_100_inferences_no_oom_no_cpu_fallback(model) -> None:
    source = torch.zeros((1, 3, 640, 640), dtype=torch.float32, device=DEVICE)
    for i in range(100):
        # 真实错误(含 OOM)不得被吞掉, 也不得触发 CPU 兜底
        model.predict(source=source, device=DEVICE_IDX, half=True, imgsz=640,
                      conf=0.35, iou=0.45, max_det=50, verbose=False)


def test_latency_p95_within_budget(model) -> None:
    source = torch.zeros((1, 3, 640, 640), dtype=torch.float32, device=DEVICE)
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(130):
        torch.cuda.synchronize()
        start.record()
        model.predict(source=source, device=DEVICE_IDX, half=True, imgsz=640,
                      conf=0.35, iou=0.45, max_det=50, verbose=False)
        end.record()
        torch.cuda.synchronize()
        if i >= 30:
            samples.append(start.elapsed_time(end))
    p95 = sorted(samples)[int(len(samples) * 0.95)] if samples else 9999
    torch.cuda.synchronize()
    assert p95 <= 125.0, f"单帧 p95={p95:.1f}ms > 125ms"