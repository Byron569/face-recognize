"""CUDA 硬守卫与 OOM 策略（第 6.5 / 6.9 节）。

禁止任何 CPU 回退：CUDA 不可用 / 索引错 / 模型或输出在 CPU / OOM 一律进入
UNAVAILABLE 并携带对应错误码，绝不返回 Normal 或切到 CPU。
"""
from __future__ import annotations

import hashlib
import pathlib
import re
from dataclasses import dataclass, field

from ..errors import (
    CudaBuildMissingError, CudaUnavailableError, CudaDeviceInvalidError,
    GPU_OOM, MODEL_HASH_MISMATCH,
)

_CUDA_RE = re.compile(r"^cuda:(\d+)$")


def parse_explicit_cuda_index(device: str) -> int:
    m = _CUDA_RE.match(device)
    if not m:
        raise ValueError(f"必须显式 cuda:N: {device!r}")
    return int(m.group(1))


def assert_gpu_ready(torch) -> None:
    if torch.version.cuda is None:
        raise CudaBuildMissingError("Torch 为 CPU-only 构建")
    if not torch.cuda.is_available():
        raise CudaUnavailableError("CUDA 不可用")
    return None


def check_device_index(torch, index: int) -> None:
    if index < 0 or index >= torch.cuda.device_count():
        raise CudaDeviceInvalidError(f"cuda index {index} 不存在")
    return None


def validate_model_sha256(model_side: tuple[str, str]) -> dict:
    model_path, sha_path = model_side
    mp = pathlib.Path(model_path)
    if not mp.is_file() or mp.stat().st_size == 0:
        raise FileNotFoundError(f"模型不存在或为空: {mp}")
    side = pathlib.Path(sha_path)
    if not side.is_file():
        raise FileNotFoundError(f"缺少 sha256 sidecar: {side}")
    expected = side.read_text().strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("sha256 文件格式非法")
    actual = hashlib.sha256(mp.read_bytes()).hexdigest()
    if actual != expected:
        raise Exception(f"sha256 mismatch for model={mp.name}")
    return {"sha256": actual}


def assert_parameter_device(parameters, *, device: str) -> None:
    for p in parameters:
        d = getattr(p, "device", None)
        if d is not None and d.type not in ("cuda",):
            raise Exception(f"模型参数不在 CUDA: {d}")
    return None


def assert_gpu_required_config(*, required: bool) -> None:
    if required is not True:
        raise ValueError("gpu.required 必须为 true")


@dataclass
class OomPolicy:
    cpu_predict_calls: int = 0
    _armed: bool = field(default=False)

    def classify(self, err: Exception) -> str:
        name = type(err).__name__
        if "OOM" in name or "OutOfMemory" in name or (hasattr(err, "__class__") and "OutOfMemory" in err.__class__.__module__):
            return GPU_OOM
        return "CUDA_RUNTIME_ERROR"

    def allowed_devices(self) -> list[str]:
        return ["cuda"]


def compute_model_sha256(model_path: str) -> str:
    return hashlib.sha256(pathlib.Path(model_path).read_bytes()).hexdigest()
