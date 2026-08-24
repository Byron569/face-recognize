"""阶段5：CUDA 硬守卫。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_monitor_pose.worker import gpu_guard as gg
from ai_monitor_pose.errors import CudaBuildMissingError, CudaUnavailableError, CudaDeviceInvalidError

import tempfile, pathlib, hashlib


def _cusession(tmp: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    model = tmp / "pose.pt"
    model.write_bytes(b"x" * 100)
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    side = tmp / "pose.sha256"
    side.write_text(sha)
    return model, side


class _FakeCuda:
    def __init__(self, available=True, version="12.6", count=1):
        self._available = available
        self._version = version
        self._count = count
    @property
    def is_available(self):
        return self
    def __bool__(self): return self._available
    def __call__(self, *a, **k): return self._available
    def device_count(self): return self._count


def _fake_torch(*, cuda_ver="12.6", available=True, count=1):
    return SimpleNamespace(
        version=SimpleNamespace(cuda=cuda_ver),
        cuda=_FakeCuda(available=available, version=cuda_ver, count=count),
    )


def test_cpu_only_torch_build_refuses_start() -> None:
    torch = _fake_torch(cuda_ver=None)
    with pytest.raises(CudaBuildMissingError):
        gg.assert_gpu_ready(torch)


def test_cuda_unavailable_refuses_start() -> None:
    torch = _fake_torch(cuda_ver="12.6", available=False)
    with pytest.raises(CudaUnavailableError):
        gg.assert_gpu_ready(torch)


def test_nonexistent_cuda_index_refuses_start() -> None:
    torch = _fake_torch(cuda_ver="12.6", available=True, count=1)
    with pytest.raises(CudaDeviceInvalidError):
        gg.check_device_index(torch, 5)


def test_missing_model_fails_before_yolo_constructor() -> None:
    torch = _fake_torch()
    with pytest.raises(FileNotFoundError):
        gg.validate_model_sha256(tmp_missing())


def tmp_missing():
    return ("C:/definitely/not/a/model.pt", "C:/definitely/not.a.sha256")


def test_hash_mismatch_fails_before_yolo_constructor(tmp_path) -> None:
    model, side = _cusession(tmp_path)
    side.write_text("deadbeef" * 8)
    with pytest.raises(Exception) as ei:
        gg.validate_model_sha256((model, side))
    assert "hash" in str(ei.value).lower() or "mismatch" in str(ei.value).lower()


def test_model_parameter_device_is_verified() -> None:
    class Param:
        device = SimpleNamespace(type="cpu")
    with pytest.raises(Exception):
        gg.assert_parameter_device([Param()], device="cuda:0")
