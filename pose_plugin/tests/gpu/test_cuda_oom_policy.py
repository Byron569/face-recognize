"""阶段5：CUDA OOM 与通用 CUDA 错误绝不回退 CPU。"""
from __future__ import annotations

import numpy as np
import pytest

from ai_monitor_pose.worker.gpu_guard import OomPolicy
from ai_monitor_pose.errors import GPU_OOM


class _CudaOOM(RuntimeError):
    pass


def test_cuda_oom_marks_unavailable_without_cpu_retry() -> None:
    p = OomPolicy()
    code = p.classify(_CudaOOM())
    assert code == GPU_OOM
    assert p.cpu_predict_calls == 0
    assert "cpu" not in p.allowed_devices()


def test_generic_cuda_error_never_invokes_cpu_predict() -> None:
    p = OomPolicy()
    code = p.classify(RuntimeError("cuda runtime error"))
    assert code == "CUDA_RUNTIME_ERROR"
    assert p.cpu_predict_calls == 0
