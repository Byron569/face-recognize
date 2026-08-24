"""阶段5：CUDA 必需性网关。"""
from __future__ import annotations

import pytest
from ai_monitor_pose.worker.gpu_guard import assert_gpu_required_config


def test_gpu_required_flagged() -> None:
    assert_gpu_required_config(required=True)
    with pytest.raises(ValueError):
        assert_gpu_required_config(required=False)
