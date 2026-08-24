"""worker.metrics 模块（阶段0.5 脚手架）。"""
from __future__ import annotations


def __getattr__(name: str):
    """阶段0.5 无行为脚手架：本模块尚未实现，任何公开符号访问即抛 NotImplementedError。"""
    raise NotImplementedError(
        f"{__name__}.{name} 尚未实现（阶段0.5 脚手架占位，等待阶段行为测试驱动）"
    )
