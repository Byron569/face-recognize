"""
vision.tasks — 可插拔视觉任务接口。

扩展任务(如跌倒检测)只需:
    1. 继承 VisionTask 并实现 should_run() / run();
    2. 在 configs/default.yaml 的 tasks: 节登记 class_path 并 enabled: true。
主循环、路由、前端零改动。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .events import PipelineContext, VisionEvent


class VisionTask(ABC):
    """视觉任务基类。"""

    # 任务名(唯一,用于日志与注册表)
    name: str = "unnamed"
    # 由配置注入的开关与运行间隔
    enabled: bool = True
    interval: int = 1

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.interval = max(1, int(cfg.get("interval", 1)))

    @abstractmethod
    def should_run(self, frame_id: int, context: PipelineContext) -> bool:
        """判断当前帧是否需要执行本任务(由 Pipeline 每帧调用)。"""

    @abstractmethod
    def run(self, frame, context: PipelineContext) -> List[VisionEvent]:
        """执行任务,返回本帧产生的事件列表。"""

    def close(self) -> None:
        """释放任务持有资源(线程/模型等),子类按需覆写。"""
