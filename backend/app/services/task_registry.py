"""
services.task_registry — 可插拔任务注册表。

从配置的 tasks: 节实例化任务(按 class_path 动态导入),
未来扩展任务(如跌倒检测)只需提供类并登记配置,无需改任何主流程代码。
"""


from __future__ import annotations
import importlib
import logging
from typing import Any, Dict, List, Optional

from vision.tasks import VisionTask

logger = logging.getLogger(__name__)


def instantiate_task(
    class_path: str,
    config: Optional[dict],
    extra_kwargs: Optional[dict] = None,
) -> VisionTask:
    """按 'package.module.ClassName' 动态导入并实例化任务。

    extra_kwargs(如 gallery / tracker / full_config)按需注入:
    任务构造器声明了对应参数则传入,否则仅传 config。
    """
    module_name, _, class_name = class_path.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"非法 class_path: {class_path}")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    extra = extra_kwargs or {}
    # 只注入构造器实际接受的参数,避免 TypeError 掩盖真实初始化错误
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        accepted = {
            k: v for k, v in extra.items()
            if k in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        }
    else:
        accepted = extra
    return cls(config or {}, **accepted)


class TaskRegistry:
    """任务注册表:加载配置 → 实例化 → 交给 Pipeline。"""

    def __init__(self, tasks_cfg: Optional[Dict[str, Any]] = None):
        self._cfg = tasks_cfg or {}
        self._tasks: List[VisionTask] = []

    def load(self, extra_kwargs: Optional[dict] = None) -> List[VisionTask]:
        """按配置实例化全部任务(已加载则直接返回)。"""
        if self._tasks:
            return self._tasks
        for name, task_cfg in self._cfg.items():
            if not isinstance(task_cfg, dict):
                logger.warning("[tasks] %s 配置非法,已跳过", name)
                continue
            enabled = bool(task_cfg.get("enabled", False))
            class_path = task_cfg.get("class_path")
            if not enabled:
                logger.info("[tasks] %s disabled", name)
                continue
            if not class_path:
                logger.warning("[tasks] %s enabled 但 class_path 为空,已跳过(预留扩展?)", name)
                continue
            try:
                task = instantiate_task(class_path, task_cfg, extra_kwargs)
                self._tasks.append(task)
                logger.info("[tasks] loaded: %s (%s)", name, class_path)
            except Exception:  # noqa: BLE001
                logger.exception("[tasks] %s 加载失败: %s", name, class_path)
        return self._tasks

    def close(self) -> None:
        for task in self._tasks:
            try:
                task.close()
            except Exception:  # noqa: BLE001
                pass
        self._tasks = []

    @property
    def tasks(self) -> List[VisionTask]:
        return list(self._tasks)

    @property
    def registered(self) -> List[dict]:
        """返回配置登记的任务清单(供 /api/tasks 展示)。"""
        out = []
        for name, cfg in self._cfg.items():
            out.append(
                {
                    "name": name,
                    "enabled": bool(cfg.get("enabled", False)),
                    "class_path": cfg.get("class_path"),
                }
            )
        return out
