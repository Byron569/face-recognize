"""
services.model_manager — 推理引擎池(单例)。

按 (模型包, 模型根, 设备, 检测尺寸, 阈值) 缓存 InsightFaceEngine,
多个摄像头共享同一份模型/显存;不同配置自动创建独立引擎。
注册/搜索接口也从这里取引擎,避免每个请求重复加载模型(旧架构的最大性能坑)。
"""


from __future__ import annotations
import logging
import threading
from typing import Dict

from vision.config import VisionConfig
from vision.engine import InsightFaceEngine

logger = logging.getLogger(__name__)

_engine_pool: "EnginePool | None" = None


def get_engine_pool() -> "EnginePool":
    global _engine_pool
    if _engine_pool is None:
        _engine_pool = EnginePool()
    return _engine_pool


class EnginePool:
    def __init__(self):
        self._lock = threading.Lock()
        self._engines: Dict[str, InsightFaceEngine] = {}

    @staticmethod
    def _key(cfg: VisionConfig) -> str:
        det = tuple(cfg.det_size) if isinstance(cfg.det_size, (list, tuple)) else str(cfg.det_size)
        return f"{cfg.model_pack}|{cfg.models_root}|{cfg.device}|{det}|{cfg.det_thresh}"

    def get(self, cfg: VisionConfig) -> InsightFaceEngine:
        key = self._key(cfg)
        with self._lock:
            engine = self._engines.get(key)
        if engine is not None:
            return engine
        logger.info("[engine-pool] creating engine: %s", key)
        engine = InsightFaceEngine(cfg)
        with self._lock:
            # 双检,避免并发重复创建
            existing = self._engines.get(key)
            if existing is not None:
                return existing
            self._engines[key] = engine
        return engine

    def close_all(self) -> None:
        with self._lock:
            engines, self._engines = self._engines, {}
        for engine in engines.values():
            try:
                engine.close()
            except Exception:  # noqa: BLE001
                pass

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._engines)
