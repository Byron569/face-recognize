"""阶段8b — 后端测试 5:pipeline_manager 生命周期关停。

不启动真实摄像头/引擎/源,只验证确定性行为:
    - stop_camera 对不存在摄像头返回 False 且幂等;
    - stop_camera 清理每摄像头状态(stream_settings / stream_metrics /
      delivery_modes / last_frames / 编码线程句柄);
    - 投递模式默认 shadow,显式设置后按摄像头解析;
    - shutdown 无摄像头、无 loop 绑定下幂等,且关闭 EventIngress;
    - 全部停用后 is_running 返回 False。

使用独立实例,不触全局单例的 engine pool / ingress,保证测试隔离。
"""
from __future__ import annotations

import asyncio

from backend.app.services.pipeline_manager import PipelineManager
from backend.app.services.event_ingress import EventIngress


class _StubEnginePool:
    """替代真实 EnginePool,承载 close_all 幂等即可。"""

    def __init__(self) -> None:
        self.closed = False

    def close_all(self) -> None:
        self.closed = True

    def get(self, vision_cfg):  # pragma: no cover - 测试不触发
        raise NotImplementedError


def _fresh_manager() -> PipelineManager:
    mgr = PipelineManager()
    mgr._engine_pool = _StubEnginePool()  # 隔离,避免关停全局池
    mgr._event_ingress = EventIngress()    # 独立 ingress,不碰全局单例
    return mgr


def test_stop_camera_absent_returns_false_and_is_idempotent() -> None:
    mgr = _fresh_manager()
    first = asyncio.run(mgr.stop_camera("nope"))
    second = asyncio.run(mgr.stop_camera("nope"))
    assert first is False and second is False


def test_stop_camera_cleans_per_camera_state() -> None:
    mgr = _fresh_manager()
    # 模拟已开启的摄像头残留状态(不启动真实线程)
    mgr._stream_settings["cam-1"] = {"max_height": 480, "jpeg_quality": 80, "push_fps": 15}
    mgr._stream_metrics.setdefault("cam-1")
    mgr._delivery_modes["cam-1"] = "alert"

    # 假编码线程句柄(未真正 start,只验证清理)
    import threading

    stop = threading.Event()
    mgr._encode_stops["cam-1"] = stop
    mgr._encode_threads["cam-1"] = None
    mgr._encode_queues["cam-1"] = None
    mgr._last_frames["cam-1"] = bytes(8)

    result = asyncio.run(mgr.stop_camera("cam-1"))

    assert result is False  # 无真实 pipeline
    assert "cam-1" not in mgr._stream_settings
    assert "cam-1" not in mgr._stream_metrics
    assert "cam-1" not in mgr._delivery_modes
    assert "cam-1" not in mgr._last_frames
    assert "cam-1" not in mgr._encode_stops
    assert "cam-1" not in mgr._encode_queues
    assert stop.is_set()  # 编码线程停止事件已触发


def test_resolve_delivery_mode_defaults_to_shadow() -> None:
    mgr = _fresh_manager()
    assert mgr._resolve_delivery_mode("unset-cam") == "shadow"


def test_delivery_mode_reflects_explicit_setting() -> None:
    mgr = _fresh_manager()
    mgr._delivery_modes["cam-x"] = "alert"
    assert mgr._resolve_delivery_mode("cam-x") == "alert"


def test_shutdown_without_cameras_is_idempotent_and_closes_ingress() -> None:
    mgr = _fresh_manager()
    ingress = mgr._event_ingress

    asyncio.run(mgr.shutdown())  # 第一次:未绑定 loop,close 早期返回
    asyncio.run(mgr.shutdown())  # 第二次:幂等,不抛错

    assert ingress._closed is True
    assert ingress._started is False
    assert mgr._engine_pool.closed is True
    assert mgr.list_cameras() == []


def test_shutdown_after_stopping_camera_leaves_no_cameras() -> None:
    mgr = _fresh_manager()
    # 模拟残留流水线条目(不真实线程;shutdown 遍历 stop_camera 应清理)
    mgr._pipelines["ghost"] = None  # type: ignore[assignment]
    mgr._delivery_modes["ghost"] = "alert"
    asyncio.run(mgr.shutdown())
    assert mgr.list_cameras() == []
    assert "ghost" not in mgr._delivery_modes


def test_is_running_false_for_absent_camera() -> None:
    mgr = _fresh_manager()
    assert mgr.is_running("absent") is False