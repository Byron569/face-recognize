"""共享 Runtime 注册表（第 6.1 节）。

只暴露 acquire / release / health_snapshot / shutdown_all。
相同 runtime_key 返回同一个 PoseRuntime；用全局配置指纹拒绝冲突；首次 acquire 绑定
event_sink（后续不同 sink 抛冲突）；引用计数为 0 时停止 Worker。acquire 一律不等待模型。
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import threading
import uuid
from typing import Any

from .config import RuntimeConfig
from .health import FallRuntimeHealthSnapshotV1, WorkerHealthV1, STOPPED
from .runtime import PoseRuntime, PoseRuntimeLease

RUNTIME_CONFIG_CONFLICT = "RUNTIME_CONFIG_CONFLICT"
RUNTIME_EVENT_SINK_CONFLICT = "RUNTIME_EVENT_SINK_CONFLICT"


def _fingerprint(config) -> str:
    d = dataclasses.asdict(config)
    canon = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class PoseRuntimeRegistry:
    _lock = threading.RLock()
    _runtimes: dict[str, PoseRuntime] = {}
    _refcounts: dict[str, int] = {}
    _sinks: dict[str, Any] = {}
    _fingerprints: dict[str, str] = {}

    @classmethod
    def acquire(cls, runtime_key: str, config: RuntimeConfig, event_sink: Any | None, *, process_factory=None, restart_limit: int = 3, retry_delay_s: float = 0.0, heartbeat_interval_s: float = 0, heartbeat_timeout_s: float = 0, mode: str = "shadow") -> PoseRuntimeLease:
        """获取（或复用）runtime_key 对应的 PoseRuntime lease。

        mode 不参与配置指纹校验：runtime 已存在时复用先创建者的 mode
        （与 event_sink 首次绑定行为一致，先到先得），仅新建时透传给 PoseRuntime。
        """
        with cls._lock:
            fp = _fingerprint(config)
            if runtime_key in cls._runtimes:
                if cls._fingerprints.get(runtime_key) != fp:
                    raise RuntimeError(RUNTIME_CONFIG_CONFLICT)
                bound = cls._sinks.get(runtime_key)
                if bound is not None and event_sink is not None and bound is not event_sink:
                    raise RuntimeError(RUNTIME_EVENT_SINK_CONFLICT)
                rt = cls._runtimes[runtime_key]
                if event_sink is not None and bound is None:
                    cls._sinks[runtime_key] = event_sink
            else:
                rt = PoseRuntime(config, process_factory=process_factory, restart_limit=restart_limit, retry_delay_s=retry_delay_s, event_sink=event_sink, heartbeat_interval_s=heartbeat_interval_s, heartbeat_timeout_s=heartbeat_timeout_s, mode=mode)
                cls._runtimes[runtime_key] = rt
                cls._fingerprints[runtime_key] = fp
                cls._sinks[runtime_key] = event_sink
                cls._refcounts[runtime_key] = 0
                rt.start()
            cls._refcounts[runtime_key] = cls._refcounts.get(runtime_key, 0) + 1
            return PoseRuntimeLease(lease_id=uuid.uuid4().hex, runtime_key=runtime_key,
                                    runtime=rt, _release_cb=cls.release)

    @classmethod
    def release(cls, lease: PoseRuntimeLease) -> None:
        if lease is None:
            return
        lease.closed = True
        key = lease.runtime_key
        with cls._lock:
            if cls._refcounts.get(key, 0) <= 0:
                return  # 幂等
            cls._refcounts[key] -= 1
            if cls._refcounts[key] > 0:
                return
            rt = cls._runtimes.pop(key, None)
            cls._fingerprints.pop(key, None)
            cls._sinks.pop(key, None)
            cls._refcounts.pop(key, None)
        if rt is not None:
            rt.stop_and_drain_blocking()

    @classmethod
    def health_snapshot(cls, runtime_key: str | None = None) -> FallRuntimeHealthSnapshotV1:
        with cls._lock:
            if runtime_key is not None:
                rt = cls._runtimes.get(runtime_key)
                if rt is not None:
                    return rt.health_snapshot(runtime_key)
                return _empty_snapshot(runtime_key)
            for k, rt in cls._runtimes.items():
                return rt.health_snapshot(k)
            return _empty_snapshot(None)

    @classmethod
    async def shutdown_all(cls, timeout_s: float = 5.0) -> None:
        with cls._lock:
            items = list(cls._runtimes.items())
            cls._runtimes.clear()
            cls._refcounts.clear()
            cls._sinks.clear()
            cls._fingerprints.clear()
        for key, rt in items:
            await asyncio.to_thread(rt.stop_and_drain_blocking, timeout_s)

    @classmethod
    def _reset(cls) -> None:
        with cls._lock:
            items = list(cls._runtimes.values())
            cls._runtimes.clear()
            cls._refcounts.clear()
            cls._sinks.clear()
            cls._fingerprints.clear()
        for rt in items:
            try:
                rt.stop_and_drain_blocking()
            except Exception:
                pass


def _empty_snapshot(runtime_key: str | None) -> FallRuntimeHealthSnapshotV1:
    return FallRuntimeHealthSnapshotV1(
        schema_version=1, enabled=False, mode=None, runtime_key=runtime_key,
        worker=WorkerHealthV1(schema_version=1, state=STOPPED, error_code=None,
                              error_message=None, worker_epoch=None, worker_pid=None,
                              cuda_device=None, cuda_device_name=None, model_sha256=None,
                              last_heartbeat_monotonic_ns=None, restart_count=0),
        gpu_metrics={}, model_metadata={}, delivery_metrics={}, cameras=(),
    )