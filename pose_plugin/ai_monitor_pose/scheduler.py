"""latest-only 公平调度器（第 6.4 节）。

父进程进程内使用；只管理描述符与 slot 状态，不执行模型。
每摄像头只保留最新 pending；全局 in-flight 第一版为 1；service-debt 选择待服务摄像头，
同分用 round-robin 游标。重复 / 倒退 / 旧 session / 旧 epoch / 已过期请求在推理前拒绝。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OfferOutcome(str, Enum):
    ACCEPTED = "accepted"
    REPLACED_OLDER_FRAME = "replaced_older_frame"
    RATE_LIMITED = "rate_limited"
    DUPLICATE_FRAME = "duplicate_frame"
    STALE_FRAME = "stale_frame"


@dataclass
class _CameraState:
    target_interval_ns: int
    latest_generation: int | None = None
    last_frame_id: int | None = None
    pending: tuple[int, int] | None = None    # (generation, frame_id)
    inflight: tuple[int, int] | None = None
    last_dispatch_ns: int | None = None
    submitted: int = 0
    replaced: int = 0
    dispatched: int = 0
    completed: int = 0


class FallScheduler:
    def __init__(self, *, target_fps: int, batch_size: int = 1) -> None:
        if batch_size != 1:
            raise ValueError("batch_size 第一版固定为 1")
        self.target_fps = target_fps
        self.batch_size = batch_size
        self._interval_ns = int(1e9 / max(1, target_fps))
        self._cameras: dict[str, _CameraState] = {}
        self._cursor = 0

    def register_camera(self, camera_id: str, *, target_fps: int | None = None) -> None:
        self._cameras[camera_id] = _CameraState(
            target_interval_ns=int(1e9 / max(1, (target_fps or self.target_fps)))
        )

    def unregister_camera(self, camera_id: str) -> None:
        self._cameras.pop(camera_id, None)

    def offer(self, camera_id: str, frame_id: int, generation: int, now_ns: int) -> OfferOutcome:
        st = self._cameras[camera_id]
        if st.last_frame_id is not None:
            if frame_id < st.last_frame_id:
                return OfferOutcome.STALE_FRAME
            if frame_id == st.last_frame_id:
                return OfferOutcome.DUPLICATE_FRAME
        st.submitted += 1
        st.latest_generation = generation
        st.last_frame_id = frame_id
        replaced = st.pending is not None or st.inflight is not None
        st.pending = (generation, frame_id)
        if replaced:
            st.replaced += 1
            return OfferOutcome.REPLACED_OLDER_FRAME
        return OfferOutcome.ACCEPTED

    def pick(self, now_ns: int):
        """返回 (camera_id, generation) 或 None；全局只允许一个 in-flight。"""
        busy = [c for c, st in self._cameras.items() if st.inflight is not None]
        if busy:
            return None
        candidates = [c for c, st in self._cameras.items() if st.pending is not None]
        if not candidates:
            return None

        def key(c: str) -> tuple[float, int]:
            st = self._cameras[c]
            if st.last_dispatch_ns is None:
                debt = float("inf")
            else:
                debt = (now_ns - st.last_dispatch_ns) / st.target_interval_ns
            order = list(self._cameras).index(c)
            return (debt, order)

        ordered = sorted(candidates, key=key)
        pick = ordered[0]
        st = self._cameras[pick]
        st.inflight = st.pending
        st.pending = None
        st.last_dispatch_ns = now_ns
        st.dispatched += 1
        return (pick, st.inflight[0])

    def complete(self, camera_id: str, generation: int, *, ok: bool = True) -> None:
        st = self._cameras[camera_id]
        if st.inflight is not None and st.inflight[0] == generation:
            st.inflight = None
            st.completed += int(ok)

    def counters(self, camera_id: str) -> dict:
        st = self._cameras[camera_id]
        return {
            "submitted": st.submitted, "replaced": st.replaced,
            "dispatched": st.dispatched, "completed": st.completed,
        }
