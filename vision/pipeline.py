"""
vision.pipeline — 单摄像头处理线程(工业化主循环)。

链路: 采集 → 检测(降频)→ 跟踪 → 可插拔任务 → 输出回调

设计原则:
    - 本类只做编排,不包含任何算法与业务规则;
    - 检测/跟踪/任务全部通过构造参数注入(依赖倒置);
    - 帧与事件通过 on_frame / on_event 回调交给后端(WS / 数据库);
    - 任何阶段异常都被隔离,单帧失败不影响流水线。
"""

from __future__ import annotations

from collections import deque

import logging
import threading
import time
from typing import Callable, List, Optional

from .camera import FrameSource
from .config import VisionConfig
from .engine import InsightFaceEngine
from .events import PipelineContext, VisionEvent
from .tasks import VisionTask
from .tracker import ByteTracker

logger = logging.getLogger(__name__)

FrameCallback = Callable[[PipelineContext], None]
EventCallback = Callable[[VisionEvent], None]


class VisionPipeline(threading.Thread):
    """单摄像头推理流水线(daemon 线程)。"""

    def __init__(
        self,
        camera_id: str,
        source: FrameSource,
        engine: InsightFaceEngine,
        tracker: ByteTracker,
        config: VisionConfig,
        tasks: Optional[List[VisionTask]] = None,
        on_frame: Optional[FrameCallback] = None,
        on_event: Optional[EventCallback] = None,
        max_reconnect_attempts: int = 0,     # 0 = 无限重连
    ):
        super().__init__(daemon=True, name=f"vision-{camera_id}")
        self.camera_id = camera_id
        self._source = source
        self._engine = engine
        self._tracker = tracker
        self._config = config
        self._tasks: List[VisionTask] = tasks or []
        self._on_frame = on_frame
        self._on_event = on_event
        self._max_reconnect_attempts = max_reconnect_attempts

        self._running = False
        self._frame_id = 0
        self._started_at: Optional[float] = None
        self._proc_ts: deque = deque()          # 最近处理帧时间戳(滑动窗口 5s)
        self._proc_lock = threading.Lock()      # pipeline 线程写 / metrics 异步读

        # 阶段耗时滚动平均(监控用)
        self._stage_ms = {
            "capture": 0.0, "detect": 0.0, "track": 0.0, "tasks": 0.0, "emit": 0.0,
        }
        self._stage_lock = threading.Lock()

    # ── 生命周期 ──────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        self._started_at = time.time()
        logger.info("[vision] pipeline %s started", self.camera_id)

        if not self._source.open():
            logger.error("[vision] pipeline %s: source open failed, exiting", self.camera_id)
            self._running = False
            return

        reconnect_failures = 0
        while self._running:
            t0 = time.perf_counter()
            ret, frame = self._source.read()
            self._record_stage("capture", time.perf_counter() - t0)

            if not ret:
                reconnect_failures += 1
                if self._max_reconnect_attempts > 0 and reconnect_failures > self._max_reconnect_attempts:
                    logger.error("[vision] pipeline %s: reconnect attempts exhausted", self.camera_id)
                    break
                if not self._source.reconnect():
                    break
                reconnect_failures = 0
                continue
            reconnect_failures = 0

            self._frame_id += 1
            try:
                self._process_frame(frame)
            except Exception:  # noqa: BLE001
                logger.exception("[vision] pipeline %s frame %s failed", self.camera_id, self._frame_id)

        self._close_tasks()
        self._source.release()
        logger.info("[vision] pipeline %s stopped (frames=%s)", self.camera_id, self._frame_id)

    def _process_frame(self, frame) -> None:
        # 0. 滚动帧率(处理吞吐,滑动窗口 5s)
        now = time.perf_counter()
        with self._proc_lock:
            self._proc_ts.append(now)
            while self._proc_ts and now - self._proc_ts[0] > 5.0:
                self._proc_ts.popleft()

        # 1. 检测(降频);非检测帧只做跟踪预测,不判定丢失
        t0 = time.perf_counter()
        if self._frame_id % self._config.det_interval == 0:
            detections = self._engine.detect(frame)
            tracks = self._tracker.update(detections, self._frame_id)
        else:
            tracks = self._tracker.skip(self._frame_id)
        self._record_stage("detect", time.perf_counter() - t0)

        # 2. 跟踪
        t0 = time.perf_counter()
        self._record_stage("track", time.perf_counter() - t0)

        # 3. 可插拔任务
        t0 = time.perf_counter()
        context = PipelineContext(
            camera_id=self.camera_id,
            frame_id=self._frame_id,
            frame=frame,
            tracks=tracks,
            observed_at_monotonic_ns=time.monotonic_ns(),
            observed_at_utc=time.time(),
        )
        for task in self._tasks:
            if not task.enabled:
                continue
            try:
                if not task.should_run(self._frame_id, context):
                    continue
                events = task.run(frame, context)
                for evt in events:
                    if evt.camera_id == "":
                        evt.camera_id = self.camera_id
                    if self._on_event:
                        self._on_event(evt)
            except Exception:  # noqa: BLE001
                logger.exception("[vision] task %s failed", task.name)
        self._record_stage("tasks", time.perf_counter() - t0)

        # 4. 输出回调(帧 + 当前 track 快照)
        t0 = time.perf_counter()
        if self._on_frame:
            self._on_frame(context)
        self._record_stage("emit", time.perf_counter() - t0)

    # ── 控制 ──────────────────────────────────────────────

    def stop(self) -> None:
        self._running = False

    def close(self, timeout: float = 5.0) -> None:
        self._running = False
        if hasattr(self._source, "request_stop"):
            self._source.request_stop()  # 唤醒断流重连循环,让线程可退出
        self.join(timeout=timeout)

    def _close_tasks(self) -> None:
        for task in self._tasks:
            try:
                task.close()
            except Exception:  # noqa: BLE001
                logger.exception("[vision] task %s close failed", task.name)

    # ── 监控 ──────────────────────────────────────────────

    def _record_stage(self, name: str, seconds: float) -> None:
        ms = seconds * 1000
        with self._stage_lock:
            self._stage_ms[name] = self._stage_ms[name] * 0.9 + ms * 0.1  # EMA 平滑

    def metrics(self) -> dict:
        with self._stage_lock:
            stages = dict(self._stage_ms)
        with self._proc_lock:
            ts = list(self._proc_ts)
        if len(ts) >= 2 and ts[-1] > ts[0]:
            fps = (len(ts) - 1) / (ts[-1] - ts[0])
        else:
            fps = 0.0
        return {
            "camera_id": self.camera_id,
            "alive": self.is_alive(),
            "frames": self._frame_id,
            "fps": round(fps, 1),
            "tracks": self._tracker.active_count,
            "uptime_seconds": int(time.time() - self._started_at) if self._started_at else 0,
            "stage_ms": {k: round(v, 1) for k, v in stages.items()},
        }
