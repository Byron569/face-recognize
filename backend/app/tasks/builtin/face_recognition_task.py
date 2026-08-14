"""
tasks.builtin.face_recognition_task — 内置人脸识别任务。

职责(与跟踪、检测完全解耦):
    - 按冷却策略决定每个 track 何时识别(新 track 优先 → 冷却到期 → 重验证);
    - 用内存底库快照(FaceGallery)做向量化比对;
    - 把身份写回 IoUTracker,并在身份变化时产出 VisionEvent("recognition")。

识别调度参数全部来自配置(vision.recognition 节),无硬编码。
"""


from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from vision.config import RecognitionConfig
from vision.events import PipelineContext, VisionEvent
from vision.tasks import VisionTask

logger = logging.getLogger(__name__)


@dataclass
class _TrackRecState:
    last_attempt_frame: int = -10**9
    last_success_frame: int = -10**9
    fail_count: int = 0
    identity: str = "Unknown"
    similarity: float = 0.0
    identity_id: str = ""


class FaceRecognitionTask(VisionTask):
    """人脸识别任务(每摄像头一个实例,状态互不干扰)。"""

    name = "face_recognition"

    def __init__(
        self,
        config: Optional[dict] = None,
        gallery=None,
        full_config: Optional[dict] = None,
        tracker=None,
    ):
        cfg = config or {}
        super().__init__(cfg)
        merged = (full_config or {}).get("vision", {}).get("recognition", {}) or {}
        merged = {**merged, **cfg}  # 任务级覆盖全局
        self._rec_cfg = RecognitionConfig.from_dict(merged)

        self._gallery = gallery
        self._tracker = tracker
        self._states: Dict[int, _TrackRecState] = {}
        self._max_per_frame = max(1, int(cfg.get("max_per_frame", 3)))
        self._log_to_db = self._rec_cfg.log_to_db
        self._event_to_db = self._rec_cfg.event_to_db

    # ── 注入(由 pipeline_manager 在组装时调用)─────────────

    def set_gallery(self, gallery) -> None:
        self._gallery = gallery

    def set_tracker(self, tracker) -> None:
        self._tracker = tracker

    # ── VisionTask 接口 ───────────────────────────────────

    def should_run(self, frame_id: int, context: PipelineContext) -> bool:
        return self._gallery is not None and bool(context.tracks)

    def run(self, frame, context: PipelineContext) -> List[VisionEvent]:
        events: List[VisionEvent] = []
        processed = 0

        for track in context.tracks:
            if processed >= self._max_per_frame:
                break
            event, attempted = self._maybe_recognize(track, context)
            if attempted:
                processed += 1  # 无论是否命中,实际比对都计入限流
            if event is not None:
                events.append(event)

        return events

    # ── 识别调度 ──────────────────────────────────────────

    def _maybe_recognize(self, track, context: PipelineContext) -> tuple[Optional[VisionEvent], bool]:
        """尝试识别一个 track。返回 (event, attempted):attempted=True 表示本帧实际执行了底库比对。"""
        st = self._states.get(track.track_id)
        if st is None:
            st = self._states[track.track_id] = _TrackRecState()

        frame_id = context.frame_id
        rec_cfg = self._rec_cfg

        # 调度优先级:新 track / 冷却到期(未识别) / 重验证(已识别)
        if st.last_attempt_frame > 0:  # 已有尝试记录 → 检查冷却
            if st.identity == "Unknown":
                effective = rec_cfg.cooldown_frames + st.fail_count * rec_cfg.failed_backoff_frames
                if rec_cfg.max_attempts > 0 and st.fail_count >= rec_cfg.max_attempts:
                    return None, False
                if frame_id - st.last_attempt_frame < effective:
                    return None, False
            else:
                if frame_id - st.last_success_frame < rec_cfg.recognized_cooldown_frames:
                    return None, False

        embedding = self._latest_embedding(track)
        if embedding is None:
            return None, False

        t0 = time.perf_counter()
        hit = self._gallery.search(np.asarray(embedding, dtype=np.float32), rec_cfg.threshold)
        latency_ms = (time.perf_counter() - t0) * 1000

        st.last_attempt_frame = frame_id
        identity_id, name, similarity = (hit[0], hit[1], hit[2]) if hit else ("", "Unknown", 0.0)

        if name == "Unknown":
            st.fail_count += 1
            # 写回跟踪器,保证前端始终拿到最新身份状态
            if self._tracker:
                self._tracker.set_identity(track.track_id, "Unknown", 0.0)
            return None, True  # 执行了比对,计入限流

        st.fail_count = 0
        st.last_success_frame = frame_id
        changed = st.identity != name
        st.identity = name
        st.similarity = similarity
        st.identity_id = identity_id

        if self._tracker:
            self._tracker.set_identity(track.track_id, name, similarity)

        logger.info(
            "[recognition] camera=%s track=%s name=%s sim=%.3f (%.1fms)",
            context.camera_id, track.track_id, name, similarity, latency_ms,
        )
        return (
            VisionEvent(
                event_type="recognition",
                camera_id=context.camera_id,
                track_id=track.track_id,
                confidence=similarity,
                payload={
                    "identity_id": identity_id,
                    "name": name,
                    "similarity": similarity,
                    "latency_ms": round(latency_ms, 2),
                    "changed": changed,
                },
            ),
            True,
        )

    @staticmethod
    def _latest_embedding(track):
        """从 tracker 的 track 对象取最新 embedding。"""
        return getattr(track, "embedding", None)
