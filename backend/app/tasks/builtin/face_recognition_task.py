"""
tasks.builtin.face_recognition_task — 内置人脸识别任务。

职责(与跟踪、检测完全解耦):
    - 按冷却策略决定每个 track 何时识别(新 track 优先 → 冷却到期 → 重验证);
    - 用内存底库快照(FaceGallery)做向量化比对;
    - 把身份写回 IoUTracker,并在身份变化时产出 VisionEvent("recognition")。

识别调度参数全部来自配置(vision.recognition 节),无硬编码。
"""


from __future__ import annotations
from collections import Counter, deque
import logging
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

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
    candidate_scores: Dict[str, Deque[float]] = field(default_factory=dict)
    candidate_info: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    sample_order: Deque[str] = field(default_factory=deque)
    sample_count: int = 0
    valid_sample_count: int = 0
    skipped_frame_count: int = 0
    skip_reasons: Counter = field(default_factory=Counter)
    last_embedding_frame_id: Optional[int] = None

    @property
    def has_pending_samples(self) -> bool:
        return self.sample_count > 0


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
        self._last_camera_id = ""
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
        self._last_camera_id = context.camera_id
        self._cleanup_states(context)

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
        st = self._states.setdefault(track.track_id, _TrackRecState())

        frame_id = context.frame_id
        rec_cfg = self._rec_cfg

        # 调度优先级:新 track / 初始稳定窗口 / 冷却到期 / 已识别重验证
        if not self._cooldown_allows(st, frame_id):
            return None, False

        embedding = self._latest_embedding(track)
        if embedding is None:
            return None, False

        embedding_frame_id = getattr(track, "embedding_frame_id", None)
        if embedding_frame_id is not None:
            if embedding_frame_id == st.last_embedding_frame_id:
                return None, False
            # 无论后续质量是否合格,同一检测帧只消费一次。
            st.last_embedding_frame_id = embedding_frame_id

        quality_reason = self._quality_skip_reason(track)
        if quality_reason is not None:
            st.skipped_frame_count += 1
            st.skip_reasons[quality_reason] += 1
            logger.info(
                "[recognition-quality] camera=%s track=%s frame=%s skipped=%s "
                "det_score=%.3f face_size=%.1f",
                context.camera_id,
                track.track_id,
                frame_id,
                quality_reason,
                float(getattr(track, "score", 0.0)),
                self._face_size(track),
            )
            return None, False

        t0 = time.perf_counter()
        # 先取回每帧最佳候选,最终是否确认由稳定聚合分数与 threshold 决定。
        hit = self._gallery.search(np.asarray(embedding, dtype=np.float32), 0.0)
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
        candidate_key = str(identity_id or name)
        queue = st.candidate_scores.setdefault(candidate_key, deque())
        st.candidate_info[candidate_key] = (str(identity_id), str(name))
        queue.append(float(similarity))
        st.sample_order.append(candidate_key)
        st.sample_count += 1
        st.valid_sample_count += 1
        self._trim_samples(st)

        min_samples = max(1, int(rec_cfg.temporal.min_valid_samples))
        if len(queue) < min_samples:
            return None, True

        candidate_scores = list(queue)
        top_k = max(1, int(rec_cfg.temporal.top_k))
        top_k_scores = sorted(candidate_scores, reverse=True)[:top_k]
        stable_score = float(sum(top_k_scores) / len(top_k_scores))
        if stable_score < rec_cfg.threshold:
            return None, True

        identity_id, name = st.candidate_info[candidate_key]
        changed = st.identity != name
        st.last_success_frame = frame_id
        st.identity = name
        st.similarity = stable_score
        st.identity_id = identity_id

        if self._tracker:
            self._tracker.set_identity(track.track_id, name, stable_score)

        logger.info(
            "[recognition] camera=%s track=%s name=%s candidate_scores=%s "
            "top_k_scores=%s stable_score=%.6f (%.1fms)",
            context.camera_id,
            track.track_id,
            name,
            [round(score, 6) for score in candidate_scores],
            [round(score, 6) for score in top_k_scores],
            stable_score,
            latency_ms,
        )
        self._clear_samples(st)
        return (
            VisionEvent(
                event_type="recognition",
                camera_id=context.camera_id,
                track_id=track.track_id,
                confidence=stable_score,
                payload={
                    "identity_id": identity_id,
                    "name": name,
                    "similarity": stable_score,
                    "stable_score": stable_score,
                    "candidate_scores": candidate_scores,
                    "top_k_scores": top_k_scores,
                    "latency_ms": round(latency_ms, 2),
                    "changed": changed,
                },
            ),
            True,
        )

    def _cooldown_allows(self, st: _TrackRecState, frame_id: int) -> bool:
        rec_cfg = self._rec_cfg
        if st.identity != "Unknown":
            return frame_id - st.last_success_frame >= rec_cfg.recognized_cooldown_frames
        if st.has_pending_samples:
            return True
        if st.last_attempt_frame == -10**9:
            return True
        if rec_cfg.max_attempts > 0 and st.fail_count >= rec_cfg.max_attempts:
            return False
        effective = rec_cfg.cooldown_frames + st.fail_count * rec_cfg.failed_backoff_frames
        return frame_id - st.last_attempt_frame >= effective

    def _quality_skip_reason(self, track) -> Optional[str]:
        quality = self._rec_cfg.quality
        if float(getattr(track, "score", 0.0)) < quality.min_det_score:
            return "low_det_score"
        if self._face_size(track) < quality.min_face_size:
            return "face_too_small"
        return None

    @staticmethod
    def _face_size(track) -> float:
        bbox = getattr(track, "bbox", (0, 0, 0, 0))
        return min(max(0.0, float(bbox[2]) - float(bbox[0])), max(0.0, float(bbox[3]) - float(bbox[1])))

    def _trim_samples(self, st: _TrackRecState) -> None:
        max_samples = max(1, int(self._rec_cfg.temporal.max_samples_per_track))
        while st.sample_count > max_samples:
            oldest_key = st.sample_order.popleft()
            oldest_queue = st.candidate_scores.get(oldest_key)
            if oldest_queue:
                oldest_queue.popleft()
                st.sample_count -= 1
                if not oldest_queue:
                    st.candidate_scores.pop(oldest_key, None)
                    st.candidate_info.pop(oldest_key, None)

    @staticmethod
    def _clear_samples(st: _TrackRecState) -> None:
        st.candidate_scores.clear()
        st.candidate_info.clear()
        st.sample_order.clear()
        st.sample_count = 0

    def _cleanup_states(self, context: PipelineContext) -> None:
        active_ids = {track.track_id for track in context.tracks}
        for track_id in list(self._states):
            if track_id in active_ids:
                continue
            st = self._states.pop(track_id)
            self._log_track_summary(context.camera_id, track_id, st)

    def _log_track_summary(self, camera_id: str, track_id: int, st: _TrackRecState) -> None:
        logger.info(
            "[recognition-track] camera=%s track=%s valid_samples=%s "
            "skipped_frames=%s skip_reasons=%s",
            camera_id,
            track_id,
            st.valid_sample_count,
            st.skipped_frame_count,
            dict(st.skip_reasons),
        )

    def close(self) -> None:
        for track_id, st in list(self._states.items()):
            self._log_track_summary(self._last_camera_id, track_id, st)
        self._states.clear()

    @staticmethod
    def _latest_embedding(track):
        """从 tracker 的 track 对象取最新 embedding。"""
        return getattr(track, "embedding", None)
