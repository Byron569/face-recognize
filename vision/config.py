"""
vision.config — 内核参数 dataclass。

所有参数由 configs/*.yaml 经 backend 加载后注入,vision 内部不读取任何全局状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


@dataclass
class TrackConfig:
    """ByteTrack 参数(见 vision/tracker.py 移植说明)。"""

    track_thresh: float = 0.5     # 高置信度检测阈值(高于此进入第一轮关联)
    low_thresh: float = 0.1       # 低置信度检测下限(低于此直接丢弃)
    match_thresh: float = 0.8     # 第一轮 IoU 关联阈值(成本 1-IoU 的上限)
    track_buffer: int = 30        # lost 保留帧数(按 frame_rate 缩放为实际帧数)
    min_hits: int = 3             # 新轨迹连续命中多少帧后确认(activate)
    frame_rate: int = 30          # 用于 lost 保留时长的缩放
    max_tracks: int = 30          # 最大活跃轨迹数

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "TrackConfig":
        cfg = cfg or {}
        return cls(
            track_thresh=float(cfg.get("track_thresh", 0.5)),
            low_thresh=float(cfg.get("low_thresh", 0.1)),
            match_thresh=float(cfg.get("match_thresh", 0.8)),
            track_buffer=int(cfg.get("track_buffer", 30)),
            min_hits=int(cfg.get("min_hits", 3)),
            frame_rate=int(cfg.get("frame_rate", 30)),
            max_tracks=int(cfg.get("max_tracks", 30)),
        )


@dataclass
class RecognitionQualityConfig:
    min_det_score: float = 0.60
    min_face_size: int = 80

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "RecognitionQualityConfig":
        cfg = cfg or {}
        return cls(
            min_det_score=float(cfg.get("min_det_score", 0.60)),
            min_face_size=int(cfg.get("min_face_size", 80)),
        )


@dataclass
class RecognitionTemporalConfig:
    min_valid_samples: int = 3
    max_samples_per_track: int = 8
    top_k: int = 3

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "RecognitionTemporalConfig":
        cfg = cfg or {}
        return cls(
            min_valid_samples=int(cfg.get("min_valid_samples", 3)),
            max_samples_per_track=int(cfg.get("max_samples_per_track", 8)),
            top_k=int(cfg.get("top_k", 3)),
        )


@dataclass
class RecognitionConfig:
    threshold: float = 0.40
    cooldown_frames: int = 300
    recognized_cooldown_frames: int = 600
    failed_backoff_frames: int = 90
    max_attempts: int = 20
    queue_size: int = 8
    log_to_db: bool = True
    event_to_db: bool = True
    quality: RecognitionQualityConfig = field(default_factory=RecognitionQualityConfig)
    temporal: RecognitionTemporalConfig = field(default_factory=RecognitionTemporalConfig)

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "RecognitionConfig":
        cfg = cfg or {}
        return cls(
            threshold=float(cfg.get("threshold", 0.40)),
            cooldown_frames=int(cfg.get("cooldown_frames", 300)),
            recognized_cooldown_frames=int(cfg.get("recognized_cooldown_frames", 600)),
            failed_backoff_frames=int(cfg.get("failed_backoff_frames", 90)),
            max_attempts=int(cfg.get("max_attempts", 20)),
            queue_size=int(cfg.get("queue_size", 8)),
            log_to_db=bool(cfg.get("log_to_db", True)),
            event_to_db=bool(cfg.get("event_to_db", True)),
            quality=RecognitionQualityConfig.from_dict(cfg.get("quality")),
            temporal=RecognitionTemporalConfig.from_dict(cfg.get("temporal")),
        )


@dataclass
class VisionConfig:
    """视觉内核完整配置。"""

    model_pack: str = "buffalo_s"
    models_root: str = "models"
    device: str = "cuda"                # cuda(默认,全部推理走 GPU)/ cpu / auto
    providers_cuda: List[str] = field(default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    providers_cpu: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    det_size: Any = (640, 640)
    det_thresh: float = 0.5
    max_faces: int = 0
    det_interval: int = 2
    track: TrackConfig = field(default_factory=TrackConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "VisionConfig":
        cfg = cfg or {}
        det_size = cfg.get("det_size", (640, 640))
        if isinstance(det_size, list):
            det_size = tuple(det_size)
        return cls(
            model_pack=str(cfg.get("model_pack", "buffalo_s")),
            models_root=str(cfg.get("models_root", "models")),
            device=str(cfg.get("device", "cuda")),
            providers_cuda=list(cfg.get("providers_cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"])),
            providers_cpu=list(cfg.get("providers_cpu", ["CPUExecutionProvider"])),
            det_size=det_size,
            det_thresh=float(cfg.get("det_thresh", 0.5)),
            max_faces=int(cfg.get("max_faces", 0)),
            det_interval=max(1, int(cfg.get("det_interval", 2))),  # 下界 1,防除零
            track=TrackConfig.from_dict(cfg.get("track")),
            recognition=RecognitionConfig.from_dict(cfg.get("recognition")),
        )
