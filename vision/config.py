"""
vision.config — 内核参数 dataclass。

所有参数由 configs/*.yaml 经 backend 加载后注入,vision 内部不读取任何全局状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


@dataclass
class TrackConfig:
    iou_threshold: float = 0.3
    max_lost: int = 15
    min_hits: int = 2
    max_tracks: int = 30

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "TrackConfig":
        cfg = cfg or {}
        return cls(
            iou_threshold=float(cfg.get("iou_threshold", 0.3)),
            max_lost=int(cfg.get("max_lost", 15)),
            min_hits=int(cfg.get("min_hits", 2)),
            max_tracks=int(cfg.get("max_tracks", 30)),
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
            det_interval=int(cfg.get("det_interval", 2)),
            track=TrackConfig.from_dict(cfg.get("track")),
            recognition=RecognitionConfig.from_dict(cfg.get("recognition")),
        )
