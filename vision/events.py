"""
vision.events — 内核数据模型。

所有跨模块(内核 ↔ 任务 ↔ 后端)的数据交换都使用这里的 dataclass,
不存在裸 dict 隐式契约。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VisionEvent:
    """任务产出的统一事件,后端据此写库 / 推 WebSocket。"""

    event_type: str
    camera_id: str = ""
    track_id: Optional[int] = None
    confidence: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "confidence": self.confidence,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


@dataclass
class FaceResult:
    """InsightFace 单张人脸检测结果(engine 输出的统一格式)。"""

    bbox: Tuple[float, float, float, float]        # x1, y1, x2, y2
    det_score: float
    kps: Optional[List[Tuple[float, float]]] = None   # 5 点关键点
    embedding: Optional[Any] = None                   # 512-d 已归一化向量(点积即余弦相似度)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class TrackResult:
    """跟踪结果:每个活跃 track 一条。识别任务会把 identity 写回这里。"""

    track_id: int
    bbox: Tuple[float, float, float, float]
    score: float = 0.0
    hits: int = 0
    confirmed: bool = False
    identity: str = "Unknown"
    similarity: float = 0.0
    embedding: Optional[Any] = None       # 该 track 最新的人脸 embedding(识别任务使用)
    embedding_frame_id: Optional[int] = None  # embedding 最近一次更新的检测帧号

    @property
    def is_identified(self) -> bool:
        return self.identity != "Unknown"

    def to_dict(self) -> Dict[str, Any]:
        """供前端渲染的稳定结构(bbox 为 [x, y, w, h])。"""
        x1, y1, x2, y2 = self.bbox
        return {
            "track_id": self.track_id,
            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "identity": self.identity,
            "confidence": round(self.similarity, 3),
        }


@dataclass
class PipelineContext:
    """每帧传给任务与回调的上下文快照。"""

    camera_id: str
    frame_id: int
    frame: Any
    tracks: List[TrackResult] = field(default_factory=list)
