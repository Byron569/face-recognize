"""
vision/ — 纯推理内核(唯一依赖 InsightFace)。

分层:
    camera.py   帧采集(本地摄像头 / RTSP / 视频文件,自动重连)
    engine.py   InsightFace 官方 FaceAnalysis 封装(检测 + 识别一体,GPU 优先)
    tracker.py  轻量 IoU 跟踪器(稳定 track_id,识别按需触发)
    tasks.py    可插拔视觉任务接口(跌倒检测等扩展的预留点)
    pipeline.py 单摄像头处理线程:采集 → 检测 → 跟踪 → 任务 → 输出回调
    events.py   内核数据模型(VisionEvent / FaceResult / TrackResult / PipelineContext)
    config.py   全部参数 dataclass(由 YAML 注入,零硬编码)

本包不依赖 backend/fastapi,可独立运行与测试。
"""

from .config import VisionConfig
from .events import VisionEvent, FaceResult, TrackResult, PipelineContext
from .tasks import VisionTask
from .engine import InsightFaceEngine
from .tracker import IoUTracker
from .pipeline import VisionPipeline

__all__ = [
    "VisionConfig",
    "VisionEvent",
    "FaceResult",
    "TrackResult",
    "PipelineContext",
    "VisionTask",
    "InsightFaceEngine",
    "IoUTracker",
    "VisionPipeline",
]
