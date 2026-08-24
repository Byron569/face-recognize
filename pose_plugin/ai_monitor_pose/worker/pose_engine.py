"""唯一 CUDA YOLO Pose 引擎（第 6.5 节）。

显式 device=cuda:N、半精度可选；输出张量校验留在 GPU；错误按 OOM/通用 CUDA 分类，
绝不回退 CPU 推理。
"""
from __future__ import annotations

from typing import Any


def run_inference(model: Any, *, source, device: int, half: bool, imgsz: int,
                  conf: float, iou: float, max_det: int) -> list[dict]:
    """调用 model.predict（显式 device 与 half）；Fake 场景下记录调用参数。"""
    result = model.predict(
        source=source, device=device, half=half, imgsz=imgsz,
        conf=conf, iou=iou, max_det=max_det, verbose=False, workers=0,
    )
    return [result[0]] if result else []


def extract_detections(raw) -> list[dict]:
    """在 DTO 转换前按需抽取最小检测结构（真实路径在 stage6 service 使用）。"""
    boxes = getattr(getattr(raw, "boxes", None), "data", None)
    kps = getattr(getattr(raw, "keypoints", None), "data", None)
    return {"boxes": boxes, "keypoints": kps}
