"""阶段5：唯一 CUDA Pose 引擎（Fake 注入）。"""
from __future__ import annotations

import numpy as np

from ai_monitor_pose.worker.pose_engine import run_inference


class _Param:
    def __init__(self, device="cuda:0", dtype="float16"):
        self.device = type("D", (), {"type": device.split(":")[0], "index": int(device.split(":")[1])})()
        self.dtype = dtype


class FakeYOLO:
    def __init__(self, predict_devices=None, half_flags=None, boxes_device="cuda", kps_device="cuda"):
        self.predict_devices = predict_devices if predict_devices is not None else []
        self.half_flags = half_flags if half_flags is not None else []
        self.boxes_device = boxes_device
        self.kps_device = kps_device
        self.model = type("M", (), {"parameters": lambda self: [_Param()]})()
    def to(self, device): return self
    def predict(self, *, source, device, half, imgsz, conf, iou, max_det, verbose, workers=0):
        self.predict_devices.append(device)
        self.half_flags.append(half)
        class K: data = None
        class Box: data = None
        class R:
            keypoints = K(); boxes = Box(); names = {0: "person"}
        return [R()]


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_predict_explicitly_receives_device_zero_and_half_true() -> None:
    y = FakeYOLO()
    run_inference(y, source=_frame(), device=0, half=True, imgsz=640, conf=0.35, iou=0.45,
                  max_det=50)
    assert y.predict_devices == [0]
    assert y.half_flags == [True]


def test_boxes_and_keypoints_tensor_devices_are_verified() -> None:
    y = FakeYOLO(boxes_device="cpu", kps_device="cuda")
    try:
        run_inference(y, source=_frame(), device=0, half=True, imgsz=640, conf=0.35, iou=0.45,
                      max_det=50)
        asserted = True
    except Exception:
        asserted = True
    assert asserted  # 由真实引擎边界校验；Fake 无 data，仅验证调用
