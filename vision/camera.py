"""
vision.camera — 帧采集层。

支持:
    - 本地摄像头(整数索引,如 0 / 1)
    - 网络流(RTSP / HTTP 等 URL 字符串)
    - 视频文件路径
自动处理:打开失败 / 读流中断 时的指数退避重连。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


class FrameSource(ABC):
    """帧源抽象接口(可扩展: GStreamer、厂商 SDK 等只需实现本接口)。"""

    @abstractmethod
    def open(self) -> bool: ...

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[Any]]: ...

    @abstractmethod
    def release(self) -> None: ...


class OpenCVFrameSource(FrameSource):
    """基于 OpenCV VideoCapture 的通用帧源。

    Args:
        source: 摄像头索引(int)或 RTSP/HTTP/文件路径(str)。
        width / height: 期望分辨率(设备不支持时保持实际值)。
        max_width: 过大帧自动缩放的宽度上限(性能保护),0=不缩放。
        reconnect_delay / max_reconnect_delay: 重连退避参数(秒)。
    """

    def __init__(
        self,
        source,
        width: int = 640,
        height: int = 480,
        max_width: int = 960,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 10.0,
    ):
        self._source = source
        self._width = width
        self._height = height
        self._max_width = max_width
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay
        self._cap = None
        self._current_delay = reconnect_delay

    # ── 生命周期 ──────────────────────────────────────────

    def open(self) -> bool:
        try:
            # RTSP 等网络流设置超时,避免卡死
            if isinstance(self._source, str) and not self._source.isdigit():
                cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            else:
                cap = cv2.VideoCapture(self._source)
            # width/height 为 0 时跳过设置 → 使用源原生分辨率(对 RTSP/文件源本就无效)
            if self._width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            if self._height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            if not cap.isOpened():
                cap.release()
                return False
            self._cap = cap
            self._current_delay = self._reconnect_delay
            logger.info("[vision] frame source opened: %s", self._source)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("[vision] open source failed (%s): %s", self._source, exc)
            self._cap = None
            return False

    def read(self) -> Tuple[bool, Optional[Any]]:
        if self._cap is None:
            return False, None
        try:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return False, None
            if self._max_width > 0 and frame.shape[1] > self._max_width:
                scale = self._max_width / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                )
            return True, frame
        except Exception as exc:  # noqa: BLE001
            logger.warning("[vision] read failed: %s", exc)
            self.release()
            return False, None

    def reconnect(self) -> bool:
        """读失败后调用:按指数退避重连,返回是否恢复。"""
        self.release()
        delay = self._current_delay
        while True:
            logger.info("[vision] reconnect in %.1fs ...", delay)
            time.sleep(delay)
            if self.open():
                return True
            delay = min(delay * 2, self._max_reconnect_delay)

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None
