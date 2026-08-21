"""Bounded WebSocket sender for the latest camera preview frame."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class LatestFrameSender:
    """Send at most one pending frame per WebSocket subscriber.

    ``offer`` is intentionally non-blocking and must be called from the
    subscriber's asyncio event loop. If a frame is already waiting while the
    socket is busy, the waiting frame is replaced by the newest one.
    """

    def __init__(
        self,
        websocket: Any,
        on_disconnect: Callable[[], Any] | None = None,
        on_sent: Callable[[], Any] | None = None,
    ) -> None:
        self.websocket = websocket
        self._on_disconnect = on_disconnect
        self._on_sent = on_sent
        self._pending: tuple[bytes, str] | None = None
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self.dropped_frames = 0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    def offer(self, frame_packet: bytes, detections_json: str) -> None:
        if self._closed:
            return
        if self._pending is not None:
            self.dropped_frames += 1
        self._pending = (frame_packet, detections_json)
        self._idle.clear()
        self._wake.set()

    async def wait_until_idle(self) -> None:
        await self._idle.wait()

    async def close(self) -> None:
        if self._closed and self._task is None:
            return
        self._closed = True
        self._pending = None
        self._wake.set()
        task = self._task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._task = None
        self._idle.set()

    async def _run(self) -> None:
        try:
            while not self._closed:
                await self._wake.wait()
                self._wake.clear()
                while not self._closed:
                    pending = self._pending
                    if pending is None:
                        self._idle.set()
                        break
                    self._pending = None
                    packet, detections_json = pending
                    try:
                        await self.websocket.send_bytes(packet)
                        await self.websocket.send_text(detections_json)
                    except Exception:  # noqa: BLE001
                        self._closed = True
                        self._pending = None
                        self._idle.set()
                        self._notify_disconnect()
                        return
                    self._notify_sent()
        except asyncio.CancelledError:
            self._idle.set()
            raise
        finally:
            self._idle.set()

    def _notify_disconnect(self) -> None:
        if self._on_disconnect is None:
            return
        try:
            result = self._on_disconnect()
            if inspect.isawaitable(result) and not isinstance(result, asyncio.Future):
                asyncio.create_task(result)
        except Exception:  # noqa: BLE001
            logger.exception("preview subscriber disconnect callback failed")

    def _notify_sent(self) -> None:
        if self._on_sent is None:
            return
        try:
            result = self._on_sent()
            if inspect.isawaitable(result) and not isinstance(result, asyncio.Future):
                asyncio.create_task(result)
        except Exception:  # noqa: BLE001
            logger.exception("preview subscriber sent callback failed")
