# Binary Preview Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Replace Base64 JSON camera preview delivery with bounded binary JPEG streaming so browser previews are smooth and a slow viewer cannot create latency for other viewers.

**Architecture:** Keep the current camera pipeline and one-per-camera JPEG encoder thread. The encoder produces one binary packet per camera. Each WebSocket subscriber owns a single-slot sender worker that replaces stale unsent data. React retains only the newest undrawn frame, decodes it with createImageBitmap, and draws it in requestAnimationFrame.

**Tech Stack:** FastAPI/Starlette WebSocket, Python asyncio, OpenCV, React 18, TypeScript, Canvas 2D, Vite.

---

## File structure

- Create: backend/app/services/stream_protocol.py — packet constants and pure pack/unpack helpers.
- Create: backend/app/services/stream_subscriber.py — one bounded sender worker per WebSocket.
- Modify: backend/app/services/pipeline_manager.py — fanout, stream settings, and counters.
- Modify: backend/app/api/ws.py and backend/app/api/system.py — async lifecycle and metrics response.
- Modify: backend/app/config.py and configs/default.yaml — matching measured defaults.
- Create: tests/test_stream_protocol.py — protocol and slow-client tests without GPU.
- Create: frontend/src/stream/frameProtocol.ts — browser packet parser.
- Modify: frontend/src/hooks/useWebSocket.ts and frontend/src/components/CameraCell.tsx — mixed payload handling and latest-frame rendering.
- Modify: README.md and docs/API.md — protocol and operator guidance.

### Task 1: Define the versioned binary packet contract

**Files:**

- Create: backend/app/services/stream_protocol.py
- Create: tests/test_stream_protocol.py

- [ ] **Step 1: Write failing round-trip tests**

~~~python
from backend.app.services.stream_protocol import (
    FRAME_PACKET_TYPE,
    pack_jpeg_frame,
    unpack_jpeg_frame,
)


def test_jpeg_packet_round_trip_preserves_frame_id_and_bytes():
    jpeg = b"\xff\xd8test-jpeg\xff\xd9"
    packet = pack_jpeg_frame(frame_id=513, jpeg=jpeg)

    assert packet[0] == FRAME_PACKET_TYPE
    assert len(packet) == 9 + len(jpeg)
    assert unpack_jpeg_frame(packet) == (513, jpeg)


def test_jpeg_packet_rejects_wrong_type_and_short_headers():
    import pytest

    with pytest.raises(ValueError, match="frame packet"):
        unpack_jpeg_frame(b"")
    with pytest.raises(ValueError, match="unsupported"):
        unpack_jpeg_frame(b"\x02" + b"\x00" * 8)
~~~

- [ ] **Step 2: Verify the test fails**

Run:

~~~powershell
python -m pytest tests/test_stream_protocol.py -q
~~~

Expected: import failure because stream_protocol.py does not exist.

- [ ] **Step 3: Implement the minimal protocol helper**

~~~python
"""Versioned binary camera-preview packet helpers."""

from __future__ import annotations

FRAME_PACKET_TYPE = 0x01
FRAME_HEADER_SIZE = 9


def pack_jpeg_frame(frame_id: int, jpeg: bytes) -> bytes:
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    return bytes((FRAME_PACKET_TYPE,)) + frame_id.to_bytes(8, "big") + jpeg


def unpack_jpeg_frame(packet: bytes) -> tuple[int, bytes]:
    if len(packet) < FRAME_HEADER_SIZE:
        raise ValueError("invalid frame packet: header is incomplete")
    if packet[0] != FRAME_PACKET_TYPE:
        raise ValueError(f"unsupported frame packet type: {packet[0]}")
    return int.from_bytes(packet[1:9], "big"), packet[9:]
~~~

The wire format is exactly: one message-type byte, eight big-endian frame-id bytes, then raw JPEG bytes. Do not add Base64, JSON, or another image copy.

- [ ] **Step 4: Verify the contract**

Run:

~~~powershell
python -m pytest tests/test_stream_protocol.py -q
~~~

Expected: 2 passed.

- [ ] **Step 5: Commit**

~~~powershell
git add backend/app/services/stream_protocol.py tests/test_stream_protocol.py
git commit -m "feat(stream): define binary JPEG frame protocol"
~~~

### Task 2: Isolate slow WebSocket viewers with a one-slot sender

**Files:**

- Create: backend/app/services/stream_subscriber.py
- Modify: tests/test_stream_protocol.py

- [ ] **Step 1: Add a failing slow-viewer test**

~~~python
import asyncio
import pytest

from backend.app.services.stream_subscriber import LatestFrameSender


class SlowWebSocket:
    def __init__(self):
        self.sent = []
        self.release_first_send = asyncio.Event()

    async def send_bytes(self, packet):
        if not self.sent:
            await self.release_first_send.wait()
        self.sent.append(("bytes", packet))

    async def send_text(self, message):
        self.sent.append(("text", message))


@pytest.mark.asyncio
async def test_slow_subscriber_replaces_pending_frame_with_latest_one():
    ws = SlowWebSocket()
    sender = LatestFrameSender(ws)
    sender.start()

    sender.offer(b"frame-1", '{"frame_id":1}')
    await asyncio.sleep(0)
    sender.offer(b"frame-2", '{"frame_id":2}')
    sender.offer(b"frame-3", '{"frame_id":3}')
    ws.release_first_send.set()
    await sender.wait_until_idle()

    assert ("bytes", b"frame-1") in ws.sent
    assert ("bytes", b"frame-3") in ws.sent
    assert ("bytes", b"frame-2") not in ws.sent
    assert sender.dropped_frames == 1
    await sender.close()
~~~

- [ ] **Step 2: Verify the test fails**

Run:

~~~powershell
python -m pytest tests/test_stream_protocol.py::test_slow_subscriber_replaces_pending_frame_with_latest_one -q
~~~

Expected: import failure for LatestFrameSender.

- [ ] **Step 3: Implement the sender public contract**

~~~python
class LatestFrameSender:
    def __init__(self, websocket, on_disconnect=None) -> None: ...
    def start(self) -> None: ...
    def offer(self, frame_packet: bytes, detections_json: str) -> None: ...
    async def wait_until_idle(self) -> None: ...
    async def close(self) -> None: ...
~~~

Implementation rules:

- offer is called only from the FastAPI event loop;
- store either no pending item or one pair of frame packet and detections JSON;
- replacing a pending item increments dropped_frames;
- one background task waits on an asyncio Event, sends bytes then text, and repeats;
- a send exception exits the worker and invokes on_disconnect; if that callback returns a coroutine, schedule it with asyncio.create_task rather than leaving it unawaited;
- close wakes and awaits/cancels the worker cleanly;
- never use an unbounded queue, a task per frame, or socket send while holding a manager lock.

- [ ] **Step 4: Verify the sender**

Run:

~~~powershell
python -m pytest tests/test_stream_protocol.py -q
~~~

Expected: all protocol and sender tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add backend/app/services/stream_subscriber.py tests/test_stream_protocol.py
git commit -m "feat(stream): isolate slow preview subscribers"
~~~

### Task 3: Integrate binary fanout and stream metrics

**Files:**

- Modify: backend/app/services/pipeline_manager.py
- Modify: backend/app/api/ws.py
- Modify: backend/app/api/system.py
- Modify: tests/test_stream_protocol.py

- [ ] **Step 1: Add a failing metric test**

~~~python
def test_stream_metrics_report_encoding_and_drop_counters():
    from backend.app.services.pipeline_manager import StreamMetrics

    metrics = StreamMetrics()
    metrics.record_enqueue()
    metrics.record_encoded(jpeg_bytes=1234)
    metrics.record_encode_drop()
    metrics.record_subscriber_drops(2)

    snapshot = metrics.snapshot()
    assert snapshot["encoded_frames"] == 1
    assert snapshot["encode_dropped_frames"] == 1
    assert snapshot["subscriber_dropped_frames"] == 2
    assert snapshot["avg_jpeg_bytes"] == 1234
~~~

- [ ] **Step 2: Verify it fails**

Run:

~~~powershell
python -m pytest tests/test_stream_protocol.py::test_stream_metrics_report_encoding_and_drop_counters -q
~~~

Expected: import failure for StreamMetrics.

- [ ] **Step 3: Make fanout bounded**

In PipelineManager replace the set-only connection registry with a per-camera mapping:

~~~python
self._subscribers: dict[str, dict[WebSocket, LatestFrameSender]] = defaultdict(dict)
self._stream_metrics: dict[str, StreamMetrics] = {}
~~~

Update the existing throttled encoder gate to inspect self._subscribers for the camera rather than the removed self._ws_connections set. No subscriber means no frame copy and no JPEG encoding.

Make registration asynchronous:

~~~python
async def register_ws(self, camera_id: str, ws: WebSocket) -> None:
    sender = LatestFrameSender(
        ws,
        on_disconnect=lambda: asyncio.create_task(self.unregister_ws(camera_id, ws)),
    )
    sender.start()
    with self._lock:
        self._subscribers[camera_id][ws] = sender


async def unregister_ws(self, camera_id: str, ws: WebSocket) -> None:
    with self._lock:
        sender = self._subscribers[camera_id].pop(ws, None)
    if sender is not None:
        await sender.close()
~~~

Keep JPEG encoding in the existing per-camera thread. After cv2.imencode, schedule a single event-loop coroutine that builds the protocol packet and calls sender.offer for a copied list of subscribers. It must not await a socket send directly.

~~~python
async def _fanout_encoded_frame(self, camera_id, jpeg, persons, frame_id):
    frame_packet = pack_jpeg_frame(frame_id, jpeg)
    detections_json = json.dumps(
        {"type": "detections", "frame_id": frame_id, "persons": persons}
    )
    with self._lock:
        subscribers = list(self._subscribers.get(camera_id, {}).values())
    for sender in subscribers:
        before = sender.dropped_frames
        sender.offer(frame_packet, detections_json)
        self._stream_metrics[camera_id].record_subscriber_drops(
            sender.dropped_frames - before
        )
~~~

In backend/app/api/ws.py, await both lifecycle calls. Expose a stream object under each camera in api/system/metrics with preview_enqueue_fps, encoded_fps, sent_fps, encode_dropped_frames, subscriber_dropped_frames, and avg_jpeg_bytes. Do not remove current metrics fields.

- [ ] **Step 4: Run server checks**

Run:

~~~powershell
python -m pytest tests/test_stream_protocol.py tests/test_backend_logic.py tests/test_vision_core.py -q
python -c "from backend.app.services.pipeline_manager import PipelineManager; print('pipeline manager import ok')"
~~~

Expected: tests pass and the import prints pipeline manager import ok.

- [ ] **Step 5: Commit**

~~~powershell
git add backend/app/services/pipeline_manager.py backend/app/api/ws.py backend/app/api/system.py backend/app/services/stream_subscriber.py tests/test_stream_protocol.py
git commit -m "feat(stream): send latest binary preview frames per viewer"
~~~

### Task 4: Make preview settings per-camera and measured

**Files:**

- Modify: configs/default.yaml
- Modify: backend/app/config.py
- Modify: backend/app/services/pipeline_manager.py
- Modify: tests/test_backend_logic.py

- [ ] **Step 1: Add configuration cascade coverage**

~~~python
def test_camera_stream_config_overrides_all_default_preview_values():
    cfg = build_camera_config(
        "desktop",
        {"stream": {"max_height": 720, "jpeg_quality": 68, "push_fps": 20}},
    )
    assert cfg["stream"] == {
        "max_height": 720,
        "jpeg_quality": 68,
        "push_fps": 20,
    }
~~~

- [ ] **Step 2: Run it**

Run:

~~~powershell
python -m pytest tests/test_backend_logic.py::test_camera_stream_config_overrides_all_default_preview_values -q
~~~

Expected: pass; it protects deep-merge behavior before defaults change.

- [ ] **Step 3: Apply explicit defaults and clamp values**

Use the same defaults in YAML and Settings:

~~~yaml
stream:
  max_height: 480
  jpeg_quality: 70
  push_fps: 20
~~~

At camera start, resolve and store per-camera settings:

~~~python
max_height = max(0, int(stream_cfg.get("max_height", settings.stream_max_height)))
jpeg_quality = min(100, max(1, int(stream_cfg.get("jpeg_quality", settings.stream_jpeg_quality))))
push_fps = min(30, max(1, int(stream_cfg.get("push_fps", settings.stream_push_fps))))
~~~

Use that camera jpeg_quality in cv2.imencode. Keep explicit database settings such as stream.max_height equal to 0 untouched; System Settings remains the operator-controlled way to move existing cameras to 480 or 720.

- [ ] **Step 4: Run configuration regression tests**

Run:

~~~powershell
python -m pytest tests/test_backend_logic.py -q
~~~

Expected: all tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add configs/default.yaml backend/app/config.py backend/app/services/pipeline_manager.py tests/test_backend_logic.py
git commit -m "perf(stream): set measured preview defaults"
~~~

### Task 5: Render binary frames in React with no decode backlog

**Files:**

- Create: frontend/src/stream/frameProtocol.ts
- Modify: frontend/src/hooks/useWebSocket.ts
- Modify: frontend/src/components/CameraCell.tsx

- [ ] **Step 1: Add the browser packet parser**

~~~ts
export const FRAME_PACKET_TYPE = 0x01;
export const FRAME_HEADER_SIZE = 9;

export type DecodedFramePacket = { frameId: number; jpeg: Blob };

export function parseFramePacket(data: ArrayBuffer): DecodedFramePacket {
  if (data.byteLength < FRAME_HEADER_SIZE) {
    throw new Error('frame packet header is incomplete');
  }
  const view = new DataView(data);
  if (view.getUint8(0) !== FRAME_PACKET_TYPE) {
    throw new Error('unsupported frame packet type');
  }
  const frameId = Number(view.getBigUint64(1, false));
  return {
    frameId,
    jpeg: new Blob([data.slice(FRAME_HEADER_SIZE)], { type: 'image/jpeg' }),
  };
}
~~~

- [ ] **Step 2: Extend the WebSocket hook safely**

Change useWebSocket to set ws.binaryType equal to arraybuffer when requested. Its onmessage handler parses only strings, drops ping messages, and forwards ArrayBuffer values unchanged. Existing AlertBanner must continue to receive JSON objects.

- [ ] **Step 3: Replace the Data URI renderer in CameraCell**

Use these refs:

~~~ts
const detectionsByFrameRef = useRef(new Map<number, Detection[]>());
const pendingFrameRef = useRef<DecodedFramePacket | null>(null);
const drawScheduledRef = useRef(false);
const canvasSizeRef = useRef({ width: 0, height: 0 });
~~~

When a binary packet arrives, parse it and replace pendingFrameRef.current. Schedule exactly one requestAnimationFrame callback. The callback takes the newest pending frame, creates an ImageBitmap from its JPEG Blob, sets the canvas pixel size only if dimensions changed, draws the bitmap, draws only detections matching the same frame id, closes the ImageBitmap, then schedules again if another pending frame arrived.

When a detections JSON message arrives, store it in a bounded Map keyed by frame_id. After drawing frame n, remove entries below n minus 30. Remove the existing new Image, Base64 Data URI, and per-frame onload logic.

- [ ] **Step 4: Build frontend**

Run in frontend:

~~~powershell
npm run build
~~~

Expected: TypeScript and Vite exit 0.

- [ ] **Step 5: Commit**

~~~powershell
git add frontend/src/stream/frameProtocol.ts frontend/src/hooks/useWebSocket.ts frontend/src/components/CameraCell.tsx
git commit -m "perf(frontend): render binary camera preview frames"
~~~

### Task 6: Verify with local video and document operator controls

**Files:**

- Modify: README.md
- Modify: docs/API.md

- [ ] **Step 1: Document the operational contract**

Document that preview resolution does not alter InsightFace inference; 480p is the default and 720p is the high-clarity LAN option. Document that existing cameras set to native must be explicitly changed in System Settings. Document the binary_jpeg_v1 9-byte header for external consumers.

- [ ] **Step 2: Measure no-viewer and viewer cases**

Use D:\test6.mp4 through cam-0. Start, inspect metrics, connect one normal browser and one deliberately throttled browser, then stop:

~~~powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/cameras/cam-0/start
Invoke-RestMethod http://127.0.0.1:8000/api/system/metrics | ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/cameras/cam-0/stop
~~~

Acceptance:

- no viewer means encoded_fps equals zero while pipeline FPS remains healthy;
- 480 or 720p at 20 FPS reaches at least 18 displayed FPS locally;
- a slow viewer increments its drop counter without lowering the fast viewer below 18 FPS;
- boxes remain aligned with their own frame IDs;
- no camera remains running after the test.

- [ ] **Step 3: Run full regression**

~~~powershell
python -m pytest tests -q
~~~

Run in frontend:

~~~powershell
npm run build
~~~

Expected: both commands exit 0.

- [ ] **Step 4: Commit docs**

~~~powershell
git add README.md docs/API.md
git commit -m "docs: explain binary preview stream operation"
~~~
