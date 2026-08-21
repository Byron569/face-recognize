import asyncio

import pytest


def test_jpeg_packet_round_trip_preserves_frame_id_and_bytes():
    from backend.app.services.stream_protocol import (
        FRAME_PACKET_TYPE,
        pack_jpeg_frame,
        unpack_jpeg_frame,
    )

    jpeg = b"\xff\xd8test-jpeg\xff\xd9"
    packet = pack_jpeg_frame(frame_id=513, jpeg=jpeg)

    assert packet[0] == FRAME_PACKET_TYPE
    assert len(packet) == 9 + len(jpeg)
    assert unpack_jpeg_frame(packet) == (513, jpeg)


def test_jpeg_packet_rejects_wrong_type_and_short_headers():
    from backend.app.services.stream_protocol import unpack_jpeg_frame

    with pytest.raises(ValueError, match="frame packet"):
        unpack_jpeg_frame(b"")
    with pytest.raises(ValueError, match="unsupported"):
        unpack_jpeg_frame(b"\x02" + b"\x00" * 8)


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


def test_slow_subscriber_replaces_pending_frame_with_latest_one():
    from backend.app.services.stream_subscriber import LatestFrameSender

    async def run_test():
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

    asyncio.run(run_test())


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


def test_preview_detection_boxes_scale_with_encoded_frame():
    from backend.app.services.pipeline_manager import _scale_persons_for_preview

    persons = [{"track_id": 1, "bbox": [10, 20, 30, 40], "identity": "Alice"}]

    assert _scale_persons_for_preview(persons, 0.5, 0.5) == [
        {"track_id": 1, "bbox": [5.0, 10.0, 15.0, 20.0], "identity": "Alice"}
    ]
