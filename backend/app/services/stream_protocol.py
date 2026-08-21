"""Versioned binary camera-preview packet helpers."""

from __future__ import annotations

FRAME_PACKET_TYPE = 0x01
FRAME_HEADER_SIZE = 9


def pack_jpeg_frame(frame_id: int, jpeg: bytes) -> bytes:
    """Encode a JPEG preview frame as ``type + uint64 frame id + JPEG``."""
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    return bytes((FRAME_PACKET_TYPE,)) + frame_id.to_bytes(8, "big") + jpeg


def unpack_jpeg_frame(packet: bytes) -> tuple[int, bytes]:
    """Decode a packet produced by :func:`pack_jpeg_frame`."""
    if len(packet) < FRAME_HEADER_SIZE:
        raise ValueError("invalid frame packet: header is incomplete")
    if packet[0] != FRAME_PACKET_TYPE:
        raise ValueError(f"unsupported frame packet type: {packet[0]}")
    return int.from_bytes(packet[1:FRAME_HEADER_SIZE], "big"), packet[FRAME_HEADER_SIZE:]
