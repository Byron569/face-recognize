"""build_trt_engine.py 纯函数单元测试（主 venv 可跑，无 TensorRT/onnx 依赖）。

覆盖三块:
    1. engine header 线格式: 4 字节小端长度 + JSON, 与 ultralytics
       nn/backends/base.py BaseBackend.engine_header 的读取逻辑 round-trip。
    2. 元数据 dict 构造: 键集合与 ultralytics engine/exporter.py L915-937 一致。
    3. 读取端语义: json round-trip 后 apply_metadata 的关键字段解析结果。
"""
from __future__ import annotations

import contextlib
import json
import struct

from scripts.build_trt_engine import (
    DEFAULT_ENGINE_ARGS,
    build_metadata,
    encode_engine_header,
    parse_engine_header,
)

FIXED_DATE = "2026-08-25T12:00:00+08:00"
FIXED_VERSION = "8.4.127"
FAKE_ENGINE = bytes(range(256)) * 8  # 模拟 TRT 序列化字节(非 JSON)


def _ultralytics_engine_header(path):
    """复刻 ultralytics/nn/backends/base.py BaseBackend.engine_header 读取逻辑。"""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(4), byteorder="little")  # 4 字节小端 JSON 长度
        if 0 < n <= f.seek(0, 2) - 4:  # 长度越界则不是 header
            f.seek(4)
            with contextlib.suppress(ValueError):  # engine 字节不是 JSON
                return 4 + n, json.loads(f.read(n))
    return 0, {}


def _meta(**overrides):
    base = dict(date=FIXED_DATE, version=FIXED_VERSION)
    base.update(overrides)
    return build_metadata(**base)


def _json_normalize(obj):
    """JSON round-trip 后的形态(int 键→str 键, tuple→list)。"""
    return json.loads(json.dumps(obj))


# ── header 线格式 ──────────────────────────────────────────────


def test_encode_header_wire_format():
    meta = _meta()
    blob = encode_engine_header(meta)
    payload = json.dumps(meta).encode("utf-8")
    assert blob[:4] == struct.pack("<I", len(payload))  # 小端无符号长度前缀
    assert blob[4:] == payload
    assert len(blob) == 4 + len(payload)


def test_header_round_trip_in_memory():
    meta = _meta()
    blob = encode_engine_header(meta) + FAKE_ENGINE
    offset, parsed = parse_engine_header(blob)
    assert offset == 4 + len(json.dumps(meta))
    assert parsed == _json_normalize(meta)
    assert blob[offset:] == FAKE_ENGINE  # engine 字节原样保留在 offset 之后


def test_header_round_trip_file(tmp_path):
    """写入文件后, 用复刻 ultralytics 读取逻辑的独立实现读回。"""
    meta = _meta()
    f = tmp_path / "yolov8n-pose.alt.engine"
    f.write_bytes(encode_engine_header(meta) + FAKE_ENGINE)
    offset, parsed = _ultralytics_engine_header(f)
    assert offset == 4 + len(json.dumps(meta))
    assert parsed == _json_normalize(meta)
    with open(f, "rb") as fh:
        fh.seek(offset)
        assert fh.read() == FAKE_ENGINE


def test_parse_header_rejects_bad_headers():
    assert parse_engine_header(b"") == (0, {})
    assert parse_engine_header(b"\x00\x00\x00\x00abc") == (0, {})  # n=0 非 header
    assert parse_engine_header(b"\xff\xff\xff\x7f" + b"x" * 8) == (0, {})  # 长度越界
    # 声称的长度未越界但内容不是 JSON → suppress(ValueError) → 无 header
    assert parse_engine_header(b"\x05\x00\x00\x00not-json") == (0, {})


def test_header_non_ascii_metadata_round_trip():
    """字节长度前缀对非 ASCII 元数据同样成立(逐字节计数)。"""
    meta = _meta(names={0: "人"})
    blob = encode_engine_header(meta) + FAKE_ENGINE
    offset, parsed = parse_engine_header(blob)
    assert parsed == _json_normalize(meta)  # 默认 ensure_ascii 转义后仍可完整读回
    assert blob[offset:] == FAKE_ENGINE


# ── 元数据键完整性(对照 exporter.py L915-937) ──────────────────


def test_metadata_keys_complete():
    meta = _meta()
    assert set(meta) == {
        "description",
        "author",
        "date",
        "version",
        "license",
        "docs",
        "stride",
        "task",
        "head",
        "batch",
        "imgsz",
        "names",
        "args",
        "channels",
        "end2end",
        "kpt_shape",
    }  # 预训练 pose 模型无 kpt_names/dla → 省略


def test_metadata_core_values():
    meta = _meta()
    assert meta["description"].startswith("Ultralytics YOLOv8n-pose model")
    assert meta["author"] == "Ultralytics"
    assert meta["date"] == FIXED_DATE
    assert meta["version"] == FIXED_VERSION
    assert meta["license"] == "AGPL-3.0 License (https://ultralytics.com/license)"
    assert meta["docs"] == "https://docs.ultralytics.com"
    assert meta["stride"] == 32
    assert meta["task"] == "pose"
    assert meta["head"] == "Pose"
    assert meta["batch"] == 1
    assert meta["imgsz"] == [640, 640]
    assert meta["names"] == {0: "person"}
    assert meta["channels"] == 3
    assert meta["end2end"] is False
    assert meta["kpt_shape"] == [17, 3]


def test_metadata_args_keys_match_engine_fmt():
    """args 键集合 == engine 格式 fmt_keys(读取端只用 nms/dynamic, 其余为导出信息)。"""
    meta = _meta()
    assert set(meta["args"]) == {
        "batch",
        "data",
        "dynamic",
        "quantize",
        "opset",
        "simplify",
        "workspace",
        "nms",
        "fraction",
    }
    assert meta["args"]["nms"] is False
    assert meta["args"]["dynamic"] is False
    assert set(DEFAULT_ENGINE_ARGS) == set(meta["args"])


def test_metadata_overrides():
    meta = _meta(imgsz=(480, 640), batch=2, names={0: "dog", 1: "cat"}, kpt_shape=(5, 3))
    assert meta["imgsz"] == [480, 640]
    assert meta["batch"] == 2
    assert meta["names"] == {0: "dog", 1: "cat"}
    assert meta["kpt_shape"] == [5, 3]
    meta2 = _meta(args={"batch": 1, "dynamic": False, "nms": True})
    assert meta2["args"]["nms"] is True  # 调用方可整体覆盖 args


# ── 读取端语义(apply_metadata 的关键字段) ──────────────────────


def test_metadata_reader_semantics_after_json_roundtrip():
    loaded = _json_normalize(_meta())
    # stride/batch/channels: int() 转换后不变
    assert int(loaded["stride"]) == 32
    assert int(loaded["batch"]) == 1
    assert int(loaded["channels"]) == 3
    # names: JSON 后键变 str(dict 非 str → 不走 literal_eval, 保持 str 键)
    assert loaded["names"] == {"0": "person"}
    # end2end / dynamic 解析(apply_metadata L214-215)
    assert (loaded.get("end2end", False) or loaded.get("args", {}).get("nms", False)) is False
    assert loaded.get("args", {}).get("dynamic", True) is False
    # imgsz/kpt_shape 为 list
    assert loaded["imgsz"] == [640, 640]
    assert loaded["kpt_shape"] == [17, 3]
