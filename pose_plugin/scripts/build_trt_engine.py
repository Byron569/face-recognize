"""绕过 modelopt 的 TensorRT engine 构建脚本（备用路径 B）。

背景: TensorRT 11 强类型网络移除了 FP16/INT8 builder flag, ultralytics 官方
导出走 nvidia-modelopt 把 FP16 烘焙进 ONNX; 本脚本用 onnxconverter-common
的 float16 转换替代 modelopt, 并手工复刻 ultralytics 的 .engine 文件格式:

    [4 字节小端 JSON 长度 N][N 字节 JSON 元数据][TRT 序列化 engine 字节]

参照(ultralytics 8.4.127):
    - 写入端:  ultralytics/utils/export/engine.py  onnx2engine 尾部
    - 读取端:  ultralytics/nn/backends/base.py     BaseBackend.engine_header /
               apply_metadata
    - 元数据:  ultralytics/engine/exporter.py      Exporter L915-937 构造

用法(worker venv):
    python scripts/build_trt_engine.py                    # 默认 FP16, 640
    python scripts/build_trt_engine.py --no-fp16          # FP32
    python scripts/build_trt_engine.py --imgsz 480 --out models/x.alt.engine
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from datetime import datetime
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ONNX = _PLUGIN_ROOT / "models" / "yolov8n-pose.onnx"
DEFAULT_OUT = _PLUGIN_ROOT / "models" / "yolov8n-pose.alt.engine"
FALLBACK_ULTRALYTICS_VERSION = "8.4.127"

# engine 格式的 fmt_keys(exporter.py export_formats() "TensorRT" 行), 值为本次导出实际设置
DEFAULT_ENGINE_ARGS = {
    "batch": 1,
    "data": None,
    "dynamic": False,
    "quantize": 16,
    "opset": None,
    "simplify": True,
    "workspace": 4.0,
    "nms": False,
    "fraction": 1.0,
}


# ── 纯函数(header 编解码 / 元数据构造, 无 TRT/onnx 依赖, 可单元测试) ──


def build_metadata(
    imgsz=(640, 640),
    batch=1,
    names=None,
    stride=32,
    task="pose",
    head="Pose",
    channels=3,
    end2end=False,
    kpt_shape=(17, 3),
    kpt_names=None,
    args=None,
    version=None,
    date=None,
) -> dict:
    """构造与 ultralytics engine 导出一致的元数据 dict(复刻 exporter.py L915-937)。

    预训练 pose 模型没有 kpt_names 属性(仅训练时注入)且无 DLA, 故二者默认省略。
    """
    if names is None:
        names = {0: "person"}
    if args is None:
        args = dict(DEFAULT_ENGINE_ARGS)
    if version is None:
        version = _ultralytics_version()
    if date is None:
        date = datetime.now().astimezone().isoformat()
    metadata = {
        "description": "Ultralytics YOLOv8n-pose model",
        "author": "Ultralytics",
        "date": date,
        "version": version,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "stride": int(stride),
        "task": task,
        "head": head,
        "batch": int(batch),
        "imgsz": list(imgsz),
        "names": dict(names),
        "args": dict(args),
        "channels": int(channels),
        "end2end": bool(end2end),
    }
    if task == "pose":
        metadata["kpt_shape"] = list(kpt_shape)
        if kpt_names is not None:
            metadata["kpt_names"] = kpt_names
    return metadata


def _ultralytics_version() -> str:
    """取已安装 ultralytics 版本, 缺失时退回已知常量(纯环境探测, 不影响格式)。"""
    try:
        import ultralytics

        return ultralytics.__version__
    except Exception:
        return FALLBACK_ULTRALYTICS_VERSION


def encode_engine_header(metadata: dict) -> bytes:
    """元数据 → 4 字节小端长度前缀 + JSON 字节(与 ultralytics 写入端等价)。"""
    payload = json.dumps(metadata).encode("utf-8")
    return struct.pack("<I", len(payload)) + payload


def parse_engine_header(data: bytes) -> tuple[int, dict]:
    """按 base.py BaseBackend.engine_header 的逻辑从字节读回 (offset, metadata)。

    非 header(空 / n=0 / 长度越界 / JSON 解析失败)统一返回 (0, {})。
    """
    if len(data) < 4:
        return 0, {}
    n = int.from_bytes(data[:4], byteorder="little")
    if 0 < n <= len(data) - 4:
        try:
            return 4 + n, json.loads(data[4 : 4 + n])
        except ValueError:
            return 0, {}
    return 0, {}


def write_engine_file(out_file, metadata: dict, engine_bytes: bytes) -> Path:
    """写 .engine 文件: 元数据 header + TRT 序列化 engine 字节。"""
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(encode_engine_header(metadata))
        f.write(engine_bytes)
    return out


# ── 重依赖函数(tensorrt / onnx 延迟导入, 模块本身可在无 TRT 环境导入) ──


def convert_onnx_to_fp16(onnx_file, out_file) -> str:
    """FP32 ONNX → FP16 ONNX(onnxconverter-common; keep_io_types 保持 IO 为 FP32)。

    与官方 modelopt 路径的 autocast.convert_to_mixed_precision(keep_io_types=True)
    等价: 内部算子 FP16, 图输入输出仍 FP32, 读取端 binding dtype 不变。
    """
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(onnx_file))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(out_file))
    return str(out_file)


def _onnx_input_shape(onnx_file) -> tuple[int, ...]:
    """读 ONNX 首个输入的 shape, 动态维度记为 -1。"""
    import onnx

    dims = onnx.load(str(onnx_file), load_external_data=False).graph.input[0].type.tensor_type.shape.dim
    return tuple(-1 if d.dim_param else int(d.dim_value) for d in dims)


def _resolve_dim(index: int, dim: int, batch: int, imgsz) -> int:
    """动态维度(-1)的构建取值: N→batch, H/W→imgsz。"""
    if dim != -1:
        return dim
    if index == 0:
        return batch
    if len(imgsz) == 2:
        if index == 2:
            return int(imgsz[0])
        if index == 3:
            return int(imgsz[1])
    return 1


def build_serialized_engine(
    onnx_file,
    workspace_gb: float = 4.0,
    batch: int = 1,
    imgsz=(640, 640),
    fp16: bool = True,
    verbose: bool = False,
) -> bytes:
    """解析 ONNX 并构建 TRT 序列化 engine, 返回 engine 字节。

    TRT11 强类型: FP16 已烘焙进 ONNX 图, 无需(也无法用)builder flag;
    无 STRONGLY_TYPED flag 的旧版本回退普通网络 + FP16 builder flag。
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    if workspace_gb and hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):  # TRT >= 10
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    else:  # TRT 7-9 兼容: 显式 batch + FP16 builder flag
        flags = (
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH")
            else 0
        )
        network = builder.create_network(flags)
        if fp16 and hasattr(trt.BuilderFlag, "FP16"):
            config.set_flag(trt.BuilderFlag.FP16)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_file)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"failed to parse ONNX file: {onnx_file}\n{errors}")

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'  input "{inp.name}" with shape{tuple(inp.shape)} {inp.dtype}')
    for out in outputs:
        print(f'  output "{out.name}" with shape{tuple(out.shape)} {out.dtype}')

    # 固定 batch 的 optimization profile: 仅含动态维度(-1)的输入需要, 静态 ONNX 跳过
    if any(d == -1 for inp in inputs for d in inp.shape):
        profile = builder.create_optimization_profile()
        for inp in inputs:
            shape = tuple(_resolve_dim(i, d, batch, imgsz) for i, d in enumerate(inp.shape))
            profile.set_shape(inp.name, min=shape, opt=shape, max=shape)
        config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed, check logs for errors")
    return bytes(engine)


# ── CLI ────────────────────────────────────────────────────────


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绕过 modelopt 的 TensorRT engine 构建(备用路径 B)")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help=f"源 FP32 ONNX (默认 {DEFAULT_ONNX})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"输出 .alt.engine (默认 {DEFAULT_OUT})")
    parser.add_argument("--imgsz", type=int, default=640, help="输入边长(正方形, 默认 640, 须与 ONNX 一致)")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True, help="FP16 烘焙(默认开)")
    parser.add_argument("--workspace", type=float, default=4.0, help="builder workspace GiB(默认 4)")
    parser.add_argument("--batch", type=int, default=1, help="固定 batch(默认 1, 须与 ONNX 一致)")
    parser.add_argument("--verbose", action="store_true", help="TRT VERBOSE 日志")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(f"ONNX file not found: {onnx_path}")

    imgsz = [args.imgsz, args.imgsz]
    # 与 ONNX 实际输入形状对齐, 元数据撒谎会导致运行期 shape 断言失败
    shape = _onnx_input_shape(onnx_path)
    if shape[0] not in (-1, args.batch) or shape[2:] not in ((-1, -1), tuple(imgsz)):
        raise SystemExit(f"ONNX 输入形状 {shape} 与 --batch {args.batch}/--imgsz {imgsz} 不一致")

    export_args = dict(DEFAULT_ENGINE_ARGS)
    export_args.update(batch=args.batch, quantize=16 if args.fp16 else None, workspace=args.workspace)
    metadata = build_metadata(imgsz=imgsz, batch=args.batch, args=export_args)
    print(f"metadata keys: {sorted(metadata)}")

    build_onnx = onnx_path
    if args.fp16:
        fp16_path = onnx_path.with_suffix(".fp16.onnx")
        print(f"[1/3] FP32 ONNX -> FP16 ONNX: {fp16_path}")
        convert_onnx_to_fp16(onnx_path, fp16_path)
        build_onnx = fp16_path

    print(f"[2/3] building TensorRT engine from {build_onnx} (workspace={args.workspace} GiB, batch={args.batch})...")
    t0 = time.time()
    engine_bytes = build_serialized_engine(
        build_onnx, workspace_gb=args.workspace, batch=args.batch, imgsz=imgsz, fp16=args.fp16, verbose=args.verbose
    )
    print(f"      engine built in {time.time() - t0:.1f}s ({len(engine_bytes) / 1e6:.1f} MB)")

    out = write_engine_file(args.out, metadata, engine_bytes)
    print(f"[3/3] wrote {out} (header {4 + len(json.dumps(metadata))} B + engine {len(engine_bytes)} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
