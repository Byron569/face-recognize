"""
vision.engine — InsightFace 官方 FaceAnalysis 封装。

不做任何自研算法,只做三件事:
    1. 按配置构造 FaceAnalysis(模型包 / 模型根目录 / provider 全部可注入);
    2. 推理优先 GPU(默认 CUDAExecutionProvider,不可用时自动降级 CPU 并告警);
    3. 把官方 Face 对象规整为内核统一的 FaceResult。
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .config import VisionConfig
from .events import FaceResult

logger = logging.getLogger(__name__)


class InsightFaceEngine:
    """InsightFace 检测 + 识别一体化推理引擎(线程安全:onnxruntime session 支持并发 run)。"""

    def __init__(self, config: VisionConfig):
        self._config = config
        self._device = self._resolve_device(config)
        self._providers = (
            config.providers_cuda if self._device == "cuda" else config.providers_cpu
        )
        self._model_root = self._resolve_model_root(config)

        # 延迟导入,保证 vision 包在未安装 insightface 时仍可被导入(如单元测试环境)
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name=config.model_pack,
            root=self._model_root,
            allowed_modules=["detection", "recognition"],
            providers=self._providers,
        )
        ctx_id = 0 if self._device == "cuda" else -1
        self._app.prepare(ctx_id=ctx_id, det_thresh=config.det_thresh, det_size=config.det_size)

        self._report_backend()

    @staticmethod
    def _resolve_model_root(config: VisionConfig) -> str:
        """把配置的 models_root 转换为 insightface 官方语义的 root。

        官方语义: FaceAnalysis(root) 实际查找 {root}/models/{model_pack}/*.onnx。

        兼容两种项目布局:
          布局A(官方): {models_root}/models/{pack}/*.onnx → 直接传 models_root
          布局B(直觉): {models_root}/{pack}/*.onnx      → 传 models_root 的父目录
        """
        from pathlib import Path

        root = Path(config.models_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        pack = config.model_pack

        official = root / "models" / pack
        direct = root / pack
        if any(direct.glob("*.onnx")):
            return str(root.parent)     # 布局B:{parent}/models/{pack} == {root}/{pack} ✓
        if any(official.glob("*.onnx")):
            return str(root)            # 布局A ✓
        logger = logging.getLogger(__name__)
        logger.warning(
            "[vision] %s 下未找到 %s 的 onnx 模型,insightface 将尝试自动下载",
            root, pack,
        )
        return str(root)

    # ── 设备解析 ──────────────────────────────────────────

    @staticmethod
    def _resolve_device(config: VisionConfig) -> str:
        if config.device in ("cuda", "cpu"):
            return config.device
        # auto: 检测可用 provider
        import onnxruntime as ort

        return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"

    def _report_backend(self) -> None:
        det_model = self._app.models.get("detection")
        session = getattr(det_model, "session", None)
        actual = session.get_providers() if session is not None else list(self._providers)
        if self._device == "cuda" and "CUDAExecutionProvider" not in actual:
            logger.warning(
                "[vision] 请求 CUDA 但实际使用 %s — 请检查 onnxruntime-gpu 与 NVIDIA 驱动",
                actual,
            )

        model_dir, detection_files, recognition_files = self._model_files()
        if not model_dir.is_dir():
            logger.warning(
                "[vision] model pack files missing: pack=%s model_dir=%s; "
                "InsightFace may attempt to auto-download",
                self._config.model_pack,
                model_dir,
            )
        if not detection_files:
            logger.warning(
                "[vision] detection model file missing: pack=%s model_dir=%s; "
                "InsightFace may attempt to auto-download",
                self._config.model_pack,
                model_dir,
            )
        if not recognition_files:
            logger.warning(
                "[vision] recognition model file missing: pack=%s model_dir=%s; "
                "InsightFace may attempt to auto-download",
                self._config.model_pack,
                model_dir,
            )
        logger.info(
            "[vision] engine ready: pack=%s detection_models=%s recognition_models=%s "
            "requested_device=%s device=%s providers=%s det_size=%s",
            self._config.model_pack,
            [str(path) for path in detection_files],
            [str(path) for path in recognition_files],
            self._config.device,
            self._device,
            actual,
            self._config.det_size,
        )

    def _model_files(self):
        from pathlib import Path

        model_dir = (Path(self._model_root) / "models" / self._config.model_pack).resolve()
        files = sorted(model_dir.glob("*.onnx"))
        detection_files = [path for path in files if path.name.lower().startswith("det")]
        recognition_files = [
            path
            for path in files
            if path not in detection_files
            and (
                path.name.lower().startswith("w600k")
                or "recogn" in path.name.lower()
                or "arcface" in path.name.lower()
            )
        ]
        if not recognition_files:
            recognition_files = [path for path in files if path not in detection_files]
        return model_dir, detection_files, recognition_files

    # ── 推理入口 ──────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[FaceResult]:
        """对单帧执行 检测 + 关键点 + embedding,返回 FaceResult 列表。"""
        faces = self._app.get(frame, max_num=self._config.max_faces)
        results: List[FaceResult] = []
        for f in faces:
            bbox = tuple(float(v) for v in f.bbox[:4])
            kps = None
            if f.kps is not None:
                kps = [(float(k[0]), float(k[1])) for k in f.kps]
            results.append(
                FaceResult(
                    bbox=bbox,
                    det_score=float(f.det_score),
                    kps=kps,
                    embedding=f.normed_embedding if f.embedding is not None else None,
                )
            )
        return results

    @property
    def device(self) -> str:
        return self._device

    @property
    def providers(self) -> List[str]:
        return list(self._providers)

    def close(self) -> None:
        # FaceAnalysis 无显式资源释放;清空引用由 GC 回收 session
        self._app = None
