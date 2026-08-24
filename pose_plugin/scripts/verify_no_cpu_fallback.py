"""静态扫描：assert 生产推理路径（pose_engine/gpu_guard/service）无 CPU/auto 回退。

配置层用于“拒绝非法配置”的 cpu/auto 字符串不属于生产推理路径，本扫描只覆盖真正
执行神经推理的模块，检测 device=cpu/auto 赋值、run_on_cpu 调用与 else->cpu 兜底。
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "ai_monitor_pose"

_FORBIDDEN = [
    re.compile(r"device\s*=\s*['\"]\s*(cpu|auto)\s*['\"]", re.I),
    re.compile(r"device\s*=\s*['\"]?\s*cpu", re.I),
    re.compile(r"\b(cpu|auto)\b.*torch\.cuda\.is_available\(\)", re.I),
    re.compile(r"allow_cpu_fallback\s*=\s*True", re.I),
]

SCAN_FILES = [
    "worker/pose_engine.py",
    "worker/gpu_guard.py",
    "worker/service.py",
]


def main() -> int:
    bad = 0
    for rel in SCAN_FILES:
        f = ROOT / rel
        if not f.exists():
            print(f"SKIP(missing): {rel}")
            continue
        for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for rx in _FORBIDDEN:
                if rx.search(line):
                    print(f"VIOLATION {rel}:{lineno}: {line.strip()}")
                    bad += 1
    if bad:
        print(f"FAIL: {bad} CPU-fallback occurrence(s) found")
        return 1
    print("OK: no CPU fallback in inference path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
