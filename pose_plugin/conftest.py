# 保证姿态项目根目录可被测试导入（tests/* 与 ai_monitor_pose 基准）。
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
