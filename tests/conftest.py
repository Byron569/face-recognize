"""pytest 配置:项目根加入 sys.path,保证 vision 可导入。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
