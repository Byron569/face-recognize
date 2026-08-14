"""backend.app.api — 路由层(薄:只做参数校验与响应,业务全部在 services)。"""

from __future__ import annotations
from .router import api_router

__all__ = ["api_router"]
