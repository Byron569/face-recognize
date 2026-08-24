from __future__ import annotations
from fastapi import APIRouter

from ..config import test_mode_enabled, test_run_id

router = APIRouter()


@router.get("/health")
async def health():
    payload = {"status": "ok"}
    # 非敏感 marker:仅显式 AI_MONITOR_TEST_MODE 启用才暴露;生产默认关闭绝不泄露
    if test_mode_enabled():
        payload["test_mode"] = True
        payload["test_run_id"] = test_run_id()
    return payload