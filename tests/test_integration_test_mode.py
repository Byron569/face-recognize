"""阶段9 宿主:集成测试模式安全门禁(默认关闭)。

验证:
    - 默认(未设 AI_MONITOR_TEST_MODE)时 /health 绝不暴露任何测试 marker,
      也不含 test_run_id;
    - 显式开启 AI_MONITOR_TEST_MODE=true 且注入 AI_MONITOR_TEST_RUN_ID 时,
      /health 返回 test_mode=true 与精确匹配的 test_run_id(非敏感 marker);
    - 仅设置 run_id 而 mode 未开时仍为关闭,不能由配置遗漏被意外打开;
    - 生产配置不能通过普通 API 修改/打开该 marker(读取即关闭,无写入口)。

先 RED(此时 config/health 的 helper 尚不存在),实现后 GREEN。
"""
from __future__ import annotations

import os
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.health import router as health_router

_MODE = "AI_MONITOR_TEST_MODE"
_RUN = "AI_MONITOR_TEST_RUN_ID"


@pytest.fixture(autouse=True)
def _clear_env():
    """每个测试前清空相关 env,隔绝跨测试泄漏。"""
    for key in (_MODE, _RUN):
        os.environ.pop(key, None)
    yield
    for key in (_MODE, _RUN):
        os.environ.pop(key, None)


@pytest.fixture()
def env_mode_on() -> dict:
    run_id = str(uuid.uuid4())
    os.environ[_MODE] = "true"
    os.environ[_RUN] = run_id
    return {"AI_MONITOR_TEST_RUN_ID": run_id}


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(health_router)
    return app


# ── helper 层 ───────────────────────────────────────────

def test_mode_defaults_off() -> None:
    from backend.app.config import test_mode_enabled

    assert test_mode_enabled() is False


def test_mode_off_means_empty_run_id() -> None:
    from backend.app.config import test_run_id

    assert test_run_id() == ""


def test_mode_on_returns_exact_run_id(env_mode_on) -> None:
    from backend.app.config import test_mode_enabled, test_run_id

    assert test_mode_enabled() is True
    assert test_run_id() == env_mode_on["AI_MONITOR_TEST_RUN_ID"]


def test_run_id_alone_keeps_mode_off() -> None:
    from backend.app.config import test_mode_enabled, test_run_id

    os.environ[_RUN] = str(uuid.uuid4())
    assert test_mode_enabled() is False
    assert test_run_id() != ""  # run_id 可读,但 mode 未开不算启用


# ── HTTP /health 层 ─────────────────────────────────────

def test_health_no_marker_when_off() -> None:
    body = TestClient(_app()).get("/health").json()
    assert body.get("status") == "ok"
    assert "test_mode" not in body
    assert "test_run_id" not in body


def test_health_run_id_alone_is_not_enough() -> None:
    os.environ[_RUN] = str(uuid.uuid4())
    body = TestClient(_app()).get("/health").json()
    assert "test_mode" not in body
    assert "test_run_id" not in body


def test_health_exposes_marker_only_when_explicitly_on(env_mode_on) -> None:
    body = TestClient(_app()).get("/health").json()
    assert body.get("test_mode") is True
    assert body.get("test_run_id") == env_mode_on["AI_MONITOR_TEST_RUN_ID"]