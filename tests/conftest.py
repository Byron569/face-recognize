"""pytest 配置与阶段8 后端测试脚手架。

- 把项目根与 backend 加入 sys.path,保证 vision / app 可导入;
- 提供仅由 DB 相关测试请求的 fixture(fall_db_engine / fall_session_factory / fall_db),
  通过 alembic 把测试库升到头,并在测试间 TRUNCATE 事件表做隔离;
- 测试库 URL 从环境变量 AI_MONITOR_TEST_DATABASE_URL 读取(默认本机 ai_monitor_test)。
  库名必须以 _test 结尾;不可达时相关测试自动 skip,不影响离线测试。
"""

from __future__ import annotations

import asyncio
import os
import sys


def _insert_paths() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    backend = os.path.join(root, "..", "backend")
    backend = os.path.abspath(backend)
    if backend not in sys.path:
        sys.path.insert(0, backend)


_insert_paths()

import pytest  # noqa: E402


def pytest_configure(config):  # noqa: ANN001
    config.addinivalue_line(
        "markers",
        "eventdb: depends on the PostgreSQL test database (AI_MONITOR_TEST_DATABASE_URL); skips if unavailable",
    )

_DEFAULT_SYNC_URL = "postgresql://postgres:123456@localhost:5432/ai_monitor_test"
_DEFAULT_ASYNC_URL = "postgresql+asyncpg://postgres:123456@localhost:5432/ai_monitor_test"


def _test_sync_url() -> str:
    return os.environ.get("AI_MONITOR_TEST_DATABASE_URL", _DEFAULT_SYNC_URL)


def _test_async_url() -> str:
    return os.environ.get("AI_MONITOR_TEST_ASYNC_URL", _DEFAULT_ASYNC_URL)


def _ensure_test_db(url: str) -> None:
    from sqlalchemy.engine import make_url

    dbname = getattr(make_url(url), "database", None)
    if not dbname or not dbname.endswith("_test"):
        raise RuntimeError(f"测试库名必须以 _test 结尾, 得到 {dbname!r}")


def _async_url(url: str) -> str:
    """把 postgres 测试库 URL 规整为 asyncpg 驱动,供 async 迁移 env 使用。"""
    if "+asyncpg://" in url:
        return url
    return url.replace("postgresql://", "postgresql+asyncpg://", 1)


def _alembic_upgrade_head(db_url: str) -> None:
    """把测试库升到 head(幂等);失败让调用方跳过。

    迁移 env 使用 async_engine_from_config,因此必须注入 asyncpg 驱动 URL。
    """
    from alembic import command
    from alembic.config import Config

    cfg = Config(os.path.join(_BACKEND_DIR, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(_BACKEND_DIR, "migrations"))
    cfg.set_main_option("sqlalchemy.url", _async_url(db_url))
    command.upgrade(cfg, "head")


_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))


@pytest.fixture(scope="session")
def fall_loop():
    """整个 DB 测试共享的单一事件循环。

    规避 asyncio.run() 每次新建 loop 导致 async engine 连接跨 loop 复用报
    “another operation is in progress” 的问题;所有 DB 协程统一在此循环上
    串行 run_until_complete。
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run_on_loop(loop, coro):
    if loop.is_running():
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


@pytest.fixture(scope="session")
def fall_db_engine(fall_loop):
    """alembic 升到 head 的 async engine(fallback 任务 DB 脚手架)。"""
    from sqlalchemy import pool
    from sqlalchemy.ext.asyncio import create_async_engine

    sync_url = _test_sync_url()
    async_url = _test_async_url()
    _ensure_test_db(sync_url)
    _ensure_test_db(async_url)
    try:
        _alembic_upgrade_head(sync_url)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"测试库不可用/迁移失败: {exc}")
    engine = create_async_engine(async_url, poolclass=pool.NullPool)
    yield engine
    run_on_loop(fall_loop, engine.dispose())


@pytest.fixture(scope="session")
def fall_session_factory(fall_db_engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    return async_sessionmaker(fall_db_engine, expire_on_commit=False)


@pytest.fixture()
def fall_db(fall_session_factory, fall_loop):
    """每个 DB 测试前后清空事件表(在共享 fall_loop 上串行执行)。"""
    from sqlalchemy import text

    async def _truncate() -> None:
        async with fall_session_factory() as s:
            await s.execute(
                text("TRUNCATE event_outbox, events, recognition_logs RESTART IDENTITY CASCADE")
            )
            await s.commit()

    run_on_loop(fall_loop, _truncate())
    yield fall_session_factory
    run_on_loop(fall_loop, _truncate())