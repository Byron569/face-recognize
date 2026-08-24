"""阶段8b — 后端测试 4:fall 迁移保护与 schema 结构(部分需测试库)。

两类覆盖:
    1. 纯函数守护:Alembic 测试库 URL 的 5 道安全校验(guard_test_url),
       非法/默认库名即抛 ValueError,绝不回退;
    2. DB 结构:升到头后 event_outbox 表与 events 新增列(dedupe_key /
       delivery_mode / incident_id / source_event_id)存在,upgrade head 幂等。
结构校验通过 fall_db_engine(session 级已升到 head)执行,不可达即整组 skip。
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.eventdb


def _run(loop, coro):
    """在共享 fall_loop 上串行执行,避免跨 loop 复用 async engine 连接。"""
    return loop.run_until_complete(coro)


# ── 纯函数守护(不需要测试库)───────────────────────────────

def test_guard_rejects_empty_url() -> None:
    from backend.scripts.test_fall_migration_cycle import guard_test_url

    with pytest.raises(ValueError):
        guard_test_url("")
    with pytest.raises(ValueError):
        guard_test_url("   ")


def test_guard_rejects_non_test_database() -> None:
    from backend.scripts.test_fall_migration_cycle import guard_test_url

    for bad in (
        "postgresql://user:pwd@localhost:5432/ai_monitor",
        "postgresql://user:pwd@localhost:5432/ai_monitor_prod",
        "postgresql://user:pwd@localhost:5432/app",
    ):
        with pytest.raises(ValueError):
            guard_test_url(bad)


def test_guard_accepts_test_database_and_returns_same_url() -> None:
    from backend.scripts.test_fall_migration_cycle import guard_test_url

    url = "postgresql://postgres:123456@localhost:5432/ai_monitor_test"
    assert guard_test_url(url) == url


def test_guard_rejects_non_default_but_also_rejects_app_default() -> None:
    """即便像 ai_monitor 这样的常见名,只要不以 _test 结尾也必须拒绝。"""
    from backend.scripts.test_fall_migration_cycle import guard_test_url

    with pytest.raises(ValueError):
        guard_test_url("postgresql://postgres:123456@localhost:5432/ai_monitor")
    assert guard_test_url("postgresql://u:p@localhost:9999/some_test")


# ── DB 结构(需测试库)─────────────────────────────────────

def test_head_has_outbox_and_new_event_columns(fall_db_engine, fall_loop) -> None:
    async def body(engine):
        from sqlalchemy import text

        async with engine.connect() as conn:
            outbox = await conn.scalar(
                text("SELECT to_regclass('public.event_outbox')")
            )
            assert outbox is not None, "event_outbox 表缺失"

            cols = {
                row[0]
                for row in (
                    await conn.execute(
                        text(
                            "SELECT column_name FROM information_schema.columns"
                            " WHERE table_name='events'"
                        )
                    )
                ).fetchall()
            }
        assert {"dedupe_key", "delivery_mode", "incident_id", "source_event_id"} <= cols, cols

    _run(fall_loop, body(fall_db_engine))


def test_upgrade_head_is_idempotent(fall_db_engine) -> None:
    """重复 upgrade head 不抛错(迁移脚本可重复执行)。"""
    import os

    from alembic import command
    from alembic.config import Config

    sync_url = os.environ.get(
        "AI_MONITOR_TEST_DATABASE_URL",
        "postgresql://postgres:123456@localhost:5432/ai_monitor_test",
    )
    async_url = sync_url if "+asyncpg://" in sync_url else sync_url.replace(
        "postgresql://", "postgresql+asyncpg://", 1
    )
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
    cfg = Config()
    cfg.set_main_option("script_location", os.path.join(backend_dir, "migrations"))
    cfg.set_main_option("sqlalchemy.url", async_url)
    command.upgrade(cfg, "head")  # 已到头再跑一次,幂等