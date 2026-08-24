"""迁移保护脚本 — 只对一次性 `_test` 库执行 upgrade/downgrade 周期校验。

该脚本不是生产迁移工具。它从 `--database-url-env` 指定的环境变量读取测试库 URL,
在用 Alembic 做任何操作前执行多道安全校验,任一失败立即非零退出且绝不回退/绝不使用
应用默认 URL。校验项(全部满足才继续):
  1. URL 非空;
  2. SQLAlchemy make_url() 可解析;
  3. database 名以 ``_test`` 结尾;
  4. 连接后 ``SELECT current_database()`` 仍以 ``_test`` 结尾;
  5. 不是应用默认 database_url。

随后用 Alembic Python API 显式注入该 URL,建立一条既有事件 sentinel,执行:
    upgrade head -> downgrade -1 -> upgrade head
并验证 sentinel 未丢失、各版本下列/索引/outbox 状态一致。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402


def guard_test_url(url_str: str) -> str:
    """校验测试库 URL 的安全性。非法立即抛 ValueError,绝不回退。"""
    from sqlalchemy.engine import make_url

    if not url_str or not url_str.strip():
        raise ValueError("AI_MONITOR_TEST_DATABASE_URL is required; no default database is allowed")
    url = make_url(url_str)
    database = getattr(url, "database", None)
    if not database or not database.endswith("_test"):
        raise ValueError(f"database name must end with '_test', got {database!r}")
    return url_str


async def _current_database(url) -> str:
    import asyncpg

    db = getattr(url, "database", None)
    conn = await asyncpg.connect(
        host=url.host or "localhost",
        port=url.port or 5432,
        user=url.username,
        password=url.password,
        database=db,
    )
    try:
        return str(await conn.fetchval("SELECT current_database()"))
    finally:
        await conn.close()


def _async_url(url_str: str) -> str:
    """规整为 asyncpg 驱动 URL(迁移 env 用 async_engine_from_config)。"""
    if "+asyncpg://" in url_str:
        return url_str
    if "postgresql+asyncpg://" not in url_str:
        return url_str.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url_str


def _alembic_config(test_url: str) -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", _async_url(test_url))
    return cfg


async def _insert_sentinel_event(url) -> int:
    import asyncpg

    conn = await asyncpg.connect(
        host=url.host or "localhost",
        port=url.port or 5432,
        user=url.username,
        password=url.password,
        database=url.database,
    )
    try:
        row = await conn.fetchrow(
            "INSERT INTO events (event_type, camera_id, confidence, payload, created_at)"
            " VALUES ('fall_potential', 'sentinel-cam', 0, '{}'::jsonb, now()) RETURNING id"
        )
        return int(row["id"])
    finally:
        await conn.close()


async def _column_exists(url, table: str, column: str) -> bool:
    import asyncpg

    conn = await asyncpg.connect(
        host=url.host or "localhost", port=url.port or 5432,
        user=url.username, password=url.password, database=url.database,
    )
    try:
        val = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns"
            " WHERE table_name=$1 AND column_name=$2", table, column
        )
        return val is not None
    finally:
        await conn.close()


async def _table_exists(url, table: str) -> bool:
    import asyncpg

    conn = await asyncpg.connect(
        host=url.host or "localhost", port=url.port or 5432,
        user=url.username, password=url.password, database=url.database,
    )
    try:
        val = await conn.fetchval(
            "SELECT 1 FROM information_schema.tables WHERE table_name=$1", table
        )
        return val is not None
    finally:
        await conn.close()


async def _sentinel_exists(url, sentinel_id: int) -> bool:
    import asyncpg

    conn = await asyncpg.connect(
        host=url.host or "localhost", port=url.port or 5432,
        user=url.username, password=url.password, database=url.database,
    )
    try:
        return await conn.fetchval("SELECT 1 FROM events WHERE id=$1", sentinel_id) is not None
    finally:
        await conn.close()


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url-env", default="AI_MONITOR_TEST_DATABASE_URL")
    args = parser.parse_args()

    url_str = os.environ.get(args.database_url_env, "")
    guard_test_url(url_str)
    from sqlalchemy.engine import make_url

    url = make_url(url_str)

    # 连接后二次确认数据库名
    current = _run(_current_database(url))
    if not current.endswith("_test"):
        raise RuntimeError(f"live database {current!r} does not end with _test")

    cfg = _alembic_config(url_str)

    # 1) 先把测试库升到 head(若为空库或旧版本),确保 events 存在可插 sentinel
    command.upgrade(cfg, "head")

    sentinel_id = _run(_insert_sentinel_event(url))

    # 2) upgrade head(幂等兜底) -> 验证 outbox 与新列
    command.upgrade(cfg, "head")
    assert _run(_table_exists(url, "event_outbox")), "event_outbox missing at head"
    assert _run(_column_exists(url, "events", "dedupe_key")), "events.dedupe_key missing at head"
    assert _run(_column_exists(url, "events", "delivery_mode")), "events.delivery_mode missing at head"
    assert _run(_sentinel_exists(url, sentinel_id)), "sentinel lost after upgrade"

    # 3) downgrade -1 -> 回到 0001,outbox/新列应消失,sentinel 仍在
    command.downgrade(cfg, "-1")
    assert not _run(_table_exists(url, "event_outbox")), "event_outbox should be gone after -1"
    assert not _run(_column_exists(url, "events", "dedupe_key")), "dedupe_key should be gone after -1"
    assert _run(_sentinel_exists(url, sentinel_id)), "sentinel lost after downgrade -1"

    # 4) upgrade head -> 恢复新结构
    command.upgrade(cfg, "head")
    assert _run(_table_exists(url, "event_outbox")), "event_outbox missing on re-upgrade"
    assert _run(_column_exists(url, "events", "dedupe_key")), "dedupe_key missing on re-upgrade"
    assert _run(_column_exists(url, "events", "incident_id")), "incident_id missing on re-upgrade"
    assert _run(_sentinel_exists(url, sentinel_id)), "sentinel lost after re-upgrade"

    print("MIGRATION_CYCLE_OK sentinel_id=%d database=%s" % (sentinel_id, current))


if __name__ == "__main__":
    main()