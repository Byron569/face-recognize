"""Worker crash-safe transition journal（第 5.8 / 6.6）。

状态机先在这里用整个事务持久化不可变 transition（WAL, synchronous=FULL, busy timeout），
commit 成功才更新内存 FSM 并对外发送。损坏时进入 UNAVAILABLE 并保留原文件。
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid

# 父端投递失败后的重试退避序列（秒，封顶 5；规格 5.8/6.6）
_RETRY_DELAYS_S = (0.25, 0.5, 1.0, 2.0, 5.0)


class WorkerJournal:
    def __init__(self, path: str, worker_instance_id: str) -> None:
        self.path = path
        self.worker_instance_id = worker_instance_id
        # 完整性检查：损坏时抛错并保留原件
        try:
            probe = sqlite3.connect(path, timeout=5)
            probe.execute("PRAGMA quick_check")
            probe.close()
        except sqlite3.DatabaseError as e:
            raise Exception(f"journal corrupt (preserved): {path}") from e
        self._conn = sqlite3.connect(path, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("""CREATE TABLE IF NOT EXISTS journal(
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE NOT NULL,
            payload_json TEXT NOT NULL DEFAULT "",
            created_at REAL NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            next_attempt_at REAL,
            last_error TEXT,
            parent_acked_at REAL)""")
        self._conn.commit()

    # 先占位（得到 sequence），再 commit 完整 payload；两段都在事务内完成
    def begin_add(self) -> int:
        # 用 WAL + IMMEDIATE 事务保证 crash-safe
        self._conn.execute("BEGIN IMMEDIATE")
        cur = self._conn.execute(
            "INSERT INTO journal(event_id, payload_json, created_at) VALUES(?,?,?)",
            (uuid.uuid4().hex, "", time.time()),
        )
        return int(cur.lastrowid)

    def commit(self, seq: int, transition) -> None:
        self._conn.execute(
            "UPDATE journal SET event_id=?, payload_json=? WHERE sequence=?",
            (transition.event_id, json.dumps(transition.to_dict(), ensure_ascii=False), seq),
        )
        self._conn.commit()

    def pending(self):
        # 退避过滤：last failure 排定的 next_attempt_at 未到时不重发，避免失败风暴；
        # 同时返回 attempt_count 供父端毒丸上限判断
        rows = self._conn.execute(
            "SELECT event_id, payload_json, attempt_count FROM journal "
            "WHERE parent_acked_at IS NULL AND payload_json <> '' "
            "AND (next_attempt_at IS NULL OR next_attempt_at <= ?) "
            "ORDER BY sequence",
            (time.time(),),
        ).fetchall()
        for eid, payload, attempts in rows:
            yield {"event_id": eid, "payload": payload, "attempt_count": int(attempts)}

    def record_attempt_failure(self, event_id: str, error: str) -> int:
        """父端投递失败：attempt_count+1、记录 last_error 并按退避推迟重试。

        返回失败后的 attempt_count（供毒丸上限判断）。
        """
        row = self._conn.execute(
            "SELECT attempt_count FROM journal WHERE event_id=?", (event_id,)
        ).fetchone()
        prev = int(row[0]) if row else 0
        delay = _RETRY_DELAYS_S[min(prev, len(_RETRY_DELAYS_S) - 1)]
        self._conn.execute(
            "UPDATE journal SET attempt_count=attempt_count+1, last_error=?, next_attempt_at=? "
            "WHERE event_id=?",
            (error, time.time() + delay, event_id),
        )
        self._conn.commit()
        return prev + 1

    def mark_parent_acked(self, event_id: str) -> None:
        self._conn.execute("UPDATE journal SET parent_acked_at=? WHERE event_id=?",
                           (time.time(), event_id))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
