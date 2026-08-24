"""父端 durable event spool（第 5.8 / 6.2.8）。

以 event_id 唯一键的 SQLite（WAL, synchronous=FULL）持久待投递队列。Worker transition
在父端 commit 后才 ACK；已投递行按保留策略分批清理，绝不清理 pending 行。容量只统计
delivered_at IS NULL 的 pending。
"""
from __future__ import annotations

import json
import sqlite3
import time


class EventSpool:
    def __init__(self, path: str, pending_capacity: int) -> None:
        self.path = path
        self.cap = int(pending_capacity)
        self._conn = sqlite3.connect(path, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("""CREATE TABLE IF NOT EXISTS events(
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE NOT NULL,
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            delivered_at REAL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT)""")
        self._conn.commit()

    def add(self, transition) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO events(event_id, payload_json, created_at) VALUES(?,?,?)",
            (transition.event_id, json.dumps(transition.to_dict(), ensure_ascii=False), time.time()),
        )
        self._conn.commit()

    def try_add(self, transition) -> bool:
        if self.pending_count() >= self.cap:
            return False
        self.add(transition)
        return True

    def pending(self):
        rows = self._conn.execute(
            "SELECT event_id, payload_json FROM events WHERE delivered_at IS NULL ORDER BY sequence"
        ).fetchall()
        for eid, payload in rows:
            yield {"event_id": eid, "payload": payload}

    def mark_delivered(self, event_id: str) -> None:
        self._conn.execute("UPDATE events SET delivered_at=? WHERE event_id=? AND delivered_at IS NULL",
                           (time.time(), event_id))
        self._conn.commit()

    def pending_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM events WHERE delivered_at IS NULL").fetchone()
        return int(row[0])

    def is_full(self) -> bool:
        return self.pending_count() >= self.cap

    def prune_delivered(self, *, retention_hours: int, retention_rows: int) -> int:
        cutoff = time.time() - retention_hours * 3600
        cur = self._conn.execute(
            "DELETE FROM events WHERE delivered_at IS NOT NULL AND (delivered_at < ? OR sequence IN "
            "(SELECT sequence FROM events WHERE delivered_at IS NOT NULL ORDER BY sequence DESC LIMIT -1 OFFSET ?))",
            (cutoff, max(retention_rows, 0)),
        )
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()
