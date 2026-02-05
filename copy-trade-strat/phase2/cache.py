from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class Phase2Cache:
    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_meta (
                    condition_id TEXT PRIMARY KEY,
                    market_id TEXT,
                    slug TEXT,
                    title TEXT,
                    category TEXT,
                    outcomes_json TEXT,
                    token_ids_json TEXT,
                    status TEXT,
                    end_time INTEGER,
                    last_refreshed_at INTEGER,
                    source TEXT
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            self.conn.commit()

    def get_meta(self, condition_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self.conn.execute(
                """
                SELECT
                    condition_id,
                    market_id,
                    slug,
                    title,
                    category,
                    outcomes_json,
                    token_ids_json,
                    status,
                    end_time,
                    last_refreshed_at,
                    source
                FROM market_meta
                WHERE condition_id = ?
                """,
                (condition_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "condition_id": row[0],
            "market_id": row[1],
            "slug": row[2],
            "title": row[3],
            "category": row[4],
            "outcomes": json.loads(row[5]) if row[5] else None,
            "token_ids": json.loads(row[6]) if row[6] else None,
            "status": row[7],
            "end_time": row[8],
            "last_refreshed_at": row[9],
            "source": row[10],
        }

    def upsert_meta(self, meta: dict[str, Any]) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO market_meta (
                    condition_id,
                    market_id,
                    slug,
                    title,
                    category,
                    outcomes_json,
                    token_ids_json,
                    status,
                    end_time,
                    last_refreshed_at,
                    source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(condition_id) DO UPDATE SET
                    market_id=excluded.market_id,
                    slug=excluded.slug,
                    title=excluded.title,
                    category=excluded.category,
                    outcomes_json=excluded.outcomes_json,
                    token_ids_json=excluded.token_ids_json,
                    status=excluded.status,
                    end_time=excluded.end_time,
                    last_refreshed_at=excluded.last_refreshed_at,
                    source=excluded.source
                """,
                (
                    meta.get("condition_id"),
                    meta.get("market_id"),
                    meta.get("slug"),
                    meta.get("title"),
                    meta.get("category"),
                    json.dumps(meta.get("outcomes")) if meta.get("outcomes") is not None else None,
                    json.dumps(meta.get("token_ids")) if meta.get("token_ids") is not None else None,
                    meta.get("status"),
                    meta.get("end_time"),
                    meta.get("last_refreshed_at"),
                    meta.get("source"),
                ),
            )
            self.conn.commit()

    def set_state(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO state (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, payload),
            )
            self.conn.commit()

    def get_state(self, key: str, default: Any | None = None) -> Any:
        with self._lock:
            cur = self.conn.execute("SELECT value FROM state WHERE key = ?", (key,))
            row = cur.fetchone()
        if not row:
            return default
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return row[0]

    def iter_missing_condition_ids(self) -> list[str]:
        with self._lock:
            cur = self.conn.execute(
                """
                SELECT DISTINCT t.condition_id
                FROM observed_trades t
                LEFT JOIN market_meta m
                  ON t.condition_id = m.condition_id
                WHERE m.condition_id IS NULL
                """
            )
            rows = cur.fetchall()
        return [row[0] for row in rows]

    def iter_new_condition_ids_since(self, last_trade_id: int) -> list[str]:
        with self._lock:
            cur = self.conn.execute(
                """
                SELECT DISTINCT condition_id
                FROM observed_trades
                WHERE id > ?
                ORDER BY id ASC
                """,
                (last_trade_id,),
            )
            rows = cur.fetchall()
            cur2 = self.conn.execute("SELECT MAX(id) FROM observed_trades")
            max_row = cur2.fetchone()
        max_id = max_row[0] if max_row and max_row[0] is not None else last_trade_id
        return [row[0] for row in rows], max_id
