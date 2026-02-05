from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any


class Phase3Store:
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
                CREATE TABLE IF NOT EXISTS copy_intents (
                    id INTEGER PRIMARY KEY,
                    created_at INTEGER,
                    target_wallet TEXT,
                    observed_trade_id INTEGER,
                    tx_hash TEXT,
                    condition_id TEXT,
                    outcome TEXT,
                    side TEXT,
                    target_price REAL,
                    target_size REAL,
                    my_size REAL,
                    my_limit_price REAL,
                    intent_status TEXT,
                    skip_reason TEXT,
                    notes TEXT
                )
                """
            )
            self.conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ux_copy_intents_observed_trade
                ON copy_intents(observed_trade_id)
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS exposure_state (
                    condition_id TEXT PRIMARY KEY,
                    open_notional REAL,
                    last_updated_at INTEGER
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS exposure_summary (
                    key TEXT PRIMARY KEY,
                    value REAL,
                    last_updated_at INTEGER
                )
                """
            )
            self.conn.commit()

    def fetch_unprocessed_trades(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            cur = self.conn.execute(
                """
                SELECT
                    t.id,
                    t.target_wallet,
                    t.proxy_wallet,
                    t.tx_hash,
                    t.condition_id,
                    t.outcome,
                    t.side,
                    t.price,
                    t.size,
                    t.timestamp,
                    t.ingested_at,
                    m.title,
                    m.category,
                    m.status,
                    m.end_time
                FROM observed_trades t
                LEFT JOIN market_meta m
                  ON t.condition_id = m.condition_id
                LEFT JOIN copy_intents c
                  ON c.observed_trade_id = t.id
                WHERE c.observed_trade_id IS NULL
                ORDER BY t.id ASC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
        trades = []
        for row in rows:
            trades.append(
                {
                    "id": row[0],
                    "target_wallet": row[1],
                    "proxy_wallet": row[2],
                    "tx_hash": row[3],
                    "condition_id": row[4],
                    "outcome": row[5],
                    "side": row[6],
                    "price": row[7],
                    "size": row[8],
                    "timestamp": row[9],
                    "ingested_at": row[10],
                    "meta_title": row[11],
                    "meta_category": row[12],
                    "meta_status": row[13],
                    "meta_end_time": row[14],
                }
            )
        return trades

    def insert_intent(self, intent: dict[str, Any]) -> bool:
        with self._lock:
            try:
                self.conn.execute(
                    """
                    INSERT INTO copy_intents (
                        created_at,
                        target_wallet,
                        observed_trade_id,
                        tx_hash,
                        condition_id,
                        outcome,
                        side,
                        target_price,
                        target_size,
                        my_size,
                        my_limit_price,
                        intent_status,
                        skip_reason,
                        notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        intent.get("created_at"),
                        intent.get("target_wallet"),
                        intent.get("observed_trade_id"),
                        intent.get("tx_hash"),
                        intent.get("condition_id"),
                        intent.get("outcome"),
                        intent.get("side"),
                        intent.get("target_price"),
                        intent.get("target_size"),
                        intent.get("my_size"),
                        intent.get("my_limit_price"),
                        intent.get("intent_status"),
                        intent.get("skip_reason"),
                        intent.get("notes"),
                    ),
                )
                self.conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def get_open_exposure(self, condition_id: str) -> float:
        with self._lock:
            cur = self.conn.execute(
                "SELECT open_notional FROM exposure_state WHERE condition_id = ?",
                (condition_id,),
            )
            row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    def set_open_exposure(self, condition_id: str, value: float, ts: int) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO exposure_state (condition_id, open_notional, last_updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(condition_id) DO UPDATE SET
                    open_notional=excluded.open_notional,
                    last_updated_at=excluded.last_updated_at
                """,
                (condition_id, value, ts),
            )
            self.conn.commit()

    def get_summary_value(self, key: str) -> float:
        with self._lock:
            cur = self.conn.execute("SELECT value FROM exposure_summary WHERE key = ?", (key,))
            row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    def set_summary_value(self, key: str, value: float, ts: int) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO exposure_summary (key, value, last_updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    last_updated_at=excluded.last_updated_at
                """,
                (key, value, ts),
            )
            self.conn.commit()

    def count_intents_last_hour(self, now_ts: int) -> int:
        with self._lock:
            cur = self.conn.execute(
                """
                SELECT COUNT(*)
                FROM copy_intents
                WHERE created_at >= ?
                  AND intent_status != 'SKIPPED'
                """,
                (now_ts - 3600,),
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0
