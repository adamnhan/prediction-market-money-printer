from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class Phase4Store:
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
                CREATE TABLE IF NOT EXISTS my_orders (
                    id INTEGER PRIMARY KEY,
                    intent_id INTEGER,
                    created_at INTEGER,
                    token_id TEXT,
                    side TEXT,
                    limit_price REAL,
                    qty REAL,
                    order_id TEXT,
                    status TEXT,
                    last_update_at INTEGER,
                    error TEXT
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS my_fills (
                    id INTEGER PRIMARY KEY,
                    order_id TEXT,
                    filled_at INTEGER,
                    fill_price REAL,
                    fill_qty REAL,
                    fee REAL,
                    tx_hash TEXT
                )
                """
            )
            self.conn.commit()

    def fetch_pending_intents(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            cur = self.conn.execute(
                """
                SELECT
                    c.id,
                    c.created_at,
                    c.target_wallet,
                    c.observed_trade_id,
                    c.tx_hash,
                    c.condition_id,
                    c.outcome,
                    c.side,
                    c.target_price,
                    c.target_size,
                    c.my_size,
                    c.my_limit_price,
                    c.intent_status,
                    m.title,
                    m.category,
                    m.status,
                    m.end_time,
                    m.token_ids_json
                FROM copy_intents c
                LEFT JOIN market_meta m
                  ON c.condition_id = m.condition_id
                WHERE c.intent_status = 'NEEDS_BOOKCHECK'
                ORDER BY c.id ASC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
        intents = []
        for row in rows:
            token_ids = None
            if row[17]:
                try:
                    token_ids = json.loads(row[17])
                except json.JSONDecodeError:
                    token_ids = None
            intents.append(
                {
                    "id": row[0],
                    "created_at": row[1],
                    "target_wallet": row[2],
                    "observed_trade_id": row[3],
                    "tx_hash": row[4],
                    "condition_id": row[5],
                    "outcome": row[6],
                    "side": row[7],
                    "target_price": row[8],
                    "target_size": row[9],
                    "my_size": row[10],
                    "my_limit_price": row[11],
                    "intent_status": row[12],
                    "meta_title": row[13],
                    "meta_category": row[14],
                    "meta_status": row[15],
                    "meta_end_time": row[16],
                    "token_ids": token_ids,
                }
            )
        return intents

    def update_intent_status(self, intent_id: int, status: str, reason: str | None = None) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE copy_intents
                SET intent_status = ?, skip_reason = ?
                WHERE id = ?
                """,
                (status, reason, intent_id),
            )
            self.conn.commit()

    def insert_order(self, order: dict[str, Any]) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO my_orders (
                    intent_id,
                    created_at,
                    token_id,
                    side,
                    limit_price,
                    qty,
                    order_id,
                    status,
                    last_update_at,
                    error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order.get("intent_id"),
                    order.get("created_at"),
                    order.get("token_id"),
                    order.get("side"),
                    order.get("limit_price"),
                    order.get("qty"),
                    order.get("order_id"),
                    order.get("status"),
                    order.get("last_update_at"),
                    order.get("error"),
                ),
            )
            self.conn.commit()

    def update_order_status(self, order_id: str, status: str, error: str | None = None) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE my_orders
                SET status = ?, last_update_at = strftime('%s','now'), error = ?
                WHERE order_id = ?
                """,
                (status, error, order_id),
            )
            self.conn.commit()

    def insert_fill(self, fill: dict[str, Any]) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO my_fills (
                    order_id,
                    filled_at,
                    fill_price,
                    fill_qty,
                    fee,
                    tx_hash
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    fill.get("order_id"),
                    fill.get("filled_at"),
                    fill.get("fill_price"),
                    fill.get("fill_qty"),
                    fill.get("fee"),
                    fill.get("tx_hash"),
                ),
            )
            self.conn.commit()
