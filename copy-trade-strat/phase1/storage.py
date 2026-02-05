from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class Storage:
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
                CREATE TABLE IF NOT EXISTS observed_trades (
                    id INTEGER PRIMARY KEY,
                    target_wallet TEXT,
                    proxy_wallet TEXT,
                    tx_hash TEXT,
                    condition_id TEXT,
                    outcome TEXT,
                    side TEXT,
                    price REAL,
                    size REAL,
                    timestamp INTEGER,
                    ingested_at INTEGER,
                    source TEXT,
                    UNIQUE(tx_hash, condition_id, side, price, size)
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observed_positions_snapshots (
                    id INTEGER PRIMARY KEY,
                    target_wallet TEXT,
                    condition_id TEXT,
                    outcome TEXT,
                    size REAL,
                    snapshot_time INTEGER
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

    def insert_trade(self, trade: dict[str, Any]) -> bool:
        with self._lock:
            try:
                self.conn.execute(
                    """
                    INSERT INTO observed_trades (
                        target_wallet,
                        proxy_wallet,
                        tx_hash,
                        condition_id,
                        outcome,
                        side,
                        price,
                        size,
                        timestamp,
                        ingested_at,
                        source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade.get("target_wallet"),
                        trade.get("proxy_wallet"),
                        trade.get("tx_hash"),
                        trade.get("condition_id"),
                        trade.get("outcome"),
                        trade.get("side"),
                        trade.get("price"),
                        trade.get("size"),
                        trade.get("timestamp"),
                        trade.get("ingested_at"),
                        trade.get("source"),
                    ),
                )
                self.conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def insert_position_snapshot(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO observed_positions_snapshots (
                    target_wallet,
                    condition_id,
                    outcome,
                    size,
                    snapshot_time
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    snapshot.get("target_wallet"),
                    snapshot.get("condition_id"),
                    snapshot.get("outcome"),
                    snapshot.get("size"),
                    snapshot.get("snapshot_time"),
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
