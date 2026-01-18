"""Persistence for shadow execution results."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from cross_venue_arb.storage.sqlite import connect


@dataclass(frozen=True)
class ShadowLegRecord:
    venue: str
    market_id: str
    outcome: str
    limit_price: float
    intended_price: float | None
    filled_price: float | None
    filled_size: float
    status: str
    reason: str | None


@dataclass(frozen=True)
class ShadowTradeRecord:
    opp_id: str
    game_key: str
    detected_ts: float
    latency_ms: int
    detected_edge: float
    detected_size: float
    status: str
    reason: str | None
    realized_pnl: float | None
    legs: list[ShadowLegRecord]


def write_shadow_trade(record: ShadowTradeRecord) -> None:
    conn = connect()
    _ensure_schema(conn)
    created_at = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO shadow_trades (
            opp_id,
            game_key,
            detected_ts,
            latency_ms,
            detected_edge,
            detected_size,
            status,
            reason,
            realized_pnl,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.opp_id,
            record.game_key,
            record.detected_ts,
            record.latency_ms,
            record.detected_edge,
            record.detected_size,
            record.status,
            record.reason,
            record.realized_pnl,
            created_at,
        ),
    )
    trade_id = cursor.lastrowid
    leg_rows = []
    for leg in record.legs:
        leg_rows.append(
            (
                trade_id,
                leg.venue,
                leg.market_id,
                leg.outcome,
                leg.limit_price,
                leg.intended_price,
                leg.filled_price,
                leg.filled_size,
                leg.status,
                leg.reason,
            )
        )
    conn.executemany(
        """
        INSERT INTO shadow_trade_legs (
            trade_id,
            venue,
            market_id,
            outcome,
            limit_price,
            intended_price,
            filled_price,
            filled_size,
            status,
            reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        leg_rows,
    )
    conn.commit()
    conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shadow_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            opp_id TEXT NOT NULL,
            game_key TEXT NOT NULL,
            detected_ts REAL NOT NULL,
            latency_ms INTEGER NOT NULL,
            detected_edge REAL NOT NULL,
            detected_size REAL NOT NULL,
            status TEXT NOT NULL,
            reason TEXT,
            realized_pnl REAL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shadow_trade_legs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER NOT NULL,
            venue TEXT NOT NULL,
            market_id TEXT NOT NULL,
            outcome TEXT NOT NULL,
            limit_price REAL NOT NULL,
            intended_price REAL,
            filled_price REAL,
            filled_size REAL NOT NULL,
            status TEXT NOT NULL,
            reason TEXT,
            FOREIGN KEY(trade_id) REFERENCES shadow_trades(id)
        )
        """
    )
