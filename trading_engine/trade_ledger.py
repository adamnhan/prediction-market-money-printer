"""
Persistent trade ledger for paper trades.

Currently stores closed trades in a local SQLite file under data/trade_ledger.sqlite.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

from .models import Position

# SQLite file alongside other data artifacts.
DB_PATH = Path(
    os.getenv(
        "TRADE_LEDGER_DB_PATH",
        Path(__file__).resolve().parent.parent / "data" / "trade_ledger.sqlite",
    )
)


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER,
    event_ticker TEXT,
    market_ticker TEXT,
    side TEXT,
    qty INTEGER,
    entry_price REAL,
    entry_ts TEXT,
    exit_price REAL,
    exit_ts TEXT,
    realized_pnl REAL,
    exit_reason TEXT,
    strategy_snapshot TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


def _ensure_db() -> sqlite3.Connection:
    """
    Opens the SQLite database and ensures the trades table exists.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(CREATE_TABLE_SQL)
    return conn


def record_trade_close(
    position: Position,
    exit_reason: str,
    strategy_snapshot: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a closed trade record to the ledger.
    """
    conn = _ensure_db()

    entry_ts = position.entry_ts.isoformat() if position.entry_ts else None
    exit_ts = position.exit_ts.isoformat() if position.exit_ts else None
    config_json = json.dumps(strategy_snapshot) if strategy_snapshot else None

    try:
        with conn:
            conn.execute(
                """
                INSERT INTO trades (
                    position_id,
                    event_ticker,
                    market_ticker,
                    side,
                    qty,
                    entry_price,
                    entry_ts,
                    exit_price,
                    exit_ts,
                    realized_pnl,
                    exit_reason,
                    strategy_snapshot
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position.id,
                    position.event_ticker,
                    position.market_ticker,
                    position.side,
                    position.qty,
                    position.entry_price,
                    entry_ts,
                    position.current_price,
                    exit_ts,
                    position.realized_pnl,
                    exit_reason,
                    config_json,
                ),
            )
    finally:
        conn.close()


def fetch_trades(
    limit: int = 100,
    offset: int = 0,
    market_ticker: Optional[str] = None,
    event_ticker: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """
    Read back trade rows for APIs/UI. Filters are optional.
    """
    try:
        conn = _ensure_db()
        conn.row_factory = sqlite3.Row

        clauses = []
        params: list[Any] = []

        if market_ticker:
            clauses.append("market_ticker = ?")
            params.append(market_ticker.upper())

        if event_ticker:
            clauses.append("event_ticker = ?")
            params.append(event_ticker.upper())

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        query = f"""
            SELECT *
            FROM trades
            {where_sql}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def compute_summary_metrics() -> Dict[str, Any]:
    """
    Compute simple performance metrics from closed trades in the ledger.
    """
    try:
        conn = _ensure_db()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM trades").fetchall()
        finally:
            conn.close()
    except Exception:
        rows = []

    trades = [dict(r) for r in rows]
    total_trades = len(trades)

    wins = [t for t in trades if (t.get("realized_pnl") or 0) > 0]
    losses = [t for t in trades if (t.get("realized_pnl") or 0) < 0]

    total_realized = sum((t.get("realized_pnl") or 0) for t in trades)
    avg_win = sum((t.get("realized_pnl") or 0) for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum((t.get("realized_pnl") or 0) for t in losses) / len(losses) if losses else 0.0

    # Hold times in seconds (using entry_ts/exit_ts if present)
    hold_seconds = []
    for t in trades:
        entry_ts = t.get("entry_ts")
        exit_ts = t.get("exit_ts")
        if not entry_ts or not exit_ts:
            continue
        try:
            start = datetime.fromisoformat(entry_ts)
            end = datetime.fromisoformat(exit_ts)
            hold_seconds.append((end - start).total_seconds())
        except Exception:
            continue

    avg_hold_seconds = sum(hold_seconds) / len(hold_seconds) if hold_seconds else 0.0

    return {
        "total_trades": total_trades,
        "win_rate": (len(wins) / total_trades) if total_trades else 0.0,
        "total_realized_pnl": total_realized,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "average_hold_seconds": avg_hold_seconds,
        # placeholders for future metrics
        "open_trades": 0,
        "closed_trades": total_trades,
    }


def compute_equity_curve() -> list[Dict[str, Any]]:
    """
    Return a realized-PnL-only equity curve as a list of points.
    Each point has: id, exit_ts, cumulative_pnl.
    """
    try:
        conn = _ensure_db()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, exit_ts, realized_pnl FROM trades ORDER BY exit_ts ASC, id ASC"
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        rows = []

    curve = []
    cumulative = 0.0
    for r in rows:
        pnl = float(r["realized_pnl"] or 0)
        cumulative += pnl
        curve.append(
            {
                "id": r["id"],
                "exit_ts": r["exit_ts"],
                "cumulative_pnl": cumulative,
            }
        )
    return curve


def compute_circuit_breaker_stats() -> Dict[str, Any]:
    """
    Aggregate stats to support circuit breakers:
    - realized PnL today
    - trades closed today
    - cumulative equity & max drawdown
    """
    try:
        conn = _ensure_db()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT exit_ts, realized_pnl FROM trades ORDER BY exit_ts ASC, id ASC"
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        rows = []

    today_str = datetime.utcnow().date().isoformat()

    today_realized = 0.0
    today_trades = 0
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for r in rows:
        exit_ts = r["exit_ts"]
        pnl = float(r["realized_pnl"] or 0.0)
        if isinstance(exit_ts, str):
            try:
                exit_date = datetime.fromisoformat(exit_ts).date().isoformat()
                if exit_date == today_str:
                    today_trades += 1
                    today_realized += pnl
            except Exception:
                pass

        cumulative += pnl
        peak = max(peak, cumulative)
        drawdown = peak - cumulative
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return {
        "today_realized_pnl": today_realized,
        "today_trades": today_trades,
        "max_drawdown": max_drawdown,
    }


def compute_per_market_summary(
    limit: int = 200,
    sort_by: str = "total_realized_pnl"
) -> list[Dict[str, Any]]:
    """
    Aggregate realized PnL and win/loss counts by market ticker.
    """
    # Safe sort fields
    sort_fields = {
        "total_realized_pnl": "total_realized_pnl",
        "trades": "trades",
        "win_rate": "win_rate",
    }
    sort_column = sort_fields.get(sort_by, "total_realized_pnl")

    limit = max(1, min(limit, 1000))

    try:
        conn = _ensure_db()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                f"""
                SELECT market_ticker,
                       COUNT(*) as trades,
                       SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
                       SUM(realized_pnl) as total_realized_pnl,
                       AVG(realized_pnl) as avg_realized_pnl,
                       CASE WHEN COUNT(*) > 0 THEN CAST(SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) ELSE 0 END as win_rate
                FROM trades
                GROUP BY market_ticker
                ORDER BY {sort_column} DESC
                LIMIT ?
                """
            , (limit,)
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        rows = []

    summary = []
    for r in rows:
        trades = r["trades"] or 0
        wins = r["wins"] or 0
        summary.append(
            {
                "market_ticker": r["market_ticker"],
                "trades": trades,
                "wins": wins,
                "losses": r["losses"] or 0,
                "win_rate": (wins / trades) if trades else 0.0,
                "total_realized_pnl": float(r["total_realized_pnl"] or 0),
                "avg_realized_pnl": float(r["avg_realized_pnl"] or 0),
            }
        )
    return summary
