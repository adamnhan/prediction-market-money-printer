from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .artifacts import Artifacts, load_artifacts


logger = logging.getLogger("nba_phase4")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


ENTRY_DELAY = timedelta(minutes=2)
COOLDOWN = timedelta(minutes=10)
TAKE_PROFIT = 0.03
STOP_LOSS = -0.05
MAX_HOLD = timedelta(minutes=10)
KILL_SWITCH_SAMPLE = 100


@dataclass(frozen=True)
class CandleRow:
    rowid: int
    market_ticker: str
    start_ts: datetime
    close: float
    trade_active: int
    ret_3: float
    vol_10: float
    vol_sum_5: float
    gap_recent_5: int
    p_open: float
    p_base: float


@dataclass
class PanicState:
    detected_ts: datetime
    market_ticker: str
    direction: str
    p_open: float
    p_base: float
    ret_3: float
    vol_10: float
    vol_sum_5: float


@dataclass
class Position:
    id: int
    market_ticker: str
    side: str
    entry_ts: datetime
    entry_price: float
    qty: int = 1


@dataclass(frozen=True)
class CandleSnapshot:
    market_ticker: str
    start_ts: datetime
    close: float
    trade_active: int
    ret_3: float
    vol_10: float
    vol_sum_5: float
    p_open: float
    p_base: float


class CandleStream:
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"missing candle db: {path}")
        self.conn = sqlite3.connect(str(path))

    def max_rowid(self) -> int:
        cursor = self.conn.execute("SELECT MAX(rowid) FROM candles")
        row = cursor.fetchone()
        if not row or row[0] is None:
            return 0
        return int(row[0])

    def fetch_since(self, last_rowid: int) -> list[CandleRow]:
        rows = self.conn.execute(
            """
            SELECT rowid,
                   market_ticker,
                   start_ts,
                   close,
                   trade_active,
                   ret_3,
                   vol_10,
                   vol_sum_5,
                   gap_recent_5,
                   p_open,
                   p_base
            FROM candles
            WHERE rowid > ?
            ORDER BY rowid
            """,
            (last_rowid,),
        ).fetchall()
        return [_row_to_candle(row) for row in rows]

    def fetch_all(self) -> list[CandleRow]:
        rows = self.conn.execute(
            """
            SELECT rowid,
                   market_ticker,
                   start_ts,
                   close,
                   trade_active,
                   ret_3,
                   vol_10,
                   vol_sum_5,
                   gap_recent_5,
                   p_open,
                   p_base
            FROM candles
            ORDER BY rowid
            """
        ).fetchall()
        return [_row_to_candle(row) for row in rows]


class PaperStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_ts TEXT NOT NULL,
                entry_price REAL NOT NULL,
                qty INTEGER,
                p_open REAL,
                p_base REAL,
                panic_ts TEXT,
                panic_ret_3 REAL,
                vol_10 REAL,
                vol_sum_5 REAL,
                quality_score REAL,
                exit_ts TEXT,
                exit_price REAL,
                exit_reason TEXT,
                pnl REAL
            )
            """
        )
        self.conn.commit()
        self._ensure_columns()

    def _ensure_columns(self) -> None:
        existing = {
            str(row[1]) for row in self.conn.execute("PRAGMA table_info(paper_trades)").fetchall()
        }
        if "exit_reason" not in existing:
            self.conn.execute("ALTER TABLE paper_trades ADD COLUMN exit_reason TEXT")
        if "pnl" not in existing:
            self.conn.execute("ALTER TABLE paper_trades ADD COLUMN pnl REAL")
        if "qty" not in existing:
            self.conn.execute("ALTER TABLE paper_trades ADD COLUMN qty INTEGER")
        self.conn.commit()

    def load_open_positions(self) -> dict[str, Position]:
        rows = self.conn.execute(
            """
            SELECT id, market_ticker, side, entry_ts, entry_price, qty
            FROM paper_trades
            WHERE exit_ts IS NULL
            """
        ).fetchall()
        positions: dict[str, Position] = {}
        for row in rows:
            qty_val = row[5] if len(row) > 5 else None
            positions[str(row[1])] = Position(
                id=int(row[0]),
                market_ticker=str(row[1]),
                side=str(row[2]),
                entry_ts=_parse_ts(row[3]),
                entry_price=_to_float(row[4]),
                qty=int(qty_val) if qty_val is not None else 1,
            )
        return positions

    def last_exit_ts(self, ticker: str) -> datetime | None:
        row = self.conn.execute(
            """
            SELECT exit_ts
            FROM paper_trades
            WHERE market_ticker = ? AND exit_ts IS NOT NULL
            ORDER BY exit_ts DESC
            LIMIT 1
            """,
            (ticker,),
        ).fetchone()
        if not row or not row[0]:
            return None
        return _parse_ts(row[0])

    def insert_entry(
        self,
        ticker: str,
        side: str,
        entry_ts: datetime,
        entry_price: float,
        panic: PanicState,
        quality_score: float,
        qty: int = 1,
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO paper_trades (
                market_ticker,
                side,
                entry_ts,
                entry_price,
                qty,
                p_open,
                p_base,
                panic_ts,
                panic_ret_3,
                vol_10,
                vol_sum_5,
                quality_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                side,
                entry_ts.isoformat(),
                entry_price,
                qty,
                panic.p_open,
                panic.p_base,
                panic.detected_ts.isoformat(),
                panic.ret_3,
                panic.vol_10,
                panic.vol_sum_5,
                quality_score,
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def update_exit(
        self,
        position_id: int,
        exit_ts: datetime,
        exit_price: float,
        exit_reason: str,
        pnl: float,
    ) -> None:
        self.conn.execute(
            """
            UPDATE paper_trades
            SET exit_ts = ?, exit_price = ?, exit_reason = ?, pnl = ?
            WHERE id = ?
            """,
            (exit_ts.isoformat(), exit_price, exit_reason, pnl, position_id),
        )
        self.conn.commit()

    def recent_mean_pnl(self, limit: int, quality_cutoff: float) -> tuple[float | None, int]:
        rows = self.conn.execute(
            """
            SELECT pnl
            FROM paper_trades
            WHERE exit_ts IS NOT NULL
              AND pnl IS NOT NULL
              AND quality_score IS NOT NULL
              AND quality_score >= ?
            ORDER BY exit_ts DESC
            LIMIT ?
            """,
            (quality_cutoff, limit),
        ).fetchall()
        values = [_to_float(row[0]) for row in rows]
        filtered = [val for val in values if _is_valid_number(val)]
        if not filtered:
            return None, 0
        mean_pnl = float(sum(filtered) / len(filtered))
        return mean_pnl, len(filtered)

    def load_recent_trades(self, limit: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT id,
                   market_ticker,
                   side,
                   entry_ts,
                   entry_price,
                   qty,
                   exit_ts,
                   exit_price,
                   exit_reason,
                   pnl
            FROM paper_trades
            WHERE exit_ts IS NOT NULL
            ORDER BY exit_ts DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        trades: list[dict[str, Any]] = []
        for row in rows:
            trades.append(
                {
                    "id": int(row[0]),
                    "market_ticker": str(row[1]),
                    "side": str(row[2]),
                    "entry_ts": _parse_ts(row[3]).isoformat(),
                    "entry_price": _to_float(row[4]),
                    "qty": int(row[5]) if row[5] is not None else 1,
                    "exit_ts": _parse_ts(row[6]).isoformat(),
                    "exit_price": _to_float(row[7]),
                    "exit_reason": row[8],
                    "pnl": _to_float(row[9]),
                }
            )
        return trades

    def closed_pnls(self) -> list[tuple[datetime, float]]:
        rows = self.conn.execute(
            """
            SELECT exit_ts, pnl
            FROM paper_trades
            WHERE exit_ts IS NOT NULL AND pnl IS NOT NULL
            ORDER BY exit_ts
            """
        ).fetchall()
        parsed: list[tuple[datetime, float]] = []
        for row in rows:
            parsed.append((_parse_ts(row[0]), _to_float(row[1])))
        return parsed


class MemoryPaperStore:
    def __init__(self) -> None:
        self._trades: list[dict[str, Any]] = []
        self._next_id = 1

    def load_open_positions(self) -> dict[str, Position]:
        positions: dict[str, Position] = {}
        for trade in self._trades:
            if trade["exit_ts"] is None:
                positions[trade["market_ticker"]] = Position(
                    id=trade["id"],
                    market_ticker=trade["market_ticker"],
                    side=trade["side"],
                    entry_ts=trade["entry_ts"],
                    entry_price=trade["entry_price"],
                    qty=trade.get("qty", 1),
                )
        return positions

    def last_exit_ts(self, ticker: str) -> datetime | None:
        for trade in reversed(self._trades):
            if trade["market_ticker"] == ticker and trade["exit_ts"] is not None:
                return trade["exit_ts"]
        return None

    def insert_entry(
        self,
        ticker: str,
        side: str,
        entry_ts: datetime,
        entry_price: float,
        panic: PanicState,
        quality_score: float,
        qty: int = 1,
    ) -> int:
        trade_id = self._next_id
        self._next_id += 1
        self._trades.append(
            {
                "id": trade_id,
                "market_ticker": ticker,
                "side": side,
                "entry_ts": entry_ts,
                "entry_price": entry_price,
                "qty": qty,
                "quality_score": quality_score,
                "exit_ts": None,
                "exit_price": None,
                "exit_reason": None,
                "pnl": None,
            }
        )
        return trade_id

    def update_exit(
        self,
        position_id: int,
        exit_ts: datetime,
        exit_price: float,
        exit_reason: str,
        pnl: float,
    ) -> None:
        for trade in self._trades:
            if trade["id"] == position_id:
                trade["exit_ts"] = exit_ts
                trade["exit_price"] = exit_price
                trade["exit_reason"] = exit_reason
                trade["pnl"] = pnl
                break

    def recent_mean_pnl(self, limit: int, quality_cutoff: float) -> tuple[float | None, int]:
        filtered: list[float] = []
        for trade in reversed(self._trades):
            if trade["exit_ts"] is None:
                continue
            quality_score = trade.get("quality_score")
            if quality_score is None or quality_score < quality_cutoff:
                continue
            pnl = trade.get("pnl")
            if pnl is not None and _is_valid_number(pnl):
                filtered.append(float(pnl))
            if len(filtered) >= limit:
                break
        if not filtered:
            return None, 0
        mean_pnl = float(sum(filtered) / len(filtered))
        return mean_pnl, len(filtered)

    def closed_pnls(self) -> list[tuple[datetime, float]]:
        pnls: list[tuple[datetime, float]] = []
        for trade in self._trades:
            if trade["exit_ts"] is not None and trade["pnl"] is not None:
                pnls.append((trade["exit_ts"], float(trade["pnl"])))
        pnls.sort(key=lambda item: item[0])
        return pnls

    def recent_trades(self, limit: int) -> list[dict[str, Any]]:
        rows = [trade for trade in self._trades if trade["exit_ts"] is not None]
        rows.sort(key=lambda item: item["exit_ts"], reverse=True)
        return rows[:limit]


class ActivityTracker:
    def __init__(self, seed: dict[str, deque[int]] | None = None) -> None:
        self._recent = seed or {}

    def update(self, ticker: str, trade_active: int) -> int:
        recent = self._recent.setdefault(ticker, deque(maxlen=3))
        recent.append(1 if trade_active else 0)
        return int(sum(recent))


class SignalEngine:
    def __init__(self, artifacts: Artifacts) -> None:
        self.artifacts = artifacts

    def detect_panic(self, candle: CandleRow) -> PanicState | None:
        if not _is_valid_number(candle.p_open):
            return None
        if not _is_valid_number(candle.ret_3):
            return None
        if not _is_valid_number(candle.vol_10) or not _is_valid_number(candle.vol_sum_5):
            return None

        if not _is_mid_confidence(candle.p_open):
            return None
        if candle.vol_10 < self.artifacts.vol_quantiles["p90"]:
            return None
        if candle.vol_sum_5 < self.artifacts.volsum_quantiles["p90"]:
            return None

        is_underdog = candle.p_open < 0.5
        if is_underdog and candle.ret_3 >= 0.05:
            direction = "UNDERDOG_UP"
        elif (not is_underdog) and candle.ret_3 <= -0.05:
            direction = "FAVORITE_DOWN"
        else:
            return None

        return PanicState(
            detected_ts=candle.start_ts,
            market_ticker=candle.market_ticker,
            direction=direction,
            p_open=candle.p_open,
            p_base=candle.p_base,
            ret_3=candle.ret_3,
            vol_10=candle.vol_10,
            vol_sum_5=candle.vol_sum_5,
        )

    def evaluate_entry(
        self,
        panic: PanicState,
        candle: CandleRow,
        active_last_3: int,
    ) -> tuple[bool, float, list[str]]:
        reasons: list[str] = []
        if not _is_valid_number(candle.close):
            reasons.append("missing_entry_price")
        if not _is_valid_number(panic.p_base):
            reasons.append("missing_p_base")
        if not _is_valid_number(candle.vol_10):
            reasons.append("missing_vol_10")
        if not _is_valid_number(candle.vol_sum_5):
            reasons.append("missing_vol_sum_5")
        if candle.gap_recent_5 != 0:
            reasons.append("gap_recent_5")
        if active_last_3 < 2:
            reasons.append("active_last_3")

        move_from_base = float("nan")
        quality_score = float("nan")
        if not reasons:
            move_from_base = abs(candle.close - panic.p_base)
            if move_from_base < 0.07:
                reasons.append("move_from_base")
            vol_10_e = candle.vol_10 - self.artifacts.vol_quantiles["p90"]
            vol_sum_5_e = candle.vol_sum_5 - self.artifacts.volsum_quantiles["p90"]
            quality_score = (
                _zscore(move_from_base, self.artifacts)
                + _zscore(vol_10_e, self.artifacts)
                + _zscore(vol_sum_5_e, self.artifacts)
            )
            if quality_score < self.artifacts.quality_cutoff:
                reasons.append("quality_score")

        return len(reasons) == 0, quality_score, reasons


class EngineState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.last_candles: dict[str, CandleSnapshot] = {}
        self.last_prices: dict[str, float] = {}
        self.positions: dict[str, Position] = {}
        self.recent_trades: deque[dict[str, Any]] = deque(maxlen=200)
        self.logs: deque[dict[str, Any]] = deque(maxlen=300)
        self.closed_pnls: list[tuple[datetime, float]] = []
        self.kill_switch = False
        self.kill_switch_sample = 0
        self.kill_switch_mean_pnl: float | None = None
        self.cooldowns: dict[str, datetime] = {}
        self.last_update: datetime | None = None

    def seed_state(
        self,
        last_candles: dict[str, CandleSnapshot],
        last_prices: dict[str, float],
        positions: dict[str, Position],
        recent_trades: list[dict[str, Any]],
        closed_pnls: list[tuple[datetime, float]],
    ) -> None:
        with self._lock:
            self.last_candles = dict(last_candles)
            self.last_prices = dict(last_prices)
            self.positions = dict(positions)
            self.recent_trades = deque(recent_trades, maxlen=self.recent_trades.maxlen)
            self.closed_pnls = list(closed_pnls)

    def update_candle(self, candle: CandleRow) -> None:
        snapshot = CandleSnapshot(
            market_ticker=candle.market_ticker,
            start_ts=candle.start_ts,
            close=candle.close,
            trade_active=candle.trade_active,
            ret_3=candle.ret_3,
            vol_10=candle.vol_10,
            vol_sum_5=candle.vol_sum_5,
            p_open=candle.p_open,
            p_base=candle.p_base,
        )
        with self._lock:
            self.last_candles[candle.market_ticker] = snapshot
            if _is_valid_number(candle.close):
                self.last_prices[candle.market_ticker] = candle.close
            self.last_update = datetime.now(timezone.utc)

    def record_entry(self, position: Position) -> None:
        with self._lock:
            self.positions[position.market_ticker] = position
            self.last_update = datetime.now(timezone.utc)

    def record_exit(self, trade: dict[str, Any], cooldown_until: datetime | None) -> None:
        with self._lock:
            self.positions.pop(trade["market_ticker"], None)
            self.recent_trades.appendleft(trade)
            self.closed_pnls.append((_parse_ts(trade["exit_ts"]), float(trade["pnl"])))
            if cooldown_until is not None:
                self.cooldowns[trade["market_ticker"]] = cooldown_until
            self.last_update = datetime.now(timezone.utc)

    def record_log(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.logs.appendleft(payload)
            self.last_update = datetime.now(timezone.utc)

    def set_kill_switch(self, enabled: bool, mean_pnl: float | None, sample: int) -> None:
        with self._lock:
            self.kill_switch = enabled
            self.kill_switch_mean_pnl = mean_pnl
            self.kill_switch_sample = sample
            self.last_update = datetime.now(timezone.utc)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            markets = sorted(self.last_candles.values(), key=lambda item: item.market_ticker)
            positions = list(self.positions.values())
            recent_trades = list(self.recent_trades)
            logs = list(self.logs)
            closed_pnls = list(self.closed_pnls)
            cooldowns = dict(self.cooldowns)
            last_prices = dict(self.last_prices)
            kill_switch = self.kill_switch
            kill_switch_mean_pnl = self.kill_switch_mean_pnl
            kill_switch_sample = self.kill_switch_sample
            last_update = self.last_update

        now = datetime.now(timezone.utc)
        cooldown_list: list[dict[str, Any]] = []
        for ticker, until in cooldowns.items():
            if until > now:
                cooldown_list.append(
                    {"market_ticker": ticker, "cooldown_until": until.isoformat()}
                )

        metrics = _metrics_from_state(closed_pnls, positions, last_prices)
        return {
            "as_of": now.isoformat(),
            "last_update": last_update.isoformat() if last_update else None,
            "metrics": metrics,
            "status": {
                "kill_switch": kill_switch,
                "kill_switch_mean_pnl": kill_switch_mean_pnl,
                "kill_switch_sample": kill_switch_sample,
                "cooldowns": sorted(cooldown_list, key=lambda item: item["cooldown_until"]),
            },
            "markets": [
                {
                    "market_ticker": m.market_ticker,
                    "start_ts": m.start_ts.isoformat(),
                    "close": m.close,
                    "trade_active": m.trade_active,
                    "ret_3": m.ret_3,
                    "vol_10": m.vol_10,
                    "vol_sum_5": m.vol_sum_5,
                    "p_open": m.p_open,
                    "p_base": m.p_base,
                }
                for m in markets
            ],
            "positions": [
                {
                    "market_ticker": p.market_ticker,
                    "side": p.side,
                    "entry_ts": p.entry_ts.isoformat(),
                    "entry_price": p.entry_price,
                    "qty": p.qty,
                    "current_price": last_prices.get(p.market_ticker),
                    "pnl": _compute_pnl(
                        p.side,
                        p.entry_price,
                        last_prices.get(p.market_ticker, p.entry_price),
                        p.qty,
                    ),
                }
                for p in positions
            ],
            "trades": recent_trades,
            "logs": logs,
        }


def _metrics_from_state(
    closed_pnls: list[tuple[datetime, float]],
    positions: list[Position],
    last_prices: dict[str, float],
) -> dict[str, Any]:
    total_pnl = float(sum(pnl for _, pnl in closed_pnls))
    wins = len([pnl for _, pnl in closed_pnls if pnl > 0])
    closed = len(closed_pnls)
    win_rate = float(wins / closed) if closed else 0.0
    drawdown = _max_drawdown(closed_pnls)
    open_pnl = 0.0
    for position in positions:
        current = last_prices.get(position.market_ticker, position.entry_price)
        open_pnl += _compute_pnl(position.side, position.entry_price, current, position.qty)

    return {
        "realized_pnl": total_pnl,
        "win_rate": win_rate,
        "drawdown": drawdown,
        "open_positions": len(positions),
        "open_pnl": open_pnl,
        "closed_trades": closed,
    }


def _max_drawdown(closed_pnls: list[tuple[datetime, float]]) -> float:
    peak = 0.0
    cumulative = 0.0
    max_dd = 0.0
    for _, pnl in closed_pnls:
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        drawdown = peak - cumulative
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def _row_to_candle(row: tuple[Any, ...]) -> CandleRow:
    return CandleRow(
        rowid=int(row[0]),
        market_ticker=str(row[1]),
        start_ts=_parse_ts(row[2]),
        close=_to_float(row[3]),
        trade_active=int(row[4]) if row[4] is not None else 0,
        ret_3=_to_float(row[5]),
        vol_10=_to_float(row[6]),
        vol_sum_5=_to_float(row[7]),
        gap_recent_5=int(row[8]) if row[8] is not None else 0,
        p_open=_to_float(row[9]),
        p_base=_to_float(row[10]),
    )


def _is_valid_number(value: float) -> bool:
    return value is not None and not math.isnan(value)


def _is_mid_confidence(p_open: float) -> bool:
    return 0.15 <= abs(p_open - 0.5) < 0.30


def _zscore(value: float, artifacts: Artifacts) -> float:
    if artifacts.zscore_std == 0:
        return 0.0
    return (value - artifacts.zscore_mean) / artifacts.zscore_std


def _parse_ts(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    return datetime.now(timezone.utc)


def _to_float(value: object) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: _sanitize_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_json(item) for item in value]
    return value


def _seed_activity(conn: sqlite3.Connection) -> dict[str, deque[int]]:
    seed: dict[str, deque[int]] = {}
    tickers = conn.execute("SELECT DISTINCT market_ticker FROM candles").fetchall()
    for row in tickers:
        ticker = str(row[0])
        recent_rows = conn.execute(
            """
            SELECT trade_active
            FROM candles
            WHERE market_ticker = ?
            ORDER BY start_ts DESC
            LIMIT 2
            """,
            (ticker,),
        ).fetchall()
        values = [int(r[0]) if r[0] is not None else 0 for r in recent_rows][::-1]
        seed[ticker] = deque(values, maxlen=3)
    return seed


def _seed_last_prices(conn: sqlite3.Connection, tickers: list[str]) -> dict[str, float]:
    last_prices: dict[str, float] = {}
    for ticker in tickers:
        row = conn.execute(
            """
            SELECT close
            FROM candles
            WHERE market_ticker = ? AND close = close
            ORDER BY start_ts DESC
            LIMIT 1
            """,
            (ticker,),
        ).fetchone()
        if row and _is_valid_number(_to_float(row[0])):
            last_prices[ticker] = _to_float(row[0])
    return last_prices


def _seed_latest_candles(conn: sqlite3.Connection) -> dict[str, CandleSnapshot]:
    rows = conn.execute(
        """
        SELECT c.market_ticker,
               c.start_ts,
               c.close,
               c.trade_active,
               c.ret_3,
               c.vol_10,
               c.vol_sum_5,
               c.p_open,
               c.p_base
        FROM candles c
        JOIN (
            SELECT market_ticker, MAX(rowid) AS max_rowid
            FROM candles
            GROUP BY market_ticker
        ) latest
        ON c.market_ticker = latest.market_ticker AND c.rowid = latest.max_rowid
        """
    ).fetchall()
    latest: dict[str, CandleSnapshot] = {}
    for row in rows:
        snapshot = CandleSnapshot(
            market_ticker=str(row[0]),
            start_ts=_parse_ts(row[1]),
            close=_to_float(row[2]),
            trade_active=int(row[3]) if row[3] is not None else 0,
            ret_3=_to_float(row[4]),
            vol_10=_to_float(row[5]),
            vol_sum_5=_to_float(row[6]),
            p_open=_to_float(row[7]),
            p_base=_to_float(row[8]),
        )
        latest[snapshot.market_ticker] = snapshot
    return latest


def _compute_pnl(side: str, entry_price: float, exit_price: float, qty: int) -> float:
    if side.upper() == "YES":
        return (exit_price - entry_price) * qty
    return (entry_price - exit_price) * qty


def _exit_signal(
    position: Position,
    candle_ts: datetime,
    price: float,
    last_price: float | None,
) -> tuple[str | None, float | None, float | None]:
    if _is_valid_number(price):
        exit_price = price
    elif last_price is not None:
        exit_price = last_price
    else:
        exit_price = position.entry_price
    pnl = _compute_pnl(position.side, position.entry_price, exit_price, position.qty)
    if _is_valid_number(price):
        if pnl >= TAKE_PROFIT:
            return "tp", exit_price, pnl
        if pnl <= STOP_LOSS:
            return "sl", exit_price, pnl
    if candle_ts >= position.entry_ts + MAX_HOLD:
        return "max_hold", exit_price, pnl
    return None, None, None


def _log_event(state: EngineState | None, event: str, **fields: Any) -> None:
    payload = {"event": event, "ts": datetime.now(timezone.utc).isoformat()}
    payload.update(fields)
    logger.info(json.dumps(payload, sort_keys=True))
    if state is not None:
        state.record_log(payload)

def run_loop(state: EngineState | None = None) -> None:
    artifacts_path = Path(os.getenv("STRATEGY_ARTIFACTS_PATH", "strategy_artifacts.json"))
    artifacts = load_artifacts(artifacts_path)

    candle_db_path = Path(os.getenv("KALSHI_CANDLE_DB_PATH", "data/phase1_candles.sqlite"))
    paper_db_path = Path(os.getenv("KALSHI_PAPER_DB_PATH", "data/paper_trades.sqlite"))

    stream = CandleStream(candle_db_path)
    store = PaperStore(paper_db_path)
    activity = ActivityTracker(_seed_activity(stream.conn))
    engine = SignalEngine(artifacts)

    last_rowid = stream.max_rowid()
    pending: dict[str, PanicState] = {}
    positions = store.load_open_positions()
    last_prices = _seed_last_prices(stream.conn, list(positions.keys()))
    kill_switch = False
    mean_pnl, sample_count = store.recent_mean_pnl(KILL_SWITCH_SAMPLE, artifacts.quality_cutoff)
    if sample_count == KILL_SWITCH_SAMPLE and mean_pnl is not None and mean_pnl < 0:
        kill_switch = True
        _log_event(
            state,
            "kill_switch_triggered",
            mean_pnl=mean_pnl,
            sample=sample_count,
        )

    if state is not None:
        recent_trades = store.load_recent_trades(state.recent_trades.maxlen)
        closed_pnls = store.closed_pnls()
        last_candles = _seed_latest_candles(stream.conn)
        state.seed_state(last_candles, last_prices, positions, recent_trades, closed_pnls)
        state.set_kill_switch(kill_switch, mean_pnl, sample_count)

    _log_event(
        state,
        "phase4_ready",
        candle_db=str(candle_db_path),
        paper_db=str(paper_db_path),
        last_rowid=last_rowid,
        open_positions=len(positions),
        artifacts=artifacts.summary(),
    )

    while True:
        rows = stream.fetch_since(last_rowid)
        if not rows:
            time.sleep(1)
            continue
        for candle in rows:
            last_rowid = candle.rowid
            active_last_3 = activity.update(candle.market_ticker, candle.trade_active)
            if _is_valid_number(candle.close):
                last_prices[candle.market_ticker] = candle.close
            if state is not None:
                state.update_candle(candle)

            position = positions.get(candle.market_ticker)
            if position:
                reason, exit_price, pnl = _exit_signal(
                    position,
                    candle.start_ts,
                    candle.close,
                    last_prices.get(candle.market_ticker),
                )
                if reason and exit_price is not None and pnl is not None:
                    store.update_exit(position.id, candle.start_ts, exit_price, reason, pnl)
                    positions.pop(candle.market_ticker, None)
                    trade_payload = {
                        "id": position.id,
                        "market_ticker": position.market_ticker,
                        "side": position.side,
                        "entry_ts": position.entry_ts.isoformat(),
                        "entry_price": position.entry_price,
                        "exit_ts": candle.start_ts.isoformat(),
                        "exit_price": exit_price,
                        "exit_reason": reason,
                        "pnl": pnl,
                    }
                    cooldown_until = candle.start_ts + COOLDOWN
                    if state is not None:
                        state.record_exit(trade_payload, cooldown_until)
                    _log_event(
                        state,
                        "exit",
                        market_ticker=position.market_ticker,
                        side=position.side,
                        reason=reason,
                        exit_ts=candle.start_ts.isoformat(),
                        exit_price=exit_price,
                        pnl=pnl,
                        entry_ts=position.entry_ts.isoformat(),
                        entry_price=position.entry_price,
                    )
                    if not kill_switch:
                        mean_pnl, sample_count = store.recent_mean_pnl(
                            KILL_SWITCH_SAMPLE, artifacts.quality_cutoff
                        )
                        if (
                            sample_count == KILL_SWITCH_SAMPLE
                            and mean_pnl is not None
                            and mean_pnl < 0
                        ):
                            kill_switch = True
                            _log_event(
                                state,
                                "kill_switch_triggered",
                                mean_pnl=mean_pnl,
                                sample=sample_count,
                            )
                            if state is not None:
                                state.set_kill_switch(kill_switch, mean_pnl, sample_count)

            panic = pending.get(candle.market_ticker)
            if panic:
                if candle.start_ts >= panic.detected_ts + ENTRY_DELAY:
                    reasons: list[str] = []
                    if kill_switch:
                        reasons.append("kill_switch")
                    if candle.market_ticker in positions:
                        reasons.append("open_position")
                    else:
                        last_exit = store.last_exit_ts(candle.market_ticker)
                        if last_exit and candle.start_ts < last_exit + COOLDOWN:
                            reasons.append("cooldown")
                    allowed, quality_score, eval_reasons = engine.evaluate_entry(
                        panic, candle, active_last_3
                    )
                    reasons.extend(eval_reasons)
                    if reasons:
                        _log_event(
                            state,
                            "skip_entry",
                            market_ticker=candle.market_ticker,
                            reasons=",".join(reasons),
                            entry_ts=candle.start_ts.isoformat(),
                            entry_price=candle.close,
                            p_base=panic.p_base,
                            active_last_3=active_last_3,
                            gap_recent_5=candle.gap_recent_5,
                            quality=quality_score if not math.isnan(quality_score) else -1.0,
                        )
                    else:
                        side = "NO" if panic.direction == "UNDERDOG_UP" else "YES"
                        position_id = store.insert_entry(
                            candle.market_ticker,
                            side,
                            candle.start_ts,
                            candle.close,
                            panic,
                            quality_score,
                        )
                        new_position = Position(
                            id=position_id,
                            market_ticker=candle.market_ticker,
                            side=side,
                            entry_ts=candle.start_ts,
                            entry_price=candle.close,
                        )
                        positions[candle.market_ticker] = new_position
                        if state is not None:
                            state.record_entry(new_position)
                        _log_event(
                            state,
                            "entry",
                            market_ticker=candle.market_ticker,
                            side=side,
                            entry_ts=candle.start_ts.isoformat(),
                            entry_price=candle.close,
                            panic_ts=panic.detected_ts.isoformat(),
                            ret_3=panic.ret_3,
                            vol_10=panic.vol_10,
                            vol_sum_5=panic.vol_sum_5,
                            quality=quality_score,
                            p_open=panic.p_open,
                            p_base=panic.p_base,
                        )
                    pending.pop(candle.market_ticker, None)
                continue

            panic = engine.detect_panic(candle)
            if panic:
                pending[candle.market_ticker] = panic
                _log_event(
                    state,
                    "panic_detected",
                    market_ticker=candle.market_ticker,
                    direction=panic.direction,
                    ts=panic.detected_ts.isoformat(),
                    ret_3=panic.ret_3,
                    vol_10=panic.vol_10,
                    vol_sum_5=panic.vol_sum_5,
                    p_open=panic.p_open,
                    p_base=panic.p_base,
                )


def run_replay() -> dict[str, Any]:
    artifacts_path = Path(os.getenv("STRATEGY_ARTIFACTS_PATH", "strategy_artifacts.json"))
    artifacts = load_artifacts(artifacts_path)
    candle_db_path = Path(os.getenv("KALSHI_CANDLE_DB_PATH", "data/phase1_candles.sqlite"))
    paper_db_path = Path(os.getenv("KALSHI_PAPER_DB_PATH", "data/paper_trades.sqlite"))

    stream = CandleStream(candle_db_path)
    store = MemoryPaperStore()
    activity = ActivityTracker()
    engine = SignalEngine(artifacts)

    pending: dict[str, PanicState] = {}
    positions = store.load_open_positions()
    last_prices: dict[str, float] = {}
    kill_switch = False
    mean_pnl, sample_count = store.recent_mean_pnl(KILL_SWITCH_SAMPLE, artifacts.quality_cutoff)
    if sample_count == KILL_SWITCH_SAMPLE and mean_pnl is not None and mean_pnl < 0:
        kill_switch = True

    _log_event(None, "replay_start", candle_db=str(candle_db_path))

    for candle in stream.fetch_all():
        active_last_3 = activity.update(candle.market_ticker, candle.trade_active)
        if _is_valid_number(candle.close):
            last_prices[candle.market_ticker] = candle.close

        position = positions.get(candle.market_ticker)
        if position:
            reason, exit_price, pnl = _exit_signal(
                position,
                candle.start_ts,
                candle.close,
                last_prices.get(candle.market_ticker),
            )
            if reason and exit_price is not None and pnl is not None:
                store.update_exit(position.id, candle.start_ts, exit_price, reason, pnl)
                positions.pop(candle.market_ticker, None)
                if not kill_switch:
                    mean_pnl, sample_count = store.recent_mean_pnl(
                        KILL_SWITCH_SAMPLE, artifacts.quality_cutoff
                    )
                    if (
                        sample_count == KILL_SWITCH_SAMPLE
                        and mean_pnl is not None
                        and mean_pnl < 0
                    ):
                        kill_switch = True

        panic = pending.get(candle.market_ticker)
        if panic:
            if candle.start_ts >= panic.detected_ts + ENTRY_DELAY:
                reasons: list[str] = []
                if kill_switch:
                    reasons.append("kill_switch")
                if candle.market_ticker in positions:
                    reasons.append("open_position")
                else:
                    last_exit = store.last_exit_ts(candle.market_ticker)
                    if last_exit and candle.start_ts < last_exit + COOLDOWN:
                        reasons.append("cooldown")
                allowed, quality_score, eval_reasons = engine.evaluate_entry(
                    panic, candle, active_last_3
                )
                reasons.extend(eval_reasons)
                if not reasons:
                    side = "NO" if panic.direction == "UNDERDOG_UP" else "YES"
                    position_id = store.insert_entry(
                        candle.market_ticker,
                        side,
                        candle.start_ts,
                        candle.close,
                        panic,
                        quality_score,
                    )
                    positions[candle.market_ticker] = Position(
                        id=position_id,
                        market_ticker=candle.market_ticker,
                        side=side,
                        entry_ts=candle.start_ts,
                        entry_price=candle.close,
                    )
                pending.pop(candle.market_ticker, None)
            continue

        panic = engine.detect_panic(candle)
        if panic:
            pending[candle.market_ticker] = panic

    closed_pnls = store.closed_pnls()
    replay_metrics = _metrics_from_state(closed_pnls, list(positions.values()), last_prices)

    stored_metrics: dict[str, Any] = {}
    if paper_db_path.exists():
        actual_store = PaperStore(paper_db_path)
        actual_closed = actual_store.closed_pnls()
        stored_metrics = _metrics_from_state(actual_closed, [], {})

    report = {
        "replay": replay_metrics,
        "stored": stored_metrics,
    }
    _log_event(None, "replay_complete", **report)
    return report


def _dashboard_html() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>NBA Engine Ops</title>
    <style>
      @import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap");
      :root {
        --bg: #0b0c12;
        --panel: rgba(17, 20, 30, 0.86);
        --panel-strong: rgba(23, 28, 40, 0.95);
        --border: rgba(90, 98, 120, 0.3);
        --text: #e7eaf6;
        --muted: #a6b0c3;
        --accent: #66d9ff;
        --accent-2: #ffd166;
        --positive: #8dffbb;
        --negative: #ff7b7b;
        --shadow: 0 20px 60px rgba(5, 8, 20, 0.55);
      }
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        font-family: "Space Grotesk", "Trebuchet MS", sans-serif;
        color: var(--text);
        background: radial-gradient(circle at top left, #1b1f2e 0%, #0b0c12 45%, #07080f 100%);
        min-height: 100vh;
      }
      body::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image: linear-gradient(120deg, rgba(102, 217, 255, 0.08) 0%, transparent 40%),
          repeating-linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0 1px, transparent 1px 6px);
        pointer-events: none;
        z-index: 0;
      }
      .page {
        position: relative;
        z-index: 1;
        max-width: 1200px;
        margin: 0 auto;
        padding: 32px 20px 64px;
        animation: pageFade 0.7s ease-out both;
      }
      header {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
      }
      .title {
        font-size: 28px;
        letter-spacing: 0.5px;
        font-weight: 700;
      }
      .subtitle {
        color: var(--muted);
        font-size: 14px;
      }
      .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: 1px solid var(--border);
        background: rgba(12, 15, 25, 0.6);
      }
      .status-pill strong {
        font-size: 11px;
      }
      .grid {
        display: grid;
        gap: 18px;
      }
      .grid.metrics {
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      }
      .grid.main {
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      }
      .card {
        padding: 18px;
        border-radius: 20px;
        background: var(--panel);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        backdrop-filter: blur(14px);
      }
      .card strong {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--muted);
      }
      .metric-value {
        font-size: 28px;
        margin-top: 10px;
        font-weight: 600;
      }
      .muted {
        color: var(--muted);
      }
      .list {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-top: 12px;
      }
      .list-item {
        padding: 12px 14px;
        border-radius: 16px;
        background: var(--panel-strong);
        border: 1px solid rgba(255, 255, 255, 0.04);
        display: grid;
        gap: 6px;
        animation: rise 0.5s ease-out both;
      }
      .row {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
      }
      .badge {
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
      .badge.positive {
        background: rgba(141, 255, 187, 0.18);
        color: var(--positive);
      }
      .badge.negative {
        background: rgba(255, 123, 123, 0.18);
        color: var(--negative);
      }
      .badge.neutral {
        background: rgba(102, 217, 255, 0.18);
        color: var(--accent);
      }
      .divider {
        height: 1px;
        background: rgba(255, 255, 255, 0.06);
        margin: 16px 0;
      }
      .footer {
        margin-top: 20px;
        font-size: 12px;
        color: var(--muted);
      }
      @keyframes pageFade {
        from {
          opacity: 0;
          transform: translateY(12px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
      @keyframes rise {
        from {
          opacity: 0;
          transform: translateY(8px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
      @media (max-width: 640px) {
        .page {
          padding: 24px 16px 48px;
        }
        .title {
          font-size: 22px;
        }
        .metric-value {
          font-size: 22px;
        }
      }
    </style>
  </head>
  <body>
    <div class="page">
      <header>
        <div>
          <div class="title">NBA Engine Ops</div>
          <div class="subtitle">Live monitoring for signals, positions, and risk controls.</div>
        </div>
        <div class="status-pill">
          <span>Kill Switch</span>
          <strong id="kill-switch">OFF</strong>
        </div>
      </header>

      <section class="grid metrics">
        <div class="card">
          <strong>Realized PnL</strong>
          <div class="metric-value" id="metric-pnl">$0.00</div>
        </div>
        <div class="card">
          <strong>Portfolio Balance</strong>
          <div class="metric-value" id="metric-balance">--</div>
        </div>
        <div class="card">
          <strong>Win Rate</strong>
          <div class="metric-value" id="metric-win">0%</div>
        </div>
        <div class="card">
          <strong>Drawdown</strong>
          <div class="metric-value" id="metric-dd">0.00</div>
        </div>
        <div class="card">
          <strong>Open Positions</strong>
          <div class="metric-value" id="metric-open">0</div>
          <div class="muted" id="metric-open-pnl">$0.00 unrealized</div>
        </div>
      </section>

      <section class="grid main" style="margin-top: 22px;">
        <div class="card">
          <strong>Live Markets</strong>
          <div class="list" id="markets-list"></div>
        </div>
        <div class="card">
          <strong>Open Positions</strong>
          <div class="list" id="positions-list"></div>
        </div>
        <div class="card">
          <strong>Recent Trades</strong>
          <div class="list" id="trades-list"></div>
        </div>
      </section>

      <section class="grid main" style="margin-top: 22px;">
        <div class="card">
          <strong>Engine Status</strong>
          <div class="list" id="status-list"></div>
        </div>
        <div class="card">
          <strong>Signal Logs</strong>
          <div class="list" id="logs-list"></div>
        </div>
      </section>

      <div class="footer" id="last-update">Last update: --</div>
    </div>

    <script>
      function money(value) {
        if (value === null || value === undefined || Number.isNaN(value)) return "--";
        return (value >= 0 ? "$" : "-$") + Math.abs(value).toFixed(2);
      }
      function pct(value) {
        if (value === null || value === undefined || Number.isNaN(value)) return "--";
        return (value * 100).toFixed(1) + "%";
      }
      function ts(value) {
        if (!value) return "--";
        const d = new Date(value);
        return isNaN(d.getTime()) ? value : d.toLocaleString();
      }
      function renderList(el, items) {
        if (!items.length) {
          const empty = document.createElement("div");
          empty.className = "muted";
          empty.textContent = "No data yet.";
          el.replaceChildren(empty);
          return;
        }
        const existing = new Map();
        Array.from(el.children).forEach((child) => {
          if (child.dataset && child.dataset.key) {
            existing.set(child.dataset.key, child);
          } else {
            child.remove();
          }
        });
        items.forEach((item, idx) => {
          let node = existing.get(item.key);
          if (!node) {
            node = document.createElement("div");
            node.className = "list-item";
            node.dataset.key = item.key;
          } else {
            existing.delete(item.key);
          }
          node.replaceChildren(...buildItemContent(item));
          node.style.animationDelay = `${idx * 60}ms`;
          el.appendChild(node);
        });
        existing.forEach((node) => node.remove());
      }
      function makeItemData(key, badgeText, badgeClass, lines) {
        return { key, badgeText, badgeClass, lines };
      }
      function buildItemContent(item) {
        const parts = [];
        if (item.badgeText) {
          const badge = document.createElement("span");
          badge.className = `badge ${item.badgeClass || "neutral"}`;
          badge.textContent = item.badgeText;
          parts.push(badge);
        }
        item.lines.forEach((entry) => {
          parts.push(line(entry[0], entry[1]));
        });
        return parts;
      }
      function line(left, right) {
        const row = document.createElement("div");
        row.className = "row";
        const leftSpan = document.createElement("div");
        leftSpan.textContent = left;
        const rightSpan = document.createElement("div");
        rightSpan.textContent = right;
        rightSpan.className = "muted";
        row.appendChild(leftSpan);
        row.appendChild(rightSpan);
        return row;
      }
      async function loadDashboard() {
        const res = await fetch("/api/dashboard");
        const data = await res.json();
        const metrics = data.metrics || {};
        document.getElementById("metric-pnl").textContent = money(metrics.realized_pnl);
        document.getElementById("metric-balance").textContent = money(metrics.portfolio_balance);
        document.getElementById("metric-win").textContent = pct(metrics.win_rate);
        document.getElementById("metric-dd").textContent = metrics.drawdown?.toFixed(2) ?? "--";
        document.getElementById("metric-open").textContent = metrics.open_positions ?? 0;
        document.getElementById("metric-open-pnl").textContent =
          money(metrics.open_pnl) + " unrealized";

        const killSwitch = data.status?.kill_switch ? "ON" : "OFF";
        document.getElementById("kill-switch").textContent = killSwitch;

        const markets = (data.markets || []).slice(0, 40).map((m, idx) =>
          makeItemData(
            m.market_ticker || `market-${idx}`,
            m.trade_active ? "Active" : "Idle",
            m.trade_active ? "positive" : "neutral",
            [
              [m.market_ticker, ts(m.start_ts)],
              ["Last close", m.close?.toFixed(4) ?? "--"],
              ["Ret 3", m.ret_3?.toFixed(4) ?? "--"],
            ]
          )
        );
        renderList(document.getElementById("markets-list"), markets);

        const positions = (data.positions || []).map((p, idx) => {
          const pnl = p.pnl ?? 0;
          const badgeClass = pnl >= 0 ? "positive" : "negative";
          return makeItemData(
            p.market_ticker || `position-${idx}`,
            pnl >= 0 ? "Up" : "Down",
            badgeClass,
            [
              [`${p.market_ticker} ${p.side}`, ts(p.entry_ts)],
              ["Entry", p.entry_price?.toFixed(4) ?? "--"],
              ["Qty", p.qty ?? 1],
              ["Now", p.current_price?.toFixed(4) ?? "--"],
              ["PnL", pnl.toFixed(4)],
            ]
          );
        });
        renderList(document.getElementById("positions-list"), positions);

        const trades = (data.trades || []).slice(0, 16).map((t, idx) => {
          const pnl = t.pnl ?? 0;
          const badgeClass = pnl >= 0 ? "positive" : "negative";
          return makeItemData(
            `trade-${t.id ?? idx}`,
            pnl >= 0 ? "Win" : "Loss",
            badgeClass,
            [
              [`${t.market_ticker} ${t.side}`, ts(t.exit_ts)],
              ["Exit", t.exit_price?.toFixed(4) ?? "--"],
              ["PnL", pnl.toFixed(4)],
              ["Reason", t.exit_reason ?? "--"],
            ]
          );
        });
        renderList(document.getElementById("trades-list"), trades);

        const status = [];
        const cooldowns = data.status?.cooldowns || [];
        status.push(
          makeItemData(
            "risk",
            "Risk",
            killSwitch === "ON" ? "negative" : "positive",
            [
              ["Kill switch", killSwitch],
              ["Sample", data.status?.kill_switch_sample ?? 0],
            ]
          )
        );
        if (cooldowns.length) {
          cooldowns.slice(0, 10).forEach((c) => {
            status.push(
              makeItemData(
                `cooldown-${c.market_ticker}`,
                "Cooldown",
                "neutral",
                [
                  [c.market_ticker, "Cooldown"],
                  ["Until", ts(c.cooldown_until)],
                ]
              )
            );
          });
        } else {
          status.push(
            makeItemData("cooldowns-none", "Cooldown", "neutral", [
              ["Cooldowns", "None active"],
            ])
          );
        }
        renderList(document.getElementById("status-list"), status);

        const logs = (data.logs || []).slice(0, 12).map((l, idx) =>
          makeItemData(
            `log-${l.ts ?? idx}-${l.event ?? ""}`,
            "Log",
            "neutral",
            [
              [l.event, ts(l.ts)],
              ["Market", l.market_ticker || "--"],
            ]
          )
        );
        renderList(document.getElementById("logs-list"), logs);

        document.getElementById("last-update").textContent =
          "Last update: " + (data.last_update ? ts(data.last_update) : "--");
      }
      loadDashboard();
      setInterval(loadDashboard, 4000);
    </script>
  </body>
</html>
"""

app = FastAPI()
ENGINE_STATE = EngineState()
ENGINE_THREAD: threading.Thread | None = None


@app.on_event("startup")
def _startup() -> None:
    global ENGINE_THREAD
    if ENGINE_THREAD is None:
        ENGINE_THREAD = threading.Thread(target=run_loop, kwargs={"state": ENGINE_STATE}, daemon=True)
        ENGINE_THREAD.start()


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(_dashboard_html())


@app.get("/api/dashboard")
def dashboard_data() -> JSONResponse:
    return JSONResponse(_sanitize_json(ENGINE_STATE.snapshot()))


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA phase4 engine with ops dashboard.")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI dashboard.")
    parser.add_argument("--host", default=os.getenv("PHASE4_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PHASE4_PORT", "8010")))
    parser.add_argument("--replay", action="store_true", help="Replay stored candles and exit.")
    args = parser.parse_args()

    if args.replay:
        report = run_replay()
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.serve:
        import uvicorn

        uvicorn.run("nba_engine.phase4:app", host=args.host, port=args.port, reload=False)
        return

    run_loop()


if __name__ == "__main__":
    main()
