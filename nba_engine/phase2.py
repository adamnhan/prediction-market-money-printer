from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .artifacts import Artifacts, load_artifacts


logger = logging.getLogger("nba_phase2")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[PHASE2] %(asctime)s %(message)s"))
    logger.addHandler(handler)


ENTRY_DELAY = timedelta(minutes=2)
COOLDOWN = timedelta(minutes=10)


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
        parsed: list[CandleRow] = []
        for row in rows:
            parsed.append(
                CandleRow(
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
            )
        return parsed


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
                p_open REAL,
                p_base REAL,
                panic_ts TEXT,
                panic_ret_3 REAL,
                vol_10 REAL,
                vol_sum_5 REAL,
                quality_score REAL,
                exit_ts TEXT,
                exit_price REAL
            )
            """
        )
        self.conn.commit()

    def has_open_position(self, ticker: str) -> bool:
        row = self.conn.execute(
            """
            SELECT 1
            FROM paper_trades
            WHERE market_ticker = ? AND exit_ts IS NULL
            LIMIT 1
            """,
            (ticker,),
        ).fetchone()
        return row is not None

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
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO paper_trades (
                market_ticker,
                side,
                entry_ts,
                entry_price,
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
                _zscore(move_from_base, self.artifacts, "move_from_base")
                + _zscore(vol_10_e, self.artifacts, "vol_10_e")
                + _zscore(vol_sum_5_e, self.artifacts, "vol_sum_5_e")
            )
            if quality_score < self.artifacts.quality_cutoff:
                reasons.append("quality_score")

        return len(reasons) == 0, quality_score, reasons


def _is_valid_number(value: float) -> bool:
    return value is not None and not math.isnan(value)


def _is_mid_confidence(p_open: float) -> bool:
    return 0.15 <= abs(p_open - 0.5) < 0.30


def _zscore(value: float, artifacts: Artifacts, feature: str | None = None) -> float:
    if feature and artifacts.zscore_features and feature in artifacts.zscore_features:
        stats = artifacts.zscore_features[feature]
        std = stats.get("std", 0.0)
        if std == 0:
            return 0.0
        return (value - stats.get("mean", 0.0)) / std
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


def run_loop() -> None:
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
    logger.info(
        "phase2_ready candle_db=%s paper_db=%s last_rowid=%d %s",
        candle_db_path,
        paper_db_path,
        last_rowid,
        artifacts.summary(),
    )

    while True:
        rows = stream.fetch_since(last_rowid)
        if not rows:
            time.sleep(1)
            continue
        for candle in rows:
            last_rowid = candle.rowid
            active_last_3 = activity.update(candle.market_ticker, candle.trade_active)
            panic = pending.get(candle.market_ticker)
            if panic:
                if candle.start_ts >= panic.detected_ts + ENTRY_DELAY:
                    reasons: list[str] = []
                    if store.has_open_position(candle.market_ticker):
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
                        logger.info(
                            "skip_entry %s reasons=%s entry_ts=%s p_entry=%.4f p_base=%.4f active_last_3=%d gap_recent_5=%d quality=%.4f",
                            candle.market_ticker,
                            ",".join(reasons),
                            candle.start_ts.isoformat(),
                            candle.close,
                            panic.p_base,
                            active_last_3,
                            candle.gap_recent_5,
                            quality_score if not math.isnan(quality_score) else -1.0,
                        )
                    else:
                        side = "NO" if panic.direction == "UNDERDOG_UP" else "YES"
                        store.insert_entry(
                            candle.market_ticker,
                            side,
                            candle.start_ts,
                            candle.close,
                            panic,
                            quality_score,
                        )
                        logger.info(
                            "entry %s side=%s entry_ts=%s entry_price=%.4f panic_ts=%s ret_3=%.4f vol_10=%.4f vol_sum_5=%.2f quality=%.4f p_open=%.4f p_base=%.4f",
                            candle.market_ticker,
                            side,
                            candle.start_ts.isoformat(),
                            candle.close,
                            panic.detected_ts.isoformat(),
                            panic.ret_3,
                            panic.vol_10,
                            panic.vol_sum_5,
                            quality_score,
                            panic.p_open,
                            panic.p_base,
                        )
                    pending.pop(candle.market_ticker, None)
                continue

            panic = engine.detect_panic(candle)
            if panic:
                pending[candle.market_ticker] = panic
                logger.info(
                    "panic_detected %s direction=%s ts=%s ret_3=%.4f vol_10=%.4f vol_sum_5=%.2f p_open=%.4f p_base=%.4f",
                    candle.market_ticker,
                    panic.direction,
                    panic.detected_ts.isoformat(),
                    panic.ret_3,
                    panic.vol_10,
                    panic.vol_sum_5,
                    panic.p_open,
                    panic.p_base,
                )


def main() -> None:
    run_loop()


if __name__ == "__main__":
    main()
