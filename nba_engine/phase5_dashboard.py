from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .artifacts import load_artifacts
from .config import load_config, load_env
from .phase5 import RestClient, _extract_balance
from .phase4 import _dashboard_html


COOLDOWN = timedelta(minutes=10)
KILL_SWITCH_SAMPLE = 100


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


def _fetch_latest_candles(conn: sqlite3.Connection, limit: int = 60) -> list[CandleSnapshot]:
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
            SELECT market_ticker, MAX(start_ts) AS max_ts
            FROM candles
            GROUP BY market_ticker
        ) latest
          ON c.market_ticker = latest.market_ticker
         AND c.start_ts = latest.max_ts
        ORDER BY c.start_ts DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    snapshots: list[CandleSnapshot] = []
    for row in rows:
        snapshots.append(
            CandleSnapshot(
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
        )
    return snapshots


def _fetch_open_positions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, market_ticker, side, entry_ts, entry_price, qty
        FROM paper_trades
        WHERE exit_ts IS NULL
        """
    ).fetchall()
    positions: list[dict[str, Any]] = []
    for row in rows:
        positions.append(
            {
                "id": int(row[0]),
                "market_ticker": str(row[1]),
                "side": str(row[2]),
                "entry_ts": _parse_ts(row[3]),
                "entry_price": _to_float(row[4]),
                "qty": int(row[5]) if row[5] is not None else 1,
            }
        )
    return positions


def _fetch_recent_trades(conn: sqlite3.Connection, limit: int = 20) -> list[dict[str, Any]]:
    rows = conn.execute(
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


def _compute_pnl(side: str, entry_price: float, exit_price: float, qty: int) -> float:
    if side.upper() == "YES":
        return (exit_price - entry_price) * qty
    return (entry_price - exit_price) * qty


def _compute_drawdown(pnls: list[tuple[datetime, float]]) -> float:
    peak = 0.0
    cumulative = 0.0
    max_dd = 0.0
    for _, pnl in pnls:
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        drawdown = peak - cumulative
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def _fetch_closed_pnls(conn: sqlite3.Connection) -> list[tuple[datetime, float]]:
    rows = conn.execute(
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


def _compute_kill_switch(
    conn: sqlite3.Connection,
    quality_cutoff: float,
) -> tuple[bool, float | None, int]:
    rows = conn.execute(
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
        (quality_cutoff, KILL_SWITCH_SAMPLE),
    ).fetchall()
    values = [_to_float(row[0]) for row in rows if row[0] is not None]
    if len(values) < KILL_SWITCH_SAMPLE:
        return False, None, len(values)
    mean_pnl = float(sum(values) / len(values))
    return mean_pnl < 0, mean_pnl, len(values)


def _active_cooldowns(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT market_ticker, MAX(exit_ts) AS last_exit
        FROM paper_trades
        WHERE exit_ts IS NOT NULL
        GROUP BY market_ticker
        """
    ).fetchall()
    now = datetime.now(timezone.utc)
    cooldowns: list[dict[str, Any]] = []
    for row in rows:
        if not row[1]:
            continue
        last_exit = _parse_ts(row[1])
        cooldown_until = last_exit + COOLDOWN
        if cooldown_until > now:
            cooldowns.append(
                {
                    "market_ticker": str(row[0]),
                    "cooldown_until": cooldown_until.isoformat(),
                }
            )
    return cooldowns


def build_dashboard_data() -> dict[str, Any]:
    env = load_env()
    mode = env.get("NBA_ENGINE_MODE", "live").lower()
    candle_db = Path(env.get("KALSHI_CANDLE_DB_PATH", "data/phase1_candles.sqlite"))
    paper_db = Path(env.get("KALSHI_PAPER_DB_PATH", "data/paper_trades.sqlite"))
    artifacts_path = Path(env.get("STRATEGY_ARTIFACTS_PATH", "strategy_artifacts.json"))
    artifacts = load_artifacts(artifacts_path)

    candles_conn = sqlite3.connect(str(candle_db))
    paper_conn = sqlite3.connect(str(paper_db))

    markets = _fetch_latest_candles(candles_conn)
    portfolio_balance = None
    if mode == "live":
        try:
            config = load_config()
            if config.kalshi_rest_url:
                client = RestClient(
                    config.kalshi_rest_url,
                    config.kalshi_key_id,
                    config.kalshi_private_key_path,
                )
                payload = client.get_portfolio()
                portfolio_balance = _extract_balance(payload)
        except Exception:
            portfolio_balance = None
    positions_raw = _fetch_open_positions(paper_conn)
    recent_trades = _fetch_recent_trades(paper_conn)
    closed_pnls = _fetch_closed_pnls(paper_conn)
    kill_switch, kill_switch_mean, kill_switch_sample = _compute_kill_switch(
        paper_conn, artifacts.quality_cutoff
    )
    cooldowns = _active_cooldowns(paper_conn)

    last_prices = {c.market_ticker: c.close for c in markets if c.close == c.close}
    positions = []
    open_pnl = 0.0
    for pos in positions_raw:
        current = last_prices.get(pos["market_ticker"], pos["entry_price"])
        pnl = _compute_pnl(pos["side"], pos["entry_price"], current, pos["qty"])
        open_pnl += pnl
        positions.append(
            {
                "market_ticker": pos["market_ticker"],
                "side": pos["side"],
                "entry_ts": pos["entry_ts"].isoformat(),
                "entry_price": pos["entry_price"],
                "qty": pos["qty"],
                "current_price": current,
                "pnl": pnl,
            }
        )

    total_pnl = float(sum(pnl for _, pnl in closed_pnls))
    wins = len([pnl for _, pnl in closed_pnls if pnl > 0])
    closed_count = len(closed_pnls)
    win_rate = float(wins / closed_count) if closed_count else 0.0
    drawdown = _compute_drawdown(closed_pnls)

    last_update = None
    if markets:
        last_update = max(m.start_ts for m in markets).isoformat()

    return {
        "metrics": {
            "realized_pnl": total_pnl,
            "win_rate": win_rate,
            "drawdown": drawdown,
            "open_positions": len(positions),
            "open_pnl": open_pnl,
            "closed_trades": closed_count,
            "portfolio_balance": portfolio_balance,
        },
        "status": {
            "kill_switch": kill_switch,
            "kill_switch_mean_pnl": kill_switch_mean,
            "kill_switch_sample": kill_switch_sample,
            "cooldowns": cooldowns,
            "mode": mode,
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
        "positions": positions,
        "trades": recent_trades,
        "logs": [],
        "last_update": last_update,
    }


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, float) and value != value:
        return None
    return value


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(_dashboard_html())


@app.get("/api/dashboard")
def dashboard_data() -> JSONResponse:
    return JSONResponse(_sanitize_json(build_dashboard_data()))
