import json
import math
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from kalshi_fetcher.kalshi_client import request
from incentives.incentives import get_active_incentives


DEFAULT_DB_PATH = "data/incentives_phase1.sqlite"
DEFAULT_TRADES_LIMIT = 200
REQUEST_PAUSE_S = 0.15
RETRY_BACKOFF_S = (0.5, 1.0, 2.0, 4.0)

MIN_TIME_REMAINING_HOURS = 12
MAX_SPREAD_CENTS = 20
MIN_DEPTH_5C = 50
MAX_IMBALANCE = 0.8

DEPTH_CAP = 500
TARGET_TRADES_15M = 20
TARGET_TIME_HOURS = 48

RISK_W_SPREAD = 1.0
RISK_W_IMBALANCE = 1.0
RISK_W_NEAR_RESOLUTION = 0.8
RISK_W_LOW_DEPTH = 0.7


def _utc_ms_now() -> int:
    return int(time.time() * 1000)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _parse_ts_to_utc_ms(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = int(value)
        if ts > 1_000_000_000_000:
            return ts
        if ts > 1_000_000_000:
            return ts * 1000
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return _parse_ts_to_utc_ms(int(stripped))
        try:
            dt = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
            return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            return None
    return None


def _coerce_price_to_cents(value: Any) -> Optional[int]:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if price <= 1.0:
        return int(round(price * 100))
    return int(round(price))


def _coerce_size(value: Any) -> Optional[float]:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return None
    return size


def _extract_levels(raw_levels: Any) -> list[tuple[int, float]]:
    levels: list[tuple[int, float]] = []
    if not raw_levels:
        return levels
    for level in raw_levels:
        if isinstance(level, dict):
            price = _coerce_price_to_cents(level.get("price"))
            size = _coerce_size(level.get("size") or level.get("qty") or level.get("count"))
        else:
            try:
                price = _coerce_price_to_cents(level[0])
                size = _coerce_size(level[1])
            except Exception:
                continue
        if price is None or size is None:
            continue
        if size > 0:
            levels.append((price, size))
    return levels


def _extract_orderbook(payload: Dict[str, Any]) -> Tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    container = payload.get("orderbook") if isinstance(payload.get("orderbook"), dict) else payload
    yes_levels = _extract_levels(container.get("yes") or container.get("bids") or [])
    no_levels = _extract_levels(container.get("no") or container.get("asks") or [])
    return yes_levels, no_levels


def _best_bid(levels: Iterable[tuple[int, float]]) -> Optional[int]:
    prices = [price for price, size in levels if size > 0]
    return max(prices) if prices else None


def _depth_within(levels: Iterable[tuple[int, float]], best: Optional[int], window: int) -> float:
    if best is None:
        return 0.0
    min_price = best - window
    return sum(size for price, size in levels if price >= min_price)


def _top_depth(levels: Iterable[tuple[int, float]], best: Optional[int]) -> float:
    if best is None:
        return 0.0
    for price, size in levels:
        if price == best:
            return size
    return 0.0


def _imbalance(top_yes: float, top_no: float) -> float:
    denom = top_yes + top_no + 1e-9
    return (top_yes - top_no) / denom


def _spread_cents(best_yes: Optional[int], best_no: Optional[int]) -> Optional[int]:
    if best_yes is None or best_no is None:
        return None
    implied_yes_ask = 100 - best_no
    return implied_yes_ask - best_yes


def _parse_market_close_ts(market: Dict[str, Any]) -> Optional[int]:
    return _parse_ts_to_utc_ms(
        market.get("close_time")
        or market.get("close_time_ts")
        or market.get("close_time_ms")
        or market.get("close_ts")
        or market.get("settle_time")
        or market.get("settle_time_ts")
    )


def _fetch_market(ticker: str) -> Dict[str, Any]:
    data = _request_with_retry(f"/markets/{ticker}")
    if isinstance(data, dict) and isinstance(data.get("market"), dict):
        return data["market"]
    return data


def _fetch_orderbook(ticker: str) -> Dict[str, Any]:
    return _request_with_retry(f"/markets/{ticker}/orderbook")


def _fetch_trades(ticker: str, limit: int = DEFAULT_TRADES_LIMIT) -> list[Dict[str, Any]]:
    data = _request_with_retry("/markets/trades", params={"ticker": ticker, "limit": limit})
    trades = data.get("trades") if isinstance(data, dict) else None
    return trades if isinstance(trades, list) else []


def _trades_activity(trades: Iterable[Dict[str, Any]], lookback_minutes: int = 15) -> Tuple[int, float]:
    cutoff_ms = _utc_ms_now() - lookback_minutes * 60 * 1000
    count = 0
    volume = 0.0
    for trade in trades:
        ts = _parse_ts_to_utc_ms(trade.get("created_time") or trade.get("timestamp") or trade.get("ts"))
        if ts is None or ts < cutoff_ms:
            continue
        count += 1
        qty = trade.get("count") or trade.get("qty") or trade.get("size") or 0
        try:
            volume += float(qty)
        except (TypeError, ValueError):
            continue
    return count, volume


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS phase1_market_snapshots (
            id INTEGER PRIMARY KEY,
            run_id TEXT NOT NULL,
            ts_utc_ms INTEGER NOT NULL,
            market_ticker TEXT NOT NULL,
            status TEXT,
            close_ts_utc_ms INTEGER,
            time_remaining_ms INTEGER,
            best_yes_bid INTEGER,
            best_no_bid INTEGER,
            spread_cents INTEGER,
            depth_yes_5c REAL,
            depth_no_5c REAL,
            top_depth_yes REAL,
            top_depth_no REAL,
            imbalance REAL,
            trades_15m INTEGER,
            volume_15m REAL,
            incentive_type TEXT,
            reward_value REAL,
            raw_market_json TEXT,
            raw_orderbook_json TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS phase1_market_decisions (
            id INTEGER PRIMARY KEY,
            run_id TEXT NOT NULL,
            ts_utc_ms INTEGER NOT NULL,
            market_ticker TEXT NOT NULL,
            eligible INTEGER NOT NULL,
            exclude_reasons TEXT,
            rank_score REAL,
            incentive_ev_proxy REAL,
            risk_score REAL
        );
        """
    )
    conn.commit()


def _fetch_events_for_series(series_ticker: str, limit: int = 200) -> list[Dict[str, Any]]:
    events: list[Dict[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        params: Dict[str, Any] = {
            "series_ticker": series_ticker,
            "with_nested_markets": "true",
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor
        data = _request_with_retry("/events", params=params)
        page_events = data.get("events") if isinstance(data, dict) else None
        if isinstance(page_events, list):
            events.extend(page_events)
        cursor = data.get("cursor") or data.get("next_cursor")
        if not cursor:
            break
    return events


def _request_with_retry(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    for attempt, backoff in enumerate((0.0,) + RETRY_BACKOFF_S):
        if backoff:
            time.sleep(backoff)
        if REQUEST_PAUSE_S:
            time.sleep(REQUEST_PAUSE_S)
        try:
            return request(endpoint, params=params)
        except Exception as exc:
            if "429" not in str(exc) and attempt == len(RETRY_BACKOFF_S):
                raise
            if "429" not in str(exc) and attempt < len(RETRY_BACKOFF_S):
                continue
            if "429" in str(exc) and attempt < len(RETRY_BACKOFF_S):
                continue
            raise


def _expand_series_markets(series_ticker: str) -> list[str]:
    markets: list[str] = []
    for event in _fetch_events_for_series(series_ticker):
        raw_markets = event.get("markets") or event.get("nested_markets") or []
        if isinstance(raw_markets, dict):
            raw_markets = raw_markets.get("markets") or []
        if not isinstance(raw_markets, list):
            continue
        for market in raw_markets:
            if isinstance(market, dict):
                ticker = market.get("ticker") or market.get("market_ticker")
            else:
                ticker = None
            if ticker:
                markets.append(str(ticker))
    return markets


def _candidate_markets(active_incentives: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    candidates: Dict[str, Dict[str, Any]] = {}
    for incentive in active_incentives:
        market_ticker = incentive.get("market_ticker")
        series_ticker = incentive.get("series_ticker")
        if market_ticker:
            candidates[str(market_ticker)] = incentive
        if series_ticker:
            for ticker in _expand_series_markets(str(series_ticker)):
                if ticker not in candidates:
                    candidates[ticker] = incentive
    return candidates


def _score_market(
    spread_cents: Optional[int],
    depth_5c: float,
    trades_15m: int,
    time_remaining_hours: float,
    incentive_type: Optional[str],
    reward_value: Optional[float],
    imbalance: float,
    near_resolution: bool,
    low_depth: bool,
) -> Tuple[float, float, float]:
    spread_score = 0.0 if spread_cents is None else max(0.0, 1.0 - spread_cents / MAX_SPREAD_CENTS)
    depth_score = max(0.0, min(1.0, math.log1p(depth_5c) / math.log1p(DEPTH_CAP)))
    activity_score = max(0.0, min(1.0, trades_15m / TARGET_TRADES_15M))
    time_score = max(0.0, min(1.0, time_remaining_hours / TARGET_TIME_HOURS))

    reward_weight = reward_value if reward_value is not None else 1.0
    incentive_base = (depth_score + activity_score) / 2.0
    if incentive_type and incentive_type.lower() == "liquidity":
        incentive_base = (depth_score + time_score) / 2.0
    elif incentive_type and incentive_type.lower() == "volume":
        incentive_base = (activity_score + depth_score) / 2.0

    incentive_ev_proxy = reward_weight * incentive_base
    risk_score = (
        RISK_W_SPREAD * (0.0 if spread_cents is None else spread_cents / MAX_SPREAD_CENTS)
        + RISK_W_IMBALANCE * abs(imbalance)
        + RISK_W_NEAR_RESOLUTION * (1.0 if near_resolution else 0.0)
        + RISK_W_LOW_DEPTH * (1.0 if low_depth else 0.0)
    )
    rank_score = (spread_score + depth_score + activity_score + time_score) / 4.0 + incentive_ev_proxy - risk_score
    return rank_score, incentive_ev_proxy, risk_score


def run_phase1(db_path: str = DEFAULT_DB_PATH, include_trades: bool = True) -> Dict[str, Any]:
    active_incentives = get_active_incentives()
    candidates = _candidate_markets(active_incentives)

    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    try:
        _ensure_schema(conn)
        run_id = uuid.uuid4().hex
        now_ms = _utc_ms_now()

        ranked_rows: list[Dict[str, Any]] = []

        for ticker, incentive in candidates.items():
            market = _fetch_market(ticker)
            status = market.get("status")
            close_ts = _parse_market_close_ts(market)
            time_remaining_ms = close_ts - now_ms if close_ts else None
            time_remaining_hours = (time_remaining_ms / 3600000.0) if time_remaining_ms else 0.0

            orderbook = _fetch_orderbook(ticker)
            yes_levels, no_levels = _extract_orderbook(orderbook)
            best_yes = _best_bid(yes_levels)
            best_no = _best_bid(no_levels)
            spread = _spread_cents(best_yes, best_no)
            depth_yes_5c = _depth_within(yes_levels, best_yes, 5)
            depth_no_5c = _depth_within(no_levels, best_no, 5)
            top_yes = _top_depth(yes_levels, best_yes)
            top_no = _top_depth(no_levels, best_no)
            imbalance = _imbalance(top_yes, top_no)

            trades_15m = 0
            volume_15m = 0.0
            if include_trades:
                trades = _fetch_trades(ticker)
                trades_15m, volume_15m = _trades_activity(trades)

            exclude_reasons: list[str] = []
            if status:
                status_norm = str(status).lower()
                if status_norm not in {"open", "active"}:
                    exclude_reasons.append(f"status={status}")
            if time_remaining_ms is not None and time_remaining_hours < MIN_TIME_REMAINING_HOURS:
                exclude_reasons.append("near_resolution")
            if spread is None:
                exclude_reasons.append("no_spread")
            if best_yes is None or best_no is None:
                exclude_reasons.append("one_sided_book")

            total_depth_5c = depth_yes_5c + depth_no_5c
            low_depth = total_depth_5c < MIN_DEPTH_5C
            if low_depth:
                exclude_reasons.append("low_depth")

            if spread is not None and spread > MAX_SPREAD_CENTS and low_depth:
                exclude_reasons.append("wide_spread")

            if abs(imbalance) > MAX_IMBALANCE:
                exclude_reasons.append("high_imbalance")

            near_resolution = time_remaining_ms is not None and time_remaining_hours < MIN_TIME_REMAINING_HOURS

            rank_score, incentive_ev_proxy, risk_score = _score_market(
                spread,
                total_depth_5c,
                trades_15m,
                time_remaining_hours,
                incentive.get("incentive_type"),
                incentive.get("reward_value"),
                imbalance,
                near_resolution,
                low_depth,
            )

            eligible = 1 if not exclude_reasons else 0

            conn.execute(
                """
                INSERT INTO phase1_market_snapshots (
                    run_id, ts_utc_ms, market_ticker, status, close_ts_utc_ms, time_remaining_ms,
                    best_yes_bid, best_no_bid, spread_cents, depth_yes_5c, depth_no_5c,
                    top_depth_yes, top_depth_no, imbalance, trades_15m, volume_15m,
                    incentive_type, reward_value, raw_market_json, raw_orderbook_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    now_ms,
                    ticker,
                    status,
                    close_ts,
                    time_remaining_ms,
                    best_yes,
                    best_no,
                    spread,
                    depth_yes_5c,
                    depth_no_5c,
                    top_yes,
                    top_no,
                    imbalance,
                    trades_15m,
                    volume_15m,
                    incentive.get("incentive_type"),
                    incentive.get("reward_value"),
                    _json_dumps(market),
                    _json_dumps(orderbook),
                ),
            )
            conn.execute(
                """
                INSERT INTO phase1_market_decisions (
                    run_id, ts_utc_ms, market_ticker, eligible, exclude_reasons,
                    rank_score, incentive_ev_proxy, risk_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    now_ms,
                    ticker,
                    eligible,
                    ",".join(exclude_reasons) if exclude_reasons else None,
                    rank_score,
                    incentive_ev_proxy,
                    risk_score,
                ),
            )

            ranked_rows.append(
                {
                    "market": ticker,
                    "spread": spread,
                    "depth_5c": total_depth_5c,
                    "imbalance": imbalance,
                    "trades_15m": trades_15m,
                    "time_remaining_h": time_remaining_hours,
                    "incentive_ev_proxy": incentive_ev_proxy,
                    "risk_score": risk_score,
                    "eligible": bool(eligible),
                }
            )

        conn.commit()

        ranked_rows.sort(key=lambda row: (row["eligible"], row["incentive_ev_proxy"], row["depth_5c"]), reverse=True)
        return {"ok": True, "run_id": run_id, "rows": ranked_rows}
    finally:
        conn.close()


def render_ranked_table(rows: Iterable[Dict[str, Any]], limit: int = 50) -> str:
    lines = [
        "market | spread | depth_5c | imbalance | trades_15m | time_remaining_h | incentive_ev_proxy | risk_score | eligible"
    ]
    for row in list(rows)[:limit]:
        line = " | ".join(
            [
                str(row.get("market") or ""),
                str(row.get("spread") or ""),
                f"{row.get('depth_5c'):.1f}" if row.get("depth_5c") is not None else "",
                f"{row.get('imbalance'):.2f}" if row.get("imbalance") is not None else "",
                str(row.get("trades_15m") or ""),
                f"{row.get('time_remaining_h'):.1f}" if row.get("time_remaining_h") is not None else "",
                f"{row.get('incentive_ev_proxy'):.3f}" if row.get("incentive_ev_proxy") is not None else "",
                f"{row.get('risk_score'):.3f}" if row.get("risk_score") is not None else "",
                "yes" if row.get("eligible") else "no",
            ]
        )
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    result = run_phase1()
    print({"ok": result["ok"], "run_id": result["run_id"], "rows": len(result["rows"])})
    print(render_ranked_table(result["rows"]))
