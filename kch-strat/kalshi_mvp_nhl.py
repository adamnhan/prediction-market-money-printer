from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import re
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
KCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(KCH_DIR))

from src import platform_ops  # type: ignore
from kalshi_mvp_state import (  # type: ignore
    ensure_day,
    get_market_state,
    load_state,
    save_state,
)
from kalshi_mvp_ws import WSOrderbookClient  # type: ignore


EXCLUDE_KEYWORDS = (
    "over",
    "under",
    "total",
    "spread",
    "points",
    "goals",
)


EVENT_TEAMS_RE = re.compile(r"-(\d{2}[A-Z]{3}\d{2})([A-Z]{3})([A-Z]{3})$")
EVENT_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")
MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


@dataclass
class MarketPair:
    event_ticker: str
    event_start_ts: int | None
    market_a: dict[str, Any]
    market_b: dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ts_to_dt(ts: int | float | None) -> datetime | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except Exception:
        return None


def _parse_iso_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _event_start_ts(event: dict[str, Any]) -> int | None:
    for key in ("event_start_ts", "start_ts", "start_time", "startDate", "start_time_iso"):
        if key in event:
            dt = _ts_to_dt(event.get(key))
            if dt:
                return int(dt.timestamp())
    for key in ("start_time", "startDate", "start_date", "start"):
        dt = _parse_iso_dt(event.get(key))
        if dt:
            return int(dt.timestamp())
    event_ticker = (event.get("event_ticker") or event.get("ticker") or "")
    dt = _parse_event_date_from_ticker(event_ticker)
    if dt:
        return int(dt.timestamp())
    return None


def _parse_event_date_from_ticker(event_ticker: str) -> datetime | None:
    match = EVENT_DATE_RE.search(event_ticker.upper())
    if not match:
        return None
    year = 2000 + int(match.group(1))
    month = MONTH_MAP.get(match.group(2))
    day = int(match.group(3))
    if not month:
        return None
    # Use noon UTC as a neutral placeholder when only the date is known.
    return datetime(year, month, day, 12, 0, tzinfo=timezone.utc)


def _market_label(market: dict[str, Any]) -> str:
    return (
        (market.get("yes_sub_title") or market.get("subtitle") or market.get("title") or "")
        .strip()
    )


def _is_winner_market(market: dict[str, Any]) -> bool:
    text = " ".join(
        str(market.get(k, "") or "")
        for k in ("title", "subtitle", "sub_title", "ticker", "yes_sub_title")
    ).lower()
    if any(keyword in text for keyword in EXCLUDE_KEYWORDS):
        return False
    status = (market.get("status") or "").lower()
    if status in ("settled", "determined", "canceled", "closed"):
        return False
    return True


def _parse_event_teams(event_ticker: str) -> tuple[str, str] | None:
    match = EVENT_TEAMS_RE.search(event_ticker.upper())
    if not match:
        return None
    return match.group(2), match.group(3)


def _is_event_winner_market(event_ticker: str, market: dict[str, Any]) -> bool:
    ticker = (market.get("ticker") or "").upper()
    if not ticker.startswith("KXNHLGAME-"):
        return False
    if not ticker.startswith(event_ticker + "-"):
        return False
    teams = _parse_event_teams(event_ticker)
    if teams:
        if ticker.endswith("-" + teams[0]) or ticker.endswith("-" + teams[1]):
            return True
    return _is_winner_market(market)


def _select_two_markets(event_ticker: str, markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = [m for m in markets if _is_event_winner_market(event_ticker, m)]
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in filtered:
        label = _market_label(m)
        if not label:
            continue
        if label in seen:
            continue
        seen.add(label)
        unique.append(m)
        if len(unique) >= 2:
            break
    return unique


def _extract_prices_from_orderbook(payload: dict[str, Any]) -> dict[str, Any]:
    book = payload.get("orderbook") if isinstance(payload.get("orderbook"), dict) else payload
    result = {
        "yes_ask": None,
        "yes_bid": None,
        "yes_ask_size": None,
        "yes_bid_size": None,
        "no_ask": None,
        "no_bid": None,
        "no_ask_size": None,
        "no_bid_size": None,
    }

    def _best_price(levels: Any, want_min: bool) -> tuple[int | None, int | None]:
        if not isinstance(levels, list):
            return None, None
        best_price = None
        best_size = None
        for level in levels:
            if not isinstance(level, (list, tuple)) or len(level) < 2:
                continue
            price = level[0]
            size = level[1]
            if price is None or size is None:
                continue
            try:
                price_i = int(price)
                size_i = int(size)
            except Exception:
                continue
            if size_i <= 0:
                continue
            if best_price is None:
                best_price = price_i
                best_size = size_i
                continue
            if want_min and price_i < best_price:
                best_price = price_i
                best_size = size_i
            if not want_min and price_i > best_price:
                best_price = price_i
                best_size = size_i
        return best_price, best_size

    if isinstance(book, dict):
        if "yes_asks" in book:
            result["yes_ask"], result["yes_ask_size"] = _best_price(book.get("yes_asks"), True)
        if "yes_bids" in book:
            result["yes_bid"], result["yes_bid_size"] = _best_price(book.get("yes_bids"), False)
        if "no_asks" in book:
            result["no_ask"], result["no_ask_size"] = _best_price(book.get("no_asks"), True)
        if "no_bids" in book:
            result["no_bid"], result["no_bid_size"] = _best_price(book.get("no_bids"), False)
        if result["yes_ask"] is None and "yes" in book:
            result["yes_ask"], result["yes_ask_size"] = _best_price(book.get("yes"), True)
        if result["no_ask"] is None and "no" in book:
            result["no_ask"], result["no_ask_size"] = _best_price(book.get("no"), True)
        if result["yes_bid"] is None and "yes" in book:
            result["yes_bid"], result["yes_bid_size"] = _best_price(book.get("yes"), False)
        if result["no_bid"] is None and "no" in book:
            result["no_bid"], result["no_bid_size"] = _best_price(book.get("no"), False)
        for key in ("yes_ask", "yes_bid", "no_ask", "no_bid"):
            if result[key] is None and key in book:
                try:
                    result[key] = int(book.get(key))
                except Exception:
                    pass
    return result


def _spread_ok(best_bid: int | None, best_ask: int | None, max_spread: int) -> bool:
    if best_bid is None or best_ask is None:
        return False
    return (best_ask - best_bid) <= max_spread


def _depth_ok(best_ask_size: int | None, min_depth: int) -> bool:
    if best_ask_size is None:
        return False
    return best_ask_size >= min_depth


def _ensure_csv(path: Path, header: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def _append_csv(path: Path, row: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


REST_LAST_TS = 0.0


async def _request_any_base(
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    min_interval_ms: int = 250,
    max_retries: int = 3,
) -> dict[str, Any]:
    global REST_LAST_TS
    attempt = 0
    backoff_s = 0.5
    while attempt <= max_retries:
        now = time.monotonic()
        wait_s = max(0.0, (min_interval_ms / 1000.0) - (now - REST_LAST_TS))
        if wait_s > 0:
            await asyncio.sleep(wait_s)
        REST_LAST_TS = time.monotonic()

        for base in (platform_ops.KALSHI_ELECTIONS_BASE_URL, platform_ops.KALSHI_BASE_URL):
            url = f"{base}{path}"
            resp = await platform_ops._request(method, url, params=params)  # type: ignore
            if resp.get("ok"):
                return resp
            if resp.get("status") == 429:
                break
        if resp.get("status") == 429:
            await asyncio.sleep(backoff_s)
            backoff_s = min(backoff_s * 2, 8.0)
            attempt += 1
            continue
        return resp
    return resp


async def _fetch_events(series_ticker: str, limit: int = 200, min_interval_ms: int = 250) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cursor = None
    while True:
        params = {"limit": limit, "series_ticker": series_ticker}
        if cursor:
            params["cursor"] = cursor
        resp = await _request_any_base("GET", "/events", params=params, min_interval_ms=min_interval_ms)
        if not resp.get("ok"):
            return events
        data = resp.get("data") or {}
        page = data.get("events") if isinstance(data, dict) else data
        if not isinstance(page, list):
            page = []
        events.extend(page)
        cursor = data.get("cursor") or data.get("next_cursor")
        if not cursor:
            break
    return events


async def _fetch_markets_for_event(
    event_ticker: str,
    limit: int = 200,
    min_interval_ms: int = 250,
) -> list[dict[str, Any]]:
    params = {"event_ticker": event_ticker, "limit": limit}
    resp = await _request_any_base("GET", "/markets", params=params, min_interval_ms=min_interval_ms)
    if not resp.get("ok"):
        return []
    data = resp.get("data") or {}
    markets = data.get("markets") if isinstance(data, dict) else data
    if not isinstance(markets, list):
        return []
    return markets


async def _fetch_orderbook(market_ticker: str, min_interval_ms: int = 250) -> dict[str, Any] | None:
    resp = await _request_any_base("GET", f"/markets/{market_ticker}/orderbook", min_interval_ms=min_interval_ms)
    if not resp.get("ok"):
        return None
    data = resp.get("data") or {}
    return data


async def _fetch_market(market_ticker: str, min_interval_ms: int = 250) -> dict[str, Any] | None:
    resp = await _request_any_base("GET", f"/markets/{market_ticker}", min_interval_ms=min_interval_ms)
    if not resp.get("ok"):
        return None
    data = resp.get("data") or {}
    if isinstance(data, dict) and "market" in data and isinstance(data["market"], dict):
        return data["market"]
    if isinstance(data, dict):
        return data
    return None


def _derive_settlement(market: dict[str, Any]) -> str | None:
    for key in ("result", "resolution", "outcome", "settlement"):
        value = market.get(key)
        if isinstance(value, str):
            value = value.lower()
            if value in ("yes", "no"):
                return value
    if market.get("settlement_value") in (0, 1):
        return "yes" if market.get("settlement_value") == 1 else "no"
    return None


def _build_status_line(
    now: datetime,
    open_count: int,
    entering_count: int,
    done_count: int,
    watching_count: int,
    ws_connected: bool,
    ws_subscribed: int,
) -> str:
    return (
        f"{now.isoformat(timespec='seconds')} status "
        f"watching={watching_count} entering={entering_count} open={open_count} done={done_count} "
        f"ws_connected={int(ws_connected)} ws_subscribed={ws_subscribed}"
    )


async def run_loop(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_path)
    trades_csv = output_dir / "kxnhl_mvp_trades.csv"
    markets_csv = output_dir / "kxnhl_mvp_markets.csv"
    daily_summary_path = output_dir / "daily_summary.json"

    _ensure_csv(trades_csv, ["time", "market", "side", "qty", "price_cents"])
    _ensure_csv(
        markets_csv,
        [
            "event_ticker",
            "entry_time",
            "ratio",
            "capital_cents",
            "market_a",
            "market_b",
            "qty_a",
            "qty_b",
            "price_a",
            "price_b",
            "status",
            "pnl_cents",
        ],
    )
    markets_header = [
        "event_ticker",
        "entry_time",
        "ratio",
        "capital_cents",
        "market_a",
        "market_b",
        "qty_a",
        "qty_b",
        "price_a",
        "price_b",
        "status",
        "pnl_cents",
    ]

    state = load_state(state_path)
    logger = logging.getLogger("kalshi_mvp_nhl")
    last_status_ts = time.monotonic()

    events_cache: dict[str, Any] = {"ts": 0.0, "events": []}
    markets_cache: dict[str, dict[str, Any]] = {}

    ws_client = WSOrderbookClient.from_env()
    ws_client.start()

    while True:
        loop_start = time.monotonic()
        now = _utc_now()
        ensure_day(state, now.date().isoformat())

        fetched_events = False
        if time.monotonic() - events_cache["ts"] > args.events_refresh_sec:
            events_cache["events"] = await _fetch_events(
                args.series_ticker,
                min_interval_ms=args.rest_min_interval_ms,
            )
            events_cache["ts"] = time.monotonic()
            fetched_events = True
        events = events_cache["events"]
        total_events = len(events)
        parsed_events = 0
        future_events = 0
        now_ts = int(now.timestamp())
        event_pairs: list[MarketPair] = []
        for event in events:
            event_ticker = (event.get("event_ticker") or event.get("ticker") or "").upper()
            if not event_ticker:
                continue
            event_ts = _event_start_ts(event)
            if event_ts is None:
                continue
            parsed_events += 1
            if event_ts < int(now.timestamp()):
                continue
            future_events += 1
            cached = markets_cache.get(event_ticker)
            if cached is None or time.monotonic() - cached["ts"] > args.markets_refresh_sec:
                markets = await _fetch_markets_for_event(
                    event_ticker,
                    min_interval_ms=args.rest_min_interval_ms,
                )
                markets_cache[event_ticker] = {"ts": time.monotonic(), "markets": markets}
            else:
                markets = cached["markets"]
            pair = _select_two_markets(event_ticker, markets)
            if len(pair) < 2:
                continue
            event_pairs.append(
                MarketPair(
                    event_ticker=event_ticker,
                    event_start_ts=event_ts,
                    market_a=pair[0],
                    market_b=pair[1],
                )
            )
        if fetched_events:
            logger.info(
                "events_summary total=%s parsed_start=%s future=%s now_ts=%s",
                total_events,
                parsed_events,
                future_events,
                now_ts,
            )
            if total_events > 0 and parsed_events == 0:
                sample = events[0]
                logger.info(
                    "events_sample_keys keys=%s values=%s",
                    sorted(list(sample.keys())),
                    {k: sample.get(k) for k in sorted(list(sample.keys()))[:6]},
                )

        open_markets = 0
        entering_markets = 0
        done_markets = 0
        watching_markets = 0
        total_deployed_cents = 0

        desired_tickers: set[str] = set()
        for pair in event_pairs:
            event_key = pair.event_ticker
            mstate = get_market_state(state, event_key)
            status = mstate.get("status") or "WATCHING"

            if status in ("OPEN", "ENTERING"):
                total_deployed_cents += int(mstate.get("entry", {}).get("capital_cents") or 0)
            if status == "OPEN":
                open_markets += 1
            elif status == "ENTERING":
                entering_markets += 1
            elif status == "DONE":
                done_markets += 1
            else:
                watching_markets += 1

            if status == "DONE":
                continue

            # Entry window check
            if pair.event_start_ts is None:
                logger.info("skip event=%s missing_start_time", event_key)
                continue
            seconds_to_start = pair.event_start_ts - int(now.timestamp())
            if seconds_to_start > args.watch_start_hours * 3600:
                continue
            if seconds_to_start < args.stop_entry_minutes * 60:
                continue

            if pair.market_a.get("ticker"):
                desired_tickers.add(pair.market_a.get("ticker"))
            if pair.market_b.get("ticker"):
                desired_tickers.add(pair.market_b.get("ticker"))

            if status == "WATCHING":
                if open_markets + entering_markets >= args.max_concurrent:
                    continue
                if args.max_daily_new and state.get("daily_new_markets", 0) >= args.max_daily_new:
                    continue
                if args.max_total_deployed and total_deployed_cents >= int(args.max_total_deployed * 100):
                    continue

                ticker_a = pair.market_a.get("ticker", "")
                ticker_b = pair.market_b.get("ticker", "")
                if not ticker_a or not ticker_b:
                    continue

                ws_prices_a = ws_client.get_best_prices(ticker_a)
                ws_prices_b = ws_client.get_best_prices(ticker_b)
                if ws_prices_a is None or ws_prices_b is None:
                    if not args.ws_price_fallback_rest:
                        continue
                    book_a = await _fetch_orderbook(ticker_a, min_interval_ms=args.rest_min_interval_ms)
                    book_b = await _fetch_orderbook(ticker_b, min_interval_ms=args.rest_min_interval_ms)
                    if not book_a or not book_b:
                        continue
                    prices_a = _extract_prices_from_orderbook(book_a)
                    prices_b = _extract_prices_from_orderbook(book_b)
                    yes_ask_a = prices_a.get("yes_ask")
                    yes_ask_b = prices_b.get("yes_ask")
                    yes_bid_a = prices_a.get("yes_bid")
                    yes_bid_b = prices_b.get("yes_bid")
                    ask_size_a = prices_a.get("yes_ask_size")
                    ask_size_b = prices_b.get("yes_ask_size")
                else:
                    yes_ask_a = ws_prices_a.yes_ask
                    yes_ask_b = ws_prices_b.yes_ask
                    yes_bid_a = ws_prices_a.yes_bid
                    yes_bid_b = ws_prices_b.yes_bid
                    ask_size_a = ws_prices_a.yes_ask_size
                    ask_size_b = ws_prices_b.yes_ask_size
                spread_a = None
                spread_b = None
                if yes_bid_a is not None and yes_ask_a is not None:
                    spread_a = yes_ask_a - yes_bid_a
                if yes_bid_b is not None and yes_ask_b is not None:
                    spread_b = yes_ask_b - yes_bid_b

                logger.info(
                    "watch event=%s a=%s b=%s ask_a=%s ask_b=%s spread_a=%s spread_b=%s depth_a=%s depth_b=%s",
                    event_key,
                    ticker_a,
                    ticker_b,
                    yes_ask_a,
                    yes_ask_b,
                    spread_a,
                    spread_b,
                    ask_size_a,
                    ask_size_b,
                )

                if yes_ask_a is None or yes_ask_b is None:
                    continue

                spread_ok = _spread_ok(yes_bid_a, yes_ask_a, args.max_spread_cents)
                spread_ok = spread_ok and _spread_ok(yes_bid_b, yes_ask_b, args.max_spread_cents)
                depth_ok = _depth_ok(ask_size_a, args.min_depth)
                depth_ok = depth_ok and _depth_ok(ask_size_b, args.min_depth)
                if not spread_ok or not depth_ok:
                    mstate["stable_count"] = 0
                    mstate["last_band_ok"] = False
                    continue

                cheap_price = min(yes_ask_a, yes_ask_b)
                band_ok = args.band_low <= cheap_price <= args.band_high
                if band_ok:
                    if mstate.get("last_band_ok"):
                        mstate["stable_count"] = int(mstate.get("stable_count", 0)) + 1
                    else:
                        mstate["stable_count"] = 1
                else:
                    mstate["stable_count"] = 0
                mstate["last_band_ok"] = band_ok

                logger.info(
                    "band event=%s cheap=%s band_ok=%s stable=%s spread_ok=%s depth_ok=%s",
                    event_key,
                    cheap_price,
                    band_ok,
                    mstate.get("stable_count"),
                    spread_ok,
                    depth_ok,
                )

                if not band_ok or mstate.get("stable_count", 0) < args.stable_n:
                    continue

                capital_cents = int(args.capital_per_market * 100)
                cheap_notional = int(capital_cents / (1.0 + args.ratio))
                exp_notional = capital_cents - cheap_notional
                if yes_ask_a <= yes_ask_b:
                    cheap_price = yes_ask_a
                    exp_price = yes_ask_b
                    cheap_ticker = ticker_a
                    exp_ticker = ticker_b
                else:
                    cheap_price = yes_ask_b
                    exp_price = yes_ask_a
                    cheap_ticker = ticker_b
                    exp_ticker = ticker_a

                cheap_qty = math.floor(cheap_notional / cheap_price)
                exp_qty = math.floor(exp_notional / exp_price)
                if cheap_qty < args.min_qty or exp_qty < args.min_qty:
                    logger.info("skip event=%s qty_low cheap=%s exp=%s", event_key, cheap_qty, exp_qty)
                    continue

                entry = {
                    "ratio": args.ratio,
                    "capital_cents": capital_cents,
                    "cheap_ticker": cheap_ticker,
                    "exp_ticker": exp_ticker,
                    "cheap_qty": cheap_qty,
                    "exp_qty": exp_qty,
                    "cheap_price": cheap_price,
                    "exp_price": exp_price,
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                }
                mstate["entry"] = entry
                mstate["entered_at"] = now.isoformat()
                mstate["entry_deadline_ts"] = int(time.time()) + args.entry_timeout_sec
                mstate["entry_timeout_sec"] = args.entry_timeout_sec
                mstate["status"] = "ENTERING"
                state["daily_new_markets"] = int(state.get("daily_new_markets", 0)) + 1

                if args.paper:
                    mstate["fills"] = [
                        {
                            "ticker": cheap_ticker,
                            "side": "yes",
                            "price": cheap_price,
                            "qty": cheap_qty,
                            "time": now.isoformat(),
                        },
                        {
                            "ticker": exp_ticker,
                            "side": "yes",
                            "price": exp_price,
                            "qty": exp_qty,
                            "time": now.isoformat(),
                        },
                    ]
                    _append_csv(trades_csv, [now.isoformat(), cheap_ticker, "yes", cheap_qty, cheap_price])
                    _append_csv(trades_csv, [now.isoformat(), exp_ticker, "yes", exp_qty, exp_price])
                    mstate["status"] = "OPEN"
                    continue

                order_ids: list[str] = []
                for ticker, qty, price in (
                    (cheap_ticker, cheap_qty, cheap_price),
                    (exp_ticker, exp_qty, exp_price),
                ):
                    payload = {
                        "ticker": ticker,
                        "type": "limit",
                        "action": "buy",
                        "side": "yes",
                        "count": qty,
                        "yes_price": price,
                        "client_order_id": f"kxnhl-mvp-{ticker}-{int(time.time()*1000)}",
                    }
                    resp = await platform_ops.place_order(payload)
                    order_id = platform_ops._extract_order_id(resp.get("data"))  # type: ignore
                    if order_id:
                        order_ids.append(order_id)
                        state.setdefault("orders", {})[order_id] = {
                            "ticker": ticker,
                            "side": "yes",
                            "price": price,
                            "qty": qty,
                            "filled": 0,
                        }
                mstate["order_ids"] = order_ids

            if status == "ENTERING" and not args.paper:
                order_ids = list(mstate.get("order_ids") or [])
                if not order_ids:
                    mstate["status"] = "WATCHING"
                    continue
                all_filled = True
                for order_id in order_ids:
                    resp = await platform_ops.get_order_status(order_id)
                    if not resp.get("ok"):
                        all_filled = False
                        continue
                    data = resp.get("data") or {}
                    status_str, filled, remaining = platform_ops._extract_fill_status(data)  # type: ignore
                    order_state = state.setdefault("orders", {}).setdefault(order_id, {})
                    prev_filled = int(order_state.get("filled", 0))
                    if filled > prev_filled:
                        delta = filled - prev_filled
                        order_state["filled"] = filled
                        mstate.setdefault("fills", []).append(
                            {
                                "ticker": order_state.get("ticker"),
                                "side": order_state.get("side"),
                                "price": order_state.get("price"),
                                "qty": delta,
                                "time": now.isoformat(),
                            }
                        )
                        _append_csv(
                            trades_csv,
                            [
                                now.isoformat(),
                                order_state.get("ticker"),
                                order_state.get("side"),
                                delta,
                                order_state.get("price"),
                            ],
                        )
                    if status_str not in ("filled",) or remaining != 0:
                        all_filled = False
                if all_filled:
                    mstate["status"] = "OPEN"
                elif int(time.time()) > int(mstate.get("entry_deadline_ts") or 0):
                    for order_id in order_ids:
                        await platform_ops.cancel_order(order_id)
                    mstate["status"] = "DONE"
                    mstate.setdefault("result", {})["reason"] = "entry_timeout"

            if status == "OPEN":
                ticker_a = pair.market_a.get("ticker", "")
                ticker_b = pair.market_b.get("ticker", "")
                mkt_a = await _fetch_market(ticker_a, min_interval_ms=args.rest_min_interval_ms) if ticker_a else None
                mkt_b = await _fetch_market(ticker_b, min_interval_ms=args.rest_min_interval_ms) if ticker_b else None
                status_a = (mkt_a.get("status") if mkt_a else "") if mkt_a else ""
                status_b = (mkt_b.get("status") if mkt_b else "") if mkt_b else ""
                done_a = str(status_a).lower() in ("settled", "determined")
                done_b = str(status_b).lower() in ("settled", "determined")
                if done_a and done_b:
                    result_a = _derive_settlement(mkt_a or {})
                    result_b = _derive_settlement(mkt_b or {})
                    pnl = 0
                    for ticker, result in ((ticker_a, result_a), (ticker_b, result_b)):
                        fills = [f for f in (mstate.get("fills") or []) if f.get("ticker") == ticker]
                        for f in fills:
                            price = int(f.get("price") or 0)
                            qty = int(f.get("qty") or 0)
                            if result == "yes":
                                pnl += qty * (100 - price)
                            elif result == "no":
                                pnl -= qty * price
                    mstate.setdefault("result", {})["pnl_cents"] = pnl
                    mstate["status"] = "DONE"

        # update market summary csv
        rows = []
        for event_key, mstate in (state.get("markets") or {}).items():
            entry = mstate.get("entry") or {}
            rows.append(
                {
                    "event_ticker": event_key,
                    "entry_time": mstate.get("entered_at"),
                    "ratio": entry.get("ratio"),
                    "capital_cents": entry.get("capital_cents"),
                    "market_a": entry.get("ticker_a"),
                    "market_b": entry.get("ticker_b"),
                    "qty_a": entry.get("cheap_qty") if entry.get("cheap_ticker") == entry.get("ticker_a") else entry.get("exp_qty"),
                    "qty_b": entry.get("cheap_qty") if entry.get("cheap_ticker") == entry.get("ticker_b") else entry.get("exp_qty"),
                    "price_a": entry.get("cheap_price") if entry.get("cheap_ticker") == entry.get("ticker_a") else entry.get("exp_price"),
                    "price_b": entry.get("cheap_price") if entry.get("cheap_ticker") == entry.get("ticker_b") else entry.get("exp_price"),
                    "status": mstate.get("status"),
                    "pnl_cents": (mstate.get("result") or {}).get("pnl_cents"),
                }
            )
        with markets_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=markets_header)
            writer.writeheader()
            if rows:
                writer.writerows(rows)

        daily_summary = {
            "day": state.get("day"),
            "daily_new_markets": state.get("daily_new_markets"),
            "open_markets": open_markets,
            "entering_markets": entering_markets,
            "done_markets": done_markets,
            "watching_markets": watching_markets,
            "total_deployed_cents": total_deployed_cents,
        }
        daily_summary_path.write_text(json.dumps(daily_summary, indent=2), encoding="utf-8")

        save_state(state_path, state)
        await ws_client.set_desired_tickers(desired_tickers)

        if time.monotonic() - last_status_ts >= 60:
            print(
                _build_status_line(
                    now,
                    open_markets,
                    entering_markets,
                    done_markets,
                    watching_markets,
                    ws_client.connected,
                    len(ws_client.active_tickers),
                )
            )
            last_status_ts = time.monotonic()

        elapsed = time.monotonic() - loop_start
        sleep_for = max(0.0, args.snapshot_interval_sec - elapsed)
        await asyncio.sleep(sleep_for)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kalshi NHL MVP band-triggered strategy")
    parser.add_argument("--series_ticker", default="KXNHLGAME")
    parser.add_argument("--band_low", type=int, default=44)
    parser.add_argument("--band_high", type=int, default=50)
    parser.add_argument("--ratio", type=float, default=0.50)
    parser.add_argument("--capital_per_market", type=float, default=100)
    parser.add_argument("--watch_start_hours", type=int, default=4)
    parser.add_argument("--stop_entry_minutes", type=int, default=10)
    parser.add_argument("--snapshot_interval_sec", type=int, default=30)
    parser.add_argument("--stable_n", type=int, default=3)
    parser.add_argument("--max_spread_cents", type=int, default=6)
    parser.add_argument("--min_depth", type=int, default=50)
    parser.add_argument("--entry_timeout_sec", type=int, default=300)
    parser.add_argument("--max_concurrent", type=int, default=10)
    parser.add_argument("--max_daily_new", type=int, default=20)
    parser.add_argument("--max_total_deployed", type=float, default=0.0)
    parser.add_argument("--min_qty", type=int, default=1)
    parser.add_argument("--paper", type=str, default="true")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "kch-strat" / "phase5_out"))
    parser.add_argument("--state_path", default=str(REPO_ROOT / "kch-strat" / "phase5_out" / "kalshi_mvp_state.json"))
    parser.add_argument("--events_refresh_sec", type=int, default=86400)
    parser.add_argument("--markets_refresh_sec", type=int, default=86400)
    parser.add_argument("--rest_min_interval_ms", type=int, default=250)
    parser.add_argument("--ws_price_fallback_rest", type=str, default="false")
    return parser


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "y")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.paper = _parse_bool(str(args.paper))
    args.max_total_deployed = float(args.max_total_deployed) if args.max_total_deployed else 0.0
    args.ws_price_fallback_rest = _parse_bool(str(args.ws_price_fallback_rest))

    logging.basicConfig(
        level=logging.INFO,
        format="[KXNHLMVP] %(asctime)s %(message)s",
    )
    asyncio.run(run_loop(args))


if __name__ == "__main__":
    main()
