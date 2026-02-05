#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import sqlite3

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nba_engine.config import load_config
from nba_engine.phase5 import RestClient, _extract_orderbook_levels, _derive_best_prices

from trade_logger import TradeLogger
from shadow_executor import execute_bundle, execute_bundle_leg_in
from heartbeat import HeartbeatState


@dataclass(frozen=True)
class LegSpec:
    ticker: str
    k: int
    w: float
    limit_price: float
    side: str
    phat: float
    delta: float
    px_bid: float | None
    px_ask: float | None
    px_used: float


def _parse_spread_threshold(market_ticker: str) -> tuple[str | None, int | None]:
    suffix = market_ticker.split("-")[-1]
    for i, ch in enumerate(suffix):
        if ch.isdigit() or ch == "-":
            team = suffix[:i]
            try:
                k = int(suffix[i:])
            except ValueError:
                return None, None
            return team, k
    return None, None


def _parse_total_threshold(market_ticker: str) -> int | None:
    suffix = market_ticker.split("-")[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _load_ladder(csv_path: str, event_key: str, family: str, team: str | None) -> list[dict[str, Any]]:
    rows = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("event_key") != event_key:
                continue
            series = row.get("series_ticker")
            ticker = row.get("market_ticker") or ""
            if family == "spread" and series == "KXNBASPREAD":
                t, k = _parse_spread_threshold(ticker)
                if t and k is not None and (team is None or t == team):
                    rows.append({"k": k, "ticker": ticker, "team": t})
            elif family == "total" and series == "KXNBATOTAL":
                k = _parse_total_threshold(ticker)
                if k is not None:
                    rows.append({"k": k, "ticker": ticker, "team": "TOTAL"})
    return rows


def _load_margins(results_csv: str) -> np.ndarray:
    margins = []
    with open(results_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            val = row.get("final_margin")
            if val is None or val == "":
                continue
            try:
                margins.append(int(val))
            except ValueError:
                continue
    if not margins:
        raise RuntimeError("no margins found")
    return np.array(margins, dtype=int)


def _p_hat(margins: np.ndarray, ks: list[int]) -> np.ndarray:
    out = []
    for k in ks:
        out.append(float(np.mean(margins >= k)))
    return np.array(out, dtype=float)


def _isotonic_decreasing(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    y = -values.astype(float)
    n = len(y)
    weights = np.ones(n, dtype=float)
    level = y.copy()
    i = 0
    while i < n - 1:
        if level[i] <= level[i + 1]:
            i += 1
            continue
        new_weight = weights[i] + weights[i + 1]
        new_level = (weights[i] * level[i] + weights[i + 1] * level[i + 1]) / new_weight
        level[i] = new_level
        weights[i] = new_weight
        level = np.delete(level, i + 1)
        weights = np.delete(weights, i + 1)
        n -= 1
        if i > 0:
            i -= 1
    expanded = np.repeat(level, weights.astype(int))
    if expanded.size != values.size:
        expanded = np.interp(np.arange(values.size), np.linspace(0, values.size - 1, level.size), level)
    return -expanded


def _top_of_book(payload: dict[str, Any]) -> dict[str, Any]:
    levels = _extract_orderbook_levels(payload)
    best_prices = _derive_best_prices(levels)
    yes_bids = sorted(levels["yes_bids"], key=lambda x: x[0], reverse=True)
    yes_asks = sorted(levels["yes_asks"], key=lambda x: x[0])
    yes_bid_sz = yes_bids[0][1] if yes_bids else None
    yes_ask_sz = yes_asks[0][1] if yes_asks else None
    return {
        "yes_bid": best_prices.get("yes_bid"),
        "yes_ask": best_prices.get("yes_ask"),
        "yes_bid_sz": yes_bid_sz,
        "yes_ask_sz": yes_ask_sz,
    }


def _price_for_leg(top: dict[str, Any], action: str) -> tuple[float | None, bool]:
    used_mid = False
    if action == "buy":
        if top.get("yes_ask") is not None:
            return float(top["yes_ask"]) / 100.0, used_mid
    if action == "sell":
        if top.get("yes_bid") is not None:
            return float(top["yes_bid"]) / 100.0, used_mid
    if top.get("yes_bid") is not None and top.get("yes_ask") is not None:
        used_mid = True
        mid = (float(top["yes_bid"]) + float(top["yes_ask"])) / 200.0
        return mid, used_mid
    return None, used_mid


def _edge_ticks(top: dict[str, Any], action: str, limit_price: float) -> int | None:
    limit_cents = int(round(limit_price * 100))
    if action == "buy":
        ask = top.get("yes_ask")
        if ask is None:
            return None
        return int(limit_cents - int(ask))
    bid = top.get("yes_bid")
    if bid is None:
        return None
    return int(int(bid) - limit_cents)


def _size_ratio(top: dict[str, Any], action: str, qty: int) -> float | None:
    if action == "buy":
        sz = top.get("yes_ask_sz")
    else:
        sz = top.get("yes_bid_sz")
    if sz is None:
        return None
    if qty <= 0:
        return 0.0
    return float(sz) / float(qty)


def _mid_price(top: dict[str, Any]) -> float | None:
    yes_bid = top.get("yes_bid")
    yes_ask = top.get("yes_ask")
    if yes_bid is not None and yes_ask is not None:
        return (float(yes_bid) + float(yes_ask)) / 200.0
    if yes_ask is not None:
        return float(yes_ask) / 100.0
    if yes_bid is not None:
        return float(yes_bid) / 100.0
    return None


def _bundle_max_loss(legs: list[LegSpec]) -> float:
    if not legs:
        return 0.0
    ks = sorted({int(leg.k) for leg in legs})
    bounds = [-10**9] + ks + [10**9]
    const_cost = 0.0
    for leg in legs:
        const_cost += leg.w * (-leg.limit_price)
    vals = []
    for j in range(len(bounds) - 1):
        lo, hi = bounds[j], bounds[j + 1]
        rep = lo if lo > -10**8 else (hi - 1)
        ind_term = 0.0
        for leg in legs:
            ind_term += leg.w * (1.0 if rep >= leg.k else 0.0)
        vals.append(ind_term + const_cost)
    return -min(vals) if vals else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase F runner (shadow execution).")
    parser.add_argument("--event-key", default="")
    parser.add_argument("--family", choices=["spread", "total"], default="spread")
    parser.add_argument("--team", default="")
    parser.add_argument("--markets-csv", default="data/nba_historical_markets.csv")
    parser.add_argument("--results-csv", default="data/nba_results.csv")
    parser.add_argument("--mode", default="shadow")
    parser.add_argument("--weights", default="-2,-1,1,2")
    parser.add_argument("--band", default="6,11")
    parser.add_argument("--max-loss", type=float, default=0.10)
    parser.add_argument("--mid-loss", type=float, default=0.10)
    parser.add_argument("--min-ev", type=float, default=0.01)
    parser.add_argument("--max-mids", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--min-edge-ticks", type=int, default=0)
    parser.add_argument("--max-edge-ticks", type=int, default=2)
    parser.add_argument("--min-size-ratio", type=float, default=1.0)
    parser.add_argument("--spread-penalty", type=float, default=0.0)
    parser.add_argument("--size-penalty", type=float, default=0.0)
    parser.add_argument("--max-bundle-loss", type=float, default=0.0)
    parser.add_argument("--use-win-prob", action="store_true")
    parser.add_argument("--win-prob-strength", type=float, default=0.5)
    parser.add_argument("--ttl-s", type=float, default=2.0)
    parser.add_argument("--poll-s", type=float, default=0.2)
    parser.add_argument("--ledger-db", default="convev options hedge strat/phase_f_ledger.sqlite")
    parser.add_argument("--log-jsonl", default="convev options hedge strat/phase_f.log.jsonl")
    parser.add_argument("--heartbeat-s", type=float, default=10.0)
    parser.add_argument("--poll-markets-s", type=float, default=86400.0)
    parser.add_argument("--loop-sleep-s", type=float, default=1.0)
    parser.add_argument("--leg-in", action="store_true")
    parser.add_argument("--max-unhedged-loss", type=float, default=0.05)
    parser.add_argument("--max-unhedged-ttl", type=float, default=15.0)
    parser.add_argument("--slope-cap", type=float, default=1.0)
    parser.add_argument("--log-books", action="store_true")
    parser.add_argument("--books-db", default="convev options hedge strat/phase_f_books.sqlite")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    run_id = uuid.uuid4().hex
    logger = TradeLogger(args.ledger_db, args.log_jsonl)
    hb = HeartbeatState()
    books_conn = sqlite3.connect(args.books_db) if args.log_books else None
    if books_conn:
        books_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS l1_books (
                id INTEGER PRIMARY KEY,
                ts INTEGER NOT NULL,
                event_key TEXT,
                market_ticker TEXT NOT NULL,
                yes_bid INTEGER,
                yes_ask INTEGER,
                yes_bid_sz INTEGER,
                yes_ask_sz INTEGER
            )
            """
        )
        books_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_l1_books_ticker_ts ON l1_books(market_ticker, ts)"
        )
        books_conn.commit()

    team = args.team.strip().upper() or None
    event_key = args.event_key.strip()

    margins = _load_margins(args.results_csv)

    _ = args.weights  # retained for CLI compatibility; weight magnitudes not used in sign-only selection
    band = tuple(int(x.strip()) for x in args.band.split(","))

    rest_client = RestClient(
        config.kalshi_rest_url or "https://api.elections.kalshi.com/trade-api/v2",
        config.kalshi_key_id,
        config.kalshi_private_key_path,
    )

    def book_provider(tickers: list[str]) -> dict[str, dict[str, Any]]:
        out = {}
        for ticker in tickers:
            try:
                out[ticker] = rest_client.get_orderbook(ticker)
            except Exception as exc:
                hb.record_error("orderbook_error")
                logger.log_error(
                    {
                        "ts": int(time.time()),
                        "event_key": ticker,
                        "error": str(exc),
                        "context": "get_orderbook",
                    }
                )
        return out

    def _fetch_markets_for_series(series_ticker: str) -> list[dict[str, Any]]:
        payload = rest_client.request(
            "GET",
            "/markets",
            params={"series_ticker": series_ticker, "status": "open", "limit": 100},
        )
        return payload.get("markets") or []

    def _discover_markets() -> dict[str, list[dict[str, Any]]]:
        by_event: dict[str, list[dict[str, Any]]] = {}
        for series in ("KXNBAGAME", "KXNBASPREAD", "KXNBATOTAL"):
            markets = _fetch_markets_for_series(series)
            for m in markets:
                ticker = m.get("ticker") or m.get("market_ticker")
                if not ticker:
                    continue
                parts = str(ticker).split("-")
                if len(parts) < 2:
                    continue
                event_key = parts[1]
                by_event.setdefault(event_key, []).append(m)
        return by_event

    def _ladder_from_markets(markets: list[dict[str, Any]], family: str, team: str | None) -> list[dict[str, Any]]:
        rows = []
        for m in markets:
            ticker = m.get("ticker") or m.get("market_ticker") or ""
            series = m.get("series_ticker") or (ticker.split("-")[0] if ticker else "")
            if family == "spread" and series == "KXNBASPREAD":
                t, k = _parse_spread_threshold(ticker)
                if t and k is not None and (team is None or t == team):
                    rows.append({"k": k, "ticker": ticker, "team": t})
            elif family == "total" and series == "KXNBATOTAL":
                k = _parse_total_threshold(ticker)
                if k is not None:
                    rows.append({"k": k, "ticker": ticker, "team": "TOTAL"})
        return rows

    def _win_markets(markets: list[dict[str, Any]]) -> dict[str, str]:
        out: dict[str, str] = {}
        for m in markets:
            ticker = m.get("ticker") or m.get("market_ticker") or ""
            series = m.get("series_ticker") or (ticker.split("-")[0] if ticker else "")
            if series != "KXNBAGAME" or not ticker:
                continue
            parts = ticker.split("-")
            if len(parts) < 3:
                continue
            team_abbr = parts[-1]
            out[team_abbr] = ticker
        return out

    def _parse_event_teams(event_key: str) -> tuple[str | None, str | None]:
        if len(event_key) < 13:
            return None, None
        teams = event_key[7:]
        if len(teams) != 6:
            return None, None
        away = teams[:3]
        home = teams[3:]
        return away, home

    last_hb = time.time()
    last_summary = time.time()
    last_market_poll = 0.0
    markets_by_event: dict[str, list[dict[str, Any]]] = {}
    market_count = 0
    last_skip_reason: dict[str, str] = {}
    last_skip_ts: dict[str, float] = {}
    while True:
        hb.signals += 1
        now = time.time()
        if not markets_by_event or (now - last_market_poll) >= args.poll_markets_s:
            markets_by_event = _discover_markets()
            last_market_poll = now
            market_count = sum(len(v) for v in markets_by_event.values())
            logger.log_heartbeat(
                {
                    **hb.snapshot(),
                    "event_count": len(markets_by_event),
                    "market_count": market_count,
                }
            )

        if event_key:
            keys_to_check = [event_key]
        else:
            keys_to_check = list(markets_by_event.keys())

        for key in keys_to_check:
            ladder = _ladder_from_markets(markets_by_event.get(key, []), args.family, team)
            if not ladder:
                reason = "NO_LADDER"
                now_ts = int(time.time())
                if last_skip_reason.get(key) != reason or (now_ts - last_skip_ts.get(key, 0)) > 300:
                    logger.log_skip({"ts": now_ts, "event_key": key, "reason": reason, "details": {}})
                    last_skip_reason[key] = reason
                    last_skip_ts[key] = now_ts
                continue
            ladder = sorted(ladder, key=lambda r: r["k"])
            ks = [row["k"] for row in ladder]
            phat = _p_hat(margins, ks)
            phat = _isotonic_decreasing(phat)
            if args.use_win_prob and args.family == "spread":
                away, home = _parse_event_teams(key)
                win_markets = _win_markets(markets_by_event.get(key, []))
                win_tickers = []
                if away and away in win_markets:
                    win_tickers.append(win_markets[away])
                if home and home in win_markets:
                    win_tickers.append(win_markets[home])
                win_books = book_provider(win_tickers) if win_tickers else {}
                p_home = None
                if home and home in win_markets:
                    top = _top_of_book(win_books.get(win_markets[home], {}))
                    p_home = _mid_price(top) if top else None
                p_away = None
                if away and away in win_markets:
                    top = _top_of_book(win_books.get(win_markets[away], {}))
                    p_away = _mid_price(top) if top else None
                # use home/away implied win prob to nudge phat
                if p_home is not None and p_away is None:
                    p_away = 1.0 - p_home
                if p_away is not None and p_home is None:
                    p_home = 1.0 - p_away
                if p_home is not None and p_away is not None:
                    # note: ladder rows carry team abbrev; adjust per team when building legs
                    implied = {"home": p_home, "away": p_away}
                else:
                    implied = {}
            else:
                implied = {}

            payloads = book_provider([row["ticker"] for row in ladder])
            tops = {t: _top_of_book(p) for t, p in payloads.items()}
            if len(ladder) < 3:
                reason = "LADDER_TOO_SHORT"
                now_ts = int(time.time())
                if last_skip_reason.get(key) != reason or (now_ts - last_skip_ts.get(key, 0)) > 300:
                    logger.log_skip({"ts": now_ts, "event_key": key, "reason": reason, "details": {"len": len(ladder)}})
                    last_skip_reason[key] = reason
                    last_skip_ts[key] = now_ts
                continue
            if any(top is None for top in tops.values()):
                reason = "MISSING_BOOK"
                now_ts = int(time.time())
                if last_skip_reason.get(key) != reason or (now_ts - last_skip_ts.get(key, 0)) > 300:
                    logger.log_skip({"ts": now_ts, "event_key": key, "reason": reason, "details": {}})
                    last_skip_reason[key] = reason
                    last_skip_ts[key] = now_ts
            if books_conn:
                now_ts = int(time.time())
                for ticker, top in tops.items():
                    books_conn.execute(
                        """
                        INSERT INTO l1_books (
                            ts, event_key, market_ticker, yes_bid, yes_ask, yes_bid_sz, yes_ask_sz
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            now_ts,
                            key,
                            ticker,
                            top.get("yes_bid"),
                            top.get("yes_ask"),
                            top.get("yes_bid_sz"),
                            top.get("yes_ask_sz"),
                        ),
                    )
                books_conn.commit()

            candidates = []
            for i in range(len(ladder) - 2):
                for j in range(i + 1, len(ladder) - 1):
                    for k in range(j + 1, len(ladder)):
                        legs_rows = [ladder[i], ladder[j], ladder[k]]
                        for s1 in (1, -1):
                            for s2 in (1, -1):
                                for s3 in (1, -1):
                                    legs = []
                                    mid_fallbacks = 0
                                    edge_vals = []
                                    size_vals = []
                                    spread_vals = []
                                    for s, row in zip((s1, s2, s3), legs_rows):
                                        top = tops.get(row["ticker"])
                                        if top is None:
                                            legs = []
                                            break
                                        action = "buy" if s > 0 else "sell"
                                        price, used_mid = _price_for_leg(top, action)
                                        if price is None:
                                            legs = []
                                            break
                                        if used_mid:
                                            mid_fallbacks += 1
                                        k_idx = ks.index(row["k"])
                                        ph = phat[k_idx]
                                        if implied:
                                            team_abbr = row.get("team")
                                            if team_abbr:
                                                if team_abbr == (home or ""):
                                                    ph = ph + args.win_prob_strength * (implied["home"] - 0.5)
                                                elif team_abbr == (away or ""):
                                                    ph = ph + args.win_prob_strength * (implied["away"] - 0.5)
                                                ph = max(0.0, min(1.0, ph))
                                        leg = LegSpec(
                                            ticker=row["ticker"],
                                            k=row["k"],
                                            w=float(s),
                                            limit_price=price,
                                            side="BUY_YES" if s > 0 else "SELL_YES",
                                            phat=ph,
                                            delta=ph - price,
                                            px_bid=top.get("yes_bid"),
                                            px_ask=top.get("yes_ask"),
                                            px_used=price,
                                        )
                                        legs.append(leg)
                                        edge = _edge_ticks(top, action, price)
                                        if edge is not None:
                                            edge_vals.append(edge)
                                        size_ratio = _size_ratio(top, action, 1)
                                        if size_ratio is not None:
                                            size_vals.append(size_ratio)
                                        if top.get("yes_bid") is not None and top.get("yes_ask") is not None:
                                            spread_vals.append((top["yes_ask"] - top["yes_bid"]) / 100.0)
                                    if not legs or mid_fallbacks > args.max_mids:
                                        continue
                                    if edge_vals:
                                        min_edge = min(edge_vals)
                                        if min_edge < args.min_edge_ticks:
                                            continue
                                        if args.max_edge_ticks and min_edge > args.max_edge_ticks:
                                            continue
                                    if size_vals and min(size_vals) < args.min_size_ratio:
                                        continue
                                    ev = sum(leg.w * (leg.phat - leg.limit_price) for leg in legs)
                                    if ev < args.min_ev:
                                        continue
                                    bundle_max_loss = _bundle_max_loss(legs)
                                    if args.max_bundle_loss and bundle_max_loss > args.max_bundle_loss:
                                        continue
                                    spread_pen = max(spread_vals) if spread_vals else 0.0
                                    size_pen = max(0.0, args.min_size_ratio - min(size_vals)) if size_vals else 0.0
                                    score = ev - (args.spread_penalty * spread_pen) - (args.size_penalty * size_pen)
                                    candidates.append((score, ev, bundle_max_loss, legs))

            if not candidates:
                reason = "NO_CANDIDATES"
                now_ts = int(time.time())
                if last_skip_reason.get(key) != reason or (now_ts - last_skip_ts.get(key, 0)) > 300:
                    logger.log_skip({"ts": now_ts, "event_key": key, "reason": reason, "details": {}})
                    last_skip_reason[key] = reason
                    last_skip_ts[key] = now_ts
                hb.skipped += 1
                continue
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, ev, bundle_max_loss, legs = candidates[0]
            bundle = {
                "run_id": run_id,
                "ts_signal": int(time.time()),
                "ts_decision": int(time.time()),
                "event_key": key,
                "series_ticker": "KXNBASPREAD" if args.family == "spread" else "KXNBATOTAL",
                "mode": args.mode,
                "decision": "SHADOW_ATTEMPT",
                "reasons": [],
                "mid_band_L": band[0],
                "mid_band_U": band[1],
                "max_loss": bundle_max_loss,
                "mid_worst": None,
                "ev_raw": ev,
                "ev_net_est": ev,
                "snapshot": {"phat": phat.tolist(), "ks": ks},
                "legs": [
                    {
                        "ticker": leg.ticker,
                        "k": leg.k,
                        "side": leg.side,
                        "qty": 1,
                        "limit_price": leg.limit_price,
                        "px_bid": leg.px_bid,
                        "px_ask": leg.px_ask,
                        "px_used": leg.px_used,
                        "phat": leg.phat,
                        "delta": leg.delta,
                    }
                    for leg in legs
                ],
            }
            bundle_id = logger.log_bundle(bundle)
            logger.log_bundle_legs(bundle_id, bundle["legs"])
            hb.bundles += 1
            if args.leg_in:
                result = execute_bundle_leg_in(
                    bundle,
                    book_provider,
                    max_unhedged_loss=args.max_unhedged_loss,
                    max_unhedged_ttl_s=args.max_unhedged_ttl,
                    slope_cap=args.slope_cap,
                    poll_s=args.poll_s,
                )
            else:
                result = execute_bundle(bundle, book_provider, ttl_s=args.ttl_s, poll_s=args.poll_s)
            fills = result.get("fills") or []
            if fills:
                fills_by_ticker = {f["ticker"]: f for f in fills}
                for leg in bundle["legs"]:
                    filled = fills_by_ticker.get(leg["ticker"])
                    if not filled:
                        continue
                    fill_price = filled.get("fill_price")
                    fill_qty = filled.get("fill_qty")
                    if fill_price is not None and fill_price > 1.0:
                        fill_price = float(fill_price) / 100.0
                    leg["fill_price"] = fill_price
                    leg["fill_qty"] = fill_qty
                logger.log_bundle_legs(bundle_id, bundle["legs"])
                logger.log_fills(bundle_id, bundle["event_key"], fills)
            if result["status"] == "FILLED":
                hb.filled += 1
                bundle["decision"] = "SHADOW_FILLED"
            elif result["status"] == "PARTIAL":
                hb.rejected += 1
                bundle["decision"] = "SHADOW_PARTIAL"
                bundle["reasons"] = result.get("reasons")
            else:
                hb.rejected += 1
                bundle["decision"] = "SHADOW_REJECT"
                bundle["reasons"] = result.get("reasons")
            if result.get("unhedged_max_loss") is not None or result.get("unhedged_time_s") is not None:
                bundle.setdefault("snapshot", {})
                bundle["snapshot"]["unhedged_max_loss"] = result.get("unhedged_max_loss")
                bundle["snapshot"]["unhedged_time_s"] = result.get("unhedged_time_s")
            logger.log_bundle(bundle)

        now = time.time()
        if now - last_hb >= args.heartbeat_s:
            hb.last_data_ts = time.time()
            logger.log_heartbeat(
                {
                    **hb.snapshot(),
                    "event_count": len(markets_by_event),
                    "market_count": market_count,
                }
            )
            last_hb = now

        if now - last_summary >= 300:
            logger.log_heartbeat(
                {
                    **hb.snapshot(),
                    "event_count": len(markets_by_event),
                    "market_count": market_count,
                    "summary": True,
                }
            )
            last_summary = now

        if args.loop_sleep_s > 0:
            time.sleep(args.loop_sleep_s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
