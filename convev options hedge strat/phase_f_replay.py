#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sqlite3

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nba_engine.phase5 import _extract_orderbook_levels, _derive_best_prices


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


def _load_win_markets(csv_path: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            series = row.get("series_ticker")
            if series != "KXNBAGAME":
                continue
            event_key = row.get("event_key") or ""
            ticker = row.get("market_ticker") or ""
            if not event_key or not ticker:
                continue
            parts = ticker.split("-")
            if len(parts) < 3:
                continue
            team_abbr = parts[-1]
            out.setdefault(event_key, {})[team_abbr] = ticker
    return out


def _load_win_markets_from_books(conn: sqlite3.Connection, event_key: str) -> dict[str, str]:
    rows = conn.execute(
        """
        SELECT DISTINCT market_ticker
        FROM l1_books
        WHERE event_key = ?
        """,
        (event_key,),
    ).fetchall()
    out: dict[str, str] = {}
    for (ticker,) in rows:
        if not ticker:
            continue
        parts = ticker.split("-")
        if len(parts) < 3:
            continue
        series = parts[0]
        if series != "KXNBAGAME":
            continue
        team_abbr = parts[-1]
        out[team_abbr] = ticker
    return out


def _load_ladder_from_books(
    conn: sqlite3.Connection, event_key: str, family: str, team: str | None
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT DISTINCT market_ticker
        FROM l1_books
        WHERE event_key = ?
        """,
        (event_key,),
    ).fetchall()
    out = []
    for (ticker,) in rows:
        if not ticker:
            continue
        series = ticker.split("-")[0]
        if family == "spread" and series == "KXNBASPREAD":
            t, k = _parse_spread_threshold(ticker)
            if t and k is not None and (team is None or t == team):
                out.append({"k": k, "ticker": ticker, "team": t})
        elif family == "total" and series == "KXNBATOTAL":
            k = _parse_total_threshold(ticker)
            if k is not None:
                out.append({"k": k, "ticker": ticker, "team": "TOTAL"})
    return out


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


def _top_of_book_from_row(row: tuple[int, int | None, int | None, int | None, int | None]) -> dict[str, Any]:
    _, yes_bid, yes_ask, yes_bid_sz, yes_ask_sz = row
    return {
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "yes_bid_sz": yes_bid_sz,
        "yes_ask_sz": yes_ask_sz,
    }


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


def _portfolio_max_loss(legs: list[dict[str, Any]]) -> float:
    if not legs:
        return 0.0
    ks = sorted(set(int(leg["k"]) for leg in legs))
    bounds = [-10**9] + ks + [10**9]
    const_cost = 0.0
    for leg in legs:
        w = leg["w"]
        p = leg["p"]
        const_cost += w * (-p)
    vals = []
    for j in range(len(bounds) - 1):
        lo, hi = bounds[j], bounds[j + 1]
        rep = lo if lo > -10**8 else (hi - 1)
        ind_term = 0.0
        for leg in legs:
            ind_term += leg["w"] * (1.0 if rep >= leg["k"] else 0.0)
        vals.append(ind_term + const_cost)
    return float(min(vals)) if vals else 0.0


def _net_slope(legs: list[dict[str, Any]]) -> float:
    return float(sum(leg["w"] for leg in legs))


def _parse_event_teams(event_key: str) -> tuple[str | None, str | None]:
    if len(event_key) < 13:
        return None, None
    teams = event_key[7:]
    if len(teams) != 6:
        return None, None
    away = teams[:3]
    home = teams[3:]
    return away, home


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Phase F selection against recorded L1 books.")
    parser.add_argument("--ledger-db", default="convev options hedge strat/phase_f_ledger.sqlite")
    parser.add_argument("--books-db", default="convev options hedge strat/phase_f_books.sqlite")
    parser.add_argument("--markets-csv", default="data/nba_historical_markets.csv")
    parser.add_argument("--results-csv", default="data/nba_results.csv")
    parser.add_argument("--out-jsonl", default="convev options hedge strat/phase_f_replay_feb4.jsonl")
    parser.add_argument("--event-prefix", default="26FEB04")
    parser.add_argument("--family", choices=["spread", "total"], default="spread")
    parser.add_argument("--team", default="")
    parser.add_argument("--min-ev", type=float, default=0.0)
    parser.add_argument("--min-edge-ticks", type=int, default=0)
    parser.add_argument("--max-edge-ticks", type=int, default=2)
    parser.add_argument("--min-size-ratio", type=float, default=1.0)
    parser.add_argument("--max-mids", type=int, default=1)
    parser.add_argument("--spread-penalty", type=float, default=0.0)
    parser.add_argument("--size-penalty", type=float, default=0.0)
    parser.add_argument("--max-bundle-loss", type=float, default=1.0)
    parser.add_argument("--use-win-prob", action="store_true")
    parser.add_argument("--win-prob-strength", type=float, default=0.5)
    parser.add_argument("--leg-in", action="store_true")
    parser.add_argument("--max-unhedged-loss", type=float, default=1.3)
    parser.add_argument("--max-unhedged-ttl", type=float, default=15.0)
    parser.add_argument("--slope-cap", type=float, default=2.0)
    parser.add_argument("--poll-s", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    team = args.team.strip().upper() or None
    run_id = uuid.uuid4().hex
    margins = _load_margins(args.results_csv)
    win_markets = _load_win_markets(args.markets_csv)

    ledger = sqlite3.connect(args.ledger_db)
    books = sqlite3.connect(args.books_db)

    event_keys = sorted(
        {
            r[0]
            for r in ledger.execute(
                "SELECT DISTINCT event_key FROM bundles WHERE event_key LIKE ?",
                (f"{args.event_prefix}%",),
            ).fetchall()
        }
    )
    if not event_keys:
        print(f"[replay] no event_keys matched prefix={args.event_prefix}")
        return 1

    ts_list = sorted(
        {
            int(r[0])
            for r in ledger.execute(
                "SELECT DISTINCT ts_decision FROM bundles WHERE event_key LIKE ?",
                (f"{args.event_prefix}%",),
            ).fetchall()
            if r[0]
        }
    )
    if not ts_list:
        print(f"[replay] no ts_decision found for prefix={args.event_prefix}")
        return 1

    # Cache orderbooks per ticker
    def _load_book_rows(ticker: str) -> list[tuple[int, int | None, int | None, int | None, int | None]]:
        rows = books.execute(
            """
            SELECT ts, yes_bid, yes_ask, yes_bid_sz, yes_ask_sz
            FROM l1_books
            WHERE market_ticker = ?
            ORDER BY ts
            """,
            (ticker,),
        ).fetchall()
        return rows

    book_cache: dict[str, list[tuple[int, int | None, int | None, int | None, int | None]]] = {}

    def _book_at(ticker: str, ts: int) -> dict[str, Any] | None:
        rows = book_cache.get(ticker)
        if rows is None:
            rows = _load_book_rows(ticker)
            book_cache[ticker] = rows
        if not rows:
            return None
        # find latest row with ts <= target
        lo, hi = 0, len(rows) - 1
        best_idx = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if rows[mid][0] <= ts:
                best_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best_idx is None:
            return None
        return _top_of_book_from_row(rows[best_idx])

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = out_path.open("w")

    total = 0
    filled = 0
    rejected = 0
    processed = 0
    last_log = time.time()
    log_every_s = 5.0

    for ts in ts_list:
        if time.time() - last_log >= log_every_s:
            print(f"[replay] progress ts={ts} processed={processed} bundles={total} filled={filled} rejected={rejected}")
            last_log = time.time()
        for event_key in event_keys:
            ladder = _load_ladder(args.markets_csv, event_key, args.family, team)
            if not ladder:
                ladder = _load_ladder_from_books(books, event_key, args.family, team)
            if not ladder:
                continue
            ladder = sorted(ladder, key=lambda r: r["k"])
            if len(ladder) < 3:
                continue
            ks = [row["k"] for row in ladder]
            phat = _isotonic_decreasing(_p_hat(margins, ks))

            implied = {}
            if args.use_win_prob and args.family == "spread":
                away, home = _parse_event_teams(event_key)
                win_for_event = win_markets.get(event_key) or _load_win_markets_from_books(books, event_key)
                p_home = None
                p_away = None
                if home and home in win_for_event:
                    top = _book_at(win_for_event[home], ts)
                    p_home = _mid_price(top) if top else None
                if away and away in win_for_event:
                    top = _book_at(win_for_event[away], ts)
                    p_away = _mid_price(top) if top else None
                if p_home is not None and p_away is None:
                    p_away = 1.0 - p_home
                if p_away is not None and p_home is None:
                    p_home = 1.0 - p_away
                if p_home is not None and p_away is not None:
                    implied = {"home": p_home, "away": p_away}

            tops = {}
            missing_book = False
            for row in ladder:
                top = _book_at(row["ticker"], ts)
                if top is None:
                    missing_book = True
                    break
                tops[row["ticker"]] = top
            if missing_book:
                continue

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
                continue

            candidates.sort(key=lambda x: x[0], reverse=True)
            _, ev, bundle_max_loss, legs = candidates[0]

            bundle = {
                "run_id": run_id,
                "ts_signal": ts,
                "ts_decision": ts,
                "event_key": event_key,
                "series_ticker": "KXNBASPREAD" if args.family == "spread" else "KXNBATOTAL",
                "decision": "SHADOW_ATTEMPT",
                "reasons": [],
                "max_loss": bundle_max_loss,
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

            total += 1

            if args.leg_in:
                result = execute_leg_in_offline(
                    bundle,
                    _book_at,
                    ts,
                    max_unhedged_loss=args.max_unhedged_loss,
                    max_unhedged_ttl_s=args.max_unhedged_ttl,
                    slope_cap=args.slope_cap,
                    poll_s=args.poll_s,
                )
            else:
                result = execute_atomic_offline(
                    bundle,
                    _book_at,
                    ts,
                    ttl_s=args.max_unhedged_ttl,
                    poll_s=args.poll_s,
                )

            fills = result.get("fills") or []
            if fills:
                fills_by_ticker = {f["ticker"]: f for f in fills}
                for leg in bundle["legs"]:
                    filled_leg = fills_by_ticker.get(leg["ticker"])
                    if not filled_leg:
                        continue
                    leg["fill_price"] = filled_leg.get("fill_price")
                    leg["fill_qty"] = filled_leg.get("fill_qty")
                    leg["fill_ts"] = filled_leg.get("fill_ts")

            status = result.get("status")
            if status == "FILLED":
                filled += 1
                bundle["decision"] = "SHADOW_FILLED"
            elif status == "PARTIAL":
                rejected += 1
                bundle["decision"] = "SHADOW_PARTIAL"
                bundle["reasons"] = result.get("reasons")
            else:
                rejected += 1
                bundle["decision"] = "SHADOW_REJECT"
                bundle["reasons"] = result.get("reasons")

            if result.get("unhedged_max_loss") is not None or result.get("unhedged_time_s") is not None:
                bundle.setdefault("snapshot", {})
                bundle["snapshot"]["unhedged_max_loss"] = result.get("unhedged_max_loss")
                bundle["snapshot"]["unhedged_time_s"] = result.get("unhedged_time_s")

            out.write(json.dumps(bundle) + "\n")
            processed += 1

    out.close()
    print(f"[replay] bundles={total} filled={filled} rejected={rejected}")
    print(f"[replay] wrote {out_path}")
    return 0


def execute_atomic_offline(
    bundle: dict[str, Any],
    book_at: callable,
    start_ts: int,
    ttl_s: float,
    poll_s: float,
) -> dict[str, Any]:
    legs = bundle.get("legs") or []
    tickers = [leg["ticker"] for leg in legs]
    t = start_ts
    end_ts = start_ts + int(ttl_s)
    last_reasons: list[str] = []
    while t <= end_ts:
        all_ok = True
        last_reasons = []
        fills = []
        for leg in legs:
            ticker = leg["ticker"]
            side = leg["side"]
            qty = int(leg.get("qty") or 1)
            limit_price = int(round(float(leg.get("limit_price")) * 100))
            top = book_at(ticker, t)
            if top is None:
                all_ok = False
                last_reasons.append(f"missing_book:{ticker}")
                continue
            if side == "BUY_YES":
                ask = top.get("yes_ask")
                ask_sz = top.get("yes_ask_sz") or 0
                if ask is None or ask > limit_price or ask_sz < qty:
                    all_ok = False
                    last_reasons.append(f"buy_not_fill:{ticker}")
                    continue
                fills.append({"ticker": ticker, "fill_price": float(ask) / 100.0, "fill_qty": qty, "fill_ts": t})
            elif side == "SELL_YES":
                bid = top.get("yes_bid")
                bid_sz = top.get("yes_bid_sz") or 0
                if bid is None or bid < limit_price or bid_sz < qty:
                    all_ok = False
                    last_reasons.append(f"sell_not_fill:{ticker}")
                    continue
                fills.append({"ticker": ticker, "fill_price": float(bid) / 100.0, "fill_qty": qty, "fill_ts": t})
            else:
                all_ok = False
                last_reasons.append(f"invalid_side:{ticker}")
        if all_ok and fills:
            return {"status": "FILLED", "reasons": [], "fills": fills}
        t += int(max(1, round(poll_s)))
    return {"status": "REJECTED", "reasons": last_reasons or ["TTL_EXPIRED"], "fills": []}


def execute_leg_in_offline(
    bundle: dict[str, Any],
    book_at: callable,
    start_ts: int,
    max_unhedged_loss: float,
    max_unhedged_ttl_s: float,
    slope_cap: float,
    poll_s: float = 0.2,
) -> dict[str, Any]:
    legs = bundle.get("legs") or []
    normalized = []
    for leg in legs:
        qty = int(leg.get("qty") or 1)
        side = leg.get("side")
        w = qty if side == "BUY_YES" else -qty
        normalized.append(
            {
                "ticker": leg["ticker"],
                "k": int(leg["k"]),
                "w": float(w),
                "p": float(leg["limit_price"]),
                "delta": float(leg.get("delta") or 0.0),
                "side": side,
                "qty": qty,
            }
        )

    remaining = normalized[:]
    filled: list[dict[str, Any]] = []
    t = start_ts
    end_ts = start_ts + int(max_unhedged_ttl_s)
    first_fill_ts: int | None = None
    max_loss_seen: float | None = None

    while remaining and t <= end_ts:
        current_max_loss = _portfolio_max_loss(filled)
        best = None
        best_improve = None
        for leg in remaining:
            trial = filled + [leg]
            trial_loss = _portfolio_max_loss(trial)
            improve = trial_loss - current_max_loss
            if best is None or improve > best_improve or (improve == best_improve and abs(leg["delta"]) > abs(best["delta"])):
                best = leg
                best_improve = improve
        if best is None:
            break

        trial = filled + [best]
        if _portfolio_max_loss(trial) < -max_unhedged_loss:
            return {
                "status": "REJECTED",
                "reasons": ["MAX_UNHEDGED_LOSS"],
                "fills": filled,
                "unhedged_max_loss": max_loss_seen,
            }
        if abs(_net_slope(trial)) > slope_cap:
            return {
                "status": "REJECTED",
                "reasons": ["SLOPE_CAP"],
                "fills": filled,
                "unhedged_max_loss": max_loss_seen,
            }

        top = book_at(best["ticker"], t)
        if top is None:
            t += int(max(1, round(poll_s)))
            continue
        limit_price = int(round(best["p"] * 100))
        if best["side"] == "BUY_YES":
            ask = top.get("yes_ask")
            ask_sz = top.get("yes_ask_sz") or 0
            if ask is None or ask > limit_price or ask_sz < best["qty"]:
                t += int(max(1, round(poll_s)))
                continue
            best["fill_price"] = float(ask) / 100.0
            best["fill_qty"] = best["qty"]
            best["fill_ts"] = t
            filled.append(best)
            if first_fill_ts is None:
                first_fill_ts = t
            max_loss_seen = current_max_loss if max_loss_seen is None else min(max_loss_seen, current_max_loss)
            remaining = [leg for leg in remaining if leg is not best]
        elif best["side"] == "SELL_YES":
            bid = top.get("yes_bid")
            bid_sz = top.get("yes_bid_sz") or 0
            if bid is None or bid < limit_price or bid_sz < best["qty"]:
                t += int(max(1, round(poll_s)))
                continue
            best["fill_price"] = float(bid) / 100.0
            best["fill_qty"] = best["qty"]
            best["fill_ts"] = t
            filled.append(best)
            if first_fill_ts is None:
                first_fill_ts = t
            max_loss_seen = current_max_loss if max_loss_seen is None else min(max_loss_seen, current_max_loss)
            remaining = [leg for leg in remaining if leg is not best]
        else:
            return {
                "status": "REJECTED",
                "reasons": ["invalid_side"],
                "fills": filled,
                "unhedged_max_loss": max_loss_seen,
            }

        t += int(max(1, round(poll_s)))

    if remaining:
        unhedged_time = 0.0
        if first_fill_ts is not None:
            unhedged_time = max(0.0, float(t - first_fill_ts))
        return {
            "status": "PARTIAL",
            "reasons": ["PARTIAL_TIMEOUT"],
            "fills": filled,
            "unhedged_time_s": unhedged_time,
            "unhedged_max_loss": max_loss_seen,
        }
    return {
        "status": "FILLED",
        "reasons": [],
        "fills": filled,
        "unhedged_time_s": 0.0 if first_fill_ts is None else float(t - first_fill_ts),
        "unhedged_max_loss": max_loss_seen,
    }


if __name__ == "__main__":
    raise SystemExit(main())
