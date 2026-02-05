#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Leg:
    k: int
    w: float
    p: float
    ticker: str


def _parse_ts(value: str) -> int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return int(dt.timestamp())


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


def _load_ladder(
    csv_path: str,
    event_key: str,
    family: str,
    team: str | None,
) -> list[dict[str, Any]]:
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


def _fetch_latest_candle(conn: sqlite3.Connection, ticker: str, ts: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT ts, close, source_json
        FROM candles
        WHERE market_ticker = ? AND ts <= ?
        ORDER BY ts DESC
        LIMIT 1
        """,
        (ticker, ts),
    ).fetchone()
    if row is None:
        row = conn.execute(
            """
            SELECT ts, close, source_json
            FROM candles
            WHERE market_ticker = ?
            ORDER BY ABS(ts - ?) ASC
            LIMIT 1
            """,
            (ticker, ts),
        ).fetchone()
    if row is None:
        return None
    ts_val, close, source_json = row
    payload = {}
    try:
        payload = json.loads(source_json) if source_json else {}
    except json.JSONDecodeError:
        payload = {}
    yes_bid = payload.get("yes_bid") or {}
    yes_ask = payload.get("yes_ask") or {}
    yes_bid_close = yes_bid.get("close")
    yes_ask_close = yes_ask.get("close")
    mid = close
    if mid is None and yes_bid_close is not None and yes_ask_close is not None:
        mid = (float(yes_bid_close) + float(yes_ask_close)) / 2.0
    return {
        "ts": ts_val,
        "yes_bid_close": yes_bid_close,
        "yes_ask_close": yes_ask_close,
        "mid": mid,
    }


def _price_for_leg(candle: dict[str, Any], action: str) -> float | None:
    if action == "buy":
        if candle.get("yes_ask_close") is not None:
            return float(candle["yes_ask_close"]) / 100.0
    if action == "sell":
        if candle.get("yes_bid_close") is not None:
            return float(candle["yes_bid_close"]) / 100.0
    if candle.get("mid") is not None:
        return float(candle["mid"]) / 100.0
    return None


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
    # Pool adjacent violators for decreasing sequence
    if len(values) == 0:
        return values
    # Convert to increasing by negation
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
        # Remove i+1
        level = np.delete(level, i + 1)
        weights = np.delete(weights, i + 1)
        n -= 1
        if i > 0:
            i -= 1
    # Expand back
    expanded = np.repeat(level, weights.astype(int))
    # If weights were not integers, fall back to linear fill
    if expanded.size != values.size:
        expanded = np.interp(np.arange(values.size), np.linspace(0, values.size - 1, level.size), level)
    return -expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase C: empirical probability + EV.")
    parser.add_argument("--event-key", required=True)
    parser.add_argument("--ts", required=True, help="Unix seconds or ISO timestamp")
    parser.add_argument("--candles-db", default="data/nba_historical_candles.sqlite")
    parser.add_argument("--markets-csv", default="data/nba_historical_markets.csv")
    parser.add_argument("--results-csv", default="data/nba_results.csv")
    parser.add_argument("--family", choices=["spread", "total"], default="spread")
    parser.add_argument("--team", default="", help="Team code for spreads (e.g., POR)")
    parser.add_argument("--smooth", action="store_true", help="Apply isotonic decreasing smoothing")
    parser.add_argument("--require-bid-ask", action="store_true")
    parser.add_argument("--k-low", type=int, default=None)
    parser.add_argument("--k-high", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ts = _parse_ts(args.ts)
    team = args.team.strip().upper() or None
    ladder = _load_ladder(args.markets_csv, args.event_key, args.family, team)
    if not ladder:
        raise SystemExit("No ladder rows found for event/team")
    ladder = sorted(ladder, key=lambda r: r["k"])
    prices: dict[str, dict[str, Any]] = {}
    with sqlite3.connect(args.candles_db) as conn:
        for row in ladder:
            candle = _fetch_latest_candle(conn, row["ticker"], ts)
            if candle is None:
                continue
            if args.require_bid_ask and (candle.get("yes_bid_close") is None or candle.get("yes_ask_close") is None):
                continue
            prices[row["ticker"]] = candle

    ks = [row["k"] for row in ladder if row["ticker"] in prices]
    if not ks:
        raise SystemExit("No usable prices for ladder at this ts")
    margins = _load_margins(args.results_csv)
    phat = _p_hat(margins, ks)
    if args.smooth:
        phat = _isotonic_decreasing(phat)

    rows = []
    for k, row in zip(ks, [r for r in ladder if r["ticker"] in prices]):
        candle = prices[row["ticker"]]
        p_buy = _price_for_leg(candle, "buy")
        p_sell = _price_for_leg(candle, "sell")
        rows.append(
            {
                "k": k,
                "ticker": row["ticker"],
                "p_buy": p_buy,
                "p_sell": p_sell,
                "phat": phat[ks.index(k)],
            }
        )

    print("[phase_c] ladder_ev")
    for r in rows:
        ev_long = (r["phat"] - r["p_buy"]) if r["p_buy"] is not None else None
        ev_short = ((1 - r["phat"]) - (1 - r["p_sell"])) if r["p_sell"] is not None else None
        print(f"k={r['k']} p_buy={r['p_buy']} p_sell={r['p_sell']} P>=k={r['phat']:.4f} ev_long={ev_long} ev_short={ev_short}")

    if args.k_low is not None and args.k_high is not None:
        low = next((r for r in rows if r["k"] == args.k_low), None)
        high = next((r for r in rows if r["k"] == args.k_high), None)
        if not low or not high:
            raise SystemExit("k_low/k_high not found in ladder")
        legs = [
            Leg(k=low["k"], w=-1.0, p=float(low["p_sell"]), ticker=low["ticker"]),
            Leg(k=high["k"], w=1.0, p=float(high["p_buy"]), ticker=high["ticker"]),
        ]
        ev = sum(leg.w * (phat[ks.index(leg.k)] - leg.p) for leg in legs)
        print("[phase_c] strangle_ev", ev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
