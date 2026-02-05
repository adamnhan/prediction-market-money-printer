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
    w: float  # + long YES, - short YES
    p: float  # entry price in dollars (0..1)
    ticker: str


@dataclass
class Portfolio:
    legs: list[Leg]

    def payoff(self, m: np.ndarray) -> np.ndarray:
        out = np.zeros_like(m, dtype=float)
        for leg in self.legs:
            out += leg.w * ((m >= leg.k).astype(float) - leg.p)
        return out

    def region_payoffs(self) -> list[tuple[tuple[int, int], float]]:
        ks = sorted(set(leg.k for leg in self.legs))
        bounds = [-10**9] + ks + [10**9]
        const_cost = sum(leg.w * (-leg.p) for leg in self.legs)
        regions = []
        for j in range(len(bounds) - 1):
            lo, hi = bounds[j], bounds[j + 1]
            rep = lo if lo > -10**8 else (hi - 1)
            ind_term = 0.0
            for leg in self.legs:
                ind_term += leg.w * (1.0 if rep >= leg.k else 0.0)
            regions.append(((lo, hi), ind_term + const_cost))
        return regions

    def risk_metrics(self, band: tuple[int, int] | None = None) -> dict[str, float]:
        regions = self.region_payoffs()
        vals = [v for _, v in regions]
        out = {"max_loss": float(min(vals)), "max_gain": float(max(vals))}
        if band is not None:
            L, U = band
            ms = np.arange(L, U)
            if len(ms) > 0:
                pv = self.payoff(ms)
                out["mid_worst"] = float(pv.min())
                out["mid_avg"] = float(pv.mean())
        return out


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


def _build_strangle(
    ladder: list[dict[str, Any]],
    prices: dict[str, dict[str, Any]],
    k_low: int,
    k_high: int,
) -> Portfolio:
    low = next((row for row in ladder if row["k"] == k_low), None)
    high = next((row for row in ladder if row["k"] == k_high), None)
    if not low or not high:
        raise RuntimeError("k_low/k_high not found in ladder")
    low_candle = prices[low["ticker"]]
    high_candle = prices[high["ticker"]]
    p_sell = _price_for_leg(low_candle, "sell")
    p_buy = _price_for_leg(high_candle, "buy")
    if p_sell is None or p_buy is None:
        raise RuntimeError("missing executable price for legs")
    legs = [
        Leg(k=k_low, w=-1.0, p=p_sell, ticker=low["ticker"]),
        Leg(k=k_high, w=1.0, p=p_buy, ticker=high["ticker"]),
    ]
    return Portfolio(legs=legs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase B payoff math + risk diagnostics.")
    parser.add_argument("--event-key", required=True)
    parser.add_argument("--ts", required=True, help="Unix seconds or ISO timestamp")
    parser.add_argument("--candles-db", default="data/nba_historical_candles.sqlite")
    parser.add_argument("--markets-csv", default="data/nba_historical_markets.csv")
    parser.add_argument("--family", choices=["spread", "total"], default="spread")
    parser.add_argument("--team", default="", help="Team code for spreads (e.g., POR)")
    parser.add_argument("--k-low", type=int, default=None)
    parser.add_argument("--k-high", type=int, default=None)
    parser.add_argument("--band", default="", help="L,U band for middle loss, e.g. 6,11")
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
            prices[row["ticker"]] = candle
    if args.k_low is not None and args.k_high is not None:
        portfolio = _build_strangle(ladder, prices, args.k_low, args.k_high)
        band = None
        if args.band:
            L, U = (int(x.strip()) for x in args.band.split(","))
            band = (L, U)
        metrics = portfolio.risk_metrics(band=band)
        print("[portfolio] legs")
        for leg in portfolio.legs:
            print(f"  k={leg.k} w={leg.w} p={leg.p:.4f} ticker={leg.ticker}")
        print("[portfolio] metrics", metrics)
        ms = np.arange(-60, 61)
        pv = portfolio.payoff(ms)
        print("[portfolio] payoff sample", list(zip(ms[:10], pv[:10])))
    else:
        print("[ladder] snapshot")
        for row in ladder:
            candle = prices.get(row["ticker"]) or {}
            yes_bid = candle.get("yes_bid_close")
            yes_ask = candle.get("yes_ask_close")
            mid = candle.get("mid")
            print(f"k={row['k']} ticker={row['ticker']} yes_bid={yes_bid} yes_ask={yes_ask} mid={mid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
