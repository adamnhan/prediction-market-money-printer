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

    def risk_metrics(self, band: tuple[int, int]) -> dict[str, float]:
        regions = self.region_payoffs()
        vals = [v for _, v in regions]
        out = {"max_loss": float(min(vals)), "max_gain": float(max(vals))}
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


def _price_for_leg(candle: dict[str, Any], action: str) -> tuple[float | None, bool]:
    used_mid = False
    if action == "buy":
        if candle.get("yes_ask_close") is not None:
            return float(candle["yes_ask_close"]) / 100.0, used_mid
    if action == "sell":
        if candle.get("yes_bid_close") is not None:
            return float(candle["yes_bid_close"]) / 100.0, used_mid
    if candle.get("mid") is not None:
        used_mid = True
        return float(candle["mid"]) / 100.0, used_mid
    return None, used_mid


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase D: signal construction + portfolio search.")
    parser.add_argument("--event-key", required=True)
    parser.add_argument("--ts", required=True)
    parser.add_argument("--candles-db", default="data/nba_historical_candles.sqlite")
    parser.add_argument("--markets-csv", default="data/nba_historical_markets.csv")
    parser.add_argument("--results-csv", default="data/nba_results.csv")
    parser.add_argument("--family", choices=["spread", "total"], default="spread")
    parser.add_argument("--team", default="", help="Team code for spreads (e.g., POR)")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--weights", default="-2,-1,1,2")
    parser.add_argument("--band", default="6,11")
    parser.add_argument("--max-loss", type=float, default=0.10)
    parser.add_argument("--mid-loss", type=float, default=0.10)
    parser.add_argument("--min-ev", type=float, default=0.01)
    parser.add_argument("--max-mids", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ts = _parse_ts(args.ts)
    team = args.team.strip().upper() or None
    weights = [int(x.strip()) for x in args.weights.split(",") if x.strip()]
    band = tuple(int(x.strip()) for x in args.band.split(","))

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

    ladder = [row for row in ladder if row["ticker"] in prices]
    if len(ladder) < 3:
        raise SystemExit("Not enough ladder points with prices")

    ks = [row["k"] for row in ladder]
    margins = _load_margins(args.results_csv)
    phat = _p_hat(margins, ks)
    if args.smooth:
        phat = _isotonic_decreasing(phat)

    candidates = []
    for i in range(len(ladder) - 2):
        for j in range(i + 1, len(ladder) - 1):
            for k in range(j + 1, len(ladder)):
                legs_rows = [ladder[i], ladder[j], ladder[k]]
                for w1 in weights:
                    for w2 in weights:
                        for w3 in weights:
                            if (w1 == 0) or (w2 == 0) or (w3 == 0):
                                continue
                            legs = []
                            mid_fallbacks = 0
                            for w, row in zip((w1, w2, w3), legs_rows):
                                candle = prices[row["ticker"]]
                                action = "buy" if w > 0 else "sell"
                                price, used_mid = _price_for_leg(candle, action)
                                if price is None:
                                    legs = []
                                    break
                                if used_mid:
                                    mid_fallbacks += 1
                                legs.append(Leg(k=row["k"], w=float(w), p=price, ticker=row["ticker"]))
                            if not legs or mid_fallbacks > args.max_mids:
                                continue
                            portfolio = Portfolio(legs=legs)
                            metrics = portfolio.risk_metrics(band=band)
                            if metrics["max_loss"] < -args.max_loss:
                                continue
                            if metrics["mid_worst"] < -args.mid_loss:
                                continue
                            ev = 0.0
                            for leg in legs:
                                k_idx = ks.index(leg.k)
                                ev += leg.w * (phat[k_idx] - leg.p)
                            if ev < args.min_ev:
                                continue
                            candidates.append(
                                {
                                    "ev": ev,
                                    "metrics": metrics,
                                    "legs": [
                                        {
                                            "ticker": leg.ticker,
                                            "k": leg.k,
                                            "w": leg.w,
                                            "p": leg.p,
                                            "side": "BUY_YES" if leg.w > 0 else "SELL_YES",
                                        }
                                        for leg in legs
                                    ],
                                }
                            )

    candidates.sort(key=lambda x: x["ev"], reverse=True)
    top = candidates[: args.top_k]
    print(f"[phase_d] candidates={len(candidates)} top={len(top)}")
    for idx, cand in enumerate(top, start=1):
        print(f"[signal {idx}] ev={cand['ev']:.4f} max_loss={cand['metrics']['max_loss']:.4f} mid_worst={cand['metrics']['mid_worst']:.4f}")
        print(json.dumps(cand, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
