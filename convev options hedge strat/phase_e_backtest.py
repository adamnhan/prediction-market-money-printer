#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

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

    def risk_metrics(self, band: tuple[int, int]) -> dict[str, float]:
        ks = sorted(set(leg.k for leg in self.legs))
        bounds = [-10**9] + ks + [10**9]
        const_cost = sum(leg.w * (-leg.p) for leg in self.legs)
        vals = []
        for j in range(len(bounds) - 1):
            lo, hi = bounds[j], bounds[j + 1]
            rep = lo if lo > -10**8 else (hi - 1)
            ind_term = 0.0
            for leg in self.legs:
                ind_term += leg.w * (1.0 if rep >= leg.k else 0.0)
            vals.append(ind_term + const_cost)
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


def _parse_event_key(event_key: str) -> tuple[str | None, str | None]:
    if len(event_key) < 13:
        return None, None
    teams = event_key[7:]
    if len(teams) != 6:
        return None, None
    away = teams[:3]
    home = teams[3:]
    return away, home


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
        "volume": payload.get("volume"),
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


def _load_margins(results_csv: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    margins = []
    rows = []
    with open(results_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            val = row.get("final_margin")
            if val is None or val == "":
                continue
            try:
                margin = int(val)
            except ValueError:
                continue
            rows.append(row)
            margins.append(margin)
    if not margins:
        raise RuntimeError("no margins found")
    return np.array(margins, dtype=int), rows


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


def _margin_for_team(event_key: str, home_score: int, away_score: int, team: str) -> int:
    away, home = _parse_event_key(event_key)
    if team == home:
        return home_score - away_score
    if team == away:
        return away_score - home_score
    return home_score - away_score


def _total_points(home_score: int, away_score: int) -> int:
    return home_score + away_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase E: settle-only backtest using candle proxies.")
    parser.add_argument("--events-csv", default="data/nba_historical_markets.csv")
    parser.add_argument("--candles-db", default="data/nba_historical_candles.sqlite")
    parser.add_argument("--results-csv", default="data/nba_results.csv")
    parser.add_argument("--family", choices=["spread", "total"], default="spread")
    parser.add_argument("--team", default="", help="Team code for spreads (leave empty for all)")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--weights", default="-2,-1,1,2")
    parser.add_argument("--band", default="6,11")
    parser.add_argument("--max-loss", type=float, default=0.10)
    parser.add_argument("--mid-loss", type=float, default=0.10)
    parser.add_argument("--min-ev", type=float, default=0.01)
    parser.add_argument("--max-mids", type=int, default=1)
    parser.add_argument("--step-minutes", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = [int(x.strip()) for x in args.weights.split(",") if x.strip()]
    band = tuple(int(x.strip()) for x in args.band.split(","))
    team = args.team.strip().upper() or None

    margins_all, result_rows = _load_margins(args.results_csv)
    phat_cache: dict[tuple[int, ...], np.ndarray] = {}

    results_by_event: dict[str, dict[str, Any]] = {}
    for row in result_rows:
        results_by_event[row["event_key"]] = row

    trades = []
    skipped = 0
    total_signals = 0

    with sqlite3.connect(args.candles_db) as conn:
        for event_key in results_by_event:
            ladder_all = _load_ladder(args.events_csv, event_key, args.family, None if args.family == "spread" else team)
            if not ladder_all:
                continue
            ladders_by_team = {}
            if args.family == "spread":
                for row in ladder_all:
                    ladders_by_team.setdefault(row["team"], []).append(row)
            else:
                ladders_by_team = {"TOTAL": ladder_all}

            for team_code, ladder in ladders_by_team.items():
                if team and args.family == "spread" and team_code != team:
                    continue
                ladder = sorted(ladder, key=lambda r: r["k"])
                tickers = [row["ticker"] for row in ladder]
                qs = ",".join("?" * len(tickers))
                rows = conn.execute(
                    f"SELECT DISTINCT ts FROM candles WHERE market_ticker IN ({qs}) ORDER BY ts",
                    tickers,
                ).fetchall()
                ts_list = [r[0] for r in rows][:: max(1, int(args.step_minutes))]
                if not ts_list:
                    continue
                ks = [row["k"] for row in ladder]
                ks_key = tuple(ks)
                if ks_key not in phat_cache:
                    phat = _p_hat(margins_all, ks)
                    if args.smooth:
                        phat = _isotonic_decreasing(phat)
                    phat_cache[ks_key] = phat
                phat = phat_cache[ks_key]

                for ts in ts_list:
                    total_signals += 1
                    prices: dict[str, dict[str, Any]] = {}
                    for row in ladder:
                        candle = _fetch_latest_candle(conn, row["ticker"], ts)
                        if candle is None:
                            continue
                        prices[row["ticker"]] = candle
                    ladder_use = [row for row in ladder if row["ticker"] in prices]
                    if len(ladder_use) < 3:
                        skipped += 1
                        continue

                candidates = []
                for i in range(len(ladder_use) - 2):
                    for j in range(i + 1, len(ladder_use) - 1):
                        for k in range(j + 1, len(ladder_use)):
                            legs_rows = [ladder_use[i], ladder_use[j], ladder_use[k]]
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
                                        candidates.append((ev, legs, metrics))
                    if not candidates:
                        skipped += 1
                        continue
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    for ev, legs, metrics in candidates[: args.top_k]:
                        trades.append(
                            {
                                "event_key": event_key,
                                "team_code": team_code,
                                "ts": ts,
                                "ev": ev,
                                "legs": legs,
                                "metrics": metrics,
                            }
                        )

    # Settle
    pnl = []
    for trade in trades:
        event_key = trade["event_key"]
        row = results_by_event.get(event_key)
        if not row:
            continue
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        if args.family == "spread":
            team_code = trade.get("team_code") or team
            if not team_code:
                team_code = trade["legs"][0].ticker.split("-")[-1][:3]
            m = _margin_for_team(event_key, home_score, away_score, team_code)
        else:
            m = _total_points(home_score, away_score)
        portfolio = Portfolio(legs=trade["legs"])
        trade_pnl = float(portfolio.payoff(np.array([m], dtype=int))[0])
        pnl.append(trade_pnl)

    if pnl:
        pnl_arr = np.array(pnl)
        print(f"[phase_e] trades={len(pnl)} avg_pnl={pnl_arr.mean():.4f} win_rate={(pnl_arr>0).mean():.3f} max_drawdown={pnl_arr.min():.4f}")
    print(f"[phase_e] total_signals={total_signals} executed={len(pnl)} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
