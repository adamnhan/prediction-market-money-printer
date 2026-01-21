#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

from market_maker.config import load_config


def _parse_ts(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.timestamp()


MONTHS = {
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


def _parse_tip_ts(ticker: str, tip_hour: int, tip_minute: int) -> Optional[float]:
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    token = parts[1]
    if len(token) < 7:
        return None
    try:
        day = int(token[0:2])
        mon = MONTHS.get(token[2:5].upper())
        year = int(token[5:7]) + 2000
    except ValueError:
        return None
    if mon is None:
        return None
    try:
        tip_dt = dt.datetime(year, mon, day, tip_hour, tip_minute, tzinfo=dt.timezone.utc)
    except ValueError:
        return None
    return tip_dt.timestamp()


def _time_bucket(minutes: Optional[float]) -> str:
    if minutes is None:
        return "unknown"
    if minutes < 0:
        return "post_tip"
    if minutes <= 30:
        return "0-30m"
    if minutes <= 60:
        return "30-60m"
    if minutes <= 120:
        return "60-120m"
    if minutes <= 360:
        return "2-6h"
    return "6h+"


def _spread_bucket(spread_cents: Optional[float]) -> str:
    if spread_cents is None:
        return "unknown"
    if spread_cents < 3:
        return "tight"
    if spread_cents <= 6:
        return "medium"
    return "wide"


def _vol_bucket(vol_cents_per_min: Optional[float]) -> str:
    if vol_cents_per_min is None:
        return "unknown"
    if vol_cents_per_min < 2:
        return "calm"
    if vol_cents_per_min < 5:
        return "choppy"
    return "fast"


def _inv_bucket(inv: Optional[int], itarget: float, imax: int) -> str:
    if inv is None:
        return "unknown"
    if inv == 0:
        return "zero"
    if inv >= max(0, imax - 2):
        return "near_max"
    if abs(inv - itarget) <= 2:
        return "near_target"
    return "other"


def _load_csv(path: Path) -> Iterable[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _build_snapshots(
    rows: Iterable[dict[str, str]],
    tip_hour: int,
    tip_minute: int,
    itarget: float,
    imax: int,
) -> dict[str, list[dict[str, Any]]]:
    per_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    last_mid: dict[str, Optional[float]] = {}
    last_ts: dict[str, Optional[float]] = {}

    for row in rows:
        ticker = row.get("market_ticker", "")
        ts = _parse_ts(row.get("ts_utc", ""))
        if not ticker or ts is None:
            continue
        best_bid = _safe_float(row.get("best_bid"))
        best_ask = _safe_float(row.get("best_ask"))
        mid_cents = _safe_float(row.get("mid_cents"))
        spread_cents = _safe_float(row.get("spread_cents"))
        inv = _safe_int(row.get("inventory"))
        net_pnl_cents = _safe_float(row.get("net_pnl_cents"))
        max_drawdown_cents = _safe_float(row.get("max_drawdown_cents"))
        hold_age_s = _safe_float(row.get("hold_age_s"))
        halt_reason = row.get("halt_reason") or None

        tip_ts = _parse_tip_ts(ticker, tip_hour, tip_minute)
        minutes_to_tip = None
        if tip_ts is not None:
            minutes_to_tip = (tip_ts - ts) / 60.0

        vol_per_min = None
        if mid_cents is not None:
            prev_ts = last_ts.get(ticker)
            prev_mid = last_mid.get(ticker)
            if prev_ts is not None and prev_mid is not None and ts > prev_ts:
                delta_min = (ts - prev_ts) / 60.0
                if delta_min > 0:
                    vol_per_min = abs(mid_cents - prev_mid) / delta_min
            last_ts[ticker] = ts
            last_mid[ticker] = mid_cents

        per_market[ticker].append(
            {
                "ts": ts,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_cents": mid_cents,
                "spread_cents": spread_cents,
                "inv": inv,
                "net_pnl_cents": net_pnl_cents,
                "max_drawdown_cents": max_drawdown_cents,
                "hold_age_s": hold_age_s,
                "halt_reason": halt_reason,
                "time_bucket": _time_bucket(minutes_to_tip),
                "spread_bucket": _spread_bucket(spread_cents),
                "vol_bucket": _vol_bucket(vol_per_min),
                "inv_bucket": _inv_bucket(inv, itarget, imax),
            }
        )

    for ticker in per_market:
        per_market[ticker].sort(key=lambda item: item["ts"])
    return per_market


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _find_snapshot(snapshots: list[dict[str, Any]], ts: float) -> Optional[dict[str, Any]]:
    times = [s["ts"] for s in snapshots]
    idx = bisect_right(times, ts) - 1
    if idx < 0:
        return None
    return snapshots[idx]


def _agg_add(stats: dict[str, list[float]], key: str, value: Optional[float]) -> None:
    if value is None:
        return
    stats[key].append(value)


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    sorted_vals = sorted(values)
    return {
        "count": len(values),
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "p25": round(sorted_vals[int(0.25 * (len(sorted_vals) - 1))], 4),
        "p75": round(sorted_vals[int(0.75 * (len(sorted_vals) - 1))], 4),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _quote_fill_efficiency(
    quotes: Iterable[dict[str, str]],
    snapshots: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"quotes": 0})
    for row in quotes:
        ticker = row.get("market_ticker", "")
        ts = _parse_ts(row.get("ts_utc", ""))
        if not ticker or ts is None:
            continue
        snap = _find_snapshot(snapshots.get(ticker, []), ts)
        if snap is None:
            continue
        key = snap["spread_bucket"]
        counts[key]["quotes"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 analysis for Shadow MM")
    parser.add_argument("--config", default="config/shadow_mm.yaml")
    parser.add_argument("--logs-dir", default="market_maker/logs")
    parser.add_argument("--out-dir", default="market_maker/analysis_outputs")
    parser.add_argument("--tip-hour", type=int, default=0)
    parser.add_argument("--tip-minute", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)

    market_rows = _load_csv(logs_dir / "market_summary.csv")
    fills_rows = _load_csv(logs_dir / "fills.csv")
    adverse_rows = _load_csv(logs_dir / "adverse.csv")
    round_rows = _load_csv(logs_dir / "round_trips.csv")
    quote_rows = _load_csv(logs_dir / "quotes.csv")

    snapshots = _build_snapshots(market_rows, args.tip_hour, args.tip_minute, config.itarget, config.imax)

    # Regime aggregations
    regimes = {
        "time_bucket": defaultdict(lambda: defaultdict(list)),
        "spread_bucket": defaultdict(lambda: defaultdict(list)),
        "vol_bucket": defaultdict(lambda: defaultdict(list)),
        "inv_bucket": defaultdict(lambda: defaultdict(list)),
    }

    # Fill efficiency based on quotes (by spread regime, to keep cost low)
    quote_counts = _quote_fill_efficiency(quote_rows, snapshots)

    fills_per_regime: dict[str, dict[str, int]] = defaultdict(lambda: {"fills": 0})
    for row in fills_rows:
        ticker = row.get("market_ticker", "")
        ts = _parse_ts(row.get("ts_utc", ""))
        if not ticker or ts is None:
            continue
        snap = _find_snapshot(snapshots.get(ticker, []), ts)
        if snap is None:
            continue
        for key in regimes:
            bucket = snap[key]
            fills_per_regime[bucket]["fills"] = fills_per_regime[bucket].get("fills", 0) + 1

    # Realized spread from round trips
    for row in round_rows:
        ticker = row.get("market_ticker", "")
        exit_ts = _parse_ts(row.get("exit_ts_utc", ""))
        gross_edge = _safe_float(row.get("gross_edge_cents"))
        hold_s = _safe_float(row.get("hold_s"))
        if not ticker or exit_ts is None:
            continue
        snap = _find_snapshot(snapshots.get(ticker, []), exit_ts)
        if snap is None:
            continue
        for key in regimes:
            bucket = snap[key]
            _agg_add(regimes[key][bucket], "gross_edge_cents", gross_edge)
            _agg_add(regimes[key][bucket], "hold_s", hold_s)

    # Adverse selection
    for row in adverse_rows:
        ticker = row.get("market_ticker", "")
        ts = _parse_ts(row.get("ts_utc", ""))
        adv1 = _safe_float(row.get("adverse_1s"))
        adv5 = _safe_float(row.get("adverse_5s"))
        adv30 = _safe_float(row.get("adverse_30s"))
        if not ticker or ts is None:
            continue
        snap = _find_snapshot(snapshots.get(ticker, []), ts)
        if snap is None:
            continue
        for key in regimes:
            bucket = snap[key]
            _agg_add(regimes[key][bucket], "adverse_1s", adv1)
            _agg_add(regimes[key][bucket], "adverse_5s", adv5)
            _agg_add(regimes[key][bucket], "adverse_30s", adv30)

    # Inventory health & drawdowns from market summary
    inv_regimes: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rows in snapshots.values():
        for snap in rows:
            for key in regimes:
                bucket = snap[key]
                inv_regimes[key][bucket].append(snap.get("inv") or 0)
                if snap.get("hold_age_s") is not None:
                    _agg_add(regimes[key][bucket], "hold_age_s", snap["hold_age_s"])
                if snap.get("max_drawdown_cents") is not None:
                    _agg_add(regimes[key][bucket], "max_drawdown_cents", snap["max_drawdown_cents"])
                if snap.get("halt_reason") == "stop_loss":
                    _agg_add(regimes[key][bucket], "stop_loss_hits", 1)

    # Build output tables
    for key, buckets in regimes.items():
        rows: list[dict[str, Any]] = []
        for bucket, stats in buckets.items():
            edge_stats = _summary_stats(stats.get("gross_edge_cents", []))
            adv1_stats = _summary_stats(stats.get("adverse_1s", []))
            adv5_stats = _summary_stats(stats.get("adverse_5s", []))
            adv30_stats = _summary_stats(stats.get("adverse_30s", []))
            hold_stats = _summary_stats(stats.get("hold_s", []))
            hold_age_stats = _summary_stats(stats.get("hold_age_s", []))
            drawdown_stats = _summary_stats(stats.get("max_drawdown_cents", []))

            inv_values = inv_regimes[key].get(bucket, [])
            inv_zero_pct = 0.0
            inv_max_pct = 0.0
            if inv_values:
                inv_zero_pct = sum(1 for v in inv_values if v == 0) / len(inv_values)
                inv_max_pct = sum(1 for v in inv_values if v >= max(1, config.imax - 1)) / len(inv_values)

            fills = fills_per_regime.get(bucket, {}).get("fills", 0)
            quote_count = quote_counts.get(bucket, {}).get("quotes", 0)
            fill_rate = fills / quote_count if quote_count else 0.0

            rows.append(
                {
                    key: bucket,
                    "fills": fills,
                    "quotes": quote_count,
                    "fill_rate": round(fill_rate, 4),
                    "gross_edge_median": edge_stats.get("median"),
                    "gross_edge_p25": edge_stats.get("p25"),
                    "gross_edge_p75": edge_stats.get("p75"),
                    "adverse_1s_mean": adv1_stats.get("mean"),
                    "adverse_5s_mean": adv5_stats.get("mean"),
                    "adverse_30s_mean": adv30_stats.get("mean"),
                    "hold_s_median": hold_stats.get("median"),
                    "hold_age_s_median": hold_age_stats.get("median"),
                    "inv_zero_pct": round(inv_zero_pct, 4),
                    "inv_max_pct": round(inv_max_pct, 4),
                    "max_drawdown_mean": drawdown_stats.get("mean"),
                    "stop_loss_hits": int(sum(stats.get("stop_loss_hits", []))) if stats.get("stop_loss_hits") else 0,
                }
            )
        _write_csv(out_dir / f"regime_{key}.csv", rows)

    # Market ranking
    market_rank_rows: list[dict[str, Any]] = []
    for ticker, rows in snapshots.items():
        if not rows:
            continue
        start_ts = rows[0]["ts"]
        end_ts = rows[-1]["ts"]
        hours = max(1e-6, (end_ts - start_ts) / 3600.0)
        net_pnl = rows[-1].get("net_pnl_cents") or 0.0
        max_dd = max((r.get("max_drawdown_cents") or 0.0) for r in rows)
        adverse_vals = []
        for row in adverse_rows:
            if row.get("market_ticker") == ticker:
                adv = _safe_float(row.get("adverse_5s"))
                if adv is not None:
                    adverse_vals.append(adv)
        adverse_mean = round(mean(adverse_vals), 4) if adverse_vals else None
        market_rank_rows.append(
            {
                "market_ticker": ticker,
                "net_pnl_per_hour": round(net_pnl / hours, 4),
                "net_pnl_cents": round(net_pnl, 2),
                "max_drawdown_cents": round(max_dd, 2),
                "drawdown_per_pnl": round(max_dd / max(1.0, abs(net_pnl)), 4),
                "adverse_5s_mean": adverse_mean,
            }
        )
    market_rank_rows.sort(key=lambda r: r["net_pnl_per_hour"], reverse=True)
    _write_csv(out_dir / "market_ranking.csv", market_rank_rows)

    print(f"Wrote regime tables and rankings to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
