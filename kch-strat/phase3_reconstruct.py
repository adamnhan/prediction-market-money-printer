#!/usr/bin/env python3
"""Phase 3: Reconstruct per-event position trajectories for NHL moneyline markets."""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _safe_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def _parse_outcomes(val: Any) -> List[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return [str(v) for v in val if v is not None]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v is not None]
            if isinstance(parsed, str):
                try:
                    parsed2 = json.loads(parsed)
                    if isinstance(parsed2, list):
                        return [str(v) for v in parsed2 if v is not None]
                except Exception:
                    return []
        except Exception:
            return []
    return []


def _price_cents(row: pd.Series) -> Optional[float]:
    val = row.get("price_cents")
    if val is not None and not (isinstance(val, float) and pd.isna(val)):
        try:
            return float(val)
        except Exception:
            return None
    price = row.get("price")
    if price is None or (isinstance(price, float) and pd.isna(price)):
        return None
    try:
        price = float(price)
    except Exception:
        return None
    if price <= 1.0:
        return price * 100.0
    return price


def _extract_raw_outcome(raw_json: Any) -> Optional[str]:
    if not isinstance(raw_json, str) or not raw_json:
        return None
    try:
        payload = json.loads(raw_json)
    except Exception:
        return None
    return _safe_str(payload.get("outcome"))


def _normalize_side(outcome: Optional[str], outcomes: List[str]) -> Optional[str]:
    if not outcome or len(outcomes) < 2:
        return None
    outcome_l = outcome.lower()
    a = outcomes[0].lower()
    b = outcomes[1].lower()
    if outcome_l == a:
        return "SIDE_A"
    if outcome_l == b:
        return "SIDE_B"
    return None


def _band_label(price: Optional[float], edges: List[float]) -> Optional[str]:
    if price is None:
        return None
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if lo <= price < hi:
            return f"{int(lo)}-{int(hi)}"
    return f"{int(edges[-2])}-{int(edges[-1])}"


def _vwap(cum_notional: float, cum_size: float) -> Optional[float]:
    if cum_size <= 0:
        return None
    return cum_notional / cum_size


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3: reconstruct per-event positions")
    parser.add_argument("--phase2_path", default="./phase2_out/trades_enriched.parquet")
    parser.add_argument("--outdir", default="./phase3_out")
    parser.add_argument("--cheap_band_edges", default="0,10,20,30,40,50")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    edges = [float(x) for x in args.cheap_band_edges.split(",") if x.strip() != ""]
    if len(edges) < 2:
        raise SystemExit("cheap_band_edges must have at least two values")

    df = pd.read_parquet(args.phase2_path)

    # Filter NHL winner markets (exclude spreads/totals)
    df = df[
        df["event_slug"].astype(str).str.startswith("nhl-")
        & ~df["slug"].astype(str).str.contains("spread|total", case=False, na=False)
    ].copy()
    if df.empty:
        raise SystemExit("no rows after filtering NHL winner markets")

    df["event_slug"] = df["event_slug"].apply(_safe_str)
    df["slug"] = df["slug"].apply(_safe_str)
    df = df[df["event_slug"].notna() & df["slug"].notna()].copy()
    df["market_key"] = df["event_slug"] + "::" + df["slug"]

    df["outcome_raw"] = df["outcome"].apply(_safe_str)
    df["outcome_raw"] = df["outcome_raw"].fillna(df["raw_json"].apply(_extract_raw_outcome))
    df["price_cents_norm"] = df.apply(_price_cents, axis=1)

    # Build per-market outcomes lookup
    outcomes_by_market: Dict[str, List[str]] = {}
    for market_key, group in df.groupby("market_key"):
        outcomes = []
        for val in group["outcomes"]:
            outcomes = _parse_outcomes(val)
            if outcomes:
                break
        outcomes_by_market[market_key] = outcomes

    # Normalize side
    def _side_norm(row: pd.Series) -> Optional[str]:
        outcomes = outcomes_by_market.get(row["market_key"], [])
        return _normalize_side(row.get("outcome_raw"), outcomes)

    df["side_norm"] = df.apply(_side_norm, axis=1)
    df = df[df["side_norm"].notna()].copy()

    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["market_key", "timestamp_dt"]).reset_index(drop=True)
    df["trade_seq"] = df.groupby("market_key").cumcount() + 1

    timeseries_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for market_key, group in df.groupby("market_key"):
        group = group.sort_values("timestamp_dt")

        cum_size = {"SIDE_A": 0.0, "SIDE_B": 0.0}
        cum_notional = {"SIDE_A": 0.0, "SIDE_B": 0.0}
        first_price = {"SIDE_A": None, "SIDE_B": None}
        first_ts = {"SIDE_A": None, "SIDE_B": None}
        first_seq = {"SIDE_A": None, "SIDE_B": None}

        for _, row in group.iterrows():
            side = row["side_norm"]
            size = float(row["size"]) if row.get("size") is not None else 0.0
            price = row["price_cents_norm"]
            if price is None:
                price = 0.0
            if first_price[side] is None and price:
                first_price[side] = price
                first_ts[side] = row["timestamp_dt"]
                first_seq[side] = int(row["trade_seq"])

            cum_size[side] += size
            cum_notional[side] += size * float(price)

            timeseries_rows.append(
                {
                    "market_key": market_key,
                    "event_slug": group["event_slug"].iloc[0],
                    "slug": group["slug"].iloc[0],
                    "timestamp": row["timestamp_dt"],
                    "trade_seq": int(row["trade_seq"]),
                    "side_norm": side,
                    "size": size,
                    "price_cents": price,
                    "cum_size_side_a": cum_size["SIDE_A"],
                    "cum_size_side_b": cum_size["SIDE_B"],
                    "cum_notional_side_a": cum_notional["SIDE_A"],
                    "cum_notional_side_b": cum_notional["SIDE_B"],
                    "vwap_side_a": _vwap(cum_notional["SIDE_A"], cum_size["SIDE_A"]),
                    "vwap_side_b": _vwap(cum_notional["SIDE_B"], cum_size["SIDE_B"]),
                }
            )

        # Determine cheap vs expensive side based on first observed prices
        fa = first_price["SIDE_A"]
        fb = first_price["SIDE_B"]
        if fa is None and fb is None:
            cheap_side = None
            exp_side = None
        elif fa is None:
            cheap_side = "SIDE_B"
            exp_side = "SIDE_A"
        elif fb is None:
            cheap_side = "SIDE_A"
            exp_side = "SIDE_B"
        else:
            if fa <= fb:
                cheap_side = "SIDE_A"
                exp_side = "SIDE_B"
            else:
                cheap_side = "SIDE_B"
                exp_side = "SIDE_A"

        total_size_cheap = cum_size.get(cheap_side, 0.0) if cheap_side else 0.0
        total_size_exp = cum_size.get(exp_side, 0.0) if exp_side else 0.0
        size_ratio = None
        if total_size_cheap > 0:
            size_ratio = total_size_exp / total_size_cheap

        has_both = cum_size["SIDE_A"] > 0 and cum_size["SIDE_B"] > 0
        hedge_delay_trades = None
        hedge_delay_seconds = None
        if has_both and cheap_side and exp_side:
            seq_cheap = first_seq[cheap_side]
            seq_exp = first_seq[exp_side]
            ts_cheap = first_ts[cheap_side]
            ts_exp = first_ts[exp_side]
            if seq_cheap is not None and seq_exp is not None:
                hedge_delay_trades = abs(int(seq_exp) - int(seq_cheap))
            if ts_cheap is not None and ts_exp is not None:
                hedge_delay_seconds = abs((ts_exp - ts_cheap).total_seconds())

        first_entry_price_cheap = first_price.get(cheap_side) if cheap_side else None
        first_entry_price_exp = first_price.get(exp_side) if exp_side else None
        cheap_price_band = _band_label(first_entry_price_cheap, edges)

        # Max adverse move for cheap side (most negative move from first price)
        max_adverse_move_cheap = None
        if cheap_side and first_entry_price_cheap is not None:
            cheap_prices = [
                r["price_cents"]
                for r in timeseries_rows
                if r["market_key"] == market_key and r["side_norm"] == cheap_side
            ]
            if cheap_prices:
                min_price = min(cheap_prices)
                max_adverse_move_cheap = min_price - first_entry_price_cheap

        if not has_both:
            position_shape = "ONE_SIDED_ONLY"
        elif size_ratio is not None and size_ratio < 0.25:
            position_shape = "CHEAP_DOMINANT_ASYM"
        else:
            position_shape = "SYMMETRIC_BOTH_SIDES"

        summary_rows.append(
            {
                "market_key": market_key,
                "event_slug": group["event_slug"].iloc[0],
                "slug": group["slug"].iloc[0],
                "first_trade_ts": group["timestamp_dt"].min(),
                "last_trade_ts": group["timestamp_dt"].max(),
                "num_trades_total": int(len(group)),
                "num_trades_cheap": int(sum(group["side_norm"] == cheap_side)) if cheap_side else 0,
                "num_trades_expensive": int(sum(group["side_norm"] == exp_side)) if exp_side else 0,
                "first_entry_price_cheap": first_entry_price_cheap,
                "first_entry_price_expensive": first_entry_price_exp,
                "total_size_cheap": total_size_cheap,
                "total_size_expensive": total_size_exp,
                "size_ratio_exp_to_cheap": size_ratio,
                "has_both_sides": has_both,
                "hedge_delay_trades": hedge_delay_trades,
                "hedge_delay_seconds": hedge_delay_seconds,
                "cheap_price_band": cheap_price_band,
                "max_adverse_move_cheap": max_adverse_move_cheap,
                "position_shape": position_shape,
            }
        )

    timeseries_df = pd.DataFrame(timeseries_rows)
    summary_df = pd.DataFrame(summary_rows)

    timeseries_df.to_parquet(os.path.join(args.outdir, "event_position_timeseries.parquet"), index=False)
    summary_df.to_parquet(os.path.join(args.outdir, "event_positions.parquet"), index=False)

    pct_both = float(summary_df["has_both_sides"].mean()) if len(summary_df) else 0.0
    median_ratio = summary_df["size_ratio_exp_to_cheap"].median()
    band_dist = summary_df["cheap_price_band"].value_counts(dropna=False).to_dict()

    report = {
        "markets": int(len(summary_df)),
        "pct_with_both_sides": pct_both,
        "median_size_ratio_exp_to_cheap": None if pd.isna(median_ratio) else float(median_ratio),
        "cheap_price_band_distribution": band_dist,
    }

    with open(os.path.join(args.outdir, "phase3_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"markets {int(len(summary_df))}")
    print(f"pct_with_both_sides {pct_both:.3f}")
    print(f"median_size_ratio_exp_to_cheap {report['median_size_ratio_exp_to_cheap']}")
    print(f"cheap_price_band_distribution {band_dist}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
