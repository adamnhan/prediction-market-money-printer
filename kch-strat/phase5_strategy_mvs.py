#!/usr/bin/env python3
"""Phase 5: Minimal Viable Strategy (MVS) simulator (paper-test only)."""

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


def _compute_drawdown(pnl_series: pd.Series) -> float:
    equity = pnl_series.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min()) if len(dd) else 0.0


def _build_market_meta(trades: pd.DataFrame) -> pd.DataFrame:
    outcomes_by_market: Dict[str, List[str]] = {}
    for market_key, group in trades.groupby("market_key"):
        outcomes = []
        for val in group["outcomes"]:
            outcomes = _parse_outcomes(val)
            if outcomes:
                break
        outcomes_by_market[market_key] = outcomes

    trades["side_norm"] = trades.apply(
        lambda r: _normalize_side(r.get("outcome_raw"), outcomes_by_market.get(r["market_key"], [])), axis=1
    )
    trades = trades[trades["side_norm"].notna()].copy()
    trades = trades.sort_values(["market_key", "timestamp_dt"])

    meta_rows: List[Dict[str, Any]] = []
    for market_key, group in trades.groupby("market_key"):
        outcomes = outcomes_by_market.get(market_key, [])
        if len(outcomes) < 2:
            continue
        side_a = group[group["side_norm"] == "SIDE_A"]["price_cents_norm"]
        side_b = group[group["side_norm"] == "SIDE_B"]["price_cents_norm"]
        if side_a.empty or side_b.empty:
            continue
        price_a = side_a.iloc[0]
        price_b = side_b.iloc[0]
        if pd.isna(price_a) or pd.isna(price_b):
            continue
        if price_a <= price_b:
            cheap_side, exp_side = "SIDE_A", "SIDE_B"
            cheap_price, exp_price = float(price_a), float(price_b)
        else:
            cheap_side, exp_side = "SIDE_B", "SIDE_A"
            cheap_price, exp_price = float(price_b), float(price_a)

        meta_rows.append(
            {
                "market_key": market_key,
                "event_slug": group["event_slug"].iloc[0],
                "slug": group["slug"].iloc[0],
                "condition_id": group["condition_id"].iloc[0],
                "outcome_a": outcomes[0],
                "outcome_b": outcomes[1],
                "cheap_side": cheap_side,
                "exp_side": exp_side,
                "cheap_price_cents": cheap_price,
                "exp_price_cents": exp_price,
                "first_trade_ts": group["timestamp_dt"].min(),
                "last_trade_ts": group["timestamp_dt"].max(),
            }
        )

    return pd.DataFrame(meta_rows)


def _winner_side(row: pd.Series) -> Optional[str]:
    win = _safe_str(row.get("winning_outcome"))
    if not win:
        return None
    if win.lower() == str(row.get("outcome_a", "")).lower():
        return "SIDE_A"
    if win.lower() == str(row.get("outcome_b", "")).lower():
        return "SIDE_B"
    return None


def _simulate_mvs(
    markets: pd.DataFrame,
    capital_per_market_cents: float,
    ratio: float,
    max_concurrent: int,
    order_by: str,
    concurrency_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if order_by not in ("resolved_time", "last_trade_ts"):
        order_by = "resolved_time"

    markets = markets.sort_values(order_by)

    open_positions: List[Dict[str, Any]] = []
    results_rows: List[Dict[str, Any]] = []

    def _release_closed(current_time):
        nonlocal open_positions
        open_positions = [p for p in open_positions if p["exit_time"] and p["exit_time"] > current_time]

    for _, row in markets.iterrows():
        entry_time = row["first_trade_ts"]
        exit_time = row["resolved_time"] or row["last_trade_ts"]
        if pd.isna(entry_time) or pd.isna(exit_time):
            continue
        _release_closed(entry_time)

        cap_total = capital_per_market_cents
        remaining_cap = max_concurrent * capital_per_market_cents - sum(p["capital_alloc_cents"] for p in open_positions)

        if concurrency_mode == "skip":
            if remaining_cap < cap_total:
                continue
            scale = 1.0
        else:
            if remaining_cap <= 0:
                continue
            scale = min(1.0, remaining_cap / cap_total)

        n_cheap = cap_total / (1 + ratio)
        n_exp = ratio * cap_total / (1 + ratio)
        q_cheap = n_cheap / row["cheap_price_cents"]
        q_exp = n_exp / row["exp_price_cents"]

        winner_side = row["winner_side"]
        settle_value = None
        if winner_side == row["cheap_side"]:
            settle_value = 100.0 * q_cheap
        elif winner_side == row["exp_side"]:
            settle_value = 100.0 * q_exp

        cost_total = n_cheap + n_exp
        pnl = (settle_value - cost_total) if settle_value is not None else None

        results_rows.append(
            {
                "market_key": row["market_key"],
                "event_slug": row["event_slug"],
                "slug": row["slug"],
                "first_trade_ts": entry_time,
                "resolved_time": row["resolved_time"],
                "last_trade_ts": row["last_trade_ts"],
                "cheap_price_cents": row["cheap_price_cents"],
                "exp_price_cents": row["exp_price_cents"],
                "cheap_side": row["cheap_side"],
                "exp_side": row["exp_side"],
                "winner_side": winner_side,
                "capital_alloc_cents": cap_total * scale,
                "scale_factor": scale,
                "cost_total_cents": cost_total * scale,
                "settle_value_cents": settle_value * scale if settle_value is not None else None,
                "pnl_cents": pnl * scale if pnl is not None else None,
            }
        )

        open_positions.append(
            {
                "market_key": row["market_key"],
                "exit_time": exit_time,
                "capital_alloc_cents": cap_total * scale,
            }
        )

    results_df = pd.DataFrame(results_rows)
    if "pnl_cents" in results_df.columns:
        results_df = results_df.dropna(subset=["pnl_cents"])
    else:
        results_df = pd.DataFrame(columns=["pnl_cents", "resolved_time", "last_trade_ts"])

    if len(results_df) == 0:
        curve_df = pd.DataFrame(columns=["timestamp", "cum_pnl_cents"])
    else:
        curve = results_df.copy()
        if "resolved_time" not in curve.columns:
            curve["resolved_time"] = pd.NaT
        if "last_trade_ts" not in curve.columns:
            curve["last_trade_ts"] = pd.NaT
        curve["exit_time"] = curve["resolved_time"].fillna(curve["last_trade_ts"])
        curve = curve.sort_values("exit_time")
        curve["cum_pnl_cents"] = curve["pnl_cents"].cumsum()
        curve_df = curve[["exit_time", "cum_pnl_cents"]].rename(columns={"exit_time": "timestamp"})

    if len(results_df) == 0:
        metrics = {
            "simulated_markets": 0,
            "total_pnl_cents": 0.0,
            "max_drawdown_cents": 0.0,
            "median_pnl_cents": 0.0,
            "p10_pnl_cents": 0.0,
            "p90_pnl_cents": 0.0,
            "best_market_pnl_cents": 0.0,
            "worst_market_pnl_cents": 0.0,
        }
    else:
        metrics = {
            "simulated_markets": int(len(results_df)),
            "total_pnl_cents": float(results_df["pnl_cents"].sum()),
            "max_drawdown_cents": _compute_drawdown(results_df.sort_values("resolved_time")["pnl_cents"]),
            "median_pnl_cents": float(results_df["pnl_cents"].median()),
            "p10_pnl_cents": float(results_df["pnl_cents"].quantile(0.1)),
            "p90_pnl_cents": float(results_df["pnl_cents"].quantile(0.9)),
            "best_market_pnl_cents": float(results_df["pnl_cents"].max()),
            "worst_market_pnl_cents": float(results_df["pnl_cents"].min()),
        }

    return results_df, curve_df, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 MVS simulator (offline, no live trading)")
    parser.add_argument("--phase2_path", default="./kch-strat/phase2_out/trades_enriched.parquet")
    parser.add_argument("--phase4_pnl_path", default="./kch-strat/phase4_out/market_pnl.parquet")
    parser.add_argument("--resolution_path", default="./kch-strat/phase4_out/market_resolution_dim.parquet")
    parser.add_argument("--resolution_manual_csv", default="./kch-strat/phase4_out/market_resolution_manual.csv")
    parser.add_argument("--outdir", default="./kch-strat/phase5_out")
    parser.add_argument("--band", default="44,50")
    parser.add_argument("--ratio", type=float, default=0.65)
    parser.add_argument("--capital_per_market", type=float, default=10000.0)
    parser.add_argument("--max_concurrent", type=int, default=10)
    parser.add_argument("--order_by", default="resolved_time")
    parser.add_argument("--concurrency_mode", default="skip", choices=["skip", "scale"])
    parser.add_argument("--sweep", default="false")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    trades = pd.read_parquet(args.phase2_path)
    trades = trades[
        trades["event_slug"].astype(str).str.startswith("nhl-")
        & ~trades["slug"].astype(str).str.contains("spread|total", case=False, na=False)
    ].copy()
    trades["event_slug"] = trades["event_slug"].apply(_safe_str)
    trades["slug"] = trades["slug"].apply(_safe_str)
    trades["market_key"] = trades["event_slug"] + "::" + trades["slug"]
    trades["outcome_raw"] = trades["outcome"].apply(_safe_str)
    trades["price_cents_norm"] = trades.apply(_price_cents, axis=1)
    trades["timestamp_dt"] = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")

    meta = _build_market_meta(trades)

    phase4 = pd.read_parquet(args.phase4_pnl_path)
    res = pd.read_parquet(args.resolution_path)
    if args.resolution_manual_csv and os.path.exists(args.resolution_manual_csv):
        manual = pd.read_csv(args.resolution_manual_csv)
        manual = manual[[c for c in manual.columns if c in ("condition_id", "winning_outcome", "resolved_time")]].copy()
        if "resolved_time" not in manual.columns:
            manual["resolved_time"] = None
        res = res.merge(manual, on="condition_id", how="left", suffixes=("", "_manual"))
        res["winning_outcome"] = res["winning_outcome_manual"].fillna(res["winning_outcome"])
        res["resolved_time"] = res["resolved_time_manual"].fillna(res["resolved_time"])
        res = res.drop(columns=[c for c in res.columns if c.endswith("_manual")])

    merged = meta.merge(res, on="condition_id", how="left").merge(
        phase4[["market_key", "cheap_price_band", "size_ratio_exp_to_cheap", "hedge_delay_seconds"]],
        on="market_key",
        how="left",
    )

    merged["winner_side"] = merged.apply(_winner_side, axis=1)
    merged["resolved_time"] = pd.to_datetime(merged["resolved_time"], utc=True, errors="coerce")

    band_lo, band_hi = [float(x) for x in args.band.split(",")]
    eligible = merged[
        merged["winner_side"].notna()
        & merged["cheap_price_cents"].between(band_lo, band_hi, inclusive="both")
    ].copy()

    eligible["resolved_time"] = eligible["resolved_time"].fillna(eligible["last_trade_ts"])

    results, curve, metrics = _simulate_mvs(
        eligible,
        capital_per_market_cents=args.capital_per_market,
        ratio=args.ratio,
        max_concurrent=args.max_concurrent,
        order_by=args.order_by,
        concurrency_mode=args.concurrency_mode,
    )

    results.to_parquet(os.path.join(args.outdir, "mvs_market_results.parquet"), index=False)
    curve.to_csv(os.path.join(args.outdir, "mvs_portfolio_curve.csv"), index=False)

    report = {
        "eligible_markets": int(len(eligible)),
        "simulated_markets": metrics["simulated_markets"],
        "total_pnl_cents": metrics["total_pnl_cents"],
        "max_drawdown_cents": metrics["max_drawdown_cents"],
        "median_pnl_cents": metrics["median_pnl_cents"],
        "p10_pnl_cents": metrics["p10_pnl_cents"],
        "p90_pnl_cents": metrics["p90_pnl_cents"],
        "best_market_pnl_cents": metrics["best_market_pnl_cents"],
        "worst_market_pnl_cents": metrics["worst_market_pnl_cents"],
        "parameters": {
            "band": args.band,
            "ratio": args.ratio,
            "capital_per_market_cents": args.capital_per_market,
            "max_concurrent": args.max_concurrent,
            "order_by": args.order_by,
            "concurrency_mode": args.concurrency_mode,
        },
    }

    with open(os.path.join(args.outdir, "phase5_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"eligible_markets {report['eligible_markets']}")
    print(f"simulated_markets {report['simulated_markets']}")
    print(f"total_pnl_cents {report['total_pnl_cents']}")
    print(f"max_drawdown_cents {report['max_drawdown_cents']}")
    print(f"median_pnl_cents {report['median_pnl_cents']} p10 {report['p10_pnl_cents']} p90 {report['p90_pnl_cents']}")
    print(f"best_market_pnl_cents {report['best_market_pnl_cents']} worst_market_pnl_cents {report['worst_market_pnl_cents']}")
    print(
        f"params band={args.band} ratio={args.ratio} capital_per_market_cents={args.capital_per_market} "
        f"max_concurrent={args.max_concurrent}"
    )

    if str(args.sweep).lower() == "true":
        sweep_rows = []
        bands = [(40, 50), (44, 50)]
        ratios = [0.50, 0.65, 0.80]
        for blo, bhi in bands:
            for r in ratios:
                elig = merged[
                    merged["winner_side"].notna()
                    & merged["cheap_price_cents"].between(blo, bhi, inclusive="both")
                ].copy()
                elig["resolved_time"] = elig["resolved_time"].fillna(elig["last_trade_ts"])
                _, _, m = _simulate_mvs(
                    elig,
                    capital_per_market_cents=args.capital_per_market,
                    ratio=r,
                    max_concurrent=args.max_concurrent,
                    order_by=args.order_by,
                    concurrency_mode=args.concurrency_mode,
                )
                sweep_rows.append(
                    {
                        "band": f"{blo},{bhi}",
                        "ratio": r,
                        "eligible_markets": int(len(elig)),
                        "simulated_markets": m["simulated_markets"],
                        "total_pnl_cents": m["total_pnl_cents"],
                        "max_drawdown_cents": m["max_drawdown_cents"],
                        "median_pnl_cents": m["median_pnl_cents"],
                    }
                )

        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df.to_csv(os.path.join(args.outdir, "sweep_summary.csv"), index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
