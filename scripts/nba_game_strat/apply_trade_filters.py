from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


def _metrics(t: pd.DataFrame) -> dict:
    if t.empty:
        return {
            "num_trades": 0,
            "win_rate": np.nan,
            "mean_pnl": np.nan,
            "median_pnl": np.nan,
            "total_pnl": 0.0,
            "max_drawdown": np.nan,
        }
    t = t.sort_values("exit_ts")
    cum = t["pnl"].cumsum()
    dd = cum - cum.cummax()
    return {
        "num_trades": int(len(t)),
        "win_rate": (t["pnl"] > 0).mean(),
        "mean_pnl": t["pnl"].mean(),
        "median_pnl": t["pnl"].median(),
        "total_pnl": t["pnl"].sum(),
        "max_drawdown": dd.min(),
    }


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return series * 0
    return (series - series.mean()) / std


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply A/B/C filters to trades_latest.csv.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/trades_latest.csv"),
        help="Input trades CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV for results table.",
    )
    args = parser.parse_args()

    try:
        trades = pd.read_csv(args.input, parse_dates=["entry_ts", "exit_ts"])
    except (FileNotFoundError, EmptyDataError):
        print(f"No trades found in {args.input}")
        return

    if trades.empty:
        print(f"No trades found in {args.input}")
        return

    baseline = trades.copy()
    baseline_n = len(baseline)
    print(f"Baseline trades: {baseline_n}")

    vol10_q = baseline["vol_10_e"].quantile([0.90, 0.92, 0.95])
    volsum_q = baseline["vol_sum_5_e"].quantile([0.90, 0.92, 0.95])

    filters = {
        "A_move>=0.07": baseline["move_from_base"] >= 0.07,
        "A_move>=0.09": baseline["move_from_base"] >= 0.09,
        "A_move>=0.11": baseline["move_from_base"] >= 0.11,
        "B_vol10>=q90": baseline["vol_10_e"] >= vol10_q.loc[0.90],
        "B_vol10>=q92": baseline["vol_10_e"] >= vol10_q.loc[0.92],
        "B_vol10>=q95": baseline["vol_10_e"] >= vol10_q.loc[0.95],
        "C_volsum>=q90": baseline["vol_sum_5_e"] >= volsum_q.loc[0.90],
        "C_volsum>=q92": baseline["vol_sum_5_e"] >= volsum_q.loc[0.92],
        "C_volsum>=q95": baseline["vol_sum_5_e"] >= volsum_q.loc[0.95],
    }

    if "was_nan_e" in baseline.columns:
        filters["D_no_stale"] = baseline["was_nan_e"] == 0
    else:
        print("Skipping D_no_stale: missing column was_nan_e")

    if "active_last_3" in baseline.columns:
        filters["E_active_last_3>=2"] = baseline["active_last_3"] >= 2
    else:
        print("Skipping E_active_last_3>=2: missing column active_last_3")

    if "gap_recent_5" in baseline.columns:
        filters["F_gap_recent_5==0"] = baseline["gap_recent_5"] == 0
    else:
        print("Skipping F_gap_recent_5==0: missing column gap_recent_5")

    if "ret1_sign_changes_5" in baseline.columns:
        filters["G_ret1_signchg>=1"] = baseline["ret1_sign_changes_5"] >= 1
    else:
        print("Skipping G_ret1_signchg>=1: missing column ret1_sign_changes_5")

    quality_score = (
        _zscore(baseline["move_from_base"])
        + _zscore(baseline["vol_10_e"])
        + _zscore(baseline["vol_sum_5_e"])
    )
    score_cut = {
        "top50": quality_score.quantile(0.50),
        "top40": quality_score.quantile(0.60),
        "top30": quality_score.quantile(0.70),
    }
    filters["H_quality_top50"] = quality_score >= score_cut["top50"]
    filters["H_quality_top40"] = quality_score >= score_cut["top40"]
    filters["H_quality_top30"] = quality_score >= score_cut["top30"]

    results = []
    for name, mask in filters.items():
        t = baseline[mask].copy()
        m = _metrics(t)
        m["variant"] = name
        m["retained_pct"] = (m["num_trades"] / baseline_n) if baseline_n else np.nan
        m["filtered_out"] = baseline_n - m["num_trades"] if baseline_n else 0
        results.append(m)
        print(f"{name}: kept {m['num_trades']} / {baseline_n} (filtered_out={m['filtered_out']})")

    results_df = pd.DataFrame(results).sort_values("total_pnl", ascending=False)
    print("\nResults (sorted by total_pnl):")
    print(results_df.to_string(index=False))

    best_mean = results_df.loc[results_df["mean_pnl"].idxmax()]
    best_total = results_df.loc[results_df["total_pnl"].idxmax()]
    pos_total = results_df[results_df["total_pnl"] > 0]
    best_dd = pos_total.loc[pos_total["max_drawdown"].idxmax()] if not pos_total.empty else None

    print(f"\nBest by mean_pnl: {best_mean['variant']}")
    print(f"Best by total_pnl: {best_total['variant']}")
    if best_dd is not None:
        print(
            "Best by max_drawdown (least negative) with positive total_pnl:"
            f" {best_dd['variant']}"
        )
    else:
        print("No variants with positive total_pnl for max_drawdown selection.")

    if "entry_ts" in baseline.columns:
        top3 = results_df.head(3)["variant"].tolist()
        oos_rows = []
        for name in top3:
            t = baseline[filters[name]].sort_values("entry_ts")
            if t.empty:
                oos_rows.append({"variant": name, "oos_mean_pnl": np.nan, "oos_total_pnl": 0.0})
                continue
            split_idx = int(len(t) * 0.7)
            oos = t.iloc[split_idx:]
            oos_rows.append(
                {
                    "variant": name,
                    "oos_mean_pnl": oos["pnl"].mean() if not oos.empty else np.nan,
                    "oos_total_pnl": oos["pnl"].sum() if not oos.empty else 0.0,
                }
            )
        oos_df = pd.DataFrame(oos_rows)
        print("\nOOS (70/30) for top 3 variants by total_pnl:")
        print(oos_df.to_string(index=False))
    else:
        print("Skipping OOS check: missing column entry_ts")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output, index=False)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
