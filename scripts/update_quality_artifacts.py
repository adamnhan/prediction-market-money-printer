#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from statistics import mean, pstdev


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    q = max(0.0, min(q, 1.0))
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return values[idx]


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _safe_pstdev(values: list[float]) -> float:
    return pstdev(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Update per-feature zscores in artifacts.")
    parser.add_argument(
        "--db",
        default="data/paper_trades.sqlite",
        help="Path to paper_trades sqlite DB.",
    )
    parser.add_argument(
        "--artifacts",
        default="strategy_artifacts.json",
        help="Path to strategy artifacts JSON.",
    )
    parser.add_argument(
        "--exclude-pnl-above",
        type=float,
        default=None,
        help="Exclude trades with pnl above this value.",
    )
    parser.add_argument(
        "--quality-quantile",
        type=float,
        default=None,
        help="If set, update quality_cutoff to this quantile of quality_score.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write updates back to the artifacts file.",
    )
    args = parser.parse_args()

    artifacts_path = Path(args.artifacts)
    raw = json.loads(artifacts_path.read_text(encoding="utf-8"))
    vol_p90 = raw["vol_quantiles"]["p90"]
    volsum_p90 = raw["volsum_quantiles"]["p90"]

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        """
        SELECT entry_price, p_base, vol_10, vol_sum_5, pnl
        FROM paper_trades
        WHERE entry_price IS NOT NULL
          AND p_base IS NOT NULL
          AND vol_10 IS NOT NULL
          AND vol_sum_5 IS NOT NULL
        """
    ).fetchall()

    move_from_base: list[float] = []
    vol_10_e: list[float] = []
    vol_sum_5_e: list[float] = []
    for entry_price, p_base, vol_10, vol_sum_5, pnl in rows:
        if args.exclude_pnl_above is not None and pnl is not None and pnl > args.exclude_pnl_above:
            continue
        move_from_base.append(abs(float(entry_price) - float(p_base)))
        vol_10_e.append(float(vol_10) - float(vol_p90))
        vol_sum_5_e.append(float(vol_sum_5) - float(volsum_p90))

    stats = {
        "move_from_base": {
            "mean": _safe_mean(move_from_base),
            "std": _safe_pstdev(move_from_base),
        },
        "vol_10_e": {"mean": _safe_mean(vol_10_e), "std": _safe_pstdev(vol_10_e)},
        "vol_sum_5_e": {
            "mean": _safe_mean(vol_sum_5_e),
            "std": _safe_pstdev(vol_sum_5_e),
        },
    }

    def zscore(value: float, feature: str) -> float:
        std = stats[feature]["std"]
        if std == 0:
            return 0.0
        return (value - stats[feature]["mean"]) / std

    quality_scores: list[float] = []
    for mfb, v10, vs5 in zip(move_from_base, vol_10_e, vol_sum_5_e):
        quality_scores.append(zscore(mfb, "move_from_base") + zscore(v10, "vol_10_e") + zscore(vs5, "vol_sum_5_e"))

    print("Feature zscore stats:")
    for key, stat in stats.items():
        print(f"  {key}: mean={stat['mean']:.6f}, std={stat['std']:.6f}")
    if quality_scores:
        print("Quality score quantiles:")
        for q in (0.5, 0.6, 0.7, 0.8, 0.9):
            print(f"  q{int(q*100)}={_percentile(quality_scores, q):.6f}")

    raw["zscore_features"] = stats
    if args.quality_quantile is not None:
        raw["quality_cutoff"] = _percentile(quality_scores, args.quality_quantile)
        print(f"Updated quality_cutoff to {raw['quality_cutoff']:.6f}")

    if args.write:
        artifacts_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote artifacts to {artifacts_path}")
    else:
        print("\nDry run (no write). Use --write to update artifacts.")


if __name__ == "__main__":
    main()
