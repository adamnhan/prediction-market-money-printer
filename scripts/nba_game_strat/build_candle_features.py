from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_STEP = 60
MINOR_GAP_MULTIPLE = 2


def add_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("ts").copy()

    group["p"] = group["yes_price"]
    group["was_nan"] = group["p"].isna().astype(int)
    group["p_filled"] = group["p"].ffill()
    group["p_is_valid"] = group["p_filled"].notna().astype(int)
    group["volume_filled"] = group["volume"].fillna(0)

    group["dt"] = group["ts"].diff()
    group["gap_flag"] = (group["dt"] >= 180).astype(int)
    group["is_step_multiple"] = (group["dt"] % EXPECTED_STEP == 0).astype(int)
    group["minor_gap_flag"] = (group["dt"] == EXPECTED_STEP * MINOR_GAP_MULTIPLE).astype(int)

    segment_id = group["gap_flag"].cumsum()
    group["segment_id"] = segment_id

    group["ret_1"] = group.groupby(segment_id)["p_filled"].diff(1)
    group["ret_3"] = group.groupby(segment_id)["p_filled"].diff(3)
    group["ret_5"] = group.groupby(segment_id)["p_filled"].diff(5)
    group["ret_10"] = group.groupby(segment_id)["p_filled"].diff(10)

    group["vol_5"] = (
        group.groupby(segment_id)["ret_1"]
        .rolling(5, min_periods=5)
        .std()
        .reset_index(level=0, drop=True)
    )
    group["vol_10"] = (
        group.groupby(segment_id)["ret_1"]
        .rolling(10, min_periods=10)
        .std()
        .reset_index(level=0, drop=True)
    )
    group["vol_20"] = (
        group.groupby(segment_id)["ret_1"]
        .rolling(20, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )

    group["vol_sum_5"] = (
        group.groupby(segment_id)["volume_filled"]
        .rolling(5, min_periods=5)
        .sum()
        .reset_index(level=0, drop=True)
    )
    group["vol_sum_10"] = (
        group.groupby(segment_id)["volume_filled"]
        .rolling(10, min_periods=10)
        .sum()
        .reset_index(level=0, drop=True)
    )
    group["trade_active"] = (group["volume_filled"] > 0).astype(int)

    return group


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = repo_root / "data" / "nba_market_candles_sample.csv"
    output_path = repo_root / "data" / "nba_market_candles_features.csv"

    df = pd.read_csv(input_path)
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df["yes_price"] = pd.to_numeric(df["yes_price"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")

    df = (
        df.groupby("market_id", group_keys=False, sort=False)
        .apply(add_features)
        .reset_index(drop=True)
    )

    df.to_csv(output_path, index=False)
    print(f"Saved feature CSV to {output_path}")


if __name__ == "__main__":
    main()
