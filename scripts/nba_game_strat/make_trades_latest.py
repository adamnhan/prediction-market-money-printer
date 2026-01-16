from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


MID_CONF_MIN = 0.15
MID_CONF_MAX = 0.30
PANIC_RET_3 = 0.05
VOL10_Q = 0.90
VOLSUM_Q = 0.90
ENTRY_DELAY_MIN = 2
MOVE_MIN = 0.07
TP = 0.03
SL = 0.05
MAX_HOLD_MIN = 10
COOLDOWN_MIN = 10
COST = 0.01


def _parse_ts(series: pd.Series) -> pd.Series:
    if is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        # Heuristic: treat as seconds if values look like epoch seconds.
        if series.dropna().max() < 1e11:
            return pd.to_datetime(series, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def _add_minutes(ts: pd.Series, minutes: int) -> pd.Series:
    if is_datetime64_any_dtype(ts):
        return ts + pd.Timedelta(minutes=minutes)
    return ts + minutes * 60


def _window_bounds(ts_val, minutes: int) -> tuple:
    if isinstance(ts_val, pd.Timestamp):
        return ts_val, ts_val + pd.Timedelta(minutes=minutes)
    return ts_val, ts_val + minutes * 60


def _simulate_trades(mkt_df: pd.DataFrame, sig_df: pd.DataFrame) -> list[dict]:
    trades = []
    mkt_df = mkt_df.sort_values("ts")
    sig_df = sig_df.sort_values("entry_ts")
    cooldown_until = pd.Timestamp.min if is_datetime64_any_dtype(mkt_df["ts"]) else -np.inf

    for row in sig_df.itertuples(index=False):
        if row.entry_ts <= cooldown_until:
            continue
        start_ts, end_ts = _window_bounds(row.entry_ts, MAX_HOLD_MIN)
        window = mkt_df[(mkt_df["ts"] > start_ts) & (mkt_df["ts"] <= end_ts)]
        if window.empty:
            continue

        if row.direction == "long":
            tp = row.entry_price + TP
            sl = row.entry_price - SL
            tp_hit = window[window["p_filled"] >= tp]
            sl_hit = window[window["p_filled"] <= sl]
        else:
            tp = row.entry_price - TP
            sl = row.entry_price + SL
            tp_hit = window[window["p_filled"] <= tp]
            sl_hit = window[window["p_filled"] >= sl]

        exit_row = None
        if (not tp_hit.empty) or (not sl_hit.empty):
            tp_ts = tp_hit["ts"].iloc[0] if not tp_hit.empty else pd.Timestamp.max
            sl_ts = sl_hit["ts"].iloc[0] if not sl_hit.empty else pd.Timestamp.max
            if tp_ts < sl_ts:
                exit_row = tp_hit.iloc[0]
            elif sl_ts < tp_ts:
                exit_row = sl_hit.iloc[0]
            else:
                exit_row = sl_hit.iloc[0] if not sl_hit.empty else tp_hit.iloc[0]
        if exit_row is None:
            exit_row = window.iloc[-1]

        exit_ts = exit_row["ts"]
        exit_price = exit_row["p_filled"]
        if row.direction == "long":
            pnl = (exit_price - row.entry_price) - COST
        else:
            pnl = (row.entry_price - exit_price) - COST

        trades.append(
            {
                "market_id": row.market_id,
                "segment_id": row.segment_id,
                "direction": row.direction,
                "entry_ts": row.entry_ts,
                "exit_ts": exit_ts,
                "entry_price": row.entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "p_base": row.p_base,
                "move_from_base": row.move_from_base,
                "vol_10_e": row.vol_10_e,
                "vol_sum_5_e": row.vol_sum_5_e,
                "ret_1_e": row.ret_1_e,
                "ret_3_e": row.ret_3_e,
                "ret_5_e": row.ret_5_e,
                "was_nan_e": row.was_nan_e,
                "trade_active_e": row.trade_active_e,
                "gap_flag_e": row.gap_flag_e,
            }
        )

        _, cooldown_until = _window_bounds(exit_ts, COOLDOWN_MIN)

    return trades


def build_trades(df: pd.DataFrame, entry_delay_min: int) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["p_open"], errors="ignore")
    df["ts"] = _parse_ts(df["ts"])
    df = df.sort_values(["market_id", "segment_id", "ts"]).reset_index(drop=True)

    non_nan = df[df["p_filled"].notna()].copy()
    p_open = (
        non_nan.sort_values(["market_id", "ts"])
        .groupby("market_id", group_keys=False)
        .head(5)
        .groupby("market_id")["p_filled"]
        .median()
        .rename("p_open")
    )
    df = df.merge(p_open, on="market_id", how="left")

    abs_conf = (df["p_open"] - 0.5).abs()
    df["mid_conf"] = (abs_conf >= MID_CONF_MIN) & (abs_conf < MID_CONF_MAX)

    q90_vol10 = df["vol_10"].quantile(VOL10_Q)
    q90_volsum = df["vol_sum_5"].quantile(VOLSUM_Q)
    panic = (df["vol_10"] >= q90_vol10) & (df["vol_sum_5"] >= q90_volsum)
    underdog = df["p_open"] < 0.5
    favorite = df["p_open"] > 0.5

    signal_short = df["mid_conf"] & panic & underdog & (df["ret_3"] >= PANIC_RET_3)
    signal_long = df["mid_conf"] & panic & favorite & (df["ret_3"] <= -PANIC_RET_3)
    mask = signal_short | signal_long

    signals = df.loc[mask, ["market_id", "segment_id", "ts", "p_open"]].copy()
    signals["direction"] = np.where(signal_short.loc[mask], "short", "long")
    signals["entry_ts"] = _add_minutes(signals["ts"], entry_delay_min)

    entry_cols = [
        "market_id",
        "segment_id",
        "ts",
        "p_filled",
        "vol_10",
        "vol_sum_5",
        "ret_1",
        "ret_3",
        "ret_5",
        "was_nan",
        "trade_active",
        "gap_flag",
    ]
    entry = df[entry_cols].rename(
        columns={
            "ts": "entry_ts",
            "p_filled": "entry_price",
            "vol_10": "vol_10_e",
            "vol_sum_5": "vol_sum_5_e",
            "ret_1": "ret_1_e",
            "ret_3": "ret_3_e",
            "ret_5": "ret_5_e",
            "was_nan": "was_nan_e",
            "trade_active": "trade_active_e",
            "gap_flag": "gap_flag_e",
        }
    )

    signals = signals.merge(entry, on=["market_id", "segment_id", "entry_ts"], how="inner")
    signals = signals[signals["entry_price"].notna()].copy()

    signals["p_base"] = signals["p_open"]
    signals["move_from_base"] = np.where(
        signals["direction"] == "short",
        signals["entry_price"] - signals["p_base"],
        signals["p_base"] - signals["entry_price"],
    )
    signals = signals[signals["move_from_base"] >= MOVE_MIN].copy()

    trades_list: list[dict] = []
    for (market_id, segment_id), g in df.groupby(["market_id", "segment_id"]):
        sig_g = signals[
            (signals["market_id"] == market_id) & (signals["segment_id"] == segment_id)
        ]
        if sig_g.empty:
            continue
        trades_list.extend(_simulate_trades(g, sig_g))

    columns = [
        "market_id",
        "segment_id",
        "direction",
        "entry_ts",
        "exit_ts",
        "entry_price",
        "exit_price",
        "pnl",
        "p_base",
        "move_from_base",
        "vol_10_e",
        "vol_sum_5_e",
        "ret_1_e",
        "ret_3_e",
        "ret_5_e",
        "was_nan_e",
        "trade_active_e",
        "gap_flag_e",
    ]
    return pd.DataFrame(trades_list, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate trades_latest.csv from candle features.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/nba_market_candles_features.csv"),
        help="Input candle feature CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/trades_latest.csv"),
        help="Output trades CSV.",
    )
    parser.add_argument(
        "--delay-min",
        type=int,
        default=ENTRY_DELAY_MIN,
        help="Entry delay in minutes after panic signal.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    trades = build_trades(df, entry_delay_min=args.delay_min)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.output, index=False)
    print(f"Saved {len(trades)} trades to {args.output}")


if __name__ == "__main__":
    main()
