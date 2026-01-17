#!/usr/bin/env python3
"""
Market-making feasibility from candle data only.

Assumptions:
- Candle data lacks bid/ask; results depend on fill model assumptions.
- Use multiple fill modes to bracket optimistic to strict outcomes.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_DATA_DIR = "./data/candles"
DEFAULT_PATTERN = "*.csv"


OHLC_ALIASES = {
    "ts": ["ts", "timestamp", "time", "datetime", "date"],
    "open": ["open", "o", "open_price"],
    "high": ["high", "h", "high_price"],
    "low": ["low", "l", "low_price"],
    "close": ["close", "c", "close_price"],
    "volume": ["volume", "v"],
}


def _find_col(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    return None


def standardize_candles(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    mapping = {}
    for key, aliases in OHLC_ALIASES.items():
        found = _find_col(cols, aliases)
        if found:
            mapping[found] = key
    df = df.rename(columns=mapping)
    missing = [k for k in ["ts", "open", "high", "low", "close"] if k not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df.sort_values("ts").reset_index(drop=True)


def load_candles_from_dir(data_dir: str, pattern: str) -> Dict[str, pd.DataFrame]:
    path_pattern = os.path.join(data_dir, pattern)
    files = sorted(glob.glob(path_pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {path_pattern}")
    out: Dict[str, pd.DataFrame] = {}
    total = len(files)
    for idx, file_path in enumerate(files, start=1):
        market = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_csv(file_path)
        df = standardize_candles(df)
        out[market] = df
        _progress(f"Loading candles", idx, total)
    return out


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["close"].diff()
    df["abs_ret"] = df["ret"].abs()
    df["range"] = df["high"] - df["low"]
    return df


def time_to_expiry_buckets(df: pd.DataFrame, bucket_edges: List[int]) -> pd.Series:
    # Approximate by bar index from end. 0 = last bar.
    idx_from_end = (len(df) - 1) - np.arange(len(df))
    labels = []
    for i in range(len(bucket_edges) - 1):
        labels.append(f"{bucket_edges[i]}-{bucket_edges[i+1] - 1}")
    buckets = pd.cut(
        idx_from_end,
        bins=bucket_edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    return buckets


def reversion_probability(
    close: np.ndarray, window: int, epsilon: float
) -> float:
    n = len(close)
    if n <= window:
        return float("nan")
    hits = 0
    total = 0
    for i in range(n - window):
        entry = close[i]
        target_low = entry - epsilon
        target_high = entry + epsilon
        future = close[i + 1 : i + window + 1]
        if np.any((future >= target_low) & (future <= target_high)):
            hits += 1
        total += 1
    return hits / total if total > 0 else float("nan")


def trendiness_metrics(close: np.ndarray) -> Dict[str, float]:
    if len(close) < 5:
        return {"autocorr_1": float("nan"), "net_to_abs": float("nan")}
    ret = np.diff(close)
    if len(ret) < 2:
        autocorr = float("nan")
    else:
        autocorr = float(np.corrcoef(ret[:-1], ret[1:])[0, 1])
    net_move = float(abs(close[-1] - close[0]))
    abs_move = float(np.sum(np.abs(ret)))
    net_to_abs = net_move / abs_move if abs_move > 0 else float("nan")
    return {"autocorr_1": autocorr, "net_to_abs": net_to_abs}


def compute_descriptive_stats(
    df: pd.DataFrame, bucket_edges: Optional[List[int]] = None
) -> pd.DataFrame:
    df = add_returns(df)
    stats = {}
    stats["median_abs_ret"] = df["abs_ret"].median()
    stats["p80_abs_ret"] = df["abs_ret"].quantile(0.80)
    stats["p90_abs_ret"] = df["abs_ret"].quantile(0.90)
    stats["p95_abs_ret"] = df["abs_ret"].quantile(0.95)
    stats["median_range"] = df["range"].median()
    stats["p90_range"] = df["range"].quantile(0.90)
    stats["reversion_p_3"] = reversion_probability(
        df["close"].to_numpy(), window=3, epsilon=0.02
    )
    stats["reversion_p_5"] = reversion_probability(
        df["close"].to_numpy(), window=5, epsilon=0.02
    )
    stats.update(trendiness_metrics(df["close"].to_numpy()))
    out = pd.DataFrame([stats])

    if bucket_edges:
        df["tte_bucket"] = time_to_expiry_buckets(df, bucket_edges)
        bucket_stats = (
            df.groupby("tte_bucket", observed=True)["abs_ret"]
            .agg(["median", lambda s: s.quantile(0.9)])
            .rename(columns={"median": "median_abs_ret", "<lambda_0>": "p90_abs_ret"})
            .reset_index()
        )
        bucket_stats["metric"] = "bucket"
        out["metric"] = "overall"
        out = pd.concat([out, bucket_stats], ignore_index=True, sort=False)
    return out


@dataclass
class FillParams:
    fill_mode: str
    slip_cents: float
    p_fill: float


@dataclass
class FeeParams:
    fee_bps: float
    fee_mode: str

    def fee_per_contract(self, price: float) -> float:
        if self.fee_mode == "zero":
            return 0.0
        # Stub: default to bps on price (price in dollars).
        return (self.fee_bps / 10000.0) * price


def _touched(low: float, high: float, price: float) -> bool:
    return low <= price <= high


def _fill_allowed(touched: bool, params: FillParams) -> bool:
    if not touched:
        return False
    if params.fill_mode == "optimistic":
        return True
    if params.fill_mode == "strict":
        return True
    if params.fill_mode == "probabilistic":
        return np.random.random() < params.p_fill
    raise ValueError(f"Unknown fill mode: {params.fill_mode}")


def _effective_price(price: float, side: str, params: FillParams) -> float:
    # slip_cents is interpreted as extra required improvement for maker fills
    slip = params.slip_cents / 100.0
    if params.fill_mode == "strict":
        if side == "bid":
            return price - slip
        return price + slip
    return price


def simulate_inventory_loop(
    df: pd.DataFrame,
    side: str,
    i_max: int,
    i_target: int,
    base_spread_cents: float,
    inv_skew_cents_per_contract: float,
    fill_params: FillParams,
    fee_params: FeeParams,
    fair_mode: str,
) -> Dict[str, float]:
    inv = 0
    cash = 0.0
    inventory_series = []
    trade_pnls = []
    max_inv = 0
    min_inv = 0
    cooldown_bid = False
    cooldown_ask = False

    base_spread = base_spread_cents / 100.0
    inv_skew = inv_skew_cents_per_contract / 100.0

    for _, row in df.iterrows():
        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])
        if fair_mode == "mid":
            fair = (high + low) / 2.0
        else:
            fair = close

        bid_offset = base_spread / 2.0 + max(0, inv - i_target) * inv_skew
        ask_offset = base_spread / 2.0 + max(0, i_target - inv) * inv_skew
        bid = fair - bid_offset
        ask = fair + ask_offset

        # Bid
        if inv < i_max and not cooldown_bid:
            bid_touch = _touched(low, high, _effective_price(bid, "bid", fill_params))
            if _fill_allowed(bid_touch, fill_params):
                inv += 1
                cash -= bid
                cash -= fee_params.fee_per_contract(bid)
                cooldown_bid = True
        # Ask
        if inv > 0 and not cooldown_ask:
            ask_touch = _touched(low, high, _effective_price(ask, "ask", fill_params))
            if _fill_allowed(ask_touch, fill_params):
                inv -= 1
                cash += ask
                cash -= fee_params.fee_per_contract(ask)
                cooldown_ask = True

        cooldown_bid = False
        cooldown_ask = False
        inventory_series.append(inv)
        max_inv = max(max_inv, inv)
        min_inv = min(min_inv, inv)

    # Mark-to-market at close
    final_price = float(df.iloc[-1]["close"])
    pnl = cash + inv * final_price
    drawdown_proxy = _drawdown_proxy(pnl_series(_equity_curve(df, inventory_series, cash)))
    return {
        "pnl": pnl,
        "max_inv": max_inv,
        "min_inv": min_inv,
        "time_at_0": inventory_series.count(0) / len(inventory_series),
        "time_at_i_max": inventory_series.count(i_max) / len(inventory_series),
        "drawdown_proxy": drawdown_proxy,
        "trades": len(trade_pnls),
    }


def simulate_take_profit_baseline(
    df: pd.DataFrame,
    threshold_cents: float,
    take_profit_cents: float,
    fill_params: FillParams,
    fee_params: FeeParams,
) -> Dict[str, float]:
    threshold = threshold_cents / 100.0
    take_profit = take_profit_cents / 100.0
    in_pos = False
    entry = 0.0
    cash = 0.0
    inv = 0
    inventory_series = []
    trades = 0

    for _, row in df.iterrows():
        close = float(row["close"])
        low = float(row["low"])
        high = float(row["high"])

        if not in_pos:
            if close <= (df["close"].shift(1).fillna(close) - threshold).iloc[0]:
                # Enter on next bar if touched
                if _fill_allowed(_touched(low, high, close), fill_params):
                    in_pos = True
                    entry = close
                    inv = 1
                    cash -= entry + fee_params.fee_per_contract(entry)
                    trades += 1
        else:
            target = entry + take_profit
            if _fill_allowed(_touched(low, high, target), fill_params):
                in_pos = False
                inv = 0
                cash += target - fee_params.fee_per_contract(target)
                trades += 1

        inventory_series.append(inv)

    final_price = float(df.iloc[-1]["close"])
    pnl = cash + inv * final_price
    drawdown_proxy = _drawdown_proxy(pnl_series(_equity_curve(df, inventory_series, cash)))
    return {
        "pnl": pnl,
        "time_at_0": inventory_series.count(0) / len(inventory_series),
        "time_at_i_max": float("nan"),
        "drawdown_proxy": drawdown_proxy,
        "trades": trades,
    }


def pnl_series(equity: np.ndarray) -> np.ndarray:
    return equity - equity[0]


def _equity_curve(df: pd.DataFrame, inventory_series: List[int], cash: float) -> np.ndarray:
    closes = df["close"].to_numpy()
    inv = np.array(inventory_series, dtype=float)
    # cash is final cash, so approximate by offsetting to final mark-to-market
    final_equity = cash + inv[-1] * closes[-1]
    equity = inv * closes
    equity = equity - equity[-1] + final_equity
    return equity


def _drawdown_proxy(equity: np.ndarray) -> float:
    peak = -np.inf
    max_dd = 0.0
    for val in equity:
        peak = max(peak, val)
        dd = peak - val
        max_dd = max(max_dd, dd)
    return max_dd


def run_strategy(
    candles: Dict[str, pd.DataFrame],
    strategy: str,
    side: str,
    fill_params: FillParams,
    fee_params: FeeParams,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    total = len(candles)
    step = 0
    for market, df in candles.items():
        step += 1
        if strategy == "inventory_loop":
            res = simulate_inventory_loop(
                df=df,
                side=side,
                i_max=args.i_max,
                i_target=args.i_target,
                base_spread_cents=args.base_spread_cents,
                inv_skew_cents_per_contract=args.inv_skew_cents_per_contract,
                fill_params=fill_params,
                fee_params=fee_params,
                fair_mode=args.fair,
            )
        else:
            res = simulate_take_profit_baseline(
                df=df,
                threshold_cents=args.entry_threshold_cents,
                take_profit_cents=args.take_profit_cents,
                fill_params=fill_params,
                fee_params=fee_params,
            )
        res["market"] = market
        res["strategy"] = strategy
        rows.append(res)
        _progress(f"Strategy {strategy}", step, total)
    return pd.DataFrame(rows)


def grid_search_inventory_loop(
    candles: Dict[str, pd.DataFrame],
    fill_params: FillParams,
    fee_params: FeeParams,
    args: argparse.Namespace,
) -> pd.DataFrame:
    offsets = [float(x) for x in args.grid.split(",")]
    results = []
    total = len(offsets) * len(offsets) * len(args.grid_i_max) * len(candles)
    step = 0
    for bid_offset in offsets:
        for ask_offset in offsets:
            for i_max in args.grid_i_max:
                for market, df in candles.items():
                    step += 1
                    res = simulate_inventory_loop(
                        df=df,
                        side=args.side,
                        i_max=i_max,
                        i_target=args.i_target,
                        base_spread_cents=bid_offset + ask_offset,
                        inv_skew_cents_per_contract=args.inv_skew_cents_per_contract,
                        fill_params=fill_params,
                        fee_params=fee_params,
                        fair_mode=args.fair,
                    )
                    res.update(
                        {
                            "market": market,
                            "bid_offset": bid_offset,
                            "ask_offset": ask_offset,
                            "i_max": i_max,
                        }
                    )
                    results.append(res)
                    _progress("Grid search", step, total)
    return pd.DataFrame(results)


def save_heatmap(df: pd.DataFrame, value_col: str, out_path: str) -> None:
    pivot = df.pivot_table(
        index="bid_offset", columns="ask_offset", values=value_col, aggfunc="mean"
    )
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label=value_col)
    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns], rotation=45)
    plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    plt.title(f"Heatmap: {value_col}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--fidelity", default="1m")
    parser.add_argument("--side", default="NO", choices=["YES", "NO"])
    parser.add_argument("--grid", default="2,4,6,8")
    parser.add_argument("--fill_mode", default="strict", choices=["optimistic", "strict", "probabilistic"])
    parser.add_argument("--slip_cents", type=float, default=1.0)
    parser.add_argument("--p_fill", type=float, default=0.5)
    parser.add_argument("--i_max", type=int, default=5)
    parser.add_argument("--i_target", type=int, default=2)
    parser.add_argument("--base_spread_cents", type=float, default=4.0)
    parser.add_argument("--inv_skew_cents_per_contract", type=float, default=0.5)
    parser.add_argument("--fee_bps", type=float, default=0.0)
    parser.add_argument("--fee_mode", default="zero", choices=["zero", "bps"])
    parser.add_argument("--fair", default="close", choices=["close", "mid"])
    parser.add_argument("--entry_threshold_cents", type=float, default=4.0)
    parser.add_argument("--take_profit_cents", type=float, default=4.0)
    parser.add_argument("--grid_i_max", default="3,5,8")
    parser.add_argument("--output_dir", default="./research_outputs")
    parser.add_argument("--stats_only", action="store_true")
    return parser.parse_args()


def _parse_int_list(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if str(x).strip() != ""]


def _progress(label: str, current: int, total: int) -> None:
    if total <= 0:
        return
    width = 30
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "." * (width - filled)
    sys.stdout.write(f"\r{label}: [{bar}] {current}/{total}")
    if current >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    args.grid_i_max = _parse_int_list(args.grid_i_max)

    candles = load_candles_from_dir(args.data_dir, args.pattern)
    fill_params = FillParams(
        fill_mode=args.fill_mode,
        slip_cents=args.slip_cents,
        p_fill=args.p_fill,
    )
    fee_params = FeeParams(
        fee_bps=args.fee_bps,
        fee_mode=args.fee_mode,
    )

    # Stats
    stats_rows = []
    total_stats = len(candles)
    step_stats = 0
    for market, df in candles.items():
        step_stats += 1
        stats = compute_descriptive_stats(df, bucket_edges=[0, 60, 120, 240, 5000])
        stats["market"] = market
        stats_rows.append(stats)
        _progress("Stats", step_stats, total_stats)
    stats_df = pd.concat(stats_rows, ignore_index=True, sort=False)
    stats_path = os.path.join(args.output_dir, "descriptive_stats.csv")
    stats_df.to_csv(stats_path, index=False)

    if args.stats_only:
        print(f"Saved stats to {stats_path}")
        return

    # Strategies
    inv_df = run_strategy(
        candles=candles,
        strategy="inventory_loop",
        side=args.side,
        fill_params=fill_params,
        fee_params=fee_params,
        args=args,
    )
    inv_path = os.path.join(args.output_dir, "inventory_loop_results.csv")
    inv_df.to_csv(inv_path, index=False)

    baseline_df = run_strategy(
        candles=candles,
        strategy="take_profit",
        side=args.side,
        fill_params=fill_params,
        fee_params=fee_params,
        args=args,
    )
    baseline_path = os.path.join(args.output_dir, "baseline_results.csv")
    baseline_df.to_csv(baseline_path, index=False)

    # Grid search
    grid_df = grid_search_inventory_loop(candles, fill_params, fee_params, args)
    grid_path = os.path.join(args.output_dir, "grid_results.csv")
    grid_df.to_csv(grid_path, index=False)

    heatmap_path = os.path.join(args.output_dir, "heatmap_pnl.png")
    save_heatmap(grid_df, "pnl", heatmap_path)

    print(f"Saved stats to {stats_path}")
    print(f"Saved inventory loop results to {inv_path}")
    print(f"Saved baseline results to {baseline_path}")
    print(f"Saved grid results to {grid_path}")
    print(f"Saved heatmap to {heatmap_path}")


if __name__ == "__main__":
    main()
