from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


SECONDS_HOUR = 3600
SECONDS_10M = 600


logger = logging.getLogger("nba_low_conf_dataset")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_market_start_times(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["event_start_ts"] = pd.to_numeric(df["event_start_ts"], errors="coerce")
    records = []
    for _, row in df.iterrows():
        game_start_ts = row["event_start_ts"]
        if pd.isna(game_start_ts):
            continue
        for col in ("market_id_1", "market_id_2"):
            market_id = row.get(col)
            if isinstance(market_id, str) and market_id.strip():
                records.append(
                    {
                        "market_ticker": market_id.strip(),
                        "game_start_ts": int(game_start_ts),
                    }
                )
    return pd.DataFrame(records)


def load_settlement_map(json_path: Path) -> dict[str, int]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    settlement: dict[str, int] = {}
    for markets in payload.values():
        for market in markets:
            ticker = market.get("ticker")
            value = market.get("settlement_value")
            if not ticker or value is None:
                continue
            settle_yes = 1 if int(value) == 100 else 0
            settlement[ticker] = settle_yes
    return settlement


def _last_close_before(group: pd.DataFrame, cutoff_ts: int) -> tuple[float, int]:
    eligible = group[(group["ts"] <= cutoff_ts) & group["close"].notna()]
    if eligible.empty:
        return np.nan, -1
    row = eligible.iloc[-1]
    return float(row["close"]), int(row["ts"])


def build_dataset(
    candles_path: Path,
    markets_csv: Path,
    markets_json: Path,
    output_path: Path,
) -> None:
    candles = pd.read_csv(candles_path)
    candles = candles.rename(columns={"market_id": "market_ticker", "yes_price": "close"})
    candles["ts"] = pd.to_numeric(candles["ts"], errors="coerce")
    candles["close"] = pd.to_numeric(candles["close"], errors="coerce")
    candles = candles.dropna(subset=["market_ticker", "ts"])
    candles["ts"] = candles["ts"].astype(int)

    markets = load_market_start_times(markets_csv)
    settle_map = load_settlement_map(markets_json)

    markets["settle_yes"] = markets["market_ticker"].map(settle_map)
    markets = markets.dropna(subset=["settle_yes"]).copy()
    markets["settle_yes"] = markets["settle_yes"].astype(int)

    grouped = candles.sort_values("ts").groupby("market_ticker", sort=False)

    rows: list[dict] = []
    total = 0
    survived = 0

    for _, row in markets.iterrows():
        total += 1
        market_ticker = row["market_ticker"]
        game_start_ts = int(row["game_start_ts"])
        t6h_ts = game_start_ts - 6 * SECONDS_HOUR
        t1h_ts = game_start_ts - SECONDS_HOUR
        t10m_ts = game_start_ts - SECONDS_10M

        if market_ticker not in grouped.groups:
            continue
        group = grouped.get_group(market_ticker)
        group = group.sort_values("ts")

        # p_t6h is the last close at or before t6h (no future candles).
        p_t6h, p_t6h_ts = _last_close_before(group, t6h_ts)
        if not np.isfinite(p_t6h):
            continue

        # p_t1h and p_t10m use last close at or before their cutoffs (no interpolation).
        p_t1h, p_t1h_ts = _last_close_before(group, t1h_ts)
        p_t10m, p_t10m_ts = _last_close_before(group, t10m_ts)

        if p_t1h_ts != -1:
            assert p_t1h_ts <= game_start_ts
        if p_t10m_ts != -1:
            assert p_t10m_ts <= game_start_ts

        post = group[(group["ts"] >= t6h_ts) & group["close"].notna()]
        # Post-t6h stats only use candles with ts >= t6h to avoid lookahead bias.
        p_max_post_t6h = float(post["close"].max()) if not post.empty else np.nan
        p_min_post_t6h = float(post["close"].min()) if not post.empty else np.nan

        high_conf = abs(p_t6h - 0.5) >= 0.30
        is_underdog = p_t6h < 0.5

        rows.append(
            {
                "market_ticker": market_ticker,
                "game_start_ts": game_start_ts,
                "t6h_ts": t6h_ts,
                "p_t6h": p_t6h,
                "is_underdog": int(is_underdog),
                "high_conf": int(high_conf),
                "settle_yes": int(row["settle_yes"]),
                "p_max_post_t6h": p_max_post_t6h,
                "p_min_post_t6h": p_min_post_t6h,
                "p_t1h": p_t1h,
                "p_t10m": p_t10m,
                "max_runup": p_max_post_t6h - p_t6h if np.isfinite(p_max_post_t6h) else np.nan,
                "max_drawdown": p_t6h - p_min_post_t6h if np.isfinite(p_min_post_t6h) else np.nan,
                "repricing_1h": p_t1h - p_t6h if np.isfinite(p_t1h) else np.nan,
                "repricing_10m": p_t10m - p_t6h if np.isfinite(p_t10m) else np.nan,
            }
        )
        survived += 1

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    logger.info("markets_total=%d markets_survived=%d output=%s", total, survived, output_path)


def main() -> None:
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(description="Build NBA low-confidence research dataset.")
    parser.add_argument(
        "--candles",
        default=str(repo_root / "data" / "nba_market_candles_sample.csv"),
        help="Path to 1-minute candle CSV.",
    )
    parser.add_argument(
        "--markets-csv",
        default=str(repo_root / "data" / "nba_event_markets_sample.csv"),
        help="Path to NBA markets CSV with game start times.",
    )
    parser.add_argument(
        "--markets-json",
        default=str(repo_root / "data" / "nba_event_markets_sample.json"),
        help="Path to NBA markets JSON with settlement values.",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root / "data" / "nba_low_conf_dataset.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    build_dataset(
        Path(args.candles),
        Path(args.markets_csv),
        Path(args.markets_json),
        Path(args.output),
    )


if __name__ == "__main__":
    main()
