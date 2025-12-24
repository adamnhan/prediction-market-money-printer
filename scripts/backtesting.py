import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import random


import pandas as pd
import requests

# --- Config ---
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_TICKERS = [
    "kxsbads",
    "kxsongsoncharttswift",
    "kxemmydseries",
    "kxrtcaptainamerica",
    "kxrtmoana2",
    "kxrtmickey17",
    "kxrtsnowwhite",
    "kxrtmufasa",
    "kxrtminecraft",
    "kxsongsoncharttswift2",
    "kxrtthemonkey",
    "kxrtnovocaine",
    "kxrtfantasticfour",
    "kxsongsoncharttswift6",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRADES_DIR = DATA_DIR / "series_trades"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRADES_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
DEFAULT_HEADERS: Dict[str, str] = {}
TIMEOUT = 30

STATUS_MAP = {
    "closed": "closed",
    "settled": "settled",
    "determined": "closed",
    "finalized": "settled",
}

# --- Rate limiting knobs (conservative) ---
MIN_SECONDS_BETWEEN_CALLS = 0.12   # ~8.3 req/sec (well under Basic 20/sec)
MAX_RETRIES = 8

_last_call_ts = 0.0

def _throttle():
    global _last_call_ts
    now = time.time()
    wait = MIN_SECONDS_BETWEEN_CALLS - (now - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.time()

def get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{BASE_URL}{path}"
    params = params or {}

    for attempt in range(MAX_RETRIES):
        _throttle()

        resp = SESSION.get(url, params=params, headers=DEFAULT_HEADERS, timeout=TIMEOUT)

        # Handle 429 with Retry-After (if provided) + exponential backoff
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    sleep_s = float(retry_after)
                except ValueError:
                    sleep_s = 1.0
            else:
                sleep_s = min(30.0, (2 ** attempt) * 0.5)  # 0.5,1,2,4,... capped
            sleep_s += random.uniform(0, 0.25)  # small jitter
            print(f"[RATE_LIMIT] 429 on {path} params={params}. sleeping {sleep_s:.2f}s (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(sleep_s)
            continue

        resp.raise_for_status()
        return resp.json()

    raise RuntimeError(f"Exceeded retries due to rate limiting for {url} params={params}")


def fetch_markets_for_series(series_ticker: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Correct v2 approach:
      - Use GET /markets
      - Filter by series_ticker
      - Only one status allowed per request -> fetch closed and settled separately
    """
    st = series_ticker.upper()
    markets: List[Dict[str, Any]] = []

    for status in ["closed", "settled"]:
        cursor: Optional[str] = None
        while True:
            params: Dict[str, Any] = {
                "series_ticker": st,
                "status": status,     # only one status per request
                "limit": limit,
            }
            if cursor:
                params["cursor"] = cursor

            data = get("/markets", params=params)  # ✅ correct endpoint
            markets.extend(data.get("markets", []))
            cursor = data.get("cursor")
            if not cursor:
                break

    return markets

def fetch_trades_for_market(ticker: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    v2 trades endpoint: GET /markets/trades (cursor paginated).
    """
    trades: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        params: Dict[str, Any] = {"ticker": ticker, "limit": limit}
        if cursor:
            params["cursor"] = cursor

        data = get("/markets/trades", params=params)  # ✅ correct endpoint
        trades.extend(data.get("trades", []))
        cursor = data.get("cursor")
        if not cursor:
            break

    return trades

def save_trades_csv(ticker: str, trades: List[Dict[str, Any]]) -> Path:
    path = TRADES_DIR / f"{ticker}.csv"
    if trades:
        pd.DataFrame(trades).to_csv(path, index=False)
        print(f"[{ticker}] saved {len(trades)} trades -> {path}")
    else:
        print(f"[{ticker}] no trades to save")
    return path

@dataclass
class StrategyParams:
    max_no_entry_price: Optional[float] = None
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    max_hold_seconds: Optional[int] = None

def load_trades_with_result(ticker: str, series_markets_df: pd.DataFrame):
    path = TRADES_DIR / f"{ticker}.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    if df.empty:
        return None, None

    df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce", utc=True)
    df = df.sort_values("created_time").reset_index(drop=True)

    # Your trade schema assumption (keep as-is)
    df["price_yes"] = df["yes_price_dollars"]
    df["price_no"] = 1.0 - df["price_yes"]

    meta = series_markets_df.loc[series_markets_df["market_ticker"] == ticker]
    result_str = str(meta.iloc[0]["result"]).lower() if not meta.empty else None
    return df, result_str

def simulate_no_with_params(ticker: str, params: StrategyParams, series_markets_df: pd.DataFrame, qty: int = 1):
    trades_df, result_str = load_trades_with_result(ticker, series_markets_df)
    if trades_df is None:
        return {"market_ticker": ticker, "had_trade": False, "pnl": 0.0, "reason": "no_trades"}

    candidates = trades_df
    if params.max_no_entry_price is not None:
        candidates = candidates[candidates["price_no"] <= params.max_no_entry_price]
    if candidates.empty:
        return {"market_ticker": ticker, "had_trade": False, "pnl": 0.0, "reason": "entry_filtered"}

    entry = candidates.iloc[0]
    entry_time = entry["created_time"]
    entry_price_no = float(entry["price_no"])
    entry_value = entry_price_no * qty

    tp_val = None if params.take_profit_pct is None else params.take_profit_pct * entry_value
    sl_val = None if params.stop_loss_pct is None else params.stop_loss_pct * entry_value

    exit_price_no = None
    exit_time = None
    exit_reason = None

    for _, row in trades_df[trades_df["created_time"] >= entry_time].iterrows():
        curr_no = float(row["price_no"])
        pnl_running = (entry_price_no - curr_no) * qty

        if tp_val is not None and pnl_running >= tp_val:
            exit_price_no = curr_no
            exit_time = row["created_time"]
            exit_reason = "take_profit"
            break

        if sl_val is not None and pnl_running <= sl_val:
            exit_price_no = curr_no
            exit_time = row["created_time"]
            exit_reason = "stop_loss"
            break

        if params.max_hold_seconds is not None:
            age = (row["created_time"] - entry_time).total_seconds()
            if age >= params.max_hold_seconds:
                exit_price_no = curr_no
                exit_time = row["created_time"]
                exit_reason = "time_expired"
                break

    if exit_price_no is None:
        payoff_yes = 1.0 if result_str == "yes" else 0.0 if result_str == "no" else float(trades_df.iloc[-1]["price_yes"])
        payoff_no = 1.0 - payoff_yes
        exit_price_no = payoff_no
        exit_time = trades_df.iloc[-1]["created_time"]
        exit_reason = "settlement"

    pnl = (entry_price_no - exit_price_no) * qty
    return {
        "market_ticker": ticker,
        "had_trade": True,
        "pnl": pnl,
        "reason": exit_reason,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price_no": entry_price_no,
        "exit_price_no": exit_price_no,
        "result": result_str,
        "params": asdict(params),
    }

def backtest_params(params: StrategyParams, tickers: List[str], series_markets_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([simulate_no_with_params(t, params, series_markets_df) for t in tickers])

def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    trades = df[df["had_trade"]]
    if trades.empty:
        return {"num_trades": 0, "mean_pnl": 0.0, "median_pnl": 0.0, "win_rate": 0.0, "min_pnl": 0.0, "max_pnl": 0.0}
    pnls = trades["pnl"]
    return {
        "num_trades": len(trades),
        "mean_pnl": pnls.mean(),
        "median_pnl": pnls.median(),
        "win_rate": (pnls > 0).mean(),
        "min_pnl": pnls.min(),
        "max_pnl": pnls.max(),
    }

def main():
    # 1) Fetch markets for all series
    all_rows = []
    for series in SERIES_TICKERS:
        mkt_raw = fetch_markets_for_series(series)
        for m in mkt_raw:
            normalized_status = STATUS_MAP.get(m.get("status"), m.get("status"))
            all_rows.append({
                "series_ticker": series,
                "market_ticker": m.get("ticker"),
                "event_ticker": m.get("event_ticker"),
                "status": m.get("status"),
                "normalized_status": normalized_status,
                "title": m.get("title"),
                "result": m.get("result"),
            })

    series_markets_df = pd.DataFrame(all_rows)
    unique_tickers = series_markets_df["market_ticker"].dropna().unique().tolist()
    print("Total markets:", len(series_markets_df))
    print("Tickers to fetch:", len(unique_tickers))

    # 2) Fetch trades for each market
    for t in unique_tickers:
        trades = fetch_trades_for_market(t, limit=1000)
        save_trades_csv(t, trades)

    # 3) Grid search
    results = []
    for max_no, tp, sl, hold in itertools.product(
        [0.25, 0.35, 0.45, 0.55, 0.65, None],
        [None, 0.05, 0.10, 0.20],
        [None, -0.05, -0.10, -0.20],
        [6*3600, 24*3600, 3*24*3600, None],
    ):
        params = StrategyParams(max_no_entry_price=max_no, take_profit_pct=tp, stop_loss_pct=sl, max_hold_seconds=hold)
        df = backtest_params(params, unique_tickers, series_markets_df)
        summary = summarize(df)
        summary.update(asdict(params))
        results.append(summary)

    grid_df = pd.DataFrame(results).sort_values(["mean_pnl", "win_rate"], ascending=[False, False]).reset_index(drop=True)
    print("Top 5 configs:")
    print(grid_df.head())

    # 4) Best params + save
    best_row = grid_df.iloc[0].to_dict()
    best_params = StrategyParams(
        max_no_entry_price=best_row["max_no_entry_price"],
        take_profit_pct=best_row["take_profit_pct"],
        stop_loss_pct=best_row["stop_loss_pct"],
        max_hold_seconds=best_row["max_hold_seconds"],
    )
    best_df = backtest_params(best_params, unique_tickers, series_markets_df)
    best_summary = summarize(best_df)
    best_payload = asdict(best_params)
    best_payload.update(best_summary)

    out_path = DATA_DIR / "best_strategy_config.json"
    out_path.write_text(json.dumps(best_payload, indent=2))
    print("Saved best config to", out_path)
    print(best_payload)

if __name__ == "__main__":
    main()
