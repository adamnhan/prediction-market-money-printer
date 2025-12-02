# kalshi_fetcher/fetch_trades.py

from kalshi_client import request  
import csv
from pathlib import Path

def get_trades_for_market(ticker, since_ts=None, limit=100):
    """
    Fetch ALL trades for a given market ticker using cursor-based pagination.

    Returns:
        List[dict]: list of raw trade dicts from the API.
    """
    all_trades = []
    cursor = None

    while True:
        params = {
            "ticker": ticker,
            "limit": limit,
        }

        if since_ts is not None:
            params["since"] = since_ts

        if cursor is not None:
            params["cursor"] = cursor

        data = request("/markets/trades", params=params)

        # Defensive: if API shape changes, fail gracefully
        page_trades = data.get("trades", [])
        all_trades.extend(page_trades)

        cursor = data.get("cursor")

        # If no cursor is returned, we've reached the last page
        if not cursor:
            break

    return all_trades

def format_trade_row(trade: dict) -> dict:
    """
    Convert a raw trade dict from the API into a clean row for CSV/output.
    Fields:
        - ticker
        - created_time
        - price        (float, 0–1)
        - count        (int, contracts)
        - taker_side   ('yes' or 'no')
        - trade_id
    """
    return {
        "ticker": trade.get("ticker"),
        "created_time": trade.get("created_time"),
        "price": trade.get("price"),
        "count": trade.get("count"),
        "taker_side": trade.get("taker_side"),
        "trade_id": trade.get("trade_id"),
    }

def save_trades_for_market(ticker: str, since_ts: str | None = None, limit: int = 100,
                           out_dir: str = "data/trades") -> None:
    """
    Fetch all trades for a ticker, format them, and save to data/trades/{ticker}.csv.

    Also prints a brief summary:
      - number of trades
      - total contracts traded (sum of 'count')
    """
    trades = get_trades_for_market(ticker, since_ts=since_ts, limit=limit)
    rows = [format_trade_row(t) for t in trades]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / f"{ticker}.csv"

    fieldnames = ["ticker", "created_time", "price", "count", "taker_side", "trade_id"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_trades = len(rows)
    total_contracts = sum((r.get("count") or 0) for r in rows)

    print(f"[summary] {ticker}: wrote {total_trades} trades to {csv_path}")
    print(f"[summary] {ticker}: total contracts traded = {total_contracts}")

def fetch_all_closed_markets_trades(
    closed_markets_csv: str = "data/closed_markets.csv",
    max_markets: int | None = None,
) -> None:
    """
    Read closed_markets.csv, loop over tickers, and fetch/save trades for each.

    Args:
        closed_markets_csv: Path to the CSV of closed/settled markets.
        max_markets: If set, limit to the first N markets (for sanity checks).
    """
    csv_path = Path(closed_markets_csv)
    if not csv_path.exists():
        print(f"[error] closed_markets.csv not found at {csv_path}")
        return

    print(f"[info] Reading closed markets from {csv_path}...")

    processed = 0

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            ticker = row.get("ticker")
            if not ticker:
                continue

            # Volume from closed_markets.csv; may be string, so coerce to int
            volume_str = row.get("volume", "") or "0"
            try:
                volume = int(volume_str)
            except ValueError:
                volume = 0

            # Skip markets that never traded
            if volume <= 0:
                print(f"[skip] {ticker}: volume=0, skipping trade fetch.")
                continue

            print(f"[info] Fetching trades for {ticker} (volume={volume})...")
            save_trades_for_market(ticker)

            processed += 1
            if max_markets is not None and processed >= max_markets:
                break

    print(f"[done] Processed {processed} markets from {csv_path}")


if __name__ == "__main__":
    # Sanity check: run end-to-end on a small subset of closed markets
    # This will:
    #   - read data/closed_markets.csv
    #   - skip markets with volume=0
    #   - fetch & save trades for up to 3 tickers
    fetch_all_closed_markets_trades(
        closed_markets_csv="../data/closed_markets.csv",
        max_markets=5000,
    )



