# kalshi_fetcher/closed_markets.py

from pathlib import Path
import csv
import time
from collections import Counter

from kalshi_client import request  # uses the helper you built in Step 1

# Folder + file where we'll store closed markets
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_PATH = DATA_DIR / "closed_markets.csv"

STATUS_MAP = {
    # What the API might return  ->  How we want to label it
    "closed": "closed",
    "settled": "settled",
    "determined": "closed",   # result known, not fully settled yet
    "finalized": "settled",   # fully finalized/payout complete
}



def fetch_closed_markets(limit: int = 1000):
    """
    Fetch markets from Kalshi that:
      - have status closed or settled (server-side)
      - have close_time <= now via max_close_ts
    """
    now_unix = int(time.time())

    params = {
        "status": "closed,settled",  # comma-separated list per Kalshi docs
        "limit": limit,
        "max_close_ts": now_unix,    # only markets that close <= now
        # you can also add:
        # "mve_filter": "exclude",   # to skip multivariate combos if desired
    }

    response = request("/markets", params=params)

    markets = response.get("markets", [])

    # Quick debug: what statuses did we actually get back?
    status_counts = Counter(m.get("status") for m in markets)
    print(f"[closed_markets] Raw status counts from API: {status_counts}")
    print(f"[closed_markets] Fetched {len(markets)} markets from API (with filters)")

    return markets



def filter_closed_or_settled(markets):
    """
    Keep only markets whose status corresponds to closed/settled.
    Normalize raw statuses (e.g. 'determined', 'finalized') into
    canonical values: 'closed' or 'settled'.
    """
    filtered = []

    for m in markets:
        raw_status = m.get("status")
        canonical_status = STATUS_MAP.get(raw_status)

        # Skip anything that isn't in our mapping
        if canonical_status is None:
            continue

        # Make a shallow copy so we don't mutate the original dict in-place
        m_copy = dict(m)
        m_copy["status"] = canonical_status
        filtered.append(m_copy)

    print(
        f"[closed_markets] After status filter: {len(filtered)} / {len(markets)} markets "
        f"are mapped to closed/settled"
    )
    return filtered



def save_closed_markets_csv(markets):
    """
    Save selected fields of closed markets to CSV at data/closed_markets.csv.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = ["ticker", "event_ticker", "status", "close_time", "volume"]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(markets)

    tickers_preview = [m.get("ticker") for m in markets[:5]]
    print(f"Saved {len(markets)} closed/settled markets to {CSV_PATH}")
    if tickers_preview:
        print("First few tickers:", ", ".join(tickers_preview))


def extract_market_fields(markets):
    """
    Return a simplified list of closed markets with only the relevant fields.
    """
    simplified = []
    for m in markets:
        simplified.append({
            "ticker": m.get("ticker"),
            "event_ticker": m.get("event_ticker"),
            "status": m.get("status"),
            "close_time": m.get("close_time"),
            "volume": m.get("volume"),
        })
    return simplified



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch closed/settled markets from Kalshi")
    parser.add_argument("--limit", type=int, default=200, help="Number of markets to fetch (default 100)")
    args = parser.parse_args()

    markets = fetch_closed_markets(limit=args.limit)
    closed_markets = filter_closed_or_settled(markets)
    clean_markets = extract_market_fields(closed_markets)
    save_closed_markets_csv(clean_markets)
    print(f"Extracted {len(clean_markets)} closed/settled markets (fields cleaned).")





if __name__ == "__main__":
    main()
