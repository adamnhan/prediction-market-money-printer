# kalshi_fetcher/closed_markets.py

from __future__ import annotations

from pathlib import Path
import csv
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from kalshi_client import request, get_event_metadata  # NOTE: get_event_metadata already used elsewhere

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
    Fetch up to `limit` markets from Kalshi that are closed/settled/determined.

    Uses pagination with `cursor` and respects the API's max page size of 1000.
    """
    print(f"[closed_markets] Fetching up to {limit} markets from /markets...")

    markets = []
    remaining = limit
    cursor = None

    while remaining > 0:
        page_limit = min(remaining, 1000)  # API max is 1000 per page

        params = {
            "limit": page_limit,
            # filter on server side to only get done markets
            "status": "closed,settled",
        }
        if cursor:
            params["cursor"] = cursor

        data = request("/markets", params=params)
        batch = data.get("markets", [])
        markets.extend(batch)

        print(f"[closed_markets] Got {len(batch)} markets in this page "
              f"(total so far: {len(markets)})")

        remaining -= len(batch)

        cursor = data.get("cursor")
        if not cursor or not batch:
            # no more pages
            break

    # Just for debugging: show raw statuses from API
    status_counts = Counter(m.get("status") for m in markets)
    print(f"[closed_markets] Raw status counts from API: {status_counts}")

    return markets



def filter_closed_or_settled(markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only markets whose status can be mapped to 'closed' or 'settled'
    using STATUS_MAP.
    """
    filtered: List[Dict[str, Any]] = []
    for m in markets:
        raw_status = m.get("status")
        label = STATUS_MAP.get(raw_status)
        if label is None:
            continue  # skip unknown statuses
        m = dict(m)  # shallow copy to avoid mutating original
        m["normalized_status"] = label
        filtered.append(m)

    print(
        f"[closed_markets] After status filter: {len(filtered)} / {len(markets)} markets "
        f"are closed/settled"
    )
    return filtered


def enrich_with_categories(markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each unique event_ticker, fetch event metadata and attach 'category'
    to each market.
    """
    # Collect unique event_tickers
    event_tickers = sorted({m.get("event_ticker") for m in markets if m.get("event_ticker")})
    print(f"[closed_markets] Unique event_tickers: {len(event_tickers)}")

    event_meta: Dict[str, Dict[str, Any]] = {}

    for et in event_tickers:
        try:
            meta = get_event_metadata(et)  # expected to return a dict with at least 'ticker' and 'category'
        except Exception as e:
            print(f"[warn] Failed to fetch metadata for {et}: {e}")
            meta = {}
        event_meta[et] = meta

    enriched: List[Dict[str, Any]] = []
    for m in markets:
        et = m.get("event_ticker")
        meta = event_meta.get(et, {})
        category = meta.get("category")
        # Attach category to each market (even if None)
        m2 = dict(m)
        m2["category"] = category
        enriched.append(m2)

    return enriched


def filter_by_result(markets: List[Dict[str, Any]], result_filter: str) -> List[Dict[str, Any]]:
    """
    Filter markets by result.

    result_filter:
      - 'both' : no filter
      - 'yes'  : only result == 'yes'
      - 'no'   : only result == 'no'
    """
    if result_filter == "both":
        return markets

    allowed = result_filter.lower()
    filtered = [m for m in markets if (m.get("result") or "").lower() == allowed]
    print(
        f"[closed_markets] After result filter ({result_filter}): "
        f"{len(filtered)} / {len(markets)} markets"
    )
    return filtered


def sample_by_category(
    markets: List[Dict[str, Any]],
    categories: Optional[List[str]],
    per_category_limit: Optional[int],
    sample_mode: str = "balanced",  # 'high', 'low', or 'balanced'
) -> List[Dict[str, Any]]:
    """
    Optionally filter to a subset of categories and take up to N markets per category.

    sample_mode:
      - 'high'      : top-N by volume (descending)
      - 'low'       : bottom-N by volume (ascending)
      - 'balanced'  : 50/50 low-volume & high-volume per category
    """
    # Normalize category names for matching
    if categories is not None:
        target_set = {c.strip() for c in categories if c.strip()}
        print(f"[closed_markets] Requested categories: {sorted(target_set)}")
    else:
        target_set = None
        print("[closed_markets] No category filter requested (keeping all categories)")

    # Group by category
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in markets:
        cat = m.get("category") or "Unknown"
        if target_set is not None and cat not in target_set:
            continue
        groups[cat].append(m)

    selected: List[Dict[str, Any]] = []

    for cat, rows in groups.items():
        # If no per-category limit, just keep everything in this category
        if per_category_limit is None or per_category_limit >= len(rows):
            print(f"[closed_markets] Category '{cat}': keeping all {len(rows)} markets")
            selected.extend(rows)
            continue

        # Common sorted lists
        rows_by_vol_asc = sorted(rows, key=lambda r: (r.get("volume") or 0))
        rows_by_vol_desc = list(reversed(rows_by_vol_asc))

        if sample_mode == "high":
            chosen = rows_by_vol_desc[:per_category_limit]

        elif sample_mode == "low":
            chosen = rows_by_vol_asc[:per_category_limit]

        elif sample_mode == "balanced":
            # 50/50 split between low-volume and high-volume
            low_count = per_category_limit // 2
            high_count = per_category_limit - low_count

            low_part = rows_by_vol_asc[:low_count]
            # Remove those low-part rows before selecting high-part, to avoid duplicates
            remaining = rows_by_vol_asc[low_count:]
            remaining_desc = list(reversed(remaining))
            high_part = remaining_desc[:high_count]

            chosen = low_part + high_part

        else:
            # Fallback: default to high-volume if an unknown mode is passed
            print(f"[closed_markets] Unknown sample_mode '{sample_mode}', defaulting to 'high'")
            chosen = rows_by_vol_desc[:per_category_limit]

        print(f"[closed_markets] Category '{cat}': keeping {len(chosen)} markets (mode={sample_mode})")
        selected.extend(chosen)

    print(
        f"[closed_markets] After category + per-category-limit + sample-mode filter: "
        f"{len(selected)} markets total"
    )
    return selected


def save_closed_markets_csv(markets: List[Dict[str, Any]]) -> None:
    """
    Save simplified fields to CSV_PATH.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Define the fields you care about downstream
    fieldnames = [
        "ticker",
        "event_ticker",
        "category",
        "normalized_status",
        "status",
        "close_time",
        "volume",
        "last_price",
        # outcome-related fields
        "result",
        "settlement_value",
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for m in markets:
            writer.writerow({
                "ticker": m.get("ticker"),
                "event_ticker": m.get("event_ticker"),
                "category": m.get("category"),
                "normalized_status": m.get("normalized_status"),
                "status": m.get("status"),
                "close_time": m.get("close_time"),
                "volume": m.get("volume"),
                "last_price": m.get("last_price"),
                "result": m.get("result"),
                "settlement_value": m.get("settlement_value"),
            })

    print(f"[closed_markets] Saved {len(markets)} markets to {CSV_PATH}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch closed/settled markets from Kalshi")

    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Number of markets to fetch from API before filtering (default 2000)",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Comma-separated list of categories to keep (e.g. 'Sports,Culture'). "
             "Default: keep all categories.",
    )
    parser.add_argument(
        "--per-category-limit",
        type=int,
        default=None,
        help="Maximum number of markets to keep per category (after filtering). "
             "Default: no per-category cap.",
    )
    parser.add_argument(
        "--result",
        choices=["both", "yes", "no"],
        default="both",
        help="Filter markets by resolved result. Default: both.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["high", "low", "balanced"],
        default="balanced",
        help="How to select within each category when per-category-limit is set. "
             "'high' = highest volume, 'low' = lowest volume, 'balanced' = 50/50 "
             "low & high volume (default).",
    )

    args = parser.parse_args()

    markets = fetch_closed_markets(limit=args.limit)
    closed_markets = filter_closed_or_settled(markets)
    closed_markets = enrich_with_categories(closed_markets)
    closed_markets = filter_by_result(closed_markets, args.result)

    # After result filter
    cat_counts = Counter(m.get("category") or "Unknown" for m in closed_markets)
    print("[debug] Categories after filters (result + status):")
    for cat, cnt in cat_counts.items():
        print(f"  {cat}: {cnt}")


    if args.categories:
        category_list = [c.strip() for c in args.categories.split(",") if c.strip()]
    else:
        category_list = None

    closed_markets = sample_by_category(
        closed_markets,
        categories=category_list,
        per_category_limit=args.per_category_limit,
        sample_mode=args.sample_mode,
    )

    save_closed_markets_csv(closed_markets)
    print(f"[closed_markets] Done. Final markets count: {len(closed_markets)}")

if __name__ == "__main__":
    main()
