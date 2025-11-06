# kalshi_fetcher/trades_enrichment.py

import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)

ENRICHED_MARKETS_PATH = os.path.join(BASE_DIR, "..", "data", "enriched_markets.csv")

# ✅ trades are inside kalshi_fetcher/data/trades
TRADES_DIR = os.path.join(BASE_DIR, "data", "trades")
ENRICHED_TRADES_DIR = os.path.join(BASE_DIR, "data", "enriched_trades")


def load_market_event_lookup() -> pd.DataFrame:
    """
    Load enriched_markets.csv and return a DataFrame mapping
    each market ticker to its event_ticker, category, and event_time.
    """
    markets_df = pd.read_csv(ENRICHED_MARKETS_PATH)

    lookup_df = markets_df[[
        "ticker",        # market ticker
        "event_ticker",
        "category",
        "event_time",
    ]].drop_duplicates()

    return lookup_df

def enrich_trades_file(trades_path: str, lookup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a path to a trades CSV and the market→event lookup,
    return an enriched DataFrame with event metadata and
    time deltas (seconds_before_event, minutes_before_event).
    """
    trades_df = pd.read_csv(trades_path)

    # Merge in event_ticker, category, event_time via market ticker
    merged = trades_df.merge(
        lookup_df,
        on="ticker",
        how="left",
    )

    # Parse timestamps as UTC datetimes
    merged["created_time_dt"] = pd.to_datetime(merged["created_time"], utc=True)
    merged["event_time_dt"] = pd.to_datetime(
        merged["event_time"], utc=True, errors="coerce"
    )

    # Compute time deltas
    delta_seconds = (merged["event_time_dt"] - merged["created_time_dt"]).dt.total_seconds()
    merged["seconds_before_event"] = delta_seconds
    merged["minutes_before_event"] = merged["seconds_before_event"] / 60.0

    return merged

def enrich_all_trades():
    """
    Loop over all trades CSVs and write enriched versions
    to ENRICHED_TRADES_DIR with the same filenames.
    """
    lookup_df = load_market_event_lookup()

    os.makedirs(ENRICHED_TRADES_DIR, exist_ok=True)

    for fname in os.listdir(TRADES_DIR):
        if not fname.endswith(".csv"):
            continue

        ticker = fname[:-4]
        trades_path = os.path.join(TRADES_DIR, fname)

        enriched_df = enrich_trades_file(trades_path, lookup_df)

        out_path = os.path.join(ENRICHED_TRADES_DIR, fname)
        enriched_df.to_csv(out_path, index=False)

        print(f"[trades] Enriched {ticker}: {len(enriched_df)} rows -> {out_path}")


def main():
    enrich_all_trades()


if __name__ == "__main__":
    main()


