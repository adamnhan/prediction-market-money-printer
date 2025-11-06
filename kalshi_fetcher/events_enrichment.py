# kalshi_fetcher/events_enrichment.py

import os
import pandas as pd

from kalshi_client import get_event_metadata  # we'll use this in the next steps

# Base dir = kalshi_fetcher/
BASE_DIR = os.path.dirname(__file__)

# ../data/closed_markets.csv relative to kalshi_fetcher/
CLOSED_MARKETS_PATH = os.path.join(BASE_DIR, "..", "data", "closed_markets.csv")
EVENTS_OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "events.csv")


def load_unique_event_tickers() -> list[str]:
    """
    Load closed_markets.csv and return a list of unique event_ticker values.
    """
    df = pd.read_csv(CLOSED_MARKETS_PATH)
    event_tickers = (
        df["event_ticker"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return event_tickers

def fetch_all_event_metadata(event_tickers: list[str]) -> pd.DataFrame:
    """
    For each event_ticker:
      - call get_event_metadata(et)
      - compute event_time proxy as the latest close_time of its markets
      - return a DataFrame with one row per event
    """
    # Load closed_markets so we can compute event_time from close_time
    closed_df = pd.read_csv(CLOSED_MARKETS_PATH)

    rows = []

    for et in event_tickers:
        try:
            event = get_event_metadata(et)
        except Exception as e:
            print(f"[events] Error fetching event {et}: {e}")
            continue

        # Filter closed_markets for this event_ticker and take the latest close_time
        event_rows = closed_df[closed_df["event_ticker"] == et]
        if not event_rows.empty:
            event_time = event_rows["close_time"].max()
        else:
            event_time = None

        rows.append(
            {
                "event_ticker": event.get("event_ticker", et),
                "title": event.get("title"),
                "category": event.get("category"),
                "series_ticker": event.get("series_ticker"),
                "event_time": event_time,
            }
        )

    return pd.DataFrame(rows)

def main():
    event_tickers = load_unique_event_tickers()
    print(f"[events] Found {len(event_tickers)} unique event_ticker values")

    df_events = fetch_all_event_metadata(event_tickers)
    print(f"[events] Retrieved metadata for {len(df_events)} events")

    # Ensure ../data exists and save events.csv there
    os.makedirs(os.path.dirname(EVENTS_OUTPUT_PATH), exist_ok=True)
    df_events.to_csv(EVENTS_OUTPUT_PATH, index=False)
    print(f"[events] Saved events to {EVENTS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
