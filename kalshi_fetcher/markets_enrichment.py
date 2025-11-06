# kalshi_fetcher/markets_enrichment.py

import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)

CLOSED_MARKETS_PATH = os.path.join(BASE_DIR, "..", "data", "closed_markets.csv")
EVENTS_PATH = os.path.join(BASE_DIR, "..", "data", "events.csv")
ENRICHED_MARKETS_PATH = os.path.join(BASE_DIR, "..", "data", "enriched_markets.csv")


def build_enriched_markets() -> pd.DataFrame:
    """
    Merge closed_markets.csv with events.csv on event_ticker
    to produce enriched_markets.csv.
    """
    markets_df = pd.read_csv(CLOSED_MARKETS_PATH)
    events_df = pd.read_csv(EVENTS_PATH)

    enriched_df = markets_df.merge(
        events_df,
        on="event_ticker",
        how="left",
        suffixes=("", "_event"),
    )

    return enriched_df


def main():
    enriched_df = build_enriched_markets()
    print(f"[markets] Enriched markets rows: {len(enriched_df)}")

    os.makedirs(os.path.dirname(ENRICHED_MARKETS_PATH), exist_ok=True)
    enriched_df.to_csv(ENRICHED_MARKETS_PATH, index=False)
    print(f"[markets] Saved enriched markets to {ENRICHED_MARKETS_PATH}")


if __name__ == "__main__":
    main()
