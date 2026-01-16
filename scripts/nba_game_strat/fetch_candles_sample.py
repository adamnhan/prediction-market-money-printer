from __future__ import annotations

import csv
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys


MAX_EVENTS = None
PERIOD_INTERVAL_MINUTES = 1


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def parse_event_date_from_ticker(event_ticker: str) -> datetime | None:
    match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", event_ticker)
    if not match:
        return None
    year = 2000 + int(match.group(1))
    month = MONTH_MAP.get(match.group(2))
    day = int(match.group(3))
    if not month:
        return None
    return datetime(year, month, day, tzinfo=timezone.utc)


def load_events(csv_path: Path, limit: int | None) -> list[dict]:
    events: list[dict] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_id = (row.get("event_id") or "").strip()
            series_ticker = (row.get("series_ticker") or "").strip()
            if not event_id or not series_ticker:
                continue
            events.append({"event_id": event_id, "series_ticker": series_ticker})
            if limit is not None and len(events) >= limit:
                break
    return events


def load_market_status_map(json_path: Path) -> dict[str, str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    status_map: dict[str, str] = {}
    for markets in data.values():
        for market in markets:
            ticker = market.get("ticker")
            if not ticker:
                continue
            status_map[ticker] = market.get("status", "")
    return status_map


def fetch_event_candles(series_ticker: str, event_ticker: str) -> dict:
    repo_root = get_repo_root()
    sys.path.insert(0, str(repo_root / "kalshi_fetcher"))
    from kalshi_client import request

    event_dt = parse_event_date_from_ticker(event_ticker)
    if not event_dt:
        raise ValueError(f"Could not parse date from event ticker: {event_ticker}")

    start_ts = int((event_dt - timedelta(hours=8)).timestamp())
    end_ts = int((event_dt + timedelta(days=1)).timestamp())

    endpoint = f"/series/{series_ticker}/events/{event_ticker}/candlesticks"
    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": PERIOD_INTERVAL_MINUTES,
    }

    backoff_s = 1.0
    while True:
        try:
            return request(endpoint, params=params)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg:
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 60.0)
                continue
            raise


def candle_yes_price(candle: dict) -> float | None:
    price = candle.get("price", {})
    close_dollars = price.get("close_dollars")
    if close_dollars is not None:
        return float(close_dollars)
    close = price.get("close")
    if close is not None:
        return float(close) / 100.0
    return None


def main() -> None:
    repo_root = get_repo_root()
    csv_path = repo_root / "data" / "nba_event_markets_sample.csv"
    json_path = repo_root / "data" / "nba_event_markets_sample.json"

    events = load_events(csv_path, MAX_EVENTS)
    status_map = load_market_status_map(json_path)

    rows: list[dict] = []

    for event in events:
        event_id = event["event_id"]
        series_ticker = event["series_ticker"]
        data = fetch_event_candles(series_ticker, event_id)
        market_tickers = data.get("market_tickers", [])
        market_candles = data.get("market_candlesticks", [])

        for market_id, candles in zip(market_tickers, market_candles):
            for candle in candles:
                ts = candle.get("end_period_ts")
                yes_price = candle_yes_price(candle)
                no_price = 1.0 - yes_price if yes_price is not None else None
                rows.append(
                    {
                        "market_id": market_id,
                        "ts": ts,
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "volume": candle.get("volume"),
                        "market_status": status_map.get(market_id, ""),
                    }
                )

        time.sleep(0.2)
        print(f"{event_id}: {len(market_tickers)} markets")

    output_csv = repo_root / "data" / "nba_market_candles_sample.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "market_id",
                "ts",
                "yes_price",
                "no_price",
                "volume",
                "market_status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved candle sample to {output_csv}")


if __name__ == "__main__":
    main()
