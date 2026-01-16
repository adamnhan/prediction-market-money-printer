from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import sys


EXCLUDE_KEYWORDS = (
    "over",
    "under",
    "total",
    "spread",
)

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


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def is_winner_yes_no_market(market: dict) -> bool:
    text = " ".join(
        str(market.get(k, "") or "")
        for k in ("title", "subtitle", "sub_title", "ticker")
    ).lower()
    if any(keyword in text for keyword in EXCLUDE_KEYWORDS):
        return False
    return True


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


def season_from_date(dt: datetime | None) -> str:
    if not dt:
        return ""
    if dt.month >= 7:
        return f"{dt.year}-{dt.year + 1}"
    return f"{dt.year - 1}-{dt.year}"


def load_events(path: Path, count: int | None) -> list[dict]:
    events = json.loads(path.read_text(encoding="utf-8"))
    tickers: list[str] = []
    selected: list[dict] = []
    for event in events:
        ticker = event.get("event_ticker")
        if ticker:
            tickers.append(ticker)
            selected.append(event)
        if count is not None and len(tickers) >= count:
            break
    return selected


def fetch_markets_for_event(event_ticker: str) -> list[dict]:
    repo_root = get_repo_root()
    sys.path.insert(0, str(repo_root / "kalshi_fetcher"))
    from kalshi_client import request

    params = {"event_ticker": event_ticker, "limit": 200}
    backoff_s = 1.0
    while True:
        try:
            data = request("/markets", params=params)
            return data.get("markets", [])
        except Exception as exc:
            msg = str(exc)
            if "429" in msg:
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 60.0)
                continue
            raise


def main() -> None:
    repo_root = get_repo_root()
    events_path = repo_root / "data" / "nba_events_kxnbagame_1000.json"
    events = load_events(events_path, None)

    results: dict[str, list[dict]] = {}
    rows: list[dict] = []

    for idx, event in enumerate(events, start=1):
        event_ticker = event.get("event_ticker", "")
        markets = fetch_markets_for_event(event_ticker)
        time.sleep(0.2)
        filtered = [m for m in markets if is_winner_yes_no_market(m)]
        results[event_ticker] = filtered

        event_dt = parse_event_date_from_ticker(event_ticker)
        event_start_ts = int(event_dt.timestamp()) if event_dt else ""
        event_date = event_dt.date().isoformat() if event_dt else ""
        season = season_from_date(event_dt)
        series_ticker = event.get("series_ticker") or "KXNBAGAME"
        event_status = filtered[0].get("status") if filtered else ""

        unique_markets: list[dict] = []
        seen_labels = set()
        for m in filtered:
            label = (m.get("yes_sub_title") or m.get("title") or "").strip()
            if not label or label in seen_labels:
                continue
            seen_labels.add(label)
            unique_markets.append(m)
            if len(unique_markets) >= 2:
                break

        market_1 = unique_markets[0] if len(unique_markets) > 0 else {}
        market_2 = unique_markets[1] if len(unique_markets) > 1 else {}

        game_id = hashlib.sha1(event_ticker.encode("utf-8")).hexdigest()
        rows.append(
            {
                "game_id": game_id,
                "series_ticker": series_ticker.lower(),
                "event_id": event_ticker,
                "event_start_ts": event_start_ts,
                "event_date": event_date,
                "market_id_1": market_1.get("ticker", ""),
                "market_label_1": market_1.get("yes_sub_title") or market_1.get("title", ""),
                "market_id_2": market_2.get("ticker", ""),
                "market_label_2": market_2.get("yes_sub_title") or market_2.get("title", ""),
                "event_status": event_status,
                "league": "NBA",
                "season": season,
            }
        )

        print(f"{event_ticker}: {len(filtered)} winner-style markets")

    output_json = repo_root / "data" / "nba_event_markets_sample.json"
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved sample markets to {output_json}")

    output_csv = repo_root / "data" / "nba_event_markets_sample.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "game_id",
                "series_ticker",
                "event_id",
                "event_start_ts",
                "event_date",
                "market_id_1",
                "market_label_1",
                "market_id_2",
                "market_label_2",
                "event_status",
                "league",
                "season",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV to {output_csv}")


if __name__ == "__main__":
    main()
