from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
import sys
import time


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_event_date(event: dict) -> date | None:
    for key in ("event_time", "close_time", "end_time", "start_time"):
        raw = event.get(key)
        if not raw:
            continue
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
        except Exception:
            continue
    return None


def fetch_events(series_ticker: str, target_count: int, status: str) -> list[dict]:
    repo_root = get_repo_root()
    sys.path.insert(0, str(repo_root / "kalshi_fetcher"))
    from kalshi_client import request

    events: list[dict] = []
    cursor = None

    per_page_limit = 100
    backoff_s = 1.0
    while len(events) < target_count:
        limit = min(per_page_limit, target_count - len(events))
        params = {"series_ticker": series_ticker, "limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor
        try:
            data = request("/events", params=params)
            backoff_s = 1.0
        except Exception as exc:
            msg = str(exc)
            if "429" in msg:
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 60.0)
                continue
            raise
        batch = data.get("events", [])
        if not batch:
            break

        events.extend(batch)
        cursor = data.get("cursor")
        if not cursor:
            break

    if not cursor:
        print("Cursor exhausted; no more events available from API.")

    return events[:target_count]


def main() -> None:
    series_ticker = "KXNBAGAME"
    target_count = 1000
    status = "settled"

    events = fetch_events(series_ticker, target_count, status)
    print(f"Fetched {len(events)} events for {series_ticker}")

    repo_root = get_repo_root()
    output_dir = repo_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "nba_events_kxnbagame_1000.json"
    output_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
