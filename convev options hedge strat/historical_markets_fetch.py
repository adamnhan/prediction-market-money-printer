#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

import requests


DEFAULT_REST_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_SERIES = ["KXNBAGAME", "KXNBASPREAD", "KXNBATOTAL"]
DEFAULT_STATUSES = ["settled"]


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _event_key_from_market_ticker(ticker: str | None) -> str | None:
    if not ticker:
        return None
    parts = ticker.split("-")
    if len(parts) < 3:
        return None
    return parts[1]


def _fetch_markets(
    rest_url: str,
    api_key: str | None,
    series_ticker: str,
    statuses: list[str],
    limit: int = 200,
    max_pages: int = 200,
) -> list[dict[str, Any]]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    markets: list[dict[str, Any]] = []
    status_filter_supported = True
    for status in statuses or [None]:
        cursor = None
        pages = 0
        while pages < max_pages:
            params = {"series_ticker": series_ticker, "limit": limit}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            response = requests.get(f"{rest_url}/markets", params=params, headers=headers, timeout=20)
            if response.status_code == 400 and status:
                print(f"[historical] status filter not supported for '{status}', falling back to local filter")
                status_filter_supported = False
                cursor = None
                pages = 0
                break
            if response.status_code == 429:
                time.sleep(2)
                continue
            response.raise_for_status()
            payload = response.json() or {}
            batch = payload.get("markets") or []
            for m in batch:
                markets.append(m)
            cursor = payload.get("cursor") or payload.get("next_cursor")
            pages += 1
            if not cursor:
                break
        if not status_filter_supported:
            break
    if not status_filter_supported:
        markets = []
        cursor = None
        pages = 0
        while pages < max_pages:
            params = {"series_ticker": series_ticker, "limit": limit}
            if cursor:
                params["cursor"] = cursor
            response = requests.get(f"{rest_url}/markets", params=params, headers=headers, timeout=20)
            if response.status_code == 429:
                time.sleep(2)
                continue
            response.raise_for_status()
            payload = response.json() or {}
            batch = payload.get("markets") or []
            for m in batch:
                status = str(m.get("status") or "").lower()
                if not statuses or status in statuses:
                    markets.append(m)
            cursor = payload.get("cursor") or payload.get("next_cursor")
            pages += 1
            if not cursor:
                break
    return markets


def _select_events(
    markets_by_event: dict[str, dict[str, list[dict[str, Any]]]],
    target_events: int,
) -> list[str]:
    candidates = []
    for event_ticker, series_map in markets_by_event.items():
        if all(series in series_map and series_map[series] for series in DEFAULT_SERIES):
            event_time = None
            for series in DEFAULT_SERIES:
                if series_map[series]:
                    event_time = _parse_iso(series_map[series][0].get("event_start_time") or series_map[series][0].get("open_time"))
                    if event_time:
                        break
            candidates.append((event_time, event_ticker))
    candidates.sort(key=lambda x: x[0] or datetime.min)
    return [event for _, event in candidates[:target_events]]


def fetch_historical(
    rest_url: str,
    api_key: str | None,
    target_events: int,
    statuses: list[str],
    out_csv: str,
    out_jsonl: str | None,
) -> int:
    markets_by_event: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for series in DEFAULT_SERIES:
        markets = _fetch_markets(rest_url, api_key, series, statuses)
        print(f"[historical] series={series} fetched={len(markets)}")
        for m in markets:
            event_key = _event_key_from_market_ticker(m.get("ticker") or m.get("market_ticker"))
            event_ticker = m.get("event_ticker")
            key = event_key or event_ticker
            if not key:
                continue
            markets_by_event[key][series].append(m)
    selected_events = _select_events(markets_by_event, target_events)
    print(f"[historical] events_with_all_series={len(selected_events)}")
    if not selected_events:
        return 0
    rows = 0
    with open(out_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "event_ticker",
                "series_ticker",
                "market_ticker",
                "status",
                "event_start_time",
                "close_time",
                "title",
                "subtitle",
                "rules_text",
                "tick_size",
                "event_key",
            ]
        )
        for event_key in selected_events:
            for series in DEFAULT_SERIES:
                for m in markets_by_event[event_key].get(series, []):
                    writer.writerow(
                        [
                            m.get("event_ticker") or event_key,
                            series,
                            m.get("ticker") or m.get("market_ticker"),
                            m.get("status"),
                            m.get("event_start_time") or m.get("open_time"),
                            m.get("close_time"),
                            m.get("title"),
                            m.get("subtitle"),
                            m.get("rules_text") or m.get("rules"),
                            m.get("tick_size"),
                            event_key,
                        ]
                    )
                    rows += 1
    if out_jsonl:
        with open(out_jsonl, "w", encoding="utf-8") as handle:
            for event_key in selected_events:
                for series in DEFAULT_SERIES:
                    for m in markets_by_event[event_key].get(series, []):
                        handle.write(json.dumps(m, separators=(",", ":"), ensure_ascii=True))
                        handle.write("\n")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical NBA markets (moneyline, spread, total).")
    parser.add_argument("--rest-url", default=DEFAULT_REST_URL)
    parser.add_argument("--api-key", default=os.getenv("KALSHI_API_KEY"))
    parser.add_argument("--events", type=int, default=100)
    parser.add_argument("--statuses", default=",".join(DEFAULT_STATUSES))
    parser.add_argument("--out-csv", default="data/nba_historical_markets.csv")
    parser.add_argument("--out-jsonl", default="data/nba_historical_markets.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    statuses = [s.strip().lower() for s in (args.statuses or "").split(",") if s.strip()]
    rows = fetch_historical(
        rest_url=args.rest_url,
        api_key=args.api_key,
        target_events=args.events,
        statuses=statuses,
        out_csv=args.out_csv,
        out_jsonl=args.out_jsonl,
    )
    print(f"[historical] wrote {rows} rows to {args.out_csv}")
    if args.out_jsonl:
        print(f"[historical] wrote raw jsonl to {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
