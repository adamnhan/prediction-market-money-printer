#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable

import requests


def _utc_ms() -> int:
    return int(time.time() * 1000)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_enriched_path() -> str:
    return os.path.join(_repo_root(), "data", "enriched_markets.csv")


def _default_output_path() -> str:
    return os.path.join(_repo_root(), "data", "l1_snapshots.jsonl")


def _parse_event_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        cleaned = value.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _load_tickers(
    enriched_path: str,
    series_prefixes: list[str] | None,
) -> list[dict]:
    tickers: list[str] = []
    rows: list[dict] = []
    with open(enriched_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker = row.get("ticker") or ""
            if not ticker:
                continue
            series = row.get("series_ticker") or ""
            if series_prefixes and series not in series_prefixes:
                continue
            if ticker not in tickers:
                tickers.append(ticker)
                rows.append(
                    {
                        "ticker": ticker,
                        "series_ticker": series,
                        "event_time": _parse_event_time(row.get("event_time")),
                    }
                )
    return rows


def _parse_levels(payload: dict) -> tuple[list[list[int]], list[list[int]]]:
    if "orderbook" in payload and isinstance(payload["orderbook"], dict):
        ob = payload["orderbook"]
    else:
        ob = payload
    yes = ob.get("yes") or ob.get("yes_bids") or []
    no = ob.get("no") or ob.get("no_asks") or []
    yes_levels = [[int(p), int(s)] for p, s in yes if p is not None and s is not None]
    no_levels = [[int(p), int(s)] for p, s in no if p is not None and s is not None]
    return yes_levels, no_levels


def _top_prices(yes_levels: list[list[int]], no_levels: list[list[int]]) -> tuple[int | None, int | None]:
    top_yes = max((p for p, _ in yes_levels), default=None)
    top_no = min((p for p, _ in no_levels), default=None)
    return top_yes, top_no


def _filter_active_tickers(
    rows: list[dict],
    now: datetime,
    pre_hours: float,
    post_hours: float,
) -> list[str]:
    active: list[str] = []
    pre_delta = timedelta(hours=pre_hours)
    post_delta = timedelta(hours=post_hours)
    for row in rows:
        event_time = row.get("event_time")
        if not isinstance(event_time, datetime):
            continue
        if event_time - pre_delta <= now <= event_time + post_delta:
            active.append(row["ticker"])
    return active


def snapshot(
    rest_url: str,
    api_key: str | None,
    tickers: Iterable[str],
    out_path: str,
    depth: int,
    per_request_sleep_s: float,
    max_retries: int,
) -> int:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    count = 0
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    with open(out_path, "a", encoding="utf-8") as handle:
        for ticker in tickers:
            url = f"{rest_url}/markets/{ticker}/orderbook"
            params = {"depth": depth}
            last_err = None
            for attempt in range(max_retries + 1):
                resp = requests.get(url, params=params, headers=headers, timeout=20)
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else min(2.0 * (2**attempt), 30.0)
                    time.sleep(delay)
                    last_err = resp
                    continue
                if resp.status_code >= 500:
                    delay = min(1.0 * (2**attempt), 10.0)
                    time.sleep(delay)
                    last_err = resp
                    continue
                resp.raise_for_status()
                last_err = None
                break
            if last_err is not None:
                last_err.raise_for_status()
            payload = resp.json()
            yes_levels, no_levels = _parse_levels(payload)
            top_yes, top_no = _top_prices(yes_levels, no_levels)
            record = {
                "ts_utc_ms": _utc_ms(),
                "ts_utc": now_iso,
                "market_ticker": ticker,
                "yes_levels": yes_levels,
                "no_levels": no_levels,
                "top_yes": top_yes,
                "top_no": top_no,
                "depth": depth,
                "raw": payload,
            }
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=True))
            handle.write("\n")
            count += 1
            if per_request_sleep_s > 0:
                time.sleep(per_request_sleep_s)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L1 snapshot poller for Kalshi NBA markets.")
    parser.add_argument("--rest-url", default="https://api.elections.kalshi.com/trade-api/v2")
    parser.add_argument("--api-key", default=os.getenv("KALSHI_API_KEY"))
    parser.add_argument("--enriched", default=_default_enriched_path())
    parser.add_argument("--series", default="KXNBAGAME,KXNBASPREAD,KXNBATOTAL")
    parser.add_argument("--out", default=_default_output_path())
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--interval-s", type=float, default=15.0)
    parser.add_argument("--iterations", type=int, default=0, help="0 = run forever")
    parser.add_argument("--per-request-sleep", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--pre-hours", type=float, default=3.0)
    parser.add_argument("--post-hours", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    series = [s.strip() for s in (args.series or "").split(",") if s.strip()]
    rows = _load_tickers(args.enriched, series)
    if not rows:
        raise SystemExit(f"No tickers found in {args.enriched} for series {series}")
    iteration = 0
    while True:
        iteration += 1
        now = datetime.now(tz=timezone.utc)
        tickers = _filter_active_tickers(rows, now, args.pre_hours, args.post_hours)
        if not tickers:
            print("[l1] no active tickers in window; sleeping")
            time.sleep(args.interval_s)
            if args.iterations and iteration >= args.iterations:
                break
            continue
        count = snapshot(
            args.rest_url,
            args.api_key,
            tickers,
            args.out,
            args.depth,
            args.per_request_sleep,
            args.max_retries,
        )
        print(f"[l1] wrote {count} snapshots to {args.out} (active={len(tickers)})")
        if args.iterations and iteration >= args.iterations:
            break
        time.sleep(args.interval_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
