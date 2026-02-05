#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from datetime import timedelta
from typing import Any

import requests


DEFAULT_REST_URL = "https://api.elections.kalshi.com/trade-api/v2"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _to_epoch_seconds(dt: datetime | None) -> int | None:
    if dt is None:
        return None
    return int(dt.timestamp())


def _init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                market_ticker TEXT NOT NULL,
                ts INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                fidelity INTEGER,
                source_json TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candles_ticker_ts ON candles(market_ticker, ts)"
        )
        conn.commit()


def _extract_candles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "candlesticks" in payload and isinstance(payload["candlesticks"], list):
        return payload["candlesticks"]
    if "candles" in payload and isinstance(payload["candles"], list):
        return payload["candles"]
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("candles"), list):
        return data["candles"]
    if isinstance(data, list):
        return data
    return []


def fetch_candles(
    rest_url: str,
    api_key: str | None,
    series_ticker: str,
    market_ticker: str,
    start_ts: int | None,
    end_ts: int | None,
    period_interval: int | None,
) -> list[dict[str, Any]]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    params: dict[str, Any] = {}
    if start_ts is not None:
        params["start_ts"] = start_ts
    if end_ts is not None:
        params["end_ts"] = end_ts
    if period_interval is not None:
        params["period_interval"] = period_interval
    url = f"{rest_url}/series/{series_ticker}/markets/{market_ticker}/candlesticks"
    resp = requests.get(url, params=params, headers=headers, timeout=20)
    if resp.status_code == 429:
        time.sleep(2)
        resp = requests.get(url, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    return _extract_candles(resp.json() or {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical candles for NBA markets.")
    parser.add_argument("--rest-url", default=DEFAULT_REST_URL)
    parser.add_argument("--api-key", default=os.getenv("KALSHI_API_KEY"))
    parser.add_argument("--csv", default="data/nba_historical_markets.csv")
    parser.add_argument("--db", default="data/nba_historical_candles.sqlite")
    parser.add_argument("--period-interval", type=int, default=60)
    parser.add_argument("--sleep-s", type=float, default=0.1)
    parser.add_argument("--max-markets", type=int, default=0, help="0 = no limit")
    parser.add_argument("--fallback-hours", type=float, default=6.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _init_db(args.db)
    total = 0
    with sqlite3.connect(args.db) as conn, open(args.csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            if args.max_markets and idx > args.max_markets:
                break
            ticker = row.get("market_ticker")
            series = row.get("series_ticker")
            if not ticker or not series:
                continue
            start_dt = _parse_iso(row.get("event_start_time"))
            end_dt = _parse_iso(row.get("close_time"))
            if start_dt and not end_dt:
                end_dt = start_dt + timedelta(hours=args.fallback_hours)
            if end_dt and not start_dt:
                start_dt = end_dt - timedelta(hours=args.fallback_hours)
            start_ts = _to_epoch_seconds(start_dt)
            end_ts = _to_epoch_seconds(end_dt)
            if start_ts is None or end_ts is None:
                continue
            if start_ts >= end_ts:
                continue
            if args.period_interval not in (1, 60, 1440):
                raise SystemExit("period_interval must be 1, 60, or 1440")
            candles = fetch_candles(
                args.rest_url,
                args.api_key,
                series,
                ticker,
                start_ts,
                end_ts,
                args.period_interval,
            )
            for candle in candles:
                ts = (
                    candle.get("end_period_ts")
                    or candle.get("ts")
                    or candle.get("t")
                    or candle.get("start_ts")
                )
                if ts is None:
                    continue
                price = candle.get("price") or {}
                if isinstance(price, dict):
                    open_px = price.get("open")
                    high_px = price.get("high")
                    low_px = price.get("low")
                    close_px = price.get("close")
                else:
                    open_px = high_px = low_px = close_px = None
                if close_px is None:
                    yes_bid = candle.get("yes_bid") or {}
                    yes_ask = candle.get("yes_ask") or {}
                    if isinstance(yes_bid, dict) and isinstance(yes_ask, dict):
                        bid_close = yes_bid.get("close")
                        ask_close = yes_ask.get("close")
                        if bid_close is not None and ask_close is not None:
                            close_px = (float(bid_close) + float(ask_close)) / 2.0
                conn.execute(
                    """
                    INSERT INTO candles (
                        market_ticker, ts, open, high, low, close, volume, fidelity, source_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticker,
                        int(ts),
                        open_px,
                        high_px,
                        low_px,
                        close_px,
                        candle.get("volume"),
                        args.period_interval,
                        json.dumps(candle, separators=(",", ":"), ensure_ascii=True),
                    ),
                )
                total += 1
            conn.commit()
            if args.sleep_s:
                time.sleep(args.sleep_s)
            if idx % 50 == 0:
                print(f"[candles] markets={idx} total_rows={total}")
    print(f"[candles] done markets={idx} total_rows={total} db={args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
