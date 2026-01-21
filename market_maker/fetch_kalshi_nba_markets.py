#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

if __package__ is None and str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from cross_venue_arb.config import CONFIG


KALSHI_DEFAULT_REST_URL = "https://api.elections.kalshi.com/trade-api/v2"


def fetch_markets(series_ticker: str, status: str, rest_url: str) -> list[dict]:
    markets: list[dict] = []
    cursor: str | None = None
    while True:
        params = {"series_ticker": series_ticker, "status": status, "limit": 100}
        if cursor:
            params["cursor"] = cursor
        response = requests.get(f"{rest_url}/markets", params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("markets") or []
        if isinstance(batch, list):
            markets.extend(batch)
        cursor = payload.get("cursor") or payload.get("next_cursor")
        if not cursor:
            break
    return markets


def _is_winner_market(market: dict) -> bool:
    market_type = str(market.get("market_type") or market.get("type") or "").lower()
    if "winner" in market_type or "moneyline" in market_type:
        return True
    title = str(market.get("title") or market.get("question") or "").lower()
    return any(term in title for term in (" win", " wins", " winner"))


def write_markets(path: Path, tickers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(tickers) + "\n"
    path.write_text(payload, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Kalshi NBA winner markets to a txt file.")
    parser.add_argument("--series-ticker", default="KXNBAGAME")
    parser.add_argument("--status", default="open")
    parser.add_argument("--output", default="config/markets_nba.txt")
    parser.add_argument("--rest-url", default=None)
    parser.add_argument("--no-filter", action="store_true")
    args = parser.parse_args()

    rest_url = args.rest_url or CONFIG.kalshi.rest_url or KALSHI_DEFAULT_REST_URL
    markets = fetch_markets(args.series_ticker, args.status, rest_url)
    if args.no_filter:
        tickers = [str(m.get("ticker") or m.get("market_ticker") or "") for m in markets]
    else:
        tickers = [
            str(m.get("ticker") or m.get("market_ticker") or "")
            for m in markets
            if _is_winner_market(m)
        ]
    tickers = sorted({t for t in tickers if t})
    write_markets(Path(args.output), tickers)
    print(f"Wrote {len(tickers)} tickers to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
