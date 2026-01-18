from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

import requests

if __package__ is None and str(Path(__file__).parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[2]))

from cross_venue_arb.config import CONFIG
from cross_venue_arb.matching.normalize import format_normalized_game, normalize_game


logger = logging.getLogger("phase1_markets")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[PHASE1] %(asctime)s %(message)s"))
    logger.addHandler(handler)


POLYMARKET_BASE_URL = "https://clob.polymarket.com"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"
KALSHI_DEFAULT_REST_URL = "https://api.elections.kalshi.com/trade-api/v2"


@dataclass
class Market:
    venue: str
    ticker: str
    title: str
    start_time_utc: str | None
    home_team: str | None
    away_team: str | None
    market_type: str = "winner"
    raw: dict[str, Any] | None = None


def _parse_timestamp(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value > 1e12:
            dt = datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
        elif value > 1e10:
            dt = datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
        else:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        return dt.isoformat()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return None
    return None


def _is_yes_no_outcome(outcomes: list[str]) -> bool:
    if len(outcomes) != 2:
        return False
    lowered = {o.strip().lower() for o in outcomes}
    return lowered == {"yes", "no"}


def _is_winner_market(market: dict) -> bool:
    market_type = str(
        market.get("market_type")
        or market.get("marketType")
        or market.get("type")
        or ""
    ).lower()
    if "winner" in market_type or "moneyline" in market_type:
        return True
    sports_type = str(market.get("sportsMarketType") or "").lower()
    if "winner" in sports_type or "moneyline" in sports_type or "win" in sports_type:
        return True
    outcomes = market.get("outcomes") or []
    if isinstance(outcomes, list) and outcomes:
        outcomes_str = [str(o) for o in outcomes]
        if len(outcomes_str) == 2 and not _is_yes_no_outcome(outcomes_str):
            return True
        if _is_yes_no_outcome(outcomes_str):
            question = str(market.get("question") or market.get("title") or "").lower()
            winner_terms = (" win", " wins", " beat", " beats", " defeat", " winner")
            return any(term in question for term in winner_terms)
    return False


def _is_nba_market(market: dict) -> bool:
    title = str(
        market.get("question")
        or market.get("title")
        or market.get("event_title")
        or ""
    ).lower()
    if "nba" in title:
        return True
    tags = market.get("tags") or market.get("categories") or []
    if isinstance(tags, list):
        for tag in tags:
            if "nba" in str(tag).lower():
                return True
    event_slug = str(market.get("event_slug") or market.get("event_slug") or "").lower()
    return "nba" in event_slug


def _extract_matchup(title: str) -> tuple[str | None, str | None]:
    separators = [" vs ", " vs. ", " @ ", " at "]
    lowered = title.lower()
    for sep in separators:
        if sep in lowered:
            parts = title.split(sep, 1)
            if len(parts) == 2:
                return parts[0].strip() or None, parts[1].strip() or None
    return None, None


def fetch_polymarket_markets() -> list[dict]:
    markets: list[dict] = []
    cursor: str | None = None
    while True:
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        logger.info("polymarket_clob_fetch cursor=%s", cursor or "none")
        response = requests.get(f"{POLYMARKET_BASE_URL}/markets", params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("markets") or payload.get("data") or []
        if isinstance(batch, list):
            markets.extend(batch)
        cursor = payload.get("next_cursor") or payload.get("cursor")
        if not cursor:
            break
    return markets


def fetch_polymarket_markets_gamma(series_id: int, tag_id: int) -> list[dict]:
    markets: list[dict] = []
    offset = 0
    limit = 100
    while True:
        params = {
            "series_id": series_id,
            "tag_id": tag_id,
            "active": "true",
            "closed": "false",
            "order": "startTime",
            "ascending": "true",
            "limit": limit,
            "offset": offset,
        }
        logger.info("polymarket_gamma_fetch offset=%d", offset)
        response = requests.get(f"{POLYMARKET_GAMMA_URL}/events", params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if offset == 0:
            if isinstance(payload, dict):
                logger.info("polymarket_gamma_payload keys=%s", sorted(payload.keys()))
            else:
                logger.info("polymarket_gamma_payload type=%s", type(payload).__name__)
        events = None
        if isinstance(payload, dict):
            events = payload.get("events")
            if events is None:
                events = payload.get("data")
        else:
            events = payload
        if not isinstance(events, list) or not events:
            break
        for event in events:
            if not isinstance(event, dict):
                continue
            event_start = event.get("startTime") or event.get("start_time")
            event_title = event.get("title") or event.get("name")
            event_tags = event.get("tags")
            event_markets = event.get("markets") or []
            if not isinstance(event_markets, list):
                continue
            for market in event_markets:
                if not isinstance(market, dict):
                    continue
                enriched = market.copy()
                enriched.setdefault("event_start_time", event_start)
                enriched.setdefault("event_title", event_title)
                if event_tags and not enriched.get("tags"):
                    enriched["tags"] = event_tags
                markets.append(enriched)
        offset += len(events)
        if len(events) < limit:
            break
    return markets


def normalize_polymarket(markets: list[dict]) -> list[Market]:
    normalized: list[Market] = []
    for market in markets:
        if not _is_nba_market(market):
            continue
        status = str(market.get("status") or "").lower()
        if status and status not in {"active", "open"}:
            continue
        if market.get("closed") is True or market.get("active") is False:
            continue
        if not _is_winner_market(market):
            continue

        ticker = str(market.get("market_id") or market.get("id") or "")
        if not ticker:
            continue
        title = str(market.get("question") or market.get("title") or "").strip()
        title_lower = title.lower()
        if any(
            term in title_lower
            for term in ("1h", "first half", "1st half", "1q", "first quarter", "1st quarter")
        ):
            continue
        start_time = _parse_timestamp(
            market.get("start_time")
            or market.get("event_start_time")
            or market.get("startTime")
            or market.get("eventStartTime")
            or market.get("gameStartTime")
            or market.get("startDateIso")
            or market.get("startDate")
        )
        home_team, away_team = _extract_matchup(title)

        normalized.append(
            Market(
                venue="polymarket",
                ticker=ticker,
                title=title,
                start_time_utc=start_time,
                home_team=home_team,
                away_team=away_team,
                raw=market,
            )
        )
    return normalized


def fetch_kalshi_markets(series_ticker: str) -> list[dict]:
    markets: list[dict] = []
    cursor: str | None = None
    base_url = CONFIG.kalshi.rest_url or KALSHI_DEFAULT_REST_URL
    while True:
        params = {"series_ticker": series_ticker, "status": "open", "limit": 100}
        if cursor:
            params["cursor"] = cursor
        response = requests.get(f"{base_url}/markets", params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("markets") or []
        if isinstance(batch, list):
            markets.extend(batch)
        cursor = payload.get("cursor") or payload.get("next_cursor")
        if not cursor:
            break
    return markets


def normalize_kalshi(markets: list[dict]) -> list[Market]:
    normalized: list[Market] = []
    for market in markets:
        ticker = str(market.get("ticker") or market.get("market_ticker") or "")
        if not ticker:
            continue
        title = str(market.get("title") or market.get("question") or "").strip()
        start_time = _parse_timestamp(
            market.get("event_start_time")
            or market.get("close_time")
            or market.get("close_ts")
            or market.get("open_time")
        )
        market_type = str(market.get("market_type") or market.get("type") or "winner")
        home_team, away_team = _extract_matchup(title)
        normalized.append(
            Market(
                venue="kalshi",
                ticker=ticker,
                title=title,
                start_time_utc=start_time,
                home_team=home_team,
                away_team=away_team,
                market_type=market_type,
                raw=market,
            )
        )
    return normalized


def write_json(path: Path, markets: list[Market]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(market) for market in markets]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _print_summary(markets: list[Market], venue: str, sample_size: int) -> None:
    subset = [m for m in markets if m.venue == venue]
    print(f"{venue.capitalize()}: {len(subset)} NBA winner markets found")
    for market in subset[:sample_size]:
        print(
            f"- {market.ticker} | {market.title} | start_time_utc={market.start_time_utc or 'n/a'}"
        )
    if sample_size < len(subset):
        print(f"- ... {len(subset) - sample_size} more")


def _print_normalized_sample(games: list[object], venue: str, sample_size: int) -> None:
    subset = [g for g in games if getattr(g, "venue", None) == venue]
    print(f"{venue.capitalize()}: {len(subset)} normalized games")
    for game in subset[:sample_size]:
        print(f"- {format_normalized_game(game)}")
    if sample_size < len(subset):
        print(f"- ... {len(subset) - sample_size} more")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 market discovery")
    parser.add_argument("--series-ticker", default="KXNBAGAME", help="Kalshi series ticker")
    parser.add_argument(
        "--polymarket-source",
        choices=("gamma", "clob"),
        default="gamma",
        help="Polymarket market source",
    )
    parser.add_argument(
        "--polymarket-series-id",
        type=int,
        default=10345,
        help="Polymarket Gamma series_id for NBA",
    )
    parser.add_argument(
        "--polymarket-tag-id",
        type=int,
        default=100639,
        help="Polymarket Gamma tag_id for NBA",
    )
    parser.add_argument(
        "--output",
        default="cross_venue_arb/data/phase1_markets.json",
        help="Output JSON path",
    )
    parser.add_argument("--sample-size", type=int, default=10, help="Sample size per venue")
    parser.add_argument(
        "--list-polymarket",
        action="store_true",
        help="List all Polymarket markets found",
    )
    args = parser.parse_args()

    if args.polymarket_source == "gamma":
        polymarket_raw = fetch_polymarket_markets_gamma(
            args.polymarket_series_id,
            args.polymarket_tag_id,
        )
    else:
        polymarket_raw = fetch_polymarket_markets()
    polymarket_markets = normalize_polymarket(polymarket_raw)
    polymarket_games = [
        normalize_game(market.raw or {}, venue="polymarket") for market in polymarket_markets
    ]

    kalshi_raw = fetch_kalshi_markets(args.series_ticker)
    kalshi_markets = normalize_kalshi(kalshi_raw)
    kalshi_games = [normalize_game(market.raw or {}, venue="kalshi") for market in kalshi_markets]

    all_markets = polymarket_markets + kalshi_markets
    write_json(Path(args.output), all_markets)

    if args.list_polymarket:
        _print_summary(all_markets, "polymarket", len(polymarket_markets))
    else:
        _print_summary(all_markets, "polymarket", args.sample_size)
    _print_summary(all_markets, "kalshi", args.sample_size)
    _print_normalized_sample(polymarket_games, "polymarket", args.sample_size)
    _print_normalized_sample(kalshi_games, "kalshi", args.sample_size)


if __name__ == "__main__":
    main()
