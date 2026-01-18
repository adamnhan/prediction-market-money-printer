from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__ is None and str(Path(__file__).parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[2]))

load_dotenv()

from cross_venue_arb.books.kalshi_ws import run as kalshi_ws
from cross_venue_arb.books.manager import BookManager
from cross_venue_arb.books.polymarket_ws import run as polymarket_ws
from cross_venue_arb.storage.mapping_registry import GameMappingRecord, read_game_mappings


@dataclass(frozen=True)
class GameView:
    event_ticker: str
    team_markets: list[dict[str, Any]]
    team_a_norm: str | None
    team_b_norm: str | None
    polymarket_asset_ids: list[str]
    polymarket_asset_id: str | None
    polymarket_outcome_map: dict[str, str]


def _format_price(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _game_label(game: GameView) -> str:
    if game.team_a_norm and game.team_b_norm:
        return f"{game.team_a_norm} @ {game.team_b_norm}"
    return game.event_ticker


async def _print_loop(manager: BookManager, games: list[GameView], interval_s: float) -> None:
    while True:
        print(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + "Z")
        for game in games:
            print(_game_label(game))
            for team_market in game.team_markets:
                ticker = team_market.get("ticker")
                label = team_market.get("team_norm") or ticker or "kalshi"
                book = manager.get_book("kalshi", ticker) if ticker else None
                bid = _format_price(book.best_bid[0]) if book and book.best_bid else "n/a"
                ask = _format_price(book.best_ask[0]) if book and book.best_ask else "n/a"
                status = "OK" if ticker and manager.is_healthy("kalshi", ticker) else "STALE"
                print(f"  Kalshi {label}: bid {bid} / ask {ask} ({status})")
            if game.polymarket_asset_id:
                book = manager.get_book("polymarket", game.polymarket_asset_id)
                bid = _format_price(book.best_bid[0]) if book and book.best_bid else "n/a"
                ask = _format_price(book.best_ask[0]) if book and book.best_ask else "n/a"
                status = (
                    "OK"
                    if manager.is_healthy("polymarket", game.polymarket_asset_id)
                    else "STALE"
                )
                print(f"  Poly (winner): bid {bid} / ask {ask} ({status})")
        print("")
        await asyncio.sleep(interval_s)


def _build_views(records: list[GameMappingRecord]) -> list[GameView]:
    views: list[GameView] = []
    for record in records:
        match_details = record.match_details or {}
        asset_ids = match_details.get("polymarket_asset_ids") or []
        outcome_map = match_details.get("polymarket_outcome_map") or {}
        preferred = None
        if record.team_b_norm and record.team_b_norm in outcome_map:
            preferred = outcome_map[record.team_b_norm]
        elif record.team_a_norm and record.team_a_norm in outcome_map:
            preferred = outcome_map[record.team_a_norm]
        if not preferred and asset_ids:
            preferred = asset_ids[0]
        if not preferred and record.polymarket_market_id:
            preferred = record.polymarket_market_id
        views.append(
            GameView(
                event_ticker=record.kalshi_event_ticker,
                team_markets=record.kalshi_team_markets,
                team_a_norm=record.team_a_norm,
                team_b_norm=record.team_b_norm,
                polymarket_asset_ids=asset_ids,
                polymarket_asset_id=preferred,
                polymarket_outcome_map=outcome_map,
            )
        )
    return views


async def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 order book streaming")
    parser.add_argument("--as-of-date", default=None, help="Registry as_of_date override")
    parser.add_argument("--interval", type=float, default=5.0, help="Print interval seconds")
    args = parser.parse_args()

    records = read_game_mappings(as_of_date=args.as_of_date)
    if not records:
        raise SystemExit("No game mappings found. Run phase3 with --write-db first.")

    games = _build_views(records)
    missing_assets = [g.event_ticker for g in games if not g.polymarket_asset_id]
    if missing_assets:
        print("Missing polymarket asset ids for:")
        for event_ticker in missing_assets:
            print(f"- {event_ticker}")
    kalshi_tickers = sorted(
        {entry.get("ticker") for record in records for entry in record.kalshi_team_markets if entry.get("ticker")}
    )
    polymarket_market_ids = sorted(
        {
            asset_id
            for view in games
            for asset_id in (view.polymarket_asset_ids or [view.polymarket_asset_id])
            if asset_id
        }
    )

    manager = BookManager()
    await asyncio.gather(
        kalshi_ws(manager, kalshi_tickers),
        polymarket_ws(manager, polymarket_market_ids),
        _print_loop(manager, games, args.interval),
    )


if __name__ == "__main__":
    asyncio.run(main())
