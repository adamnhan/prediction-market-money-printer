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

from cross_venue_arb.arb.opportunity import Opportunity
from cross_venue_arb.arb.phase5 import OpportunityTracker, Phase5Config, evaluate_game
from cross_venue_arb.books.kalshi_ws import run as kalshi_ws
from cross_venue_arb.books.manager import BookManager
from cross_venue_arb.books.polymarket_ws import run as polymarket_ws
from cross_venue_arb.storage.mapping_registry import GameMappingRecord, read_game_mappings


@dataclass(frozen=True)
class GameView:
    record: GameMappingRecord
    polymarket_asset_ids: list[str]


def _build_views(records: list[GameMappingRecord]) -> list[GameView]:
    views: list[GameView] = []
    for record in records:
        details = record.match_details or {}
        asset_ids = details.get("polymarket_asset_ids") or []
        views.append(GameView(record=record, polymarket_asset_ids=asset_ids))
    return views


def _print_opportunity(opp: Opportunity) -> None:
    if opp.reject_reason:
        print(f"REJECT {opp.game_key} {opp.direction} reason={opp.reject_reason}")
        return
    print(
        f"OPP {opp.game_key} {opp.direction} edge={opp.edge_per_contract:.4f} "
        f"size={opp.size_max:.1f} profit={opp.expected_profit_max:.4f}"
    )
    for leg in opp.legs:
        print(
            f"  {leg.venue} {leg.market_id} BUY {leg.outcome} "
            f"ask={leg.limit_price:.4f} size={leg.available_size:.2f}"
        )


async def _arb_loop(
    manager: BookManager,
    views: list[GameView],
    config: Phase5Config,
    interval_s: float,
) -> None:
    tracker = OpportunityTracker(config.emit_cooldown_s)
    while True:
        for view in views:
            for opp in evaluate_game(view.record, manager, config):
                if opp.reject_reason:
                    if tracker.should_emit(f"rej:{opp.game_key}:{opp.direction}:{opp.reject_reason}", 0.0):
                        _print_opportunity(opp)
                    continue
                key = f"opp:{opp.game_key}:{opp.direction}"
                confirmed = tracker.track_streak(key, True, config.confirm_ticks)
                if confirmed and tracker.should_emit(key, opp.edge_per_contract):
                    _print_opportunity(opp)
        await asyncio.sleep(interval_s)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 arbitrage detector")
    parser.add_argument("--as-of-date", default=None, help="Registry as_of_date override")
    parser.add_argument("--interval", type=float, default=2.0, help="Loop interval seconds")
    parser.add_argument("--min-size", type=float, default=10.0, help="Minimum size")
    parser.add_argument("--min-edge", type=float, default=0.010, help="Minimum edge per contract")
    parser.add_argument("--buffer", type=float, default=0.005, help="Buffer per contract")
    parser.add_argument("--kalshi-fee", type=float, default=0.001, help="Kalshi fee per contract")
    parser.add_argument("--poly-fee", type=float, default=0.0, help="Polymarket fee per contract")
    parser.add_argument("--confirm-ticks", type=int, default=2, help="Consecutive ticks required")
    args = parser.parse_args()

    records = read_game_mappings(as_of_date=args.as_of_date)
    if not records:
        raise SystemExit("No game mappings found. Run phase3 with --write-db first.")

    views = _build_views(records)
    kalshi_tickers = sorted(
        {
            entry.get("ticker")
            for record in records
            for entry in record.kalshi_team_markets
            if entry.get("ticker")
        }
    )
    polymarket_asset_ids = sorted(
        {asset_id for view in views for asset_id in view.polymarket_asset_ids if asset_id}
    )

    manager = BookManager()
    config = Phase5Config(
        min_size=args.min_size,
        min_edge=args.min_edge,
        buffer_per_contract=args.buffer,
        kalshi_fee_per_contract=args.kalshi_fee,
        polymarket_fee_per_contract=args.poly_fee,
        confirm_ticks=args.confirm_ticks,
    )

    await asyncio.gather(
        kalshi_ws(manager, kalshi_tickers),
        polymarket_ws(manager, polymarket_asset_ids),
        _arb_loop(manager, views, config, args.interval),
    )


if __name__ == "__main__":
    asyncio.run(main())
