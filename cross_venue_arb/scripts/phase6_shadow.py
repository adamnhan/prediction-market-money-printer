from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

if __package__ is None and str(Path(__file__).parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[2]))

load_dotenv()

from cross_venue_arb.arb.maker_taker import MakerTakerConfig, MakerTakerCoordinator
from cross_venue_arb.arb.phase6 import persist_shadow_trade
from cross_venue_arb.books.kalshi_ws import run as kalshi_ws
from cross_venue_arb.books.manager import BookManager
from cross_venue_arb.books.polymarket_ws import run as polymarket_ws
from cross_venue_arb.storage.mapping_registry import read_game_mappings


async def _shadow_loop(manager: BookManager, records, config: MakerTakerConfig, interval_s: float, max_active: int) -> None:
    coordinator = MakerTakerCoordinator(config, max_active_global=max_active)
    while True:
        for record in records:
            trades = coordinator.step_game(record, manager)
            for trade in trades:
                persist_shadow_trade(trade)
        await asyncio.sleep(interval_s)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 shadow execution")
    parser.add_argument("--as-of-date", default=None, help="Registry as_of_date override")
    parser.add_argument("--edge-target", type=float, default=0.015)
    parser.add_argument("--buffer", type=float, default=0.005)
    parser.add_argument("--kalshi-fee", type=float, default=0.001)
    parser.add_argument("--poly-fee", type=float, default=0.0)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--quote-size", type=float, default=1.0)
    parser.add_argument("--ttl", type=float, default=30.0)
    parser.add_argument("--max-active-global", type=int, default=5)
    parser.add_argument("--reprice-interval", type=float, default=1.0)
    parser.add_argument("--reprice-threshold-ticks", type=int, default=1)
    parser.add_argument("--allow-reprice-up", action="store_true")
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    records = read_game_mappings(as_of_date=args.as_of_date)
    if not records:
        raise SystemExit("No game mappings found. Run phase3 with --write-db first.")

    kalshi_tickers = sorted(
        {
            entry.get("ticker")
            for record in records
            for entry in record.kalshi_team_markets
            if entry.get("ticker")
        }
    )
    polymarket_asset_ids = sorted(
        {
            asset_id
            for record in records
            for asset_id in (record.match_details.get("polymarket_asset_ids") or [])
            if asset_id
        }
    )

    manager = BookManager()
    config = MakerTakerConfig(
        edge_target=args.edge_target,
        buffer_per_contract=args.buffer,
        kalshi_fee_per_contract=args.kalshi_fee,
        polymarket_fee_per_contract=args.poly_fee,
        tick_size=args.tick_size,
        quote_size=args.quote_size,
        ttl_seconds=args.ttl,
        min_reprice_interval_s=args.reprice_interval,
        reprice_threshold_ticks=args.reprice_threshold_ticks,
        allow_reprice_up=args.allow_reprice_up,
    )

    await asyncio.gather(
        kalshi_ws(manager, kalshi_tickers),
        polymarket_ws(manager, polymarket_asset_ids),
        _shadow_loop(manager, records, config, args.interval, args.max_active_global),
    )


if __name__ == "__main__":
    asyncio.run(main())
