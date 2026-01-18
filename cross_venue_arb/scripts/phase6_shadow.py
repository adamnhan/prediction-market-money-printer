from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

if __package__ is None and str(Path(__file__).parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[2]))

load_dotenv()

from cross_venue_arb.arb.phase5 import OpportunityTracker, Phase5Config, evaluate_game
from cross_venue_arb.arb.phase6 import Phase6Config, persist_shadow_trade, simulate_opportunity
from cross_venue_arb.books.kalshi_ws import run as kalshi_ws
from cross_venue_arb.books.manager import BookManager
from cross_venue_arb.books.polymarket_ws import run as polymarket_ws
from cross_venue_arb.storage.mapping_registry import read_game_mappings


async def _shadow_loop(manager: BookManager, records, p5: Phase5Config, p6: Phase6Config) -> None:
    tracker = OpportunityTracker(p5.emit_cooldown_s)
    while True:
        for record in records:
            for opp in evaluate_game(record, manager, p5):
                if opp.reject_reason:
                    continue
                key = f"opp:{opp.game_key}:{opp.direction}"
                confirmed = tracker.track_streak(key, True, p5.confirm_ticks)
                if confirmed and tracker.should_emit(key, opp.edge_per_contract):
                    record = simulate_opportunity(opp, manager, p6)
                    persist_shadow_trade(record)
        await asyncio.sleep(1.0)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 shadow execution")
    parser.add_argument("--as-of-date", default=None, help="Registry as_of_date override")
    parser.add_argument("--min-size", type=float, default=10.0)
    parser.add_argument("--min-edge", type=float, default=0.010)
    parser.add_argument("--buffer", type=float, default=0.005)
    parser.add_argument("--kalshi-fee", type=float, default=0.001)
    parser.add_argument("--poly-fee", type=float, default=0.0)
    parser.add_argument("--confirm-ticks", type=int, default=2)
    parser.add_argument("--latency-min", type=int, default=150)
    parser.add_argument("--latency-max", type=int, default=300)
    parser.add_argument("--slippage-ticks", type=int, default=1)
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
    p5 = Phase5Config(
        min_size=args.min_size,
        min_edge=args.min_edge,
        buffer_per_contract=args.buffer,
        kalshi_fee_per_contract=args.kalshi_fee,
        polymarket_fee_per_contract=args.poly_fee,
        confirm_ticks=args.confirm_ticks,
    )
    p6 = Phase6Config(
        latency_min_ms=args.latency_min,
        latency_max_ms=args.latency_max,
        slippage_ticks=args.slippage_ticks,
        buffer_per_contract=args.buffer,
        kalshi_fee_per_contract=args.kalshi_fee,
        polymarket_fee_per_contract=args.poly_fee,
    )

    await asyncio.gather(
        kalshi_ws(manager, kalshi_tickers),
        polymarket_ws(manager, polymarket_asset_ids),
        _shadow_loop(manager, records, p5, p6),
    )


if __name__ == "__main__":
    asyncio.run(main())
