from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
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
from cross_venue_arb.safety import assert_no_live_trading
from cross_venue_arb.storage.mapping_registry import read_game_mappings


logger = logging.getLogger("phase6_shadow")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[PHASE6] %(asctime)s %(message)s"))
    logger.addHandler(handler)


@dataclass
class SyncState:
    last_kalshi_ts: float | None = None
    last_poly_ts: float | None = None
    kalshi_dirty: bool = False
    poly_dirty: bool = False


def _kalshi_tickers(record) -> list[str]:
    return [entry.get("ticker") for entry in record.kalshi_team_markets if entry.get("ticker")]


def _polymarket_assets(record) -> list[str]:
    details = record.match_details or {}
    asset_ids = details.get("polymarket_asset_ids") or []
    assets = [asset_id for asset_id in asset_ids if asset_id]
    if assets:
        return assets
    outcome_map = details.get("polymarket_outcome_map") or {}
    return [asset_id for asset_id in outcome_map.values() if asset_id]


def _mark_dirty(state: SyncState, manager: BookManager, record) -> None:
    kalshi_ts = manager.latest_update_ts("kalshi", _kalshi_tickers(record))
    if kalshi_ts is not None and (state.last_kalshi_ts is None or kalshi_ts > state.last_kalshi_ts):
        state.last_kalshi_ts = kalshi_ts
        state.kalshi_dirty = True
    poly_ts = manager.latest_update_ts("polymarket", _polymarket_assets(record))
    if poly_ts is not None and (state.last_poly_ts is None or poly_ts > state.last_poly_ts):
        state.last_poly_ts = poly_ts
        state.poly_dirty = True


async def _shadow_loop(manager: BookManager, records, p5: Phase5Config, p6: Phase6Config) -> None:
    tracker = OpportunityTracker(p5.emit_cooldown_s)
    states: dict[str, SyncState] = {}
    while True:
        for record in records:
            state = states.setdefault(record.game_key, SyncState())
            _mark_dirty(state, manager, record)
            if not (state.kalshi_dirty and state.poly_dirty):
                continue
            state.kalshi_dirty = False
            state.poly_dirty = False
            for opp in evaluate_game(record, manager, p5):
                if opp.reject_reason:
                    continue
                key = f"opp:{opp.game_key}:{opp.direction}"
                confirmed = tracker.track_streak(key, True, p5.confirm_ticks)
                if confirmed and tracker.should_emit(key, opp.edge_per_contract):
                    logger.info(
                        "opp_detected game=%s dir=%s edge=%.4f size=%.1f",
                        opp.game_key,
                        opp.direction,
                        opp.edge_per_contract,
                        opp.size_max,
                    )
                    record = simulate_opportunity(opp, manager, p6)
                    persist_shadow_trade(record)
        await asyncio.sleep(1.0)


async def _subscription_status(
    manager: BookManager,
    kalshi_tickers: list[str],
    polymarket_asset_ids: list[str],
    interval_s: float = 30.0,
) -> None:
    while True:
        def _summarize(venue: str, ids: list[str]) -> tuple[int, int, int]:
            total = len(ids)
            seen = 0
            healthy = 0
            for market_id in ids:
                book = manager.get_book(venue, market_id)
                if book is None:
                    continue
                if book.has_snapshot:
                    seen += 1
                if manager.is_healthy(venue, market_id):
                    healthy += 1
            stale = max(seen - healthy, 0)
            return total, seen, healthy, stale

        k_total, k_seen, k_healthy, k_stale = _summarize("kalshi", kalshi_tickers)
        p_total, p_seen, p_healthy, p_stale = _summarize("polymarket", polymarket_asset_ids)
        logger.info(
            "subscription_status kalshi total=%d seen=%d healthy=%d stale=%d",
            k_total,
            k_seen,
            k_healthy,
            k_stale,
        )
        logger.info(
            "subscription_status polymarket total=%d seen=%d healthy=%d stale=%d",
            p_total,
            p_seen,
            p_healthy,
            p_stale,
        )
        await asyncio.sleep(interval_s)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 shadow execution")
    parser.add_argument("--as-of-date", default=None, help="Registry as_of_date override")
    parser.add_argument("--min-size", type=float, default=10.0)
    parser.add_argument("--min-edge", type=float, default=0.010)
    parser.add_argument("--buffer", type=float, default=0.005)
    parser.add_argument("--kalshi-fee", type=float, default=0.001)
    parser.add_argument("--poly-fee", type=float, default=0.0)
    parser.add_argument("--confirm-ticks", type=int, default=2)
    parser.add_argument("--fresh-ms", type=int, default=1500)
    parser.add_argument("--sync-ms", type=int, default=400)
    parser.add_argument("--latency-min", type=int, default=150)
    parser.add_argument("--latency-max", type=int, default=300)
    parser.add_argument("--slippage-ticks", type=int, default=1)
    args = parser.parse_args()
    assert_no_live_trading()

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
        fresh_ms=args.fresh_ms,
        sync_ms=args.sync_ms,
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
        _subscription_status(manager, kalshi_tickers, polymarket_asset_ids),
    )


if __name__ == "__main__":
    asyncio.run(main())
