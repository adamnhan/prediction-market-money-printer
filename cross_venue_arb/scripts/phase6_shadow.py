from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
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
from cross_venue_arb.safety import assert_no_live_trading
from cross_venue_arb.storage.mapping_registry import read_game_mappings

logger = logging.getLogger("phase6_shadow")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[PHASE6] %(asctime)s %(message)s"))
    logger.addHandler(handler)

_KALSHI_DATE_RE = re.compile(r"-(\d{2}[A-Z]{3}\d{2})")


def _parse_start_time(value: object) -> datetime | None:
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _extract_start_date(record) -> datetime.date | None:
    details = record.match_details or {}
    for key in ("kalshi_start_time_utc", "polymarket_start_time_utc", "start_time_utc", "event_start_time"):
        dt = _parse_start_time(details.get(key))
        if dt is not None:
            return dt.date()
    ticker = record.kalshi_event_ticker or ""
    match = _KALSHI_DATE_RE.search(ticker)
    if match:
        date_str = match.group(1)
        try:
            return datetime.strptime(date_str.title(), "%d%b%y").date()
        except ValueError:
            return None
    return None


def _is_today_record(record, today: datetime.date) -> bool:
    start_date = _extract_start_date(record)
    if start_date is None:
        return False
    return start_date == today


def _earliest_start_date(records) -> datetime.date | None:
    dates = [d for record in records if (d := _extract_start_date(record)) is not None]
    if not dates:
        return None
    return min(dates)


async def _shadow_loop(manager: BookManager, records, config: MakerTakerConfig, interval_s: float, max_active: int) -> None:
    coordinator = MakerTakerCoordinator(config, max_active_global=max_active)
    while True:
        for record in records:
            trades = coordinator.step_game(record, manager)
            for trade in trades:
                persist_shadow_trade(trade)
        await asyncio.sleep(interval_s)


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
    parser.add_argument(
        "--today-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only include games with start dates matching today (UTC).",
    )
    parser.add_argument(
        "--earliest-bucket-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only include games from the earliest start-date bucket available.",
    )
    parser.add_argument("--as-of-date", default=None, help="Registry as_of_date override")
    parser.add_argument("--edge-target", type=float, default=0.005)
    parser.add_argument("--buffer", type=float, default=0.003)
    parser.add_argument("--kalshi-fee", type=float, default=0.001)
    parser.add_argument("--poly-fee", type=float, default=0.0)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--quote-size", type=float, default=1.0)
    parser.add_argument("--ttl", type=float, default=15.0)
    parser.add_argument("--max-active-global", type=int, default=5)
    parser.add_argument("--reprice-interval", type=float, default=1.0)
    parser.add_argument("--reprice-threshold-ticks", type=int, default=2)
    parser.add_argument("--allow-reprice-up", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    assert_no_live_trading()

    records = read_game_mappings(as_of_date=args.as_of_date)
    if not records:
        raise SystemExit("No game mappings found. Run phase3 with --write-db first.")
    if args.today_only:
        today = datetime.now(timezone.utc).date()
        records = [record for record in records if _is_today_record(record, today)]
        if not records:
            raise SystemExit("No game mappings found for today. Re-run phase3 or disable --today-only.")
    if args.earliest_bucket_only:
        earliest_date = _earliest_start_date(records)
        if earliest_date is None:
            raise SystemExit(
                "No start dates available for earliest-bucket filter. Re-run phase3 or disable --earliest-bucket-only."
            )
        records = [record for record in records if _extract_start_date(record) == earliest_date]
        if not records:
            raise SystemExit("No game mappings found for earliest bucket. Re-run phase3 or disable --earliest-bucket-only.")

    kalshi_tickers = sorted(
        {
            entry.get("ticker")
            for record in records
            for entry in record.kalshi_team_markets
            if entry.get("ticker")
        }
    )
    logger.info("kalshi_tickers count=%d tickers=%s", len(kalshi_tickers), ",".join(kalshi_tickers))
    polymarket_asset_ids = sorted(
        {
            asset_id
            for record in records
            for asset_id in (record.match_details.get("polymarket_asset_ids") or [])
            if asset_id
        }
    )
    logger.info(
        "polymarket_asset_ids count=%d assets=%s",
        len(polymarket_asset_ids),
        ",".join(polymarket_asset_ids),
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
