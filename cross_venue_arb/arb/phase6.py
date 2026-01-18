"""Shadow execution simulator for Phase 6."""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass

from cross_venue_arb.arb.opportunity import Opportunity
from cross_venue_arb.books.manager import BookManager
from cross_venue_arb.storage.shadow_trades import ShadowLegRecord, ShadowTradeRecord, write_shadow_trade


@dataclass(frozen=True)
class Phase6Config:
    latency_min_ms: int = 150
    latency_max_ms: int = 300
    slippage_ticks: int = 1
    tick_size: float = 0.01
    buffer_per_contract: float = 0.005
    kalshi_fee_per_contract: float = 0.001
    polymarket_fee_per_contract: float = 0.0


def simulate_opportunity(
    opp: Opportunity,
    manager: BookManager,
    config: Phase6Config,
) -> ShadowTradeRecord:
    latency_ms = random.randint(config.latency_min_ms, config.latency_max_ms)
    time.sleep(latency_ms / 1000.0)

    opp_id = uuid.uuid4().hex
    leg_results: list[ShadowLegRecord] = []
    filled_sizes: list[float] = []
    fill_prices: list[float] = []
    reasons: list[str] = []

    for leg in opp.legs:
        leg_record = _fill_leg(leg, manager, config, opp.size_max)
        leg_results.append(leg_record)
        filled_sizes.append(leg_record.filled_size)
        if leg_record.filled_price is not None:
            fill_prices.append(leg_record.filled_price)
        if leg_record.reason:
            reasons.append(leg_record.reason)

    filled_a, filled_b = filled_sizes[0], filled_sizes[1] if len(filled_sizes) > 1 else 0.0
    hedge_size = min(filled_a, filled_b)
    if hedge_size <= 0:
        status = "MISSED" if max(filled_a, filled_b) == 0 else "LEGGED"
    elif filled_a == filled_b == opp.size_max:
        status = "FULL_FILL"
    elif filled_a > 0 and filled_b > 0:
        status = "PARTIAL"
    else:
        status = "LEGGED"

    realized_pnl = None
    if hedge_size > 0 and len(fill_prices) >= 2:
        fee_total = config.kalshi_fee_per_contract + config.polymarket_fee_per_contract
        price_sum = fill_prices[0] + fill_prices[1]
        realized_pnl = hedge_size * (1.0 - price_sum - fee_total - config.buffer_per_contract)

    reason = ",".join(sorted(set(reasons))) if reasons else None
    return ShadowTradeRecord(
        opp_id=opp_id,
        game_key=opp.game_key,
        detected_ts=opp.ts,
        latency_ms=latency_ms,
        detected_edge=opp.edge_per_contract,
        detected_size=opp.size_max,
        status=status,
        reason=reason,
        realized_pnl=realized_pnl,
        legs=leg_results,
    )


def _fill_leg(
    leg,
    manager: BookManager,
    config: Phase6Config,
    requested_size: float,
) -> ShadowLegRecord:
    book = manager.get_book(leg.venue, leg.market_id)
    if not book or not manager.is_healthy(leg.venue, leg.market_id):
        return ShadowLegRecord(
            venue=leg.venue,
            market_id=leg.market_id,
            outcome=leg.outcome,
            limit_price=leg.limit_price,
            intended_price=None,
            filled_price=None,
            filled_size=0.0,
            status="NO_BOOK",
            reason="stale_book",
        )

    best_ask = book.best_ask[0] if book.best_ask else None
    best_size = book.best_ask_size()
    if best_ask is None or best_size is None:
        return ShadowLegRecord(
            venue=leg.venue,
            market_id=leg.market_id,
            outcome=leg.outcome,
            limit_price=leg.limit_price,
            intended_price=best_ask,
            filled_price=None,
            filled_size=0.0,
            status="NO_ASK",
            reason="missing_ask",
        )

    effective_price = best_ask + (config.slippage_ticks * config.tick_size)
    if effective_price > leg.limit_price:
        return ShadowLegRecord(
            venue=leg.venue,
            market_id=leg.market_id,
            outcome=leg.outcome,
            limit_price=leg.limit_price,
            intended_price=best_ask,
            filled_price=None,
            filled_size=0.0,
            status="NO_FILL",
            reason="price_moved",
        )

    fill_size = min(requested_size, float(best_size))
    status = "FULL" if fill_size >= requested_size else "PARTIAL"
    return ShadowLegRecord(
        venue=leg.venue,
        market_id=leg.market_id,
        outcome=leg.outcome,
        limit_price=leg.limit_price,
        intended_price=best_ask,
        filled_price=effective_price,
        filled_size=fill_size,
        status=status,
        reason=None,
    )


def persist_shadow_trade(record: ShadowTradeRecord) -> None:
    write_shadow_trade(record)
