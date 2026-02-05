"""Maker-taker quoting and shadow hedge simulation."""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass

from cross_venue_arb.books.manager import BookManager
from cross_venue_arb.storage.mapping_registry import GameMappingRecord
from cross_venue_arb.storage.shadow_trades import ShadowLegRecord, ShadowTradeRecord


logger = logging.getLogger("maker_taker")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[MAKER_TAKER] %(asctime)s %(message)s"))
    logger.addHandler(handler)


@dataclass(frozen=True)
class MakerTakerConfig:
    edge_target: float = 0.005
    buffer_per_contract: float = 0.003
    kalshi_fee_per_contract: float = 0.001
    polymarket_fee_per_contract: float = 0.0
    tick_size: float = 0.01
    quote_size: float = 1.0
    ttl_seconds: float = 15.0
    min_reprice_interval_s: float = 1.0
    reprice_threshold_ticks: int = 2
    allow_reprice_up: bool = True
    fresh_s: float = 1.0
    sync_s: float = 0.25
    use_dirty_sync: bool = True


@dataclass(frozen=True)
class QuoteIntent:
    game_key: str
    team_to_buy: str
    hedge_team: str
    kalshi_ticker: str
    poly_asset_id: str
    price: float
    size: float
    poly_ask: float
    edge_per_contract: float
    reason: str | None = None


@dataclass
class KalshiOrderState:
    order_id: str
    game_key: str
    team_norm: str
    hedge_team: str
    ticker: str
    poly_asset_id: str
    side: str
    price: float
    size: float
    status: str
    created_ts: float
    ttl_seconds: float
    filled_size: float = 0.0
    avg_fill_price: float | None = None
    edge_per_contract: float = 0.0
    poly_ask_at_quote: float = 0.0


@dataclass
class GameState:
    status: str = "IDLE"
    active_order: KalshiOrderState | None = None
    last_reprice_ts: float | None = None
    last_intent: QuoteIntent | None = None
    last_decision_ts: float | None = None


class MakerTakerCoordinator:
    def __init__(self, config: MakerTakerConfig, *, max_active_global: int = 5) -> None:
        self._config = config
        self._max_active_global = max_active_global
        self._states: dict[str, GameState] = {}
        self._placed_count = 0
        self._fill_count = 0
        self._status_counts: dict[str, int] = {}
        self._counter_window_start = time.monotonic()
        self._counter_interval_s = 600.0

    def step_game(self, record: GameMappingRecord, manager: BookManager) -> list[ShadowTradeRecord]:
        now = time.monotonic()
        state = self._states.setdefault(record.game_key, GameState())
        trades: list[ShadowTradeRecord] = []

        if state.active_order is None:
            if self._active_order_count() >= self._max_active_global:
                return trades
            intent = _pick_best_intent(record, manager, self._config, state.last_decision_ts)
            state.last_decision_ts = now
            if intent is None or intent.reason:
                state.last_intent = intent
                return trades
            logger.info(
                "intent game=%s team=%s kalshi_ticker=%s poly_asset_id=%s poly_ask=%.4f edge_per_contract=%.4f",
                intent.game_key,
                intent.team_to_buy,
                intent.kalshi_ticker,
                intent.poly_asset_id,
                intent.poly_ask,
                intent.edge_per_contract,
            )
            state.active_order = _place_order(intent, now, self._config)
            self._note_order_placed(now)
            state.status = "QUOTE_PLACED"
            state.last_reprice_ts = now
            state.last_intent = intent
            return trades

        order = state.active_order
        book = manager.get_book("kalshi", order.ticker)
        best_bid = book.best_bid[0] if book and book.best_bid else None
        best_ask = book.best_ask[0] if book and book.best_ask else None
        distance_ticks = None
        if best_bid is not None and self._config.tick_size > 0:
            distance_ticks = (best_bid - order.price) / self._config.tick_size
        fee_total = self._config.kalshi_fee_per_contract + self._config.polymarket_fee_per_contract
        bid_max = 1.0 - order.poly_ask_at_quote - fee_total - self._config.buffer_per_contract - self._config.edge_target
        logger.info(
            "active_order game=%s team=%s my_price=%.4f best_bid=%s best_ask=%s distance_ticks=%s age_s=%.2f poly_ask=%.4f bid_max=%.4f edge_per_contract=%.4f",
            order.game_key,
            order.team_norm,
            order.price,
            f"{best_bid:.4f}" if best_bid is not None else "n/a",
            f"{best_ask:.4f}" if best_ask is not None else "n/a",
            f"{distance_ticks:.2f}" if distance_ticks is not None else "n/a",
            now - order.created_ts,
            order.poly_ask_at_quote,
            bid_max,
            order.edge_per_contract,
        )
        if now - order.created_ts >= order.ttl_seconds:
            order.status = "EXPIRED"
            state.active_order = None
            state.status = "IDLE"
            return trades

        refreshed = _build_intent_for_order(
            record,
            manager,
            self._config,
            order,
            last_decision_ts=state.last_decision_ts,
        )
        state.last_decision_ts = now
        if refreshed is None or refreshed.reason:
            order.status = "CANCELED"
            state.active_order = None
            state.status = "IDLE"
            state.last_intent = refreshed
            return trades

        state.last_intent = refreshed
        if _should_reprice(order, refreshed, state.last_reprice_ts, now, self._config):
            state.active_order = _place_order(refreshed, now, self._config)
            self._note_order_placed(now)
            state.last_reprice_ts = now
            state.status = "QUOTE_PLACED"
            order.status = "CANCELED"
            return trades

        fill_size = _maybe_fill_order(order, manager)
        if fill_size is not None and fill_size > 0:
            fill_price = order.price
            order.filled_size += fill_size
            order.avg_fill_price = fill_price
            if order.filled_size >= order.size:
                order.status = "FILLED"
                state.active_order = None
                state.status = "IDLE"
            else:
                order.status = "PARTIAL"
                state.status = "HEDGE_PENDING"
            trade = _simulate_hedge(record, order, fill_price, fill_size, now, manager, self._config)
            trades.append(trade)
            self._note_trade(trade, now)
        return trades

    def _active_order_count(self) -> int:
        count = 0
        for state in self._states.values():
            order = state.active_order
            if order and order.status in {"OPEN", "PARTIAL"}:
                count += 1
        return count

    def _note_order_placed(self, now: float) -> None:
        self._placed_count += 1
        self._maybe_log_summary(now)

    def _note_trade(self, trade: ShadowTradeRecord, now: float) -> None:
        self._fill_count += 1
        status = trade.status.lower()
        self._status_counts[status] = self._status_counts.get(status, 0) + 1
        self._maybe_log_summary(now)

    def _maybe_log_summary(self, now: float) -> None:
        elapsed = now - self._counter_window_start
        if elapsed < self._counter_interval_s:
            return
        logger.info(
            "orders_placed count=%d fills=%d status=%s window_s=%d",
            self._placed_count,
            self._fill_count,
            ",".join(f"{k}:{v}" for k, v in sorted(self._status_counts.items())) or "none",
            int(elapsed),
        )
        self._placed_count = 0
        self._fill_count = 0
        self._status_counts = {}
        self._counter_window_start = now


def _pick_best_intent(
    record: GameMappingRecord,
    manager: BookManager,
    config: MakerTakerConfig,
    last_decision_ts: float | None,
) -> QuoteIntent | None:
    intents = build_quote_intents(record, manager, config, last_decision_ts)
    candidates = [intent for intent in intents if intent.reason is None]
    if not candidates:
        return intents[0] if intents else None
    return max(candidates, key=lambda item: (item.edge_per_contract, item.price))


def build_quote_intents(
    record: GameMappingRecord,
    manager: BookManager,
    config: MakerTakerConfig,
    last_decision_ts: float | None,
) -> list[QuoteIntent]:
    team_map = _kalshi_team_map(record.kalshi_team_markets)
    if len(team_map) < 2:
        return [QuoteIntent(record.game_key, "", "", "", "", 0.0, 0.0, 0.0, 0.0, "missing_teams")]

    teams = list(team_map.keys())[:2]
    team_a, team_b = teams[0], teams[1]
    poly_outcome_map = record.match_details.get("polymarket_outcome_map", {})
    poly_asset_a = poly_outcome_map.get(team_a)
    poly_asset_b = poly_outcome_map.get(team_b)

    intents: list[QuoteIntent] = []
    intents.append(
        _build_intent(
            record,
            manager,
            config,
            team_to_buy=team_a,
            hedge_team=team_b,
            poly_asset_id=poly_asset_b,
            last_decision_ts=last_decision_ts,
        )
    )
    intents.append(
        _build_intent(
            record,
            manager,
            config,
            team_to_buy=team_b,
            hedge_team=team_a,
            poly_asset_id=poly_asset_a,
            last_decision_ts=last_decision_ts,
        )
    )
    return intents


def _build_intent(
    record: GameMappingRecord,
    manager: BookManager,
    config: MakerTakerConfig,
    *,
    team_to_buy: str,
    hedge_team: str,
    poly_asset_id: str | None,
    last_decision_ts: float | None,
) -> QuoteIntent:
    kalshi_ticker = _kalshi_team_map(record.kalshi_team_markets).get(team_to_buy)
    if not kalshi_ticker:
        return QuoteIntent(record.game_key, team_to_buy, hedge_team, "", "", 0.0, 0.0, 0.0, 0.0, "missing_kalshi_ticker")
    if not poly_asset_id:
        return QuoteIntent(
            record.game_key,
            team_to_buy,
            hedge_team,
            kalshi_ticker,
            "",
            0.0,
            0.0,
            0.0,
            0.0,
            "missing_polymarket_asset",
        )

    if not manager.is_healthy("kalshi", kalshi_ticker):
        return QuoteIntent(
            record.game_key,
            team_to_buy,
            hedge_team,
            kalshi_ticker,
            poly_asset_id,
            0.0,
            0.0,
            0.0,
            0.0,
            "stale_kalshi_book",
        )
    if not manager.is_healthy("polymarket", poly_asset_id):
        return QuoteIntent(
            record.game_key,
            team_to_buy,
            hedge_team,
            kalshi_ticker,
            poly_asset_id,
            0.0,
            0.0,
            0.0,
            0.0,
            "stale_polymarket_book",
        )

    book_kalshi = manager.get_book("kalshi", kalshi_ticker)
    book_poly = manager.get_book("polymarket", poly_asset_id)
    if not book_kalshi or not book_poly:
        return QuoteIntent(
            record.game_key,
            team_to_buy,
            hedge_team,
            kalshi_ticker,
            poly_asset_id,
            0.0,
            0.0,
            0.0,
            0.0,
            "missing_book",
        )

    if not _books_synced(book_kalshi, book_poly, time.monotonic(), config, last_decision_ts):
        return QuoteIntent(
            record.game_key,
            team_to_buy,
            hedge_team,
            kalshi_ticker,
            poly_asset_id,
            0.0,
            0.0,
            0.0,
            0.0,
            "unsynced_books",
        )

    best_bid = book_kalshi.best_bid[0] if book_kalshi.best_bid else None
    best_ask = book_kalshi.best_ask[0] if book_kalshi.best_ask else None
    poly_ask = book_poly.best_ask[0] if book_poly.best_ask else None
    if best_bid is None or best_ask is None:
        return QuoteIntent(
            record.game_key,
            team_to_buy,
            hedge_team,
            kalshi_ticker,
            poly_asset_id,
            0.0,
            0.0,
            0.0,
            0.0,
            "missing_kalshi_quotes",
        )
    if poly_ask is None:
        return QuoteIntent(
            record.game_key,
            team_to_buy,
            hedge_team,
            kalshi_ticker,
            poly_asset_id,
            0.0,
            0.0,
            0.0,
            0.0,
            "missing_polymarket_ask",
        )

    fee_total = config.kalshi_fee_per_contract + config.polymarket_fee_per_contract
    bid_max = 1.0 - poly_ask - fee_total - config.buffer_per_contract - config.edge_target
    if bid_max <= 0:
        return QuoteIntent(
            record.game_key,
            team_to_buy,
            hedge_team,
            kalshi_ticker,
            poly_asset_id,
            0.0,
            0.0,
            poly_ask,
            0.0,
            "edge_unachievable",
        )

    price_cap = min(bid_max, best_ask - config.tick_size)
    if price_cap <= 0:
        return QuoteIntent(
            record.game_key,
            team_to_buy,
            hedge_team,
            kalshi_ticker,
            poly_asset_id,
            0.0,
            0.0,
            poly_ask,
            0.0,
            "crosses_spread",
        )

    quote_price = _round_down_to_tick(price_cap, config.tick_size)
    if quote_price < (best_bid + config.tick_size):
        return QuoteIntent(
            record.game_key,
            team_to_buy,
            hedge_team,
            kalshi_ticker,
            poly_asset_id,
            quote_price,
            0.0,
            poly_ask,
            0.0,
            "not_inside_spread",
        )

    edge_per_contract = 1.0 - (quote_price + poly_ask) - fee_total - config.buffer_per_contract
    return QuoteIntent(
        record.game_key,
        team_to_buy,
        hedge_team,
        kalshi_ticker,
        poly_asset_id,
        quote_price,
        config.quote_size,
        poly_ask,
        edge_per_contract,
    )


def _build_intent_for_order(
    record: GameMappingRecord,
    manager: BookManager,
    config: MakerTakerConfig,
    order: KalshiOrderState,
    *,
    last_decision_ts: float | None,
) -> QuoteIntent | None:
    return _build_intent(
        record,
        manager,
        config,
        team_to_buy=order.team_norm,
        hedge_team=order.hedge_team,
        poly_asset_id=order.poly_asset_id,
        last_decision_ts=last_decision_ts,
    )


def _place_order(intent: QuoteIntent, now: float, config: MakerTakerConfig) -> KalshiOrderState:
    return KalshiOrderState(
        order_id=uuid.uuid4().hex,
        game_key=intent.game_key,
        team_norm=intent.team_to_buy,
        hedge_team=intent.hedge_team,
        ticker=intent.kalshi_ticker,
        poly_asset_id=intent.poly_asset_id,
        side="BUY",
        price=intent.price,
        size=intent.size,
        status="OPEN",
        created_ts=now,
        ttl_seconds=config.ttl_seconds,
        filled_size=0.0,
        avg_fill_price=None,
        edge_per_contract=intent.edge_per_contract,
        poly_ask_at_quote=intent.poly_ask,
    )


def _should_reprice(
    order: KalshiOrderState,
    intent: QuoteIntent,
    last_reprice_ts: float | None,
    now: float,
    config: MakerTakerConfig,
) -> bool:
    if last_reprice_ts is None:
        return False
    if now - last_reprice_ts < config.min_reprice_interval_s:
        return False
    threshold = config.reprice_threshold_ticks * config.tick_size
    if intent.price + threshold <= order.price:
        return True
    if config.allow_reprice_up and intent.price >= order.price + threshold:
        return True
    return False


def _maybe_fill_order(order: KalshiOrderState, manager: BookManager) -> float | None:
    book = manager.get_book("kalshi", order.ticker)
    if not book or not manager.is_healthy("kalshi", order.ticker):
        return None
    best_ask = book.best_ask[0] if book.best_ask else None
    best_ask_size = book.best_ask_size()
    if best_ask is None or best_ask_size is None:
        return None
    if best_ask > order.price:
        return None
    remaining = max(order.size - order.filled_size, 0.0)
    return min(remaining, float(best_ask_size))


def _simulate_hedge(
    record: GameMappingRecord,
    order: KalshiOrderState,
    fill_price: float,
    fill_size: float,
    now: float,
    manager: BookManager,
    config: MakerTakerConfig,
) -> ShadowTradeRecord:
    poly_book = manager.get_book("polymarket", order.poly_asset_id)
    hedge_price = None
    hedge_size = 0.0
    reason_parts: list[str] = []

    if not poly_book or not manager.is_healthy("polymarket", order.poly_asset_id):
        reason_parts.append("stale_polymarket_book")
    else:
        poly_ask = poly_book.best_ask[0] if poly_book.best_ask else None
        poly_size = poly_book.best_ask_size()
        if poly_ask is None or poly_size is None:
            reason_parts.append("missing_polymarket_ask")
        else:
            hedge_price = poly_ask
            hedge_size = min(fill_size, float(poly_size))
            if hedge_size < fill_size:
                reason_parts.append("hedge_size_short")

    realized_edge = None
    realized_pnl = None
    status = "HEDGE_MISSED"
    if hedge_price is not None and hedge_size > 0:
        fee_total = config.kalshi_fee_per_contract + config.polymarket_fee_per_contract
        realized_edge = 1.0 - (fill_price + hedge_price) - fee_total - config.buffer_per_contract
        realized_pnl = hedge_size * realized_edge
        if realized_edge < 0:
            reason_parts.append("adverse_selection")
        status = "HEDGED" if hedge_size == fill_size else "PARTIAL_HEDGE"
    latency_ms = int((now - order.created_ts) * 1000)
    reason = ",".join(sorted(set(reason_parts))) if reason_parts else None
    legs = [
        ShadowLegRecord(
            venue="kalshi",
            market_id=order.ticker,
            outcome=order.team_norm,
            limit_price=order.price,
            intended_price=order.price,
            filled_price=fill_price,
            filled_size=fill_size,
            status="FILLED",
            reason=None,
        ),
        ShadowLegRecord(
            venue="polymarket",
            market_id=order.poly_asset_id,
            outcome=order.hedge_team,
            limit_price=hedge_price or 0.0,
            intended_price=hedge_price,
            filled_price=hedge_price,
            filled_size=hedge_size,
            status=status,
            reason=reason,
        ),
    ]
    return ShadowTradeRecord(
        opp_id=uuid.uuid4().hex,
        game_key=record.game_key,
        detected_ts=order.created_ts,
        latency_ms=latency_ms,
        detected_edge=order.edge_per_contract,
        detected_size=fill_size,
        status=status,
        reason=reason,
        realized_pnl=realized_pnl,
        legs=legs,
    )


def _kalshi_team_map(team_markets: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in team_markets:
        team = entry.get("team_norm")
        ticker = entry.get("ticker")
        if team and ticker:
            mapping[team] = ticker
    return mapping


def _books_synced(
    book_a: object,
    book_b: object,
    now: float,
    config: MakerTakerConfig,
    last_decision_ts: float | None,
) -> bool:
    ts_a = getattr(book_a, "last_update_ts", None)
    ts_b = getattr(book_b, "last_update_ts", None)
    if ts_a is None or ts_b is None:
        return False
    if now - ts_a > config.fresh_s:
        return False
    if now - ts_b > config.fresh_s:
        return False
    if config.use_dirty_sync:
        if last_decision_ts is None:
            return True
        return ts_a > last_decision_ts and ts_b > last_decision_ts
    if config.sync_s > 0 and abs(ts_a - ts_b) > config.sync_s:
        return False
    return True


def _round_down_to_tick(price: float, tick_size: float) -> float:
    if tick_size <= 0:
        return price
    return math.floor(price / tick_size) * tick_size
