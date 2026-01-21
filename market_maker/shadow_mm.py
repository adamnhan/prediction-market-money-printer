from __future__ import annotations

import asyncio
import csv
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, Tuple

from cross_venue_arb.books.manager import BookManager

from market_maker.config import ShadowMMConfig
from market_maker.kalshi_ws_client import WsMetrics


logger = logging.getLogger("shadow_mm")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[SHADOW_MM] %(asctime)s %(message)s"))
    logger.addHandler(handler)


@dataclass
class MarketStats:
    ticks: int = 0
    halted_ticks: int = 0
    inv_zero_ticks: int = 0
    inv_max_ticks: int = 0
    fills: int = 0
    buy_fills: int = 0
    sell_fills: int = 0


@dataclass
class Quote:
    side: str
    price_cents: float
    size: int
    created_ts: float
    last_fill_ts: float | None = None
    touch_start_ts: float | None = None


@dataclass
class AdverseRecord:
    ticker: str
    side: str
    fill_ts: float
    mid_at_fill: float
    quote_price_cents: float
    targets: Dict[str, float]
    results: Dict[str, float] = field(default_factory=dict)


@dataclass
class MarketState:
    inventory: int = 0
    cash_cents: float = 0.0
    avg_cost_cents: float = 0.0
    realized_pnl_cents: float = 0.0
    fees_paid_cents: float = 0.0
    halted_until: float = 0.0
    halt_reason: str | None = None
    hold_start_ts: float | None = None
    peak_net_pnl_cents: float = 0.0
    max_drawdown_cents: float = 0.0
    mid_history: Deque[Tuple[float, float]] = field(default_factory=deque)
    fill_history: Deque[float] = field(default_factory=deque)
    adverse_pending: Deque[AdverseRecord] = field(default_factory=deque)
    stats: MarketStats = field(default_factory=MarketStats)
    last_spread_cents: float | None = None
    bid_quote: Quote | None = None
    ask_quote: Quote | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_log_headers(path: Path, headers: Iterable[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(headers))


def _clamp_cents(value: float) -> float:
    return max(1.0, min(99.0, value))


def _spread_cents(bid: float, ask: float) -> float:
    return (ask - bid) * 100.0


def _mid_cents(bid: float, ask: float) -> float:
    return (bid + ask) * 50.0


def _compute_quotes(
    mid_cents: float,
    inventory: int,
    config: ShadowMMConfig,
) -> tuple[float | None, float | None]:
    half_spread = config.base_spread_cents / 2.0
    skew = config.k_cents_per_contract * (inventory - config.itarget)
    bid_cents = mid_cents - half_spread - skew
    ask_cents = mid_cents + half_spread + skew
    bid = None
    ask = None
    if inventory < config.imax:
        bid = _clamp_cents(bid_cents)
    if inventory > 0:
        ask = _clamp_cents(ask_cents)
    return bid, ask


def _update_volatility_halt(
    state: MarketState,
    mid_cents: float,
    now: float,
    config: ShadowMMConfig,
) -> bool:
    state.mid_history.append((now, mid_cents))
    while state.mid_history and now - state.mid_history[0][0] > config.vol_window_s:
        state.mid_history.popleft()
    if state.mid_history:
        oldest_ts, oldest_mid = state.mid_history[0]
        if now - oldest_ts >= config.vol_window_s:
            if abs(mid_cents - oldest_mid) > config.vol_move_cents:
                state.halted_until = now + config.halt_s
                return True
    return False


def _append_quote_log(
    path: Path,
    row: list[object],
) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)


def _append_summary_log(
    path: Path,
    row: list[object],
) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)


def _append_fill_log(path: Path, row: list[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)


def _append_adverse_log(path: Path, row: list[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)


def _price_close(a: float, b: float, tick_cents: float) -> bool:
    return abs(a - b) <= tick_cents


def _compute_fill_probability(
    spread_cents: float,
    best_size: float | None,
    mid_move_cents: float,
    config: ShadowMMConfig,
) -> float:
    size_bonus = 0.0
    if best_size is not None:
        size_bonus = max(0.0, (1.0 / max(1.0, best_size)) - 0.01) * 100.0
    p = config.p_fill_base
    p += config.p_fill_spread_bonus_per_cent * max(0.0, spread_cents - config.min_spread_cents)
    p += config.p_fill_size_bonus_per_contract * size_bonus
    p -= config.p_fill_vol_penalty_per_cent * max(0.0, mid_move_cents)
    return max(0.0, min(1.0, p))


def _quote_is_stale(quote: Quote, now: float, config: ShadowMMConfig) -> bool:
    return (now - quote.created_ts) >= config.quote_ttl_s


def _maybe_replace_quote(
    quote: Quote | None,
    side: str,
    new_price_cents: float | None,
    now: float,
    config: ShadowMMConfig,
) -> Quote | None:
    if new_price_cents is None:
        return None
    if quote is None:
        return Quote(side=side, price_cents=new_price_cents, size=config.quote_size, created_ts=now)
    if _quote_is_stale(quote, now, config):
        return Quote(side=side, price_cents=new_price_cents, size=config.quote_size, created_ts=now)
    if abs(new_price_cents - quote.price_cents) >= config.top_of_book_tick_cents:
        return Quote(side=side, price_cents=new_price_cents, size=config.quote_size, created_ts=now)
    return quote


def _book_mid_move(state: MarketState, now: float, window_s: float) -> float:
    if not state.mid_history:
        return 0.0
    oldest_ts, oldest_mid = state.mid_history[0]
    if now - oldest_ts < window_s:
        return 0.0
    return abs(state.mid_history[-1][1] - oldest_mid)


def _prune_fill_history(state: MarketState, now: float) -> None:
    while state.fill_history and now - state.fill_history[0] > 60.0:
        state.fill_history.popleft()


def _record_adverse(
    state: MarketState,
    ticker: str,
    side: str,
    fill_ts: float,
    mid_at_fill: float,
    quote_price_cents: float,
) -> None:
    targets = {
        "1s": fill_ts + 1.0,
        "5s": fill_ts + 5.0,
        "30s": fill_ts + 30.0,
    }
    state.adverse_pending.append(
        AdverseRecord(
            ticker=ticker,
            side=side,
            fill_ts=fill_ts,
            mid_at_fill=mid_at_fill,
            quote_price_cents=quote_price_cents,
            targets=targets,
        )
    )


def _maybe_finalize_adverse(
    state: MarketState,
    adverse_path: Path,
    now: float,
) -> None:
    if not state.adverse_pending:
        return
    for _ in range(len(state.adverse_pending)):
        record = state.adverse_pending.popleft()
        for label, target_ts in record.targets.items():
            if label in record.results:
                continue
            for ts, mid in state.mid_history:
                if ts >= target_ts:
                    record.results[label] = mid
                    break
        if len(record.results) < len(record.targets):
            state.adverse_pending.append(record)
            continue
        adverse_1s = record.results["1s"] - record.mid_at_fill
        adverse_5s = record.results["5s"] - record.mid_at_fill
        adverse_30s = record.results["30s"] - record.mid_at_fill
        if record.side == "ask":
            adverse_1s *= -1.0
            adverse_5s *= -1.0
            adverse_30s *= -1.0
        _append_adverse_log(
            adverse_path,
            [
                _utc_now(),
                record.ticker,
                record.side,
                round(record.quote_price_cents, 2),
                round(record.mid_at_fill, 2),
                round(record.results["1s"], 2),
                round(record.results["5s"], 2),
                round(record.results["30s"], 2),
                round(adverse_1s, 4),
                round(adverse_5s, 4),
                round(adverse_30s, 4),
            ],
        )


def _apply_fee(size: int, config: ShadowMMConfig) -> float:
    return float(size) * config.fee_per_contract_cents


def _mark_price_cents(best_bid: float | None, best_ask: float | None, config: ShadowMMConfig) -> float | None:
    if best_bid is None or best_ask is None:
        return None
    if config.mark_price_mode == "mid":
        return _mid_cents(best_bid, best_ask)
    return best_bid * 100.0


def _net_pnl_cents(state: MarketState, mark_cents: float | None) -> float:
    unrealized = 0.0
    if mark_cents is not None and state.inventory != 0:
        unrealized = state.inventory * (mark_cents - state.avg_cost_cents)
    return state.realized_pnl_cents + unrealized - state.fees_paid_cents


async def quote_loop(
    manager: BookManager,
    tickers: Iterable[str],
    config: ShadowMMConfig,
    metrics: WsMetrics,
) -> None:
    states: Dict[str, MarketState] = {ticker: MarketState() for ticker in tickers}
    quote_path = Path(config.log_dir) / "quotes.csv"
    summary_path = Path(config.log_dir) / "summary.csv"
    fills_path = Path(config.log_dir) / "fills.csv"
    adverse_path = Path(config.log_dir) / "adverse.csv"
    market_summary_path = Path(config.log_dir) / "market_summary.csv"
    roundtrip_path = Path(config.log_dir) / "round_trips.csv"
    _ensure_log_headers(
        quote_path,
        [
            "ts_utc",
            "market_ticker",
            "best_bid",
            "best_ask",
            "spread_cents",
            "fair_cents",
            "bid_quote_cents",
            "ask_quote_cents",
            "bid_quote_age_s",
            "ask_quote_age_s",
            "inventory",
            "enabled",
            "block_reason",
        ],
    )
    _ensure_log_headers(
        summary_path,
        [
            "ts_utc",
            "fills",
            "pnl_cents",
            "avg_spread_cents",
            "pct_halted",
            "inv_zero_pct",
            "inv_max_pct",
            "update_rate_mps",
            "reconnects",
            "total_net_pnl_cents",
            "total_fees_cents",
            "total_exposure_cents",
        ],
    )
    _ensure_log_headers(
        fills_path,
        [
            "ts_utc",
            "market_ticker",
            "side",
            "fill_price_cents",
            "size",
            "mid_at_fill",
            "quote_price_cents",
            "fill_reason",
        ],
    )
    _ensure_log_headers(
        adverse_path,
        [
            "ts_utc",
            "market_ticker",
            "side",
            "quote_price_cents",
            "mid_at_fill",
            "mid_1s",
            "mid_5s",
            "mid_30s",
            "adverse_1s",
            "adverse_5s",
            "adverse_30s",
        ],
    )
    _ensure_log_headers(
        market_summary_path,
        [
            "ts_utc",
            "market_ticker",
            "inventory",
            "avg_cost_cents",
            "best_bid",
            "best_ask",
            "mid_cents",
            "spread_cents",
            "realized_pnl_cents",
            "unrealized_pnl_cents",
            "fees_paid_cents",
            "net_pnl_cents",
            "fills_last_minute",
            "halted",
            "halt_reason",
            "exposure_cents",
            "inv_concentration",
            "hold_age_s",
            "max_drawdown_cents",
        ],
    )
    _ensure_log_headers(
        roundtrip_path,
        [
            "ts_utc",
            "market_ticker",
            "entry_ts_utc",
            "exit_ts_utc",
            "hold_s",
            "avg_cost_cents",
            "exit_price_cents",
            "qty",
            "gross_edge_cents",
        ],
    )

    last_summary_ts = time.monotonic()
    last_health_ts = time.monotonic()
    last_msg_count = metrics.msg_count
    last_summary_msg_count = metrics.msg_count

    while True:
        loop_start = time.monotonic()
        total_spread = 0.0
        spread_samples = 0
        total_net_pnl = 0.0
        total_fees = 0.0
        total_exposure = 0.0
        total_inventory = 0

        market_snapshots: dict[str, dict[str, object]] = {}

        for ticker in tickers:
            state = states[ticker]
            state.stats.ticks += 1

            book = manager.get_book("kalshi", ticker)
            now = time.monotonic()
            best_bid = book.best_bid[0] if book and book.best_bid else None
            best_ask = book.best_ask[0] if book and book.best_ask else None
            mark_cents = _mark_price_cents(best_bid, best_ask, config)
            if now >= state.halted_until:
                state.halt_reason = None

            enabled = False
            block_reason = ""
            intended_bid = None
            intended_ask = None
            bid_age = None
            ask_age = None

            if best_bid is None or best_ask is None:
                block_reason = "no_book"
            elif best_bid >= best_ask:
                block_reason = "crossed"
            else:
                spread_cents = _spread_cents(best_bid, best_ask)
                state.last_spread_cents = spread_cents
                total_spread += spread_cents
                spread_samples += 1
                state.mid_history.append((now, _mid_cents(best_bid, best_ask)))
                while state.mid_history and now - state.mid_history[0][0] > 60.0:
                    state.mid_history.popleft()
                if spread_cents < config.min_spread_cents:
                    block_reason = "spread_too_tight"
                else:
                    age = manager.last_update_age("kalshi", ticker)
                    if age is None or age > config.staleness_s:
                        block_reason = "stale"
                    else:
                        fair_cents = _mid_cents(best_bid, best_ask)
                        triggered = _update_volatility_halt(state, fair_cents, now, config)
                        if triggered:
                            block_reason = "vol_halt_triggered"
                            state.halt_reason = "vol_halt"
                        elif now < state.halted_until:
                            block_reason = state.halt_reason or "vol_halt_active"
                        else:
                            intended_bid, intended_ask = _compute_quotes(
                                fair_cents, state.inventory, config
                            )
                            # Max-hold and stop-loss rules.
                            hold_age = None
                            if state.hold_start_ts is not None:
                                hold_age = now - state.hold_start_ts
                            net_pnl = _net_pnl_cents(state, mark_cents)
                            if hold_age is not None and hold_age >= config.max_hold_s:
                                block_reason = "max_hold"
                                state.halted_until = max(state.halted_until, now + config.halt_s)
                                state.halt_reason = "max_hold"
                                if config.force_unwind_on_max_hold and state.inventory > 0:
                                    intended_bid = None
                                    if best_bid is not None:
                                        fill_price = best_bid * 100.0
                                        size = state.inventory
                                        fee = _apply_fee(size, config)
                                        state.realized_pnl_cents += (
                                            fill_price - state.avg_cost_cents
                                        ) * size
                                        state.cash_cents += fill_price * size
                                        state.fees_paid_cents += fee
                                        state.inventory = 0
                                        state.stats.fills += 1
                                        state.stats.sell_fills += 1
                                        state.fill_history.append(now)
                                        _append_fill_log(
                                            fills_path,
                                            [
                                                _utc_now(),
                                                ticker,
                                                "ask",
                                                round(fill_price, 2),
                                                size,
                                                round(fair_cents, 2),
                                                round(fill_price, 2),
                                                "force_unwind",
                                            ],
                                        )
                                        _record_adverse(
                                            state,
                                            ticker,
                                            "ask",
                                            now,
                                            fair_cents,
                                            fill_price,
                                        )
                                        if state.hold_start_ts is not None:
                                            hold_s = now - state.hold_start_ts
                                            _append_roundtrip_log(
                                                roundtrip_path,
                                                [
                                                    _utc_now(),
                                                    ticker,
                                                    datetime.fromtimestamp(
                                                        state.hold_start_ts, tz=timezone.utc
                                                    ).isoformat(),
                                                    _utc_now(),
                                                    round(hold_s, 2),
                                                    round(state.avg_cost_cents, 2),
                                                    round(fill_price, 2),
                                                    size,
                                                    round(fill_price - state.avg_cost_cents, 2),
                                                ],
                                            )
                                        state.avg_cost_cents = 0.0
                                        state.hold_start_ts = None
                            if net_pnl <= -config.stop_loss_cents:
                                block_reason = "stop_loss"
                                state.halted_until = max(state.halted_until, now + config.stop_loss_halt_s)
                                state.halt_reason = "stop_loss"
                                if config.force_unwind_on_stop_loss and state.inventory > 0:
                                    intended_bid = None
                                    if best_bid is not None:
                                        fill_price = best_bid * 100.0
                                        size = state.inventory
                                        fee = _apply_fee(size, config)
                                        state.realized_pnl_cents += (
                                            fill_price - state.avg_cost_cents
                                        ) * size
                                        state.cash_cents += fill_price * size
                                        state.fees_paid_cents += fee
                                        state.inventory = 0
                                        state.stats.fills += 1
                                        state.stats.sell_fills += 1
                                        state.fill_history.append(now)
                                        _append_fill_log(
                                            fills_path,
                                            [
                                                _utc_now(),
                                                ticker,
                                                "ask",
                                                round(fill_price, 2),
                                                size,
                                                round(fair_cents, 2),
                                                round(fill_price, 2),
                                                "force_unwind",
                                            ],
                                        )
                                        _record_adverse(
                                            state,
                                            ticker,
                                            "ask",
                                            now,
                                            fair_cents,
                                            fill_price,
                                        )
                                        if state.hold_start_ts is not None:
                                            hold_s = now - state.hold_start_ts
                                            _append_roundtrip_log(
                                                roundtrip_path,
                                                [
                                                    _utc_now(),
                                                    ticker,
                                                    datetime.fromtimestamp(
                                                        state.hold_start_ts, tz=timezone.utc
                                                    ).isoformat(),
                                                    _utc_now(),
                                                    round(hold_s, 2),
                                                    round(state.avg_cost_cents, 2),
                                                    round(fill_price, 2),
                                                    size,
                                                    round(fill_price - state.avg_cost_cents, 2),
                                                ],
                                            )
                                        state.avg_cost_cents = 0.0
                                        state.hold_start_ts = None

                            state.bid_quote = _maybe_replace_quote(
                                state.bid_quote, "bid", intended_bid, now, config
                            )
                            state.ask_quote = _maybe_replace_quote(
                                state.ask_quote, "ask", intended_ask, now, config
                            )
                            enabled = state.bid_quote is not None or state.ask_quote is not None
                            if not enabled:
                                block_reason = "inventory_limit"

                            # Shadow fills: conservative maker model.
                            _prune_fill_history(state, now)
                            if len(state.fill_history) < config.max_fills_per_minute:
                                best_bid_cents = best_bid * 100.0
                                best_ask_cents = best_ask * 100.0
                                mid_cents = fair_cents
                                mid_move_cents = _book_mid_move(state, now, config.vol_window_s)

                                if state.bid_quote and state.inventory < config.imax:
                                    bid_quote = state.bid_quote
                                    bid_age = now - bid_quote.created_ts
                                    top_ok = _price_close(
                                        bid_quote.price_cents,
                                        best_bid_cents,
                                        config.top_of_book_tick_cents,
                                    )
                                    touch_ok = best_ask_cents <= bid_quote.price_cents
                                    cross_ok = bid_quote.price_cents >= best_ask_cents
                                    if touch_ok:
                                        if bid_quote.touch_start_ts is None:
                                            bid_quote.touch_start_ts = now
                                    else:
                                        bid_quote.touch_start_ts = None

                                    touch_ready = (
                                        bid_quote.touch_start_ts is not None
                                        and (now - bid_quote.touch_start_ts) * 1000.0 >= config.touch_ms
                                    )
                                    if cross_ok or (touch_ready and top_ok):
                                        if bid_quote.last_fill_ts is None or (
                                            now - bid_quote.last_fill_ts
                                        ) >= 1.0 / config.max_fills_per_quote_per_s:
                                            p_fill = _compute_fill_probability(
                                                spread_cents,
                                                book.best_ask[1] if book and book.best_ask else None,
                                                mid_move_cents,
                                                config,
                                            )
                                            if cross_ok or random.random() < p_fill:
                                                fill_price = best_ask_cents
                                                fee = _apply_fee(bid_quote.size, config)
                                                total_cost = fill_price * bid_quote.size
                                                new_inv = state.inventory + bid_quote.size
                                                if state.inventory == 0:
                                                    state.avg_cost_cents = fill_price
                                                    state.hold_start_ts = now
                                                else:
                                                    state.avg_cost_cents = (
                                                        state.avg_cost_cents * state.inventory + total_cost
                                                    ) / new_inv
                                                state.inventory = new_inv
                                                state.cash_cents -= total_cost
                                                state.fees_paid_cents += fee
                                                state.stats.fills += 1
                                                state.stats.buy_fills += 1
                                                state.fill_history.append(now)
                                                bid_quote.last_fill_ts = now
                                                _append_fill_log(
                                                    fills_path,
                                                    [
                                                        _utc_now(),
                                                        ticker,
                                                        "bid",
                                                        round(fill_price, 2),
                                                        bid_quote.size,
                                                        round(mid_cents, 2),
                                                        round(bid_quote.price_cents, 2),
                                                        "cross" if cross_ok else "touch",
                                                    ],
                                                )
                                                _record_adverse(
                                                    state,
                                                    ticker,
                                                    "bid",
                                                    now,
                                                    mid_cents,
                                                    bid_quote.price_cents,
                                                )

                                if state.ask_quote and state.inventory > 0:
                                    ask_quote = state.ask_quote
                                    ask_age = now - ask_quote.created_ts
                                    top_ok = _price_close(
                                        ask_quote.price_cents,
                                        best_ask_cents,
                                        config.top_of_book_tick_cents,
                                    )
                                    touch_ok = best_bid_cents >= ask_quote.price_cents
                                    cross_ok = ask_quote.price_cents <= best_bid_cents
                                    if touch_ok:
                                        if ask_quote.touch_start_ts is None:
                                            ask_quote.touch_start_ts = now
                                    else:
                                        ask_quote.touch_start_ts = None

                                    touch_ready = (
                                        ask_quote.touch_start_ts is not None
                                        and (now - ask_quote.touch_start_ts) * 1000.0 >= config.touch_ms
                                    )
                                    if cross_ok or (touch_ready and top_ok):
                                        if ask_quote.last_fill_ts is None or (
                                            now - ask_quote.last_fill_ts
                                        ) >= 1.0 / config.max_fills_per_quote_per_s:
                                            p_fill = _compute_fill_probability(
                                                spread_cents,
                                                book.best_bid[1] if book and book.best_bid else None,
                                                mid_move_cents,
                                                config,
                                            )
                                            if cross_ok or random.random() < p_fill:
                                                fill_price = best_bid_cents
                                                fee = _apply_fee(ask_quote.size, config)
                                                proceeds = fill_price * ask_quote.size
                                                state.realized_pnl_cents += (
                                                    fill_price - state.avg_cost_cents
                                                ) * ask_quote.size
                                                state.inventory -= ask_quote.size
                                                state.cash_cents += proceeds
                                                state.fees_paid_cents += fee
                                                state.stats.fills += 1
                                                state.stats.sell_fills += 1
                                                state.fill_history.append(now)
                                                ask_quote.last_fill_ts = now
                                                _append_fill_log(
                                                    fills_path,
                                                    [
                                                        _utc_now(),
                                                        ticker,
                                                        "ask",
                                                        round(fill_price, 2),
                                                        ask_quote.size,
                                                        round(mid_cents, 2),
                                                        round(ask_quote.price_cents, 2),
                                                        "cross" if cross_ok else "touch",
                                                    ],
                                                )
                                                _record_adverse(
                                                    state,
                                                    ticker,
                                                    "ask",
                                                    now,
                                                    mid_cents,
                                                    ask_quote.price_cents,
                                                )
                                                if state.inventory == 0 and state.hold_start_ts is not None:
                                                    hold_s = now - state.hold_start_ts
                                                    _append_roundtrip_log(
                                                        roundtrip_path,
                                                        [
                                                            _utc_now(),
                                                            ticker,
                                                            datetime.fromtimestamp(
                                                                state.hold_start_ts, tz=timezone.utc
                                                            ).isoformat(),
                                                            _utc_now(),
                                                            round(hold_s, 2),
                                                            round(state.avg_cost_cents, 2),
                                                            round(fill_price, 2),
                                                            ask_quote.size,
                                                            round(fill_price - state.avg_cost_cents, 2),
                                                        ],
                                                    )
                                                    state.avg_cost_cents = 0.0
                                                    state.hold_start_ts = None

            if not enabled:
                state.stats.halted_ticks += 1
            if state.inventory == 0:
                state.stats.inv_zero_ticks += 1
            if state.inventory >= config.imax:
                state.stats.inv_max_ticks += 1

            fair_cents = None
            if best_bid is not None and best_ask is not None:
                fair_cents = _mid_cents(best_bid, best_ask)

            if state.bid_quote:
                bid_age = now - state.bid_quote.created_ts
            if state.ask_quote:
                ask_age = now - state.ask_quote.created_ts

            _append_quote_log(
                quote_path,
                [
                    _utc_now(),
                    ticker,
                    best_bid,
                    best_ask,
                    state.last_spread_cents,
                    fair_cents,
                    state.bid_quote.price_cents if state.bid_quote else None,
                    state.ask_quote.price_cents if state.ask_quote else None,
                    round(bid_age, 3) if bid_age is not None else None,
                    round(ask_age, 3) if ask_age is not None else None,
                    state.inventory,
                    enabled,
                    block_reason,
                ],
            )
            _maybe_finalize_adverse(state, adverse_path, now)

            fills_last_minute = len(state.fill_history)
            unrealized = 0.0
            if mark_cents is not None and state.inventory != 0:
                unrealized = state.inventory * (mark_cents - state.avg_cost_cents)
            net_pnl = state.realized_pnl_cents + unrealized - state.fees_paid_cents
            if net_pnl > state.peak_net_pnl_cents:
                state.peak_net_pnl_cents = net_pnl
            drawdown = state.peak_net_pnl_cents - net_pnl
            if drawdown > state.max_drawdown_cents:
                state.max_drawdown_cents = drawdown
            exposure = state.inventory * max(0.0, (100.0 - state.avg_cost_cents))
            inv_concentration = state.inventory / max(1.0, config.imax)
            hold_age = (now - state.hold_start_ts) if state.hold_start_ts else None
            market_snapshots[ticker] = {
                "ts_utc": _utc_now(),
                "inventory": state.inventory,
                "avg_cost_cents": round(state.avg_cost_cents, 2),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_cents": round(fair_cents, 2) if fair_cents is not None else None,
                "spread_cents": state.last_spread_cents,
                "realized_pnl_cents": round(state.realized_pnl_cents, 2),
                "unrealized_pnl_cents": round(unrealized, 2),
                "fees_paid_cents": round(state.fees_paid_cents, 2),
                "net_pnl_cents": round(net_pnl, 2),
                "fills_last_minute": fills_last_minute,
                "halted": now < state.halted_until,
                "halt_reason": state.halt_reason,
                "exposure_cents": round(exposure, 2),
                "inv_concentration": round(inv_concentration, 4),
                "hold_age_s": round(hold_age, 2) if hold_age is not None else None,
                "max_drawdown_cents": round(state.max_drawdown_cents, 2),
            }
            total_net_pnl += net_pnl
            total_fees += state.fees_paid_cents
            total_exposure += exposure
            total_inventory += state.inventory

        now = time.monotonic()
        if now - last_health_ts >= config.health_every_s:
            valid_markets = [t for t in tickers if manager.is_healthy("kalshi", t)]
            stale_markets = [
                t
                for t in tickers
                if (manager.last_update_age("kalshi", t) or 0.0) > config.staleness_s
            ]
            msg_delta = metrics.msg_count - last_msg_count
            elapsed = now - last_health_ts
            update_rate = msg_delta / elapsed if elapsed > 0 else 0.0
            last_msg_count = metrics.msg_count
            last_health_ts = now
            logger.info(
                "health valid=%d stale=%d update_rate=%.2f msg/s reconnects=%d",
                len(valid_markets),
                len(stale_markets),
                update_rate,
                metrics.reconnect_count,
            )

        if now - last_summary_ts >= config.summary_every_s:
            total_fills = sum(state.stats.fills for state in states.values())
            total_cash = sum(state.cash_cents for state in states.values())
            avg_spread = total_spread / spread_samples if spread_samples else 0.0

            total_ticks = sum(state.stats.ticks for state in states.values())
            halted_ticks = sum(state.stats.halted_ticks for state in states.values())
            inv_zero_ticks = sum(state.stats.inv_zero_ticks for state in states.values())
            inv_max_ticks = sum(state.stats.inv_max_ticks for state in states.values())

            pct_halted = halted_ticks / total_ticks if total_ticks else 0.0
            inv_zero_pct = inv_zero_ticks / total_ticks if total_ticks else 0.0
            inv_max_pct = inv_max_ticks / total_ticks if total_ticks else 0.0

            mid_mark = 0.0
            if spread_samples:
                mid_mark = sum(
                    _mid_cents(book.best_bid[0], book.best_ask[0])
                    for book in (
                        manager.get_book("kalshi", ticker) for ticker in tickers
                    )
                    if book and book.best_bid and book.best_ask
                )
                mid_mark = mid_mark / spread_samples

            pnl_cents = total_cash + total_inventory * mid_mark
            msg_delta = metrics.msg_count - last_summary_msg_count
            summary_elapsed = now - last_summary_ts
            update_rate = msg_delta / summary_elapsed if summary_elapsed > 0 else 0.0
            _append_summary_log(
                summary_path,
                [
                    _utc_now(),
                    total_fills,
                    round(pnl_cents, 2),
                    round(avg_spread, 2),
                    round(pct_halted, 4),
                    round(inv_zero_pct, 4),
                    round(inv_max_pct, 4),
                    round(update_rate, 2),
                    metrics.reconnect_count,
                    round(total_net_pnl, 2),
                    round(total_fees, 2),
                    round(total_exposure, 2),
                ],
            )
            for ticker, snap in market_snapshots.items():
                _append_market_summary_log(
                    market_summary_path,
                    [
                        snap["ts_utc"],
                        ticker,
                        snap["inventory"],
                        snap["avg_cost_cents"],
                        snap["best_bid"],
                        snap["best_ask"],
                        snap["mid_cents"],
                        snap["spread_cents"],
                        snap["realized_pnl_cents"],
                        snap["unrealized_pnl_cents"],
                        snap["fees_paid_cents"],
                        snap["net_pnl_cents"],
                        snap["fills_last_minute"],
                        snap["halted"],
                        snap["halt_reason"],
                        snap["exposure_cents"],
                        snap["inv_concentration"],
                        snap["hold_age_s"],
                        snap["max_drawdown_cents"],
                    ],
                )
            last_summary_ts = now
            last_summary_msg_count = metrics.msg_count

        elapsed = time.monotonic() - loop_start
        sleep_for = max(0.0, config.quote_cadence_s - elapsed)
        await asyncio.sleep(sleep_for)
def _append_market_summary_log(path: Path, row: list[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)


def _append_roundtrip_log(path: Path, row: list[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)
