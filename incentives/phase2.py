from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import random
import re
import sqlite3
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from dotenv import load_dotenv

_REPO_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_REPO_ENV, override=False)

from cross_venue_arb.books.manager import BookManager
from cross_venue_arb.books.kalshi_ws import run as kalshi_ws_run
from cross_venue_arb.config import CONFIG
from incentives.incentives import get_active_incentives, incentives_fresh, sync_incentive_programs
from kalshi_fetcher.kalshi_client import request as public_request
from nba_engine.phase5 import RestClient


logger = logging.getLogger("phase2_mm")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[PHASE2] %(asctime)s %(message)s"))
    logger.addHandler(handler)



STATE_FLAT = "FLAT"
STATE_ACCUMULATE = "ACCUMULATE"
STATE_DISTRIBUTE = "DISTRIBUTE"
STATE_FLATTEN = "FLATTEN"
STATE_PAUSE = "PAUSE"


@dataclass
class Phase2Config:
    log_dir: str = "market_maker/logs"
    tick_s: float = 2.0
    positions_refresh_s: float = 5.0
    portfolio_refresh_s: float = 10.0
    cancel_grace_s: float = 10.0
    incentives_refresh_s: float = 60.0
    market_refresh_s: float = 300.0
    max_order_age_s: float = 45.0
    reprice_threshold_cents: int = 2
    rest_fallback_age_s: float = 5.0
    yes_cap: int = 25
    no_cap: int = 25
    net_cap: int = 20
    notional_cap_cents: int = 2500
    max_hold_minutes: int = 120
    min_time_remaining_hours: float = 12.0
    max_spread_cents: int = 20
    min_depth_5c: int = 50
    max_imbalance: float = 0.8
    max_imbalance_hard: float = 0.99
    base_bid_size: int = 1
    base_ask_size: int = 1
    max_bid_size: int = 3
    max_ask_size: int = 3
    edge_min_cents: int = 3
    edge_inv_skew_cents: int = 2
    edge_risk_cents: int = 2
    ask_edge_discount_cents: int = 1
    skew_k_cents: int = 3
    step_in_min_spread_cents: int = 6
    step_in_max_cents: int = 2
    step_out_distribute_cents: int = 1
    step_out_flatten_cents: int = 0
    depth_scale_threshold: float = 500.0
    min_reprice_s: float = 4.0
    bad_window_minutes: int = 15
    bad_window_cash_cents: int = 150
    bad_window_pause_minutes: int = 5
    incentive_mm_count_est: int = 10
    incentive_ev_norm: float = 1.0
    incentive_discount_per_score: float = 0.5
    incentive_discount_max: int = 2
    ev_weight_incentive: float = 1.0
    ev_weight_spread: float = 1.0
    ev_weight_adverse: float = 1.0
    ev_weight_inventory: float = 0.5
    tier_a_pct: float = 0.2
    tier_b_pct: float = 0.7
    adverse_step_in_max: float = 0.6
    adverse_tier_downgrade: float = 0.8
    adverse_discount_zero: float = 0.9
    kill_max_drawdown_cents: int = 300
    kill_inventory_stuck_minutes: int = 240
    kill_adverse_risk: float = 0.95
    kill_adverse_minutes: int = 20
    kill_on_incentive_removed: bool = True
    drop_on_incentive_removed: bool = False
    drop_closed_markets: bool = True
    max_market_loss_cents: int = 0
    max_hour_loss_cents: int = 0
    max_day_loss_cents: int = 0
    hard_cap: int = 0
    soft_throttle_cash_cents: int = 150
    soft_throttle_minutes: int = 15
    soft_throttle_adverse: float = 0.85
    markets_file: str = "config/markets_nba.txt"
    phase1_db_path: str = "data/incentives_phase1.sqlite"
    dry_run: bool = False
    paper_mode: bool = False
    starting_cash_dollars: float = 100.0
    bot_budget_dollars: float = 0.0
    scale_risk: bool = True
    max_fill_prob: float = 0.35
    request_pause_s: float = 0.2
    retry_backoff_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    log_every_s: float = 30.0
    diag_every_s: float = 1800.0
    max_watchlist: int = 5
    incentives_sync_s: float = 300.0
    paper_state_path: str = "data/phase2_paper_state.json"
    paper_state_sync_s: float = 30.0


@dataclass
class InventoryState:
    inv_yes: int = 0
    inv_no: int = 0
    avg_cost_yes: float = 0.0
    avg_cost_no: float = 0.0


@dataclass
class PaperPortfolio:
    cash_cents: int
    positions: Dict[str, InventoryState] = field(default_factory=dict)

    def get_inventory(self, ticker: str) -> InventoryState:
        return self.positions.setdefault(ticker, InventoryState())

    def apply_fill(self, ticker: str, side: str, action: str, price_cents: int, qty: int) -> None:
        inv = self.get_inventory(ticker)
        cost_cents = price_cents * qty
        side = side.lower()
        action = action.lower()

        if action == "buy":
            if side == "yes":
                inv.avg_cost_yes = _weighted_avg(inv.avg_cost_yes, inv.inv_yes, price_cents, qty)
                inv.inv_yes += qty
            elif side == "no":
                inv.avg_cost_no = _weighted_avg(inv.avg_cost_no, inv.inv_no, price_cents, qty)
                inv.inv_no += qty
            self.cash_cents -= cost_cents
        elif action == "sell":
            if side == "yes":
                inv.inv_yes = max(inv.inv_yes - qty, 0)
                if inv.inv_yes == 0:
                    inv.avg_cost_yes = 0.0
            elif side == "no":
                inv.inv_no = max(inv.inv_no - qty, 0)
                if inv.inv_no == 0:
                    inv.avg_cost_no = 0.0
            self.cash_cents += cost_cents


@dataclass
class MarketMetrics:
    best_yes_bid: Optional[int] = None
    best_no_bid: Optional[int] = None
    implied_yes_ask: Optional[int] = None
    implied_no_ask: Optional[int] = None
    spread_cents: Optional[int] = None
    depth_yes_5c: float = 0.0
    depth_no_5c: float = 0.0
    top_yes: float = 0.0
    top_no: float = 0.0
    imbalance: float = 0.0
    time_remaining_hours: Optional[float] = None
    status: Optional[str] = None
    healthy: bool = False


@dataclass
class WorkingOrder:
    order_id: str
    market_ticker: str
    side: str
    action: str
    price_cents: int
    qty: int
    created_ts: float
    cancel_requested_ts: Optional[float] = None


@dataclass
class MarketState:
    hold_start_ts: Dict[str, float] = field(default_factory=dict)
    last_market_fetch_ts: Dict[str, float] = field(default_factory=dict)
    last_positions_fetch_ts: float = 0.0
    last_incentives_fetch_ts: float = 0.0
    market_cache: Dict[str, dict[str, Any]] = field(default_factory=dict)
    working_orders: Dict[str, list[WorkingOrder]] = field(default_factory=dict)
    incentives_active: set[str] = field(default_factory=set)
    incentives_by_ticker: Dict[str, dict[str, Any]] = field(default_factory=dict)
    ev_tiers: Dict[str, str] = field(default_factory=dict)
    ev_scores: Dict[str, float] = field(default_factory=dict)
    incentive_discounts: Dict[str, int] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_csv(path: Path, headers: Iterable[str]) -> None:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(headers))


def _log_csv(path: Path, row: list[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)


def _parse_ts_to_utc_ms(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = int(value)
        if ts > 1_000_000_000_000:
            return ts
        if ts > 1_000_000_000:
            return ts * 1000
        return None
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            return None
    return None


def _depth_within(levels: dict[float, float], best: Optional[float], window: float) -> float:
    if best is None:
        return 0.0
    min_price = best - window
    return sum(size for price, size in levels.items() if price >= min_price)


def _best_price(levels: dict[float, float]) -> Optional[float]:
    prices = [price for price, size in levels.items() if size > 0]
    return max(prices) if prices else None


def _top_depth(levels: dict[float, float], best: Optional[float]) -> float:
    if best is None:
        return 0.0
    return float(levels.get(best, 0.0))


def _imbalance(top_yes: float, top_no: float) -> float:
    denom = top_yes + top_no + 1e-9
    return (top_yes - top_no) / denom


def _spread_cents(best_yes: Optional[int], best_no: Optional[int]) -> Optional[int]:
    if best_yes is None or best_no is None:
        return None
    return (100 - best_no) - best_yes


def _parse_positions(payload: dict[str, Any]) -> dict[str, InventoryState]:
    positions: dict[str, InventoryState] = {}
    rows = payload.get("positions") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return positions
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = row.get("market_ticker") or row.get("ticker")
        if not ticker:
            continue
        side = str(row.get("side") or row.get("position_side") or "").lower()
        qty_raw = row.get("quantity") or row.get("count") or row.get("qty") or row.get("position")
        avg_price = row.get("avg_price") or row.get("average_price") or row.get("avg_cost")
        try:
            qty = int(qty_raw)
        except (TypeError, ValueError):
            qty = 0
        try:
            avg = float(avg_price) if avg_price is not None else 0.0
        except (TypeError, ValueError):
            avg = 0.0
        if avg > 1.0:
            avg = avg / 100.0
        inv = positions.setdefault(str(ticker), InventoryState())
        if side == "yes":
            inv.inv_yes += qty
            inv.avg_cost_yes = avg or inv.avg_cost_yes
        elif side == "no":
            inv.inv_no += qty
            inv.avg_cost_no = avg or inv.avg_cost_no
    return positions


def _extract_cash_cents(payload: dict[str, Any]) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    unit_override = os.getenv("PHASE2_CASH_UNIT", "").strip().lower()
    candidates: list[Any] = []
    for key in (
        "available_cash",
        "cash",
        "cash_balance",
        "balance",
        "available",
        "buying_power",
        "usd_balance",
    ):
        if key in payload:
            candidates.append(payload.get(key))
    portfolio = payload.get("portfolio") if isinstance(payload.get("portfolio"), dict) else None
    if portfolio:
        for key in ("available_cash", "cash", "balance", "buying_power", "usd_balance"):
            if key in portfolio:
                candidates.append(portfolio.get(key))
    for value in candidates:
        try:
            if value is None:
                continue
            if unit_override in {"cents", "cent"}:
                return int(round(float(value)))
            if unit_override in {"dollars", "usd"}:
                return int(round(float(value) * 100))
            raw = float(value)
        except (TypeError, ValueError):
            continue
        if isinstance(value, int):
            return int(raw)
        if isinstance(value, float):
            if raw.is_integer():
                return int(raw)
            return int(round(raw * 100))
        if isinstance(value, str) and "." in value:
            return int(round(raw * 100))
        if abs(raw) >= 10_000:
            return int(round(raw))
        return int(round(raw))
    return None


def _filter_positions(
    positions: dict[str, InventoryState], tickers: set[str]
) -> dict[str, InventoryState]:
    if not tickers:
        return positions
    return {ticker: inv for ticker, inv in positions.items() if ticker in tickers}


def _order_closed(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    order = payload.get("order") if isinstance(payload.get("order"), dict) else payload
    if not isinstance(order, dict):
        return False
    status = str(
        order.get("status")
        or order.get("state")
        or order.get("order_status")
        or order.get("order_state")
        or ""
    ).lower()
    return status in {
        "canceled",
        "cancelled",
        "filled",
        "executed",
        "complete",
        "completed",
        "closed",
        "expired",
        "settled",
        "resolved",
    }


def decide_state(
    inv: InventoryState,
    *,
    net: int,
    time_in_inventory_min: Optional[float],
    market_health: bool,
    incentives_alive: bool,
    time_remaining_hours: Optional[float],
    config: Phase2Config,
) -> str:
    total_inv = inv.inv_yes + inv.inv_no
    near_resolution = (
        time_remaining_hours is not None and time_remaining_hours < config.min_time_remaining_hours
    )
    if not incentives_alive:
        return STATE_FLATTEN if total_inv > 0 else STATE_PAUSE
    if not market_health or near_resolution:
        return STATE_FLATTEN if total_inv > 0 else STATE_PAUSE
    if time_in_inventory_min is not None and time_in_inventory_min > config.max_hold_minutes:
        return STATE_FLATTEN
    if inv.inv_yes >= config.yes_cap or inv.inv_no >= config.no_cap or abs(net) >= config.net_cap:
        return STATE_DISTRIBUTE
    if total_inv == 0:
        return STATE_FLAT
    if total_inv < max(1, (config.yes_cap + config.no_cap) // 4):
        return STATE_ACCUMULATE
    return STATE_DISTRIBUTE


def _eligible_from_metrics(metrics: MarketMetrics, config: Phase2Config) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if metrics.status:
        status_norm = str(metrics.status).lower()
        if status_norm not in {"open", "active"}:
            reasons.append(f"status={metrics.status}")
    if metrics.time_remaining_hours is not None and metrics.time_remaining_hours < config.min_time_remaining_hours:
        reasons.append("near_resolution")
    total_depth = metrics.depth_yes_5c + metrics.depth_no_5c
    if total_depth < config.min_depth_5c:
        reasons.append("low_depth")
    if metrics.spread_cents is None:
        reasons.append("no_spread")
    elif metrics.spread_cents > config.max_spread_cents and total_depth < config.min_depth_5c:
        reasons.append("wide_spread")
    if abs(metrics.imbalance) > config.max_imbalance_hard:
        reasons.append("high_imbalance")
    return len(reasons) == 0, reasons


def _size_ladder(
    inv: InventoryState, config: Phase2Config, imbalance: float, spread_cents: Optional[int], depth_5c: float
) -> tuple[int, int]:
    util_yes = inv.inv_yes / max(1, config.yes_cap)
    util_no = inv.inv_no / max(1, config.no_cap)
    util = (util_yes + util_no) / 2.0
    bid_size = config.base_bid_size if util < 0.5 else 1
    ask_size = config.base_ask_size if util < 0.7 else 1
    if spread_cents is not None and spread_cents >= 8 and depth_5c >= config.depth_scale_threshold and util < 0.5:
        bid_size = min(config.max_bid_size, max(1, bid_size + 1))
        ask_size = min(config.max_ask_size, max(1, ask_size + 1))
    if abs(imbalance) > config.max_imbalance:
        bid_size = 1
        ask_size = 1
    bid_size = min(config.max_bid_size, max(1, bid_size))
    ask_size = min(config.max_ask_size, max(1, ask_size))
    return bid_size, ask_size


def _free_inventory(inv: InventoryState, working: list[WorkingOrder], side: str) -> int:
    side = side.lower()
    open_asks = sum(o.qty for o in working if o.side == side and o.action == "sell")
    if side == "yes":
        return max(inv.inv_yes - open_asks, 0)
    if side == "no":
        return max(inv.inv_no - open_asks, 0)
    return 0


def _notional_cents(inv: InventoryState) -> float:
    yes_notional = inv.inv_yes * inv.avg_cost_yes * 100.0
    no_notional = inv.inv_no * inv.avg_cost_no * 100.0
    return yes_notional + no_notional


def _weighted_avg(avg_cents: float, qty: int, price_cents: int, add_qty: int) -> float:
    total_qty = qty + add_qty
    if total_qty <= 0:
        return 0.0
    return ((avg_cents * qty) + (price_cents * add_qty)) / total_qty


def _fill_probability(metrics: MarketMetrics, side: str) -> float:
    base = 0.05
    if metrics.spread_cents is not None:
        base += min(0.2, metrics.spread_cents / 100.0)
    depth = metrics.top_yes if side == "yes" else metrics.top_no
    if depth:
        base += min(0.1, 1.0 / max(depth, 1.0))
    return min(0.5, max(0.01, base))


def _avg_cost_cents(value: float) -> float:
    if value <= 1.0:
        return value * 100.0
    return value


def _market_pnl_cents(inv: InventoryState, mids: tuple[Optional[int], Optional[int]]) -> Optional[int]:
    yes_mid, no_mid = mids
    if inv.inv_yes + inv.inv_no <= 0:
        return 0
    if inv.inv_yes and yes_mid is None:
        return None
    if inv.inv_no and no_mid is None:
        return None
    yes_cost = _avg_cost_cents(inv.avg_cost_yes)
    no_cost = _avg_cost_cents(inv.avg_cost_no)
    pnl = 0.0
    if inv.inv_yes and yes_mid is not None:
        pnl += (yes_mid - yes_cost) * inv.inv_yes
    if inv.inv_no and no_mid is not None:
        pnl += (no_mid - no_cost) * inv.inv_no
    return int(round(pnl))


def _compute_target_edge(
    inv: InventoryState,
    metrics: MarketMetrics,
    config: Phase2Config,
    edge_min_override: Optional[int] = None,
) -> int:
    util_yes = inv.inv_yes / max(1, config.yes_cap)
    util_no = inv.inv_no / max(1, config.no_cap)
    inv_util = (util_yes + util_no) / 2.0
    risk_score = min(1.0, abs(metrics.imbalance))
    base_edge = edge_min_override if edge_min_override is not None else config.edge_min_cents
    edge = base_edge + config.edge_inv_skew_cents * inv_util + config.edge_risk_cents * risk_score
    return int(round(edge))


def _apply_skew(price_cents: int, skew_cents: int, direction: int) -> int:
    if direction > 0:
        return max(1, price_cents - max(0, skew_cents))
    if direction < 0:
        return min(99, price_cents - min(0, skew_cents))
    return price_cents


def _spread_quality_score(metrics: MarketMetrics, config: Phase2Config) -> float:
    if metrics.spread_cents is None:
        return 0.0
    spread_score = max(0.0, 1.0 - (metrics.spread_cents / max(1.0, config.max_spread_cents)))
    depth = metrics.depth_yes_5c + metrics.depth_no_5c
    depth_score = min(1.0, math.log1p(depth) / math.log1p(max(1.0, config.depth_scale_threshold)))
    return 0.5 * spread_score + 0.5 * depth_score


def _adverse_risk(metrics: MarketMetrics) -> float:
    return min(1.0, abs(metrics.imbalance))


def _inventory_risk(inv: InventoryState, config: Phase2Config) -> float:
    net = inv.inv_yes - inv.inv_no
    return min(1.0, abs(net) / max(1, config.net_cap))


def _mid_price_cents(best_bid: Optional[int], implied_ask: Optional[int]) -> Optional[int]:
    if best_bid is not None and implied_ask is not None:
        return int(round((best_bid + implied_ask) / 2))
    return best_bid if best_bid is not None else implied_ask


def _equity_cents(
    cash_cents: int,
    positions: Dict[str, InventoryState],
    mids: Dict[str, tuple[Optional[int], Optional[int]]],
) -> int:
    equity = cash_cents
    for ticker, inv in positions.items():
        yes_mid, no_mid = mids.get(ticker, (None, None))
        if yes_mid is None:
            yes_mid = int(round(_avg_cost_cents(inv.avg_cost_yes))) if inv.avg_cost_yes else None
        if no_mid is None:
            no_mid = int(round(_avg_cost_cents(inv.avg_cost_no))) if inv.avg_cost_no else None
        if yes_mid is not None:
            equity += inv.inv_yes * yes_mid
        if no_mid is not None:
            equity += inv.inv_no * no_mid
    return int(round(equity))


def _incentive_ev_score(
    incentive: Optional[dict[str, Any]],
    fills_per_hour: float,
    config: Phase2Config,
    now_ms: int,
) -> float:
    if not incentive:
        return 0.0
    reward = incentive.get("reward_value") or 0.0
    try:
        reward_value = float(reward)
    except (TypeError, ValueError):
        reward_value = 0.0
    if reward_value >= 1_000_000:
        reward_value = reward_value / 100.0
    start_ms = incentive.get("start_ts_utc_ms")
    end_ms = incentive.get("end_ts_utc_ms")
    duration_h = None
    if isinstance(start_ms, int) and isinstance(end_ms, int) and end_ms > start_ms:
        duration_h = max((end_ms - start_ms) / 3600000.0, 1.0)
    if duration_h is None:
        duration_h = 24.0
    incentive_type = str(incentive.get("incentive_type") or "").lower()
    if incentive_type == "liquidity":
        ev = reward_value / duration_h
        ev = ev / max(1, config.incentive_mm_count_est)
    elif incentive_type == "volume":
        ev = reward_value / duration_h
    else:
        ev = 0.0
    return ev / max(1.0, config.incentive_ev_norm)


def _assign_tiers(scores: dict[str, float], config: Phase2Config) -> dict[str, str]:
    if not scores:
        return {}
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    n = len(ordered)
    a_cut = max(1, int(round(n * config.tier_a_pct)))
    b_cut = max(a_cut, int(round(n * config.tier_b_pct)))
    tiers: dict[str, str] = {}
    for idx, (ticker, _) in enumerate(ordered):
        if idx < a_cut:
            tiers[ticker] = "A"
        elif idx < b_cut:
            tiers[ticker] = "B"
        else:
            tiers[ticker] = "C"
    return tiers


def _tier_discount(tier: str, config: Phase2Config) -> int:
    if tier == "A":
        return min(config.incentive_discount_max, 2)
    if tier == "B":
        return min(config.incentive_discount_max, 1)
    return 0


def _can_place_bid(inv: InventoryState, net: int, side: str, config: Phase2Config) -> bool:
    side = side.lower()
    if side == "yes":
        return inv.inv_yes < config.yes_cap and net < config.net_cap
    if side == "no":
        return inv.inv_no < config.no_cap and net > -config.net_cap
    return False


def _build_desired_orders(
    state: str,
    inv: InventoryState,
    metrics: MarketMetrics,
    working: list[WorkingOrder],
    config: Phase2Config,
    *,
    edge_min_override: Optional[int] = None,
    allow_step_in: bool = True,
) -> list[dict[str, Any]]:
    desired: list[dict[str, Any]] = []
    net = inv.inv_yes - inv.inv_no
    depth_5c = metrics.depth_yes_5c + metrics.depth_no_5c
    bid_size, ask_size = _size_ladder(inv, config, metrics.imbalance, metrics.spread_cents, depth_5c)
    free_yes = _free_inventory(inv, working, "yes")
    free_no = _free_inventory(inv, working, "no")

    if metrics.best_yes_bid is None or metrics.best_no_bid is None:
        return desired
    if metrics.spread_cents is not None and metrics.spread_cents <= 0:
        return desired

    yes_bid = metrics.best_yes_bid
    no_bid = metrics.best_no_bid
    yes_ask = metrics.implied_yes_ask
    no_ask = metrics.implied_no_ask

    spread = metrics.spread_cents or 0
    if allow_step_in and spread >= config.step_in_min_spread_cents and depth_5c >= (2 * config.min_depth_5c):
        step_in = min(config.step_in_max_cents, max(0, spread // 4))
    else:
        step_in = 0
    target_edge = _compute_target_edge(inv, metrics, config, edge_min_override=edge_min_override)
    skew_cents = int(round(config.skew_k_cents * (net / max(1, config.net_cap))))

    yes_bid = min(99, max(1, yes_bid + step_in))
    no_bid = min(99, max(1, no_bid + step_in))

    if state in {STATE_FLAT, STATE_ACCUMULATE, STATE_DISTRIBUTE}:
        if _notional_cents(inv) < config.notional_cap_cents and _can_place_bid(inv, net, "yes", config):
            adj_bid = _apply_skew(yes_bid, skew_cents, 1)
            desired.append({"side": "yes", "action": "buy", "price": adj_bid, "qty": bid_size})
        if _notional_cents(inv) < config.notional_cap_cents and _can_place_bid(inv, net, "no", config):
            adj_bid = _apply_skew(no_bid, -skew_cents, -1)
            desired.append({"side": "no", "action": "buy", "price": adj_bid, "qty": bid_size})

    if state in {STATE_DISTRIBUTE, STATE_FLATTEN}:
        if free_yes > 0 and yes_ask is not None:
            step_out = config.step_out_flatten_cents if state == STATE_FLATTEN else config.step_out_distribute_cents
            ask_anchor = max(1, min(99, yes_ask - step_out))
            ask_edge = max(0, target_edge - config.ask_edge_discount_cents)
            cost_anchor = min(99, max(1, int(round(_avg_cost_cents(inv.avg_cost_yes) + ask_edge))))
            ask_price = min(cost_anchor, ask_anchor) if state == STATE_FLATTEN else max(cost_anchor, ask_anchor)
            ask_price = _apply_skew(ask_price, skew_cents, 1)
            desired.append({"side": "yes", "action": "sell", "price": ask_price, "qty": min(ask_size, free_yes)})
        if free_no > 0 and no_ask is not None:
            step_out = config.step_out_flatten_cents if state == STATE_FLATTEN else config.step_out_distribute_cents
            ask_anchor = max(1, min(99, no_ask - step_out))
            ask_edge = max(0, target_edge - config.ask_edge_discount_cents)
            cost_anchor = min(99, max(1, int(round(_avg_cost_cents(inv.avg_cost_no) + ask_edge))))
            ask_price = min(cost_anchor, ask_anchor) if state == STATE_FLATTEN else max(cost_anchor, ask_anchor)
            ask_price = _apply_skew(ask_price, -skew_cents, -1)
            desired.append({"side": "no", "action": "sell", "price": ask_price, "qty": min(ask_size, free_no)})

    return desired


def _order_key(order: WorkingOrder) -> str:
    return f"{order.side}:{order.action}"


def _select_best_order(orders: list[WorkingOrder], side: str, action: str) -> Optional[WorkingOrder]:
    for order in orders:
        if order.side == side and order.action == action:
            return order
    return None


def _needs_reprice(
    order: WorkingOrder, target_price: int, now: float, config: Phase2Config
) -> bool:
    if now - order.created_ts < config.min_reprice_s:
        return False
    return abs(order.price_cents - target_price) >= config.reprice_threshold_cents


def _build_metrics_from_book(
    book_yes: dict[float, float],
    book_no: dict[float, float],
    status: Optional[str],
    time_remaining_hours: Optional[float],
    healthy: bool,
) -> MarketMetrics:
    best_yes = _best_price(book_yes)
    best_no = _best_price(book_no)
    metrics = MarketMetrics(
        best_yes_bid=int(round(best_yes * 100)) if best_yes is not None else None,
        best_no_bid=int(round(best_no * 100)) if best_no is not None else None,
        status=status,
        time_remaining_hours=time_remaining_hours,
        healthy=healthy,
    )
    if metrics.best_yes_bid is not None and metrics.best_no_bid is not None:
        metrics.implied_yes_ask = 100 - metrics.best_no_bid
        metrics.implied_no_ask = 100 - metrics.best_yes_bid
        metrics.spread_cents = _spread_cents(metrics.best_yes_bid, metrics.best_no_bid)
    metrics.depth_yes_5c = _depth_within(book_yes, best_yes, 0.05)
    metrics.depth_no_5c = _depth_within(book_no, best_no, 0.05)
    metrics.top_yes = _top_depth(book_yes, best_yes)
    metrics.top_no = _top_depth(book_no, best_no)
    metrics.imbalance = _imbalance(metrics.top_yes, metrics.top_no)
    return metrics


def _parse_orderbook(payload: dict[str, Any]) -> tuple[dict[float, float], dict[float, float]]:
    container = payload.get("orderbook") if isinstance(payload.get("orderbook"), dict) else payload
    yes_levels: dict[float, float] = {}
    no_levels: dict[float, float] = {}
    for level in container.get("yes") or []:
        try:
            price = float(level[0]) / 100.0
            size = float(level[1])
        except Exception:
            continue
        if size > 0:
            yes_levels[price] = size
    for level in container.get("no") or []:
        try:
            price = float(level[0]) / 100.0
            size = float(level[1])
        except Exception:
            continue
        if size > 0:
            no_levels[price] = size
    return yes_levels, no_levels


async def _fetch_market(rest: Optional[RestClient], ticker: str, config: Phase2Config) -> dict[str, Any]:
    if rest is None:
        return await asyncio.to_thread(
            _request_with_retry,
            f"/markets/{ticker}",
            pause_s=config.request_pause_s,
            backoff=config.retry_backoff_s,
        )
    return await asyncio.to_thread(rest.request, "GET", f"/markets/{ticker}")


async def _fetch_positions(rest: Optional[RestClient], config: Phase2Config) -> dict[str, Any]:
    if rest is None:
        return {"positions": []}
    return await asyncio.to_thread(rest.request, "GET", "/portfolio/positions")


async def _fetch_portfolio(rest: Optional[RestClient]) -> dict[str, Any]:
    if rest is None:
        return {}
    return await asyncio.to_thread(rest.get_portfolio)


async def _fetch_orderbook_rest(
    rest: Optional[RestClient], ticker: str, config: Phase2Config
) -> dict[str, Any]:
    if rest is None:
        return await asyncio.to_thread(
            _request_with_retry,
            f"/markets/{ticker}/orderbook",
            pause_s=config.request_pause_s,
            backoff=config.retry_backoff_s,
        )
    return await asyncio.to_thread(rest.get_orderbook, ticker)


def _refresh_incentives(state: MarketState) -> None:
    active = get_active_incentives()
    tickers: set[str] = set()
    incentives_by_ticker: Dict[str, dict[str, Any]] = {}
    for row in active:
        market_ticker = row.get("market_ticker")
        if market_ticker:
            tickers.add(str(market_ticker))
            incentives_by_ticker[str(market_ticker)] = row
    state.incentives_active = tickers
    state.incentives_by_ticker = incentives_by_ticker


_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")
_INT_RE = re.compile(r"^-?\d+$")


def _parse_value(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    if _INT_RE.match(value):
        return int(value)
    if _FLOAT_RE.match(value):
        return float(value)
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_value(value)
    return data


def load_markets(path: str | Path) -> list[str]:
    markets: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        markets.append(line.upper())
    return markets


def _parse_date(value: object) -> Optional[datetime.date]:
    if not value:
        return None
    try:
        text = str(value)
        if "T" in text:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        return datetime.fromisoformat(text).date()
    except Exception:
        return None


def _load_paper_state(path: Path) -> Optional[tuple[PaperPortfolio, dict[str, Any]]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("paper_state_load_failed path=%s err=%s", path, exc)
        return None
    cash_cents = int(raw.get("cash_cents", 0))
    portfolio = PaperPortfolio(cash_cents=cash_cents)
    positions = raw.get("positions")
    if isinstance(positions, dict):
        for ticker, pdata in positions.items():
            if not isinstance(pdata, dict):
                continue
            inv = portfolio.get_inventory(str(ticker).upper())
            try:
                inv.inv_yes = int(pdata.get("inv_yes", 0))
                inv.inv_no = int(pdata.get("inv_no", 0))
                inv.avg_cost_yes = float(pdata.get("avg_cost_yes", 0.0))
                inv.avg_cost_no = float(pdata.get("avg_cost_no", 0.0))
            except Exception:
                continue
    meta = {
        "day_start_date": raw.get("day_start_date"),
        "day_start_equity": raw.get("day_start_equity"),
        "peak_equity": raw.get("peak_equity"),
        "worst_window_cash": raw.get("worst_window_cash"),
    }
    return portfolio, meta


def _save_paper_state(
    path: Path,
    portfolio: PaperPortfolio,
    day_start: datetime.date,
    day_start_equity: int,
    peak_equity: int,
    worst_window_cash: int,
) -> None:
    positions: dict[str, dict[str, float | int]] = {}
    for ticker, inv in portfolio.positions.items():
        positions[ticker] = {
            "inv_yes": inv.inv_yes,
            "inv_no": inv.inv_no,
            "avg_cost_yes": inv.avg_cost_yes,
            "avg_cost_no": inv.avg_cost_no,
        }
    payload = {
        "cash_cents": portfolio.cash_cents,
        "positions": positions,
        "day_start_date": day_start.isoformat(),
        "day_start_equity": int(day_start_equity),
        "peak_equity": int(peak_equity),
        "worst_window_cash": int(worst_window_cash),
        "updated_ts": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp_path.replace(path)


def _request_with_retry(
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    pause_s: float = 0.0,
    backoff: Iterable[float] = (),
) -> Dict[str, Any]:
    delays = (0.0, *backoff)
    for idx, delay in enumerate(delays):
        if delay:
            time.sleep(delay)
        if pause_s:
            time.sleep(pause_s)
        try:
            return public_request(endpoint, params=params)
        except Exception as exc:
            msg = str(exc)
            if idx == len(delays) - 1:
                raise
            if any(code in msg for code in ("429", "500", "502", "503", "504")):
                continue
            continue


def _parse_bool(raw: object, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def load_markets_from_phase1(db_path: str | Path) -> list[str]:
    db_file = Path(db_path)
    if not db_file.exists():
        return []
    conn = sqlite3.connect(str(db_file))
    try:
        row = conn.execute(
            "SELECT run_id FROM phase1_market_decisions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return []
        run_id = row[0]
        rows = conn.execute(
            """
            SELECT market_ticker
            FROM phase1_market_decisions
            WHERE run_id = ? AND eligible = 1
            """,
            (run_id,),
        ).fetchall()
        return [str(r[0]) for r in rows if r and r[0]]
    finally:
        conn.close()


def load_top_markets_from_phase1(db_path: str | Path, limit: int) -> list[str]:
    if limit <= 0:
        return load_markets_from_phase1(db_path)
    db_file = Path(db_path)
    if not db_file.exists():
        return []
    conn = sqlite3.connect(str(db_file))
    try:
        row = conn.execute(
            "SELECT run_id FROM phase1_market_decisions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return []
        run_id = row[0]
        rows = conn.execute(
            """
            SELECT d.market_ticker
            FROM phase1_market_decisions d
            WHERE d.run_id = ?
            ORDER BY d.rank_score DESC
            LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
        return [str(r[0]) for r in rows if r and r[0]]
    finally:
        conn.close()


def load_phase2_config(path: str | Path) -> Phase2Config:
    values = load_simple_yaml(path)
    config = Phase2Config(
        log_dir=values.get("log_dir", Phase2Config.log_dir),
        tick_s=float(values.get("phase2_tick_s", Phase2Config.tick_s)),
        positions_refresh_s=float(
            values.get("phase2_positions_refresh_s", Phase2Config.positions_refresh_s)
        ),
        portfolio_refresh_s=float(
            values.get("phase2_portfolio_refresh_s", Phase2Config.portfolio_refresh_s)
        ),
        cancel_grace_s=float(
            values.get("phase2_cancel_grace_s", Phase2Config.cancel_grace_s)
        ),
        incentives_refresh_s=float(
            values.get("phase2_incentives_refresh_s", Phase2Config.incentives_refresh_s)
        ),
        market_refresh_s=float(values.get("phase2_market_refresh_s", Phase2Config.market_refresh_s)),
        max_order_age_s=float(values.get("phase2_max_order_age_s", Phase2Config.max_order_age_s)),
        reprice_threshold_cents=int(
            values.get("phase2_reprice_threshold_cents", Phase2Config.reprice_threshold_cents)
        ),
        rest_fallback_age_s=float(
            values.get("phase2_rest_fallback_age_s", Phase2Config.rest_fallback_age_s)
        ),
        yes_cap=int(values.get("phase2_yes_cap", Phase2Config.yes_cap)),
        no_cap=int(values.get("phase2_no_cap", Phase2Config.no_cap)),
        net_cap=int(values.get("phase2_net_cap", Phase2Config.net_cap)),
        notional_cap_cents=int(
            values.get("phase2_notional_cap_cents", Phase2Config.notional_cap_cents)
        ),
        max_hold_minutes=int(values.get("phase2_max_hold_minutes", Phase2Config.max_hold_minutes)),
        min_time_remaining_hours=float(
            values.get("phase2_min_time_remaining_hours", Phase2Config.min_time_remaining_hours)
        ),
        max_spread_cents=int(values.get("phase2_max_spread_cents", Phase2Config.max_spread_cents)),
        min_depth_5c=int(values.get("phase2_min_depth_5c", Phase2Config.min_depth_5c)),
        max_imbalance=float(values.get("phase2_max_imbalance", Phase2Config.max_imbalance)),
        max_imbalance_hard=float(
            values.get("phase2_max_imbalance_hard", Phase2Config.max_imbalance_hard)
        ),
        base_bid_size=int(values.get("phase2_base_bid_size", Phase2Config.base_bid_size)),
        base_ask_size=int(values.get("phase2_base_ask_size", Phase2Config.base_ask_size)),
        max_bid_size=int(values.get("phase2_max_bid_size", Phase2Config.max_bid_size)),
        max_ask_size=int(values.get("phase2_max_ask_size", Phase2Config.max_ask_size)),
        edge_min_cents=int(values.get("phase2_edge_min_cents", Phase2Config.edge_min_cents)),
        edge_inv_skew_cents=int(
            values.get("phase2_edge_inv_skew_cents", Phase2Config.edge_inv_skew_cents)
        ),
        edge_risk_cents=int(values.get("phase2_edge_risk_cents", Phase2Config.edge_risk_cents)),
        ask_edge_discount_cents=int(
            values.get("phase2_ask_edge_discount_cents", Phase2Config.ask_edge_discount_cents)
        ),
        skew_k_cents=int(values.get("phase2_skew_k_cents", Phase2Config.skew_k_cents)),
        step_in_min_spread_cents=int(
            values.get("phase2_step_in_min_spread_cents", Phase2Config.step_in_min_spread_cents)
        ),
        step_in_max_cents=int(
            values.get("phase2_step_in_max_cents", Phase2Config.step_in_max_cents)
        ),
        step_out_distribute_cents=int(
            values.get(
                "phase2_step_out_distribute_cents",
                Phase2Config.step_out_distribute_cents,
            )
        ),
        step_out_flatten_cents=int(
            values.get("phase2_step_out_flatten_cents", Phase2Config.step_out_flatten_cents)
        ),
        depth_scale_threshold=float(
            values.get("phase2_depth_scale_threshold", Phase2Config.depth_scale_threshold)
        ),
        min_reprice_s=float(values.get("phase2_min_reprice_s", Phase2Config.min_reprice_s)),
        bad_window_minutes=int(
            values.get("phase2_bad_window_minutes", Phase2Config.bad_window_minutes)
        ),
        bad_window_cash_cents=int(
            values.get("phase2_bad_window_cash_cents", Phase2Config.bad_window_cash_cents)
        ),
        bad_window_pause_minutes=int(
            values.get("phase2_bad_window_pause_minutes", Phase2Config.bad_window_pause_minutes)
        ),
        incentive_mm_count_est=int(
            values.get("phase2_incentive_mm_count_est", Phase2Config.incentive_mm_count_est)
        ),
        incentive_ev_norm=float(
            values.get("phase2_incentive_ev_norm", Phase2Config.incentive_ev_norm)
        ),
        incentive_discount_per_score=float(
            values.get(
                "phase2_incentive_discount_per_score",
                Phase2Config.incentive_discount_per_score,
            )
        ),
        incentive_discount_max=int(
            values.get("phase2_incentive_discount_max", Phase2Config.incentive_discount_max)
        ),
        ev_weight_incentive=float(
            values.get("phase2_ev_weight_incentive", Phase2Config.ev_weight_incentive)
        ),
        ev_weight_spread=float(
            values.get("phase2_ev_weight_spread", Phase2Config.ev_weight_spread)
        ),
        ev_weight_adverse=float(
            values.get("phase2_ev_weight_adverse", Phase2Config.ev_weight_adverse)
        ),
        ev_weight_inventory=float(
            values.get("phase2_ev_weight_inventory", Phase2Config.ev_weight_inventory)
        ),
        tier_a_pct=float(values.get("phase2_tier_a_pct", Phase2Config.tier_a_pct)),
        tier_b_pct=float(values.get("phase2_tier_b_pct", Phase2Config.tier_b_pct)),
        adverse_step_in_max=float(
            values.get("phase2_adverse_step_in_max", Phase2Config.adverse_step_in_max)
        ),
        adverse_tier_downgrade=float(
            values.get("phase2_adverse_tier_downgrade", Phase2Config.adverse_tier_downgrade)
        ),
        adverse_discount_zero=float(
            values.get("phase2_adverse_discount_zero", Phase2Config.adverse_discount_zero)
        ),
        kill_max_drawdown_cents=int(
            values.get("phase2_kill_max_drawdown_cents", Phase2Config.kill_max_drawdown_cents)
        ),
        kill_inventory_stuck_minutes=int(
            values.get(
                "phase2_kill_inventory_stuck_minutes",
                Phase2Config.kill_inventory_stuck_minutes,
            )
        ),
        kill_adverse_risk=float(
            values.get("phase2_kill_adverse_risk", Phase2Config.kill_adverse_risk)
        ),
        kill_adverse_minutes=int(
            values.get("phase2_kill_adverse_minutes", Phase2Config.kill_adverse_minutes)
        ),
        kill_on_incentive_removed=_parse_bool(
            values.get(
                "phase2_kill_on_incentive_removed",
                Phase2Config.kill_on_incentive_removed,
            ),
            default=Phase2Config.kill_on_incentive_removed,
        ),
        drop_on_incentive_removed=_parse_bool(
            values.get(
                "phase2_drop_on_incentive_removed",
                Phase2Config.drop_on_incentive_removed,
            ),
            default=Phase2Config.drop_on_incentive_removed,
        ),
        drop_closed_markets=_parse_bool(
            values.get("phase2_drop_closed_markets", Phase2Config.drop_closed_markets),
            default=Phase2Config.drop_closed_markets,
        ),
        max_market_loss_cents=int(
            values.get("phase2_max_market_loss_cents", Phase2Config.max_market_loss_cents)
        ),
        max_hour_loss_cents=int(
            values.get("phase2_max_hour_loss_cents", Phase2Config.max_hour_loss_cents)
        ),
        max_day_loss_cents=int(
            values.get("phase2_max_day_loss_cents", Phase2Config.max_day_loss_cents)
        ),
        hard_cap=int(values.get("phase2_hard_cap", Phase2Config.hard_cap)),
        soft_throttle_cash_cents=int(
            values.get("phase2_soft_throttle_cash_cents", Phase2Config.soft_throttle_cash_cents)
        ),
        soft_throttle_minutes=int(
            values.get("phase2_soft_throttle_minutes", Phase2Config.soft_throttle_minutes)
        ),
        soft_throttle_adverse=float(
            values.get("phase2_soft_throttle_adverse", Phase2Config.soft_throttle_adverse)
        ),
        markets_file=values.get("markets_file", Phase2Config.markets_file),
        phase1_db_path=values.get("phase2_phase1_db_path", Phase2Config.phase1_db_path),
        dry_run=_parse_bool(
            values.get("phase2_dry_run", os.getenv("PHASE2_DRY_RUN", "1")),
            default=Phase2Config.dry_run,
        ),
        paper_mode=_parse_bool(
            values.get("phase2_paper_mode", os.getenv("PHASE2_PAPER_MODE", "1")),
            default=Phase2Config.paper_mode,
        ),
        starting_cash_dollars=float(
            values.get("phase2_starting_cash_dollars", Phase2Config.starting_cash_dollars)
        ),
        bot_budget_dollars=float(values.get("phase2_bot_budget_dollars", 0.0) or 0.0),
        scale_risk=_parse_bool(
            values.get("phase2_scale_risk", os.getenv("PHASE2_SCALE_RISK", "1")),
            default=Phase2Config.scale_risk,
        ),
        max_fill_prob=float(values.get("phase2_max_fill_prob", Phase2Config.max_fill_prob)),
        request_pause_s=float(values.get("phase2_request_pause_s", Phase2Config.request_pause_s)),
        retry_backoff_s=tuple(
            float(x)
            for x in str(
                values.get(
                    "phase2_retry_backoff_s",
                    ",".join(map(str, Phase2Config.retry_backoff_s)),
                )
            ).split(",")
            if str(x).strip()
        ),
        log_every_s=float(values.get("phase2_log_every_s", Phase2Config.log_every_s)),
        diag_every_s=float(values.get("phase2_diag_every_s", Phase2Config.diag_every_s)),
        max_watchlist=int(values.get("phase2_max_watchlist", Phase2Config.max_watchlist)),
        incentives_sync_s=float(
            values.get("phase2_incentives_sync_s", Phase2Config.incentives_sync_s)
        ),
        paper_state_path=values.get(
            "phase2_paper_state_path", Phase2Config.paper_state_path
        ),
        paper_state_sync_s=float(
            values.get("phase2_paper_state_sync_s", Phase2Config.paper_state_sync_s)
        ),
    )
    if config.scale_risk:
        scale = config.starting_cash_dollars / 100.0
        if scale > 0 and abs(scale - 1.0) > 1e-6:
            scaled_fields = [
                "notional_cap_cents",
                "bad_window_cash_cents",
                "kill_max_drawdown_cents",
                "max_market_loss_cents",
                "max_hour_loss_cents",
                "max_day_loss_cents",
                "hard_cap",
                "soft_throttle_cash_cents",
            ]
            for field in scaled_fields:
                if f"phase2_{field}" in values:
                    continue
                current = getattr(config, field)
                if current <= 0:
                    continue
                scaled = int(round(current * scale))
                setattr(config, field, max(1, scaled))
    return config


def _ensure_rest_client() -> RestClient:
    if not CONFIG.kalshi.rest_url:
        raise RuntimeError("KALSHI_REST_URL is required for phase2 trading")
    if not CONFIG.kalshi.key_id or not CONFIG.kalshi.private_key_path:
        raise RuntimeError("KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH are required for phase2 trading")
    key_path = Path(CONFIG.kalshi.private_key_path)
    return RestClient(
        CONFIG.kalshi.rest_url,
        CONFIG.kalshi.key_id,
        key_path,
        order_url=CONFIG.kalshi.order_url or CONFIG.kalshi.rest_url,
    )


async def phase2_loop(
    manager: BookManager,
    tickers: Iterable[str],
    config: Phase2Config,
) -> None:
    rest = None if config.paper_mode else _ensure_rest_client()
    tickers = [t.upper() for t in tickers]
    tickers_set = set(tickers)
    dry_actions = config.dry_run or config.paper_mode
    logger.info(
        "boot_mode paper_mode=%s dry_run=%s rest_url=%s order_url=%s",
        int(config.paper_mode),
        int(config.dry_run),
        CONFIG.kalshi.rest_url if not config.paper_mode else "paper",
        CONFIG.kalshi.order_url if not config.paper_mode else "paper",
    )

    state = MarketState()
    positions_cache: dict[str, InventoryState] = {}
    portfolio = PaperPortfolio(cash_cents=int(config.starting_cash_dollars * 100))
    live_cash_cents: Optional[int] = None
    paper_state_path = (
        Path(config.paper_state_path) if config.paper_mode and config.paper_state_path else None
    )
    paper_state_meta: Optional[dict[str, Any]] = None
    if paper_state_path:
        loaded = _load_paper_state(paper_state_path)
        if loaded:
            portfolio, paper_state_meta = loaded
    last_log_ts = 0.0
    last_diag_ts = 0.0
    last_incentives_sync_ts = 0.0
    last_summary_ts = 0.0
    last_portfolio_fetch_ts = 0.0
    summary_fill_events = 0
    summary_actions = 0
    summary_start_cash = portfolio.cash_cents
    diag_state_counts: Dict[str, int] = {}
    diag_exclusion_counts: Dict[str, int] = {}
    diag_desired_orders = 0
    diag_created_orders = 0
    diag_canceled_orders = 0
    diag_fill_events = 0
    diag_samples: list[dict[str, Any]] = []
    cash_window: deque[tuple[float, int]] = deque()
    hourly_equity_window: deque[tuple[float, int]] = deque()
    pause_until_ts = 0.0
    fill_history: Dict[str, deque[float]] = {}
    fill_cash_history: Dict[str, deque[tuple[float, int]]] = {}
    latest_ev_rows: Dict[str, dict[str, Any]] = {}
    adverse_history: Dict[str, deque[float]] = {}
    soft_cash_window: deque[tuple[float, int]] = deque()
    soft_throttle_until = 0.0
    kill_triggered = False
    last_mids: Dict[str, tuple[Optional[int], Optional[int]]] = {}
    day_start = datetime.now(timezone.utc).date()
    hourly_halt_until = 0.0
    daily_halt_day: Optional[datetime.date] = None
    boot_equity = _equity_cents(portfolio.cash_cents, portfolio.positions, {})
    day_start_equity = boot_equity
    peak_equity = boot_equity
    day_start_equity_set = config.paper_mode
    worst_window_cash = 0
    if paper_state_meta:
        loaded_day = _parse_date(paper_state_meta.get("day_start_date"))
        if loaded_day == day_start:
            try:
                day_start_equity = int(paper_state_meta.get("day_start_equity", day_start_equity))
                peak_equity = int(paper_state_meta.get("peak_equity", peak_equity))
                worst_window_cash = int(
                    paper_state_meta.get("worst_window_cash", worst_window_cash)
                )
            except Exception:
                pass
    last_dashboard_ts = 0.0
    last_incentives_active: set[str] = set()
    last_paper_state_ts = 0.0
    inactive_tickers: set[str] = set()
    paused_tickers: Dict[str, datetime.date] = {}
    hard_cap_tickers: set[str] = set()

    decisions_path = Path(config.log_dir) / "phase2_decisions.csv"
    actions_path = Path(config.log_dir) / "phase2_actions.csv"
    fills_path = Path(config.log_dir) / "phase2_fills.csv"
    summary_path = Path(config.log_dir) / "phase2_summary.csv"
    phase4_path = Path(config.log_dir) / "phase4_ev.csv"
    phase4_validation_path = Path(config.log_dir) / "phase4_validation.csv"
    _ensure_csv(
        decisions_path,
        [
            "ts_utc",
            "market_ticker",
            "state",
            "eligible",
            "exclude_reasons",
            "inv_yes",
            "inv_no",
            "net",
            "spread_cents",
            "depth_5c",
            "imbalance",
            "time_remaining_h",
            "incentives_alive",
        ],
    )
    _ensure_csv(
        actions_path,
        [
            "ts_utc",
            "market_ticker",
            "action",
            "side",
            "price_cents",
            "qty",
            "order_id",
            "reason",
        ],
    )
    _ensure_csv(
        fills_path,
        [
            "ts_utc",
            "market_ticker",
            "side",
            "action",
            "price_cents",
            "qty",
            "cash_cents",
        ],
    )
    _ensure_csv(
        summary_path,
        [
            "ts_utc",
            "fills",
            "actions",
            "fill_rate",
            "net_cash_change",
            "inv_yes",
            "inv_no",
        ],
    )
    _ensure_csv(
        phase4_path,
        [
            "ts_utc",
            "market_ticker",
            "tier",
            "incentive_ev_score",
            "spread_quality_score",
            "adverse_risk",
            "inventory_risk",
            "total_ev_score",
            "incentive_discount",
            "fills_30m",
        ],
    )
    _ensure_csv(
        phase4_validation_path,
        [
            "ts_utc",
            "market_ticker",
            "tier",
            "incentive_discount",
            "adverse_risk",
            "price_pnl_cents",
            "incentive_revenue_est_cents",
            "net_pnl_cents",
        ],
    )

    async def _cancel_all_orders(reason: str) -> None:
        for ticker, orders in list(state.working_orders.items()):
            for order in list(orders):
                if dry_actions:
                    orders.remove(order)
                    _log_csv(
                        actions_path,
                        [
                            _utc_now(),
                            ticker,
                            "cancel",
                            order.side,
                            order.price_cents,
                            order.qty,
                            order.order_id,
                            reason,
                        ],
                    )
                    continue
                try:
                    await asyncio.to_thread(rest.cancel_order, order.order_id)
                except Exception as exc:
                    logger.warning(
                        "action=cancel status=failed ticker=%s order_id=%s err=%s",
                        ticker,
                        order.order_id,
                        exc,
                    )
                else:
                    logger.info(
                        "action=cancel status=ok ticker=%s order_id=%s",
                        ticker,
                        order.order_id,
                    )
                    order.cancel_requested_ts = time.monotonic()
                    _log_csv(
                        actions_path,
                        [
                            _utc_now(),
                            ticker,
                            "cancel",
                            order.side,
                            order.price_cents,
                            order.qty,
                            order.order_id,
                            reason,
                        ],
                    )

    async def _cancel_orders_for_ticker(ticker: str, reason: str) -> None:
        orders = state.working_orders.get(ticker)
        if not orders:
            return
        for order in list(orders):
            if dry_actions:
                orders.remove(order)
                _log_csv(
                    actions_path,
                    [
                        _utc_now(),
                        ticker,
                        "cancel",
                        order.side,
                        order.price_cents,
                        order.qty,
                        order.order_id,
                        reason,
                    ],
                )
                continue
            try:
                await asyncio.to_thread(rest.cancel_order, order.order_id)
            except Exception as exc:
                logger.warning(
                    "action=cancel status=failed ticker=%s order_id=%s err=%s",
                    ticker,
                    order.order_id,
                    exc,
                )
            else:
                logger.info(
                    "action=cancel status=ok ticker=%s order_id=%s",
                    ticker,
                    order.order_id,
                )
                order.cancel_requested_ts = time.monotonic()
                _log_csv(
                    actions_path,
                    [
                        _utc_now(),
                        ticker,
                        "cancel",
                        order.side,
                        order.price_cents,
                        order.qty,
                        order.order_id,
                        reason,
                    ],
                )

    while True:
        now = time.monotonic()
        incentives_ok = incentives_fresh()
        temp_scores: Dict[str, float] = {}
        temp_discounts: Dict[str, int] = {}
        temp_ev_rows: Dict[str, dict[str, Any]] = {}
        equity_now = _equity_cents(portfolio.cash_cents, portfolio.positions, last_mids)
        cash_window.append((now, equity_now))
        hourly_equity_window.append((now, equity_now))
        soft_cash_window.append((now, equity_now))
        window_s = config.bad_window_minutes * 60
        soft_window_s = config.soft_throttle_minutes * 60
        hour_window_s = 3600.0
        while cash_window and now - cash_window[0][0] > window_s:
            cash_window.popleft()
        while hourly_equity_window and now - hourly_equity_window[0][0] > hour_window_s:
            hourly_equity_window.popleft()
        while soft_cash_window and now - soft_cash_window[0][0] > soft_window_s:
            soft_cash_window.popleft()
        if cash_window:
            equity_change = equity_now - cash_window[0][1]
            if equity_change <= -config.bad_window_cash_cents and now >= pause_until_ts:
                pause_until_ts = now + (config.bad_window_pause_minutes * 60)
                logger.info(
                    "bad_window_brake equity_change=%s pause_minutes=%s",
                    equity_change,
                    config.bad_window_pause_minutes,
                )
        if soft_cash_window:
            soft_change = equity_now - soft_cash_window[0][1]
            if soft_change <= -config.soft_throttle_cash_cents and now >= soft_throttle_until:
                soft_throttle_until = now + (config.soft_throttle_minutes * 60)
                logger.info(
                    "soft_throttle equity_change=%s pause_minutes=%s",
                    soft_change,
                    config.soft_throttle_minutes,
                )
        if (
            config.max_hour_loss_cents > 0
            and now >= hourly_halt_until
            and hourly_equity_window
        ):
            hour_change = equity_now - hourly_equity_window[0][1]
            if hour_change <= -config.max_hour_loss_cents:
                hourly_halt_until = now + 3600.0
                await _cancel_all_orders("hourly_loss_cap")
                logger.error(
                    "hourly_loss_cap hit change_cents=%s halt_minutes=60",
                    hour_change,
                )
        kill_reason = ""
        if not config.paper_mode and now - state.last_positions_fetch_ts >= config.positions_refresh_s:
            payload = await _fetch_positions(rest, config)
            positions_cache = _filter_positions(_parse_positions(payload), tickers_set)
            state.last_positions_fetch_ts = now
        if not config.paper_mode and now - last_portfolio_fetch_ts >= config.portfolio_refresh_s:
            try:
                portfolio_payload = await _fetch_portfolio(rest)
                live_cash_cents = _extract_cash_cents(portfolio_payload)
            except Exception as exc:
                logger.warning("portfolio_fetch_failed err=%s", exc)
            last_portfolio_fetch_ts = now

        if now - state.last_incentives_fetch_ts >= config.incentives_refresh_s:
            last_incentives_active = set(state.incentives_active)
            _refresh_incentives(state)
            state.last_incentives_fetch_ts = now
        if now - last_incentives_sync_ts >= config.incentives_sync_s:
            try:
                await asyncio.to_thread(sync_incentive_programs)
            except Exception as exc:
                logger.warning("incentives_sync_failed err=%s", exc)
            last_incentives_sync_ts = now

        global_halt = (now < hourly_halt_until) or (daily_halt_day == day_start)
        for ticker in tickers:
            if kill_triggered:
                break
            if ticker in inactive_tickers:
                continue
            paused_until = paused_tickers.get(ticker)
            if paused_until is not None:
                if paused_until > day_start:
                    continue
                paused_tickers.pop(ticker, None)
            book = manager.get_book("kalshi", ticker)
            market_payload = state.market_cache.get(ticker)
            time_remaining_hours = None
            status = None

            last_market_ts = state.last_market_fetch_ts.get(ticker, 0.0)
            if market_payload is None or now - last_market_ts >= config.market_refresh_s:
                try:
                    market_payload = await _fetch_market(rest, ticker, config)
                except Exception as exc:
                    logger.warning("market_fetch_failed ticker=%s err=%s", ticker, exc)
                    if market_payload is None:
                        continue
                else:
                    state.market_cache[ticker] = market_payload
                    state.last_market_fetch_ts[ticker] = now

            if isinstance(market_payload, dict):
                market_obj = market_payload.get("market") if isinstance(market_payload.get("market"), dict) else market_payload
                status = market_obj.get("status")
                close_ts = _parse_ts_to_utc_ms(
                    market_obj.get("close_time")
                    or market_obj.get("close_time_ts")
                    or market_obj.get("close_time_ms")
                    or market_obj.get("close_ts")
                    or market_obj.get("settle_time")
                )
                if close_ts:
                    time_remaining_hours = max((close_ts - int(time.time() * 1000)) / 3600000.0, 0.0)
            status_norm = str(status).lower() if status is not None else ""
            if config.drop_closed_markets and status_norm in {
                "closed",
                "settled",
                "resolved",
                "finalized",
                "expired",
            }:
                await _cancel_orders_for_ticker(ticker, f"market_{status_norm}")
                inactive_tickers.add(ticker)
                logger.info("ticker_deactivated ticker=%s reason=market_%s", ticker, status_norm)
                continue

            if book and book.is_healthy(max_age_s=config.rest_fallback_age_s, now_ts=time.monotonic()):
                book_yes = book.bids
                book_no = book.raw_no_bids or {}
                metrics = _build_metrics_from_book(
                    book_yes,
                    book_no,
                    status=status,
                    time_remaining_hours=time_remaining_hours,
                    healthy=True,
                )
            else:
                try:
                    orderbook_payload = await _fetch_orderbook_rest(rest, ticker, config)
                    book_yes, book_no = _parse_orderbook(orderbook_payload)
                    metrics = _build_metrics_from_book(
                        book_yes,
                        book_no,
                        status=status,
                        time_remaining_hours=time_remaining_hours,
                        healthy=bool(book_yes or book_no),
                    )
                except Exception as exc:
                    logger.warning("orderbook_fetch_failed ticker=%s err=%s", ticker, exc)
                    continue

            if config.paper_mode:
                inv = portfolio.get_inventory(ticker)
            else:
                inv = positions_cache.get(ticker, InventoryState())
            net = inv.inv_yes - inv.inv_no
            total_inv = inv.inv_yes + inv.inv_no

            working = state.working_orders.setdefault(ticker, [])
            if not config.paper_mode and working:
                for order in list(working):
                    if order.cancel_requested_ts is None:
                        continue
                    if time.monotonic() - order.cancel_requested_ts < config.cancel_grace_s:
                        continue
                    try:
                        payload = await asyncio.to_thread(rest.get_order, order.order_id)
                    except Exception as exc:
                        logger.warning(
                            "action=cancel status=check_failed ticker=%s order_id=%s err=%s",
                            ticker,
                            order.order_id,
                            exc,
                        )
                        continue
                    if _order_closed(payload):
                        working.remove(order)
                        logger.info(
                            "action=cancel status=confirmed ticker=%s order_id=%s",
                            ticker,
                            order.order_id,
                        )

            if inv.inv_yes + inv.inv_no > 0 and ticker not in state.hold_start_ts:
                state.hold_start_ts[ticker] = time.monotonic()
            if inv.inv_yes + inv.inv_no == 0:
                state.hold_start_ts.pop(ticker, None)
            hold_min = None
            if ticker in state.hold_start_ts:
                hold_min = (time.monotonic() - state.hold_start_ts[ticker]) / 60.0

            incentives_alive = ticker in state.incentives_active if state.incentives_active else True
            if (
                incentives_ok
                and ticker in last_incentives_active
                and ticker not in state.incentives_active
            ):
                if config.drop_on_incentive_removed:
                    await _cancel_orders_for_ticker(ticker, "incentive_removed")
                    inactive_tickers.add(ticker)
                    logger.info("ticker_deactivated ticker=%s reason=incentive_removed", ticker)
                    continue
                if config.kill_on_incentive_removed:
                    kill_triggered = True
                    kill_reason = f"incentive_removed ticker={ticker}"
                    break
            history = fill_history.setdefault(ticker, deque())
            cutoff = now - 1800.0
            while history and history[0] < cutoff:
                history.popleft()
            fills_30m = len(history)
            fills_per_hour = fills_30m * 2
            incentive = state.incentives_by_ticker.get(ticker)
            incentive_ev = _incentive_ev_score(
                incentive,
                fills_per_hour=fills_per_hour,
                config=config,
                now_ms=int(time.time() * 1000),
            )
            spread_quality = _spread_quality_score(metrics, config)
            adverse_risk = _adverse_risk(metrics)
            adverse_track = adverse_history.setdefault(ticker, deque())
            if adverse_risk >= config.kill_adverse_risk:
                adverse_track.append(now)
                cutoff_adverse = now - (config.kill_adverse_minutes * 60)
                while adverse_track and adverse_track[0] < cutoff_adverse:
                    adverse_track.popleft()
                if adverse_track and now - adverse_track[0] >= config.kill_adverse_minutes * 60:
                    kill_triggered = True
                    kill_reason = f"adverse_persist ticker={ticker} risk={adverse_risk:.3f}"
                    break
            else:
                adverse_track.clear()
            inventory_risk = _inventory_risk(inv, config)
            total_ev = (
                config.ev_weight_incentive * incentive_ev
                + config.ev_weight_spread * spread_quality
                - config.ev_weight_adverse * adverse_risk
                - config.ev_weight_inventory * inventory_risk
            )
            temp_scores[ticker] = total_ev
            temp_discounts[ticker] = 0
            temp_ev_rows[ticker] = {
                "incentive_ev": incentive_ev,
                "spread_quality": spread_quality,
                "adverse_risk": adverse_risk,
                "inventory_risk": inventory_risk,
                "total_ev": total_ev,
                "discount": 0,
                "fills_30m": fills_30m,
                "tier": "B",
            }
            market_health = metrics.spread_cents is not None and metrics.depth_yes_5c + metrics.depth_no_5c > 0
            eligible, exclude_reasons = _eligible_from_metrics(metrics, config)
            state_name = decide_state(
                inv,
                net=net,
                time_in_inventory_min=hold_min,
                market_health=market_health,
                incentives_alive=incentives_alive and incentives_ok,
                time_remaining_hours=metrics.time_remaining_hours,
                config=config,
            )
            if config.hard_cap > 0:
                if total_inv >= config.hard_cap:
                    if ticker not in hard_cap_tickers:
                        await _cancel_orders_for_ticker(ticker, "hard_cap")
                        logger.error(
                            "hard_cap hit ticker=%s total_inv=%s cap=%s",
                            ticker,
                            total_inv,
                            config.hard_cap,
                        )
                        hard_cap_tickers.add(ticker)
                    state_name = STATE_FLATTEN
                elif ticker in hard_cap_tickers:
                    hard_cap_tickers.remove(ticker)
            if global_halt:
                state_name = STATE_FLATTEN if total_inv > 0 else STATE_PAUSE
            if hold_min is not None and hold_min >= config.kill_inventory_stuck_minutes:
                kill_triggered = True
                kill_reason = f"inventory_stuck ticker={ticker} minutes={hold_min:.1f}"
                break
            if now < pause_until_ts:
                state_name = STATE_FLATTEN if (inv.inv_yes + inv.inv_no) > 0 else STATE_PAUSE

            working = state.working_orders.setdefault(ticker, [])
            tier = state.ev_tiers.get(ticker, "B")
            if adverse_risk > config.adverse_tier_downgrade and tier == "A":
                tier = "B"
            discount = _tier_discount(tier, config)
            if adverse_risk > config.adverse_discount_zero:
                discount = 0
                if tier == "A":
                    tier = "B"
            if adverse_risk >= config.soft_throttle_adverse and now >= soft_throttle_until:
                soft_throttle_until = now + (config.soft_throttle_minutes * 60)
                logger.info(
                    "soft_throttle adverse_risk=%s pause_minutes=%s ticker=%s",
                    round(adverse_risk, 3),
                    config.soft_throttle_minutes,
                    ticker,
                )
            soft_throttle_active = now < soft_throttle_until
            effective_tier = tier
            if soft_throttle_active:
                if tier == "A":
                    effective_tier = "B"
                elif tier == "B":
                    effective_tier = "C"
                else:
                    effective_tier = "C"
                discount = _tier_discount(effective_tier, config)
            if ticker in temp_ev_rows:
                temp_ev_rows[ticker]["tier"] = effective_tier
                temp_ev_rows[ticker]["discount"] = discount
            if effective_tier == "A":
                edge_min_override = max(1, config.edge_min_cents - discount)
            elif effective_tier == "C":
                edge_min_override = config.edge_min_cents + 2
            else:
                edge_min_override = max(1, config.edge_min_cents - min(1, discount))
            allow_step_in = (
                incentive_ev > 0
                and adverse_risk < config.adverse_step_in_max
                and effective_tier == "A"
                and not soft_throttle_active
            )

            if config.paper_mode and working:
                for order in list(working):
                    if order.action == "buy":
                        best = metrics.best_yes_bid if order.side == "yes" else metrics.best_no_bid
                        if best is None or order.price_cents < best:
                            continue
                    else:
                        best = metrics.implied_yes_ask if order.side == "yes" else metrics.implied_no_ask
                        if best is None or order.price_cents > best:
                            continue
                    fill_prob = min(config.max_fill_prob, _fill_probability(metrics, order.side))
                    if random.random() <= fill_prob:
                        portfolio.apply_fill(
                            ticker, order.side, order.action, order.price_cents, order.qty
                        )
                        working.remove(order)
                        fill_history.setdefault(ticker, deque()).append(time.monotonic())
                        cash_delta = order.price_cents * order.qty
                        if order.action == "buy":
                            cash_delta = -cash_delta
                        fill_cash_history.setdefault(ticker, deque()).append(
                            (time.monotonic(), cash_delta)
                        )
                        _log_csv(
                            fills_path,
                            [
                                _utc_now(),
                                ticker,
                                order.side,
                                order.action,
                                order.price_cents,
                                order.qty,
                                portfolio.cash_cents,
                            ],
                        )
                        diag_fill_events += 1
                        summary_fill_events += 1

            yes_mid = _mid_price_cents(metrics.best_yes_bid, metrics.implied_yes_ask)
            no_mid = _mid_price_cents(metrics.best_no_bid, metrics.implied_no_ask)
            last_mids[ticker] = (yes_mid, no_mid)

            if config.max_market_loss_cents > 0:
                pnl_cents = _market_pnl_cents(inv, last_mids[ticker])
                if pnl_cents is not None and pnl_cents <= -config.max_market_loss_cents:
                    await _cancel_orders_for_ticker(ticker, "market_loss_cap")
                    paused_tickers[ticker] = day_start + timedelta(days=1)
                    logger.error(
                        "market_loss_cap hit ticker=%s pnl_cents=%s pause_until_utc=%s",
                        ticker,
                        pnl_cents,
                        day_start + timedelta(days=1),
                    )
                    continue

            desired = _build_desired_orders(
                state_name,
                inv,
                metrics,
                working,
                config,
                edge_min_override=edge_min_override,
                allow_step_in=allow_step_in,
            )
            if soft_throttle_active:
                for order in desired:
                    order["qty"] = 1
            diag_state_counts[state_name] = diag_state_counts.get(state_name, 0) + 1
            if not eligible:
                key = ";".join(exclude_reasons) if exclude_reasons else "ineligible"
                diag_exclusion_counts[key] = diag_exclusion_counts.get(key, 0) + 1
            diag_desired_orders += len(desired)
            if len(diag_samples) < 5:
                diag_samples.append(
                    {
                        "ticker": ticker,
                        "state": state_name,
                        "eligible": eligible,
                        "reasons": ";".join(exclude_reasons) if exclude_reasons else "",
                        "spread": metrics.spread_cents,
                        "depth": round(metrics.depth_yes_5c + metrics.depth_no_5c, 2),
                        "imbalance": round(metrics.imbalance, 3),
                        "desired": len(desired),
                    }
                )

            cancels: list[WorkingOrder] = []
            creates: list[dict[str, Any]] = []

            for order in list(working):
                if order.cancel_requested_ts is not None:
                    continue
                age = time.monotonic() - order.created_ts
                if age > config.max_order_age_s:
                    cancels.append(order)
                    continue
                target = next((d for d in desired if d["side"] == order.side and d["action"] == order.action), None)
                if target is None:
                    cancels.append(order)
                    continue
                if _needs_reprice(order, target["price"], time.monotonic(), config):
                    cancels.append(order)

            for target in desired:
                existing = _select_best_order(working, target["side"], target["action"])
                if existing is None:
                    creates.append(target)
                    continue
                if existing.cancel_requested_ts is not None:
                    continue
                if existing in cancels:
                    continue

            for order in cancels:
                if dry_actions:
                    working.remove(order)
                    diag_canceled_orders += 1
                    summary_actions += 1
                    _log_csv(
                        actions_path,
                        [
                            _utc_now(),
                            ticker,
                            "cancel",
                            order.side,
                            order.price_cents,
                            order.qty,
                            order.order_id,
                            "dry_run",
                        ],
                    )
                    continue
                try:
                    await asyncio.to_thread(rest.cancel_order, order.order_id)
                except Exception as exc:
                    logger.warning(
                        "action=cancel status=failed ticker=%s order_id=%s err=%s",
                        ticker,
                        order.order_id,
                        exc,
                    )
                else:
                    diag_canceled_orders += 1
                    summary_actions += 1
                    logger.info(
                        "action=cancel status=ok ticker=%s order_id=%s",
                        ticker,
                        order.order_id,
                    )
                    order.cancel_requested_ts = time.monotonic()
                    _log_csv(
                        actions_path,
                        [_utc_now(), ticker, "cancel", order.side, order.price_cents, order.qty, order.order_id, "refresh"],
                    )

            for target in creates:
                if not eligible and state_name != STATE_FLATTEN:
                    continue
                if state_name == STATE_PAUSE:
                    continue
                if target["action"] == "buy":
                    if config.paper_mode and portfolio.cash_cents <= 0:
                        continue
                    if not config.paper_mode and live_cash_cents is not None:
                        cash_limit = live_cash_cents
                        if config.bot_budget_dollars > 0:
                            cash_limit = min(cash_limit, int(config.bot_budget_dollars * 100))
                        if cash_limit <= 0:
                            continue
                diag_created_orders += 1
                payload = {
                    "ticker": ticker,
                    "type": "limit",
                    "action": target["action"],
                    "side": target["side"],
                    "count": target["qty"],
                    "client_order_id": f"phase2-{ticker}-{target['side']}-{target['action']}-{int(time.time()*1000)}",
                }
                if target["side"] == "yes":
                    payload["yes_price"] = target["price"]
                else:
                    payload["no_price"] = target["price"]
                if dry_actions:
                    order_id = f"dry-{ticker}-{target['side']}-{target['action']}-{int(time.time()*1000)}"
                else:
                    try:
                        response = await asyncio.to_thread(rest.place_order, payload)
                    except Exception as exc:
                        logger.warning(
                            "order_failed ticker=%s side=%s action=%s err=%s",
                            ticker,
                            target["side"],
                            target["action"],
                            exc,
                        )
                        continue
                    order_id = None
                    if isinstance(response, dict):
                        order_id = response.get("order_id") or response.get("id")
                        if isinstance(response.get("order"), dict):
                            order_id = response["order"].get("order_id") or response["order"].get("id") or order_id
                    if order_id is None:
                        continue
                order = WorkingOrder(
                    order_id=str(order_id),
                    market_ticker=ticker,
                    side=target["side"],
                    action=target["action"],
                    price_cents=target["price"],
                    qty=target["qty"],
                    created_ts=time.monotonic(),
                )
                working.append(order)
                summary_actions += 1
                _log_csv(
                    actions_path,
                    [_utc_now(), ticker, "place", order.side, order.price_cents, order.qty, order.order_id, state_name],
                )

            _log_csv(
                decisions_path,
                [
                    _utc_now(),
                    ticker,
                    state_name,
                    int(eligible),
                    ";".join(exclude_reasons),
                    inv.inv_yes,
                    inv.inv_no,
                    net,
                    metrics.spread_cents or "",
                    round(metrics.depth_yes_5c + metrics.depth_no_5c, 2),
                    round(metrics.imbalance, 3),
                    round(metrics.time_remaining_hours, 2) if metrics.time_remaining_hours is not None else "",
                    int(incentives_alive),
                ],
            )

        if kill_triggered:
            await _cancel_all_orders(kill_reason or "kill_switch")
            logger.error("kill_switch reason=%s", kill_reason or "unknown")
            raise SystemExit("Kill switch triggered; manual restart required.")

        if config.paper_mode:
            equity_cents = _equity_cents(portfolio.cash_cents, portfolio.positions, last_mids)
            positions_view = portfolio.positions
            cash_cents_view = portfolio.cash_cents
        else:
            cash_cents_view = live_cash_cents or 0
            if config.bot_budget_dollars > 0:
                cash_cents_view = min(cash_cents_view, int(config.bot_budget_dollars * 100))
            equity_cents = _equity_cents(cash_cents_view, positions_cache, last_mids)
            positions_view = positions_cache
            if not day_start_equity_set and live_cash_cents is not None:
                day_start_equity = equity_cents
                peak_equity = equity_cents
                day_start_equity_set = True
        if datetime.now(timezone.utc).date() != day_start:
            day_start = datetime.now(timezone.utc).date()
            day_start_equity = equity_cents
            peak_equity = equity_cents
            worst_window_cash = 0
            daily_halt_day = None
            hourly_equity_window.clear()
            paused_tickers = {k: v for k, v in paused_tickers.items() if v > day_start}
        peak_equity = max(peak_equity, equity_cents)
        day_pnl = equity_cents - day_start_equity
        if day_pnl < worst_window_cash:
            worst_window_cash = day_pnl
        if (
            config.max_day_loss_cents > 0
            and day_pnl <= -config.max_day_loss_cents
            and daily_halt_day != day_start
        ):
            daily_halt_day = day_start
            await _cancel_all_orders("daily_loss_cap")
            logger.error(
                "daily_loss_cap hit day_pnl_cents=%s halt_until_utc=%s",
                day_pnl,
                day_start + timedelta(days=1),
            )
        if (
            paper_state_path
            and config.paper_mode
            and now - last_paper_state_ts >= config.paper_state_sync_s
        ):
            try:
                _save_paper_state(
                    paper_state_path,
                    portfolio,
                    day_start,
                    day_start_equity,
                    peak_equity,
                    worst_window_cash,
                )
                last_paper_state_ts = now
            except Exception as exc:
                logger.warning("paper_state_save_failed path=%s err=%s", paper_state_path, exc)
        if config.kill_max_drawdown_cents > 0 and (
            equity_cents <= peak_equity - config.kill_max_drawdown_cents
        ):
            kill_triggered = True
            kill_reason = f"drawdown equity_cents={equity_cents} peak_cents={peak_equity}"
            await _cancel_all_orders(kill_reason)
            logger.error("kill_switch reason=%s", kill_reason)
            raise SystemExit("Kill switch triggered; manual restart required.")

        state.ev_tiers = _assign_tiers(temp_scores, config)
        for ticker, tier in state.ev_tiers.items():
            temp_discounts[ticker] = _tier_discount(tier, config)
            if ticker in temp_ev_rows:
                temp_ev_rows[ticker]["discount"] = temp_discounts[ticker]
        state.ev_scores = temp_scores
        state.incentive_discounts = temp_discounts
        latest_ev_rows = temp_ev_rows

        if now - last_diag_ts >= config.diag_every_s:
            window_s = config.diag_every_s
            window_h = window_s / 3600.0
            top_exclusions = sorted(
                diag_exclusion_counts.items(), key=lambda kv: kv[1], reverse=True
            )[:3]
            exclusion_summary = "|".join(f"{k}:{v}" for k, v in top_exclusions)
            state_summary = "|".join(f"{k}:{v}" for k, v in sorted(diag_state_counts.items()))
            logger.info(
                "diag_status desired=%s created=%s canceled=%s fills=%s states=%s exclusions=%s",
                diag_desired_orders,
                diag_created_orders,
                diag_canceled_orders,
                diag_fill_events,
                state_summary or "none",
                exclusion_summary or "none",
            )
            if diag_samples:
                sample_str = " | ".join(
                    f"{row['ticker']} state={row['state']} eligible={int(row['eligible'])} "
                    f"reasons={row['reasons']} spread={row['spread']} depth={row['depth']} "
                    f"imb={row['imbalance']} desired={row['desired']}"
                    for row in diag_samples
                )
                logger.info("diag_samples %s", sample_str)
            if latest_ev_rows:
                for ticker, ev in latest_ev_rows.items():
                    _log_csv(
                        phase4_path,
                        [
                            _utc_now(),
                            ticker,
                            ev.get("tier", "B"),
                            round(ev["incentive_ev"], 6),
                            round(ev["spread_quality"], 6),
                            round(ev["adverse_risk"], 6),
                            round(ev["inventory_risk"], 6),
                            round(ev["total_ev"], 6),
                            ev["discount"],
                            ev["fills_30m"],
                        ],
                    )
                    cash_hist = fill_cash_history.setdefault(ticker, deque())
                    while cash_hist and cash_hist[0][0] < now - window_s:
                        cash_hist.popleft()
                    price_pnl = sum(delta for _, delta in cash_hist)
                    incentive_rev = ev["incentive_ev"] * window_h
                    net_pnl = price_pnl + incentive_rev
                    _log_csv(
                        phase4_validation_path,
                        [
                            _utc_now(),
                            ticker,
                            ev.get("tier", "B"),
                            ev.get("discount", 0),
                            round(ev.get("adverse_risk", 0.0), 6),
                            price_pnl,
                            round(incentive_rev, 6),
                            round(net_pnl, 6),
                        ],
                    )
            last_diag_ts = now
            diag_state_counts = {}
            diag_exclusion_counts = {}
            diag_desired_orders = 0
            diag_created_orders = 0
            diag_canceled_orders = 0
            diag_fill_events = 0
            diag_samples = []

        if now - last_log_ts >= config.log_every_s:
            if config.paper_mode:
                total_inv_yes = sum(p.inv_yes for p in portfolio.positions.values())
                total_inv_no = sum(p.inv_no for p in portfolio.positions.values())
                logger.info(
                    "paper_status cash_cents=%s inv_yes=%s inv_no=%s markets=%s",
                    portfolio.cash_cents,
                    total_inv_yes,
                    total_inv_no,
                    len(tickers),
                )
            else:
                total_inv_yes = sum(p.inv_yes for p in positions_cache.values())
                total_inv_no = sum(p.inv_no for p in positions_cache.values())
                logger.info(
                    "live_status inv_yes=%s inv_no=%s markets=%s",
                    total_inv_yes,
                    total_inv_no,
                    len(tickers),
                )
            last_log_ts = now

        if now - last_dashboard_ts >= max(60.0, config.diag_every_s / 2):
            tier_counts = Counter(state.ev_tiers.values())
            total_inv_yes = sum(p.inv_yes for p in positions_view.values())
            total_inv_no = sum(p.inv_no for p in positions_view.values())
            logger.info(
                "dashboard cash_cents=%s equity_cents=%s inv_yes=%s inv_no=%s tiers=A:%s B:%s C:%s day_pnl_cents=%s worst_window_cents=%s soft_throttle=%s",
                cash_cents_view,
                equity_cents,
                total_inv_yes,
                total_inv_no,
                tier_counts.get("A", 0),
                tier_counts.get("B", 0),
                tier_counts.get("C", 0),
                day_pnl,
                worst_window_cash,
                int(now < soft_throttle_until),
            )
            last_dashboard_ts = now

        if now - last_summary_ts >= config.diag_every_s:
            net_cash_change = portfolio.cash_cents - summary_start_cash
            fill_rate = summary_fill_events / summary_actions if summary_actions else 0.0
            total_inv_yes = sum(p.inv_yes for p in portfolio.positions.values())
            total_inv_no = sum(p.inv_no for p in portfolio.positions.values())
            _log_csv(
                summary_path,
                [
                    _utc_now(),
                    summary_fill_events,
                    summary_actions,
                    round(fill_rate, 4),
                    net_cash_change,
                    total_inv_yes,
                    total_inv_no,
                ],
            )
            last_summary_ts = now
            summary_fill_events = 0
            summary_actions = 0
            summary_start_cash = portfolio.cash_cents

        await asyncio.sleep(config.tick_s)


async def _run(config_path: Optional[str]) -> None:
    if config_path:
        config = load_phase2_config(config_path)
    else:
        config = Phase2Config()
    tickers = load_top_markets_from_phase1(config.phase1_db_path, config.max_watchlist)
    if not tickers:
        raise SystemExit(f"No markets found in {config.phase1_db_path}")

    manager = BookManager(stale_after_s=config.rest_fallback_age_s)
    await asyncio.gather(
        kalshi_ws_run(manager, tickers),
        phase2_loop(manager, tickers, config),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase2 incentives market maker")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config with phase2_* values (markets_file required if different)",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
