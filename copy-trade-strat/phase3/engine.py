from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from config import Phase3Config
from storage import Phase3Store


SKIP_SIDE_NOT_ALLOWED = "SKIP_SIDE_NOT_ALLOWED"
SKIP_NO_META = "SKIP_NO_META"
SKIP_MARKET_INACTIVE = "SKIP_MARKET_INACTIVE"
SKIP_CATEGORY = "SKIP_CATEGORY"
SKIP_STALE = "SKIP_STALE"
SKIP_TOO_SMALL = "SKIP_TOO_SMALL"
SKIP_EXPOSURE_MARKET_CAP = "SKIP_EXPOSURE_MARKET_CAP"
SKIP_EXPOSURE_TOTAL_CAP = "SKIP_EXPOSURE_TOTAL_CAP"
SKIP_DAILY_NOTIONAL_CAP = "SKIP_DAILY_NOTIONAL_CAP"
SKIP_RATE_LIMIT = "SKIP_RATE_LIMIT"


def _normalize_category(value: str | None) -> set[str]:
    if not value:
        return set()
    parts = [part.strip().lower() for part in value.split(",") if part.strip()]
    return set(parts)


def _daily_key(now_ts: int) -> str:
    dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
    return f"daily_notional_{dt.strftime('%Y-%m-%d')}"


def _build_intent(
    trade: dict[str, Any],
    status: str,
    skip_reason: str | None,
    my_size: float | None,
    my_limit_price: float | None,
    notes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "created_at": int(time.time()),
        "target_wallet": trade.get("target_wallet"),
        "observed_trade_id": trade["id"],
        "tx_hash": trade.get("tx_hash"),
        "condition_id": trade.get("condition_id"),
        "outcome": trade.get("outcome"),
        "side": trade.get("side"),
        "target_price": trade.get("price"),
        "target_size": trade.get("size"),
        "my_size": my_size,
        "my_limit_price": my_limit_price,
        "intent_status": status,
        "skip_reason": skip_reason,
        "notes": json.dumps(notes) if notes else None,
    }


def decide_trade(
    store: Phase3Store,
    trade: dict[str, Any],
    cfg: Phase3Config,
) -> dict[str, Any]:
    now_ts = int(time.time())

    if trade.get("side", "").upper() not in cfg.copy_types:
        return _build_intent(trade, "SKIPPED", SKIP_SIDE_NOT_ALLOWED, None, None)

    if not trade.get("meta_title"):
        return _build_intent(trade, "SKIPPED", SKIP_NO_META, None, None)

    status = (trade.get("meta_status") or "").lower()
    if status in {"closed", "resolved", "settled"}:
        return _build_intent(trade, "SKIPPED", SKIP_MARKET_INACTIVE, None, None)

    if cfg.categories_allowlist:
        trade_categories = _normalize_category(trade.get("meta_category"))
        allow = {c.lower() for c in cfg.categories_allowlist}
        if trade_categories and trade_categories.isdisjoint(allow):
            return _build_intent(trade, "SKIPPED", SKIP_CATEGORY, None, None)
        if not trade_categories:
            return _build_intent(trade, "SKIPPED", SKIP_CATEGORY, None, None)

    age_s = now_ts - int(trade.get("timestamp") or now_ts)
    if age_s > cfg.max_staleness_seconds:
        return _build_intent(trade, "SKIPPED", SKIP_STALE, None, None)

    target_price = float(trade.get("price") or 0.0)
    my_size = float(cfg.fixed_copy_size_shares)
    my_notional = my_size * target_price

    if my_notional < cfg.min_my_trade_size_notional_usd:
        return _build_intent(trade, "SKIPPED", SKIP_TOO_SMALL, None, None)

    per_market_open = store.get_open_exposure(trade["condition_id"])
    total_open = store.get_summary_value("total_open_notional")
    daily_key = _daily_key(now_ts)
    daily_used = store.get_summary_value(daily_key)

    if per_market_open + my_notional > cfg.max_open_notional_per_market_usd:
        return _build_intent(trade, "SKIPPED", SKIP_EXPOSURE_MARKET_CAP, None, None)
    if total_open + my_notional > cfg.max_open_notional_total_usd:
        return _build_intent(trade, "SKIPPED", SKIP_EXPOSURE_TOTAL_CAP, None, None)
    if daily_used + my_notional > cfg.max_daily_notional_usd:
        return _build_intent(trade, "SKIPPED", SKIP_DAILY_NOTIONAL_CAP, None, None)

    if store.count_intents_last_hour(now_ts) >= cfg.max_trades_per_hour:
        return _build_intent(trade, "SKIPPED", SKIP_RATE_LIMIT, None, None)

    slip = cfg.max_slippage_cents / 100.0
    my_limit_price = min(target_price + slip, 0.99)

    intent = _build_intent(
        trade,
        "NEEDS_BOOKCHECK",
        None,
        my_size,
        my_limit_price,
        notes={"config": asdict(cfg), "age_s": age_s},
    )

    # Update exposure for paper intents.
    store.set_open_exposure(trade["condition_id"], per_market_open + my_notional, now_ts)
    store.set_summary_value("total_open_notional", total_open + my_notional, now_ts)
    store.set_summary_value(daily_key, daily_used + my_notional, now_ts)

    return intent
