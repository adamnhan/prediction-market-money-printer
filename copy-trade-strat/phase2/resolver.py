from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests

from cache import Phase2Cache
from fetchers import DEFAULT_BASE_URL, fetch_market_by_condition_id


logger = logging.getLogger("copy_trade_phase2")


ACTIVE_TTL_S = 6 * 60 * 60
INACTIVE_TTL_S = 7 * 24 * 60 * 60


def _parse_time(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value > 10_000_000_000:
            return int(value / 1000)
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.astimezone(timezone.utc).timestamp())
        except ValueError:
            return None
    return None


def _extract_outcomes(payload: dict[str, Any]) -> list[str] | None:
    outcomes = payload.get("outcomes") or payload.get("outcomeNames") or payload.get("outcome_names")
    if isinstance(outcomes, list) and outcomes:
        return [str(x) for x in outcomes]
    return None


def _extract_token_ids(payload: dict[str, Any]) -> dict[str, Any] | None:
    tokens = (
        payload.get("clobTokenIds")
        or payload.get("clob_token_ids")
        or payload.get("outcomeTokenIds")
        or payload.get("outcome_token_ids")
        or payload.get("tokenIds")
    )
    if isinstance(tokens, dict):
        return tokens
    if isinstance(tokens, list):
        outcomes = _extract_outcomes(payload)
        if outcomes and len(outcomes) == len(tokens):
            return dict(zip(outcomes, tokens))
    return None


def _map_status(payload: dict[str, Any]) -> str | None:
    for key in ("status", "marketStatus", "market_status", "state"):
        if key in payload and payload[key] is not None:
            return str(payload[key])
    if payload.get("resolved") is True:
        return "resolved"
    if payload.get("closed") is True:
        return "closed"
    return None


def _canonicalize_market(payload: dict[str, Any], condition_id: str) -> dict[str, Any]:
    title = payload.get("title") or payload.get("question") or payload.get("name")
    category = payload.get("category") or payload.get("categories")
    if isinstance(category, list):
        category = ",".join(str(x) for x in category)

    return {
        "condition_id": condition_id,
        "market_id": payload.get("id") or payload.get("market_id") or payload.get("marketId"),
        "slug": payload.get("slug") or payload.get("marketSlug") or payload.get("market_slug"),
        "title": title,
        "category": category,
        "outcomes": _extract_outcomes(payload),
        "token_ids": _extract_token_ids(payload),
        "status": _map_status(payload),
        "end_time": _parse_time(payload.get("end_time") or payload.get("endTime") or payload.get("closeTime")),
        "source": "data_api",
    }


def _is_stale(meta: dict[str, Any], now_ts: int) -> bool:
    last_ref = meta.get("last_refreshed_at") or 0
    status = (meta.get("status") or "").lower()
    ttl = ACTIVE_TTL_S
    if status in {"closed", "resolved", "settled"}:
        ttl = INACTIVE_TTL_S
    return (now_ts - int(last_ref)) > ttl


def resolve_condition(
    cache: Phase2Cache,
    condition_id: str,
    session: requests.Session | None = None,
) -> dict[str, Any] | None:
    now_ts = int(time.time())
    cached = cache.get_meta(condition_id)
    if cached and not _is_stale(cached, now_ts):
        return cached

    base_url = os.getenv("COPY_TRADE_DATA_API_BASE", DEFAULT_BASE_URL)
    sess = session or requests.Session()
    markets = fetch_market_by_condition_id(sess, condition_id, base_url=base_url)
    if not markets:
        return cached

    # Try to find exact condition_id match when present.
    chosen = None
    for market in markets:
        raw_cond = market.get("conditionId") or market.get("condition_id")
        if raw_cond and str(raw_cond).lower() == condition_id.lower():
            chosen = market
            break
    if chosen is None:
        chosen = markets[0]

    meta = _canonicalize_market(chosen, condition_id)
    meta["last_refreshed_at"] = now_ts
    cache.upsert_meta(meta)
    return meta
