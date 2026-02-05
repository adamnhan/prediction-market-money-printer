from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from dateutil import parser as dt_parser


def _to_timestamp(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Handle ms timestamps by downscaling if needed.
        if value > 10_000_000_000:
            return int(value / 1000)
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            dt = dt_parser.isoparse(value)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except (ValueError, TypeError):
            return None
    return None


def _get_first(payload: dict[str, Any], keys: list[str]) -> Any | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def normalize_trade(raw: dict[str, Any], target_wallet: str) -> dict[str, Any] | None:
    tx_hash = _get_first(raw, ["transactionHash", "transaction_hash", "txHash", "tx_hash"])
    condition_id = _get_first(raw, ["conditionId", "condition_id"])
    side = _get_first(raw, ["side", "direction"])
    price = _get_first(raw, ["price", "pricePerShare", "price_per_share"])
    size = _get_first(raw, ["size", "amount", "quantity"])
    outcome = _get_first(raw, ["outcome", "outcomeToken", "outcome_token"])
    proxy_wallet = _get_first(raw, ["proxyWallet", "proxy_wallet"])
    timestamp = _get_first(raw, ["timestamp", "createdAt", "created_at", "time"])

    if tx_hash is None or condition_id is None or side is None or price is None or size is None:
        return None

    try:
        price_f = float(price)
        size_f = float(size)
    except (TypeError, ValueError):
        return None

    ts = _to_timestamp(timestamp)
    if ts is None:
        ts = int(time.time())

    return {
        "observed_trade_id": None,
        "target_wallet": target_wallet,
        "proxy_wallet": proxy_wallet,
        "tx_hash": str(tx_hash),
        "condition_id": str(condition_id),
        "outcome": outcome,
        "side": str(side).upper(),
        "price": price_f,
        "size": size_f,
        "timestamp": ts,
        "ingested_at": int(time.time()),
        "source": "data_api",
    }


def normalize_position(raw: dict[str, Any], target_wallet: str, snapshot_time: int) -> dict[str, Any] | None:
    condition_id = _get_first(raw, ["conditionId", "condition_id"])
    outcome = _get_first(raw, ["outcome", "outcomeToken", "outcome_token"])
    size = _get_first(raw, ["size", "balance", "amount"])
    if condition_id is None or size is None:
        return None
    try:
        size_f = float(size)
    except (TypeError, ValueError):
        return None
    return {
        "target_wallet": target_wallet,
        "condition_id": str(condition_id),
        "outcome": outcome,
        "size": size_f,
        "snapshot_time": snapshot_time,
    }
