from __future__ import annotations

from typing import Any


def trade_dedupe_key(trade: dict[str, Any]) -> tuple[Any, ...]:
    return (
        trade.get("tx_hash"),
        trade.get("condition_id"),
        trade.get("side"),
        trade.get("price"),
        trade.get("size"),
    )
