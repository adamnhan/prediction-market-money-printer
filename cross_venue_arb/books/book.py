"""Order book representation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrderBook:
    venue: str
    market_id: str
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
