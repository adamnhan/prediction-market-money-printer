"""Arbitrage opportunity models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Leg:
    venue: str
    market_id: str
    action: str
    outcome: str
    limit_price: float
    available_size: float


@dataclass(frozen=True)
class Opportunity:
    game_key: str
    ts: float
    direction: str
    size_max: float
    edge_per_contract: float
    expected_profit_max: float
    legs: list[Leg]
    reject_reason: str | None = None
