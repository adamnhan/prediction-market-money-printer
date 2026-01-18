"""Arbitrage signal definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ArbSignal:
    left_venue: str
    right_venue: str
    market_id: str
    edge_bps: float
