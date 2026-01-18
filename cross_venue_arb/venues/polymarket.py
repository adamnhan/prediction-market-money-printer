"""Polymarket venue adapter skeleton."""

from __future__ import annotations

from dataclasses import dataclass

from cross_venue_arb.config import CONFIG


@dataclass
class PolymarketVenue:
    name: str = "polymarket"

    def connect(self) -> None:
        if not CONFIG.polymarket.api_url:
            raise RuntimeError("POLYMARKET_API_URL is not set")

    def fetch_markets(self) -> list[dict]:
        return []
