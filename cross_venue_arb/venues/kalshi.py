"""Kalshi venue adapter skeleton."""

from __future__ import annotations

from dataclasses import dataclass

from cross_venue_arb.config import CONFIG


@dataclass
class KalshiVenue:
    name: str = "kalshi"

    def connect(self) -> None:
        if not CONFIG.kalshi.ws_url:
            raise RuntimeError("KALSHI_WS_URL is not set")

    def fetch_markets(self) -> list[dict]:
        return []
