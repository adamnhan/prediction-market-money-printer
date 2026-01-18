"""Shared configuration for cross-venue arbitrage."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class KalshiConfig:
    key_id: str | None = os.getenv("KALSHI_KEY_ID")
    private_key_path: str | None = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    ws_url: str | None = os.getenv("KALSHI_WS_URL")
    rest_url: str | None = os.getenv("KALSHI_REST_URL")
    order_url: str | None = os.getenv("KALSHI_ORDER_URL")


@dataclass(frozen=True)
class PolymarketConfig:
    api_key: str | None = os.getenv("POLYMARKET_API_KEY")
    api_secret: str | None = os.getenv("POLYMARKET_API_SECRET")
    api_url: str | None = os.getenv("POLYMARKET_API_URL")
    ws_url: str | None = os.getenv("POLYMARKET_WS_URL")


@dataclass(frozen=True)
class AppConfig:
    kalshi: KalshiConfig = KalshiConfig()
    polymarket: PolymarketConfig = PolymarketConfig()
    db_path: str = os.getenv("CROSS_VENUE_ARB_DB", "./data/cross_venue_arb.sqlite")


CONFIG = AppConfig()
