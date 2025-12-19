# trading_engine/market_utils.py

from typing import Any, Dict, Optional


def map_status_from_target(target: Dict[str, Any]) -> tuple[str, str]:
    """
    Given a raw market dict from Kalshi, extract:
      - raw_status (original)
      - status (normalized: 'open' | 'not_open' | 'closed' | 'unknown')
    """
    raw_status = (
        target.get("lifecycle_status")
        or target.get("status")
        or target.get("state")
        or "unknown"
    )

    raw_status_norm = raw_status.lower() if isinstance(raw_status, str) else "unknown"

    status = "unknown"
    if raw_status_norm in ("open", "trading", "active", "initialized"):
        status = "open"
    elif raw_status_norm in ("pending", "upcoming"):
        status = "not_open"
    elif raw_status_norm in ("closed", "settled", "expired"):
        status = "closed"

    return raw_status, status


def extract_yes_price_from_target(target: Dict[str, Any]) -> Optional[float]:
    """
    Best-effort extraction of a YES price in 0–1 range from the market payload.
    Mirrors the existing logic in TradingEngine.update_market_metadata.
    """
    data_price_yes = (
        target.get("last_price_yes")
        or target.get("last_price")
        or target.get("yes_last_price")
    )
    best_bid_yes = target.get("yes_bid") or target.get("best_bid_yes")
    best_ask_yes = target.get("yes_ask") or target.get("best_ask_yes")

    price_yes: Optional[float] = None

    # Use mid of bid/ask if available
    if best_bid_yes is not None and best_ask_yes is not None:
        try:
            price_yes = (float(best_bid_yes) + float(best_ask_yes)) / 2.0
        except (TypeError, ValueError):
            price_yes = None

    # Fallback to last price fields
    if price_yes is None and data_price_yes is not None:
        try:
            price_yes = float(data_price_yes)
        except (TypeError, ValueError):
            price_yes = None

    # Normalize from cents to 0–1 if needed
    if price_yes is not None and price_yes > 1.0:
        price_yes = price_yes / 100.0

    # Clamp to [0,1]
    if price_yes is not None:
        price_yes = max(0.0, min(1.0, price_yes))

    return price_yes
