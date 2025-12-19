# trading_engine/models.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal, Dict, Any


@dataclass
class MarketEntry:
    """
    A single market we are tracking under an event.
    """
    event_ticker: str
    market_ticker: str

    # lifecycle/status (we'll populate later from real data)
    status: str = "unknown"  # e.g. "not_open", "open", "closed"

    # last known yes/no prices from WebSocket or REST
    last_price_yes: Optional[float] = None
    last_price_no: Optional[float] = None

    # timestamp of the last update
    last_update_ts: Optional[str] = None



@dataclass
class Position:
    # Core identity / linkages
    id: int = -1
    event_ticker: str = ""
    market_ticker: str = ""
    side: Literal["NO"] = "NO"

    # Sizing & prices
    qty: int = 0
    entry_price: float = 0.0
    current_price: float = 0.0  # mark price for NO

    # Lifecycle
    status: str = "open"  # "open" | "closed" | etc.
    entry_ts: Optional[datetime] = None
    exit_ts: Optional[datetime] = None

    # PnL
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

@dataclass
class MarketUpdate:
    """
    Normalized market update coming from WebSocket (or other live feeds).

    Prices are in Kalshi "price int" units (0–100), consistent with Position.entry_price/current_price.
    """
    market_ticker: str
    msg_type: str                      # "orderbook_snapshot", "orderbook_delta", etc.

    best_yes: Optional[int] = None     # from snapshots (top of yes ladder)
    best_no: Optional[int] = None      # from snapshots (top of no ladder)

    last_price: Optional[int] = None   # from deltas: price that changed
    last_side: Optional[str] = None    # "yes" | "no" from deltas

    ts: Optional[str] = None           # ISO timestamp if present

    raw: Optional[Dict[str, Any]] = None  # full raw WS message for debugging
