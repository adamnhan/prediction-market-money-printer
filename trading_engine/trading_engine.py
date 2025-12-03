from typing import Dict, List, Any, Optional, Literal
from src.event_service import fetch_event_metadata
from dataclasses import dataclass
from src import platform_ops  
from datetime import datetime

@dataclass
class MarketEntry:
    """
    A single market we are tracking under an event.
    """
    event_ticker: str
    market_ticker: str

    # lifecycle/status (we'll populate later from real data)
    status: str = "unknown"  # e.g. "not_open", "open", "closed"

    # simple price placeholders (YES/NO last or mid prices)
    last_price_yes: Optional[float] = None
    last_price_no: Optional[float] = None

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



class TradingEngine:
    def __init__(self):
        # Events + logs
        self._events: Dict[str, Dict[str, Any]] = {}
        self._logs: List[str] = []

        # Capital tracking (private but consistent)
        self._total_capital = 10000.0
        self._used_capital = 0.0

        # Markets keyed by market_ticker
        self.markets: Dict[str, MarketEntry] = {}

        # Position storage + ID generator
        self._next_position_id = 1
        self.positions: Dict[int, Position] = {}  # position_id -> Position



    # ---- Core actions ----

    async def add_event(self, ticker: str) -> None:
        ticker = ticker.upper()
        if ticker in self._events:
            self.log(f"[EVENT] {ticker} already tracked")
            return

        meta = await fetch_event_metadata(ticker)

        if meta.get("ok"):
            event = meta["event"]
            title = event.get("title") or ticker
            status = event.get("status") or "unknown"
            category = event.get("category")
            sub_category = event.get("sub_category")

            self._events[ticker] = {
                "ticker": ticker,
                "title": title,
                "status": status,
                "category": category,
                "sub_category": sub_category,
                "price": 0.42,
                "pnl": 0.0,
            }

            self.log(
                f"[EVENT] added {ticker} "
                f"title={title!r} status={status} category={category} sub_category={sub_category}"
            )

        else:
            err = meta.get("error", "unknown error from Kalshi")
            self.log(f"[EVENT] failed to fetch metadata for {ticker}: {err} — using fallback")

            self._events[ticker] = {
                "ticker": ticker,
                "title": f"Fallback {ticker}",
                "status": "unknown",
                "category": None,
                "sub_category": None,
                "price": 0.42,
                "pnl": 0.0,
            }

    def remove_event(self, ticker: str) -> None:
        if ticker in self._events:
            self._events.pop(ticker)
            self.log(f"[EVENT] removed {ticker}")

    def set_total_capital(self, amount: float):
        self._total_capital = float(amount)
        self.log(f"Set total capital to {self._total_capital}")

    # ---- Read-only views ----

    def get_state(self):
        """Return a snapshot of current engine state for the UI."""
        open_positions = [p for p in self.positions.values() if p.status == "open"]
        total_unrealized = sum(p.unrealized_pnl for p in open_positions)
        capital_at_risk = sum(p.entry_price * p.qty for p in open_positions)
        closed_positions = [p for p in self.positions.values() if p.status == "closed"]
        total_realized = sum(p.realized_pnl for p in closed_positions)

        return {
            "events": [
                {
                    "ticker": t,
                    "title": self._events[t].get("title", t),
                    "price": self._events[t].get("price", 0.42),
                    "pnl": self._events[t].get("pnl", 0.0),
                    "status": self._events[t].get("status", "unknown"),
                    "category": self._events[t].get("category"),
                    "sub_category": self._events[t].get("sub_category"),
                }
                for t in self._events
            ],
            "logs": list(self._logs),
            "positions": self.get_positions(),
            "capital": {
                "total": self._total_capital,  # starting capital (fixed)
                "used": self._used_capital,
                # available cash = starting capital - used + realized PnL
                "available": self._total_capital - self._used_capital + total_realized,
                "capital_at_risk": capital_at_risk,
                "unrealized_pnl": total_unrealized,
                "realized_pnl": total_realized,
            },

            "markets": {mt: vars(m) for mt, m in self.markets.items()},
        }


    def add_market_for_event(self, event_ticker: str, market_ticker: str):
        """
        Link a chosen market to an event. For now, just store it.
        Later we'll fetch market metadata and status.
        """
        market_ticker = market_ticker.upper()

        entry = MarketEntry(
            event_ticker=event_ticker.upper(),
            market_ticker=market_ticker,
        )

        self.markets[market_ticker] = entry
        self.log(f"[MARKET] Added market {market_ticker} for event {event_ticker}")

    async def update_market_metadata(self, market_ticker: str):
        """
        Fetch + update market metadata by calling platform_ops.get_event
        for the parent event, then locating this market inside it.
        This avoids relying on get_market(), which is failing with DNS errors.
        """
        market_ticker = market_ticker.upper()

        if market_ticker not in self.markets:
            self.log(f"[MARKET] Attempted metadata update for unknown market {market_ticker}")
            return

        market = self.markets[market_ticker]
        event_ticker = market.event_ticker

        try:
            resp = await platform_ops.get_event(event_ticker)
        except Exception as e:
            self.log(f"[MARKET] Error fetching event {event_ticker} for market {market_ticker}: {e}")
            return

        if not resp or not resp.get("ok"):
            self.log(
                f"[MARKET] Failed to fetch event {event_ticker} for market {market_ticker}: {resp!r}"
            )
            return

        data = resp.get("data") or {}

        # Event payload should have a list of markets somewhere
        raw_markets = data.get("markets") or data.get("markets_info") or []

        # Find this specific market inside the event's markets list
        target = None
        for m in raw_markets:
            mt = m.get("market_ticker") or m.get("ticker") or m.get("id")
            if isinstance(mt, str) and mt.upper() == market_ticker:
                target = m
                break

        if target is None:
            self.log(
                f"[MARKET] Could not find market {market_ticker} inside event {event_ticker}; "
                f"available keys={[m.get('market_ticker') or m.get('ticker') or m.get('id') for m in raw_markets]}"
            )
            return

        # 🔍 DEBUG: see where status actually lives
        self.log(
            f"[MARKET] DEBUG {market_ticker} "
            f"state={target.get('state')!r}, "
            f"status={target.get('status')!r}, "
            f"lifecycle_status={target.get('lifecycle_status')!r}"
        )

        # --- Status mapping ---
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

        market.status = status

        # --- Simple price extraction (best effort; tweak keys as needed) ---
        data_price_yes = (
            target.get("last_price_yes")
            or target.get("last_price")
            or target.get("yes_last_price")
        )
        best_bid_yes = target.get("yes_bid") or target.get("best_bid_yes")
        best_ask_yes = target.get("yes_ask") or target.get("best_ask_yes")

        price_yes = None

        if best_bid_yes is not None and best_ask_yes is not None:
            try:
                price_yes = (float(best_bid_yes) + float(best_ask_yes)) / 2.0
            except (TypeError, ValueError):
                price_yes = None

        if price_yes is None and data_price_yes is not None:
            try:
                price_yes = float(data_price_yes)
            except (TypeError, ValueError):
                price_yes = None

        # 🔧 Normalize from cents to 0–1 if needed
        if price_yes is not None and price_yes > 1.0:
            price_yes = price_yes / 100.0

        # Optional: clamp into [0,1] just to be safe
        if price_yes is not None:
            price_yes = max(0.0, min(1.0, price_yes))

        market.last_price_yes = price_yes
        market.last_price_no = 1.0 - price_yes if price_yes is not None else None

        self.log(
            f"[MARKET] Updated metadata for {market_ticker}: "
            f"raw_status={raw_status}, mapped_status={status}, "
            f"price_yes={market.last_price_yes}, price_no={market.last_price_no}"
        )

         # If the market has closed, auto-close any open positions in it
        if status == "closed":
            self._auto_close_positions_for_market(market_ticker)

        # --- Update positions for this market (mark-to-market for NO) ---
        if market.last_price_no is not None:
            for pos in self.positions.values():
                if pos.market_ticker == market_ticker and pos.status == "open":
                    pos.current_price = market.last_price_no
                    # Simple PnL: profit if current NO price < entry NO price
                    pos.unrealized_pnl = (pos.entry_price - pos.current_price) * pos.qty


    def maybe_open_no_position_for_market(self, market_ticker: str):
        """
        If the market is open and we don't already have a NO position
        for it, open a very simple paper NO position with a fixed qty
        and placeholder price.
        """
        market_ticker = market_ticker.upper()
        market = self.markets.get(market_ticker)

        if market is None:
            self.log(f"[TRADE] Cannot evaluate NO entry: unknown market {market_ticker}")
            return

        if market.status != "open":
            self.log(
                f"[TRADE] Market {market_ticker} not open yet "
                f"(status={market.status}); skipping NO entry."
            )
            return

        # Avoid duplicate NO positions for the same market
        
        for pos in self.positions.values():
            if pos.market_ticker == market_ticker and pos.side == "NO":
                self.log(
                    f"[TRADE] Already have a NO position in {market_ticker}; "
                    f"skipping new entry."
                )
                return


        # For now: fixed position size + dummy price
        qty = 1
        entry_price = market.last_price_no if market.last_price_no is not None else 0.5

        self.open_paper_position(
            market_ticker=market_ticker,
            side="NO",
            qty=qty,
            entry_price=entry_price,
        )

        self.log(
            f"[TRADE] (simple rule) Opened paper NO position in {market_ticker} "
            f"with qty={qty}, entry_price={entry_price}"
        )


    def open_paper_position(
        self,
        market_ticker: str,
        side: str,
        qty: int,
        entry_price: float,
    ):
        """
        Create a paper position in memory.
        No capital adjustments yet; that comes later.
        """
        market_ticker = market_ticker.upper()
        side = side.upper()

        if side not in ("YES", "NO"):
            self.log(f"[TRADE] Invalid side '{side}' for market {market_ticker}")
            return

        if qty <= 0:
            self.log(f"[TRADE] Invalid qty {qty} for market {market_ticker}")
            return

        # Try to infer the parent event from the tracked markets
        event_ticker = ""
        market_entry = self.markets.get(market_ticker)
        if market_entry is not None:
            event_ticker = market_entry.event_ticker

        position = Position(
            id=self._allocate_position_id(),
            event_ticker=event_ticker,
            market_ticker=market_ticker,
            side="NO",  # for now we only support NO in our flow
            qty=qty,
            entry_price=entry_price,
            current_price=entry_price,  # initial mark = entry
            status="open",
            entry_ts=datetime.utcnow(),
            realized_pnl=0.0,
            unrealized_pnl=0.0,
        )

        self.positions[position.id] = position

        self.log(
            f"[TRADE] Opened PAPER {position.side} position: "
            f"market={market_ticker}, qty={qty}, entry={entry_price}, "
            f"pos_id={position.id}"
        )

    def close_position(self, position_id: int):
        """
        Close an open paper position:
        - capture exit price (current mark)
        - compute realized PnL
        - update lifecycle fields
        - free capital (simple model)
        """
        pos = self.positions.get(position_id)
        if pos is None:
            self.log(f"[TRADE] Attempted close on unknown position id={position_id}")
            return

        if pos.status != "open":
            self.log(f"[TRADE] Position {position_id} already closed")
            return

        # Determine exit price (use latest current_price)
        exit_price = pos.current_price

        # Realized PnL for a NO position:
        # profit if price goes down
        pos.realized_pnl = (pos.entry_price - exit_price) * pos.qty
        pos.unrealized_pnl = 0.0  # no longer active

        # Mark lifecycle
        pos.status = "closed"
        pos.exit_ts = datetime.utcnow()

        # Free capital (simple model = entry_price * qty)
        freed_capital = pos.entry_price * pos.qty
        self._used_capital = max(0, self._used_capital - freed_capital)

        self.log(
            f"[TRADE] Closed position {position_id}: "
            f"exit_price={exit_price}, realized_pnl={pos.realized_pnl}"
        )


    # ---- Helpers / stubs ----

    def log(self, message: str):
        self._logs.append(message)

    def get_positions(self):
        """
        Return a JSON-serializable view of all positions.
        """
        return [vars(p) for p in self.positions.values()]

    def get_logs(self):
        return list(self._logs)
    
    def _allocate_position_id(self) -> int:
        pid = self._next_position_id
        self._next_position_id += 1
        return pid
    
    def _auto_close_positions_for_market(self, market_ticker: str):
        """
        Auto-close all open positions in a market when the market is closed/settled.
        Uses the latest NO price as the exit price if available.
        """
        market_ticker = market_ticker.upper()
        market = self.markets.get(market_ticker)
        if market is None:
            return

        exit_price = market.last_price_no

        for pos in self.positions.values():
            if pos.market_ticker == market_ticker and pos.status == "open":
                # If we have a fresh NO price, use it as the mark before closing
                if exit_price is not None:
                    pos.current_price = exit_price
                self.close_position(pos.id)
