from typing import Dict, List, Any, Optional, Literal
from src.event_service import fetch_event_metadata
from dataclasses import asdict
from src import platform_ops
from datetime import datetime, timezone
from trading_engine.strategy_config import StrategyConfig
from trading_engine.models import MarketEntry, Position, MarketUpdate  
from trading_engine.strategy_helpers import (
    can_open_new_position,
    pnl_exit_reason,
    time_exit_reason,
    strategy_exit_reason,
)
from trading_engine.capital import CapitalState
from trading_engine.market_utils import (
    map_status_from_target,
    extract_yes_price_from_target,
)






class TradingEngine:
    def __init__(self, strategy_config: Optional[StrategyConfig] = None,):
        # Events + logs
        self._events: Dict[str, Dict[str, Any]] = {}
        self._logs: List[str] = []

        # Capital tracking (private but consistent)
        self.capital = CapitalState(total=10000.0)

        # Markets keyed by market_ticker
        self.markets: Dict[str, MarketEntry] = {}

        # Position storage + ID generator
        self._next_position_id = 1
        self.positions: Dict[int, Position] = {}  # position_id -> Position
        self.strategy_config: StrategyConfig = strategy_config or StrategyConfig()

        # NEW: strategy-related counters (Phase 4 – not used yet)
        self._auto_entries_count = 0
        self._auto_exits_count = 0
        self._skipped_entries_due_to_risk = 0

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
        self.capital.total = float(amount)
        self.log(f"Set total capital to {self.capital.total}")

    # ---- Read-only views ----

    def get_state(self):
        """Return a snapshot of current engine state for the UI."""
        positions_list = list(self.positions.values())
        capital = self.capital.compute_totals(positions_list)

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
            "capital": capital,
            "strategy_config": asdict(self.strategy_config),
            "strategy_counters": {
                "auto_entries": self._auto_entries_count,
                "auto_exits": self._auto_exits_count,
                "skipped_entries_due_to_risk": self._skipped_entries_due_to_risk,
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
        raw_status, status = map_status_from_target(target)
        market.status = status

        # --- Simple price extraction (best effort; tweak keys as needed) ---
        price_yes = extract_yes_price_from_target(target)

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

        # NEW: if the market is open, let the strategy try an auto NO entry
        if status == "open":
            self.auto_open_no_for_market(market_ticker)

        # --- Update positions for this market (mark-to-market for NO) ---

        if market.last_price_no is not None:
            # Use list(...) in case positions change while iterating
            for pos in list(self.positions.values()):
                if pos.market_ticker == market_ticker and pos.status == "open":
                    pos.current_price = market.last_price_no
                    # Simple PnL: profit if current NO price < entry NO price
                    pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.qty

                    # Unified strategy-driven exit check (PnL + time)
                    exit_reason = strategy_exit_reason(pos, self.strategy_config)

                    # Only act if we still have an open position and a reason
                    if exit_reason is not None and pos.status == "open":
                        # PnL-based reasons come back as "pnl_take_profit" / "pnl_stop_loss"
                        if exit_reason.startswith("pnl_"):
                            pnl_reason = exit_reason.replace("pnl_", "")

                            # Keep existing PnL log style
                            self.log(
                                f"PNL_EXIT_SIGNAL: pos_id={pos.id} "
                                f"market={market_ticker} reason={pnl_reason} "
                                f"unrealized={pos.unrealized_pnl}"
                            )

                            self._auto_exits_count += 1
                            self.log(
                                f"AUTO_EXIT_PNL: pos_id={pos.id} "
                                f"market={market_ticker} reason={pnl_reason} "
                                f"unrealized={pos.unrealized_pnl}"
                            )
                            self.close_position(pos.id, reason=exit_reason)

                        # Time-based exits: "time_expired"
                        elif exit_reason == "time_expired":
                            age = (
                                datetime.utcnow() - pos.entry_ts
                            ).total_seconds() if pos.entry_ts else None

                            self.log(
                                f"TIME_EXIT_SIGNAL: pos_id={pos.id} "
                                f"market={market_ticker} reason={exit_reason} "
                                f"age={age}"
                            )

                            self._auto_exits_count += 1
                            self.log(
                                f"AUTO_EXIT_TIME: pos_id={pos.id} "
                                f"market={market_ticker} reason={exit_reason} "
                                f"age={age}"
                            )
                            self.close_position(pos.id, reason=exit_reason)


    def auto_open_no_for_market(self, market_ticker: str) -> None:
        """
        Strategy-driven wrapper around maybe_open_no_position_for_market.

        If this call results in one or more new open NO positions for the
        given market, it increments the auto_entries counter and logs them.

        Phase 4 note: this is not yet called from the update loop. It only
        becomes active once wired into the automation flow.
        """
        market_ticker = market_ticker.upper()

        # Snapshot open position IDs for this market before
        before_ids = {
            p.id
            for p in self.positions.values()
            if p.market_ticker == market_ticker and p.status == "open"
        }

        # Reuse existing logic (risk checks, market status, duplicates, etc.)
        self.maybe_open_no_position_for_market(market_ticker)

        # Snapshot after calling the manual entry function
        after_ids = {
            p.id
            for p in self.positions.values()
            if p.market_ticker == market_ticker and p.status == "open"
        }

        new_ids = after_ids - before_ids
        if new_ids:
            for pid in new_ids:
                self._auto_entries_count += 1
                self.log(
                    f"AUTO_ENTRY_NO: pos_id={pid} market={market_ticker}"
                )

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
            if pos.market_ticker == market_ticker and (pos.side or "").upper() == "NO" and (pos.status or "").lower() == "open":
                self.log(
                    f"[TRADE] Already have a NO position in {market_ticker}; "
                    f"skipping new entry."
                )
                return


        # For now: fixed position size + dummy price
        qty = 1

        # Require a sane NO price in dollars before entering.
        entry_price = market.last_price_no
        if entry_price is None:
            self.log(f"[TRADE] No live NO price for {market_ticker}; skipping entry.")
            return

        entry_price = float(entry_price)

        # Kalshi prices should be in (0, 1]. If outside, don't trade.
        if entry_price <= 0.0 or entry_price > 1.0:
            self.log(
                f"[TRADE] Invalid NO price for entry: market={market_ticker} price_no={entry_price}; skipping entry."
            )
            return



        # --- NEW: strategy risk checks (Phase 4: log-only, no enforcement yet) ---
        if not can_open_new_position(
            strategy_config=self.strategy_config,
            positions=list(self.positions.values()),
            market_ticker=market_ticker,
            qty=qty,
            entry_price=entry_price,
            total_capital=self.capital.total,
        ):

            self._skipped_entries_due_to_risk += 1
            self.log(
                f"ENTRY_SKIPPED_RISK: market={market_ticker} qty={qty} price={entry_price}"
            )
            return 
        
                # Optional: only enter if NO price is cheap enough
        cfg = self.strategy_config
        if cfg.max_no_entry_price is not None and entry_price is not None:
            if entry_price > cfg.max_no_entry_price:
                self.log(
                    f"ENTRY_SKIPPED_PRICE: market={market_ticker} "
                    f"price_no={entry_price} max_no_entry_price={cfg.max_no_entry_price}"
                )
                self._skipped_entries_due_to_risk += 1
                return


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

        self.capital.reserve(entry_price, qty)

        self.log(
            f"[TRADE] Opened PAPER {position.side} position: "
            f"market={market_ticker}, qty={qty}, entry={entry_price}, "
            f"pos_id={position.id}"
        )

    def close_position(self, position_id: int, reason: Optional[str] = None):
        """
        Close an open paper position:
        - capture exit price (current mark)
        - compute realized PnL
        - update lifecycle fields
        - free capital (simple model)
        """
        pos = self.positions.get(position_id)

        self.log(
            f"[TRADE] close_position called for id={position_id}, "
            f"found={pos is not None}, status={getattr(pos, 'status', None)}"
        )

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
        pos.realized_pnl = (exit_price - pos.entry_price) * pos.qty
        pos.unrealized_pnl = 0.0  # no longer active

        # Mark lifecycle
        pos.status = "closed"
        pos.exit_ts = datetime.utcnow()

        # Free capital (simple model = entry_price * qty)
        self.capital.release(pos.entry_price, pos.qty)

        self.log(
            f"[TRADE] Closed position {position_id}: "
            f"exit_price={exit_price}, realized_pnl={pos.realized_pnl}"
        )

    def on_market_update(self, update: MarketUpdate) -> None:
        """
        Receive a normalized live market update from WebSocket.

        Responsibilities:
        1) Log the update (compact)
        2) Update MarketEntry.last_price_yes/last_price_no/last_update_ts
        3) Mark-to-market open NO positions using last_price_no (update current_price + unrealized_pnl)
        """

        # --- 1) Log ---
        if len(self._logs) > 5000:
            self._logs = self._logs[-2500:]

        # --- 2) Update internal market state ---
        market_ticker = (update.market_ticker or "").upper()
        if not market_ticker:
            return

        market = self.markets.get(market_ticker)

        # WS may arrive before REST attach/load; create a stub entry if needed
        if market is None:
            market = MarketEntry(
                event_ticker="unknown",  # backfill later when REST knows it
                market_ticker=market_ticker,
            )
            self.markets[market_ticker] = market

        # best_yes/best_no should be produced by ws_client book logic
        # WS sends prices in cents (0-100). Store dollars everywhere.
        if update.best_yes is not None:
            market.last_price_yes = float(update.best_yes) / 100.0

        if update.best_no is not None:
            market.last_price_no = float(update.best_no) / 100.0


        if update.ts is not None:
            market.last_update_ts = update.ts

        # --- 3) Mark-to-market open NO positions (your strategy) ---
        # If we don't have a NO price, we can't MTM.
        if market.last_price_no is None:
            return

        live_no = float(market.last_price_no)

        for pos in self.positions.values():
            pos_mt = (getattr(pos, "market_ticker", "") or "").upper()
            if pos_mt != market_ticker:
                continue

            status = (getattr(pos, "status", "") or "").lower()
            if status != "open":
                continue

            side = (getattr(pos, "side", "") or "").lower()
            if side != "no":
                continue

            pos.current_price = live_no

            qty = float(getattr(pos, "qty", 1) or 1)
            entry = float(getattr(pos, "entry_price", 0) or 0)

            # Unrealized PnL in price units
            pos.unrealized_pnl = (entry - pos.current_price) * qty

            
            if len(self._logs) > 5000:
                self._logs = self._logs[-2500:]
            
        # --- 4) WS-driven auto-exits (only for this market) ---
        now = datetime.now(timezone.utc)

        positions_to_close: list[tuple[int, str]] = []

        for pos in self.positions.values():
            pos_mt = (getattr(pos, "market_ticker", "") or "").upper()
            if pos_mt != market_ticker:
                continue

            status = (getattr(pos, "status", "") or "").lower()
            if status != "open":
                continue

            side = (getattr(pos, "side", "") or "").lower()
            if side != "no":
                continue

            # Guard: don't evaluate exits until we have the basics
            if getattr(pos, "entry_price", None) is None:
                continue
            if getattr(pos, "current_price", None) is None:
                continue
            entry_ts = getattr(pos, "entry_ts", None)
            if entry_ts is None:
                continue

            # Guard: don't auto-exit in the same instant as entry (protect against bad timestamps)
            try:
                # entry_ts might be a datetime or a string in your model; handle datetime only here
                if hasattr(entry_ts, "tzinfo"):
                    # normalize naive -> UTC
                    if entry_ts.tzinfo is None:
                        entry_ts = entry_ts.replace(tzinfo=timezone.utc)
                    if (now - entry_ts).total_seconds() < 2:
                        continue
            except Exception:
                # if timestamp math fails, don't auto-exit
                continue

            # Call your existing exit logic
            decision = None
            try:
                decision = strategy_exit_reason(pos, self.strategy_config, now)
            except TypeError:
                decision = strategy_exit_reason(pos, self.strategy_config)

            # Normalize decision into (should_exit, reason)
            should_exit = False
            reason = None

            if decision is None or decision is False:
                should_exit = False
            elif isinstance(decision, tuple) and len(decision) >= 1 and isinstance(decision[0], bool):
                should_exit = decision[0]
                reason = str(decision[1]) if len(decision) > 1 else "exit"
            elif isinstance(decision, dict) and "exit" in decision:
                should_exit = bool(decision.get("exit"))
                reason = str(decision.get("reason") or decision.get("exit_reason") or "exit")
            elif isinstance(decision, str):
                # treat common "no exit" strings as no-exit
                if decision.strip().lower() in ("", "none", "no_exit", "hold", "keep", "continue"):
                    should_exit = False
                else:
                    should_exit = True
                    reason = decision
            else:
                # last resort: don't accidentally close on truthy junk
                should_exit = False

            if should_exit:
                positions_to_close.append((pos.id, reason or "exit"))

        for position_id, reason in positions_to_close:
            self._logs.append(f"[AUTO_EXIT_WS] pos={position_id} market={market_ticker} reason={reason}")
            self.close_position(position_id)



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
    
    def _auto_close_positions_for_market(self, market_ticker: str) -> None:
        """
        Auto-close all open positions in a market when the market is closed/settled.
        Uses the latest NO price as the exit price if available.
        """
        market_ticker = market_ticker.upper()
        market = self.markets.get(market_ticker)
        if market is None:
            return

        exit_price = market.last_price_no

        # Use items() so we have both pos_id and pos for logging/closing
        for pos_id, pos in list(self.positions.items()):
            if pos.market_ticker == market_ticker and pos.status == "open":
                # If we have a fresh NO price, use it as the mark before closing
                if exit_price is not None:
                    pos.current_price = exit_price

                # NEW: track and log auto exits
                self._auto_exits_count += 1
                self.log(
                    f"AUTO_EXIT: pos_id={pos_id} market={market_ticker} reason=market_closed"
                )

                self.close_position(pos_id, reason="market_closed")
