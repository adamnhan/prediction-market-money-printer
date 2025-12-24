from typing import Dict, List, Any, Optional, Literal
from dataclasses import asdict
from src.event_service import fetch_event_metadata
from src import platform_ops
from datetime import datetime, timezone, timedelta
from trading_engine.strategy_config import StrategyConfig
from trading_engine.models import MarketEntry, Position, MarketUpdate  
from trading_engine.strategy_helpers import (
    can_open_new_position,
    pnl_exit_reason,
    time_exit_reason,
    strategy_exit_reason,
)
from src.ws_client import request_unsubscribe
from trading_engine.capital import CapitalState
from trading_engine import engine_state_store
from trading_engine.trade_ledger import record_trade_close
from trading_engine.trade_ledger import compute_circuit_breaker_stats
from trading_engine.market_utils import (
    map_status_from_target,
    extract_yes_price_from_target,
)






class TradingEngine:
    def __init__(self, strategy_config: Optional[StrategyConfig] = None,):
        # Events + logs
        self._events: Dict[str, Dict[str, Any]] = {}
        self._logs: List[str] = []

        # Operator flags (to be persisted)
        self.pause_entries: bool = False
        self.pause_all: bool = False

        # Capital tracking (private but consistent)
        self.capital = CapitalState(total=10000.0)

        # Circuit breaker state
        self._cooldown_until: datetime | None = None

        # Markets keyed by market_ticker
        self.markets: Dict[str, MarketEntry] = {}
        # Markets we already traded/closed and should not auto-reenter
        self._retired_markets: set[str] = set()

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

            # If removing the event detaches its markets, unsubscribe them
            to_remove = [
                mt for mt, m in list(self.markets.items()) if m.event_ticker == ticker
            ]
            for mt in to_remove:
                self.remove_market(mt)

    def set_total_capital(self, amount: float):
        self.capital.total = float(amount)
        self.log(f"Set total capital to {self.capital.total}")

    # ---- Read-only views ----

    def get_state(self):
        """Return a snapshot of current engine state for the UI."""
        positions_list = list(self.positions.values())
        capital = self.capital.compute_totals(positions_list)

        cb_stats = self._circuit_breaker_stats()

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
            "operator_flags": {
                "pause_entries": self.pause_entries,
                "pause_all": self.pause_all,
            },
            "circuit_breakers": {
                "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
                "today_realized_pnl": cb_stats.get("today_realized_pnl"),
                "today_trades": cb_stats.get("today_trades"),
                "max_drawdown": cb_stats.get("max_drawdown"),
                "limits": {
                    "daily_loss_limit": self.strategy_config.daily_loss_limit,
                    "max_drawdown": self.strategy_config.max_drawdown,
                    "max_trades_per_day": self.strategy_config.max_trades_per_day,
                    "cooldown_minutes_after_stop": self.strategy_config.cooldown_minutes_after_stop,
                },
            },
            "strategy_counters": {
                "auto_entries": self._auto_entries_count,
                "auto_exits": self._auto_exits_count,
                "skipped_entries_due_to_risk": self._skipped_entries_due_to_risk,
                "last_risk_skip_reason": getattr(self, "_last_risk_skip_reason", None),
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
        # Optional: if previously retired, keep it retired to avoid re-entry
        self._persist_control_state()

    def remove_market(self, market_ticker: str) -> None:
        market_ticker = market_ticker.upper()
        if market_ticker not in self.markets:
            return

        # Mark as retired to block future auto-entries
        self._retired_markets.add(market_ticker)

        # Don't unsubscribe if we still have open positions in the market
        has_open = any(
            (p.market_ticker or "").upper() == market_ticker and (p.status or "").lower() == "open"
            for p in self.positions.values()
        )
        if has_open:
            self.log(
                f"[MARKET] Skipping unsubscribe for {market_ticker}: open positions exist"
            )
            return

        self.markets.pop(market_ticker, None)
        self.log(f"[MARKET] Removed market {market_ticker}")
        request_unsubscribe(market_ticker)
        # Persist control state after market removal
        self._persist_control_state()

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
            self._maybe_trigger_auto_no_entry(market_ticker, source="REST")

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


    def auto_open_no_for_market(self, market_ticker: str, source: str = "WS") -> None:
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
                    f"AUTO_ENTRY_NO_{source}: pos_id={pid} market={market_ticker}"
                )

    def maybe_open_no_position_for_market(self, market_ticker: str):
        """
        If the market is open and we don't already have a NO position
        for it, open a very simple paper NO position with a fixed qty
        and placeholder price.
        """
        market_ticker = market_ticker.upper()
        market = self.markets.get(market_ticker)

        # Global operator gates
        if getattr(self, "pause_all", False):
            self.log(f"[TRADE] Global pause_all active; skipping entry for {market_ticker}")
            return
        if getattr(self, "pause_entries", False):
            self.log(f"[TRADE] Entries paused; skipping entry for {market_ticker}")
            return

        if market is None:
            self.log(f"[TRADE] Cannot evaluate NO entry: unknown market {market_ticker}")
            return

        # Do not re-enter a market we've retired after closing
        if market_ticker in getattr(self, "_retired_markets", set()):
            self.log(f"[TRADE] Market {market_ticker} retired; skipping re-entry.")
            return

        # Do not re-enter a market once we've traded it and detached it
        if market.status == "closed":
            self.log(f"[TRADE] Market {market_ticker} marked closed; skipping re-entry.")
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
        allowed, risk_reason = can_open_new_position(
            strategy_config=self.strategy_config,
            positions=list(self.positions.values()),
            market_ticker=market_ticker,
            qty=qty,
            entry_price=entry_price,
            total_capital=self.capital.total,
        )

        if not allowed:

            self._skipped_entries_due_to_risk += 1
            self._last_risk_skip_reason = risk_reason
            self.log(
                f"ENTRY_SKIPPED_RISK: market={market_ticker} qty={qty} price={entry_price} reason={risk_reason}"
            )
            return 

        # Circuit breakers: enforce operator safety limits
        cb_allowed, cb_reason = self._circuit_breaker_allows_entry()
        if not cb_allowed:
            self._skipped_entries_due_to_risk += 1
            self._last_risk_skip_reason = cb_reason
            self.log(
                f"ENTRY_SKIPPED_CIRCUIT: market={market_ticker} qty={qty} price={entry_price} reason={cb_reason}"
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

    def _maybe_trigger_auto_no_entry(
        self, market_ticker: str, source: str = "WS", prev_price_no: float | None = None
    ) -> None:
        """
        Centralized hook for auto NO entries (WS or REST).
        Ensures market is open, price is present, and optional price-change gating.
        """
        market_ticker = market_ticker.upper()
        market = self.markets.get(market_ticker)
        if market is None:
            return

        if market.status != "open":
            return

        if market.last_price_no is None:
            return

        if prev_price_no is not None and market.last_price_no == prev_price_no:
            self.log(
                f"[AUTO_ENTRY_SKIP_{source}] market={market_ticker} price_no={market.last_price_no} reason=same_price"
            )
            return

        self.log(
            f"[AUTO_ENTRY_CHECK_{source}] market={market_ticker} price_no={market.last_price_no}"
        )
        self.auto_open_no_for_market(market_ticker, source=source)


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

        # Persist control/trading state best-effort for crash resilience
        self._persist_control_state()

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

        # Persist closed trade to ledger; don't let ledger errors break trading flow.
        try:
            record_trade_close(
                position=pos,
                exit_reason=reason or "exit",
                strategy_snapshot=asdict(self.strategy_config) if self.strategy_config else None,
            )
        except Exception as e:
            self.log(f"[TRADE_LEDGER] failed to record trade {position_id}: {e}")

        # If stop-loss triggered, start cooldown (if configured)
        if reason and "stop_loss" in str(reason) and self.strategy_config.cooldown_minutes_after_stop:
            try:
                minutes = int(self.strategy_config.cooldown_minutes_after_stop)
                self._cooldown_until = datetime.utcnow() + timedelta(minutes=minutes)
                self.log(f"[COOLDOWN] stop_loss triggered; cooling until {self._cooldown_until.isoformat()}")
            except Exception as e:
                self.log(f"[COOLDOWN] failed to start cooldown: {e}")

        # If this was the last open position in the market, detach/unsubscribe
        self._detach_market_if_idle(pos.market_ticker)

        # Persist control/trading state best-effort for crash resilience
        self._persist_control_state()

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
        prev_price_no = market.last_price_no if market is not None else None

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

        # If we see live prices but no status yet, assume open for WS-driven flow
        if market.status not in ("open", "closed") and (
            market.last_price_no is not None or market.last_price_yes is not None
        ):
            market.status = "open"
            self.log(f"[MARKET_STATUS_WS] market={market_ticker} status=open (via live update)")

        # Evaluate NO entries using live WS price when the market is open
        self._maybe_trigger_auto_no_entry(
            market_ticker, source="WS", prev_price_no=prev_price_no
        )

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

        if getattr(self, "pause_all", False):
            self.log(f"[AUTO_EXIT_SKIP] pause_all active; skipping WS auto exits for {market_ticker}")
            return

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
            self._auto_exits_count += 1
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

    def close_all_positions(self) -> int:
        """
        Close all open positions (paper). Returns count closed.
        """
        closed = 0
        # Use list copy to avoid mutation during iteration.
        for pos_id, pos in list(self.positions.items()):
            if getattr(pos, "status", "").lower() == "open":
                self.close_position(pos_id, reason="operator_close_all")
                closed += 1
        # Persist after bulk close
        self._persist_control_state()
        return closed

    def set_operator_flags(
        self, pause_entries: Optional[bool] = None, pause_all: Optional[bool] = None
    ) -> dict[str, bool]:
        """
        Update operator pause flags and persist if anything changed.
        """
        changed = False
        if pause_entries is not None and bool(pause_entries) != self.pause_entries:
            self.pause_entries = bool(pause_entries)
            changed = True
        if pause_all is not None and bool(pause_all) != self.pause_all:
            self.pause_all = bool(pause_all)
            changed = True

        if changed:
            self.log(
                f"[OPERATOR_FLAGS] pause_entries={self.pause_entries} pause_all={self.pause_all}"
            )
            self._persist_control_state()

        return {"pause_entries": self.pause_entries, "pause_all": self.pause_all}

    def _circuit_breaker_stats(self) -> Dict[str, Any]:
        try:
            return compute_circuit_breaker_stats()
        except Exception as e:
            self.log(f"[CIRCUIT_BREAKER] stats error: {e}")
            return {"today_realized_pnl": 0.0, "today_trades": 0, "max_drawdown": 0.0}

    # ---- Persistence helpers ----

    def get_control_state_snapshot(self) -> Dict[str, Any]:
        """
        Minimal control snapshot for persistence/resume (no sensitive data).
        """
        return {
            "attached_markets": list(self.markets.keys()),
            "retired_markets": sorted(self._retired_markets),
            "operator_flags": {
                "pause_entries": self.pause_entries,
                "pause_all": self.pause_all,
            },
            "strategy_config": asdict(self.strategy_config) if self.strategy_config else None,
            "positions": self.get_positions(),
            "capital": {"total": self.capital.total, "used": self.capital.used},
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
        }

    def apply_control_state_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Restore minimal control state. Does not open positions or resubscribe yet.
        """
        if not isinstance(snapshot, dict):
            return

        # Restore operator flags
        flags = snapshot.get("operator_flags", {})
        if isinstance(flags, dict):
            self.pause_entries = bool(flags.get("pause_entries", self.pause_entries))
            self.pause_all = bool(flags.get("pause_all", self.pause_all))

        # Restore retired markets
        retired = snapshot.get("retired_markets")
        if isinstance(retired, (list, tuple, set)):
            self._retired_markets = {str(m).upper() for m in retired if m}

        # Restore attached markets as stubs (actual metadata/WS handled elsewhere)
        attached = snapshot.get("attached_markets")
        if isinstance(attached, (list, tuple, set)):
            for m in attached:
                if not m:
                    continue
                ticker = str(m).upper()
                if ticker not in self.markets:
                    self.markets[ticker] = MarketEntry(
                        event_ticker="unknown", market_ticker=ticker
                    )

        # Optional strategy config restoration (best-effort)
        cfg = snapshot.get("strategy_config")
        if isinstance(cfg, dict):
            try:
                self.strategy_config = StrategyConfig(**cfg)
            except Exception:
                # Leave existing config untouched on failure
                pass

        # Restore cooldown
        cooldown_val = snapshot.get("cooldown_until")
        if isinstance(cooldown_val, str):
            try:
                self._cooldown_until = datetime.fromisoformat(cooldown_val)
            except Exception:
                self._cooldown_until = None
        else:
            self._cooldown_until = None

        # Restore positions (paper). Best-effort; ignore malformed entries.
        restored_positions = {}
        max_pid = 0
        positions = snapshot.get("positions")
        if isinstance(positions, list):
            for raw in positions:
                if not isinstance(raw, dict):
                    continue
                try:
                    pid = int(raw.get("id") or 0)
                except Exception:
                    pid = 0
                status = raw.get("status") or "open"
                qty = raw.get("qty") or 0
                entry_price = raw.get("entry_price") or 0.0
                current_price = raw.get("current_price")
                if current_price is None:
                    current_price = entry_price
                try:
                    qty_f = float(qty)
                    entry_f = float(entry_price)
                    current_f = float(current_price)
                    computed_unrealized = (
                        (entry_f - current_f) * qty_f if status.lower() == "open" else 0.0
                    )
                except Exception:
                    computed_unrealized = 0.0
                    current_price = entry_price

                pos = Position(
                    id=pid,
                    event_ticker=(raw.get("event_ticker") or "").upper(),
                    market_ticker=(raw.get("market_ticker") or "").upper(),
                    side=(raw.get("side") or "NO").upper(),
                    qty=qty,
                    entry_price=entry_price,
                    current_price=current_price,
                    status=status,
                    entry_ts=raw.get("entry_ts"),
                    exit_ts=raw.get("exit_ts"),
                    realized_pnl=raw.get("realized_pnl") or 0.0,
                    unrealized_pnl=raw.get("unrealized_pnl") or computed_unrealized,
                )
                if pid > 0:
                    max_pid = max(max_pid, pid)
                restored_positions[pos.id] = pos

        if restored_positions:
            self.positions = restored_positions
            self._next_position_id = max_pid + 1

        # Restore capital totals/usage (best-effort)
        cap = snapshot.get("capital")
        if isinstance(cap, dict):
            try:
                self.capital.total = float(cap.get("total", self.capital.total))
            except Exception:
                pass
        # Recompute used capital from open positions to ensure consistency
        try:
            self.capital.used = sum(
                (p.entry_price or 0.0) * (p.qty or 0)
                for p in self.positions.values()
                if (p.status or "").lower() == "open"
            )
        except Exception:
            # leave as-is on error
            pass
    
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

        # After bulk auto-closes, detach market if no open positions remain
        self._detach_market_if_idle(market_ticker)

    def _detach_market_if_idle(self, market_ticker: str) -> None:
        """
        Remove a market + unsubscribe when no open positions remain.
        Avoids re-entry on the same market without explicit re-attach.
        """
        market_ticker = (market_ticker or "").upper()
        if not market_ticker:
            return

        has_open = any(
            (p.market_ticker or "").upper() == market_ticker
            and (p.status or "").lower() == "open"
            for p in self.positions.values()
        )
        if has_open:
            return

        # Mark as retired to block future auto-entries
        self._retired_markets.add(market_ticker)
        self.remove_market(market_ticker)

    def on_market_status(self, market_ticker: str, raw_status: str | None) -> None:
        """
        Handle lifecycle updates from WS (preferred over REST polling).
        Maps the raw status to open/closed where possible and triggers auto-closes.
        """
        ticker = (market_ticker or "").upper()
        if not ticker:
            return

        raw = (raw_status or "").lower()
        mapped = raw

        # Map Kalshi lifecycle event_types and statuses to open/closed
        if raw in ("open", "trading", "live", "activated", "created"):
            mapped = "open"
        elif raw in (
            "closed",
            "settled",
            "resolved",
            "halted",
            "deactivated",
            "determined",
        ):
            mapped = "closed"
        elif not raw:
            mapped = "unknown"

        market = self.markets.get(ticker)
        if market is None:
            market = MarketEntry(event_ticker="unknown", market_ticker=ticker, status=mapped)
            self.markets[ticker] = market
        else:
            prev = market.status
            if prev != mapped:
                self.log(
                    f"[MARKET_STATUS_WS] market={ticker} status={mapped} raw={raw_status} prev={prev}"
                )
            market.status = mapped

        if mapped == "closed":
            self._auto_close_positions_for_market(ticker)
            # If no positions remain, detach and unsubscribe to keep streams clean
            self._detach_market_if_idle(ticker)
            # Persist control state when lifecycle forces detach/retire
            self._persist_control_state()

    def _circuit_breaker_allows_entry(self) -> tuple[bool, str]:
        """
        Enforce circuit breakers using ledger-derived stats and cooldowns.
        """
        cfg = self.strategy_config

        # Cooldown after a stop-loss
        if cfg.cooldown_minutes_after_stop and self._cooldown_until:
            if datetime.utcnow() < self._cooldown_until:
                return False, "cooldown_active"
            # cooldown expired
            self._cooldown_until = None

        stats = compute_circuit_breaker_stats()

        daily_limit = cfg.daily_loss_limit
        if daily_limit is not None and stats["today_realized_pnl"] <= -abs(daily_limit):
            return False, "daily_loss_cap"

        dd_cap = cfg.max_drawdown
        if dd_cap is not None and stats["max_drawdown"] >= abs(dd_cap):
            return False, "max_drawdown_cap"

        max_trades = cfg.max_trades_per_day
        if max_trades is not None and stats["today_trades"] >= max_trades:
            return False, "max_trades_per_day"

        return True, "ok"

    def detach_all_markets(self, force: bool = False) -> int:
        """
        Detach/unsubscribe all tracked markets.
        If force=True, unsubscribe even if open positions exist.
        Returns count of markets removed.
        """
        count = 0
        for mt in list(self.markets.keys()):
            ticker = mt.upper()
            if force:
                # Mark retired and drop without the open-position guard.
                self._retired_markets.add(ticker)
                self.markets.pop(ticker, None)
                request_unsubscribe(ticker)
                count += 1
            else:
                before = len(self.markets)
                self.remove_market(ticker)
                if len(self.markets) < before:
                    count += 1
        # Persist updated control state after bulk detach
        self._persist_control_state()
        return count

    def _persist_control_state(self) -> None:
        """
        Best-effort persistence for control/trading state.
        """
        try:
            engine_state_store.save_state(self.get_control_state_snapshot())
        except Exception as e:
            self.log(f"[PERSIST] failed to save state: {e}")
