"""Order book representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class OrderBook:
    venue: str
    market_id: str
    bids: Dict[float, float] = field(default_factory=dict)
    asks: Dict[float, float] = field(default_factory=dict)
    best_bid: tuple[float, float] | None = None
    best_ask: tuple[float, float] | None = None
    last_update_ts: float | None = None
    last_exchange_ts: float | None = None
    has_snapshot: bool = False
    raw_no_bids: Dict[float, float] | None = None

    def apply_snapshot(
        self,
        bids: Dict[float, float],
        asks: Dict[float, float],
        exchange_ts: float | None,
        recv_ts: float,
        *,
        raw_no_bids: Dict[float, float] | None = None,
    ) -> None:
        self.bids = dict(bids)
        self.asks = dict(asks)
        self.raw_no_bids = dict(raw_no_bids) if raw_no_bids is not None else None
        self.has_snapshot = True
        self._finalize_update(exchange_ts, recv_ts)

    def apply_delta(
        self,
        changes: list[tuple[str, float, float]],
        exchange_ts: float | None,
        recv_ts: float,
        *,
        raw_no_changes: list[tuple[float, float]] | None = None,
    ) -> None:
        for side, price, size in changes:
            if side == "bid":
                _apply_level(self.bids, price, size)
            elif side == "ask":
                _apply_level(self.asks, price, size)
        if raw_no_changes is not None:
            if self.raw_no_bids is None:
                self.raw_no_bids = {}
            for price, size in raw_no_changes:
                _apply_level(self.raw_no_bids, price, size)
        self._finalize_update(exchange_ts, recv_ts)

    def is_healthy(self, max_age_s: float = 30.0, *, now_ts: float | None = None) -> bool:
        if not self.has_snapshot:
            return False
        if self.best_bid is None or self.best_ask is None:
            return False
        if self.last_update_ts is None:
            return False
        if now_ts is None:
            return True
        return (now_ts - self.last_update_ts) <= max_age_s

    def _finalize_update(self, exchange_ts: float | None, recv_ts: float) -> None:
        self.last_exchange_ts = exchange_ts
        self.last_update_ts = recv_ts
        self.best_bid = _best_level(self.bids, best="bid")
        self.best_ask = _best_level(self.asks, best="ask")

    def best_ask_size(self) -> float | None:
        if self.best_ask is None:
            return None
        return float(self.best_ask[1])

    def best_bid_size(self) -> float | None:
        if self.best_bid is None:
            return None
        return float(self.best_bid[1])


def _apply_level(book: Dict[float, float], price: float, size: float) -> None:
    if size <= 0:
        book.pop(price, None)
    else:
        book[price] = size


def _best_level(book: Dict[float, float], *, best: str) -> tuple[float, float] | None:
    if not book:
        return None
    if best == "bid":
        price = max(book.keys())
    else:
        price = min(book.keys())
    return (price, book[price])
