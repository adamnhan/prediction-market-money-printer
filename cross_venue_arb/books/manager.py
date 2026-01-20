"""Book manager and health checks."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Iterable

from cross_venue_arb.books.book import OrderBook


class BookManager:
    def __init__(self, stale_after_s: float = 30.0) -> None:
        self._books: dict[str, Dict[str, OrderBook]] = defaultdict(dict)
        self._stale_after_s = stale_after_s

    def get_book(self, venue: str, market_id: str) -> OrderBook | None:
        return self._books.get(venue, {}).get(market_id)

    def ensure_book(self, venue: str, market_id: str) -> OrderBook:
        book = self._books[venue].get(market_id)
        if book is None:
            book = OrderBook(venue=venue, market_id=market_id)
            self._books[venue][market_id] = book
        return book

    def snapshot(
        self,
        venue: str,
        market_id: str,
        bids: dict[float, float],
        asks: dict[float, float],
        exchange_ts: float | None = None,
        *,
        raw_no_bids: dict[float, float] | None = None,
    ) -> None:
        recv_ts = time.monotonic()
        book = self.ensure_book(venue, market_id)
        book.apply_snapshot(bids, asks, exchange_ts, recv_ts, raw_no_bids=raw_no_bids)

    def delta(
        self,
        venue: str,
        market_id: str,
        changes: list[tuple[str, float, float]],
        exchange_ts: float | None = None,
        *,
        raw_no_changes: list[tuple[float, float]] | None = None,
    ) -> None:
        recv_ts = time.monotonic()
        book = self.ensure_book(venue, market_id)
        if not book.has_snapshot:
            return
        book.apply_delta(changes, exchange_ts, recv_ts, raw_no_changes=raw_no_changes)

    def is_healthy(self, venue: str, market_id: str) -> bool:
        book = self.get_book(venue, market_id)
        if book is None or book.last_update_ts is None:
            return False
        age = time.monotonic() - book.last_update_ts
        if age > self._stale_after_s:
            return False
        return book.has_snapshot and book.best_bid is not None and book.best_ask is not None

    def last_update_age(self, venue: str, market_id: str) -> float | None:
        book = self.get_book(venue, market_id)
        if book is None or book.last_update_ts is None:
            return None
        return time.monotonic() - book.last_update_ts

    def latest_update_ts(self, venue: str, market_ids: Iterable[str]) -> float | None:
        latest: float | None = None
        for market_id in market_ids:
            book = self.get_book(venue, market_id)
            if book is None or book.last_update_ts is None:
                continue
            latest = book.last_update_ts if latest is None else max(latest, book.last_update_ts)
        return latest

    def venues(self) -> list[str]:
        return list(self._books.keys())

    def markets(self, venue: str) -> list[str]:
        return list(self._books.get(venue, {}).keys())
