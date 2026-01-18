"""Websocket consumer placeholder."""

from __future__ import annotations

from collections.abc import Iterable

from cross_venue_arb.books.book import OrderBook


def stream_books() -> Iterable[OrderBook]:
    return []
