from __future__ import annotations

import math
import time
from typing import Any

from clob_client import ClobWrapper
from storage import Phase4Store


SKIP_SPREAD_TOO_WIDE = "SKIP_SPREAD_TOO_WIDE"
SKIP_INSUFFICIENT_DEPTH = "SKIP_INSUFFICIENT_DEPTH"
FAILED_NO_TOKEN_ID = "FAILED_NO_TOKEN_ID"
FAILED_BOOK_UNAVAILABLE = "FAILED_BOOK_UNAVAILABLE"
FAILED_ORDER_PLACE = "FAILED_ORDER_PLACE"


def _extract_token_id(intent: dict[str, Any]) -> str | None:
    token_map = intent.get("token_ids") or {}
    outcome = intent.get("outcome")
    if not token_map or not outcome:
        return None
    # Try exact, case-insensitive, and common YES/NO aliases.
    if outcome in token_map:
        return str(token_map[outcome])
    for key, value in token_map.items():
        if str(key).lower() == str(outcome).lower():
            return str(value)
    aliases = {"YES": "Yes", "NO": "No"}
    if str(outcome).upper() in aliases and aliases[str(outcome).upper()] in token_map:
        return str(token_map[aliases[str(outcome).upper()]])
    return None


def _normalize_book(book: Any) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    bids = []
    asks = []

    if isinstance(book, dict):
        raw_bids = book.get("bids") or []
        raw_asks = book.get("asks") or []
    else:
        raw_bids = getattr(book, "bids", []) or []
        raw_asks = getattr(book, "asks", []) or []

    for entry in raw_bids:
        price = entry.get("price") if isinstance(entry, dict) else entry[0]
        size = entry.get("size") if isinstance(entry, dict) else entry[1]
        bids.append((float(price), float(size)))
    for entry in raw_asks:
        price = entry.get("price") if isinstance(entry, dict) else entry[0]
        size = entry.get("size") if isinstance(entry, dict) else entry[1]
        asks.append((float(price), float(size)))

    bids.sort(key=lambda x: x[0], reverse=True)
    asks.sort(key=lambda x: x[0])
    return bids, asks


def _has_depth_for_buy(asks: list[tuple[float, float]], limit_price: float, qty: float) -> bool:
    remaining = qty
    for price, size in asks:
        if price > limit_price:
            break
        remaining -= size
        if remaining <= 1e-9:
            return True
    return False


def process_intent(
    store: Phase4Store,
    client: ClobWrapper,
    intent: dict[str, Any],
    max_spread_cents: int,
    ttl_seconds: float,
    dry_run: bool,
    killswitch: bool,
) -> tuple[str, str | None]:
    token_id = _extract_token_id(intent)
    if not token_id:
        return "FAILED", FAILED_NO_TOKEN_ID

    try:
        book = client.get_orderbook(token_id)
    except Exception:
        return "FAILED", FAILED_BOOK_UNAVAILABLE

    bids, asks = _normalize_book(book)
    if not bids or not asks:
        return "FAILED", FAILED_BOOK_UNAVAILABLE

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    spread_cents = int(round((best_ask - best_bid) * 100))
    if spread_cents > max_spread_cents:
        return "SKIPPED_BOOKCHECK", SKIP_SPREAD_TOO_WIDE

    side = str(intent.get("side") or "BUY").upper()
    qty = float(intent.get("my_size") or 0.0)
    limit_price = float(intent.get("my_limit_price") or 0.0)

    if side == "BUY" and not _has_depth_for_buy(asks, limit_price, qty):
        return "SKIPPED_BOOKCHECK", SKIP_INSUFFICIENT_DEPTH

    if dry_run or killswitch:
        return "APPROVED_READY", None

    try:
        order_resp = client.place_limit_order(token_id, side, limit_price, qty)
    except Exception as exc:
        return "FAILED", f"{FAILED_ORDER_PLACE}:{exc}"

    order_id = None
    if isinstance(order_resp, dict):
        order_id = order_resp.get("orderID") or order_resp.get("orderId") or order_resp.get("id")
    order_record = {
        "intent_id": intent["id"],
        "created_at": int(time.time()),
        "token_id": token_id,
        "side": side,
        "limit_price": limit_price,
        "qty": qty,
        "order_id": order_id,
        "status": "PLACED",
        "last_update_at": int(time.time()),
        "error": None,
    }
    store.insert_order(order_record)

    time.sleep(ttl_seconds)

    if order_id:
        client.cancel(order_id)
        store.update_order_status(order_id, "CANCELLED")
    return "CANCELLED_TTL", None
