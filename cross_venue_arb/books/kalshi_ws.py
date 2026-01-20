"""Kalshi websocket client for order book updates."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
from urllib.parse import urlsplit
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import websockets
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from cross_venue_arb.books.manager import BookManager
from cross_venue_arb.config import CONFIG


logger = logging.getLogger("kalshi_ws")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[KALSHI_WS] %(asctime)s %(message)s"))
    logger.addHandler(handler)


def _load_private_key() -> object | None:
    if not CONFIG.kalshi.private_key_path:
        logger.error("KALSHI_PRIVATE_KEY_PATH not set.")
        return None
    try:
        pem = Path(CONFIG.kalshi.private_key_path).read_text(encoding="utf-8")
        return serialization.load_pem_private_key(
            pem.encode("utf-8"),
            password=None,
            backend=default_backend(),
        )
    except Exception as exc:
        logger.error("Failed to load Kalshi private key: %s", exc)
        return None


def _build_auth_headers(private_key: object, request_path: str) -> dict[str, str] | None:
    if not CONFIG.kalshi.key_id:
        logger.error("KALSHI_KEY_ID not set.")
        return None
    now = datetime.now(timezone.utc)
    timestamp_ms = int(now.timestamp() * 1000)
    message = f"{timestamp_ms}GET{request_path}".encode("utf-8")
    try:
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
    except Exception as exc:
        logger.error("Failed to sign Kalshi WS auth message: %s", exc)
        return None
    signature_b64 = base64.b64encode(signature).decode("utf-8")
    return {
        "KALSHI-ACCESS-KEY": CONFIG.kalshi.key_id,
        "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        "KALSHI-ACCESS-SIGNATURE": signature_b64,
    }


def _parse_exchange_ts(value: object) -> float | None:
    if isinstance(value, (int, float)):
        if value > 1e12:
            return float(value) / 1000.0
        return float(value)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt.timestamp()
    return None


def _convert_levels(levels: Iterable[Iterable[object]]) -> dict[float, float]:
    parsed: dict[float, float] = {}
    for raw_price, raw_size in levels:
        try:
            price = float(raw_price) / 100.0
            size = float(raw_size)
        except (TypeError, ValueError):
            continue
        if size > 0:
            parsed[price] = size
    return parsed


def _convert_no_levels_to_asks(no_levels: dict[float, float]) -> dict[float, float]:
    asks: dict[float, float] = {}
    for price, size in no_levels.items():
        ask_price = round(1.0 - price, 4)
        asks[ask_price] = size
    return asks


async def run(
    manager: BookManager,
    tickers: Iterable[str],
) -> None:
    ws_url = CONFIG.kalshi.ws_url or "wss://api.elections.kalshi.com/trade-api/ws/v2"
    request_path = urlsplit(ws_url).path or "/trade-api/ws/v2"
    private_key = _load_private_key()
    if private_key is None:
        return

    tickers = [ticker.upper() for ticker in tickers]
    backoff = 1.0

    while True:
        try:
            headers = _build_auth_headers(private_key, request_path)
            if headers is None:
                raise RuntimeError("missing_kalshi_auth")
            async with websockets.connect(ws_url, additional_headers=headers) as websocket:
                logger.info("connected")
                backoff = 1.0
                for idx, ticker in enumerate(tickers, start=1):
                    msg = {
                        "id": idx,
                        "cmd": "subscribe",
                        "params": {
                            "channels": ["orderbook_snapshot", "orderbook_delta"],
                            "market_ticker": ticker,
                        },
                    }
                    await websocket.send(json.dumps(msg))
                async for raw in websocket:
                    data = json.loads(raw)
                    msg_type = data.get("type")
                    msg = data.get("msg") or {}
                    market_ticker = msg.get("market_ticker")
                    if not market_ticker:
                        continue
                    exchange_ts = _parse_exchange_ts(msg.get("ts"))
                    if msg_type == "orderbook_snapshot":
                        yes_levels = _convert_levels(msg.get("yes") or [])
                        no_levels = _convert_levels(msg.get("no") or [])
                        asks = _convert_no_levels_to_asks(no_levels)
                        manager.snapshot(
                            "kalshi",
                            market_ticker,
                            bids=yes_levels,
                            asks=asks,
                            exchange_ts=exchange_ts,
                            raw_no_bids=no_levels,
                        )
                    elif msg_type == "orderbook_delta":
                        price_raw = msg.get("price")
                        delta_raw = msg.get("delta")
                        side = msg.get("side")
                        if side not in {"yes", "no"}:
                            continue
                        try:
                            price = float(price_raw) / 100.0
                            delta = float(delta_raw)
                        except (TypeError, ValueError):
                            continue
                        changes: list[tuple[str, float, float]] = []
                        raw_no_changes: list[tuple[float, float]] | None = None
                        book = manager.get_book("kalshi", market_ticker)
                        if side == "yes":
                            current = book.bids.get(price, 0.0) if book else 0.0
                            new_size = current + delta
                            changes.append(("bid", price, new_size))
                        else:
                            ask_price = round(1.0 - price, 4)
                            current = book.asks.get(ask_price, 0.0) if book else 0.0
                            new_size = current + delta
                            changes.append(("ask", ask_price, new_size))
                            raw_no_changes = [(price, new_size)]
                        manager.delta(
                            "kalshi",
                            market_ticker,
                            changes,
                            exchange_ts=exchange_ts,
                            raw_no_changes=raw_no_changes,
                        )
        except Exception as exc:
            logger.error("ws error: %s", exc)

        sleep_for = backoff + random.uniform(0, backoff / 2)
        await asyncio.sleep(sleep_for)
        backoff = min(backoff * 2, 60.0)
