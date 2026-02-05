from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


logger = logging.getLogger("kalshi_mvp_ws")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[KXNHLMVP-WS] %(asctime)s %(message)s"))
    logger.addHandler(handler)

if load_dotenv:
    load_dotenv()


@dataclass
class BestPrices:
    yes_bid: int | None
    yes_bid_size: int | None
    yes_ask: int | None
    yes_ask_size: int | None
    no_bid: int | None
    no_bid_size: int | None
    no_ask: int | None
    no_ask_size: int | None


def _load_private_key(path: str) -> object | None:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("private_key_read_failed err=%s", exc)
        return None
    try:
        return serialization.load_pem_private_key(
            raw.encode("utf-8"),
            password=None,
            backend=default_backend(),
        )
    except Exception as exc:
        logger.error("private_key_parse_failed err=%s", exc)
        return None


def _build_ws_auth_headers(key_id: str, private_key: object, ws_url: str) -> dict[str, str] | None:
    try:
        path = "/trade-api/ws/v2"
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        message = f"{timestamp_ms}GET{path}".encode("utf-8")
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        signature_b64 = base64.b64encode(signature).decode("utf-8")
    except Exception as exc:
        logger.error("ws_auth_failed err=%s", exc)
        return None
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        "KALSHI-ACCESS-SIGNATURE": signature_b64,
    }


class WSOrderbookClient:
    def __init__(self, ws_url: str, key_id: str, private_key_path: str) -> None:
        self.ws_url = ws_url
        self.key_id = key_id
        self.private_key_path = private_key_path
        self.private_key = _load_private_key(private_key_path)
        self.books: dict[str, dict[str, dict[int, int]]] = {}
        self.active_tickers: set[str] = set()
        self.pending_subscribe: dict[int, str] = {}
        self.market_to_sid: dict[str, int] = {}
        self.subscribe_queue: asyncio.Queue[str] = asyncio.Queue()
        self.unsubscribe_queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self.connected = False

    @classmethod
    def from_env(cls) -> "WSOrderbookClient":
        ws_url = os.getenv("KALSHI_WS_URL", "wss://api.elections.kalshi.com/trade-api/ws/v2")
        key_id = os.getenv("KALSHI_KEY_ID")
        private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        if not key_id or not private_key_path:
            raise RuntimeError("Missing KALSHI_KEY_ID or KALSHI_PRIVATE_KEY_PATH for websocket auth.")
        return cls(ws_url, key_id, private_key_path)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_loop())

    async def set_desired_tickers(self, tickers: set[str]) -> None:
        tickers = {t.upper() for t in tickers if t}
        for ticker in tickers - self.active_tickers:
            await self.subscribe_queue.put(ticker)
        for ticker in self.active_tickers - tickers:
            await self.unsubscribe_queue.put(ticker)

    def get_best_prices(self, ticker: str) -> BestPrices | None:
        book = self.books.get(ticker.upper())
        if not book:
            return None
        yes_book = book.get("yes", {})
        no_book = book.get("no", {})
        yes_bid, yes_bid_size = self._best_bid(yes_book)
        no_bid, no_bid_size = self._best_bid(no_book)
        yes_ask = 100 - no_bid if no_bid is not None else None
        no_ask = 100 - yes_bid if yes_bid is not None else None
        yes_ask_size = no_bid_size if no_bid is not None else None
        no_ask_size = yes_bid_size if yes_bid is not None else None
        return BestPrices(
            yes_bid=yes_bid,
            yes_bid_size=yes_bid_size,
            yes_ask=yes_ask,
            yes_ask_size=yes_ask_size,
            no_bid=no_bid,
            no_bid_size=no_bid_size,
            no_ask=no_ask,
            no_ask_size=no_ask_size,
        )

    @staticmethod
    def _best_bid(side_book: dict[int, int]) -> tuple[int | None, int | None]:
        best_price = None
        best_size = None
        for price, size in side_book.items():
            if size <= 0:
                continue
            if best_price is None or price > best_price:
                best_price = price
                best_size = size
        return best_price, best_size

    async def _run_loop(self) -> None:
        if self.private_key is None:
            logger.error("ws_disabled reason=missing_private_key")
            return

        backoff = 1.0
        while True:
            headers = _build_ws_auth_headers(self.key_id, self.private_key, self.ws_url)
            if headers is None:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            try:
                async with websockets.connect(self.ws_url, additional_headers=headers) as websocket:
                    logger.info("ws_connected url=%s", self.ws_url)
                    self.connected = True
                    self.pending_subscribe.clear()
                    self.market_to_sid.clear()

                    async def subscribe_worker() -> None:
                        sub_id = 1
                        while True:
                            ticker = await self.subscribe_queue.get()
                            if ticker in self.active_tickers:
                                continue
                            payload = {
                                "id": sub_id,
                                "cmd": "subscribe",
                                "params": {
                                    "channels": ["orderbook_snapshot", "orderbook_delta", "market_lifecycle_v2"],
                                    "market_ticker": ticker,
                                },
                            }
                            self.pending_subscribe[sub_id] = ticker
                            await websocket.send(json.dumps(payload))
                            self.active_tickers.add(ticker)
                            sub_id += 1

                    async def unsubscribe_worker() -> None:
                        unsub_id = 10_000
                        while True:
                            ticker = await self.unsubscribe_queue.get()
                            sid = self.market_to_sid.get(ticker)
                            if sid is None:
                                self.active_tickers.discard(ticker)
                                continue
                            payload = {"id": unsub_id, "cmd": "unsubscribe", "params": {"sids": [sid]}}
                            await websocket.send(json.dumps(payload))
                            self.active_tickers.discard(ticker)
                            unsub_id += 1

                    asyncio.create_task(subscribe_worker())
                    asyncio.create_task(unsubscribe_worker())

                    async for message in websocket:
                        data = json.loads(message)
                        msg_type = data.get("type")
                        msg = data.get("msg") or {}
                        if msg_type == "subscribed":
                            cmd_id = data.get("id")
                            sid = (data.get("msg") or {}).get("sid")
                            if isinstance(cmd_id, int) and isinstance(sid, int):
                                ticker = self.pending_subscribe.pop(cmd_id, None)
                                if ticker:
                                    self.market_to_sid[ticker] = sid
                            continue

                        ticker = (msg.get("market_ticker") or msg.get("ticker") or "").upper()
                        if not ticker:
                            continue
                        if ticker not in self.books:
                            self.books[ticker] = {"yes": {}, "no": {}}
                        book = self.books[ticker]

                        if msg_type == "orderbook_snapshot":
                            book["yes"] = {int(p): int(sz) for p, sz in (msg.get("yes") or [])}
                            book["no"] = {int(p): int(sz) for p, sz in (msg.get("no") or [])}
                        elif msg_type == "orderbook_delta":
                            price = msg.get("price")
                            delta = msg.get("delta")
                            side = msg.get("side")
                            if isinstance(price, int) and isinstance(delta, int) and side in ("yes", "no"):
                                prev = book[side].get(price, 0)
                                new_sz = prev + delta
                                if new_sz <= 0:
                                    book[side].pop(price, None)
                                else:
                                    book[side][price] = new_sz
            except Exception as exc:
                logger.warning("ws_disconnected err=%s", exc)
                self.connected = False
                delay = backoff + random.uniform(0, backoff / 2)
                await asyncio.sleep(delay)
                backoff = min(backoff * 2, 30.0)
