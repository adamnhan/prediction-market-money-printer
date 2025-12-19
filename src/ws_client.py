# src/ws_client.py
import os
import asyncio
import logging
import base64
from datetime import datetime, timezone
import json


import websockets
from dotenv import load_dotenv

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from trading_engine.models import MarketUpdate  
from trading_engine.trading_engine import TradingEngine

# --- Logging setup ---
logger = logging.getLogger("ws_client")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[WS] %(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

# --- Load .env into process env vars ---
load_dotenv()


# --- Config ---
KALSHI_WS_URL = os.getenv(
    "KALSHI_WS_URL",
    # adjust if you're using non-elections; check your Kalshi docs
    "wss://api.elections.kalshi.com/trade-api/ws/v2",
)

KALSHI_KEY_ID = os.getenv("KALSHI_KEY_ID")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")

SUBSCRIBE_QUEUE: asyncio.Queue[str] = asyncio.Queue()
# Simple in-memory book: market_ticker -> {"yes": {price: size}, "no": {price: size}}
BOOKS: dict[str, dict[str, dict[int, int]]] = {}

UNSUBSCRIBE_QUEUE: asyncio.Queue[str] = asyncio.Queue()

# Track which subscription id (sid) belongs to which market ticker
MARKET_TO_SID: dict[str, int] = {}

# Track which outgoing subscribe command id maps to which market ticker
PENDING_SUBSCRIBE_IDS: dict[int, str] = {}


def load_private_key():
    if not KALSHI_KEY_ID or not KALSHI_PRIVATE_KEY_PATH:
        logger.error("Missing KALSHI_KEY_ID or KALSHI_PRIVATE_KEY_PATH in environment.")
        return None

    try:
        with open(KALSHI_PRIVATE_KEY_PATH, "r") as f:
            private_key_pem = f.read()
        logger.info("Loaded private key PEM successfully.")
    except Exception as exc:
        logger.error(f"Failed to read private key PEM: {exc}", exc_info=True)
        return None

    try:
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode("utf-8"),
            password=None,
            backend=default_backend(),
        )
        logger.info("Parsed private key object successfully.")
    except Exception as exc:
        logger.error(f"Failed to parse private key PEM: {exc}", exc_info=True)
        return None

    return private_key

def build_ws_auth_headers(private_key):
    """
    Build Kalshi WebSocket auth headers using RSA-PSS signature.

    Signature is based on: <timestamp_ms><HTTP_METHOD><PATH>
    For WS, HTTP_METHOD = "GET" and PATH = "/trade-api/ws/v2".
    """
    # Timestamp in milliseconds
    now = datetime.now(timezone.utc)
    timestamp_ms = int(now.timestamp() * 1000)
    timestamp_str = str(timestamp_ms)

    method = "GET"
    path = "/trade-api/ws/v2"  # path component of KALSHI_WS_URL

    message = f"{timestamp_str}{method}{path}".encode("utf-8")

    try:
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
        logger.error(f"Failed to sign WS auth message: {exc}", exc_info=True)
        return None

    headers = {
        "KALSHI-ACCESS-KEY": KALSHI_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_str,
        "KALSHI-ACCESS-SIGNATURE": signature_b64,
    }

    logger.info("Generated Kalshi WS auth headers successfully.")
    return headers

async def connect_and_listen(engine: TradingEngine):
    logger.info(f"Connecting to Kalshi WebSocket at: {KALSHI_WS_URL}")

    private_key = load_private_key()
    if private_key is None:
        return

    headers = build_ws_auth_headers(private_key)
    if headers is None:
        return

    async with websockets.connect(
        KALSHI_WS_URL,
        additional_headers=headers,
    ) as websocket:
        logger.info("WebSocket connection established.")

        # --- subscription worker (runs concurrently) ---
        async def subscription_worker():
            sub_id = 1
            while True:
                ticker = await SUBSCRIBE_QUEUE.get()
                msg = {
                    "id": sub_id,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["orderbook_delta"],
                        "market_ticker": ticker,
                    },
                }
                PENDING_SUBSCRIBE_IDS[sub_id] = ticker

                await websocket.send(json.dumps(msg))
                logger.info(f"Subscribed to {ticker}")
                sub_id += 1

        # start subscription loop
        asyncio.create_task(subscription_worker())
        async def unsubscribe_worker():
            unsub_id = 10_000  # separate id space from subscribe ids
            while True:
                ticker = await UNSUBSCRIBE_QUEUE.get()
                ticker = ticker.upper()

                sid = MARKET_TO_SID.get(ticker)
                if sid is None:
                    logger.info(f"Unsubscribe requested for {ticker}, but no sid known yet.")
                    continue

                msg = {
                    "id": unsub_id,
                    "cmd": "unsubscribe",
                    "params": {"sids": [sid]},
                }
                await websocket.send(json.dumps(msg))
                logger.info(f"Unsubscribed from {ticker} (sid={sid})")
                unsub_id += 1

        asyncio.create_task(unsubscribe_worker())

        # --- receive loop ---
        async for message in websocket:
            try:
                data = json.loads(message)
                # Map Kalshi "subscribed" response -> sid -> ticker
                if data.get("type") == "subscribed":
                    cmd_id = data.get("id")
                    sid = (data.get("msg") or {}).get("sid")
                    if isinstance(cmd_id, int) and isinstance(sid, int):
                        ticker = PENDING_SUBSCRIBE_IDS.pop(cmd_id, None)
                        if ticker:
                            MARKET_TO_SID[ticker] = sid
                            logger.info(f"[WS] Subscription confirmed: {ticker} -> sid={sid}")

                normalized = normalize_orderbook_message(data)
                update = MarketUpdate(**normalized)
                engine.on_market_update(update)
            except Exception as exc:
                logger.error(f"Failed to process WS message: {exc}", exc_info=True)


def _best_price(side_book: dict[int, int]) -> int | None:
    # "best" = lowest price level with positive size (matches snapshot ordering you saw)
    levels = [p for p, sz in side_book.items() if sz and sz > 0]
    return max(levels) if levels else None


def normalize_orderbook_message(data: dict) -> dict:
    msg_type = data.get("type")
    msg = data.get("msg") or {}

    mt = msg.get("market_ticker")

    update = {
        "msg_type": msg_type,
        "market_ticker": mt,
        "best_yes": None,
        "best_no": None,
        "last_price": None,
        "last_side": None,
        "ts": msg.get("ts"),
        "raw": data,
    }

    # ignore messages without a ticker (e.g., "subscribed")
    if not mt:
        return update

    # ensure book exists
    if mt not in BOOKS:
        BOOKS[mt] = {"yes": {}, "no": {}}

    book = BOOKS[mt]

    if msg_type == "orderbook_snapshot":
        # snapshot format: yes/no lists of [price, size]
        book["yes"] = {int(p): int(sz) for p, sz in (msg.get("yes") or [])}
        book["no"] = {int(p): int(sz) for p, sz in (msg.get("no") or [])}

    elif msg_type == "orderbook_delta":
        # delta format: {price, delta, side}
        price = msg.get("price")
        delta = msg.get("delta")
        side = msg.get("side")

        update["last_price"] = price
        update["last_side"] = side

        if isinstance(price, int) and isinstance(delta, int) and side in ("yes", "no"):
            prev = book[side].get(price, 0)
            new_sz = prev + delta
            if new_sz <= 0:
                book[side].pop(price, None)
            else:
                book[side][price] = new_sz

    # compute best after applying snapshot/delta
    update["best_yes"] = _best_price(book["yes"])
    update["best_no"] = _best_price(book["no"])

    return update

def request_subscribe(market_ticker: str) -> None:
    """
    Called by the backend when the user adds a market.
    Queue it up for the WS loop to subscribe.
    """
    if market_ticker:
        SUBSCRIBE_QUEUE.put_nowait(market_ticker)

def request_unsubscribe(market_ticker: str) -> None:
    if market_ticker:
        UNSUBSCRIBE_QUEUE.put_nowait(market_ticker.upper())

if __name__ == "__main__":
    engine = TradingEngine()
    asyncio.run(connect_and_listen(engine))
