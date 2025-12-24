# src/ws_client.py
import os
import asyncio
import logging
import base64
import random
from datetime import datetime, timezone, timedelta
import json
from typing import TYPE_CHECKING


import websockets
from dotenv import load_dotenv

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from trading_engine.models import MarketUpdate

if TYPE_CHECKING:
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
# Track active subscriptions so we can avoid duplicate subscribe/unsubscribe
ACTIVE_SUBSCRIPTIONS: set[str] = set()

# Track which subscription id (sid) belongs to which market ticker
MARKET_TO_SID: dict[str, int] = {}

# Track which outgoing subscribe command id maps to which market ticker
PENDING_SUBSCRIBE_IDS: dict[int, str] = {}

# Lightweight connection state for health checks
WS_STATE = {
    "connected": False,
    "last_connect_ts": None,
    "last_message_ts": None,
    "last_error": None,
    "stale": False,
}


def _log_event(event: str, **fields: object) -> None:
    """Structured-ish logging for WS operational events."""
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info(f"[{event}] {payload}".strip())


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

async def _connect_once(engine: "TradingEngine"):
    _log_event("WS_CONNECTING", url=KALSHI_WS_URL)

    WS_STATE["connected"] = False
    WS_STATE["last_error"] = None
    WS_STATE["stale"] = False

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
        _log_event("WS_CONNECTED")
        WS_STATE["connected"] = True
        WS_STATE["last_connect_ts"] = datetime.now(timezone.utc).isoformat()

        # Reset subscription tracking on each (re)connect; sids are connection-scoped
        ACTIVE_SUBSCRIPTIONS.clear()
        MARKET_TO_SID.clear()
        PENDING_SUBSCRIBE_IDS.clear()

        # On new connection, re-subscribe to already tracked markets
        for ticker in engine.markets.keys():
            SUBSCRIBE_QUEUE.put_nowait(ticker)
            _log_event("WS_RESUBSCRIBE_QUEUED", ticker=ticker)

        # --- subscription worker (runs concurrently) ---
        async def subscription_worker():
            sub_id = 1
            while True:
                ticker = await SUBSCRIBE_QUEUE.get()
                ticker = ticker.upper()

                if ticker in ACTIVE_SUBSCRIPTIONS or ticker in PENDING_SUBSCRIBE_IDS.values():
                    _log_event("WS_SUBSCRIBE_SKIP_DUP", ticker=ticker)
                    continue

                msg = {
                    "id": sub_id,
                    "cmd": "subscribe",
                    "params": {
                        # Subscribe to orderbook deltas and lifecycle updates for this market
                        "channels": ["orderbook_delta", "market_lifecycle_v2"],
                        "market_ticker": ticker,
                    },
                }
                PENDING_SUBSCRIBE_IDS[sub_id] = ticker

                await websocket.send(json.dumps(msg))
                _log_event("WS_SUBSCRIBE_SENT", ticker=ticker, cmd_id=sub_id)
                sub_id += 1
                ACTIVE_SUBSCRIPTIONS.add(ticker)

        # start subscription loop
        asyncio.create_task(subscription_worker())
        async def unsubscribe_worker():
            unsub_id = 10_000  # separate id space from subscribe ids
            while True:
                ticker = await UNSUBSCRIBE_QUEUE.get()
                ticker = ticker.upper()

                sid = MARKET_TO_SID.get(ticker)
                if sid is None:
                    logger.info(f"Unsubscribe requested for {ticker}, but no sid known yet; queueing removal.")
                    ACTIVE_SUBSCRIPTIONS.discard(ticker)
                    continue

                msg = {
                    "id": unsub_id,
                    "cmd": "unsubscribe",
                    "params": {"sids": [sid]},
                }
                await websocket.send(json.dumps(msg))
                _log_event("WS_UNSUBSCRIBE_SENT", ticker=ticker, sid=sid, cmd_id=unsub_id)
                unsub_id += 1
                ACTIVE_SUBSCRIPTIONS.discard(ticker)

        asyncio.create_task(unsubscribe_worker())

        # --- receive loop ---
        async for message in websocket:
            WS_STATE["last_message_ts"] = datetime.now(timezone.utc).isoformat()
            WS_STATE["stale"] = False
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                # WS lifecycle/state updates (prefer over REST polling)
                if msg_type in {"market_state", "market_status", "market_lifecycle", "market_lifecycle_v2"}:
                    msg = data.get("msg") or {}
                    ticker = (msg.get("market_ticker") or msg.get("ticker") or "").upper()
                    # v2 lifecycle uses event_type (created, activated, deactivated, determined, settled)
                    state = msg.get("state") or msg.get("status") or msg.get("event_type")

                    # Ignore lifecycle noise for markets we didn't subscribe to
                    if ticker and state:
                        if ticker not in ACTIVE_SUBSCRIPTIONS and ticker not in PENDING_SUBSCRIBE_IDS.values():
                            logger.info(f"[WS] Ignoring lifecycle for unsubscribed market {ticker} state={state}")
                            continue
                        _log_event("WS_LIFECYCLE", ticker=ticker, state=state)
                        engine.on_market_status(ticker, state)
                    continue

                # Map Kalshi "subscribed" response -> sid -> ticker
                if msg_type == "subscribed":
                    cmd_id = data.get("id")
                    sid = (data.get("msg") or {}).get("sid")
                    if isinstance(cmd_id, int) and isinstance(sid, int):
                        ticker = PENDING_SUBSCRIBE_IDS.pop(cmd_id, None)
                        if ticker:
                            MARKET_TO_SID[ticker] = sid
                            ACTIVE_SUBSCRIPTIONS.add(ticker.upper())
                            _log_event("WS_SUBSCRIBED", ticker=ticker, sid=sid, cmd_id=cmd_id)

                normalized = normalize_orderbook_message(data)
                update = MarketUpdate(**normalized)
                engine.on_market_update(update)
            except Exception as exc:
                WS_STATE["last_error"] = str(exc)
                logger.error(f"Failed to process WS message: {exc}", exc_info=True)


async def connect_and_listen(engine: "TradingEngine"):
    """
    Persistent WS loop with exponential backoff + jitter on reconnect.
    """
    backoff = 1.0
    max_backoff = 60.0
    stale_threshold = timedelta(seconds=60)

    while True:
        try:
            await _connect_once(engine)
            backoff = 1.0
        except Exception as exc:
            WS_STATE["connected"] = False
            WS_STATE["last_error"] = str(exc)
            logger.error(f"WebSocket connection error: {exc}", exc_info=True)
            _log_event("WS_RECONNECTING_AFTER_ERROR", error=str(exc))
        finally:
            WS_STATE["connected"] = False

        # Staleness check before sleeping
        try:
            last_msg = WS_STATE.get("last_message_ts")
            if last_msg:
                last_dt = datetime.fromisoformat(last_msg)
                if datetime.now(timezone.utc) - last_dt > stale_threshold:
                    WS_STATE["stale"] = True
                    _log_event("WS_STALE", last_message_ts=last_msg)
        except Exception:
            pass

        sleep_for = backoff + random.uniform(0, backoff / 2)
        _log_event("WS_RECONNECT_DELAY", sleep_for=f"{sleep_for:.1f}", backoff=f"{backoff:.1f}")
        await asyncio.sleep(sleep_for)
        backoff = min(backoff * 2, max_backoff)

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
        market_ticker = market_ticker.upper()
        if market_ticker in ACTIVE_SUBSCRIPTIONS:
            logger.info(f"[WS] Skip duplicate subscribe request for {market_ticker}")
            return
        SUBSCRIBE_QUEUE.put_nowait(market_ticker)

def request_unsubscribe(market_ticker: str) -> None:
    if market_ticker:
        UNSUBSCRIBE_QUEUE.put_nowait(market_ticker.upper())

if __name__ == "__main__":
    from trading_engine.trading_engine import TradingEngine

    engine = TradingEngine()
    asyncio.run(connect_and_listen(engine))
