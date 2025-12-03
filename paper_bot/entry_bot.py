# paper_bot/entry_bot.py

import asyncio
import base64
import json
import time
from pathlib import Path
from datetime import datetime, timezone


import websockets
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from .store import TradeRow, append_trade
from .config import KALSHI_WS_URL, KALSHI_KEY_ID, KALSHI_PRIVATE_KEY_PATH, JUST_OPEN_WINDOW_SEC
from .rest import request

# -------- Private key + signing helpers --------

_private_key_cache = None


def _load_private_key():
    """
    Load and cache the RSA private key used for signing requests.
    """
    global _private_key_cache
    if _private_key_cache is None:
        if not KALSHI_PRIVATE_KEY_PATH:
            raise RuntimeError("KALSHI_PRIVATE_KEY_PATH is not set in the environment/config.")

        key_path = Path(KALSHI_PRIVATE_KEY_PATH)
        if not key_path.exists():
            raise FileNotFoundError(f"Private key file not found at: {key_path}")

        with key_path.open("rb") as f:
            _private_key_cache = serialization.load_pem_private_key(
                f.read(),
                password=None,
            )
    return _private_key_cache


def _sign_pss_text(private_key, text: str) -> str:
    """
    Sign the given text with RSA-PSS + SHA256 and return a base64 string.
    """
    message = text.encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")

def _create_ws_headers() -> dict:
    """
    Create Kalshi WebSocket authentication headers.

    Per Kalshi docs, the message to sign is:
        timestamp + "GET" + "/trade-api/ws/v2"
    """
    if not KALSHI_KEY_ID:
        raise RuntimeError("KALSHI_KEY_ID is not set in the environment/config.")

    private_key = _load_private_key()
    timestamp = str(int(time.time() * 1000))

    path = "/trade-api/ws/v2"  # must match WS path
    msg_string = timestamp + "GET" + path

    signature = _sign_pss_text(private_key, msg_string)

    return {
        "Content-Type": "application/json",
        "KALSHI-ACCESS-KEY": KALSHI_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }

pending_markets = {}  # make sure this is defined at module level
TARGET_CATEGORIES = {"culture", "politics", "economics", "mentions", "sports"}

def handle_market_lifecycle(msg):
    """
    Looser version:
    - Use open_ts and 1-week window.
    - Fetch category via the event (preferred) or market.
    - Only add markets whose category is in TARGET_CATEGORIES.
    """
    global pending_markets

    inner = msg.get("msg")
    if not inner:
        print("[DEBUG][SKIP] lifecycle: no inner msg")
        return

    ticker = inner.get("market_ticker")
    open_ts = inner.get("open_ts")

    if not ticker:
        print("[DEBUG][SKIP] lifecycle: no market_ticker")
        return

    if open_ts is None:
        print(f"[DEBUG][SKIP] {ticker}: no open_ts")
        return

    if ticker in pending_markets:
        return  # already watching this one

    try:
        open_dt = datetime.fromtimestamp(open_ts, tz=timezone.utc)
    except Exception as e:
        print(f"[DEBUG][SKIP] {ticker}: bad open_ts={open_ts} err={e}")
        return

    now = datetime.now(timezone.utc)
    delta = (now - open_dt).total_seconds()

    from .config import JUST_OPEN_WINDOW_SEC, TARGET_CATEGORIES

    if delta < 0:
        print(f"[DEBUG][SKIP] {ticker}: open_ts in future (delta={delta}s)")
        return

    if delta > JUST_OPEN_WINDOW_SEC:
        print(
            f"[DEBUG][SKIP] {ticker}: too old (delta={delta}s > "
            f"window={JUST_OPEN_WINDOW_SEC}s)"
        )
        return

    # ----- resolve category via event -----
    category = "unknown"
    try:
        additional = inner.get("additional_metadata") or {}
        event_ticker = additional.get("event_ticker")

        # Fallback: if lifecycle doesn't carry event_ticker, ask /markets
        if not event_ticker:
            m_resp = request(f"/markets/{ticker}")
            market = m_resp.get("market") or {}
            event_ticker = market.get("event_ticker")

        # Prefer category from the event
        if event_ticker:
            e_resp = request(f"/events/{event_ticker}")
            event = e_resp.get("event") or {}
            cat = (event.get("category") or "").lower()
        else:
            cat = ""

        # Fallback: category directly on market
        if not cat:
            m_resp = request(f"/markets/{ticker}")
            market = m_resp.get("market") or {}
            cat = (market.get("category") or "").lower()

        # Heuristic: NFL tickers -> sports
        if not cat:
            full_ticker = (ticker or "").upper()
            if "NFL" in full_ticker:
                cat = "sports"

        if cat:
            category = cat
    except Exception as e:
        print(f"[DEBUG][SKIP] {ticker}: category resolution error: {e}")
        return

    if category not in TARGET_CATEGORIES:
        return

    pending_markets[ticker] = {
        "open_time": open_dt,
        "category": category,
    }

    print(
        f"[LIFECYCLE] ADDED (filtered): {ticker} "
        f"category={category} open_dt={open_dt}"
    )


def handle_ticker(msg):
    """
    Handles ticker messages.

    Structure:
    {
      "type": "ticker",
      "sid": 2,
      "msg": {
        "market_id": "...",
        "market_ticker": "...",
        "price_dollars": "0.8800",
        ...
      }
    }
    """
    global pending_markets

    inner = msg.get("msg")  # <-- note: 'msg', not 'event'
    if not inner:
        return

    ticker = inner.get("market_ticker")
    if not ticker:
        return

    # Debug: see if this ticker is one we’re tracking
    if ticker in pending_markets:
        print(f"[DEBUG] Ticker for pending market: {ticker}")

    info = pending_markets.get(ticker)
    if info is None:
        return  # not one of our "just opened" (pending) markets

    price_dollars = inner.get("price_dollars")
    if price_dollars is None:
        return

    try:
        entry_price_yes = float(price_dollars)
    except (TypeError, ValueError):
        print(f"[DEBUG] Bad price_dollars for {ticker}: {price_dollars}")
        return

    if not (0.0 <= entry_price_yes <= 1.0):
        print(f"[DEBUG] Out-of-range price for {ticker}: {entry_price_yes}")
        return

    entry_price_no = 1.0 - entry_price_yes
    now = datetime.now(timezone.utc)

       # --- Try to improve category at trade time (via market -> event) ---
    category = info.get("category", "unknown")
    if category == "unknown":
        try:
            # 1) Get the market so we can find its event_ticker
            m_resp = request(f"/markets/{ticker}")
            market = m_resp.get("market") or {}
            event_ticker = market.get("event_ticker")

            cat = ""

            # 2) Prefer category from the event object
            if event_ticker:
                e_resp = request(f"/events/{event_ticker}")
                event = e_resp.get("event") or {}
                cat = (event.get("category") or "").lower()

            # 3) Fallback: category on market (if present at all)
            if not cat:
                cat = (market.get("category") or "").lower()

            # 4) Heuristic: treat obvious NFL tickers as sports
            if not cat:
                full_ticker = (market.get("ticker") or ticker or "").upper()
                if "NFL" in full_ticker:
                    cat = "sports"

            if cat:
                category = cat

        except Exception as e:
            print(f"[DEBUG] category lookup failed at trade time for {ticker}: {e}")

    row = TradeRow(
        timestamp=now.isoformat(),
        market_ticker=ticker,
        category=category,
        open_time=info["open_time"].isoformat(),
        entry_price_yes=entry_price_yes,
        entry_price_no=entry_price_no,
        entry_side="NO",
    )

    append_trade(row)
    pending_markets.pop(ticker, None)

    print(
        f"[TRADE] {ticker}: long NO at {entry_price_no:.3f} "
        f"(YES={entry_price_yes:.3f}) category={category}"
    )

def fetch_category_for_ticker(ticker: str) -> str:
    """
    Look up the market, then its series, and return the category in lowercase.
    Returns "" if anything fails.
    """
    try:
        # 1) Get the market
        resp = request(f"/markets/{ticker}")
        market = resp.get("market") or {}
        series_ticker = market.get("series_ticker")
        if not series_ticker:
            print(f"[DEBUG] {ticker}: no series_ticker on market")
            return ""

        # 2) Get the series to read its category
        series_resp = request(f"/series/{series_ticker}")
        series = series_resp.get("series") or {}
        category = (series.get("category") or "").lower()
        if not category:
            print(f"[DEBUG] {ticker}: series {series_ticker} has empty category")
        return category
    except Exception as e:
        print(f"[DEBUG] fetch_category_for_ticker error for {ticker}: {e}")
        return ""


# -------- Subscription helpers --------

async def _subscribe(ws):
    """
    Subscribe to market_lifecycle_v2 and ticker channels.
    We don't filter by market yet (we'll add smarter logic later).
    """
    msg = {
        "id": 1,
        "cmd": "subscribe",
        "params": {
            "channels": [
                "market_lifecycle_v2",
                "ticker",
            ]
            # No market_ticker filter => receive all markets for these channels
        },
    }
    await ws.send(json.dumps(msg))
    print("Sent subscribe command:", msg)

# Track markets waiting for first price update
msg_count = 0

# -------- Entry bot skeleton --------

async def run_entry_bot():
    """
    Connect to Kalshi WebSocket, subscribe to lifecycle + ticker,
    and print any incoming messages.
    """
    global msg_count
    headers = _create_ws_headers()

    print("Connecting to:", KALSHI_WS_URL)
    print("Using KEY_ID:", KALSHI_KEY_ID)
    print("Timestamp:", headers["KALSHI-ACCESS-TIMESTAMP"])
    try:
        async with websockets.connect(KALSHI_WS_URL, additional_headers=headers) as ws:
            print(f"Connected to Kalshi WebSocket: {KALSHI_WS_URL}")

            # Subscribe to channels
            await _subscribe(ws)

            # For now, just listen + print
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    print("Raw WS message (non-JSON):", raw)
                    continue

                msg_type = msg.get("type")

                # ---- Lifecycle handling ----
                if msg_type == "market_lifecycle_v2":
                    # print("\n\n===== LIFECYCLE RAW EVENT =====")
                    # print(json.dumps(msg, indent=2))
                    # print("================================\n")
                    handle_market_lifecycle(msg)
                    continue


                # ---- Ticker handling (we fill this in next microstep) ----
                if msg_type == "ticker":
                    handle_ticker(msg)   # we'll add this next
                    continue

                msg_count += 1
                if msg_count % 50 == 0:
                    print("[DEBUG] Pending markets:", pending_markets)
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C inside run_entry_bot, closing connection...")
    finally:
            print(f"[EXIT] Total messages processed: {msg_count}")
            print(f"[EXIT] Pending markets at shutdown: {list(pending_markets.keys())}")

if __name__ == "__main__":
    asyncio.run(run_entry_bot())
