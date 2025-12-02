"""
module: src.platform_ops

Same functionality as the previous platform_ops but moved into `src/` for
better package layout. This file contains async helpers for Kalshi API
communication.
"""

from typing import Any, Dict, Optional
import os
import asyncio
import httpx


KALSHI_BASE_URL = "https://api.kalshi.com/trade-api/v2"
KALSHI_ELECTIONS_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# --- Event & Market Data ---
GET_EVENTS = f"{KALSHI_ELECTIONS_BASE_URL}/events"
GET_EVENT = lambda event_ticker: f"{KALSHI_ELECTIONS_BASE_URL}/events/{event_ticker}"
GET_MARKET = lambda market_ticker: f"{KALSHI_BASE_URL}/markets/{market_ticker}"
GET_MARKET_CANDLES = lambda market_ticker: f"{KALSHI_BASE_URL}/markets/{market_ticker}/candles"
GET_MARKET_TRADES = lambda market_ticker: f"{KALSHI_BASE_URL}/markets/{market_ticker}/trades"

# --- Trading Operations ---
PLACE_ORDER = f"{KALSHI_BASE_URL}/orders"
CANCEL_ORDER = lambda order_id: f"{KALSHI_BASE_URL}/orders/{order_id}/cancel"
GET_ORDER_STATUS = lambda order_id: f"{KALSHI_BASE_URL}/orders/{order_id}"
GET_POSITIONS = f"{KALSHI_BASE_URL}/positions"
GET_PORTFOLIO = f"{KALSHI_BASE_URL}/portfolio"

# --- misc ---
DEFAULT_TIMEOUT = 20


def _build_headers(api_key: Optional[str] = None, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers: Dict[str, str] = {"Accept": "application/json"}
    key = api_key or os.getenv("KALSHI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
        headers["X-API-Key"] = key
    if extra:
        headers.update(extra)
    return headers


async def _request(method: str, url: str, params: Optional[Dict[str, Any]] = None, json: Optional[Any] = None,
                   api_key: Optional[str] = None, headers: Optional[Dict[str, str]] = None,
                   timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    hdrs = _build_headers(api_key=api_key, extra=headers)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, url, params=params, json=json, headers=hdrs)
        status = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = resp.text

        if 200 <= status < 300:
            return {"ok": True, "status": status, "data": body}

        err_msg = body.get("error") if isinstance(body, dict) else body
        return {"ok": False, "status": status, "error": err_msg}
    except httpx.RequestError as e:
        return {"ok": False, "status": None, "error": str(e)}


# --- wrapper helpers ---
async def get_events(limit: int = 100, cursor: Optional[str] = None, with_nested_markets: Optional[bool] = None,
                     status: Optional[str] = None, series_ticker: Optional[str] = None,
                     min_close_ts: Optional[int] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {"limit": limit}
    if cursor: params["cursor"] = cursor
    if with_nested_markets is not None: params["with_nested_markets"] = str(with_nested_markets).lower()
    if status: params["status"] = status
    if series_ticker: params["series_ticker"] = series_ticker
    if min_close_ts is not None: params["min_close_ts"] = min_close_ts
    return await _request("GET", GET_EVENTS, params=params, api_key=api_key)


async def get_event(event_ticker: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    return await _request("GET", GET_EVENT(event_ticker), api_key=api_key)


async def get_market(market_ticker: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    url = GET_MARKET(market_ticker)
    print(f"[platform_ops] DEBUG get_market URL={url}")   # <-- TEMP LOG
    return await _request("GET", url, api_key=api_key)


async def get_market_candles(market_ticker: str, start_ts: Optional[int] = None, end_ts: Optional[int] = None,
                             fidelity: Optional[int] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if start_ts is not None: params["startTs"] = start_ts
    if end_ts is not None: params["endTs"] = end_ts
    if fidelity is not None: params["fidelity"] = fidelity
    return await _request("GET", GET_MARKET_CANDLES(market_ticker), params=params or None, api_key=api_key)


async def get_market_trades(market_ticker: str, limit: int = 100, offset: int = 0, api_key: Optional[str] = None) -> Dict[str, Any]:
    params = {"limit": limit, "offset": offset}
    return await _request("GET", GET_MARKET_TRADES(market_ticker), params=params, api_key=api_key)


# --- trading helpers ---
async def place_order(order_payload: Dict[str, Any], api_key: Optional[str] = None) -> Dict[str, Any]:
    return await _request("POST", PLACE_ORDER, json=order_payload, api_key=api_key)


async def cancel_order(order_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    return await _request("POST", CANCEL_ORDER(order_id), api_key=api_key)


async def get_order_status(order_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    return await _request("GET", GET_ORDER_STATUS(order_id), api_key=api_key)


async def get_positions(api_key: Optional[str] = None) -> Dict[str, Any]:
    return await _request("GET", GET_POSITIONS, api_key=api_key)


async def get_portfolio(api_key: Optional[str] = None) -> Dict[str, Any]:
    return await _request("GET", GET_PORTFOLIO, api_key=api_key)


if __name__ == "__main__":
    async def _demo():
        print("Demo: get_events(limit=3)")
        r = await get_events(limit=3)
        if r["ok"]:
            print("events:", r["data"] if isinstance(r.get("data"), list) else r["data"].get("events"))
        else:
            print("get_events failed:", r.get("error"))

    asyncio.run(_demo())
