"""Polymarket websocket client for order book updates."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from typing import Iterable

import websockets

from cross_venue_arb.books.manager import BookManager
from cross_venue_arb.config import CONFIG


logger = logging.getLogger("polymarket_ws")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[POLY_WS] %(asctime)s %(message)s"))
    logger.addHandler(handler)


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


def _parse_levels(levels: Iterable[object]) -> dict[float, float]:
    parsed: dict[float, float] = {}
    for entry in levels:
        if isinstance(entry, dict):
            price = entry.get("price")
            size = entry.get("size")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            price, size = entry[0], entry[1]
        else:
            continue
        try:
            price = float(price)
            size = float(size)
        except (TypeError, ValueError):
            continue
        if size > 0:
            parsed[price] = size
    return parsed


def _parse_changes(raw: object) -> list[tuple[str, float, float]]:
    changes: list[tuple[str, float, float]] = []
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                side = entry.get("side") or entry.get("type")
                price = entry.get("price")
                size = entry.get("size") or entry.get("amount")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
                side, price, size = entry[0], entry[1], entry[2]
            else:
                continue
            if side is None:
                continue
            side_norm = str(side).lower()
            if side_norm in {"buy", "bid"}:
                side_norm = "bid"
            elif side_norm in {"sell", "ask"}:
                side_norm = "ask"
            else:
                continue
            try:
                price = float(price)
                size = float(size)
            except (TypeError, ValueError):
                continue
            changes.append((side_norm, price, size))
    return changes


def _extract_market_id(msg: dict) -> str | None:
    for key in ("asset_id", "assetId", "token_id", "tokenId", "id", "market"):
        value = msg.get(key)
        if value:
            return str(value)
    return None


async def run(manager: BookManager, market_ids: Iterable[str]) -> None:
    ws_urls = [
        url.strip().rstrip("/")
        for url in (CONFIG.polymarket.ws_url or "wss://ws-subscriptions-clob.polymarket.com/ws/market").split(",")
        if url.strip()
    ]
    market_ids = [str(market_id) for market_id in market_ids]

    subscribe_msg = {"type": "market", "assets_ids": market_ids}
    logger.info("subscribe assets_ids=%d", len(market_ids))

    backoff = 1.0
    while True:
        try:
            last_error: Exception | None = None
            websocket = None
            for ws_url in ws_urls:
                try:
                    websocket = await websockets.connect(ws_url)
                    logger.info("connected url=%s", ws_url)
                    break
                except Exception as exc:
                    last_error = exc
                    logger.error("ws error: %s (url=%s)", exc, ws_url)
                    await asyncio.sleep(0.5)
            if websocket is None:
                if last_error:
                    raise last_error
                raise RuntimeError("No websocket URL available")

            async with websocket:
                backoff = 1.0
                await websocket.send(json.dumps(subscribe_msg))
                async for raw in websocket:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("raw=%s", raw[:500])
                    data = json.loads(raw)
                    payloads = data if isinstance(data, list) else [data]
                    for payload in payloads:
                        if not isinstance(payload, dict):
                            continue
                        if payload.get("event_type") == "error":
                            logger.error("ws error payload=%s", payload)
                            continue
                        market_id = _extract_market_id(payload)
                        if not market_id:
                            continue
                        exchange_ts = _parse_exchange_ts(
                            payload.get("timestamp") or payload.get("ts") or payload.get("time")
                        )
                        bids = payload.get("bids")
                        asks = payload.get("asks")
                        if bids is not None and asks is not None:
                            manager.snapshot(
                                "polymarket",
                                market_id,
                                bids=_parse_levels(bids),
                                asks=_parse_levels(asks),
                                exchange_ts=exchange_ts,
                            )
                            continue
                        changes = _parse_changes(
                            payload.get("price_changes")
                            or payload.get("changes")
                            or payload.get("updates")
                            or []
                        )
                        if changes:
                            manager.delta("polymarket", market_id, changes, exchange_ts=exchange_ts)
        except Exception as exc:
            logger.error("ws error: %s", exc)

        sleep_for = backoff + random.uniform(0, backoff / 2)
        await asyncio.sleep(sleep_for)
        backoff = min(backoff * 2, 60.0)
