from __future__ import annotations

import logging
from typing import Any

import requests


logger = logging.getLogger("copy_trade_phase2")


DEFAULT_BASE_URL = "https://gamma-api.polymarket.com"


def _extract_market_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        if "markets" in payload and isinstance(payload["markets"], list):
            return payload["markets"]
        if "events" in payload and isinstance(payload["events"], list):
            return payload["events"]
    return []


def _fetch_json(session: requests.Session, url: str, params: dict[str, Any]) -> Any:
    resp = session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_market_by_condition_id(
    session: requests.Session,
    condition_id: str,
    base_url: str = DEFAULT_BASE_URL,
) -> list[dict[str, Any]]:
    # Best-effort strategies; not all endpoints support condition_id filtering.
    strategies = [
        (f"{base_url}/markets", {"conditionId": condition_id}),
        (f"{base_url}/markets", {"condition_id": condition_id}),
        (f"{base_url}/markets", {"query": condition_id}),
        (f"{base_url}/events", {"query": condition_id}),
    ]

    for url, params in strategies:
        try:
            payload = _fetch_json(session, url, params)
        except Exception as exc:
            logger.debug("fetch_failed url=%s err=%s", url, exc)
            continue
        markets = _extract_market_list(payload)
        if markets:
            return markets
    return []
