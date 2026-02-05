from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


STATE_VERSION = 1


def _default_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "day": "",
        "daily_new_markets": 0,
        "max_daily_new_markets": None,
        "markets": {},
        "orders": {},
    }


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _default_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _default_state()
    if not isinstance(payload, dict):
        return _default_state()
    if payload.get("version") != STATE_VERSION:
        payload.setdefault("version", STATE_VERSION)
    payload.setdefault("day", "")
    payload.setdefault("daily_new_markets", 0)
    payload.setdefault("markets", {})
    payload.setdefault("orders", {})
    return payload


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def ensure_day(state: dict[str, Any], day_str: str) -> None:
    if state.get("day") != day_str:
        state["day"] = day_str
        state["daily_new_markets"] = 0


def get_market_state(state: dict[str, Any], market_key: str) -> dict[str, Any]:
    markets = state.setdefault("markets", {})
    if market_key not in markets:
        markets[market_key] = {
            "status": "WATCHING",
            "stable_count": 0,
            "last_band_ok": False,
            "entered_at": None,
            "entry_deadline_ts": None,
            "entry_timeout_sec": None,
            "order_ids": [],
            "fills": [],
            "entry": {},
            "result": {},
        }
    return markets[market_key]

