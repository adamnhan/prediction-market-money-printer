from __future__ import annotations

import logging
import time
from typing import Any

import requests

from normalizer import normalize_trade
from state import set_state
from storage import Storage


logger = logging.getLogger("copy_trade_phase1")


DATA_API_TRADES_URL = "https://data-api.polymarket.com/trades"


def _extract_trade_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        if "trades" in payload and isinstance(payload["trades"], list):
            return payload["trades"]
    return []


def poll_trades(
    storage: Storage,
    target_wallet: str,
    limit: int = 100,
    session: requests.Session | None = None,
) -> None:
    sess = session or requests.Session()
    params = {"user": target_wallet, "limit": limit}
    resp = sess.get(DATA_API_TRADES_URL, params=params, timeout=10)
    resp.raise_for_status()
    trades_raw = _extract_trade_list(resp.json())

    new_count = 0
    dupe_count = 0
    latest_ts = None

    for raw in trades_raw:
        trade = normalize_trade(raw, target_wallet)
        if trade is None:
            continue
        inserted = storage.insert_trade(trade)
        if inserted:
            new_count += 1
            age_s = max(0.0, trade["ingested_at"] - trade["timestamp"])
            logger.info(
                "new_trade %s %s cond=%s price=%.4f size=%.4f age=%.1fs",
                trade["side"],
                trade.get("outcome") or "-",
                trade["condition_id"],
                trade["price"],
                trade["size"],
                age_s,
            )
        else:
            dupe_count += 1
        if latest_ts is None or trade["timestamp"] > latest_ts:
            latest_ts = trade["timestamp"]

    if latest_ts is not None:
        set_state(storage, "last_trade_timestamp", latest_ts)
    set_state(storage, "last_poll_time", int(time.time()))

    logger.info("polled_trades=%d new=%d dupes=%d", len(trades_raw), new_count, dupe_count)


def poll_trades_loop(
    storage: Storage,
    target_wallet: str,
    poll_interval_s: float = 2.0,
    limit: int = 100,
) -> None:
    session = requests.Session()
    while True:
        try:
            poll_trades(storage, target_wallet, limit=limit, session=session)
        except Exception as exc:
            logger.warning("trades_poll_error err=%s", exc)
        time.sleep(poll_interval_s)
