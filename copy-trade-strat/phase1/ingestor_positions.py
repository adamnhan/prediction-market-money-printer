from __future__ import annotations

import logging
import time
from typing import Any

import requests

from normalizer import normalize_position
from storage import Storage


logger = logging.getLogger("copy_trade_phase1")


DATA_API_POSITIONS_URL = "https://data-api.polymarket.com/positions"


def _extract_position_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        if "positions" in payload and isinstance(payload["positions"], list):
            return payload["positions"]
    return []


def poll_positions(
    storage: Storage,
    target_wallet: str,
    session: requests.Session | None = None,
) -> None:
    sess = session or requests.Session()
    params = {"user": target_wallet}
    resp = sess.get(DATA_API_POSITIONS_URL, params=params, timeout=10)
    resp.raise_for_status()
    positions_raw = _extract_position_list(resp.json())

    snapshot_time = int(time.time())
    inserted = 0

    for raw in positions_raw:
        snapshot = normalize_position(raw, target_wallet, snapshot_time)
        if snapshot is None:
            continue
        storage.insert_position_snapshot(snapshot)
        inserted += 1

    logger.info("polled_positions=%d stored=%d", len(positions_raw), inserted)


def poll_positions_loop(
    storage: Storage,
    target_wallet: str,
    poll_interval_s: float = 45.0,
) -> None:
    session = requests.Session()
    while True:
        try:
            poll_positions(storage, target_wallet, session=session)
        except Exception as exc:
            logger.warning("positions_poll_error err=%s", exc)
        time.sleep(poll_interval_s)
