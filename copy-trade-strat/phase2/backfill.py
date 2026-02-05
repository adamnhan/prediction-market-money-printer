from __future__ import annotations

import logging
import time

import requests

from cache import Phase2Cache
from resolver import resolve_condition


logger = logging.getLogger("copy_trade_phase2")


def backfill_missing(cache: Phase2Cache, sleep_s: float = 0.25) -> None:
    missing = cache.iter_missing_condition_ids()
    if not missing:
        logger.info("backfill_missing none")
        return
    logger.info("backfill_missing count=%d", len(missing))
    session = requests.Session()
    resolved = 0
    for condition_id in missing:
        meta = resolve_condition(cache, condition_id, session=session)
        if meta:
            resolved += 1
        time.sleep(sleep_s)
    logger.info("backfill_done resolved=%d total=%d", resolved, len(missing))
