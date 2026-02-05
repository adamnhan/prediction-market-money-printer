from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

PHASE2_DIR = Path(__file__).resolve().parent
if str(PHASE2_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE2_DIR))

from backfill import backfill_missing
from cache import Phase2Cache
from resolver import resolve_condition


logger = logging.getLogger("copy_trade_phase2")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[PHASE2] %(asctime)s %(message)s"))
    logger.addHandler(handler)


def _poll_live(cache: Phase2Cache, poll_interval_s: float) -> None:
    last_id = cache.get_state("phase2_last_trade_id", 0) or 0
    while True:
        condition_ids, max_id = cache.iter_new_condition_ids_since(int(last_id))
        if condition_ids:
            logger.info("live_scan new_trade_ids=%d new_conditions=%d", max_id - last_id, len(condition_ids))
        resolved = 0
        for condition_id in condition_ids:
            meta = resolve_condition(cache, condition_id)
            if meta:
                resolved += 1
        if max_id != last_id:
            cache.set_state("phase2_last_trade_id", max_id)
        if condition_ids:
            logger.info("live_resolve resolved=%d total=%d", resolved, len(condition_ids))
        time.sleep(poll_interval_s)


def main() -> None:
    db_path = os.getenv("COPY_TRADE_DB_PATH", "data/copy_trade_phase1.sqlite")
    mode = os.getenv("COPY_TRADE_PHASE2_MODE", "both").lower()
    poll_interval_s = float(os.getenv("COPY_TRADE_PHASE2_POLL_S", "5"))

    logger.info("phase2_start db=%s mode=%s poll_s=%.1f", db_path, mode, poll_interval_s)
    cache = Phase2Cache(db_path)

    if mode in {"backfill", "both"}:
        backfill_missing(cache)

    if mode in {"live", "both"}:
        _poll_live(cache, poll_interval_s)


if __name__ == "__main__":
    main()
