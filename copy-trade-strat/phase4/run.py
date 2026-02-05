from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

PHASE4_DIR = Path(__file__).resolve().parent
if str(PHASE4_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE4_DIR))

from clob_client import ClobWrapper, RestOrderbookClient, load_config_from_env
from engine import process_intent
from storage import Phase4Store


logger = logging.getLogger("copy_trade_phase4")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[PHASE4] %(asctime)s %(message)s"))
    logger.addHandler(handler)


def main() -> None:
    db_path = os.getenv("COPY_TRADE_DB_PATH", "data/copy_trade_phase1.sqlite")
    poll_s = float(os.getenv("COPY_TRADE_PHASE4_POLL_S", "2"))
    ttl_s = float(os.getenv("COPY_TRADE_PHASE4_TTL_S", "3"))
    max_spread_cents = int(os.getenv("COPY_TRADE_MAX_SPREAD_CENTS", "2"))
    dry_run = os.getenv("COPY_TRADE_DRY_RUN", "true").lower() == "true"
    killswitch = os.getenv("COPY_TRADE_KILLSWITCH", "true").lower() == "true"

    logger.info(
        "phase4_start db=%s poll_s=%.1f ttl_s=%.1f dry_run=%s killswitch=%s",
        db_path,
        poll_s,
        ttl_s,
        dry_run,
        killswitch,
    )

    store = Phase4Store(db_path)
    try:
        client = ClobWrapper(load_config_from_env())
    except Exception as exc:
        if dry_run or killswitch:
            logger.warning("clob_client_unavailable fallback=rest_readonly err=%s", exc)
            client = RestOrderbookClient()
        else:
            raise

    while True:
        intents = store.fetch_pending_intents(limit=25)
        for intent in intents:
            status, reason = process_intent(
                store,
                client,
                intent,
                max_spread_cents=max_spread_cents,
                ttl_seconds=ttl_s,
                dry_run=dry_run,
                killswitch=killswitch,
            )
            store.update_intent_status(intent["id"], status, reason)
            if status in {"SKIPPED_BOOKCHECK", "FAILED"}:
                logger.info("intent=%s reason=%s id=%s", status, reason, intent["id"])
            else:
                logger.info("intent=%s id=%s", status, intent["id"])
        time.sleep(poll_s)


if __name__ == "__main__":
    main()
