from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

PHASE3_DIR = Path(__file__).resolve().parent
if str(PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE3_DIR))

from config import Phase3Config, load_phase0_config
from engine import decide_trade
from storage import Phase3Store


logger = logging.getLogger("copy_trade_phase3")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[PHASE3] %(asctime)s %(message)s"))
    logger.addHandler(handler)


def _load_config() -> Phase3Config:
    phase0_path = os.getenv("COPY_TRADE_PHASE0_CONFIG", "copy-trade-strat/phase0_config.yaml")
    if Path(phase0_path).exists():
        cfg = load_phase0_config(phase0_path)
    else:
        cfg = Phase3Config()
    return cfg


def _log_intent(intent: dict) -> None:
    if intent.get("intent_status") == "SKIPPED":
        logger.info(
            "skip=%s trade_id=%s market=%s",
            intent.get("skip_reason"),
            intent.get("observed_trade_id"),
            intent.get("condition_id"),
        )
    else:
        logger.info(
            "intent=%s my_size=%.2f cap=%.4f market=%s",
            intent.get("intent_status"),
            intent.get("my_size"),
            intent.get("my_limit_price"),
            intent.get("condition_id"),
        )


def main() -> None:
    db_path = os.getenv("COPY_TRADE_DB_PATH", "data/copy_trade_phase1.sqlite")
    poll_s = float(os.getenv("COPY_TRADE_PHASE3_POLL_S", "2"))
    cfg = _load_config()

    logger.info(
        "phase3_start db=%s poll_s=%.1f fixed_copy_size_shares=%.2f",
        db_path,
        poll_s,
        cfg.fixed_copy_size_shares,
    )

    store = Phase3Store(db_path)

    while True:
        trades = store.fetch_unprocessed_trades(limit=200)
        for trade in trades:
            intent = decide_trade(store, trade, cfg)
            inserted = store.insert_intent(intent)
            if inserted:
                _log_intent(intent)
        time.sleep(poll_s)


if __name__ == "__main__":
    main()
