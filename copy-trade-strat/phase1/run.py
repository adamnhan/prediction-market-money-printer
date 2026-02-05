from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

PHASE1_DIR = Path(__file__).resolve().parent
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from ingestor_positions import poll_positions_loop
from ingestor_trades import poll_trades_loop
from state import PHASE0_CONFIG_DEFAULT, load_target_wallet
from storage import Storage


logger = logging.getLogger("copy_trade_phase1")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[PHASE1] %(asctime)s %(message)s"))
    logger.addHandler(handler)


def _resolve_target_wallet() -> str:
    phase0_path = os.getenv("COPY_TRADE_PHASE0_CONFIG", PHASE0_CONFIG_DEFAULT)
    wallet = load_target_wallet(phase0_path)
    if not wallet:
        raise RuntimeError(
            "missing target wallet: set COPY_TRADE_TARGET_WALLET or update phase0_config.yaml"
        )
    if "..." in wallet:
        logger.warning("target_wallet looks redacted; update for live data access")
    return wallet


def main() -> None:
    db_path = os.getenv("COPY_TRADE_DB_PATH", "data/copy_trade_phase1.sqlite")
    poll_trades_s = float(os.getenv("COPY_TRADE_TRADES_POLL_S", "2"))
    poll_positions_s = float(os.getenv("COPY_TRADE_POSITIONS_POLL_S", "45"))
    target_wallet = _resolve_target_wallet()

    logger.info(
        "phase1_start target_wallet=%s db=%s trades_poll_s=%.1f positions_poll_s=%.1f",
        target_wallet,
        db_path,
        poll_trades_s,
        poll_positions_s,
    )

    storage = Storage(db_path)

    trades_thread = threading.Thread(
        target=poll_trades_loop,
        args=(storage, target_wallet, poll_trades_s),
        daemon=True,
    )
    positions_thread = threading.Thread(
        target=poll_positions_loop,
        args=(storage, target_wallet, poll_positions_s),
        daemon=True,
    )
    trades_thread.start()
    positions_thread.start()

    trades_thread.join()
    positions_thread.join()


if __name__ == "__main__":
    main()
