from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from storage import Storage


PHASE0_CONFIG_DEFAULT = "copy-trade-strat/phase0_config.yaml"


def _read_target_wallet_from_phase0(path: Path) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("target_wallet:"):
            value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None


def load_target_wallet(config_path: str | None = None) -> str | None:
    env_value = os.getenv("COPY_TRADE_TARGET_WALLET")
    if env_value:
        return env_value
    phase0_path = Path(config_path or PHASE0_CONFIG_DEFAULT)
    return _read_target_wallet_from_phase0(phase0_path)


def set_state(storage: Storage, key: str, value: Any) -> None:
    storage.set_state(key, value)


def get_state(storage: Storage, key: str, default: Any | None = None) -> Any:
    return storage.get_state(key, default)
