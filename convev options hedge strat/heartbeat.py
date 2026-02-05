#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class HeartbeatState:
    start_ts: float = field(default_factory=time.time)
    signals: int = 0
    bundles: int = 0
    filled: int = 0
    rejected: int = 0
    skipped: int = 0
    errors: Dict[str, int] = field(default_factory=dict)
    last_data_ts: float | None = None

    def record_error(self, key: str) -> None:
        self.errors[key] = self.errors.get(key, 0) + 1

    def snapshot(self) -> dict:
        return {
            "uptime_s": int(time.time() - self.start_ts),
            "signals": self.signals,
            "bundles": self.bundles,
            "filled": self.filled,
            "rejected": self.rejected,
            "skipped": self.skipped,
            "errors": self.errors,
            "last_data_ts": self.last_data_ts,
        }

