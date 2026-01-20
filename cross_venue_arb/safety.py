"""Safety checks for cross-venue arb."""

from __future__ import annotations

import os


def assert_no_live_trading() -> None:
    allow = os.getenv("CROSS_VENUE_ARB_ALLOW_TRADING", "0").strip().lower()
    if allow in {"1", "true", "yes", "y"}:
        raise SystemExit(
            "Live trading is disabled for cross_venue_arb. "
            "Unset CROSS_VENUE_ARB_ALLOW_TRADING to continue."
        )
