"""SQLite helpers for cross-venue arb."""

from __future__ import annotations

import sqlite3

from cross_venue_arb.config import CONFIG


def connect() -> sqlite3.Connection:
    return sqlite3.connect(CONFIG.db_path)
