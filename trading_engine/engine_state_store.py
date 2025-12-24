"""
Simple persistence for engine control state.

Stores a minimal snapshot in SQLite so the engine can resume after restarts.
This module is intentionally standalone (not yet wired into the engine).
"""

from __future__ import annotations

import json
import os
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

# SQLite file colocated with other data artifacts.
DB_PATH = Path(
    os.getenv(
        "ENGINE_STATE_DB_PATH",
        Path(__file__).resolve().parent.parent / "data" / "engine_state.sqlite",
    )
)


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS engine_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    snapshot TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


DEFAULT_STATE: Dict[str, Any] = {
    "attached_markets": [],
    "retired_markets": [],
    "operator_flags": {
        "pause_entries": False,
        "pause_all": False,
    },
    # Optional snapshot of last-used strategy config (JSON-serializable dict)
    "strategy_config": None,
    # Optional trading state (paper only)
    "positions": [],
    "capital": {"total": 0.0, "used": 0.0},
    "cooldown_until": None,
}


def _ensure_db() -> sqlite3.Connection:
    """Open the SQLite DB and ensure the table exists."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(CREATE_TABLE_SQL)
    return conn


def _normalize_snapshot(snapshot: Dict[str, Any] | Any) -> Dict[str, Any]:
    """
    Coerce arbitrary input into the minimal persisted schema.
    Only keeps non-sensitive control state relevant for restart/resume.
    """
    normalized = deepcopy(DEFAULT_STATE)

    if not isinstance(snapshot, dict):
        return normalized

    normalized["positions"] = _normalize_positions(snapshot.get("positions"))
    normalized["capital"] = _normalize_capital(snapshot.get("capital"))
    normalized["cooldown_until"] = _normalize_iso_dt(snapshot.get("cooldown_until"))

    attached = snapshot.get("attached_markets")
    if isinstance(attached, (list, tuple, set)):
        normalized["attached_markets"] = [str(m).upper() for m in attached if m]

    retired = snapshot.get("retired_markets")
    if isinstance(retired, (list, tuple, set)):
        # Store as a sorted list for stable JSON output.
        normalized["retired_markets"] = sorted({str(m).upper() for m in retired if m})

    flags = snapshot.get("operator_flags")
    if isinstance(flags, dict):
        normalized["operator_flags"]["pause_entries"] = bool(
            flags.get("pause_entries", normalized["operator_flags"]["pause_entries"])
        )
        normalized["operator_flags"]["pause_all"] = bool(
            flags.get("pause_all", normalized["operator_flags"]["pause_all"])
        )

    if "strategy_config" in snapshot:
        # Strategy config is treated as opaque JSON-serializable data.
        normalized["strategy_config"] = snapshot.get("strategy_config")

    return normalized


def _normalize_positions(raw_positions: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_positions, (list, tuple)):
        return []

    cleaned: list[dict[str, Any]] = []
    for pos in raw_positions:
        if not isinstance(pos, dict):
            continue

        def _dt(val: Any) -> Any:
            # Convert datetimes to ISO strings; leave other types as-is.
            try:
                from datetime import datetime

                if isinstance(val, datetime):
                    return val.isoformat()
            except Exception:
                pass
            return val

        cleaned.append(
            {
                "id": pos.get("id"),
                "event_ticker": (pos.get("event_ticker") or "").upper(),
                "market_ticker": (pos.get("market_ticker") or "").upper(),
                "side": (pos.get("side") or "").upper(),
                "qty": pos.get("qty"),
                "entry_price": pos.get("entry_price"),
                "current_price": pos.get("current_price"),
                "status": pos.get("status"),
                "entry_ts": _dt(pos.get("entry_ts")),
                "exit_ts": _dt(pos.get("exit_ts")),
                "realized_pnl": pos.get("realized_pnl"),
                "unrealized_pnl": pos.get("unrealized_pnl"),
            }
        )

    return cleaned


def _normalize_capital(raw_capital: Any) -> dict[str, float]:
    default_cap = deepcopy(DEFAULT_STATE["capital"])
    if not isinstance(raw_capital, dict):
        return default_cap

    total = raw_capital.get("total", default_cap["total"])
    used = raw_capital.get("used", default_cap["used"])

    try:
        total_f = float(total)
    except Exception:
        total_f = default_cap["total"]

    try:
        used_f = float(used)
    except Exception:
        used_f = default_cap["used"]

    return {"total": total_f, "used": used_f}


def _normalize_iso_dt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        from datetime import datetime

        if isinstance(value, datetime):
            return value.isoformat()
    except Exception:
        pass
    return None


def save_state(snapshot: Dict[str, Any]) -> None:
    """
    Persist the given snapshot. Non-serializable payloads are caller's responsibility.
    """
    normalized = _normalize_snapshot(snapshot)
    payload = json.dumps(normalized)

    conn = _ensure_db()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO engine_state (id, snapshot, updated_at)
                VALUES (1, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    snapshot=excluded.snapshot,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (payload,),
            )
    finally:
        conn.close()


def load_state() -> Dict[str, Any]:
    """
    Load the last saved snapshot.
    Returns defaults if no snapshot exists or if deserialization fails.
    """
    try:
        conn = _ensure_db()
        try:
            row = conn.execute(
                "SELECT snapshot FROM engine_state WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()
    except Exception:
        row = None

    if not row:
        return deepcopy(DEFAULT_STATE)

    raw = row[0] if isinstance(row, (list, tuple)) else None
    if not isinstance(raw, str):
        return deepcopy(DEFAULT_STATE)

    try:
        data = json.loads(raw)
    except Exception:
        return deepcopy(DEFAULT_STATE)

    return _normalize_snapshot(data)
