import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from kalshi_fetcher.kalshi_client import request


INCENTIVES_ENDPOINT = "/incentive_programs"
DEFAULT_DB_PATH = "data/incentives.sqlite"

INCENTIVE_BEHAVIOR = {
    "liquidity": "resting liquidity rewarded",
    "volume": "executed volume rewarded",
}


def _utc_ms_now() -> int:
    return int(time.time() * 1000)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True, sort_keys=True)


def _parse_ts_to_utc_ms(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = int(value)
        if ts > 1_000_000_000_000:
            return ts
        if ts > 1_000_000_000:
            return ts * 1000
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return _parse_ts_to_utc_ms(int(stripped))
        try:
            dt = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
            return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            return None
    return None


def _hash_payload(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _extract_reward(program: Dict[str, Any]) -> Tuple[Optional[float], Optional[str], Optional[str], Optional[str]]:
    reward_raw = program.get("reward")
    reward_value = None
    reward_currency = None
    reward_units = None
    reward_text = None

    if isinstance(reward_raw, dict):
        reward_value = reward_raw.get("amount") or reward_raw.get("value")
        reward_currency = reward_raw.get("currency") or reward_raw.get("currency_code")
        reward_units = reward_raw.get("units") or reward_raw.get("unit")
        reward_text = reward_raw.get("text") or reward_raw.get("description")
    elif reward_raw is not None:
        reward_text = str(reward_raw)

    if reward_value is None:
        reward_value = program.get("reward_amount") or program.get("amount") or program.get("period_reward")

    if reward_text is None and reward_value is not None:
        reward_text = str(reward_value)

    try:
        reward_value = float(reward_value) if reward_value is not None else None
    except (TypeError, ValueError):
        reward_value = None

    return reward_value, reward_currency, reward_units, reward_text


def _extract_program_id(program: Dict[str, Any], raw_json: str) -> str:
    for key in ("program_id", "id", "incentive_program_id"):
        value = program.get(key)
        if value:
            return str(value)
    return f"hash_{_hash_payload(raw_json)}"


def normalize_program(program: Dict[str, Any]) -> Dict[str, Any]:
    raw_json = _json_dumps(program)
    program_id = _extract_program_id(program, raw_json)
    status = program.get("status") or program.get("state")
    incentive_type = program.get("incentive_type") or program.get("type") or program.get("program_type")
    market_ticker = program.get("market_ticker") or program.get("ticker")
    series_ticker = program.get("series_ticker")
    start_ts = _parse_ts_to_utc_ms(
        program.get("start_ts")
        or program.get("start_time")
        or program.get("start_time_ts")
        or program.get("start_time_ms")
        or program.get("start_date")
    )
    end_ts = _parse_ts_to_utc_ms(
        program.get("end_ts")
        or program.get("end_time")
        or program.get("end_time_ts")
        or program.get("end_time_ms")
        or program.get("end_date")
    )
    paid_out = program.get("paid_out")
    rules_text = program.get("rules_text") or program.get("rules") or program.get("terms")
    caps_text = program.get("caps") or program.get("caps_text")
    reward_value, reward_currency, reward_units, reward_text = _extract_reward(program)

    normalized = {
        "program_id": program_id,
        "status": status,
        "incentive_type": incentive_type,
        "market_ticker": market_ticker,
        "series_ticker": series_ticker,
        "paid_out": paid_out,
        "reward_value": reward_value,
        "reward_currency": reward_currency,
        "reward_units": reward_units,
        "reward_text": reward_text,
        "start_ts_utc_ms": start_ts,
        "end_ts_utc_ms": end_ts,
        "rules_text": rules_text,
        "caps_text": caps_text,
        "raw_json": raw_json,
    }
    return normalized


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS incentive_programs (
            program_id TEXT PRIMARY KEY,
            status TEXT,
            incentive_type TEXT,
            market_ticker TEXT,
            series_ticker TEXT,
            paid_out INTEGER,
            reward_value REAL,
            reward_currency TEXT,
            reward_units TEXT,
            reward_text TEXT,
            start_ts_utc_ms INTEGER,
            end_ts_utc_ms INTEGER,
            rules_text TEXT,
            caps_text TEXT,
            raw_json TEXT NOT NULL,
            raw_hash TEXT NOT NULL,
            first_seen_utc_ms INTEGER NOT NULL,
            last_seen_utc_ms INTEGER NOT NULL,
            last_changed_utc_ms INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_incentives_status ON incentive_programs (status);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_incentives_market ON incentive_programs (market_ticker);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_incentives_series ON incentive_programs (series_ticker);"
    )
    _ensure_column(conn, "incentive_programs", "paid_out", "INTEGER")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS incentive_responses (
            id INTEGER PRIMARY KEY,
            fetched_at_utc_ms INTEGER NOT NULL,
            cursor TEXT,
            response_json TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS incentive_sync_state (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )
    conn.execute("DROP VIEW IF EXISTS incentives_active;")
    conn.execute(
        """
        CREATE VIEW incentives_active AS
        SELECT *
        FROM incentive_programs
        WHERE (start_ts_utc_ms IS NULL OR start_ts_utc_ms <= (strftime('%s','now') * 1000))
          AND (end_ts_utc_ms IS NULL OR end_ts_utc_ms >= (strftime('%s','now') * 1000))
          AND (paid_out IS NULL OR paid_out = 0);
        """
    )
    conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_def: str) -> None:
    existing_cols = {
        row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing_cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")


def _fetch_incentive_pages(limit: int = 200) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    programs: list[Dict[str, Any]] = []
    responses: list[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        params: Dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        data = request(INCENTIVES_ENDPOINT, params=params)
        responses.append(data)
        page_programs = data.get("incentive_programs") or data.get("programs") or data.get("data") or []
        if not isinstance(page_programs, list):
            page_programs = []
        programs.extend(page_programs)
        cursor = data.get("cursor") or data.get("next_cursor")
        if not cursor:
            break
    return programs, responses


def _log_event(event: str, **fields: Any) -> None:
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[INCENTIVES] {event} {payload}".strip())


def _set_state(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        """
        INSERT INTO incentive_sync_state (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, str(value)),
    )


def _get_state(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM incentive_sync_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def sync_incentive_programs(db_path: str = DEFAULT_DB_PATH, limit: int = 200) -> Dict[str, Any]:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    try:
        _ensure_schema(conn)
        now_ms = _utc_ms_now()
        _set_state(conn, "last_poll_utc_ms", now_ms)
        programs, responses = _fetch_incentive_pages(limit=limit)

        for response in responses:
            conn.execute(
                """
                INSERT INTO incentive_responses (fetched_at_utc_ms, cursor, response_json)
                VALUES (?, ?, ?)
                """,
                (
                    now_ms,
                    response.get("cursor"),
                    _json_dumps(response),
                ),
            )

        seen_ids: set[str] = set()
        added = 0
        changed = 0

        for program in programs:
            if not isinstance(program, dict):
                continue
            normalized = normalize_program(program)
            program_id = normalized["program_id"]
            raw_hash = _hash_payload(normalized["raw_json"])
            seen_ids.add(program_id)

            existing = conn.execute(
                "SELECT raw_hash, last_changed_utc_ms FROM incentive_programs WHERE program_id = ?",
                (program_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO incentive_programs (
                        program_id, status, incentive_type, market_ticker, series_ticker,
                        paid_out, reward_value, reward_currency, reward_units, reward_text,
                        start_ts_utc_ms, end_ts_utc_ms, rules_text, caps_text,
                        raw_json, raw_hash, first_seen_utc_ms, last_seen_utc_ms, last_changed_utc_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        program_id,
                        normalized["status"],
                        normalized["incentive_type"],
                        normalized["market_ticker"],
                        normalized["series_ticker"],
                        normalized["paid_out"],
                        normalized["reward_value"],
                        normalized["reward_currency"],
                        normalized["reward_units"],
                        normalized["reward_text"],
                        normalized["start_ts_utc_ms"],
                        normalized["end_ts_utc_ms"],
                        normalized["rules_text"],
                        normalized["caps_text"],
                        normalized["raw_json"],
                        raw_hash,
                        now_ms,
                        now_ms,
                        now_ms,
                    ),
                )
                added += 1
                _log_event("INCENTIVE_ADDED", program_id=program_id, status=normalized["status"])
            else:
                prior_hash = existing[0]
                prior_changed_ms = existing[1] or now_ms
                update_changed = prior_hash != raw_hash
                last_changed_ms = now_ms if update_changed else prior_changed_ms
                if update_changed:
                    changed += 1
                    _log_event("INCENTIVE_CHANGED", program_id=program_id, status=normalized["status"])
                conn.execute(
                    """
                    UPDATE incentive_programs
                    SET status = ?,
                        incentive_type = ?,
                        market_ticker = ?,
                        series_ticker = ?,
                        paid_out = ?,
                        reward_value = ?,
                        reward_currency = ?,
                        reward_units = ?,
                        reward_text = ?,
                        start_ts_utc_ms = ?,
                        end_ts_utc_ms = ?,
                        rules_text = ?,
                        caps_text = ?,
                        raw_json = ?,
                        raw_hash = ?,
                        last_seen_utc_ms = ?,
                        last_changed_utc_ms = ?
                    WHERE program_id = ?
                    """,
                    (
                        normalized["status"],
                        normalized["incentive_type"],
                        normalized["market_ticker"],
                        normalized["series_ticker"],
                        normalized["paid_out"],
                        normalized["reward_value"],
                        normalized["reward_currency"],
                        normalized["reward_units"],
                        normalized["reward_text"],
                        normalized["start_ts_utc_ms"],
                        normalized["end_ts_utc_ms"],
                        normalized["rules_text"],
                        normalized["caps_text"],
                        normalized["raw_json"],
                        raw_hash,
                        now_ms,
                        last_changed_ms,
                        program_id,
                    ),
                )

        if seen_ids:
            placeholders = ",".join("?" for _ in seen_ids)
            missing_rows = conn.execute(
                f"""
                SELECT program_id FROM incentive_programs
                WHERE program_id NOT IN ({placeholders})
                  AND status != 'inactive'
                """,
                tuple(seen_ids),
            ).fetchall()
        else:
            missing_rows = conn.execute(
                "SELECT program_id FROM incentive_programs WHERE status != 'inactive'"
            ).fetchall()

        removed = 0
        for (program_id,) in missing_rows:
            conn.execute(
                """
                UPDATE incentive_programs
                SET status = 'inactive',
                    last_changed_utc_ms = ?
                WHERE program_id = ?
                """,
                (now_ms, program_id),
            )
            removed += 1
            _log_event("INCENTIVE_REMOVED", program_id=program_id)

        _set_state(conn, "last_ok_utc_ms", now_ms)
        conn.commit()

        return {
            "ok": True,
            "programs": len(programs),
            "added": added,
            "changed": changed,
            "removed": removed,
            "last_poll_utc_ms": now_ms,
        }
    finally:
        conn.close()


def get_active_incentives(db_path: str = DEFAULT_DB_PATH) -> list[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT program_id, status, incentive_type, market_ticker, series_ticker,
                   paid_out,
                   reward_value, reward_currency, reward_units, reward_text,
                   start_ts_utc_ms, end_ts_utc_ms, rules_text, caps_text
            FROM incentives_active
            ORDER BY incentive_type, market_ticker, series_ticker
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _format_ts(ms: Optional[int]) -> str:
    if ms is None:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def render_active_table(active_rows: Iterable[Dict[str, Any]]) -> str:
    lines = ["market_ticker | series_ticker | incentive_type | reward | start | end"]
    for row in active_rows:
        reward = row.get("reward_text") or row.get("reward_value")
        line = " | ".join(
            [
                str(row.get("market_ticker") or ""),
                str(row.get("series_ticker") or ""),
                str(row.get("incentive_type") or ""),
                str(reward or ""),
                _format_ts(row.get("start_ts_utc_ms")),
                _format_ts(row.get("end_ts_utc_ms")),
            ]
        )
        lines.append(line)
    return "\n".join(lines)


def build_incentive_explainers(active_rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    explainers: Dict[str, Dict[str, Any]] = {}
    for row in active_rows:
        program_id = row.get("program_id")
        if not program_id:
            continue
        incentive_type = (row.get("incentive_type") or "").lower()
        explainers[str(program_id)] = {
            "incentive_type": row.get("incentive_type"),
            "rewarded_behavior": INCENTIVE_BEHAVIOR.get(incentive_type, "unknown"),
            "how_rewards_stop": "end time or program updated",
            "caps_or_edge_cases": row.get("caps_text"),
        }
    return explainers


def incentives_fresh(db_path: str = DEFAULT_DB_PATH, max_age_minutes: int = 5) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        last_poll = _get_state(conn, "last_poll_utc_ms")
        if not last_poll:
            return False
        age_ms = _utc_ms_now() - int(last_poll)
        return age_ms <= max_age_minutes * 60 * 1000
    finally:
        conn.close()


if __name__ == "__main__":
    result = sync_incentive_programs()
    print(result)
    active = get_active_incentives()
    print(render_active_table(active))
