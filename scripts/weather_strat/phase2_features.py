from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Iterable

from dateutil import parser as date_parser
from zoneinfo import ZoneInfo


LOCAL_TZ = ZoneInfo("America/New_York")
LEAD_BUCKETS = [12, 24, 48, 72, 96]

logger = logging.getLogger("weather_phase2")


@dataclass
class PointDailyMetrics:
    tmax: float | None
    hours_present: int
    coverage_ratio: float | None
    first_valid_time_et: str | None
    last_valid_time_et: str | None


def _to_et(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=LOCAL_TZ)
    return dt.astimezone(LOCAL_TZ)


def _parse_dt(value: str) -> datetime:
    dt = date_parser.parse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    return dt


def _lead_hours(asof_time_et: datetime, target_date_et: date) -> float:
    target_start = datetime.combine(target_date_et, time.min, tzinfo=LOCAL_TZ)
    return (target_start - asof_time_et).total_seconds() / 3600.0


def _lead_bucket(lead_hours: float) -> int | None:
    bucket = None
    for candidate in LEAD_BUCKETS:
        if lead_hours >= candidate:
            bucket = candidate
    return bucket


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] + (values[hi] - values[lo]) * frac


def _load_raw_json(conn: sqlite3.Connection, raw_hash: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT payload FROM raw_payloads WHERE hash = ?", (raw_hash,)).fetchone()
    if not row:
        return None
    payload = row[0]
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _build_point_metrics(periods: list[dict], target_date_et: date) -> PointDailyMetrics:
    temps: list[float] = []
    hours_present = 0
    first_valid: datetime | None = None
    last_valid: datetime | None = None

    for period in periods:
        start_time = period.get("startTime")
        if not start_time:
            continue
        try:
            dt = _parse_dt(start_time)
        except (ValueError, TypeError):
            continue
        dt_et = _to_et(dt)
        if dt_et.date() != target_date_et:
            continue
        temp = period.get("temperature")
        if temp is None:
            continue
        try:
            temp_f = float(temp)
        except (ValueError, TypeError):
            continue
        hours_present += 1
        temps.append(temp_f)
        if first_valid is None or dt_et < first_valid:
            first_valid = dt_et
        if last_valid is None or dt_et > last_valid:
            last_valid = dt_et

    tmax = max(temps) if temps else None
    coverage_ratio = (hours_present / 24.0) if hours_present else None
    return PointDailyMetrics(
        tmax=tmax,
        hours_present=hours_present,
        coverage_ratio=coverage_ratio,
        first_valid_time_et=first_valid.isoformat() if first_valid else None,
        last_valid_time_et=last_valid.isoformat() if last_valid else None,
    )


def init_feature_tables(conn: sqlite3.Connection) -> None:
    _ensure_source_columns(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS point_daily_features (
            asof_time_et TEXT NOT NULL,
            target_date_et TEXT NOT NULL,
            lead_hours_raw REAL NOT NULL,
            lead_bucket_hours INTEGER,
            tmax_point_f REAL,
            coverage_ratio REAL,
            hours_present INTEGER,
            first_valid_time_et TEXT,
            last_valid_time_et TEXT,
            grid_id TEXT,
            grid_x INTEGER,
            grid_y INTEGER,
            raw_json_hash TEXT,
            PRIMARY KEY (asof_time_et, target_date_et)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ens_daily_features (
            asof_time_et TEXT NOT NULL,
            target_date_et TEXT NOT NULL,
            lead_hours_raw REAL NOT NULL,
            lead_bucket_hours INTEGER,
            member_id TEXT NOT NULL,
            tmax_member_f REAL,
            slice_count INTEGER,
            slice_coverage_ratio REAL,
            raw_grib_subset_hash TEXT,
            PRIMARY KEY (asof_time_et, target_date_et, member_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ens_daily_agg (
            asof_time_et TEXT NOT NULL,
            target_date_et TEXT NOT NULL,
            members_present INTEGER,
            tmax_p10 REAL,
            tmax_p50 REAL,
            tmax_p90 REAL,
            PRIMARY KEY (asof_time_et, target_date_et)
        )
        """
    )


def _ensure_source_columns(conn: sqlite3.Connection) -> None:
    def _add_column(table: str, column: str, col_type: str) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    _add_column("fcst_ens_snapshots", "slice_count", "INTEGER")
    _add_column("fcst_ens_snapshots", "slice_coverage_ratio", "REAL")


def build_point_features(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT DISTINCT asof_time_et, raw_json_hash, grid_id, grid_x, grid_y
        FROM fcst_point_snapshots
        WHERE raw_json_hash IS NOT NULL
        """
    ).fetchall()
    inserted = 0
    for asof_time_et, raw_hash, grid_id, grid_x, grid_y in rows:
        payload = _load_raw_json(conn, raw_hash)
        if not payload:
            logger.warning("Missing raw payload for hash=%s", raw_hash)
            continue
        periods = payload.get("properties", {}).get("periods", [])
        targets = conn.execute(
            """
            SELECT target_date_et
            FROM fcst_point_snapshots
            WHERE asof_time_et = ?
            """,
            (asof_time_et,),
        ).fetchall()
        asof_dt = _parse_dt(asof_time_et)
        for (target_date_str,) in targets:
            target_date = date_parser.parse(target_date_str).date()
            metrics = _build_point_metrics(periods, target_date)
            lead_hours_raw = _lead_hours(asof_dt, target_date)
            lead_bucket = _lead_bucket(lead_hours_raw)
            conn.execute(
                """
                INSERT OR REPLACE INTO point_daily_features(
                    asof_time_et,
                    target_date_et,
                    lead_hours_raw,
                    lead_bucket_hours,
                    tmax_point_f,
                    coverage_ratio,
                    hours_present,
                    first_valid_time_et,
                    last_valid_time_et,
                    grid_id,
                    grid_x,
                    grid_y,
                    raw_json_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asof_time_et,
                    target_date_str,
                    lead_hours_raw,
                    lead_bucket,
                    metrics.tmax,
                    metrics.coverage_ratio,
                    metrics.hours_present,
                    metrics.first_valid_time_et,
                    metrics.last_valid_time_et,
                    grid_id,
                    grid_x,
                    grid_y,
                    raw_hash,
                ),
            )
            inserted += 1
    logger.info("Built point_daily_features rows=%d", inserted)
    return inserted


def build_ens_features(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT asof_time_et, target_date_et, member_id, tmax_member_f, slice_count, slice_coverage_ratio, raw_grib_subset_hash
        FROM fcst_ens_snapshots
        """
    ).fetchall()
    inserted = 0
    for asof_time_et, target_date_et, member_id, tmax, slice_count, slice_cov, raw_hash in rows:
        asof_dt = _parse_dt(asof_time_et)
        target_date = date_parser.parse(target_date_et).date()
        lead_hours_raw = _lead_hours(asof_dt, target_date)
        lead_bucket = _lead_bucket(lead_hours_raw)
        conn.execute(
            """
            INSERT OR REPLACE INTO ens_daily_features(
                asof_time_et,
                target_date_et,
                lead_hours_raw,
                lead_bucket_hours,
                member_id,
                tmax_member_f,
                slice_count,
                slice_coverage_ratio,
                raw_grib_subset_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asof_time_et,
                target_date_et,
                lead_hours_raw,
                lead_bucket,
                member_id,
                tmax,
                slice_count,
                slice_cov,
                raw_hash,
            ),
        )
        inserted += 1
    logger.info("Built ens_daily_features rows=%d", inserted)
    return inserted


def build_ens_agg(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT asof_time_et, target_date_et, tmax_member_f
        FROM ens_daily_features
        WHERE tmax_member_f IS NOT NULL
        """
    ).fetchall()
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for asof_time_et, target_date_et, tmax in rows:
        grouped[(asof_time_et, target_date_et)].append(float(tmax))

    inserted = 0
    for (asof_time_et, target_date_et), values in grouped.items():
        conn.execute(
            """
            INSERT OR REPLACE INTO ens_daily_agg(
                asof_time_et,
                target_date_et,
                members_present,
                tmax_p10,
                tmax_p50,
                tmax_p90
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                asof_time_et,
                target_date_et,
                len(values),
                _quantile(values, 0.10),
                _quantile(values, 0.50),
                _quantile(values, 0.90),
            ),
        )
        inserted += 1
    logger.info("Built ens_daily_agg rows=%d", inserted)
    return inserted


def main() -> None:
    logging.basicConfig(level=os.getenv("WEATHER_LOG_LEVEL", "INFO"))
    parser = argparse.ArgumentParser(description="Phase 2 feature construction")
    parser.add_argument("--db", default=os.getenv("WEATHER_DB_PATH", "data/weather_phase1.sqlite"))
    parser.add_argument("--build-ens-agg", action="store_true", help="Also build ens_daily_agg")
    args = parser.parse_args()

    with sqlite3.connect(args.db) as conn:
        init_feature_tables(conn)
        build_point_features(conn)
        build_ens_features(conn)
        if args.build_ens_agg:
            build_ens_agg(conn)


if __name__ == "__main__":
    main()
