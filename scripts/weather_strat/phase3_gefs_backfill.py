from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time as time_mod
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable

import requests
import sys
from dateutil import parser as date_parser
from zoneinfo import ZoneInfo

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.weather_strat import phase1_weather_pipeline as phase1

LOCAL_TZ = ZoneInfo("America/New_York")
logger = logging.getLogger("gefs_backfill")


@dataclass
class RunConfig:
    lead_hours: int
    run_time_utc: datetime


def _parse_date(value: str) -> date:
    return date_parser.parse(value).date()


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end date must be >= start date")
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def _latest_gefs_cycle(run_time_utc: datetime) -> datetime:
    cycle_hours = [0, 6, 12, 18]
    hour = max(h for h in cycle_hours if h <= run_time_utc.hour)
    return run_time_utc.replace(hour=hour, minute=0, second=0, microsecond=0)


def _run_time_for_target(target_date: date, lead_hours: int) -> datetime:
    target_start_et = datetime.combine(target_date, time.min, tzinfo=LOCAL_TZ)
    run_time_utc = (target_start_et - timedelta(hours=lead_hours)).astimezone(timezone.utc)
    return _latest_gefs_cycle(run_time_utc)


def _load_members() -> list[str]:
    env = os.getenv("GEFS_MEMBERS")
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]
    return ["gec00", "gep01"]


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ens_probabilities (
            run_time_utc TEXT NOT NULL,
            target_date_et TEXT NOT NULL,
            lead_bucket_hours INTEGER NOT NULL,
            strike REAL NOT NULL,
            p_raw REAL,
            members_present INTEGER,
            tmax_p10 REAL,
            tmax_p50 REAL,
            tmax_p90 REAL,
            updated_at_et TEXT,
            PRIMARY KEY (run_time_utc, target_date_et, lead_bucket_hours, strike)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ens_calibration_bins (
            lead_bucket_hours INTEGER NOT NULL,
            strike REAL NOT NULL,
            bin_lo REAL NOT NULL,
            bin_hi REAL NOT NULL,
            n INTEGER,
            event_rate REAL,
            updated_at_et TEXT,
            PRIMARY KEY (lead_bucket_hours, strike, bin_lo, bin_hi)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ens_calibration_apply (
            lead_bucket_hours INTEGER NOT NULL,
            strike REAL NOT NULL,
            p_raw REAL NOT NULL,
            p_cal REAL,
            PRIMARY KEY (lead_bucket_hours, strike, p_raw)
        )
        """
    )


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


def _latest_truth_by_date(conn: sqlite3.Connection) -> dict[str, float]:
    rows = conn.execute(
        """
        SELECT date_et, tmax_obs_f, report_issued_at_et
        FROM obs_daily
        WHERE tmax_obs_f IS NOT NULL
        """
    ).fetchall()
    truth: dict[str, tuple[float, str]] = {}
    for date_et, tmax, issued_at in rows:
        issued_key = issued_at or ""
        current = truth.get(date_et)
        if current is None or issued_key > current[1]:
            truth[date_et] = (float(tmax), issued_key)
    return {k: v[0] for k, v in truth.items()}


def _ens_members_for(conn: sqlite3.Connection, run_time_et: str, target_date_et: str) -> list[float]:
    rows = conn.execute(
        """
        SELECT tmax_member_f
        FROM fcst_ens_snapshots
        WHERE asof_time_et = ? AND target_date_et = ? AND tmax_member_f IS NOT NULL
        """,
        (run_time_et, target_date_et),
    ).fetchall()
    return [float(r[0]) for r in rows]


def backfill_gefs(
    conn: sqlite3.Connection,
    start_date: date,
    end_date: date,
    lead_hours: int,
    min_slices_ratio: float,
) -> None:
    session = requests.Session()
    members = _load_members()

    for d in _date_range(start_date, end_date):
        run_time_utc = _run_time_for_target(d, lead_hours)
        logger.info("Backfill target=%s run=%s lead=%dh", d.isoformat(), run_time_utc.isoformat(), lead_hours)
        for member in members:
            phase1.ingest_gefs_member(
                conn,
                session,
                member,
                run_time_utc,
                [d],
                phase1.DEFAULT_LAT,
                phase1.DEFAULT_LON,
                step_hours=6,
            )


def backfill_gefs_chunked(
    conn: sqlite3.Connection,
    start_date: date,
    end_date: date,
    lead_hours: int,
    chunk_size: int,
    pause_seconds: float,
) -> None:
    dates = _date_range(start_date, end_date)
    for i in range(0, len(dates), chunk_size):
        chunk = dates[i : i + chunk_size]
        if not chunk:
            continue
        logger.info(
            "Backfill chunk %s -> %s (%d days)",
            chunk[0].isoformat(),
            chunk[-1].isoformat(),
            len(chunk),
        )
        backfill_gefs(conn, chunk[0], chunk[-1], lead_hours, min_slices_ratio=0.5)
        metrics = phase1.gefs_metrics_snapshot()
        if metrics:
            logger.info("GEFS metrics so far: %s", json.dumps(metrics, sort_keys=True))
        if i + chunk_size < len(dates) and pause_seconds > 0:
            logger.info("Pausing %.1fs between chunks", pause_seconds)
            time_mod.sleep(pause_seconds)


def build_ens_probabilities(
    conn: sqlite3.Connection,
    start_date: date,
    end_date: date,
    lead_hours: int,
    strikes: list[float],
    min_members: int,
) -> None:
    _ensure_tables(conn)
    now = datetime.now(tz=LOCAL_TZ).isoformat()

    for d in _date_range(start_date, end_date):
        run_time_utc = _run_time_for_target(d, lead_hours)
        run_time_et = run_time_utc.astimezone(LOCAL_TZ).isoformat()
        values = _ens_members_for(conn, run_time_et, d.isoformat())
        if len(values) < min_members:
            continue
        p10 = _quantile(values, 0.10)
        p50 = _quantile(values, 0.50)
        p90 = _quantile(values, 0.90)
        for strike in strikes:
            p_raw = sum(1 for v in values if v > strike) / len(values)
            conn.execute(
                """
                INSERT OR REPLACE INTO ens_probabilities(
                    run_time_utc,
                    target_date_et,
                    lead_bucket_hours,
                    strike,
                    p_raw,
                    members_present,
                    tmax_p10,
                    tmax_p50,
                    tmax_p90,
                    updated_at_et
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_time_utc.isoformat(),
                    d.isoformat(),
                    lead_hours,
                    float(strike),
                    p_raw,
                    len(values),
                    p10,
                    p50,
                    p90,
                    now,
                ),
            )


def calibrate_ensemble(
    conn: sqlite3.Connection,
    lead_hours: int,
    strikes: list[float],
    bins: int,
) -> None:
    _ensure_tables(conn)
    truth = _latest_truth_by_date(conn)
    conn.execute("DELETE FROM ens_calibration_bins WHERE lead_bucket_hours = ?", (lead_hours,))
    conn.execute("DELETE FROM ens_calibration_apply WHERE lead_bucket_hours = ?", (lead_hours,))

    for strike in strikes:
        rows = conn.execute(
            """
            SELECT run_time_utc, target_date_et, p_raw
            FROM ens_probabilities
            WHERE lead_bucket_hours = ? AND strike = ? AND p_raw IS NOT NULL
            """,
            (lead_hours, float(strike)),
        ).fetchall()
        if not rows:
            continue
        # bucket by p_raw
        bins_edges = [i / bins for i in range(bins + 1)]
        bin_counts = [0] * bins
        bin_events = [0] * bins
        for _, target_date_et, p_raw in rows:
            y = truth.get(target_date_et)
            if y is None:
                continue
            event = 1 if y > strike else 0
            idx = min(bins - 1, int(p_raw * bins))
            bin_counts[idx] += 1
            bin_events[idx] += event
        now = datetime.now(tz=LOCAL_TZ).isoformat()
        for i in range(bins):
            n = bin_counts[i]
            if n == 0:
                continue
            bin_lo = bins_edges[i]
            bin_hi = bins_edges[i + 1]
            event_rate = bin_events[i] / n
            conn.execute(
                """
                INSERT OR REPLACE INTO ens_calibration_bins(
                    lead_bucket_hours, strike, bin_lo, bin_hi, n, event_rate, updated_at_et
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (lead_hours, float(strike), bin_lo, bin_hi, n, event_rate, now),
            )
            # store a simple mapping at the bin center
            p_raw_center = (bin_lo + bin_hi) / 2
            conn.execute(
                """
                INSERT OR REPLACE INTO ens_calibration_apply(
                    lead_bucket_hours, strike, p_raw, p_cal
                ) VALUES (?, ?, ?, ?)
                """,
                (lead_hours, float(strike), p_raw_center, event_rate),
            )


def main() -> None:
    logging.basicConfig(level=os.getenv("WEATHER_LOG_LEVEL", "INFO"))
    parser = argparse.ArgumentParser(description="GEFS-first probability engine")
    parser.add_argument("--db", default=os.getenv("WEATHER_DB_PATH", "data/weather_phase1.sqlite"))
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--lead-hours", type=int, default=48)
    parser.add_argument("--strike", action="append", type=float, default=[])
    parser.add_argument("--min-members", type=int, default=2)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=7, help="Chunk size in days for backfill")
    parser.add_argument("--chunk-pause", type=float, default=10.0, help="Pause seconds between chunks")
    parser.add_argument("--build-probs", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    strikes = args.strike or []

    with sqlite3.connect(args.db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        if args.backfill:
            backfill_gefs_chunked(
                conn,
                start_date,
                end_date,
                args.lead_hours,
                args.chunk_size,
                args.chunk_pause,
            )
        if args.build_probs and strikes:
            build_ens_probabilities(conn, start_date, end_date, args.lead_hours, strikes, args.min_members)
        if args.calibrate and strikes:
            calibrate_ensemble(conn, args.lead_hours, strikes, args.bins)


if __name__ == "__main__":
    main()
