from __future__ import annotations

import argparse
import logging
import os
import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable

import requests
from dateutil import parser as date_parser
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.weather_strat import phase1_weather_pipeline as phase1
from scripts.weather_strat import phase2_features as phase2
from scripts.weather_strat import phase3_modeling as phase3


LOCAL_TZ = ZoneInfo("America/New_York")
logger = logging.getLogger("weather_pipeline")


def _parse_date(value: str) -> date:
    return date_parser.parse(value).date()


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end date must be >= start date")
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def _now_et() -> datetime:
    return datetime.now(tz=LOCAL_TZ)


def _latest_gefs_run(now_utc: datetime) -> datetime:
    cycle_hours = [0, 6, 12, 18]
    hour = max(h for h in cycle_hours if h <= now_utc.hour)
    return now_utc.replace(hour=hour, minute=0, second=0, microsecond=0)


def _load_members() -> list[str]:
    env = os.getenv("GEFS_MEMBERS")
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]
    return ["gec00", "gep01"]


def _targets_for_range(start: date, end: date, asof: date) -> list[date]:
    targets = [d for d in _date_range(start, end) if d >= asof]
    return targets


def run_pipeline(
    db_path: str,
    start_date: date,
    end_date: date,
    run_point: bool,
    run_gefs: bool,
    run_features: bool,
    run_phase3: bool,
    gefs_run_times: list[datetime],
    min_coverage: float,
    min_members: int,
    strikes: list[float],
) -> None:
    phase1.init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        session = requests.Session()

        # Truth for the requested date range
        for d in _date_range(start_date, end_date):
            phase1.ingest_truth(conn, d, phase1.DEFAULT_STATION, session, limit=500)

        asof_time = _now_et()
        today = asof_time.date()
        targets = _targets_for_range(start_date, end_date, today)

        if run_point and targets:
            phase1.ingest_point_forecasts(
                conn,
                asof_time,
                targets,
                phase1.DEFAULT_LAT,
                phase1.DEFAULT_LON,
                session,
            )

        if run_gefs and targets:
            members = _load_members()
            if not gefs_run_times:
                gefs_run_times = [_latest_gefs_run(datetime.now(timezone.utc))]
            for run_time in gefs_run_times:
                for member in members:
                    phase1.ingest_gefs_member(
                        conn,
                        session,
                        member,
                        run_time,
                        targets,
                        phase1.DEFAULT_LAT,
                        phase1.DEFAULT_LON,
                        step_hours=6,
                    )

        if run_features:
            phase2.init_feature_tables(conn)
            phase2.build_point_features(conn)
            phase2.build_ens_features(conn)
            phase2.build_ens_agg(conn)

        if run_phase3:
            phase3.build_point_error_model(conn, min_coverage=min_coverage)
            if strikes:
                phase3.build_eval_metrics(conn, strikes, min_coverage, min_members)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=os.getenv("WEATHER_LOG_LEVEL", "INFO"))

    parser = argparse.ArgumentParser(description="Weather strategy end-to-end pipeline")
    parser.add_argument("--db", default=os.getenv("WEATHER_DB_PATH", "data/weather_phase1.sqlite"))
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--no-point", action="store_true")
    parser.add_argument("--no-gefs", action="store_true")
    parser.add_argument("--no-features", action="store_true")
    parser.add_argument("--no-phase3", action="store_true")
    parser.add_argument("--gefs-run-time-utc", action="append", default=[])
    parser.add_argument("--min-coverage", type=float, default=0.9)
    parser.add_argument("--min-members", type=int, default=2)
    parser.add_argument("--strike", action="append", type=float, default=[])
    args = parser.parse_args()

    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    gefs_run_times = [date_parser.parse(rt).astimezone(timezone.utc) for rt in args.gefs_run_time_utc]

    run_pipeline(
        db_path=args.db,
        start_date=start_date,
        end_date=end_date,
        run_point=not args.no_point,
        run_gefs=not args.no_gefs,
        run_features=not args.no_features,
        run_phase3=not args.no_phase3,
        gefs_run_times=gefs_run_times,
        min_coverage=args.min_coverage,
        min_members=args.min_members,
        strikes=args.strike,
    )


if __name__ == "__main__":
    main()
