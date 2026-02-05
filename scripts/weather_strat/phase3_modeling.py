from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from dateutil import parser as date_parser
from zoneinfo import ZoneInfo


LOCAL_TZ = ZoneInfo("America/New_York")
DEFAULT_MIN_MEMBERS = 2

logger = logging.getLogger("weather_phase3")


@dataclass
class ErrorStats:
    n: int
    mean: float | None
    stdev: float | None
    p05: float | None
    p25: float | None
    p50: float | None
    p75: float | None
    p95: float | None


def _to_et(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=LOCAL_TZ)
    return dt.astimezone(LOCAL_TZ)


def _parse_dt(value: str) -> datetime:
    dt = date_parser.parse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    return dt


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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mu = _mean(values)
    if mu is None:
        return None
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _lead_bucket(lead_hours: float, buckets: list[int]) -> int | None:
    bucket = None
    for candidate in buckets:
        if lead_hours >= candidate:
            bucket = candidate
    return bucket


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


def _collect_point_errors(
    conn: sqlite3.Connection, min_coverage: float
) -> dict[int, list[float]]:
    truth = _latest_truth_by_date(conn)
    rows = conn.execute(
        """
        SELECT target_date_et, lead_bucket_hours, tmax_point_f, coverage_ratio
        FROM point_daily_features
        WHERE tmax_point_f IS NOT NULL
        """
    ).fetchall()
    errors: dict[int, list[float]] = {}
    for target_date_et, lead_bucket, f_point, coverage in rows:
        if coverage is not None and coverage < min_coverage:
            continue
        if lead_bucket is None:
            continue
        y = truth.get(target_date_et)
        if y is None:
            continue
        err = y - float(f_point)
        errors.setdefault(int(lead_bucket), []).append(err)
    return errors


def _insert_point_error_stats(conn: sqlite3.Connection, errors: dict[int, list[float]]) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS point_error_stats (
            lead_bucket_hours INTEGER PRIMARY KEY,
            n INTEGER,
            mean REAL,
            stdev REAL,
            p05 REAL,
            p25 REAL,
            p50 REAL,
            p75 REAL,
            p95 REAL,
            updated_at_et TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS point_error_samples (
            lead_bucket_hours INTEGER,
            err REAL
        )
        """
    )
    conn.execute("DELETE FROM point_error_stats")
    conn.execute("DELETE FROM point_error_samples")
    for bucket, errs in errors.items():
        stats = ErrorStats(
            n=len(errs),
            mean=_mean(errs),
            stdev=_stdev(errs),
            p05=_quantile(errs, 0.05),
            p25=_quantile(errs, 0.25),
            p50=_quantile(errs, 0.50),
            p75=_quantile(errs, 0.75),
            p95=_quantile(errs, 0.95),
        )
        for err in errs:
            conn.execute(
                "INSERT INTO point_error_samples(lead_bucket_hours, err) VALUES(?, ?)",
                (bucket, float(err)),
            )
        conn.execute(
            """
            INSERT INTO point_error_stats(
                lead_bucket_hours, n, mean, stdev, p05, p25, p50, p75, p95, updated_at_et
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bucket,
                stats.n,
                stats.mean,
                stats.stdev,
                stats.p05,
                stats.p25,
                stats.p50,
                stats.p75,
                stats.p95,
                datetime.now(tz=LOCAL_TZ).isoformat(),
            ),
        )


def _point_prob(conn: sqlite3.Connection, lead_bucket: int, f_point: float, strike: float) -> float | None:
    rows = conn.execute(
        "SELECT err FROM point_error_samples WHERE lead_bucket_hours = ?",
        (lead_bucket,),
    ).fetchall()
    errs = [float(r[0]) for r in rows]
    if not errs:
        return None
    threshold = strike - f_point
    return sum(1 for e in errs if e > threshold) / len(errs)


def _ens_prob(
    conn: sqlite3.Connection, asof_time_et: str, target_date_et: str, strike: float, min_members: int
) -> tuple[float | None, int]:
    rows = conn.execute(
        """
        SELECT tmax_member_f
        FROM ens_daily_features
        WHERE asof_time_et = ? AND target_date_et = ? AND tmax_member_f IS NOT NULL
        """,
        (asof_time_et, target_date_et),
    ).fetchall()
    values = [float(r[0]) for r in rows]
    if len(values) < min_members:
        return None, len(values)
    prob = sum(1 for v in values if v > strike) / len(values)
    return prob, len(values)


def _brier_score(preds: list[float], events: list[int]) -> float | None:
    if not preds:
        return None
    return sum((p - e) ** 2 for p, e in zip(preds, events)) / len(preds)


def build_point_error_model(conn: sqlite3.Connection, min_coverage: float) -> None:
    errors = _collect_point_errors(conn, min_coverage)
    _insert_point_error_stats(conn, errors)
    logger.info("Built point_error_stats buckets=%d", len(errors))


def build_eval_metrics(
    conn: sqlite3.Connection,
    strikes: Iterable[float],
    min_coverage: float,
    min_members: int,
) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS phase3_metrics (
            model TEXT NOT NULL,
            lead_bucket_hours INTEGER,
            strike REAL NOT NULL,
            n INTEGER,
            brier REAL,
            updated_at_et TEXT
        )
        """
    )
    conn.execute("DELETE FROM phase3_metrics")

    truth = _latest_truth_by_date(conn)
    point_rows = conn.execute(
        """
        SELECT asof_time_et, target_date_et, lead_bucket_hours, tmax_point_f, coverage_ratio
        FROM point_daily_features
        WHERE tmax_point_f IS NOT NULL
        """
    ).fetchall()

    for strike in strikes:
        preds_point: dict[int, list[float]] = {}
        events_point: dict[int, list[int]] = {}
        preds_ens: dict[int, list[float]] = {}
        events_ens: dict[int, list[int]] = {}

        for asof_time_et, target_date_et, lead_bucket, f_point, coverage in point_rows:
            if coverage is not None and coverage < min_coverage:
                continue
            if lead_bucket is None:
                continue
            y = truth.get(target_date_et)
            if y is None:
                continue
            lead_bucket = int(lead_bucket)
            p_point = _point_prob(conn, lead_bucket, float(f_point), strike)
            if p_point is not None:
                preds_point.setdefault(lead_bucket, []).append(p_point)
                events_point.setdefault(lead_bucket, []).append(1 if y > strike else 0)

            p_ens, members = _ens_prob(conn, asof_time_et, target_date_et, strike, min_members)
            if p_ens is not None:
                preds_ens.setdefault(lead_bucket, []).append(p_ens)
                events_ens.setdefault(lead_bucket, []).append(1 if y > strike else 0)

        for lead_bucket, preds in preds_point.items():
            brier = _brier_score(preds, events_point.get(lead_bucket, []))
            conn.execute(
                """
                INSERT INTO phase3_metrics(model, lead_bucket_hours, strike, n, brier, updated_at_et)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "point",
                    lead_bucket,
                    strike,
                    len(preds),
                    brier,
                    datetime.now(tz=LOCAL_TZ).isoformat(),
                ),
            )
        for lead_bucket, preds in preds_ens.items():
            brier = _brier_score(preds, events_ens.get(lead_bucket, []))
            conn.execute(
                """
                INSERT INTO phase3_metrics(model, lead_bucket_hours, strike, n, brier, updated_at_et)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "ens",
                    lead_bucket,
                    strike,
                    len(preds),
                    brier,
                    datetime.now(tz=LOCAL_TZ).isoformat(),
                ),
            )


def predict_probability(
    conn: sqlite3.Connection,
    asof_time_et: str,
    target_date_et: str,
    strike: float,
    min_members: int,
    blend_weight: float | None,
) -> dict[str, float | None]:
    row = conn.execute(
        """
        SELECT lead_bucket_hours, tmax_point_f
        FROM point_daily_features
        WHERE asof_time_et = ? AND target_date_et = ?
        """,
        (asof_time_et, target_date_et),
    ).fetchone()
    if not row:
        return {
            "p_point": None,
            "p_ens": None,
            "p_blend": None,
            "members_present": None,
        }
    lead_bucket, f_point = row
    if lead_bucket is None or f_point is None:
        p_point = None
    else:
        p_point = _point_prob(conn, int(lead_bucket), float(f_point), strike)

    p_ens, members = _ens_prob(conn, asof_time_et, target_date_et, strike, min_members)

    p_blend = None
    if blend_weight is not None and p_point is not None and p_ens is not None:
        p_blend = blend_weight * p_ens + (1 - blend_weight) * p_point

    return {
        "p_point": p_point,
        "p_ens": p_ens,
        "p_blend": p_blend,
        "members_present": members,
    }


def main() -> None:
    logging.basicConfig(level=os.getenv("WEATHER_LOG_LEVEL", "INFO"))
    parser = argparse.ArgumentParser(description="Phase 3 probability modeling")
    parser.add_argument("--db", default=os.getenv("WEATHER_DB_PATH", "data/weather_phase1.sqlite"))
    parser.add_argument("--min-coverage", type=float, default=0.9)
    parser.add_argument("--min-members", type=int, default=DEFAULT_MIN_MEMBERS)
    parser.add_argument("--strike", action="append", type=float, default=[])
    parser.add_argument("--blend-weight", type=float, default=None)
    parser.add_argument("--predict", action="store_true", help="Return p_point/p_ens for a given snapshot")
    parser.add_argument("--asof", default=None)
    parser.add_argument("--target-date", default=None)
    args = parser.parse_args()

    with sqlite3.connect(args.db) as conn:
        build_point_error_model(conn, args.min_coverage)
        if args.strike:
            build_eval_metrics(conn, args.strike, args.min_coverage, args.min_members)
        if args.predict:
            if not args.asof or not args.target_date or not args.strike:
                raise SystemExit("--predict requires --asof, --target-date, and --strike")
            result = predict_probability(
                conn,
                args.asof,
                args.target_date,
                args.strike[0],
                args.min_members,
                args.blend_weight,
            )
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
