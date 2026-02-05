from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable, Optional

import requests
import time as time_mod
from requests.exceptions import RequestException
from dateutil import parser as date_parser
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
import logging


LOCAL_TZ = ZoneInfo("America/New_York")
# GEFS_URL_TEMPLATE should point to the NOMADS filter endpoint (base URL only).
# Default: https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl
# The script builds query params to pull small subregions (2m temperature only).
NWS_POINTS_BASE = "https://api.weather.gov/points"
NWS_PRODUCTS_BASE = "https://api.weather.gov/products"
NWS_PRODUCTS_TYPE = "CLI"
NWS_PRODUCTS_LOCATION = "OKX"  # deprecated for CLI discovery (location endpoint is unreliable)

DEFAULT_STATION = "Central Park NY"
DEFAULT_LAT = 40.77898
DEFAULT_LON = -73.96925
DEFAULT_SOURCE_NWS = "NWS CLI"
DEFAULT_SOURCE_POINT = "api.weather.gov"
DEFAULT_SOURCE_GEFS = "GEFS NOMADS 0.25"
DEFAULT_SOURCE_NCEI = "NCEI GHCND"

logger = logging.getLogger("weather_phase1")
_gefs_metrics = Counter()


@dataclass
class GridMeta:
    grid_id: str
    grid_x: int
    grid_y: int
    forecast_hourly_url: str


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_db_path() -> str:
    return os.path.join(_repo_root(), "data", "weather_phase1.sqlite")


def _now_et() -> datetime:
    return datetime.now(tz=LOCAL_TZ)


def _ensure_tz(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=LOCAL_TZ)
    return dt.astimezone(LOCAL_TZ)


def _to_et(dt: datetime) -> datetime:
    return dt.astimezone(LOCAL_TZ)


def _parse_dt(value: str) -> datetime:
    dt = date_parser.parse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    return dt


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _iso(dt: datetime) -> str:
    return dt.astimezone(LOCAL_TZ).isoformat()


def _date_iso(d: date) -> str:
    return d.isoformat()


def _lead_hours(asof_time_et: datetime, target_date_et: date) -> float:
    target_start = datetime.combine(target_date_et, time.min, tzinfo=LOCAL_TZ)
    delta = target_start - asof_time_et
    return delta.total_seconds() / 3600.0


def _bucket_target_dates(base_date: date, days_ahead: int) -> list[date]:
    return [base_date + timedelta(days=offset) for offset in range(1, days_ahead + 1)]


def init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_payloads (
                hash TEXT PRIMARY KEY,
                content_type TEXT,
                pulled_at_et TEXT NOT NULL,
                payload BLOB NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS obs_daily (
                id INTEGER PRIMARY KEY,
                date_et TEXT NOT NULL,
                station TEXT NOT NULL,
                tmax_obs_f REAL,
                source TEXT NOT NULL,
                report_type TEXT,
                report_issued_at_et TEXT,
                pulled_at_et TEXT NOT NULL,
                raw_text_hash TEXT,
                raw_text_len INTEGER,
                product_id TEXT,
                UNIQUE(date_et, station, report_type, report_issued_at_et)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fcst_point_snapshots (
                id INTEGER PRIMARY KEY,
                asof_time_et TEXT NOT NULL,
                target_date_et TEXT NOT NULL,
                lead_hours REAL NOT NULL,
                tmax_point_f REAL,
                coverage_ratio REAL,
                source TEXT NOT NULL,
                lat REAL,
                lon REAL,
                grid_id TEXT,
                grid_x INTEGER,
                grid_y INTEGER,
                raw_json_hash TEXT,
                UNIQUE(asof_time_et, target_date_et)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fcst_ens_snapshots (
                id INTEGER PRIMARY KEY,
                run_time_et TEXT,
                asof_time_et TEXT NOT NULL,
                target_date_et TEXT NOT NULL,
                lead_hours REAL NOT NULL,
                member_id TEXT NOT NULL,
                tmax_member_f REAL,
                slice_count INTEGER,
                slice_coverage_ratio REAL,
                source TEXT NOT NULL,
                lat REAL,
                lon REAL,
                raw_grib_subset_hash TEXT,
                UNIQUE(asof_time_et, target_date_et, member_id)
            )
            """
        )
        _ensure_columns(conn)
    logger.info("Initialized DB at %s", db_path)


def _ensure_columns(conn: sqlite3.Connection) -> None:
    def _add_column(table: str, column: str, col_type: str) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    _add_column("obs_daily", "product_id", "TEXT")
    _add_column("fcst_point_snapshots", "coverage_ratio", "REAL")
    _add_column("fcst_ens_snapshots", "slice_count", "INTEGER")
    _add_column("fcst_ens_snapshots", "slice_coverage_ratio", "REAL")


def _store_raw_payload(conn: sqlite3.Connection, payload: bytes, content_type: str, pulled_at_et: str) -> str:
    payload_hash = _hash_bytes(payload)
    conn.execute(
        "INSERT OR IGNORE INTO raw_payloads(hash, content_type, pulled_at_et, payload) VALUES(?,?,?,?)",
        (payload_hash, content_type, pulled_at_et, payload),
    )
    return payload_hash


def _nws_headers() -> dict[str, str]:
    contact = os.getenv("NWS_CONTACT", "contact@example.com")
    user_agent = os.getenv("NWS_USER_AGENT", f"weather-bot/phase1 ({contact})")
    return {"User-Agent": user_agent, "Accept": "application/geo+json"}


def _nomads_headers() -> dict[str, str]:
    contact = os.getenv("NWS_CONTACT", "contact@example.com")
    user_agent = os.getenv("NWS_USER_AGENT", f"weather-bot/phase1 ({contact})")
    return {"User-Agent": user_agent}


def _gefs_archive_base() -> str:
    return os.getenv("GEFS_ARCHIVE_BASE_URL", "https://noaa-gefs-pds.s3.amazonaws.com")


def _gefs_archive_url(run_time_utc: datetime, member_id: str, fhour: int) -> str:
    dir_template = os.getenv("GEFS_ARCHIVE_DIR_TEMPLATE", "/gefs.{date_ymd}/{run_hour}/atmos/pgrb2sp25")
    file_template = os.getenv("GEFS_ARCHIVE_FILE_TEMPLATE", "{member}.t{run_hour}z.pgrb2s.0p25.f{fhour:03d}")
    rel_dir = dir_template.format(
        date_ymd=run_time_utc.strftime("%Y%m%d"),
        run_hour=run_time_utc.strftime("%H"),
    ).lstrip("/")
    filename = file_template.format(
        member=member_id,
        run_hour=run_time_utc.strftime("%H"),
        fhour=fhour,
    )
    return f"{_gefs_archive_base().rstrip('/')}/{rel_dir}/{filename}"


def fetch_cli_products(session: requests.Session) -> list[dict]:
    # Use the global CLI list; location endpoints are unreliable for CLINYC.
    url = f"{NWS_PRODUCTS_BASE}/types/{NWS_PRODUCTS_TYPE}"
    logger.info("Fetching CLI products url=%s", url)
    resp = session.get(url, headers=_nws_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("@graph", [])


def fetch_cli_product_text(session: requests.Session, product_id: str) -> dict:
    url = f"{NWS_PRODUCTS_BASE}/{product_id}"
    logger.debug("Fetching CLI product id=%s", product_id)
    resp = session.get(url, headers=_nws_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def _extract_report_date(text: str) -> Optional[date]:
    for line in text.splitlines():
        if "CLIMATE SUMMARY FOR" in line.upper():
            try:
                return date_parser.parse(line, fuzzy=True).date()
            except (ValueError, TypeError):
                continue
    return None


def _extract_tmax(text: str) -> Optional[float]:
    lines = text.splitlines()
    in_temp_section = False
    for line in lines:
        upper = line.upper()
        if "TEMPERATURE" in upper and "(F)" in upper:
            in_temp_section = True
            continue
        if in_temp_section and not line.strip():
            in_temp_section = False
        if in_temp_section and upper.strip().startswith("MAXIMUM"):
            tokens = [tok for tok in line.split() if tok.replace(".", "", 1).lstrip("-").isdigit()]
            if tokens:
                try:
                    return float(tokens[0])
                except ValueError:
                    continue
    # fallback: any line with MAXIMUM and a number
    for line in lines:
        upper = line.upper()
        if "MAXIMUM" in upper:
            tokens = [tok for tok in line.split() if tok.replace(".", "", 1).lstrip("-").isdigit()]
            if tokens:
                try:
                    return float(tokens[0])
                except ValueError:
                    continue
    return None


def _report_type(text: str) -> str:
    if "PRELIMINARY" in text.upper():
        return "prelim"
    return "daily"


def ingest_truth(
    conn: sqlite3.Connection,
    target_date: date,
    station: str,
    session: requests.Session,
    limit: int,
) -> None:
    logger.info("Ingesting CLI truth date=%s station=%s", _date_iso(target_date), station)
    products = fetch_cli_products(session)
    window_start = datetime.combine(target_date, time.min, tzinfo=LOCAL_TZ)
    window_end = window_start + timedelta(days=2)
    matching: list[dict] = []
    candidates: list[dict] = []
    scanned = 0
    for product in products:
        product_id = product.get("id")
        if not product_id:
            continue
        issuing_office = str(product.get("issuingOffice", "")).upper()
        wmo_collective_id = str(product.get("wmoCollectiveId", "")).upper()
        issuance_time = product.get("issuanceTime") or product.get("issuance_time")
        if issuing_office != "KOKX" or wmo_collective_id != "CDUS41":
            continue
        if issuance_time:
            try:
                issued_dt = _parse_dt(str(issuance_time)).astimezone(LOCAL_TZ)
            except (ValueError, TypeError):
                issued_dt = None
            if issued_dt and not (window_start <= issued_dt <= window_end):
                continue
        candidates.append(product)

    if limit and len(candidates) > limit:
        candidates = candidates[:limit]

    for product in candidates:
        product_id = product.get("id")
        if not product_id:
            continue
        detail = fetch_cli_product_text(session, product_id)
        scanned += 1
        text = detail.get("productText", "")
        text_upper = text.upper()
        if "CLINYC" not in text_upper and "CENTRAL PARK NY" not in text_upper:
            continue
        report_date = _extract_report_date(text)
        if report_date != target_date:
            continue
        matching.append(detail)

    pulled_at = _iso(_now_et())
    logger.info(
        "Candidates=%d scanned=%d matched=%d for date=%s",
        len(candidates),
        scanned,
        len(matching),
        _date_iso(target_date),
    )
    for detail in matching:
        text = detail.get("productText", "")
        tmax = _extract_tmax(text)
        issued = detail.get("issued") or detail.get("issuanceTime") or detail.get("issuance_time")
        issued_dt = _parse_dt(issued) if issued else None
        report_type = _report_type(text)
        payload = text.encode("utf-8", errors="ignore")
        raw_hash = _store_raw_payload(conn, payload, "text/plain", pulled_at)
        conn.execute(
            """
            INSERT OR IGNORE INTO obs_daily(
                date_et,
                station,
                tmax_obs_f,
                source,
                report_type,
                report_issued_at_et,
                pulled_at_et,
                raw_text_hash,
                raw_text_len,
                product_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _date_iso(target_date),
                station,
                tmax,
                DEFAULT_SOURCE_NWS,
                report_type,
                _iso(_ensure_tz(issued_dt)) if issued_dt else None,
                pulled_at,
                raw_hash,
                len(payload),
                detail.get("id"),
            ),
        )
    logger.info("Truth ingest complete date=%s rows=%d", _date_iso(target_date), len(matching))
    if not matching:
        _ingest_truth_ncei(conn, target_date, station, session)


def _ingest_truth_ncei(
    conn: sqlite3.Connection,
    target_date: date,
    station: str,
    session: requests.Session,
) -> None:
    token = os.getenv("NCEI_TOKEN")
    if not token:
        logger.warning("NCEI_TOKEN not set; skipping NCEI truth fallback")
        return
    station_id = os.getenv("NCEI_STATION_ID", "GHCND:USW00094728")
    url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    params = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "startdate": target_date.isoformat(),
        "enddate": target_date.isoformat(),
        "datatypeid": "TMAX",
        "units": "standard",
        "limit": 10,
    }
    headers = {"token": token}
    resp = None
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=45)
        except RequestException as exc:
            last_err = exc
            logger.warning("NCEI truth request failed attempt=%d error=%s", attempt + 1, exc)
            continue
        if resp.status_code >= 400:
            logger.warning("NCEI truth fetch failed status=%s url=%s", resp.status_code, resp.url)
            return
        break
    if resp is None:
        logger.warning("NCEI truth request failed after retries error=%s", last_err)
        return
    data = resp.json()
    results = data.get("results", [])
    if not results:
        logger.warning("NCEI truth empty for date=%s station=%s", target_date.isoformat(), station_id)
        return
    tmax_val = None
    for row in results:
        if str(row.get("datatype", "")).upper() == "TMAX":
            tmax_val = row.get("value")
            break
    if tmax_val is None:
        logger.warning("NCEI truth missing TMAX for date=%s station=%s", target_date.isoformat(), station_id)
        return
    try:
        tmax = float(tmax_val)
        # If NCEI returns tenths of C, convert to F.
        if tmax > 150:
            tmax = (tmax / 10.0) * 9.0 / 5.0 + 32.0
    except (TypeError, ValueError):
        logger.warning("NCEI truth invalid TMAX value=%s", tmax_val)
        return
    pulled_at = _iso(_now_et())
    payload = json.dumps(data, sort_keys=True).encode("utf-8")
    raw_hash = _store_raw_payload(conn, payload, "application/json", pulled_at)
    conn.execute(
        """
        INSERT OR IGNORE INTO obs_daily(
            date_et,
            station,
            tmax_obs_f,
            source,
            report_type,
            report_issued_at_et,
            pulled_at_et,
            raw_text_hash,
            raw_text_len,
            product_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _date_iso(target_date),
            station,
            tmax,
            DEFAULT_SOURCE_NCEI,
            None,
            None,
            pulled_at,
            raw_hash,
            len(payload),
            None,
        ),
    )
    logger.info("NCEI truth ingested date=%s tmax=%.1f", target_date.isoformat(), tmax)


def fetch_grid_meta(session: requests.Session, lat: float, lon: float) -> GridMeta:
    url = f"{NWS_POINTS_BASE}/{lat},{lon}"
    logger.info("Fetching grid meta lat=%.5f lon=%.5f", lat, lon)
    resp = session.get(url, headers=_nws_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    props = data.get("properties", {})
    grid_id = props.get("gridId")
    grid_x = props.get("gridX")
    grid_y = props.get("gridY")
    forecast_hourly_url = props.get("forecastHourly")
    if not all([grid_id, grid_x, grid_y, forecast_hourly_url]):
        raise RuntimeError("Missing grid metadata from api.weather.gov/points response")
    return GridMeta(grid_id=grid_id, grid_x=int(grid_x), grid_y=int(grid_y), forecast_hourly_url=forecast_hourly_url)


def _compute_tmax_from_hourly(
    periods: list[dict], target_date_et: date
) -> tuple[Optional[float], Optional[float]]:
    temps: list[float] = []
    total = 0
    for period in periods:
        start_time = period.get("startTime")
        temp = period.get("temperature")
        if start_time is None:
            continue
        try:
            dt = _parse_dt(start_time)
        except (ValueError, TypeError):
            continue
        dt_et = _to_et(dt)
        if dt_et.date() != target_date_et:
            continue
        total += 1
        if temp is None:
            continue
        try:
            temps.append(float(temp))
        except (ValueError, TypeError):
            continue
    if not temps:
        return None, None
    coverage = len(temps) / total if total else None
    return max(temps), coverage


def ingest_point_forecasts(
    conn: sqlite3.Connection,
    asof_time_et: datetime,
    target_dates: Iterable[date],
    lat: float,
    lon: float,
    session: requests.Session,
) -> None:
    target_list = list(target_dates)
    logger.info(
        "Ingesting point forecasts asof=%s targets=%s",
        _iso(asof_time_et),
        [d.isoformat() for d in target_list],
    )
    grid = fetch_grid_meta(session, lat, lon)
    resp = session.get(grid.forecast_hourly_url, headers=_nws_headers(), timeout=30)
    resp.raise_for_status()
    payload = resp.content
    pulled_at = _iso(_now_et())
    raw_hash = _store_raw_payload(conn, payload, resp.headers.get("Content-Type", "application/json"), pulled_at)
    data = resp.json()
    periods = data.get("properties", {}).get("periods", [])

    for target_date in target_list:
        tmax, coverage_ratio = _compute_tmax_from_hourly(periods, target_date)
        lead_hours = _lead_hours(asof_time_et, target_date)
        conn.execute(
            """
            INSERT OR IGNORE INTO fcst_point_snapshots(
                asof_time_et,
                target_date_et,
                lead_hours,
                tmax_point_f,
                coverage_ratio,
                source,
                lat,
                lon,
                grid_id,
                grid_x,
                grid_y,
                raw_json_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _iso(asof_time_et),
                _date_iso(target_date),
                lead_hours,
                tmax,
                coverage_ratio,
                DEFAULT_SOURCE_POINT,
                lat,
                lon,
                grid.grid_id,
                grid.grid_x,
                grid.grid_y,
                raw_hash,
            ),
        )
    logger.info("Point forecast ingest complete rows=%d raw_hash=%s", len(target_list), raw_hash)


def _gefs_filter_base() -> str:
    return os.getenv(
        "GEFS_URL_TEMPLATE",
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl",
    )


def _gefs_filter_params(
    run_time_utc: datetime,
    member_id: str,
    fhour: int,
    lat: float,
    lon: float,
) -> dict[str, str | float]:
    dir_template = os.getenv("GEFS_DIR_TEMPLATE", "/gefs.{date_ymd}/{run_hour}/atmos/pgrb2sp25")
    file_template = os.getenv("GEFS_FILE_TEMPLATE", "{member}.t{run_hour}z.pgrb2s.0p25.f{fhour:03d}")
    return {
        "dir": dir_template.format(
            date_ymd=run_time_utc.strftime("%Y%m%d"),
            run_hour=run_time_utc.strftime("%H"),
        ),
        "file": file_template.format(
            member=member_id,
            run_hour=run_time_utc.strftime("%H"),
            fhour=fhour,
        ),
        "var_TMP": "on",
        "lev_2_m_above_ground": "on",
        "subregion": "",
        "leftlon": lon - 0.5,
        "rightlon": lon + 0.5,
        "toplat": lat + 0.5,
        "bottomlat": lat - 0.5,
    }


def _download_gefs_grib(
    session: requests.Session,
    url: str,
    params: dict[str, str | float],
    run_time_utc: datetime,
    member_id: str,
    fhour: int,
) -> bytes | None:
    resp = session.get(url, params=params, headers=_nomads_headers(), timeout=60)
    if resp.status_code in {403, 404}:
        logger.warning("GEFS filter unavailable status=%s url=%s", resp.status_code, resp.url)
        _gefs_metrics[f"filter_{resp.status_code}"] += 1
        return _download_gefs_archive(session, run_time_utc, member_id, fhour)
    resp.raise_for_status()
    content = resp.content
    if not content.startswith(b"GRIB"):
        logger.warning(
            "GEFS response not GRIB: url=%s ct=%s len=%d",
            resp.url,
            resp.headers.get("Content-Type"),
            len(content),
        )
        _gefs_metrics["filter_non_grib"] += 1
        return _download_gefs_archive(session, run_time_utc, member_id, fhour)
    _gefs_metrics["filter_ok"] += 1
    return content


def _download_gefs_archive(
    session: requests.Session, run_time_utc: datetime, member_id: str, fhour: int
) -> bytes | None:
    url = _gefs_archive_url(run_time_utc, member_id, fhour)
    logger.info("GEFS archive fetch url=%s", url)
    backoff = 2
    for attempt in range(4):
        resp = session.get(url, headers=_nomads_headers(), timeout=120)
        if resp.status_code == 503:
            logger.warning("GEFS archive slow down (503) attempt=%d url=%s", attempt + 1, resp.url)
            _gefs_metrics["archive_503"] += 1
            time_mod.sleep(backoff)
            backoff *= 2
            continue
        if resp.status_code in {403, 404}:
            logger.warning("GEFS archive missing status=%s url=%s", resp.status_code, resp.url)
            _gefs_metrics[f"archive_{resp.status_code}"] += 1
            return None
        resp.raise_for_status()
        content = resp.content
        if not content.startswith(b"GRIB"):
            logger.warning(
                "GEFS archive response not GRIB: url=%s ct=%s len=%d",
                resp.url,
                resp.headers.get("Content-Type"),
                len(content),
            )
            _gefs_metrics["archive_non_grib"] += 1
            return None
        logger.info("GEFS archive download ok url=%s bytes=%d", resp.url, len(content))
        _gefs_metrics["archive_ok"] += 1
        return content
    logger.warning("GEFS archive slow down persisted; giving up url=%s", url)
    _gefs_metrics["archive_giveup"] += 1
    return None


def _parse_grib_t2m_f(grib_bytes: bytes, lat: float, lon: float) -> Optional[float]:
    try:
        import pygrib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pygrib is required to parse GEFS grib files. Install pygrib and eccodes.") from exc

    with tempfile.NamedTemporaryFile(suffix=".grib2") as tmp:
        tmp.write(grib_bytes)
        tmp.flush()
        with pygrib.open(tmp.name) as grbs:
            grb = None
            selectors = [
                {"shortName": "2t"},
                {"shortName": "t2m"},
                {"name": "2 metre temperature"},
                {"name": "Temperature", "typeOfLevel": "heightAboveGround", "level": 2},
            ]
            for selector in selectors:
                try:
                    matches = grbs.select(**selector)
                except Exception:
                    continue
                if matches:
                    grb = matches[0]
                    break
            if grb is None:
                try:
                    sample = [msg.shortName for msg in grbs[:5]]
                    logger.warning("No 2m temperature matches found; sample shortNames=%s", sample)
                except Exception:
                    logger.warning("No 2m temperature matches found and unable to sample messages.")
                return None
            values = grb.values
            lats, lons = grb.latlons()
            # nearest neighbor lookup
            dist = (lats - lat) ** 2 + (lons - lon) ** 2
            idx = dist.argmin()
            temp_k = float(values.flat[idx])
            return (temp_k - 273.15) * 9.0 / 5.0 + 32.0


def ingest_gefs_member(
    conn: sqlite3.Connection,
    session: requests.Session,
    member_id: str,
    run_time_utc: datetime,
    target_dates: Iterable[date],
    lat: float,
    lon: float,
    step_hours: int,
) -> None:
    run_time_utc = run_time_utc.astimezone(timezone.utc)
    asof_time_et = run_time_utc.astimezone(LOCAL_TZ)
    pulled_at = _iso(_now_et())
    base_url = _gefs_filter_base()
    target_list = list(target_dates)
    logger.info(
        "Ingesting GEFS member=%s run=%s targets=%s step=%dh",
        member_id,
        run_time_utc.isoformat(),
        [d.isoformat() for d in target_list],
        step_hours,
    )

    for target_date in target_list:
        target_start_et = datetime.combine(target_date, time.min, tzinfo=LOCAL_TZ)
        target_end_et = target_start_et + timedelta(days=1)
        temps: list[float] = []
        hashes: list[str] = []
        target_start_utc = target_start_et.astimezone(timezone.utc)
        target_end_utc = target_end_et.astimezone(timezone.utc)
        fhour_end = int((target_end_utc - run_time_utc).total_seconds() / 3600)
        fhours: list[int] = []
        for fhour in range(0, fhour_end + 1, step_hours):
            valid_time_utc = run_time_utc + timedelta(hours=fhour)
            if valid_time_utc < target_start_utc or valid_time_utc >= target_end_utc:
                continue
            fhours.append(fhour)

        logger.info("GEFS member=%s date=%s fhours=%s", member_id, _date_iso(target_date), fhours)
        expected_slices = len(fhours)
        for fhour in fhours:
            params = _gefs_filter_params(run_time_utc, member_id, fhour, lat, lon)
            grib_bytes = _download_gefs_grib(session, base_url, params, run_time_utc, member_id, fhour)
            if not grib_bytes:
                continue
            raw_hash = _store_raw_payload(conn, grib_bytes, "application/grib2", pulled_at)
            hashes.append(raw_hash)
            temp_f = _parse_grib_t2m_f(grib_bytes, lat, lon)
            if temp_f is not None:
                temps.append(temp_f)

        tmax = max(temps) if temps else None
        slice_count = len(temps)
        slice_coverage_ratio = (slice_count / expected_slices) if expected_slices else None
        combo_hash = _hash_bytes("".join(hashes).encode("utf-8")) if hashes else None
        lead_hours = _lead_hours(asof_time_et, target_date)
        conn.execute(
            """
            INSERT OR IGNORE INTO fcst_ens_snapshots(
                run_time_et,
                asof_time_et,
                target_date_et,
                lead_hours,
                member_id,
                tmax_member_f,
                slice_count,
                slice_coverage_ratio,
                source,
                lat,
                lon,
                raw_grib_subset_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _iso(asof_time_et),
                _iso(asof_time_et),
                _date_iso(target_date),
                lead_hours,
                member_id,
                tmax,
                slice_count,
                slice_coverage_ratio,
                DEFAULT_SOURCE_GEFS,
                lat,
                lon,
                combo_hash,
            ),
        )
        logger.info(
            "GEFS member=%s date=%s lead_hours=%.1f tmax=%s",
            member_id,
            _date_iso(target_date),
            lead_hours,
            "None" if tmax is None else f"{tmax:.2f}",
        )
    logger.info("GEFS ingest complete member=%s", member_id)


def gefs_metrics_snapshot() -> dict[str, int]:
    return dict(_gefs_metrics)


def _parse_date_list(values: list[str]) -> list[date]:
    dates: list[date] = []
    for value in values:
        dates.append(date_parser.parse(value).date())
    return dates


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=os.getenv("WEATHER_LOG_LEVEL", "INFO"))

    parser = argparse.ArgumentParser(description="Phase 1 weather data plumbing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-db", help="Initialize the weather phase1 sqlite DB")
    init_parser.add_argument("--db", default=os.getenv("WEATHER_DB_PATH", _default_db_path()))

    truth_parser = subparsers.add_parser("fetch-truth", help="Fetch NWS CLI climate truth")
    truth_parser.add_argument("--db", default=os.getenv("WEATHER_DB_PATH", _default_db_path()))
    truth_parser.add_argument("--date", required=True, help="Target date (YYYY-MM-DD) in ET")
    truth_parser.add_argument("--station", default=DEFAULT_STATION)
    truth_parser.add_argument(
        "--cli-location",
        default=os.getenv("NWS_CLI_LOCATION"),
        help="Deprecated. CLI discovery uses the global list instead of locations.",
    )
    truth_parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max CLI candidate products to fetch full text for (default 200).",
    )

    truth_range_parser = subparsers.add_parser(
        "fetch-truth-range", help="Fetch CLI/NCEI truth for a date range"
    )
    truth_range_parser.add_argument("--db", default=os.getenv("WEATHER_DB_PATH", _default_db_path()))
    truth_range_parser.add_argument("--start-date", required=True)
    truth_range_parser.add_argument("--end-date", required=True)
    truth_range_parser.add_argument("--station", default=DEFAULT_STATION)
    truth_range_parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max CLI candidate products to fetch full text for (default 200).",
    )

    point_parser = subparsers.add_parser("fetch-point", help="Fetch NWS point forecast snapshots")
    point_parser.add_argument("--db", default=os.getenv("WEATHER_DB_PATH", _default_db_path()))
    point_parser.add_argument("--asof", default=None, help="As-of time (defaults to now ET)")
    point_parser.add_argument("--days", type=int, default=3, help="Days ahead to snapshot")
    point_parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    point_parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    point_parser.add_argument("--target-date", action="append", default=[], help="Explicit target date override")

    gefs_parser = subparsers.add_parser("fetch-gefs", help="Fetch GEFS ensemble snapshots")
    gefs_parser.add_argument("--db", default=os.getenv("WEATHER_DB_PATH", _default_db_path()))
    gefs_parser.add_argument("--run-time-utc", required=True, help="Run time in UTC, e.g. 2026-01-27T00:00Z")
    gefs_parser.add_argument("--member", action="append", required=True, help="Member id (e.g. gec00, gep01)")
    gefs_parser.add_argument("--days", type=int, default=3, help="Days ahead to snapshot")
    gefs_parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    gefs_parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    gefs_parser.add_argument("--step-hours", type=int, default=6, help="Forecast hour step to sample")
    gefs_parser.add_argument("--target-date", action="append", default=[], help="Explicit target date override")

    cleanup_parser = subparsers.add_parser("cleanup-db", help="Delete rows with null key fields")
    cleanup_parser.add_argument("--db", default=os.getenv("WEATHER_DB_PATH", _default_db_path()))
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show counts only, do not delete.",
    )

    args = parser.parse_args()

    if args.command == "init-db":
        init_db(args.db)
        return

    with sqlite3.connect(args.db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        session = requests.Session()

        if args.command == "fetch-truth":
            target_date = date_parser.parse(args.date).date()
            if args.cli_location:
                os.environ["NWS_CLI_LOCATION"] = args.cli_location
            ingest_truth(conn, target_date, args.station, session, args.limit)
            return
        if args.command == "fetch-truth-range":
            start_date = date_parser.parse(args.start_date).date()
            end_date = date_parser.parse(args.end_date).date()
            if end_date < start_date:
                raise SystemExit("end-date must be >= start-date")
            current = start_date
            while current <= end_date:
                ingest_truth(conn, current, args.station, session, args.limit)
                current += timedelta(days=1)
            return

        if args.command == "fetch-point":
            asof_time = _parse_dt(args.asof) if args.asof else _now_et()
            if args.target_date:
                target_dates = _parse_date_list(args.target_date)
            else:
                target_dates = _bucket_target_dates(asof_time.date(), args.days)
            ingest_point_forecasts(conn, asof_time, target_dates, args.lat, args.lon, session)
            return

        if args.command == "fetch-gefs":
            run_time_utc = _parse_dt(args.run_time_utc).astimezone(timezone.utc)
            if args.target_date:
                target_dates = _parse_date_list(args.target_date)
            else:
                target_dates = _bucket_target_dates(run_time_utc.astimezone(LOCAL_TZ).date(), args.days)
            for member in args.member:
                ingest_gefs_member(
                    conn,
                    session,
                    member,
                    run_time_utc,
                    target_dates,
                    args.lat,
                    args.lon,
                    args.step_hours,
                )
            return

        if args.command == "cleanup-db":
            targets = {
                "obs_daily": "report_issued_at_et IS NULL",
                "fcst_ens_snapshots": "tmax_member_f IS NULL",
                "fcst_point_snapshots": "coverage_ratio IS NULL",
            }
            for table, where_clause in targets.items():
                count = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where_clause}").fetchone()[0]
                if args.dry_run:
                    logger.info("cleanup-db %s rows_to_delete=%d", table, count)
                else:
                    conn.execute(f"DELETE FROM {table} WHERE {where_clause}")
                    logger.info("cleanup-db %s deleted=%d", table, count)
            return


if __name__ == "__main__":
    main()
from collections import Counter
