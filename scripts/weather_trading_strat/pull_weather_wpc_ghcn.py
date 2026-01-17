from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dateutil import parser as date_parser
from dotenv import load_dotenv
from zoneinfo import ZoneInfo


SECONDS_HOUR = 3600
ANCHOR_HOURS = 48
LOCAL_TZ = ZoneInfo("America/New_York")

WPC_BASE = "https://mapservices.weather.noaa.gov/vector/rest/services/precip/wpc_prob_winter_precip/MapServer"
WPC_LAYER_MAP = {
    (1, 4): 1,
    (1, 8): 2,
    (1, 12): 3,
    (2, 4): 6,
    (2, 8): 7,
    (2, 12): 8,
    (3, 4): 11,
    (3, 8): 12,
    (3, 12): 13,
}
WPC_BANDS = {
    "Slight (10-39%)": (0.10, 0.39),
    "Moderate (40-69%)": (0.40, 0.69),
    "High (70-100%)": (0.70, 1.00),
    "NONE": (0.00, 0.09),
}

CITY_COORDS = {
    "NYC": (-73.96925, 40.77898),
    "BOS": (-71.00975, 42.36057),
}
CITY_STATIONS = {
    "NYC": "GHCND:USW00094728",
    "BOS": "GHCND:USW00014739",
}


logger = logging.getLogger("weather_research")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


@dataclass
class WpcResult:
    day_bucket: int
    layer_id: int
    outlook: str
    prob_lo: float
    prob_hi: float
    issue_time: Any
    valid_time: Any
    start_time: Any
    end_time: Any
    product: Any


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_epoch_seconds(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return int(value.timestamp())
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        num = float(value)
        if num > 1e12:
            num /= 1000.0
        elif num > 1e10:
            num /= 1000.0
        return int(num)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            num = float(text)
            if num > 1e12:
                num /= 1000.0
            elif num > 1e10:
                num /= 1000.0
            return int(num)
        except ValueError:
            pass
        dt = date_parser.parse(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    return None


def _parse_ts_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        max_val = numeric.max()
        if max_val > 1e12:
            numeric = (numeric / 1000.0).astype("Int64")
        elif max_val > 1e10:
            numeric = (numeric / 1000.0).astype("Int64")
        else:
            numeric = numeric.astype("Int64")
    if numeric.isna().any():
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        if parsed.notna().any():
            fallback = (parsed.astype("int64") // 1_000_000_000).astype("Int64")
            numeric = numeric.fillna(fallback)
    return numeric


def _normalize_threshold(row: pd.Series) -> int | None:
    for col in ("threshold_in", "threshold", "threshold_inches", "strike_in", "strike"):
        if col in row and pd.notna(row[col]):
            try:
                return int(float(row[col]))
            except (TypeError, ValueError):
                continue
    ticker = str(row.get("market_ticker", "")).upper()
    for candidate in (4, 8, 12):
        if f"{candidate}\"" in ticker or f"{candidate}IN" in ticker:
            return candidate
    return None


def _normalize_city(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().upper()
    if text in CITY_COORDS:
        return text
    if text in ("NEW YORK", "NEW YORK CITY", "NYC"):
        return "NYC"
    if text in ("BOSTON", "BOS"):
        return "BOS"
    return None


def _determine_day_bucket(lead_hours: float) -> int | None:
    if lead_hours <= 0:
        return None
    if lead_hours <= 24:
        return 1
    if lead_hours <= 48:
        return 2
    if lead_hours <= 72:
        return 3
    return None


def _find_candle_file(candles_dir: Path, market_ticker: str, index: dict[str, list[Path]]) -> Path | None:
    key = market_ticker.strip()
    paths = index.get(key)
    if paths:
        return paths[0]
    return None


def _build_candle_index(candles_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in candles_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".csv", ".parquet", ".json", ".jsonl"}:
            continue
        index.setdefault(path.stem, []).append(path)
    return index


def _load_candles(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    return pd.read_csv(path)


def _last_close_before(candles: pd.DataFrame, cutoff_ts: int) -> float | None:
    if candles.empty:
        return None
    ts = _parse_ts_series(candles["ts"])
    closes = pd.to_numeric(candles["close"], errors="coerce")
    eligible = candles[(ts.notna()) & (ts <= cutoff_ts) & closes.notna()].copy()
    if eligible.empty:
        return None
    eligible = eligible.assign(ts=ts.loc[eligible.index], close=closes.loc[eligible.index])
    eligible = eligible.sort_values("ts")
    return float(eligible.iloc[-1]["close"])


def _query_wpc(
    session: requests.Session, day_bucket: int, threshold_in: int, lon: float, lat: float
) -> WpcResult | None:
    layer_id = WPC_LAYER_MAP.get((day_bucket, threshold_in))
    if layer_id is None:
        return None
    params = {
        "f": "json",
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "product,valid_time,issue_time,start_time,end_time,outlook",
        "returnGeometry": "false",
    }
    url = f"{WPC_BASE}/{layer_id}/query"
    response = session.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    features = payload.get("features", []) or []
    if not features:
        prob_lo, prob_hi = WPC_BANDS["NONE"]
        return WpcResult(
            day_bucket=day_bucket,
            layer_id=layer_id,
            outlook="NONE",
            prob_lo=prob_lo,
            prob_hi=prob_hi,
            issue_time=None,
            valid_time=None,
            start_time=None,
            end_time=None,
            product=None,
        )
    attrs = features[0].get("attributes", {}) or {}
    outlook = attrs.get("outlook")
    if outlook not in WPC_BANDS:
        outlook = "NONE"
    prob_lo, prob_hi = WPC_BANDS[outlook]
    return WpcResult(
        day_bucket=day_bucket,
        layer_id=layer_id,
        outlook=outlook,
        prob_lo=prob_lo,
        prob_hi=prob_hi,
        issue_time=attrs.get("issue_time"),
        valid_time=attrs.get("valid_time"),
        start_time=attrs.get("start_time"),
        end_time=attrs.get("end_time"),
        product=attrs.get("product"),
    )


def _normalize_date_value(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.date().isoformat()
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = date_parser.parse(text)
        return dt.date().isoformat()
    except (ValueError, TypeError):
        return None


def _resolution_date_local(resolution_ts: int) -> str:
    dt = datetime.fromtimestamp(resolution_ts, tz=LOCAL_TZ)
    return dt.date().isoformat()


def _fetch_ghcn_snow(
    session: requests.Session,
    token: str,
    station_id: str,
    start_date: str,
    end_date: str,
) -> tuple[float | None, list[dict[str, Any]]]:
    headers = {"token": token}
    params = {
        "datasetid": "GHCND",
        "datatypeid": "SNOW",
        "stationid": station_id,
        "startdate": start_date,
        "enddate": end_date,
        "limit": 1000,
        "offset": 1,
    }
    results: list[dict[str, Any]] = []
    while True:
        response = session.get(
            "https://www.ncdc.noaa.gov/cdo-web/api/v2/data",
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("results", []) or []
        results.extend(batch)
        metadata = payload.get("metadata", {}) or {}
        resultset = metadata.get("resultset", {}) or {}
        count = resultset.get("count", len(results))
        if len(results) >= count:
            break
        params["offset"] = params["offset"] + params["limit"]
    total = None
    values = [item.get("value") for item in results if item.get("value") is not None]
    if values:
        total = float(sum(values))
    return total, results


def load_kalshi_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Kalshi dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=suffix == ".jsonl")
    else:
        df = pd.read_csv(path)
    return df


def build_dataset(
    kalshi_dataset: Path,
    candles_dir: Path,
    token_env: str,
    output_dir: Path,
) -> None:
    df = load_kalshi_dataset(kalshi_dataset)
    if df.empty:
        raise ValueError("Kalshi dataset is empty.")

    token = os.getenv(token_env)
    if not token:
        raise ValueError(f"Missing NCEI token in env var {token_env}.")

    output_dir.mkdir(parents=True, exist_ok=True)

    candle_index = _build_candle_index(candles_dir)
    session = requests.Session()

    skip_reasons: dict[str, int] = {}
    wpc_misses = 0
    obs_failures = 0
    obs_cache: dict[tuple[str, str, str], float | None] = {}

    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        market_ticker = str(row.get("market_ticker", "")).strip()
        if not market_ticker:
            skip_reasons["missing_market_ticker"] = skip_reasons.get("missing_market_ticker", 0) + 1
            continue

        city = _normalize_city(row.get("city"))
        if not city:
            skip_reasons["unsupported_city"] = skip_reasons.get("unsupported_city", 0) + 1
            continue

        threshold_in = _normalize_threshold(row)
        if threshold_in not in (4, 8, 12):
            skip_reasons["unsupported_threshold"] = skip_reasons.get("unsupported_threshold", 0) + 1
            continue

        resolution_val = row.get("resolution_ts")
        if pd.isna(resolution_val) or resolution_val is None:
            resolution_val = row.get("event_end_ts")
        resolution_ts = _to_epoch_seconds(resolution_val)
        if resolution_ts is None:
            skip_reasons["missing_resolution_ts"] = skip_reasons.get("missing_resolution_ts", 0) + 1
            continue

        t_anchor = resolution_ts - ANCHOR_HOURS * SECONDS_HOUR
        lead_hours = (resolution_ts - t_anchor) / SECONDS_HOUR
        day_bucket = _determine_day_bucket(lead_hours)
        if not day_bucket:
            skip_reasons["outside_wpc_window"] = skip_reasons.get("outside_wpc_window", 0) + 1
            continue

        candle_path = _find_candle_file(candles_dir, market_ticker, candle_index)
        if not candle_path:
            skip_reasons["missing_candles"] = skip_reasons.get("missing_candles", 0) + 1
            continue

        candles = _load_candles(candle_path)
        if "close" not in candles.columns and "yes_price" in candles.columns:
            candles = candles.rename(columns={"yes_price": "close"})
        if "ts" not in candles.columns or "close" not in candles.columns:
            skip_reasons["invalid_candle_format"] = skip_reasons.get("invalid_candle_format", 0) + 1
            continue

        p_market = _last_close_before(candles, t_anchor)
        if p_market is None:
            skip_reasons["missing_anchor_price"] = skip_reasons.get("missing_anchor_price", 0) + 1
            continue

        lon, lat = CITY_COORDS[city]
        try:
            wpc = _query_wpc(session, day_bucket, threshold_in, lon, lat)
        except Exception:
            skip_reasons["wpc_query_failed"] = skip_reasons.get("wpc_query_failed", 0) + 1
            continue
        if wpc is None:
            skip_reasons["missing_wpc_layer"] = skip_reasons.get("missing_wpc_layer", 0) + 1
            continue
        if wpc.outlook == "NONE":
            wpc_misses += 1

        ghcn_station_id = CITY_STATIONS[city]
        window_start_date = _normalize_date_value(row.get("window_start_date"))
        window_end_date = _normalize_date_value(row.get("window_end_date"))
        used_default_window = False
        if not window_start_date or not window_end_date:
            window_start_date = _resolution_date_local(resolution_ts)
            window_end_date = window_start_date
            used_default_window = True

        if window_start_date > window_end_date:
            window_start_date, window_end_date = window_end_date, window_start_date

        cache_key = (ghcn_station_id, window_start_date, window_end_date)
        if cache_key in obs_cache:
            obs_total_raw = obs_cache[cache_key]
        else:
            try:
                obs_total_raw, _ = _fetch_ghcn_snow(session, token, *cache_key)
            except Exception:
                obs_failures += 1
                obs_total_raw = None
            obs_cache[cache_key] = obs_total_raw

        obs_total_in = None
        obs_event_hit = None
        if obs_total_raw is not None:
            obs_total_in = obs_total_raw / 25.4
            obs_event_hit = 1 if obs_total_in >= threshold_in else 0

        if used_default_window:
            logger.info(
                "default_window_used market=%s date=%s",
                market_ticker,
                window_start_date,
            )

        rows.append(
            {
                "market_ticker": market_ticker,
                "city": city,
                "threshold_in": threshold_in,
                "resolution_ts": resolution_ts,
                "t_anchor": t_anchor,
                "p_market": p_market,
                "wpc_day_bucket": wpc.day_bucket,
                "wpc_layer_id": wpc.layer_id,
                "wpc_issue_time": wpc.issue_time,
                "wpc_valid_time": wpc.valid_time,
                "wpc_start_time": wpc.start_time,
                "wpc_end_time": wpc.end_time,
                "wpc_outlook": wpc.outlook,
                "wpc_prob_lo": wpc.prob_lo,
                "wpc_prob_hi": wpc.prob_hi,
                "ghcn_station_id": ghcn_station_id,
                "obs_total_snow_raw": obs_total_raw,
                "obs_total_snow_in": obs_total_in,
                "obs_event_hit": obs_event_hit,
            }
        )

    output_df = pd.DataFrame(rows)
    csv_path = output_dir / "weather_research_v0.csv"
    parquet_path = output_dir / "weather_research_v0.parquet"
    output_df.to_csv(csv_path, index=False)

    try:
        output_df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        raise RuntimeError("Parquet output failed. Install pyarrow or fastparquet.") from exc

    logger.info("markets_processed=%d output_csv=%s output_parquet=%s", len(output_df), csv_path, parquet_path)
    if skip_reasons:
        logger.info("markets_skipped=%s", json.dumps(skip_reasons, sort_keys=True))
    logger.info("wpc_query_misses=%d observation_fetch_failures=%d", wpc_misses, obs_failures)


def main() -> None:
    load_dotenv()
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(description="Build WPC/GHCN research dataset.")
    parser.add_argument("--kalshi_dataset", required=True, help="Path to Kalshi market dataset.")
    parser.add_argument("--candles_dir", required=True, help="Directory with 1-minute candle files per market.")
    parser.add_argument(
        "--ncei_token_env",
        default="NCEI_TOKEN",
        help="Environment variable name holding the NCEI token.",
    )
    args = parser.parse_args()

    build_dataset(
        Path(args.kalshi_dataset),
        Path(args.candles_dir),
        args.ncei_token_env,
        repo_root / "research_outputs",
    )


if __name__ == "__main__":
    main()
