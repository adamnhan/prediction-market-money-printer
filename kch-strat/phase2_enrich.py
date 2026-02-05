#!/usr/bin/env python3
"""Phase 2 enrichment for Polymarket wallet trades using Gamma API."""

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


HEX_66_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")


def _safe_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    return s if s else None


def _safe_list(val: Any) -> List[Any]:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def _get_any(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _parse_ts(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if val > 1e12:
            return dt.datetime.fromtimestamp(val / 1000.0, tz=dt.timezone.utc).isoformat()
        return dt.datetime.fromtimestamp(val, tz=dt.timezone.utc).isoformat()
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return dt.datetime.fromisoformat(s).isoformat()
        except Exception:
            pass
        try:
            from dateutil import parser  # type: ignore

            return parser.isoparse(val).isoformat()
        except Exception:
            return None
    return None


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=True)


def _request_with_backoff(
    session: requests.Session, url: str, params: Dict[str, Any], sleep_s: float, max_retries: int = 6
) -> Tuple[Optional[requests.Response], Optional[str]]:
    backoff = 1.0
    last_err = None
    for _ in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            last_err = str(exc)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue

        if resp.status_code in (429, 500, 502, 503, 504):
            last_err = f"HTTP {resp.status_code}"
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    time.sleep(float(retry_after))
                except Exception:
                    time.sleep(backoff)
            else:
                time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue

        time.sleep(sleep_s)
        return resp, None

    return None, last_err


def _parse_gamma_page(resp_json: Any, kind: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    next_cursor = None
    if isinstance(resp_json, list):
        return resp_json, None
    if not isinstance(resp_json, dict):
        return [], None

    # Common shapes
    for key in ("data", kind, f"{kind}s"):
        if key in resp_json and isinstance(resp_json[key], list):
            data = resp_json[key]
            break
    else:
        data = []

    for cursor_key in ("next", "nextCursor", "next_cursor", "cursor", "nextPage"):
        if cursor_key in resp_json and resp_json[cursor_key]:
            next_cursor = str(resp_json[cursor_key])
            break

    return data, next_cursor


def _load_cached_pages(cache_dir: str, prefix: str) -> Tuple[List[Dict[str, Any]], Optional[int], Optional[str]]:
    if not os.path.isdir(cache_dir):
        return [], None, None
    pages = []
    offsets = []
    cursors = []
    for name in sorted(os.listdir(cache_dir)):
        if not name.startswith(prefix):
            continue
        path = os.path.join(cache_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            pages.append(payload)
            if isinstance(payload, dict):
                offsets.append(payload.get("offset"))
                cursors.append(payload.get("next_cursor"))
        except Exception:
            continue
    last_offset = None
    last_cursor = None
    if offsets:
        last_offset = offsets[-1]
    if cursors:
        last_cursor = cursors[-1]
    return pages, last_offset, last_cursor


def _fetch_gamma_paginated(
    session: requests.Session,
    base_url: str,
    kind: str,
    limit: int,
    sleep_s: float,
    max_pages: int,
    cache_dir: str,
    cache_prefix: str,
    resume: bool,
    required_ids: Optional[set],
    required_ratio: float,
    log_fn,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []
    os.makedirs(cache_dir, exist_ok=True)

    start_page = 0
    offset = 0
    cursor = None

    if resume:
        cached, last_offset, last_cursor = _load_cached_pages(cache_dir, cache_prefix)
        for payload in cached:
            if isinstance(payload, dict) and "data" in payload:
                rows.extend(payload["data"])
        if cached:
            start_page = len(cached)
            if last_offset is not None:
                try:
                    offset = int(last_offset)
                except Exception:
                    offset = 0
            cursor = last_cursor
            log_fn(f"loaded {len(cached)} cached {kind} pages")

    needed = required_ids or set()
    total_needed = len(needed)
    found = set()

    for page_idx in range(start_page, max_pages):
        params: Dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        else:
            params["offset"] = offset

        url = f"{base_url}/{kind}"
        resp, err = _request_with_backoff(session, url, params, sleep_s)
        if resp is None:
            warnings.append(f"{kind}_request_failed:{err}")
            break
        if resp.status_code != 200:
            warnings.append(f"{kind}_http_{resp.status_code}")
            break

        try:
            resp_json = resp.json()
        except Exception:
            warnings.append(f"{kind}_invalid_json")
            break

        data, next_cursor = _parse_gamma_page(resp_json, kind)
        if not data:
            break

        rows.extend(data)

        cache_payload = {
            "page": page_idx,
            "offset": offset,
            "cursor": cursor,
            "next_cursor": next_cursor,
            "data": data,
        }
        cache_path = os.path.join(cache_dir, f"{cache_prefix}{page_idx}.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_payload, f)

        # Early stop if we've found all required ids and coverage is high
        if needed:
            for m in data:
                cid = _safe_str(_get_any(m, ["conditionId", "condition_id"]))
                if cid:
                    found.add(cid)
            if total_needed > 0 and len(found) >= total_needed and required_ratio >= 0.7:
                break

        if next_cursor:
            cursor = next_cursor
        else:
            offset += len(data)
            if len(data) < limit:
                break

        log_fn(f"fetched {kind} page={page_idx} rows={len(data)} total={len(rows)}")

    if len(rows) == 0:
        warnings.append(f"{kind}_no_data")

    return rows, warnings


def _fetch_events_by_id(
    session: requests.Session,
    base_url: str,
    event_ids: Iterable[str],
    sleep_s: float,
    cache_dir: str,
    resume: bool,
    log_fn,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []
    os.makedirs(cache_dir, exist_ok=True)

    total = len(list(event_ids))
    for idx, event_id in enumerate(event_ids):
        if not event_id:
            continue
        cache_path = os.path.join(cache_dir, f"event_id_{event_id}.json")
        if resume and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    rows.append(payload)
                    continue
            except Exception:
                pass

        url = f"{base_url}/events/{event_id}"
        resp, err = _request_with_backoff(session, url, {}, sleep_s)
        if resp is None:
            warnings.append(f"events_id_request_failed:{event_id}:{err}")
            continue
        if resp.status_code != 200:
            warnings.append(f"events_id_http_{resp.status_code}:{event_id}")
            continue

        try:
            payload = resp.json()
        except Exception:
            warnings.append(f"events_id_invalid_json:{event_id}")
            continue

        def _cond_matches(item: Dict[str, Any], expected: str) -> bool:
            got = _safe_str(_get_any(item, ["conditionId", "condition_id"]))
            return bool(got) and got.lower() == expected.lower()

        if isinstance(payload, list):
            matched = 0
            for item in payload:
                if isinstance(item, dict) and _cond_matches(item, condition_id):
                    rows.append(item)
                    matched += 1
            if matched == 0:
                warnings.append(f"markets_condition_no_match:{condition_id}")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        elif isinstance(payload, dict):
            if _cond_matches(payload, condition_id):
                rows.append(payload)
            else:
                warnings.append(f"markets_condition_no_match:{condition_id}")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

        if idx % 100 == 0 and total:
            log_fn(f"fetched events by id {idx} of {total}")

    if len(rows) == 0:
        warnings.append("events_id_no_data")
    return rows, warnings


def _fetch_events_by_slug(
    session: requests.Session,
    base_url: str,
    slugs: Iterable[str],
    sleep_s: float,
    cache_dir: str,
    resume: bool,
    log_fn,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []
    os.makedirs(cache_dir, exist_ok=True)

    slugs_list = [s for s in slugs if s]
    total = len(slugs_list)
    for idx, slug in enumerate(slugs_list):
        cache_path = os.path.join(cache_dir, f"event_slug_{slug}.json")
        if resume and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, dict):
                            rows.append(item)
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
                    continue
            except Exception:
                pass

        url = f"{base_url}/events"
        resp, err = _request_with_backoff(session, url, {"slug": slug}, sleep_s)
        if resp is None:
            warnings.append(f"events_slug_request_failed:{slug}:{err}")
            continue
        if resp.status_code != 200:
            warnings.append(f"events_slug_http_{resp.status_code}:{slug}")
            continue

        try:
            payload = resp.json()
        except Exception:
            warnings.append(f"events_slug_invalid_json:{slug}")
            continue

        def _slug_matches(item: Dict[str, Any], expected: str) -> bool:
            got = _safe_str(_get_any(item, ["slug", "event_slug"]))
            return bool(got) and got.lower() == expected.lower()

        if isinstance(payload, list):
            matched = 0
            for item in payload:
                if isinstance(item, dict) and _slug_matches(item, slug):
                    rows.append(item)
                    matched += 1
            if matched == 0:
                warnings.append(f"events_slug_no_match:{slug}")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        elif isinstance(payload, dict):
            if _slug_matches(payload, slug):
                rows.append(payload)
            else:
                warnings.append(f"events_slug_no_match:{slug}")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

        if idx % 100 == 0 and total:
            log_fn(f"fetched events by slug {idx} of {total}")

    if len(rows) == 0:
        warnings.append("events_slug_no_data")
    return rows, warnings


def _fetch_markets_by_id(
    session: requests.Session,
    base_url: str,
    market_ids: Iterable[str],
    sleep_s: float,
    cache_dir: str,
    resume: bool,
    log_fn,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []
    os.makedirs(cache_dir, exist_ok=True)

    market_ids_list = [m for m in market_ids if m]
    total = len(market_ids_list)
    for idx, market_id in enumerate(market_ids_list):
        cache_path = os.path.join(cache_dir, f"market_id_{market_id}.json")
        if resume and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    rows.append(payload)
                    continue
            except Exception:
                pass

        url = f"{base_url}/markets/{market_id}"
        resp, err = _request_with_backoff(session, url, {}, sleep_s)
        if resp is None:
            warnings.append(f"markets_id_request_failed:{market_id}:{err}")
            continue
        if resp.status_code != 200:
            warnings.append(f"markets_id_http_{resp.status_code}:{market_id}")
            continue

        try:
            payload = resp.json()
        except Exception:
            warnings.append(f"markets_id_invalid_json:{market_id}")
            continue

        if isinstance(payload, dict):
            rows.append(payload)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

        if idx % 100 == 0 and total:
            log_fn(f"fetched markets by id {idx} of {total}")

    if len(rows) == 0:
        warnings.append("markets_id_no_data")
    return rows, warnings


def _fetch_markets_by_condition_id(
    session: requests.Session,
    base_url: str,
    condition_ids: Iterable[str],
    sleep_s: float,
    cache_dir: str,
    resume: bool,
    log_fn,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []
    os.makedirs(cache_dir, exist_ok=True)

    condition_ids_list = [c for c in condition_ids if c]
    total = len(condition_ids_list)
    for idx, condition_id in enumerate(condition_ids_list):
        cache_path = os.path.join(cache_dir, f"condition_id_{condition_id}.json")
        def _cond_matches(item: Dict[str, Any], expected: str) -> bool:
            got = _safe_str(_get_any(item, ["conditionId", "condition_id"]))
            return bool(got) and got.lower() == expected.lower()

        if resume and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    matched = 0
                    for item in payload:
                        if isinstance(item, dict) and _cond_matches(item, condition_id):
                            rows.append(item)
                            matched += 1
                    if matched == 0:
                        warnings.append(f"markets_condition_no_match:{condition_id}")
                    continue
                if isinstance(payload, dict):
                    if _cond_matches(payload, condition_id):
                        rows.append(payload)
                    else:
                        warnings.append(f"markets_condition_no_match:{condition_id}")
                    continue
            except Exception:
                pass

        url = f"{base_url}/markets"
        resp, err = _request_with_backoff(
            session,
            url,
            {"conditionId": condition_id},
            sleep_s,
        )
        if resp is None:
            warnings.append(f"markets_condition_request_failed:{condition_id}:{err}")
            continue
        if resp.status_code != 200:
            warnings.append(f"markets_condition_http_{resp.status_code}:{condition_id}")
            continue

        try:
            payload = resp.json()
        except Exception:
            warnings.append(f"markets_condition_invalid_json:{condition_id}")
            continue

        if isinstance(payload, list):
            matched = 0
            for item in payload:
                if isinstance(item, dict) and _cond_matches(item, condition_id):
                    rows.append(item)
                    matched += 1
            if matched == 0:
                warnings.append(f"markets_condition_no_match:{condition_id}")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        elif isinstance(payload, dict):
            if _cond_matches(payload, condition_id):
                rows.append(payload)
            else:
                warnings.append(f"markets_condition_no_match:{condition_id}")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

        if idx % 100 == 0 and total:
            log_fn(f"fetched markets by conditionId {idx} of {total}")

    if len(rows) == 0:
        warnings.append("markets_condition_no_data")
    return rows, warnings


def _fetch_markets_by_slug(
    session: requests.Session,
    base_url: str,
    slugs: Iterable[str],
    sleep_s: float,
    cache_dir: str,
    resume: bool,
    log_fn,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []
    os.makedirs(cache_dir, exist_ok=True)

    slugs_list = [s for s in slugs if s]
    total = len(slugs_list)
    for idx, slug in enumerate(slugs_list):
        cache_path = os.path.join(cache_dir, f"slug_{slug}.json")
        def _slug_matches(item: Dict[str, Any], expected: str) -> bool:
            got = _safe_str(_get_any(item, ["slug", "market_slug"]))
            return bool(got) and got.lower() == expected.lower()

        if resume and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    matched = 0
                    for item in payload:
                        if isinstance(item, dict) and _slug_matches(item, slug):
                            rows.append(item)
                            matched += 1
                    if matched == 0:
                        warnings.append(f"markets_slug_no_match:{slug}")
                    continue
                if isinstance(payload, dict):
                    if _slug_matches(payload, slug):
                        rows.append(payload)
                    else:
                        warnings.append(f"markets_slug_no_match:{slug}")
                    continue
            except Exception:
                pass

        url = f"{base_url}/markets"
        resp, err = _request_with_backoff(session, url, {"slug": slug}, sleep_s)
        if resp is None:
            warnings.append(f"markets_slug_request_failed:{slug}:{err}")
            continue
        if resp.status_code != 200:
            warnings.append(f"markets_slug_http_{resp.status_code}:{slug}")
            continue

        try:
            payload = resp.json()
        except Exception:
            warnings.append(f"markets_slug_invalid_json:{slug}")
            continue

        if isinstance(payload, list):
            matched = 0
            for item in payload:
                if isinstance(item, dict) and _slug_matches(item, slug):
                    rows.append(item)
                    matched += 1
            if matched == 0:
                warnings.append(f"markets_slug_no_match:{slug}")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        elif isinstance(payload, dict):
            if _slug_matches(payload, slug):
                rows.append(payload)
            else:
                warnings.append(f"markets_slug_no_match:{slug}")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

        if idx % 100 == 0 and total:
            log_fn(f"fetched markets by slug {idx} of {total}")

    if len(rows) == 0:
        warnings.append("markets_slug_no_data")
    return rows, warnings


def _normalize_condition_id(market_id: Any) -> Optional[str]:
    s = _safe_str(market_id)
    if not s:
        return None
    if HEX_66_RE.match(s):
        return s.lower()
    return None


def _looks_yes_no(outcomes: List[str]) -> bool:
    yn = {"yes", "no", "y", "n", "true", "false"}
    if len(outcomes) != 2:
        return False
    return all(o.strip().lower() in yn for o in outcomes if isinstance(o, str))


def _guess_league(text: str) -> Optional[str]:
    t = text.lower()
    if "nhl" in t or "hockey" in t:
        return "NHL"
    if "nba" in t:
        return "NBA"
    if "wnba" in t:
        return "WNBA"
    if "mlb" in t or "baseball" in t:
        return "MLB"
    if "nfl" in t or "football" in t:
        return "NFL"
    if "ncaa" in t:
        return "NCAA"
    if "premier league" in t or "epl" in t:
        return "EPL"
    if "mls" in t:
        return "MLS"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 Polymarket wallet enrichment")
    parser.add_argument("--phase1_path", default="./kch-strat/phase1_out/raw_trades.parquet")
    parser.add_argument("--outdir", default="./kch-strat/phase2_out")
    parser.add_argument("--gamma_base", default="https://gamma-api.polymarket.com")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--max_pages_markets", type=int, default=5000)
    parser.add_argument("--max_pages_events", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cache_dir = os.path.join(args.outdir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    def log_fn(msg: str) -> None:
        ts = dt.datetime.now(tz=dt.timezone.utc).isoformat()
        print(f"[{ts}] {msg}")

    if not os.path.exists(args.phase1_path):
        print(f"missing phase1 parquet: {args.phase1_path}")
        return 2

    phase1_path = args.phase1_path
    if not os.path.exists(phase1_path) and os.path.exists("./phase1_out/raw_trades.parquet"):
        phase1_path = "./phase1_out/raw_trades.parquet"
    trades = pd.read_parquet(phase1_path)
    if "market_id" not in trades.columns:
        print("phase1 parquet missing market_id")
        return 2

    trades["market_id_raw"] = trades["market_id"].apply(_safe_str)
    trades["condition_id_norm"] = trades["market_id"].apply(_normalize_condition_id)
    if "raw_json" in trades.columns:
        def _extract_json_field(raw: Any, key: str) -> Optional[str]:
            if not isinstance(raw, str) or not raw:
                return None
            try:
                payload = json.loads(raw)
            except Exception:
                return None
            val = payload.get(key)
            return _safe_str(val)

        trades["market_slug_raw"] = trades["raw_json"].apply(lambda s: _extract_json_field(s, "slug"))
        trades["event_slug_raw"] = trades["raw_json"].apply(lambda s: _extract_json_field(s, "eventSlug"))

    unique_markets_in_trades = trades["market_id_raw"].nunique(dropna=True)
    cond_set = set([c for c in trades["condition_id_norm"].dropna().unique()])
    cond_ratio = len(cond_set) / unique_markets_in_trades if unique_markets_in_trades else 0.0

    session = requests.Session()

    # Prefer fetch-by-conditionId (these look like conditionIds in activity data),
    # then by slug, then by id (numeric), and fallback to pagination.
    market_ids_list = [m for m in trades["market_id_raw"].dropna().unique().tolist() if m]
    markets_rows, markets_warnings = _fetch_markets_by_condition_id(
        session,
        args.gamma_base,
        market_ids_list,
        args.sleep,
        os.path.join(cache_dir, "markets_by_condition"),
        args.resume,
        log_fn,
    )

    # Try by slug if conditionId lookups were sparse.
    slugs_list = []
    if "market_slug_raw" in trades.columns:
        slugs_list = [s for s in trades["market_slug_raw"].dropna().unique().tolist() if s]
    if not markets_rows and slugs_list:
        slug_rows, slug_warnings = _fetch_markets_by_slug(
            session,
            args.gamma_base,
            slugs_list,
            args.sleep,
            os.path.join(cache_dir, "markets_by_slug"),
            args.resume,
            log_fn,
        )
        markets_rows = markets_rows + slug_rows
        markets_warnings = markets_warnings + slug_warnings

    # Fallback to id endpoint (numeric ids) if needed.
    if not markets_rows:
        id_rows, id_warnings = _fetch_markets_by_id(
            session,
            args.gamma_base,
            market_ids_list,
            args.sleep,
            os.path.join(cache_dir, "markets_by_id"),
            args.resume,
            log_fn,
        )
        markets_rows = markets_rows + id_rows
        markets_warnings = markets_warnings + id_warnings

    # Fallback pagination if per-id coverage is low.
    markets_rows_by_id = markets_rows
    if not markets_rows or len(markets_rows) < max(50, int(len(market_ids_list) * 0.7)):
        paginated_rows, paginated_warnings = _fetch_gamma_paginated(
            session,
            args.gamma_base,
            "markets",
            args.limit,
            args.sleep,
            args.max_pages_markets,
            cache_dir,
            "markets_page_",
            args.resume,
            cond_set,
            cond_ratio,
            log_fn,
        )
        markets_rows = markets_rows_by_id + paginated_rows
        markets_warnings = markets_warnings + paginated_warnings

    # Build lookups
    market_by_condition: Dict[str, Dict[str, Any]] = {}
    by_id: Dict[str, str] = {}
    by_slug: Dict[str, str] = {}

    for m in markets_rows:
        condition_id = _safe_str(_get_any(m, ["conditionId", "condition_id"]))
        if condition_id:
            condition_id = condition_id.lower()
        market_id = _safe_str(_get_any(m, ["id", "marketId", "market_id"]))
        slug = _safe_str(_get_any(m, ["slug", "market_slug"]))

        if condition_id and condition_id not in market_by_condition:
            market_by_condition[condition_id] = m
        if market_id and condition_id:
            by_id[str(market_id)] = condition_id
        if slug and condition_id:
            by_slug[slug.lower()] = condition_id

    # Resolve condition_id_norm for missing
    def _resolve_condition_id(row: pd.Series) -> Optional[str]:
        if pd.notna(row.get("condition_id_norm")):
            return row["condition_id_norm"]
        raw = _safe_str(row.get("market_id_raw"))
        if not raw:
            return None
        if raw in by_id:
            return by_id[raw]
        raw_lower = raw.lower()
        if raw_lower in by_slug:
            return by_slug[raw_lower]
        return None

    trades["condition_id_norm"] = trades.apply(_resolve_condition_id, axis=1)

    # Build markets_dim
    markets_dim_rows: List[Dict[str, Any]] = []
    for condition_id, m in market_by_condition.items():
        outcomes = _get_any(m, ["outcomes", "outcome", "outcomeNames"]) or []
        if isinstance(outcomes, str):
            try:
                parsed = json.loads(outcomes)
                outcomes = parsed if isinstance(parsed, list) else []
            except Exception:
                outcomes = []
        if not isinstance(outcomes, list):
            outcomes = []
        clob = _get_any(m, ["clobTokenIds", "clob_token_ids", "clobTokens"]) or None
        question = _get_any(m, ["question", "title", "name"])
        event_id = _get_any(m, ["eventId", "event_id"])
        event_slug = _get_any(m, ["eventSlug", "event_slug"])
        start_time = _get_any(m, ["startTime", "start_date", "startDate"])
        end_time = _get_any(m, ["endTime", "end_date", "endDate", "closeTime"])
        active_val = _get_any(m, ["active", "isActive"])
        closed_val = _get_any(m, ["closed", "isClosed"])

        markets_dim_rows.append(
            {
                "condition_id": condition_id,
                "market_id": _safe_str(_get_any(m, ["id", "marketId", "market_id"])),
                "slug": _safe_str(_get_any(m, ["slug", "market_slug"])),
                "question": _safe_str(question),
                "outcomes": _canonical_json(outcomes),
                "clob_token_ids": _canonical_json(clob) if clob is not None else None,
                "event_id": _safe_str(event_id),
                "event_slug": _safe_str(event_slug),
                "market_start_time": _parse_ts(start_time),
                "market_end_time": _parse_ts(end_time),
                "active": active_val,
                "closed": closed_val,
                "raw_json": _canonical_json(m),
            }
        )

    markets_dim = pd.DataFrame(markets_dim_rows)
    if markets_dim.empty:
        markets_dim = pd.DataFrame(
            columns=[
                "condition_id",
                "market_id",
                "slug",
                "question",
                "outcomes",
                "clob_token_ids",
                "event_id",
                "event_slug",
                "market_start_time",
                "market_end_time",
                "active",
                "closed",
                "raw_json",
            ]
        )
    else:
        markets_dim = markets_dim.drop_duplicates(subset=["condition_id"])

    # Build events_dim
    # Collect event ids and slugs from markets_dim and raw trades (if present)
    event_ids = set([e for e in markets_dim.get("event_id", pd.Series()).dropna().unique()])
    if "event_id" in trades.columns:
        event_ids.update([e for e in trades["event_id"].dropna().unique()])
    event_ids_list = [str(e) for e in event_ids if e is not None]

    event_slugs = set([e for e in markets_dim.get("event_slug", pd.Series()).dropna().unique()])
    if "event_slug_raw" in trades.columns:
        event_slugs.update([e for e in trades["event_slug_raw"].dropna().unique()])
    event_slugs_list = [str(e) for e in event_slugs if e is not None]

    # Prefer per-id fetch; avoid full pagination to prevent OOM
    if event_ids_list:
        events_rows, events_warnings = _fetch_events_by_id(
            session,
            args.gamma_base,
            event_ids_list,
            args.sleep,
            os.path.join(cache_dir, "events_by_id"),
            args.resume,
            log_fn,
        )
    else:
        events_rows, events_warnings = [], ["events_id_list_empty_skipped_pagination"]

    if event_slugs_list:
        events_rows_by_slug, events_warnings_by_slug = _fetch_events_by_slug(
            session,
            args.gamma_base,
            event_slugs_list,
            args.sleep,
            os.path.join(cache_dir, "events_by_slug"),
            args.resume,
            log_fn,
        )
        events_rows = events_rows + events_rows_by_slug
        events_warnings = events_warnings + events_warnings_by_slug

    events_dim_rows: List[Dict[str, Any]] = []
    for e in events_rows:
        event_id = _safe_str(_get_any(e, ["id", "eventId", "event_id"]))
        slug = _safe_str(_get_any(e, ["slug", "event_slug"]))
        title = _safe_str(_get_any(e, ["title", "name", "question"]))
        category = _safe_str(_get_any(e, ["category", "category_slug"]))
        tags = _get_any(e, ["tags", "tag", "categories"]) or []
        start_time = _get_any(e, ["startTime", "start_date", "startDate"])
        end_time = _get_any(e, ["endTime", "end_date", "endDate", "closeTime"])

        events_dim_rows.append(
            {
                "event_id": event_id,
                "slug": slug,
                "title": title,
                "category": category,
                "start_time": _parse_ts(start_time),
                "end_time": _parse_ts(end_time),
                "tags": _canonical_json(tags),
                "raw_json": _canonical_json(e),
            }
        )

    events_dim = pd.DataFrame(events_dim_rows)
    if events_dim.empty:
        events_dim = pd.DataFrame(
            columns=[
                "event_id",
                "slug",
                "title",
                "category",
                "start_time",
                "end_time",
                "tags",
                "raw_json",
            ]
        )
    else:
        events_dim = events_dim.drop_duplicates(subset=["event_id"])

    # Join
    trades_enriched = trades.merge(
        markets_dim,
        how="left",
        left_on="condition_id_norm",
        right_on="condition_id",
        suffixes=("", "_market"),
    )

    # Promote market-derived event_id when missing on trades.
    if "event_id_market" in trades_enriched.columns:
        trades_enriched["event_id"] = trades_enriched["event_id"].fillna(trades_enriched["event_id_market"])
    if "event_slug_raw" in trades_enriched.columns:
        trades_enriched["event_slug"] = trades_enriched["event_slug_raw"]
    if "event_slug_market" in trades_enriched.columns:
        trades_enriched["event_slug"] = trades_enriched["event_slug"].fillna(
            trades_enriched["event_slug_market"]
        )

    # Normalize event_id types for merge
    if "event_id" in trades_enriched.columns:
        trades_enriched["event_id"] = trades_enriched["event_id"].apply(_safe_str)
    if "event_id" in events_dim.columns:
        events_dim["event_id"] = events_dim["event_id"].apply(_safe_str)

    trades_enriched = trades_enriched.merge(
        events_dim,
        how="left",
        left_on="event_id",
        right_on="event_id",
        suffixes=("", "_event"),
    )
    # Secondary join on event slug for rows missing event_id matches.
    missing_event = trades_enriched["title"].isna()
    if missing_event.any() and "event_slug" in trades_enriched.columns:
        events_by_slug = events_dim.rename(columns={"slug": "event_slug"})
        trades_enriched = trades_enriched.merge(
            events_by_slug,
            how="left",
            left_on="event_slug",
            right_on="event_slug",
            suffixes=("", "_event_slug"),
        )
        for col in ("title", "category", "start_time", "end_time", "tags", "raw_json"):
            slug_col = f"{col}_event_slug"
            if slug_col in trades_enriched.columns:
                trades_enriched[col] = trades_enriched[col].fillna(trades_enriched[slug_col])

    trades_enriched["is_joined_market"] = trades_enriched["condition_id"].notna()
    trades_enriched["is_joined_event"] = trades_enriched["title"].notna()

    # Classification heuristics
    def _infer_market_type(outcomes_json: Any) -> Optional[str]:
        try:
            outcomes = json.loads(outcomes_json) if isinstance(outcomes_json, str) else []
        except Exception:
            outcomes = []
        outcomes = [o for o in outcomes if isinstance(o, str)]
        if len(outcomes) == 2 and not _looks_yes_no(outcomes):
            return "moneyline"
        return None

    def _infer_league(row: pd.Series) -> Optional[str]:
        pieces = []
        for key in ("title", "question", "category", "tags"):
            val = row.get(key)
            if isinstance(val, str):
                pieces.append(val)
            elif val is not None and key == "tags":
                try:
                    pieces.append(" ".join(json.loads(val)))
                except Exception:
                    pass
        text = " ".join(pieces)
        return _guess_league(text)

    def _infer_is_nhl(row: pd.Series) -> bool:
        text = " ".join(
            [
                str(row.get("title") or ""),
                str(row.get("question") or ""),
                str(row.get("category") or ""),
                str(row.get("tags") or ""),
            ]
        ).lower()
        if "nhl" in text or "hockey" in text:
            return True
        try:
            outcomes = json.loads(row.get("outcomes") or "[]")
        except Exception:
            outcomes = []
        outcomes = [o for o in outcomes if isinstance(o, str)]
        if len(outcomes) == 2 and not _looks_yes_no(outcomes) and "hockey" in text:
            return True
        return False

    trades_enriched["market_type_guess"] = trades_enriched["outcomes"].apply(_infer_market_type)
    trades_enriched["league_guess"] = trades_enriched.apply(_infer_league, axis=1)
    trades_enriched["is_nhl_guess"] = trades_enriched.apply(_infer_is_nhl, axis=1)

    # Reports
    rows_in = len(trades)
    rows_out = len(trades_enriched)
    pct_market = float(trades_enriched["is_joined_market"].mean()) if rows_out else 0.0
    pct_event = float(trades_enriched["is_joined_event"].mean()) if rows_out else 0.0
    nhl_count = int(trades_enriched["is_nhl_guess"].sum())
    unique_markets_joined = trades_enriched["condition_id"].nunique(dropna=True)

    missing_market_mask = ~trades_enriched["is_joined_market"]
    sample_missing_market_ids = (
        trades_enriched.loc[missing_market_mask, "market_id_raw"].dropna().unique().tolist()[:50]
    )

    missing_event_mask = trades_enriched["is_joined_market"] & ~trades_enriched["is_joined_event"]
    sample_missing_event_ids = (
        trades_enriched.loc[missing_event_mask, "event_id"].dropna().unique().tolist()[:50]
    )

    report = {
        "wallet": trades["wallet"].iloc[0] if "wallet" in trades.columns and rows_in else None,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "pct_trades_joined_market": pct_market,
        "pct_trades_joined_event": pct_event,
        "unique_markets_in_trades": int(unique_markets_in_trades),
        "unique_markets_joined": int(unique_markets_joined),
        "nhl_guess_trade_count": nhl_count,
        "sample_missing_market_ids": sample_missing_market_ids,
        "sample_missing_event_ids": sample_missing_event_ids,
        "market_fetch_warnings": markets_warnings,
        "event_fetch_warnings": events_warnings,
        "warnings": [],
    }

    if pct_market < 0.95:
        report["warnings"].append(
            "market_join_rate_below_0_95: likely causes include gamma API limits, missing conditionId in trades, or market_id not matching id/slug"
        )

    # Write outputs
    markets_dim.to_parquet(os.path.join(args.outdir, "markets_dim.parquet"), index=False)
    events_dim.to_parquet(os.path.join(args.outdir, "events_dim.parquet"), index=False)
    trades_enriched.to_parquet(os.path.join(args.outdir, "trades_enriched.parquet"), index=False)

    with open(os.path.join(args.outdir, "phase2_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"summary rows={rows_out} market_join={pct_market:.3f} event_join={pct_event:.3f} nhl_guess={nhl_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
