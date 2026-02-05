#!/usr/bin/env python3
"""Phase 1 Polymarket wallet ingestion."""

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

DEFAULT_WALLET = "0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee"
BASE_URL = "https://data-api.polymarket.com"


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _safe_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except Exception:
        return None


def _get_any(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _parse_ts(val: Any) -> Optional[dt.datetime]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        # Heuristic: milliseconds if too large
        if val > 1e12:
            return dt.datetime.fromtimestamp(val / 1000.0, tz=dt.timezone.utc)
        return dt.datetime.fromtimestamp(val, tz=dt.timezone.utc)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return dt.datetime.fromisoformat(s)
        except Exception:
            pass
        try:
            from dateutil import parser  # type: ignore

            return parser.isoparse(val)
        except Exception:
            return None
    return None


def _normalize_price(raw_price: Any) -> Tuple[Optional[float], Optional[int]]:
    price = _safe_float(raw_price)
    if price is None:
        return None, None
    # If price looks like a probability (0-1), treat as dollars
    if price <= 1.0:
        return price, int(round(price * 100))
    # If price looks like cents (1-100), convert to dollars
    if 1.0 < price <= 100.0:
        return price / 100.0, int(round(price))
    # Otherwise assume already dollars
    return price, int(round(price * 100))


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=True)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _synthesize_trade_id(
    tx_hash: Optional[str],
    log_index: Optional[Any],
    ts: Optional[dt.datetime],
    market_id: Optional[str],
    side: Optional[str],
    price: Optional[float],
    size: Optional[float],
) -> str:
    if tx_hash:
        if log_index is not None:
            return f"{tx_hash}:{log_index}"
        return tx_hash
    ts_s = ts.isoformat() if ts else "unknown_ts"
    return f"{ts_s}:{market_id}:{side}:{price}:{size}"


def _row_from_obj(obj: Dict[str, Any], source: str, wallet: str) -> Tuple[Dict[str, Any], bool, str]:
    raw_json = _canonical_json(obj)
    raw_hash = _hash_text(raw_json)
    trade_id = _get_any(obj, ["trade_id", "id", "fillId", "fill_id"])
    has_native_trade_id = trade_id is not None
    tx_hash = _get_any(obj, ["tx_hash", "txHash", "hash"])
    log_index = _get_any(obj, ["log_index", "logIndex", "logIndexHex"])
    market_id = _get_any(obj, ["market_id", "marketId", "conditionId", "condition_id"])
    event_id = _get_any(obj, ["event_id", "eventId"])
    side = _get_any(obj, ["side", "takerSide", "makerSide", "direction"])
    outcome = _get_any(obj, ["outcome", "outcomeId", "outcome_id", "token"])
    raw_price = _get_any(obj, ["price", "pricePerShare", "avgPrice", "fillPrice"])
    price, price_cents = _normalize_price(raw_price)
    size = _safe_float(_get_any(obj, ["size", "shares", "amount", "quantity"]))
    amount_usd = _safe_float(_get_any(obj, ["amount_usd", "amount", "value", "cost"]))
    maker_taker = _get_any(obj, ["maker_taker", "role", "sideType", "type"])
    ts = _parse_ts(_get_any(obj, ["timestamp", "createdAt", "created_at", "time", "blockTime"]))
    block_number = _safe_int(_get_any(obj, ["block_number", "blockNumber", "block"]))

    if not trade_id:
        trade_id = _synthesize_trade_id(tx_hash, log_index, ts, market_id, side, price, size)

    row = {
        "source": source,
        "trade_id": str(trade_id),
        "tx_hash": tx_hash,
        "wallet": wallet,
        "market_id": market_id,
        "event_id": event_id,
        "side": side,
        "outcome": outcome,
        "price": price,
        "price_cents": price_cents,
        "size": size,
        "amount_usd": amount_usd,
        "maker_taker": maker_taker,
        "timestamp": ts.isoformat() if ts else None,
        "block_number": block_number,
        "raw_json": raw_json,
    }
    return row, has_native_trade_id, raw_hash


def _request_with_backoff(
    session: requests.Session, url: str, params: Dict[str, Any], sleep_s: float, max_retries: int = 6
) -> Tuple[Optional[requests.Response], Optional[str]]:
    backoff = 1.0
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            last_err = str(exc)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue

        if resp.status_code in (429, 500, 502, 503, 504):
            last_err = f"HTTP {resp.status_code}"
            # Honor Retry-After if present
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


def _fetch_pages(
    session: requests.Session,
    endpoint: str,
    wallet: str,
    limit: int,
    sleep_s: float,
    max_pages: Optional[int],
    start_offset: int,
    pages_dir: Optional[str],
    debug_pages_dir: Optional[str],
    debug_pages_max: int,
    log_fn,
) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    errors: List[Dict[str, Any]] = []
    offset = start_offset
    page_num = 0
    seen_page_sigs = set()

    while True:
        if max_pages is not None and page_num >= max_pages:
            warnings.append(f"max_pages_reached_at_offset_{offset}")
            break

        params = {"user": wallet, "limit": limit, "offset": offset}
        if endpoint == "trades":
            params["takerOnly"] = "false"

        url = f"{BASE_URL}/{endpoint}"
        resp, err = _request_with_backoff(session, url, params, sleep_s)
        if resp is None:
            warnings.append(f"request_failed_offset_{offset}:{err}")
            break

        if resp.status_code != 200:
            body_snippet = resp.text[:500]
            errors.append(
                {
                    "endpoint": endpoint,
                    "offset": offset,
                    "status_code": resp.status_code,
                    "body_snippet": body_snippet,
                }
            )
            if resp.status_code == 400:
                body_lower = body_snippet.lower()
                if (
                    "offset" in body_lower
                    and (
                        "range" in body_lower
                        or "exceed" in body_lower
                        or "greater" in body_lower
                        or "max" in body_lower
                        or "limit" in body_lower
                    )
                ):
                    warnings.append(f"http_400_eof_offset_{offset}")
                    break
            warnings.append(f"http_{resp.status_code}_offset_{offset}")
            break

        try:
            data = resp.json()
        except Exception:
            warnings.append(f"invalid_json_offset_{offset}")
            break

        if not isinstance(data, list):
            warnings.append(f"unexpected_payload_offset_{offset}")
            break

        page_len = len(data)
        sig = None
        if page_len:
            sig = _hash_text(_canonical_json(data))
            if sig in seen_page_sigs:
                warnings.append(f"repeated_page_offset_{offset}")
                break
            seen_page_sigs.add(sig)

        if pages_dir:
            os.makedirs(pages_dir, exist_ok=True)
            page_path = os.path.join(pages_dir, f"page_{offset}.json")
            with open(page_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

        if debug_pages_dir and page_num < debug_pages_max:
            os.makedirs(debug_pages_dir, exist_ok=True)
            debug_path = os.path.join(
                debug_pages_dir, f"debug_page_{page_num}_offset_{offset}.json"
            )
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

        rows.extend(data)
        log_fn(f"offset={offset} fetched={page_len} cumulative={len(rows)}")

        if page_len < limit:
            break

        offset += page_len
        page_num += 1

    return rows, warnings, errors


def _dedupe_rows(
    rows: List[Dict[str, Any]], source: str, wallet: str
) -> Tuple[List[Dict[str, Any]], int, Dict[str, int]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    dupes = 0
    breakdown = {"trade_id": 0, "raw_hash": 0, "trade_id_collision": 0}
    trade_id_to_hash: Dict[str, str] = {}
    for obj in rows:
        row, has_native_trade_id, raw_hash = _row_from_obj(obj, source, wallet)
        trade_id = row["trade_id"]
        if has_native_trade_id:
            prev = trade_id_to_hash.get(trade_id)
            if prev is not None and prev != raw_hash:
                breakdown["trade_id_collision"] += 1
            else:
                trade_id_to_hash[trade_id] = raw_hash
            key = ("id", trade_id)
        else:
            key = ("raw", raw_hash)
        if key in seen:
            dupes += 1
            if has_native_trade_id:
                breakdown["trade_id"] += 1
            else:
                breakdown["raw_hash"] += 1
            continue
        seen.add(key)
        out.append(row)
    return out, dupes, breakdown


def _write_outputs(rows: List[Dict[str, Any]], outdir: str, fmt: str) -> Tuple[Optional[str], Optional[str]]:
    os.makedirs(outdir, exist_ok=True)
    parquet_path = os.path.join(outdir, "raw_trades.parquet")
    csv_path = os.path.join(outdir, "raw_trades.csv")

    if fmt in ("parquet", "both"):
        try:
            import pandas as pd  # type: ignore

            df = pd.DataFrame(rows)
            df.to_parquet(parquet_path, index=False)
        except Exception as exc:
            raise RuntimeError(f"failed_to_write_parquet:{exc}")

    if fmt in ("csv", "both"):
        try:
            import pandas as pd  # type: ignore

            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
        except Exception as exc:
            raise RuntimeError(f"failed_to_write_csv:{exc}")

    return (parquet_path if fmt in ("parquet", "both") else None), (
        csv_path if fmt in ("csv", "both") else None
    )


def _load_resume_state(outdir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(outdir, "resume_state.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_resume_state(outdir: str, endpoint: str, offset: int) -> None:
    path = os.path.join(outdir, "resume_state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"endpoint": endpoint, "offset": offset}, f)


def _fetch_with_limit_fallback(
    session: requests.Session,
    endpoint: str,
    wallet: str,
    limit: int,
    sleep_s: float,
    max_pages: Optional[int],
    start_offset: int,
    pages_dir: Optional[str],
    debug_pages_dir: Optional[str],
    debug_pages_max: int,
    log_fn,
) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]], int]:
    rows, warnings, errors = _fetch_pages(
        session,
        endpoint,
        wallet,
        limit,
        sleep_s,
        max_pages,
        start_offset,
        pages_dir=pages_dir,
        debug_pages_dir=debug_pages_dir,
        debug_pages_max=debug_pages_max,
        log_fn=log_fn,
    )

    if any(w.startswith("repeated_page_") for w in warnings) and limit > 200:
        log_fn(f"repeated page detected at limit={limit}; retrying with limit=200 from offset=0")
        rows2, warnings2, errors2 = _fetch_pages(
            session,
            endpoint,
            wallet,
            200,
            sleep_s,
            max_pages,
            0,
            pages_dir=pages_dir,
            debug_pages_dir=debug_pages_dir,
            debug_pages_max=debug_pages_max,
            log_fn=log_fn,
        )
        warnings2.append(f"limit_reduced_from_{limit}_to_200")
        warnings2.append("retry_from_offset_0")
        # Prefer the retried dataset if it's larger or if it paginates cleanly
        if rows2 and (
            len(rows2) > len(rows)
            or not any(w.startswith("repeated_page_") for w in warnings2)
        ):
            return rows2, warnings + warnings2, errors + errors2, 200
        warnings += warnings2
        errors += errors2

    return rows, warnings, errors, limit


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 Polymarket wallet ingestion")
    parser.add_argument("--wallet", default=DEFAULT_WALLET)
    parser.add_argument("--outdir", default="./kch-strat/phase1_out")
    parser.add_argument("--endpoint", choices=["auto", "trades", "activity"], default="auto")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max_pages", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--format", choices=["parquet", "csv", "both"], default="parquet")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "phase1_log.txt")

    def log_fn(msg: str) -> None:
        ts = dt.datetime.now(tz=dt.timezone.utc).isoformat()
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    session = requests.Session()
    debug_pages_dir = os.path.join(args.outdir, "pages")
    debug_pages_max = 3

    report: Dict[str, Any] = {
        "wallet": args.wallet,
        "endpoint": None,
        "rows": 0,
        "unique_trade_ids": 0,
        "duplicate_rows_removed": 0,
        "duplicate_breakdown": {},
        "first_ts_utc": None,
        "last_ts_utc": None,
        "unique_markets_count": 0,
        "pagination_warnings": [],
        "pagination_complete": None,
        "offset_ignored_suspected": None,
        "repeat_page_count": 0,
        "limit_requested": args.limit,
        "limit_used": None,
        "http_error_responses": [],
        "eof_via_http_400": False,
        "traded_check": None,
        "errors": [],
    }

    # Validate wallet has traded
    traded_url = f"{BASE_URL}/traded"
    try:
        resp = session.get(traded_url, params={"user": args.wallet}, timeout=30)
        report["traded_check"] = {
            "status_code": resp.status_code,
            "body": resp.text[:500],
        }
    except Exception as exc:
        report["traded_check"] = {"error": str(exc)}

    endpoints = [args.endpoint] if args.endpoint != "auto" else ["activity", "trades"]

    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    used_endpoint = None
    used_source = None
    used_limit: Optional[int] = None
    best_rows: List[Dict[str, Any]] = []
    best_warnings: List[str] = []
    best_endpoint: Optional[str] = None
    best_limit: Optional[int] = None
    error_responses: List[Dict[str, Any]] = []

    for endpoint in endpoints:
        start_offset = 0
        if args.resume:
            state = _load_resume_state(args.outdir)
            if state and state.get("endpoint") == endpoint:
                start_offset = int(state.get("offset", 0))
            else:
                # fallback: infer from existing file length if possible
                parquet_path = os.path.join(args.outdir, "raw_trades.parquet")
                csv_path = os.path.join(args.outdir, "raw_trades.csv")
                try:
                    import pandas as pd  # type: ignore

                    if os.path.exists(parquet_path):
                        start_offset = len(pd.read_parquet(parquet_path))
                    elif os.path.exists(csv_path):
                        start_offset = len(pd.read_csv(csv_path))
                except Exception:
                    pass

        log_fn(f"starting endpoint={endpoint} offset={start_offset}")
        page_rows, page_warnings, page_errors, limit_used = _fetch_with_limit_fallback(
            session,
            endpoint,
            args.wallet,
            args.limit,
            args.sleep,
            args.max_pages,
            start_offset,
            pages_dir=os.path.join(args.outdir, "pages") if args.resume else None,
            debug_pages_dir=debug_pages_dir,
            debug_pages_max=debug_pages_max,
            log_fn=log_fn,
        )

        error_responses.extend(page_errors)

        if page_rows:
            # Track the best dataset seen so far (by row count)
            if not best_rows or len(page_rows) > len(best_rows):
                best_rows = page_rows
                best_warnings = page_warnings
                best_endpoint = endpoint
                best_limit = limit_used

            # If no pagination repetition, accept immediately
            if not any(w.startswith("repeated_page_") for w in page_warnings):
                rows = page_rows
                warnings = page_warnings
                used_endpoint = endpoint
                used_source = "data_api_trades" if endpoint == "trades" else "data_api_activity"
                used_limit = limit_used
                _save_resume_state(args.outdir, endpoint, start_offset + len(page_rows))
                break

        warnings.extend(page_warnings)

    if not rows:
        if best_rows:
            rows = best_rows
            warnings = warnings + best_warnings
            used_endpoint = best_endpoint
            used_source = "data_api_trades" if best_endpoint == "trades" else "data_api_activity"
            used_limit = best_limit
            _save_resume_state(args.outdir, best_endpoint, len(best_rows))
        else:
            report["endpoint"] = used_endpoint
            report["pagination_warnings"] = warnings
            report["errors"].append("no_data_returned")
            report_path = os.path.join(args.outdir, "phase1_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            log_fn("no data returned from any endpoint")
            return 2

    deduped, dupes, dup_breakdown = _dedupe_rows(rows, used_source, args.wallet)

    # Report stats
    report["endpoint"] = used_endpoint
    report["rows"] = len(deduped)
    report["unique_trade_ids"] = len({r["trade_id"] for r in deduped})
    report["duplicate_rows_removed"] = dupes
    report["duplicate_breakdown"] = dup_breakdown
    report["unique_markets_count"] = len({r["market_id"] for r in deduped if r["market_id"]})
    report["pagination_warnings"] = warnings
    report["limit_used"] = used_limit
    report["http_error_responses"] = error_responses

    eof_via_http_400 = any(w.startswith("http_400_eof_") for w in warnings)
    report["eof_via_http_400"] = eof_via_http_400

    repeat_page_count = len([w for w in warnings if w.startswith("repeated_page_")])
    report["repeat_page_count"] = repeat_page_count
    report["offset_ignored_suspected"] = repeat_page_count > 0
    incomplete_prefixes = (
        "repeated_page_",
        "max_pages_reached",
        "request_failed_",
        "http_",
        "invalid_json",
        "unexpected_payload",
    )
    report["pagination_complete"] = not any(
        w.startswith(incomplete_prefixes) and not w.startswith("http_400_eof_")
        for w in warnings
    )

    ts_vals = [r["timestamp"] for r in deduped if r["timestamp"]]
    if ts_vals:
        report["first_ts_utc"] = min(ts_vals)
        report["last_ts_utc"] = max(ts_vals)

    try:
        _write_outputs(deduped, args.outdir, args.format)
    except Exception as exc:
        report["errors"].append(str(exc))
        report_path = os.path.join(args.outdir, "phase1_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log_fn(f"failed writing outputs: {exc}")
        return 3

    report_path = os.path.join(args.outdir, "phase1_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log_fn(f"completed rows={report['rows']} dupes_removed={dupes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
