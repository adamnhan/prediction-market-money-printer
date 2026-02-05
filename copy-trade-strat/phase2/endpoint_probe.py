from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from typing import Any

import requests


DEFAULT_BASE_URL = "https://gamma-api.polymarket.com"


def _summarize_json(payload: Any) -> str:
    if isinstance(payload, list):
        return f"json:list len={len(payload)} first_keys={list(payload[0].keys())[:8] if payload else []}"
    if isinstance(payload, dict):
        return f"json:dict keys={list(payload.keys())[:12]}"
    return f"json:{type(payload).__name__}"


def _try_request(session: requests.Session, method: str, url: str, params: dict[str, Any]) -> None:
    try:
        resp = session.request(method, url, params=params, timeout=15)
    except Exception as exc:
        print(f"[ERR] {method} {url} params={params} err={exc}")
        return
    ctype = resp.headers.get("content-type", "")
    snippet = resp.text[:500].replace("\n", " ")
    print(f"[RES] {method} {url} params={params} status={resp.status_code} ctype={ctype}")
    if "application/json" in ctype:
        try:
            payload = resp.json()
            print(f"      {_summarize_json(payload)}")
        except json.JSONDecodeError:
            print("      json:decode_error")
    else:
        print(f"      body_snippet={snippet}")


def _load_condition_id(db_path: str) -> str | None:
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error:
        return None
    try:
        cur = conn.execute(
            "SELECT condition_id FROM observed_trades ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-id", help="Condition id to test (optional)")
    parser.add_argument(
        "--db-path",
        default=os.getenv("COPY_TRADE_DB_PATH", "data/copy_trade_phase1.sqlite"),
        help="SQLite DB path (used if condition-id is not provided)",
    )
    parser.add_argument("--base-url", default=os.getenv("COPY_TRADE_DATA_API_BASE", DEFAULT_BASE_URL))
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    cond = args.condition_id or _load_condition_id(args.db_path)
    if not cond:
        raise SystemExit("No condition_id provided and none found in DB")
    base = args.base_url.rstrip("/")
    session = requests.Session()

    tests: list[tuple[str, str, dict[str, Any]]] = [
        ("GET", f"{base}/markets", {"conditionId": cond}),
        ("GET", f"{base}/markets", {"condition_id": cond}),
        ("GET", f"{base}/markets", {"query": cond}),
        ("GET", f"{base}/markets", {"search": cond}),
        ("GET", f"{base}/markets", {"slug": cond}),
        ("GET", f"{base}/markets", {"limit": 1}),
        ("GET", f"{base}/events", {"conditionId": cond}),
        ("GET", f"{base}/events", {"condition_id": cond}),
        ("GET", f"{base}/events", {"query": cond}),
        ("GET", f"{base}/events", {"search": cond}),
        ("GET", f"{base}/events", {"limit": 1}),
    ]

    for method, url, params in tests:
        _try_request(session, method, url, params)
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
