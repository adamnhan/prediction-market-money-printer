#!/usr/bin/env python3
"""
L2 orderbook sanity checks for l2_orderbook.sqlite.

Usage:
  python scripts/l2_sanity_checks.py data/l2_orderbook.sqlite

Optional:
  --gap-minutes 20
  --backward-threshold-seconds 60
  --pretip
  --tip-hour 0
  --tip-minute 0
  --top-n 20
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sqlite3
import sys
from typing import Dict, Iterable, List, Optional, Tuple


MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity checks for L2 orderbook data.")
    parser.add_argument("sqlite_path", help="Path to l2_orderbook.sqlite")
    parser.add_argument("--gap-minutes", type=int, default=20, help="Gap threshold in minutes")
    parser.add_argument(
        "--backward-threshold-seconds",
        type=int,
        default=60,
        help="Backward jump threshold in seconds",
    )
    parser.add_argument(
        "--pretip",
        action="store_true",
        help="Compute pre-tip share using date encoded in market_ticker",
    )
    parser.add_argument("--tip-hour", type=int, default=0, help="Tip hour (UTC)")
    parser.add_argument("--tip-minute", type=int, default=0, help="Tip minute (UTC)")
    parser.add_argument("--top-n", type=int, default=20, help="How many tickers to list in alerts")
    return parser.parse_args()


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def query_all(conn: sqlite3.Connection, sql: str, params: Tuple = ()) -> List[sqlite3.Row]:
    cur = conn.execute(sql, params)
    return cur.fetchall()


def query_one(conn: sqlite3.Connection, sql: str, params: Tuple = ()) -> sqlite3.Row:
    cur = conn.execute(sql, params)
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("Query returned no rows")
    return row


def parse_tip_ms(market_ticker: str, tip_hour: int, tip_minute: int) -> Optional[int]:
    match = re.search(r"-(\d{2}[A-Z]{3}\d{2})", market_ticker)
    if not match:
        return None
    token = match.group(1)
    day = int(token[0:2])
    mon = token[2:5]
    year = int(token[5:7])
    month = MONTHS.get(mon)
    if month is None:
        return None
    year += 2000
    try:
        tip_dt = dt.datetime(
            year, month, day, tip_hour, tip_minute, tzinfo=dt.timezone.utc
        )
    except ValueError:
        return None
    return int(tip_dt.timestamp() * 1000)


def format_ts(ms: Optional[int]) -> str:
    if ms is None:
        return "n/a"
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).isoformat()


def print_section(title: str) -> None:
    print("")
    print(title)
    print("-" * len(title))


def main() -> int:
    args = parse_args()
    gap_threshold_ms = args.gap_minutes * 60 * 1000
    backward_threshold_ms = args.backward_threshold_seconds * 1000

    conn = connect(args.sqlite_path)

    # Coverage summary
    total_markets = query_one(
        conn, "SELECT COUNT(DISTINCT market_ticker) AS cnt FROM l2_messages"
    )["cnt"]
    total_msgs = query_one(conn, "SELECT COUNT(*) AS cnt FROM l2_messages")["cnt"]

    print_section("Coverage")
    print(f"Unique markets (l2_messages): {total_markets}")
    print(f"Total messages: {total_msgs}")

    # Hours per market
    hours_rows = query_all(
        conn,
        """
        SELECT market_ticker,
               COUNT(DISTINCT CAST(ts_utc_ms / 3600000 AS INTEGER)) AS hours_distinct,
               MIN(ts_utc_ms) AS min_ts,
               MAX(ts_utc_ms) AS max_ts,
               COUNT(*) AS msg_count
        FROM l2_messages
        GROUP BY market_ticker
        """,
    )

    hours_by_market: Dict[str, sqlite3.Row] = {row["market_ticker"]: row for row in hours_rows}

    # Message mix
    print_section("Message Mix")
    mix_rows = query_all(
        conn, "SELECT channel, COUNT(*) AS cnt FROM l2_messages GROUP BY channel"
    )
    for row in mix_rows:
        print(f"{row['channel']}: {row['cnt']}")

    mix_by_market_rows = query_all(
        conn,
        """
        SELECT market_ticker,
               SUM(CASE WHEN channel = 'orderbook_snapshot' THEN 1 ELSE 0 END) AS snapshots,
               SUM(CASE WHEN channel = 'orderbook_delta' THEN 1 ELSE 0 END) AS deltas
        FROM l2_messages
        GROUP BY market_ticker
        """,
    )
    missing_snapshots = []
    missing_deltas = []
    for row in mix_by_market_rows:
        if row["snapshots"] == 0:
            missing_snapshots.append(row["market_ticker"])
        if row["deltas"] == 0:
            missing_deltas.append(row["market_ticker"])

    print(f"Markets missing snapshots: {len(missing_snapshots)}")
    if missing_snapshots:
        print("Examples:", ", ".join(missing_snapshots[: args.top_n]))
    print(f"Markets missing deltas: {len(missing_deltas)}")
    if missing_deltas:
        print("Examples:", ", ".join(missing_deltas[: args.top_n]))

    # Timestamp checks
    print_section("Timestamps")
    ts_rows = query_all(
        conn,
        """
        WITH ordered AS (
            SELECT market_ticker,
                   ts_utc_ms,
                   LAG(ts_utc_ms) OVER (
                       PARTITION BY market_ticker
                       ORDER BY ts_utc_ms
                   ) AS prev_ts
            FROM l2_messages
        )
        SELECT market_ticker,
               SUM(CASE
                       WHEN prev_ts IS NOT NULL
                        AND ts_utc_ms < prev_ts - ?
                       THEN 1 ELSE 0 END) AS backward_jumps,
               MAX(CASE
                       WHEN prev_ts IS NOT NULL
                       THEN ts_utc_ms - prev_ts
                       ELSE 0 END) AS max_gap_ms,
               SUM(CASE
                       WHEN prev_ts IS NOT NULL
                        AND ts_utc_ms - prev_ts > ?
                       THEN 1 ELSE 0 END) AS gap_count
        FROM ordered
        GROUP BY market_ticker
        """,
        (backward_threshold_ms, gap_threshold_ms),
    )

    backward_markets = [row for row in ts_rows if row["backward_jumps"] > 0]
    gap_markets = [row for row in ts_rows if row["gap_count"] > 0]

    print(f"Markets with backward jumps > {args.backward_threshold_seconds}s: {len(backward_markets)}")
    if backward_markets:
        examples = ", ".join([row["market_ticker"] for row in backward_markets[: args.top_n]])
        print("Examples:", examples)

    print(f"Markets with gaps > {args.gap_minutes} minutes: {len(gap_markets)}")
    if gap_markets:
        examples = ", ".join([row["market_ticker"] for row in gap_markets[: args.top_n]])
        print("Examples:", examples)

    # Coverage details
    print_section("Coverage Details (per market)")
    print("market_ticker | hours_distinct | msg_count | min_ts | max_ts")
    for ticker, row in sorted(hours_by_market.items()):
        print(
            f"{ticker} | {row['hours_distinct']} | {row['msg_count']} | "
            f"{format_ts(row['min_ts'])} | {format_ts(row['max_ts'])}"
        )

    # Pre-tip coverage (optional)
    if args.pretip:
        print_section("Pre-tip Coverage (estimated from ticker date)")
        tip_cache: Dict[str, Optional[int]] = {}
        pre_counts: Dict[str, int] = {}
        post_counts: Dict[str, int] = {}
        total_counts: Dict[str, int] = {}

        cur = conn.execute("SELECT market_ticker, ts_utc_ms FROM l2_messages")
        while True:
            rows = cur.fetchmany(50000)
            if not rows:
                break
            for row in rows:
                ticker = row["market_ticker"]
                ts = row["ts_utc_ms"]
                if ticker not in tip_cache:
                    tip_cache[ticker] = parse_tip_ms(
                        ticker, args.tip_hour, args.tip_minute
                    )
                tip_ms = tip_cache[ticker]
                if tip_ms is None:
                    continue
                total_counts[ticker] = total_counts.get(ticker, 0) + 1
                if ts < tip_ms:
                    pre_counts[ticker] = pre_counts.get(ticker, 0) + 1
                else:
                    post_counts[ticker] = post_counts.get(ticker, 0) + 1

        if not total_counts:
            print("No tickers had a parsable date; pre-tip check skipped.")
        else:
            mostly_pretip = []
            for ticker, total in total_counts.items():
                pre = pre_counts.get(ticker, 0)
                if total > 0 and pre / total >= 0.8:
                    mostly_pretip.append(ticker)
            print(f"Tickers with >=80% pre-tip messages: {len(mostly_pretip)}")
            if mostly_pretip:
                print("Examples:", ", ".join(mostly_pretip[: args.top_n]))

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
