#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import time
from collections import defaultdict
from typing import Any

from nba_engine.config import load_config
from nba_engine.phase5 import RestClient


def _init_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bundle_settlements (
            bundle_id TEXT PRIMARY KEY,
            event_key TEXT,
            final_margin INTEGER,
            pnl_realized REAL,
            computed_ts INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bundle_settlements_filled (
            bundle_id TEXT PRIMARY KEY,
            event_key TEXT,
            final_margin INTEGER,
            pnl_realized REAL,
            computed_ts INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events_settlement (
            event_key TEXT PRIMARY KEY,
            final_margin INTEGER,
            settle_ts INTEGER,
            source TEXT
        )
        """
    )
    conn.commit()


def _load_unsettled_bundles(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT b.bundle_id, b.event_key, b.ts_signal, b.decision
        FROM bundles b
        LEFT JOIN bundle_settlements s ON s.bundle_id = b.bundle_id
        WHERE s.bundle_id IS NULL
        """
    ).fetchall()
    return [{"bundle_id": r[0], "event_key": r[1], "ts_signal": r[2], "decision": r[3]} for r in rows]


def _load_bundle_legs(conn: sqlite3.Connection, bundle_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT ticker, k, side, qty, limit_price
        FROM bundle_legs
        WHERE bundle_id = ?
        """,
        (bundle_id,),
    ).fetchall()
    out = []
    for row in rows:
        out.append(
            {
                "ticker": row[0],
                "k": int(row[1]),
                "side": row[2],
                "qty": int(row[3]),
                "limit_price": float(row[4]),
            }
        )
    return out


def _market_settlement_value(market: dict[str, Any]) -> float | None:
    if market is None:
        return None
    value = market.get("settlement_value")
    if value is not None:
        try:
            return float(value) / 100.0
        except Exception:
            pass
    result = (market.get("result") or "").lower()
    if result == "yes":
        return 1.0
    if result == "no":
        return 0.0
    return None


def _bundle_pnl(legs: list[dict[str, Any]], settlement_yes: dict[str, float]) -> float:
    pnl = 0.0
    for leg in legs:
        ticker = leg["ticker"]
        settle = settlement_yes.get(ticker)
        if settle is None:
            return float("nan")
        qty = leg["qty"]
        p = leg["limit_price"]
        side = leg["side"]
        w = qty if side == "BUY_YES" else -qty
        pnl += w * (settle - p)
    return pnl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update bundle settlements via Kalshi market resolution.")
    parser.add_argument("--ledger-db", default="convev options hedge strat/phase_f_ledger.sqlite")
    parser.add_argument("--sleep-s", type=float, default=0.2)
    parser.add_argument("--max-bundles", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="Log progress and skip reasons.")
    parser.add_argument(
        "--debug-one",
        action="store_true",
        help="Print the raw market payload for the first status/missing settlement failure.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Emit a progress line every N bundles (0 disables).",
    )
    parser.add_argument(
        "--backfill-filled",
        action="store_true",
        help="Populate bundle_settlements_filled from existing settlements without API calls.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    client = RestClient(
        config.kalshi_rest_url or "https://api.elections.kalshi.com/trade-api/v2",
        config.kalshi_key_id,
        config.kalshi_private_key_path,
    )
    with sqlite3.connect(args.ledger_db) as conn:
        _init_tables(conn)
        if args.backfill_filled:
            rows = conn.execute(
                """
                SELECT s.bundle_id, s.event_key, s.final_margin, s.pnl_realized, s.computed_ts, b.decision
                FROM bundle_settlements s
                JOIN bundles b ON b.bundle_id = s.bundle_id
                """
            ).fetchall()
            inserted = 0
            for r in rows:
                if r[5] not in {"SHADOW_FILLED", "SHADOW_PARTIAL"}:
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO bundle_settlements_filled (
                        bundle_id, event_key, final_margin, pnl_realized, computed_ts
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (r[0], r[1], r[2], r[3], r[4]),
                )
                inserted += 1
            conn.commit()
            print(f"[settlement_updater] backfill_filled inserted={inserted}", flush=True)
            return 0
        bundles = _load_unsettled_bundles(conn)
        if args.max_bundles:
            bundles = bundles[: args.max_bundles]
        if args.verbose:
            print(f"[settlement_updater] unsettled_bundles={len(bundles)}", flush=True)
        debug_emitted = False
        market_cache: dict[str, dict[str, Any]] = {}
        total = len(bundles)
        processed = 0
        settled = 0
        t0 = time.time()
        for bundle in bundles:
            processed += 1
            bundle_id = bundle["bundle_id"]
            legs = _load_bundle_legs(conn, bundle_id)
            if not legs:
                if args.verbose:
                    print(f"[settlement_updater] bundle_id={bundle_id} skip=no_legs", flush=True)
                continue
            settlement_yes: dict[str, float] = {}
            all_settled = True
            tickers = list(dict.fromkeys(leg["ticker"] for leg in legs))
            for ticker in tickers:
                if ticker in market_cache:
                    market = market_cache[ticker]
                else:
                    try:
                        market = client.get_market(ticker)
                    except Exception as exc:
                        all_settled = False
                        if args.verbose:
                            print(
                                f"[settlement_updater] bundle_id={bundle_id} ticker={ticker} "
                                f"skip=api_error error={exc}",
                                flush=True,
                            )
                        break
                    # RestClient returns {"market": {...}} for get_market
                    if isinstance(market, dict) and "market" in market and isinstance(market.get("market"), dict):
                        market = market["market"]
                    market_cache[ticker] = market
                    if args.sleep_s:
                        time.sleep(args.sleep_s)
                status = (market.get("status") or "").lower()
                if status not in {"settled", "finalized", "determined"}:
                    all_settled = False
                    if args.verbose:
                        print(
                            f"[settlement_updater] bundle_id={bundle_id} ticker={ticker} "
                            f"skip=status status={status}",
                            flush=True,
                        )
                    if args.debug_one and not debug_emitted:
                        print(
                            f"[settlement_updater] debug ticker={ticker} market={market}",
                            flush=True,
                        )
                        debug_emitted = True
                    break
                settle_val = _market_settlement_value(market)
                if settle_val is None:
                    all_settled = False
                    if args.verbose:
                        print(
                            f"[settlement_updater] bundle_id={bundle_id} ticker={ticker} "
                            "skip=missing_settlement_value",
                            flush=True,
                        )
                    if args.debug_one and not debug_emitted:
                        print(
                            f"[settlement_updater] debug ticker={ticker} market={market}",
                            flush=True,
                        )
                        debug_emitted = True
                    break
                settlement_yes[ticker] = settle_val
            if not all_settled:
                continue
            pnl = _bundle_pnl(legs, settlement_yes)
            if pnl != pnl:
                if args.verbose:
                    print(
                        f"[settlement_updater] bundle_id={bundle_id} skip=pnl_nan",
                        flush=True,
                    )
                continue
            conn.execute(
                """
                INSERT OR REPLACE INTO bundle_settlements (
                    bundle_id, event_key, final_margin, pnl_realized, computed_ts
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (bundle_id, bundle["event_key"], None, pnl, int(time.time())),
            )
            conn.commit()
            if bundle.get("decision") in {"SHADOW_FILLED", "SHADOW_PARTIAL"}:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO bundle_settlements_filled (
                        bundle_id, event_key, final_margin, pnl_realized, computed_ts
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (bundle_id, bundle["event_key"], None, pnl, int(time.time())),
                )
                conn.commit()
            settled += 1
            if args.verbose:
                print(
                    f"[settlement_updater] bundle_id={bundle_id} event_key={bundle['event_key']} "
                    f"pnl_realized={pnl:.4f}",
                    flush=True,
                )
            if args.progress_every and processed % args.progress_every == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0.0
                print(
                    f"[settlement_updater] progress {processed}/{total} "
                    f"settled={settled} rate={rate:.2f}/s",
                    flush=True,
                )
        if args.progress_every:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0.0
            print(
                f"[settlement_updater] done {processed}/{total} "
                f"settled={settled} rate={rate:.2f}/s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
