#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from collections import defaultdict
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate leg-centric guardrails on settled bundles.")
    parser.add_argument("--ledger-db", default="convev options hedge strat/phase_f_ledger.sqlite")
    parser.add_argument("--results-csv", default="data/nba_results.csv")
    parser.add_argument(
        "--table",
        default="bundle_settlements_filled",
        help="Settlement table to use (default: bundle_settlements_filled).",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include non-filled bundles when leg fill data is present.",
    )
    parser.add_argument("--max-trades-per-event", type=int, default=300)
    parser.add_argument("--max-event-drawdown", type=float, default=-25.0)
    parser.add_argument("--max-daily-drawdown", type=float, default=-50.0)
    parser.add_argument("--ev-buckets", type=int, default=10)
    parser.add_argument("--debug-bundle-id", default="", help="Print per-leg settlement for one bundle.")
    parser.add_argument("--debug-first-n", type=int, default=0, help="Print per-leg settlement for first N accepted.")
    return parser.parse_args()


def _parse_spread_threshold(market_ticker: str) -> tuple[str | None, int | None]:
    suffix = market_ticker.split("-")[-1]
    for i, ch in enumerate(suffix):
        if ch.isdigit() or ch == "-":
            team = suffix[:i]
            try:
                k = int(suffix[i:])
            except ValueError:
                return None, None
            return team, k
    return None, None


def _parse_total_threshold(market_ticker: str) -> int | None:
    suffix = market_ticker.split("-")[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _load_results(path: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            event_key = (row.get("event_key") or "").strip()
            if not event_key:
                continue
            try:
                final_margin = int(row.get("final_margin") or 0)
            except ValueError:
                continue
            try:
                home_score = int(row.get("home_score") or 0)
                away_score = int(row.get("away_score") or 0)
            except ValueError:
                home_score = 0
                away_score = 0
            out[event_key] = {
                "home": (row.get("home_team_abbr") or "").strip(),
                "away": (row.get("away_team_abbr") or "").strip(),
                "final_margin": final_margin,
                "total_points": home_score + away_score,
            }
    return out


def _settlement_yes(ticker: str, results: dict[str, dict[str, Any]]) -> float | None:
    parts = ticker.split("-")
    if len(parts) < 3:
        return None
    series = parts[0]
    event_key = parts[1]
    meta = results.get(event_key)
    if meta is None:
        return None
    if series == "KXNBASPREAD":
        team, k = _parse_spread_threshold(ticker)
        if team is None or k is None:
            return None
        threshold = k + 0.5
        margin = meta["final_margin"]
        if team == meta["home"]:
            return 1.0 if margin > threshold else 0.0
        if team == meta["away"]:
            return 1.0 if margin < -threshold else 0.0
        return None
    if series == "KXNBATOTAL":
        k = _parse_total_threshold(ticker)
        if k is None:
            return None
        threshold = k + 0.5
        total_points = meta["total_points"]
        return 1.0 if total_points > threshold else 0.0
    return None


def _portfolio_max_loss(legs: list[dict[str, Any]]) -> float:
    if not legs:
        return 0.0
    ks = sorted(set(int(leg["k"]) for leg in legs))
    bounds = [-10**9] + ks + [10**9]
    const_cost = 0.0
    for leg in legs:
        w = leg["w"]
        p = leg["p"]
        const_cost += w * (-p)
    vals = []
    for j in range(len(bounds) - 1):
        lo, hi = bounds[j], bounds[j + 1]
        rep = lo if lo > -10**8 else (hi - 1)
        ind_term = 0.0
        for leg in legs:
            ind_term += leg["w"] * (1.0 if rep >= leg["k"] else 0.0)
        vals.append(ind_term + const_cost)
    return float(min(vals)) if vals else 0.0


def _load_bundles(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    if table not in {"bundle_settlements", "bundle_settlements_filled"}:
        raise ValueError(f"unsupported table: {table}")
    rows = conn.execute(
        f"""
        SELECT
            b.bundle_id, b.ts_signal, b.event_key, b.decision, b.ev_net_est, b.ev_raw,
            l.leg_idx, l.ticker, l.k, l.side, l.qty, l.limit_price, l.px_used,
            l.fill_qty, l.fill_price, l.fill_ts
        FROM bundles b
        JOIN {table} s ON s.bundle_id = b.bundle_id
        JOIN bundle_legs l ON l.bundle_id = b.bundle_id
        ORDER BY b.ts_signal ASC, b.bundle_id ASC, l.leg_idx ASC
        """
    ).fetchall()
    bundles: dict[str, dict[str, Any]] = {}
    for r in rows:
        bundle_id = r[0]
        bundle = bundles.setdefault(
            bundle_id,
            {
                "bundle_id": bundle_id,
                "ts_signal": int(r[1] or 0),
                "event_key": r[2] or "",
                "decision": r[3] or "",
                "ev_net_est": r[4],
                "ev_raw": r[5],
                "legs": [],
            },
        )
        bundle["legs"].append(
            {
                "leg_idx": r[6],
                "ticker": r[7],
                "k": r[8],
                "side": r[9],
                "qty": r[10],
                "limit_price": r[11],
                "px_used": r[12],
                "fill_qty": r[13],
                "fill_price": r[14],
                "fill_ts": r[15],
            }
        )
    return list(bundles.values())


def _leg_fill(leg: dict[str, Any], decision: str, include_rejected: bool) -> tuple[int, float] | None:
    fill_qty = leg.get("fill_qty")
    fill_price = leg.get("fill_price")
    if fill_qty is not None and fill_price is not None and int(fill_qty) > 0:
        qty = int(fill_qty)
        price = float(fill_price)
        if price > 1.0:
            price /= 100.0
        return qty, price
    if decision in {"SHADOW_FILLED", "SHADOW_PARTIAL"}:
        qty = int(leg.get("qty") or 1)
        price = leg.get("limit_price")
        if price is None:
            price = leg.get("px_used")
        if price is None:
            return None
        return qty, float(price)
    if include_rejected:
        # no explicit fill info; cannot assume partial fills
        return None
    return None


def _debug_bundle(
    bundle: dict[str, Any],
    results: dict[str, dict[str, Any]],
    include_rejected: bool,
) -> None:
    event_key = bundle["event_key"]
    meta = results.get(event_key)
    margin = meta["final_margin"] if meta else None
    print(f"[debug] bundle_id={bundle['bundle_id']} event_key={event_key} margin={margin}")
    total = 0.0
    for leg in bundle["legs"]:
        fill = _leg_fill(leg, bundle["decision"], include_rejected)
        qty = fill[0] if fill else None
        price = fill[1] if fill else None
        settle = _settlement_yes(leg["ticker"], results)
        side = leg["side"]
        pnl = None
        if fill and settle is not None:
            if side == "BUY_YES":
                pnl = qty * (settle - price)
            elif side == "SELL_YES":
                pnl = -qty * (settle - price)
            total += pnl
        print(
            f"[debug] leg_idx={leg['leg_idx']} ticker={leg['ticker']} side={side} "
            f"qty={leg.get('qty')} fill_qty={leg.get('fill_qty')} "
            f"price={price} settle={settle} pnl={pnl}"
        )
    print(f"[debug] bundle_pnl={total:.4f} ev_net_est={bundle.get('ev_net_est')} ev_raw={bundle.get('ev_raw')}")


def main() -> int:
    args = parse_args()
    results = _load_results(args.results_csv)
    with sqlite3.connect(args.ledger_db) as conn:
        bundles = _load_bundles(conn, args.table)

    total_bundles = 0
    wins = 0
    pnl_sum = 0.0
    worst = None
    best = None

    event_trades: dict[str, int] = {}
    event_pnl: dict[str, float] = {}
    global_pnl = 0.0

    accepted = 0
    skipped = 0

    ev_pairs: list[tuple[float, float]] = []
    max_loss_vals: list[float] = []
    missing_settlement = 0
    missing_fills = 0

    debug_target = args.debug_bundle_id.strip() or None
    debug_left = max(0, args.debug_first_n)
    for bundle in bundles:
        event_key = bundle["event_key"]
        decision = bundle["decision"]
        if not args.include_rejected and decision not in {"SHADOW_FILLED", "SHADOW_PARTIAL"}:
            skipped += 1
            continue
        if args.max_trades_per_event and event_trades.get(event_key, 0) >= args.max_trades_per_event:
            skipped += 1
            continue
        if args.max_event_drawdown and event_pnl.get(event_key, 0.0) <= args.max_event_drawdown:
            skipped += 1
            continue
        if args.max_daily_drawdown and global_pnl <= args.max_daily_drawdown:
            skipped += 1
            continue

        legs_filled = []
        bundle_pnl = 0.0
        for leg in bundle["legs"]:
            fill = _leg_fill(leg, decision, args.include_rejected)
            if fill is None:
                missing_fills += 1
                continue
            qty, price = fill
            settle = _settlement_yes(leg["ticker"], results)
            if settle is None:
                missing_settlement += 1
                continue
            side = leg["side"]
            if side == "BUY_YES":
                pnl = qty * (settle - price)
                w = qty
            elif side == "SELL_YES":
                pnl = -qty * (settle - price)
                w = -qty
            else:
                continue
            bundle_pnl += pnl
            legs_filled.append({"k": leg["k"], "w": w, "p": price})

        if not legs_filled:
            skipped += 1
            continue

        accepted += 1
        total_bundles += 1
        pnl_sum += bundle_pnl
        global_pnl += bundle_pnl
        event_trades[event_key] = event_trades.get(event_key, 0) + 1
        event_pnl[event_key] = event_pnl.get(event_key, 0.0) + bundle_pnl
        if bundle_pnl > 0:
            wins += 1
        worst = bundle_pnl if worst is None else min(worst, bundle_pnl)
        best = bundle_pnl if best is None else max(best, bundle_pnl)

        ev = bundle.get("ev_net_est")
        if ev is None:
            ev = bundle.get("ev_raw")
        if ev is not None:
            ev_pairs.append((float(ev), bundle_pnl))
        max_loss_vals.append(_portfolio_max_loss(legs_filled))
        if debug_target and bundle["bundle_id"] == debug_target:
            _debug_bundle(bundle, results, args.include_rejected)
            debug_target = None
        if debug_left > 0:
            _debug_bundle(bundle, results, args.include_rejected)
            debug_left -= 1

    avg = pnl_sum / total_bundles if total_bundles else 0.0
    win_rate = wins / total_bundles if total_bundles else 0.0

    print("execution_summary")
    print(
        f"bundles={total_bundles} total_pnl={pnl_sum:.4f} avg_pnl={avg:.4f} "
        f"win_rate={win_rate:.3f}"
    )
    print(f"best={best if best is not None else 0:.4f} worst={worst if worst is not None else 0:.4f}")
    print(f"accepted={accepted} skipped={skipped}")
    print(f"missing_fills={missing_fills} missing_settlement={missing_settlement}")
    if max_loss_vals:
        min_loss = min(max_loss_vals)
        max_loss = max(max_loss_vals)
        avg_loss = sum(max_loss_vals) / len(max_loss_vals)
        print(
            f"unhedged_max_loss avg={avg_loss:.4f} min={min_loss:.4f} max={max_loss:.4f}"
        )

    print("")
    print("by_event")
    for event_key, pnl in sorted(event_pnl.items(), key=lambda x: x[1], reverse=True):
        trades = event_trades[event_key]
        avg_evt = pnl / trades if trades else 0.0
        print(f"{event_key} bundles={trades} pnl={pnl:.4f} avg_pnl={avg_evt:.4f}")

    if ev_pairs:
        print("")
        print("ev_calibration")
        ev_pairs.sort(key=lambda x: x[0])
        bucket_count = max(1, args.ev_buckets)
        n = len(ev_pairs)
        bucket_size = max(1, math.ceil(n / bucket_count))
        for i in range(bucket_count):
            start = i * bucket_size
            end = min(n, (i + 1) * bucket_size)
            if start >= n:
                break
            bucket = ev_pairs[start:end]
            ev_avg = sum(x[0] for x in bucket) / len(bucket)
            pnl_avg = sum(x[1] for x in bucket) / len(bucket)
            win_rate_b = sum(1 for x in bucket if x[1] > 0) / len(bucket)
            print(
                f"bucket={i+1} n={len(bucket)} ev_avg={ev_avg:.4f} "
                f"pnl_avg={pnl_avg:.4f} win_rate={win_rate_b:.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
