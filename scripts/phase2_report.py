#!/usr/bin/env python3
"""
Summarize Phase2 bot performance over a recent window (default: 24 hours).
Reads market_maker/logs CSVs and optionally phase2.nohup.out for dashboard stats.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional


_DASH_RE = re.compile(
    r"\[PHASE2\] (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ dashboard "
    r"cash_cents=(?P<cash>-?\d+) equity_cents=(?P<equity>-?\d+) "
    r"inv_yes=(?P<inv_yes>-?\d+) inv_no=(?P<inv_no>-?\d+) .*? "
    r"day_pnl_cents=(?P<pnl>-?\d+) .*? soft_throttle=(?P<throttle>\d+)"
)
_SOFT_RE = re.compile(r"\[PHASE2\] (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ soft_throttle")


@dataclass
class WindowStats:
    start: datetime
    end: datetime


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _load_csv(path: Path) -> Iterable[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _window_bounds(rows: Iterable[dict[str, str]], hours: float, now_override: Optional[str]) -> WindowStats:
    if now_override:
        end = _parse_iso(now_override)
        if end is None:
            raise SystemExit(f"Invalid --now value: {now_override}")
    else:
        end = None
        for row in rows:
            ts = _parse_iso(row.get("ts_utc", ""))
            if ts and (end is None or ts > end):
                end = ts
        if end is None:
            end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    return WindowStats(start=start, end=end)


def _filter_rows(rows: Iterable[dict[str, str]], start: datetime, end: datetime) -> list[dict[str, str]]:
    out = []
    for row in rows:
        ts = _parse_iso(row.get("ts_utc", ""))
        if ts is None:
            continue
        if start <= ts <= end:
            out.append(row)
    return out


def _summarize_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    if not rows:
        return {}
    fills = 0
    actions = 0
    net_cash = 0
    last = rows[-1]
    for row in rows:
        fills += int(row.get("fills", "0") or 0)
        actions += int(row.get("actions", "0") or 0)
        net_cash += int(row.get("net_cash_change", "0") or 0)
    return {
        "fills": fills,
        "actions": actions,
        "fill_rate": (fills / actions) if actions else 0.0,
        "net_cash_change": net_cash,
        "inv_yes_last": int(last.get("inv_yes", "0") or 0),
        "inv_no_last": int(last.get("inv_no", "0") or 0),
    }


def _summarize_fills(rows: list[dict[str, str]]) -> dict[str, object]:
    if not rows:
        return {}
    total_qty = 0
    total_price_qty = 0
    side_counts = Counter()
    ticker_qty = Counter()
    for row in rows:
        qty = int(row.get("qty", "0") or 0)
        price = int(row.get("price_cents", "0") or 0)
        side = row.get("side", "")
        ticker = row.get("market_ticker", "")
        total_qty += qty
        total_price_qty += price * qty
        if side:
            side_counts[side] += qty
        if ticker:
            ticker_qty[ticker] += qty
    avg_price = (total_price_qty / total_qty) if total_qty else 0.0
    top_tickers = ticker_qty.most_common(5)
    return {
        "fills": len(rows),
        "total_qty": total_qty,
        "avg_price_cents": round(avg_price, 2),
        "side_qty": dict(side_counts),
        "top_tickers_by_qty": top_tickers,
    }


def _summarize_actions(rows: list[dict[str, str]]) -> dict[str, object]:
    if not rows:
        return {}
    action_counts = Counter()
    reason_counts = Counter()
    for row in rows:
        action_counts[row.get("action", "")] += 1
        reason = row.get("reason", "")
        if reason:
            reason_counts[reason] += 1
    return {
        "actions": len(rows),
        "action_counts": dict(action_counts),
        "top_reasons": reason_counts.most_common(5),
    }


def _summarize_decisions(rows: list[dict[str, str]]) -> dict[str, object]:
    if not rows:
        return {}
    eligible = 0
    ineligible = 0
    reasons = Counter()
    states = Counter()
    for row in rows:
        if row.get("eligible", "0") == "1":
            eligible += 1
        else:
            ineligible += 1
            for reason in (row.get("exclude_reasons") or "").split(";"):
                reason = reason.strip()
                if reason:
                    reasons[reason] += 1
        state = row.get("state", "")
        if state:
            states[state] += 1
    return {
        "eligible": eligible,
        "ineligible": ineligible,
        "top_exclusions": reasons.most_common(5),
        "top_states": states.most_common(5),
    }


def _summarize_nohup(path: Path, hours: float) -> dict[str, object]:
    if not path.exists():
        return {}
    dashboards = []
    soft_throttle_count = 0
    last_ts = None
    with path.open("r", errors="ignore") as f:
        for line in f:
            m = _DASH_RE.search(line)
            if m:
                ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
                dashboards.append(
                    (
                        ts,
                        int(m.group("cash")),
                        int(m.group("equity")),
                        int(m.group("pnl")),
                        int(m.group("inv_yes")),
                        int(m.group("inv_no")),
                        int(m.group("throttle")),
                    )
                )
                last_ts = ts
                continue
            if _SOFT_RE.search(line):
                soft_throttle_count += 1
    if not dashboards or last_ts is None:
        return {}
    window_start = last_ts - timedelta(hours=hours)
    in_window = [d for d in dashboards if d[0] >= window_start]
    if not in_window:
        return {}
    start = in_window[0]
    end = in_window[-1]
    pnls = [d[3] for d in in_window]
    equities = [d[2] for d in in_window]
    return {
        "window_start_local": window_start,
        "window_end_local": last_ts,
        "dashboard_start": {
            "ts": start[0],
            "cash": start[1],
            "equity": start[2],
            "day_pnl": start[3],
            "inv_yes": start[4],
            "inv_no": start[5],
            "soft_throttle": start[6],
        },
        "dashboard_end": {
            "ts": end[0],
            "cash": end[1],
            "equity": end[2],
            "day_pnl": end[3],
            "inv_yes": end[4],
            "inv_no": end[5],
            "soft_throttle": end[6],
        },
        "equity_change": end[2] - start[2],
        "day_pnl_min": min(pnls),
        "day_pnl_max": max(pnls),
        "soft_throttle_events": soft_throttle_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase2 performance over a recent window.")
    parser.add_argument("--hours", type=float, default=24.0, help="Window size in hours (default: 24).")
    parser.add_argument("--log-dir", type=Path, default=Path("market_maker/logs"))
    parser.add_argument("--nohup", type=Path, default=Path("phase2.nohup.out"))
    parser.add_argument("--now", type=str, default=None, help="Override window end time (ISO8601, UTC).")
    args = parser.parse_args()

    summary_rows = _load_csv(args.log_dir / "phase2_summary.csv")
    fills_rows = _load_csv(args.log_dir / "phase2_fills.csv")
    actions_rows = _load_csv(args.log_dir / "phase2_actions.csv")
    decisions_rows = _load_csv(args.log_dir / "phase2_decisions.csv")

    window = _window_bounds(summary_rows or decisions_rows or fills_rows, args.hours, args.now)

    summary = _summarize_summary(_filter_rows(summary_rows, window.start, window.end))
    fills = _summarize_fills(_filter_rows(fills_rows, window.start, window.end))
    actions = _summarize_actions(_filter_rows(actions_rows, window.start, window.end))
    decisions = _summarize_decisions(_filter_rows(decisions_rows, window.start, window.end))
    nohup = _summarize_nohup(args.nohup, args.hours)

    print("Phase2 report")
    print(f"UTC window: {window.start.isoformat()} -> {window.end.isoformat()}")

    if summary:
        print("\nSummary (phase2_summary.csv)")
        print(f"- fills: {summary['fills']}")
        print(f"- actions: {summary['actions']}")
        print(f"- fill_rate: {summary['fill_rate']:.4f}")
        print(f"- net_cash_change: {summary['net_cash_change']} cents")
        print(f"- inv_yes_last: {summary['inv_yes_last']}  inv_no_last: {summary['inv_no_last']}")
    else:
        print("\nSummary (phase2_summary.csv): no rows in window")

    if fills:
        print("\nFills (phase2_fills.csv)")
        print(f"- fills: {fills['fills']}  total_qty: {fills['total_qty']}  avg_price_cents: {fills['avg_price_cents']}")
        if fills["side_qty"]:
            print(f"- side_qty: {fills['side_qty']}")
        if fills["top_tickers_by_qty"]:
            print("- top_tickers_by_qty:")
            for ticker, qty in fills["top_tickers_by_qty"]:
                print(f"  - {ticker}: {qty}")
    else:
        print("\nFills (phase2_fills.csv): no rows in window")

    if actions:
        print("\nActions (phase2_actions.csv)")
        print(f"- actions: {actions['actions']}")
        print(f"- action_counts: {actions['action_counts']}")
        if actions["top_reasons"]:
            print("- top_reasons:")
            for reason, count in actions["top_reasons"]:
                print(f"  - {reason}: {count}")
    else:
        print("\nActions (phase2_actions.csv): no rows in window")

    if decisions:
        print("\nDecisions (phase2_decisions.csv)")
        print(f"- eligible: {decisions['eligible']}  ineligible: {decisions['ineligible']}")
        if decisions["top_states"]:
            print("- top_states:")
            for state, count in decisions["top_states"]:
                print(f"  - {state}: {count}")
        if decisions["top_exclusions"]:
            print("- top_exclusions:")
            for reason, count in decisions["top_exclusions"]:
                print(f"  - {reason}: {count}")
    else:
        print("\nDecisions (phase2_decisions.csv): no rows in window")

    if nohup:
        print("\nDashboard (phase2.nohup.out, local time)")
        print(f"- local_window: {nohup['window_start_local']} -> {nohup['window_end_local']}")
        print(f"- equity_change: {nohup['equity_change']} cents")
        print(f"- day_pnl_min: {nohup['day_pnl_min']}  day_pnl_max: {nohup['day_pnl_max']}")
        print(f"- soft_throttle_events: {nohup['soft_throttle_events']}")
        start = nohup["dashboard_start"]
        end = nohup["dashboard_end"]
        print(f"- dashboard_start: {start['ts']} cash={start['cash']} equity={start['equity']} pnl={start['day_pnl']}")
        print(f"- dashboard_end:   {end['ts']} cash={end['cash']} equity={end['equity']} pnl={end['day_pnl']}")
    else:
        print("\nDashboard (phase2.nohup.out): no dashboard lines in window")


if __name__ == "__main__":
    main()
