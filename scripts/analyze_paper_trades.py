#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable


@dataclass(frozen=True)
class TradeRow:
    trade_id: int
    market_ticker: str
    side: str
    entry_ts: datetime
    entry_price: float
    qty: int
    exit_ts: datetime
    exit_price: float | None
    exit_reason: str | None
    pnl: float
    quality_score: float | None
    panic_ret_3: float | None
    vol_10: float | None
    vol_sum_5: float | None
    p_open: float | None
    p_base: float | None


def _parse_ts(value: object) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_pnl(side: str, entry_price: float, exit_price: float, qty: int) -> float:
    if side.upper() == "YES":
        return (exit_price - entry_price) * qty
    return (entry_price - exit_price) * qty


def _compute_drawdown(pnls: Iterable[float]) -> float:
    peak = 0.0
    cumulative = 0.0
    max_dd = 0.0
    for pnl in pnls:
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        drawdown = peak - cumulative
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def _fmt_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:,.4f}"


def _fmt_pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value * 100:.2f}%"


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def _safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return median(values)


def _safe_pstdev(values: list[float]) -> float | None:
    if not values:
        return None
    return pstdev(values)


def _row_count(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _load_log_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("event"):
                events.append(payload)
    return events


def _filter_log_events(events: list[dict[str, Any]], since_hours: float | None) -> list[dict[str, Any]]:
    if since_hours is None:
        return events
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    filtered: list[dict[str, Any]] = []
    for event in events:
        raw_ts = event.get("ts")
        if not raw_ts:
            continue
        ts = _parse_ts(raw_ts)
        if ts >= cutoff:
            filtered.append(event)
    return filtered


def _summarize_gate_blocks(events: list[dict[str, Any]]) -> tuple[int, Counter[str]]:
    panic_detected = 0
    failed_counts: Counter[str] = Counter()
    blocked_total = 0
    for event in events:
        if event.get("event") == "panic_detected":
            panic_detected += 1
            continue
        if event.get("event") != "skip_entry":
            continue
        blocked_total += 1
        failed_reason = str(event.get("failed_reason") or "")
        for reason in [r.strip() for r in failed_reason.split(",") if r.strip()]:
            failed_counts[reason] += 1
    return panic_detected, failed_counts, blocked_total


def _load_closed_trades(
    conn: sqlite3.Connection,
    exclude_pnl_above: float | None,
) -> list[TradeRow]:
    rows = conn.execute(
        """
        SELECT id,
               market_ticker,
               side,
               entry_ts,
               entry_price,
               qty,
               exit_ts,
               exit_price,
               exit_reason,
               pnl,
               quality_score,
               panic_ret_3,
               vol_10,
               vol_sum_5,
               p_open,
               p_base
        FROM paper_trades
        WHERE exit_ts IS NOT NULL
        ORDER BY exit_ts
        """
    ).fetchall()
    trades: list[TradeRow] = []
    for row in rows:
        trade_id = int(row[0])
        market_ticker = str(row[1])
        side = str(row[2])
        entry_ts = _parse_ts(row[3])
        entry_price = float(row[4]) if row[4] is not None else float("nan")
        qty = int(row[5]) if row[5] is not None else 1
        exit_ts = _parse_ts(row[6])
        exit_price = _to_float(row[7])
        exit_reason = row[8]
        pnl = _to_float(row[9])
        if pnl is None and exit_price is not None and not math.isnan(entry_price):
            pnl = _compute_pnl(side, entry_price, exit_price, qty)
        pnl = pnl if pnl is not None else float("nan")
        quality_score = _to_float(row[10])
        panic_ret_3 = _to_float(row[11])
        vol_10 = _to_float(row[12])
        vol_sum_5 = _to_float(row[13])
        p_open = _to_float(row[14])
        p_base = _to_float(row[15])
        if exclude_pnl_above is not None and not math.isnan(pnl) and pnl > exclude_pnl_above:
            continue
        trades.append(
            TradeRow(
                trade_id=trade_id,
                market_ticker=market_ticker,
                side=side,
                entry_ts=entry_ts,
                entry_price=entry_price,
                qty=qty,
                exit_ts=exit_ts,
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl=pnl,
                quality_score=quality_score,
                panic_ret_3=panic_ret_3,
                vol_10=vol_10,
                vol_sum_5=vol_sum_5,
                p_open=p_open,
                p_base=p_base,
            )
        )
    return trades


def _load_open_positions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, market_ticker, side, entry_ts, entry_price, qty
        FROM paper_trades
        WHERE exit_ts IS NULL
        """
    ).fetchall()
    positions: list[dict[str, Any]] = []
    for row in rows:
        positions.append(
            {
                "id": int(row[0]),
                "market_ticker": str(row[1]),
                "side": str(row[2]),
                "entry_ts": _parse_ts(row[3]).isoformat(),
                "entry_price": float(row[4]) if row[4] is not None else None,
                "qty": int(row[5]) if row[5] is not None else 1,
            }
        )
    return positions


def _summarize_trades(trades: list[TradeRow]) -> dict[str, Any]:
    pnls = [t.pnl for t in trades if not math.isnan(t.pnl)]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    flats = [p for p in pnls if p == 0]

    durations = [
        (t.exit_ts - t.entry_ts).total_seconds()
        for t in trades
        if t.exit_ts and t.entry_ts
    ]

    summary = {
        "closed_trades": len(trades),
        "gross_pnl": sum(pnls) if pnls else 0.0,
        "avg_pnl": _safe_mean(pnls),
        "median_pnl": _safe_median(pnls),
        "pnl_std": _safe_pstdev(pnls),
        "win_rate": (len(wins) / len(pnls)) if pnls else None,
        "loss_rate": (len(losses) / len(pnls)) if pnls else None,
        "flat_rate": (len(flats) / len(pnls)) if pnls else None,
        "avg_win": _safe_mean(wins),
        "avg_loss": _safe_mean(losses),
        "profit_factor": (sum(wins) / abs(sum(losses))) if losses else None,
        "avg_hold_seconds": _safe_mean(durations),
        "median_hold_seconds": _safe_median(durations),
        "max_drawdown": _compute_drawdown(pnls) if pnls else None,
        "largest_win": max(pnls) if pnls else None,
        "largest_loss": min(pnls) if pnls else None,
    }
    return summary


def _pnl_breakdown(trades: list[TradeRow]) -> dict[str, Any]:
    core_reasons = {"tp", "sl", "max_hold"}
    pnls = [t.pnl for t in trades if not math.isnan(t.pnl)]
    core_pnls = [
        t.pnl
        for t in trades
        if not math.isnan(t.pnl) and (t.exit_reason or "") in core_reasons
    ]
    stale_pnls = [
        t.pnl
        for t in trades
        if not math.isnan(t.pnl) and (t.exit_reason or "") == "stale_feed"
    ]
    return {
        "total_pnl": sum(pnls) if pnls else 0.0,
        "core_pnl": sum(core_pnls) if core_pnls else 0.0,
        "stale_feed_pnl": sum(stale_pnls) if stale_pnls else 0.0,
        "core_trades": sum(
            1 for t in trades if (t.exit_reason or "") in core_reasons
        ),
        "stale_feed_trades": sum(
            1 for t in trades if (t.exit_reason or "") == "stale_feed"
        ),
    }


def _bucket_quality(trades: list[TradeRow]) -> dict[str, int]:
    buckets = {
        "<0": 0,
        "0-0.25": 0,
        "0.25-0.5": 0,
        "0.5-0.75": 0,
        "0.75-1.0": 0,
        ">=1": 0,
        "missing": 0,
    }
    for t in trades:
        q = t.quality_score
        if q is None or math.isnan(q):
            buckets["missing"] += 1
        elif q < 0:
            buckets["<0"] += 1
        elif q < 0.25:
            buckets["0-0.25"] += 1
        elif q < 0.5:
            buckets["0.25-0.5"] += 1
        elif q < 0.75:
            buckets["0.5-0.75"] += 1
        elif q < 1.0:
            buckets["0.75-1.0"] += 1
        else:
            buckets[">=1"] += 1
    return buckets


def _by_market(trades: list[TradeRow]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for t in trades:
        bucket = out.setdefault(
            t.market_ticker,
            {
                "market_ticker": t.market_ticker,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "flat": 0,
                "gross_pnl": 0.0,
                "avg_pnl": None,
            },
        )
        if not math.isnan(t.pnl):
            bucket["gross_pnl"] += t.pnl
            if t.pnl > 0:
                bucket["wins"] += 1
            elif t.pnl < 0:
                bucket["losses"] += 1
            else:
                bucket["flat"] += 1
        bucket["trades"] += 1
    for item in out.values():
        if item["trades"]:
            item["avg_pnl"] = item["gross_pnl"] / item["trades"]
    return sorted(out.values(), key=lambda x: x["gross_pnl"], reverse=True)


def _by_side(trades: list[TradeRow]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for t in trades:
        side = t.side.upper()
        bucket = out.setdefault(
            side,
            {
                "side": side,
                "trades": 0,
                "gross_pnl": 0.0,
                "avg_pnl": None,
                "wins": 0,
                "losses": 0,
                "flat": 0,
            },
        )
        if not math.isnan(t.pnl):
            bucket["gross_pnl"] += t.pnl
            if t.pnl > 0:
                bucket["wins"] += 1
            elif t.pnl < 0:
                bucket["losses"] += 1
            else:
                bucket["flat"] += 1
        bucket["trades"] += 1
    for item in out.values():
        if item["trades"]:
            item["avg_pnl"] = item["gross_pnl"] / item["trades"]
    return sorted(out.values(), key=lambda x: x["side"])


def _by_exit_reason(trades: list[TradeRow]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for t in trades:
        reason = t.exit_reason or "UNKNOWN"
        bucket = out.setdefault(
            reason,
            {
                "exit_reason": reason,
                "trades": 0,
                "gross_pnl": 0.0,
                "avg_pnl": None,
            },
        )
        if not math.isnan(t.pnl):
            bucket["gross_pnl"] += t.pnl
        bucket["trades"] += 1
    for item in out.values():
        if item["trades"]:
            item["avg_pnl"] = item["gross_pnl"] / item["trades"]
    return sorted(out.values(), key=lambda x: x["gross_pnl"], reverse=True)


def _by_day(trades: list[TradeRow]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for t in trades:
        day = t.exit_ts.date().isoformat()
        bucket = out.setdefault(
            day,
            {
                "date": day,
                "trades": 0,
                "gross_pnl": 0.0,
                "avg_pnl": None,
                "wins": 0,
                "losses": 0,
                "flat": 0,
            },
        )
        if not math.isnan(t.pnl):
            bucket["gross_pnl"] += t.pnl
            if t.pnl > 0:
                bucket["wins"] += 1
            elif t.pnl < 0:
                bucket["losses"] += 1
            else:
                bucket["flat"] += 1
        bucket["trades"] += 1
    for item in out.values():
        if item["trades"]:
            item["avg_pnl"] = item["gross_pnl"] / item["trades"]
    return sorted(out.values(), key=lambda x: x["date"])


def _trade_to_row(trade: TradeRow) -> dict[str, Any]:
    duration = (trade.exit_ts - trade.entry_ts).total_seconds()
    return {
        "id": trade.trade_id,
        "market_ticker": trade.market_ticker,
        "side": trade.side,
        "entry_ts": trade.entry_ts.isoformat(),
        "entry_price": trade.entry_price,
        "qty": trade.qty,
        "exit_ts": trade.exit_ts.isoformat(),
        "exit_price": trade.exit_price,
        "exit_reason": trade.exit_reason,
        "pnl": trade.pnl,
        "quality_score": trade.quality_score,
        "panic_ret_3": trade.panic_ret_3,
        "vol_10": trade.vol_10,
        "vol_sum_5": trade.vol_sum_5,
        "p_open": trade.p_open,
        "p_base": trade.p_base,
        "hold_seconds": duration,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_report(
    label: str,
    summary: dict[str, Any],
    pnl_breakdown: dict[str, Any] | None,
    open_positions: list[dict[str, Any]],
    by_market: list[dict[str, Any]],
    by_side: list[dict[str, Any]],
    by_exit: list[dict[str, Any]],
    by_day: list[dict[str, Any]],
    quality_buckets: dict[str, int],
    missing_exit_price: int,
    missing_pnl: int,
) -> None:
    print(label)
    print("=" * len(label))
    print(f"Closed trades: {summary['closed_trades']}")
    print(f"Open trades:   {len(open_positions)}")
    if pnl_breakdown is not None:
        print(
            "Core PnL (tp/sl/max_hold): "
            f"{_fmt_float(pnl_breakdown['core_pnl'])} "
            f"({pnl_breakdown['core_trades']} trades)"
        )
        print(
            "Stale feed PnL: "
            f"{_fmt_float(pnl_breakdown['stale_feed_pnl'])} "
            f"({pnl_breakdown['stale_feed_trades']} trades)"
        )
        print(f"Total PnL:     {_fmt_float(pnl_breakdown['total_pnl'])}")
    print(f"Gross PnL:     {_fmt_float(summary['gross_pnl'])}")
    print(f"Avg PnL:       {_fmt_float(summary['avg_pnl'])}")
    print(f"Median PnL:    {_fmt_float(summary['median_pnl'])}")
    print(f"PnL Std:       {_fmt_float(summary['pnl_std'])}")
    print(f"Win rate:      {_fmt_pct(summary['win_rate'])}")
    print(f"Loss rate:     {_fmt_pct(summary['loss_rate'])}")
    print(f"Flat rate:     {_fmt_pct(summary['flat_rate'])}")
    print(f"Avg win:       {_fmt_float(summary['avg_win'])}")
    print(f"Avg loss:      {_fmt_float(summary['avg_loss'])}")
    print(f"Profit factor: {_fmt_float(summary['profit_factor'])}")
    print(f"Avg hold:      {_fmt_float(summary['avg_hold_seconds'])} sec")
    print(f"Median hold:   {_fmt_float(summary['median_hold_seconds'])} sec")
    print(f"Max drawdown:  {_fmt_float(summary['max_drawdown'])}")
    print(f"Largest win:   {_fmt_float(summary['largest_win'])}")
    print(f"Largest loss:  {_fmt_float(summary['largest_loss'])}")
    print(f"Missing pnl rows: {missing_pnl}")
    print(f"Missing exit_price rows: {missing_exit_price}")

    print("\nBy side")
    for row in by_side:
        print(
            f"  {row['side']}: trades={row['trades']}, gross_pnl={_fmt_float(row['gross_pnl'])}, avg_pnl={_fmt_float(row['avg_pnl'])}"
        )

    print("\nTop markets by PnL")
    for row in by_market[:10]:
        print(
            f"  {row['market_ticker']}: trades={row['trades']}, gross_pnl={_fmt_float(row['gross_pnl'])}, avg_pnl={_fmt_float(row['avg_pnl'])}"
        )

    print("\nBottom markets by PnL")
    for row in sorted(by_market, key=lambda x: x["gross_pnl"])[:10]:
        print(
            f"  {row['market_ticker']}: trades={row['trades']}, gross_pnl={_fmt_float(row['gross_pnl'])}, avg_pnl={_fmt_float(row['avg_pnl'])}"
        )

    print("\nExit reasons")
    for row in by_exit:
        print(
            f"  {row['exit_reason']}: trades={row['trades']}, gross_pnl={_fmt_float(row['gross_pnl'])}, avg_pnl={_fmt_float(row['avg_pnl'])}"
        )

    print("\nQuality score buckets")
    for key, value in quality_buckets.items():
        print(f"  {key}: {value}")

    if by_day:
        print("\nDaily PnL (last 10 days)")
        for row in by_day[-10:]:
            print(
                f"  {row['date']}: trades={row['trades']}, gross_pnl={_fmt_float(row['gross_pnl'])}"
            )


def _score_value(trade: TradeRow, column: str) -> float | None:
    if column == "quality_score":
        return trade.quality_score
    if column == "panic_ret_3":
        return trade.panic_ret_3
    if column == "vol_10":
        return trade.vol_10
    if column == "vol_sum_5":
        return trade.vol_sum_5
    if column == "p_open":
        return trade.p_open
    if column == "p_base":
        return trade.p_base
    return None


def _filter_by_score(
    trades: list[TradeRow],
    column: str,
    drop_bottom_percent: float,
    use_abs: bool,
) -> tuple[list[TradeRow], dict[str, Any]]:
    scores: list[float] = []
    for trade in trades:
        value = _score_value(trade, column)
        if value is None or math.isnan(value):
            continue
        scores.append(abs(value) if use_abs else value)
    if not scores:
        return trades, {"threshold": None, "dropped_missing": 0, "dropped_low": 0}
    scores.sort()
    drop_fraction = max(0.0, min(drop_bottom_percent / 100.0, 1.0))
    idx = int(len(scores) * drop_fraction)
    idx = min(max(idx, 0), len(scores) - 1)
    threshold = scores[idx]
    filtered: list[TradeRow] = []
    dropped_missing = 0
    dropped_low = 0
    for trade in trades:
        value = _score_value(trade, column)
        if value is None or math.isnan(value):
            dropped_missing += 1
            continue
        score = abs(value) if use_abs else value
        if score < threshold:
            dropped_low += 1
            continue
        filtered.append(trade)
    meta = {
        "threshold": threshold,
        "dropped_missing": dropped_missing,
        "dropped_low": dropped_low,
    }
    return filtered, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paper_trades sqlite DB.")
    parser.add_argument(
        "--db",
        default="data/paper_trades.sqlite",
        help="Path to paper_trades sqlite DB.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output directory for CSV/JSON reports.",
    )
    parser.add_argument(
        "--exclude-pnl-above",
        type=float,
        default=None,
        help="Exclude closed trades with pnl above this value.",
    )
    parser.add_argument(
        "--exclude-exit-reason",
        default=None,
        help="Exclude closed trades with this exit_reason value.",
    )
    parser.add_argument(
        "--since-hours",
        type=float,
        default=None,
        help="Only include closed trades with exit_ts within the last X hours (UTC).",
    )
    parser.add_argument(
        "--nba-log",
        default=None,
        help="Optional nba_engine.log path for gate analysis.",
    )
    parser.add_argument(
        "--log-since-hours",
        type=float,
        default=None,
        help="Only include log events within the last X hours (UTC).",
    )
    parser.add_argument(
        "--score-column",
        default=None,
        choices=["quality_score", "panic_ret_3", "vol_10", "vol_sum_5", "p_open", "p_base"],
        help="Score column for dropping bottom trades.",
    )
    parser.add_argument(
        "--drop-bottom-percent",
        type=float,
        default=None,
        help="Drop the bottom X percent by score column.",
    )
    parser.add_argument(
        "--score-abs",
        action="store_true",
        help="Use absolute value of score column before filtering.",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    closed_trades_all = _load_closed_trades(conn, args.exclude_pnl_above)
    if args.since_hours is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=args.since_hours)
        closed_trades_all = [t for t in closed_trades_all if t.exit_ts >= cutoff]
    open_positions = _load_open_positions(conn)

    summary = _summarize_trades(closed_trades_all)
    pnl_breakdown = _pnl_breakdown(closed_trades_all)
    by_market = _by_market(closed_trades_all)
    by_side = _by_side(closed_trades_all)
    by_exit = _by_exit_reason(closed_trades_all)
    by_day = _by_day(closed_trades_all)
    quality_buckets = _bucket_quality(closed_trades_all)

    missing_exit_price = _row_count(
        conn,
        """
        SELECT COUNT(*)
        FROM paper_trades
        WHERE exit_ts IS NOT NULL AND exit_price IS NULL
        """,
    )
    missing_pnl = _row_count(
        conn,
        """
        SELECT COUNT(*)
        FROM paper_trades
        WHERE exit_ts IS NOT NULL AND pnl IS NULL
        """,
    )

    _print_report(
        "Paper Trades Analysis (baseline)",
        summary,
        pnl_breakdown,
        open_positions,
        by_market,
        by_side,
        by_exit,
        by_day,
        quality_buckets,
        missing_exit_price,
        missing_pnl,
    )

    stale_filtered = [trade for trade in closed_trades_all if (trade.exit_reason or "") != "stale_feed"]
    if len(stale_filtered) != len(closed_trades_all):
        stale_summary = _summarize_trades(stale_filtered)
        stale_breakdown = _pnl_breakdown(stale_filtered)
        stale_by_market = _by_market(stale_filtered)
        stale_by_side = _by_side(stale_filtered)
        stale_by_exit = _by_exit_reason(stale_filtered)
        stale_by_day = _by_day(stale_filtered)
        stale_quality = _bucket_quality(stale_filtered)
        print("")
        _print_report(
            "Paper Trades Analysis (excluding stale_feed)",
            stale_summary,
            stale_breakdown,
            open_positions,
            stale_by_market,
            stale_by_side,
            stale_by_exit,
            stale_by_day,
            stale_quality,
            missing_exit_price,
            missing_pnl,
        )

    if args.exclude_exit_reason:
        custom_filtered = [
            trade for trade in closed_trades_all if (trade.exit_reason or "") != args.exclude_exit_reason
        ]
        if len(custom_filtered) != len(closed_trades_all):
            custom_summary = _summarize_trades(custom_filtered)
            custom_breakdown = _pnl_breakdown(custom_filtered)
            custom_by_market = _by_market(custom_filtered)
            custom_by_side = _by_side(custom_filtered)
            custom_by_exit = _by_exit_reason(custom_filtered)
            custom_by_day = _by_day(custom_filtered)
            custom_quality = _bucket_quality(custom_filtered)
            print("")
            _print_report(
                f"Paper Trades Analysis (excluding {args.exclude_exit_reason})",
                custom_summary,
                custom_breakdown,
                open_positions,
                custom_by_market,
                custom_by_side,
                custom_by_exit,
                custom_by_day,
                custom_quality,
                missing_exit_price,
                missing_pnl,
            )

    score_meta: dict[str, Any] | None = None
    score_filtered: list[TradeRow] | None = None
    if args.score_column and args.drop_bottom_percent is not None:
        score_filtered, score_meta = _filter_by_score(
            closed_trades_all,
            args.score_column,
            args.drop_bottom_percent,
            args.score_abs,
        )
        score_summary = _summarize_trades(score_filtered)
        score_breakdown = _pnl_breakdown(score_filtered)
        score_by_market = _by_market(score_filtered)
        score_by_side = _by_side(score_filtered)
        score_by_exit = _by_exit_reason(score_filtered)
        score_by_day = _by_day(score_filtered)
        score_quality = _bucket_quality(score_filtered)
        print("")
        _print_report(
            f"Score-Filtered ({args.score_column}, bottom {args.drop_bottom_percent:.1f}%)",
            score_summary,
            score_breakdown,
            open_positions,
            score_by_market,
            score_by_side,
            score_by_exit,
            score_by_day,
            score_quality,
            missing_exit_price,
            missing_pnl,
        )
        if score_meta:
            print(
                f"Score filter threshold: {_fmt_float(score_meta['threshold'])}, "
                f"dropped_low={score_meta['dropped_low']}, dropped_missing={score_meta['dropped_missing']}"
            )

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(out_dir / "by_market.csv", by_market)
        _write_csv(out_dir / "by_side.csv", by_side)
        _write_csv(out_dir / "by_exit_reason.csv", by_exit)
        _write_csv(out_dir / "by_day.csv", by_day)
        _write_csv(out_dir / "closed_trades.csv", [_trade_to_row(t) for t in closed_trades_all])
        _write_csv(out_dir / "open_positions.csv", open_positions)
        if score_filtered is not None:
            _write_csv(out_dir / "score_filtered_trades.csv", [_trade_to_row(t) for t in score_filtered])
        with (out_dir / "summary.json").open("w") as handle:
            json.dump(
                {
                    "summary": summary,
                    "quality_buckets": quality_buckets,
                    "open_positions": open_positions,
                    "missing_exit_price": missing_exit_price,
                    "missing_pnl": missing_pnl,
                    "score_filter": score_meta,
                },
                handle,
                indent=2,
            )
        print(f"\nWrote reports to: {out_dir}")

    if args.nba_log:
        log_path = Path(args.nba_log)
        events = _load_log_events(log_path)
        log_since_hours = args.log_since_hours if args.log_since_hours is not None else args.since_hours
        events = _filter_log_events(events, log_since_hours)
        panic_detected, failed_counts, blocked_total = _summarize_gate_blocks(events)
        print("\nNBA Engine Gate Analysis (nba_engine.log)")
        if log_since_hours is not None:
            print(f"- window: last {log_since_hours} hours (UTC)")
        if not events:
            print("- no events found")
        else:
            print(f"- panic_detected: {panic_detected}")
            print(f"- blocked_trades: {blocked_total}")
            if failed_counts:
                top = failed_counts.most_common(10)
                print("- failed_reason breakdown:")
                for reason, count in top:
                    pct = (count / blocked_total * 100.0) if blocked_total else 0.0
                    print(f"  - {reason}: {count} ({pct:.1f}%)")
            else:
                print("- failed_reason breakdown: none")


if __name__ == "__main__":
    main()
