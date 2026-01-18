from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from collections import Counter
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nba_engine.artifacts import load_artifacts
from nba_engine.phase4 import (
    ActivityTracker,
    CandleRow,
    CandleStream,
    COOLDOWN,
    ENTRY_DELAY,
    KILL_SWITCH_SAMPLE,
    MemoryPaperStore,
    PanicState,
    Position,
    SignalEngine,
    _exit_signal,
)
from nba_engine.phase5 import (
    EntryOrder,
    ExitOrder,
    MemoryOrderLedger,
    PaperOrderAdapter,
    entry_decision,
    _entry_order_key,
    _exit_order_key,
)


def _safe_ts(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _trade_payload(trade: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": trade.get("id"),
        "market_ticker": trade.get("market_ticker"),
        "side": trade.get("side"),
        "entry_ts": _safe_ts(trade.get("entry_ts")),
        "entry_price": trade.get("entry_price"),
        "exit_ts": _safe_ts(trade.get("exit_ts")),
        "exit_price": trade.get("exit_price"),
        "exit_reason": trade.get("exit_reason"),
        "pnl": trade.get("pnl"),
        "quality_score": trade.get("quality_score"),
        "qty": trade.get("qty"),
    }


def _log_debug(
    sink: list[dict[str, Any]] | None,
    handle: Any,
    payload: dict[str, Any],
) -> None:
    if sink is not None:
        sink.append(payload)
    if handle is not None:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def replay(
    candle_db_path: Path,
    artifacts_path: Path,
    *,
    debug: bool = False,
    debug_log: Path | None = None,
    price_scale: float = 1.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    artifacts = load_artifacts(artifacts_path)
    stream = CandleStream(candle_db_path)
    store = MemoryPaperStore()
    ledger = MemoryOrderLedger()
    engine = SignalEngine(artifacts)
    adapter = PaperOrderAdapter(store, ledger)
    activity = ActivityTracker()

    pending: dict[str, PanicState] = {}
    positions = store.load_open_positions()
    last_prices: dict[str, float] = {}
    kill_switch = False
    mean_pnl, sample_count = store.recent_mean_pnl(KILL_SWITCH_SAMPLE, artifacts.quality_cutoff)
    if sample_count == KILL_SWITCH_SAMPLE and mean_pnl is not None and mean_pnl < 0:
        kill_switch = True

    debug_events: list[dict[str, Any]] | None = [] if debug and debug_log is None else None
    debug_handle = debug_log.open("w", encoding="utf-8") if debug_log else None
    reason_counts: Counter[str] = Counter()
    panic_count = 0
    entry_allowed = 0
    entry_blocked = 0

    for candle in stream.fetch_all():
        if price_scale != 1.0:
            candle = CandleRow(
                rowid=candle.rowid,
                market_ticker=candle.market_ticker,
                start_ts=candle.start_ts,
                close=candle.close * price_scale,
                trade_active=candle.trade_active,
                ret_3=candle.ret_3,
                vol_10=candle.vol_10,
                vol_sum_5=candle.vol_sum_5,
                gap_recent_5=candle.gap_recent_5,
                p_open=candle.p_open * price_scale,
                p_base=candle.p_base * price_scale,
            )
        active_last_3 = activity.update(candle.market_ticker, candle.trade_active)
        if candle.close == candle.close:
            last_prices[candle.market_ticker] = candle.close

        position = positions.get(candle.market_ticker)
        if position:
            reason, exit_price, pnl = _exit_signal(
                position,
                candle.start_ts,
                candle.close,
                last_prices.get(candle.market_ticker),
            )
            if reason and exit_price is not None and pnl is not None:
                result = adapter.submit_exit(
                    ExitOrder(
                        order_key=_exit_order_key(position),
                        position=position,
                        exit_ts=candle.start_ts,
                        exit_price=exit_price,
                        exit_reason=reason,
                        pnl=pnl,
                    )
                )
                if result.accepted:
                    positions.pop(candle.market_ticker, None)
                if not kill_switch:
                    mean_pnl, sample_count = store.recent_mean_pnl(
                        KILL_SWITCH_SAMPLE, artifacts.quality_cutoff
                    )
                    if (
                        sample_count == KILL_SWITCH_SAMPLE
                        and mean_pnl is not None
                        and mean_pnl < 0
                    ):
                        kill_switch = True

        panic = pending.get(candle.market_ticker)
        if panic:
            if candle.start_ts >= panic.detected_ts + ENTRY_DELAY:
                allowed, quality_score, reasons = entry_decision(
                    panic=panic,
                    candle=candle,
                    active_last_3=active_last_3,
                    engine=engine,
                    positions=positions,
                    store=store,
                    kill_switch=kill_switch,
                )
                if allowed and not reasons:
                    side = "NO" if panic.direction == "UNDERDOG_UP" else "YES"
                    result = adapter.submit_entry(
                        EntryOrder(
                            order_key=_entry_order_key(panic),
                            market_ticker=candle.market_ticker,
                            side=side,
                            qty=1,
                            entry_ts=candle.start_ts,
                            entry_price=candle.close,
                            panic=panic,
                            quality_score=quality_score,
                        )
                    )
                    if result.accepted and result.order_id is not None:
                        positions[candle.market_ticker] = Position(
                            id=result.order_id,
                            market_ticker=candle.market_ticker,
                            side=side,
                            entry_ts=candle.start_ts,
                            entry_price=candle.close,
                            qty=1,
                        )
                    entry_allowed += 1
                    if debug:
                        _log_debug(
                            debug_events,
                            debug_handle,
                            {
                                "event": "entry_allowed",
                                "market_ticker": candle.market_ticker,
                                "ts": candle.start_ts.isoformat(),
                                "direction": panic.direction,
                                "quality_score": quality_score,
                            },
                        )
                else:
                    entry_blocked += 1
                    for reason in reasons:
                        reason_counts[reason] += 1
                    if debug:
                        _log_debug(
                            debug_events,
                            debug_handle,
                            {
                                "event": "entry_blocked",
                                "market_ticker": candle.market_ticker,
                                "ts": candle.start_ts.isoformat(),
                                "direction": panic.direction,
                                "quality_score": quality_score,
                                "reasons": reasons,
                            },
                        )
                pending.pop(candle.market_ticker, None)
            continue

        panic = engine.detect_panic(candle)
        if panic:
            pending[candle.market_ticker] = panic
            panic_count += 1
            if debug:
                _log_debug(
                    debug_events,
                    debug_handle,
                    {
                        "event": "panic_detected",
                        "market_ticker": candle.market_ticker,
                        "ts": candle.start_ts.isoformat(),
                        "direction": panic.direction,
                        "p_open": panic.p_open,
                        "p_base": panic.p_base,
                        "ret_3": panic.ret_3,
                        "vol_10": panic.vol_10,
                        "vol_sum_5": panic.vol_sum_5,
                    },
                )

    if debug_handle is not None:
        debug_handle.close()

    trades: list[dict[str, Any]] = list(getattr(store, "_trades", []))
    closed_trades = [trade for trade in trades if trade.get("exit_ts") is not None]
    total_pnl = float(
        sum(float(trade["pnl"]) for trade in closed_trades if trade.get("pnl") is not None)
    )
    report = {
        "realized_pnl": total_pnl,
        "entries": len(trades),
        "closed_trades": len(closed_trades),
        "open_trades": len(trades) - len(closed_trades),
    }
    debug_summary = {
        "panic_detected": panic_count,
        "entry_allowed": entry_allowed,
        "entry_blocked": entry_blocked,
        "entry_blocked_reasons": dict(reason_counts),
        "debug_events": debug_events if debug_events is not None else None,
        "debug_log": str(debug_log) if debug_log else None,
    }
    return report, trades, debug_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay NBA engine entry/exit logic against a candle SQLite DB."
    )
    parser.add_argument(
        "--candle-db",
        default=os.getenv("KALSHI_CANDLE_DB_PATH", "data/phase1_candles.sqlite"),
        help="Path to phase1 candle sqlite DB.",
    )
    parser.add_argument(
        "--artifacts",
        default=os.getenv("STRATEGY_ARTIFACTS_PATH", "strategy_artifacts.json"),
        help="Path to strategy artifacts JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of trades to print per section (0 disables listing).",
    )
    parser.add_argument(
        "--include-open",
        action="store_true",
        help="Include open trades in the printed output.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log entry decisions and panic detections.",
    )
    parser.add_argument(
        "--debug-log",
        default=None,
        help="Write debug events to a JSONL file instead of stdout.",
    )
    parser.add_argument(
        "--price-scale",
        type=float,
        default=1.0,
        help="Scale price fields (close/p_open/p_base) before evaluation.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print candle DB schema and basic coverage stats, then exit.",
    )
    args = parser.parse_args()

    debug_log = Path(args.debug_log) if args.debug_log else None
    if args.inspect:
        conn = sqlite3.connect(Path(args.candle_db))
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        print(json.dumps({"tables": tables}, indent=2, sort_keys=True))
        try:
            cols = conn.execute("PRAGMA table_info(candles)").fetchall()
        except sqlite3.Error as exc:
            print(json.dumps({"candles_schema_error": str(exc)}, indent=2, sort_keys=True))
            conn.close()
            return
        schema = [{"cid": c[0], "name": c[1], "type": c[2]} for c in cols]
        print(json.dumps({"candles_columns": schema}, indent=2, sort_keys=True))
        row_count = conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
        print(json.dumps({"candles_rows": row_count}, indent=2, sort_keys=True))
        required = [
            "p_open",
            "p_base",
            "ret_3",
            "vol_10",
            "vol_sum_5",
            "trade_active",
            "gap_recent_5",
            "close",
        ]
        coverage = {}
        for col in required:
            try:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM candles WHERE {col} IS NOT NULL"
                ).fetchone()[0]
            except sqlite3.Error:
                count = None
            coverage[col] = count
        print(json.dumps({"candles_non_null": coverage}, indent=2, sort_keys=True))
        sample = conn.execute(
            "SELECT market_ticker, start_ts, close, p_open, p_base, ret_3, vol_10, vol_sum_5, trade_active, gap_recent_5 "
            "FROM candles LIMIT 5"
        ).fetchall()
        print(json.dumps({"candles_sample": sample}, indent=2, sort_keys=True))
        conn.close()
        return

    report, trades, debug_summary = replay(
        Path(args.candle_db),
        Path(args.artifacts),
        debug=args.debug,
        debug_log=debug_log,
        price_scale=args.price_scale,
    )
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.debug:
        print("\ndebug_summary")
        print(json.dumps(debug_summary, indent=2, sort_keys=True))

    if args.limit <= 0:
        return

    closed_trades = [trade for trade in trades if trade.get("exit_ts") is not None]
    closed_trades.sort(key=lambda trade: trade.get("exit_ts"))
    if closed_trades:
        print("\nclosed_trades_sample")
        for trade in closed_trades[: args.limit]:
            print(json.dumps(_trade_payload(trade), sort_keys=True))

    if args.include_open:
        open_trades = [trade for trade in trades if trade.get("exit_ts") is None]
        open_trades.sort(key=lambda trade: trade.get("entry_ts"))
        if open_trades:
            print("\nopen_trades_sample")
            for trade in open_trades[: args.limit]:
                print(json.dumps(_trade_payload(trade), sort_keys=True))


if __name__ == "__main__":
    main()
