from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

if __package__ is None and str(Path(__file__).parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[2]))

from cross_venue_arb.storage.sqlite import connect


def _parse_ts(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 shadow execution report")
    parser.add_argument("--since", type=float, default=None, help="Only include trades after timestamp")
    args = parser.parse_args()

    conn = connect()
    rows = conn.execute(
        """
        SELECT game_key,
               detected_ts,
               latency_ms,
               detected_edge,
               detected_size,
               status,
               realized_pnl
        FROM shadow_trades
        """
    ).fetchall()
    conn.close()

    total = 0
    by_status: dict[str, int] = defaultdict(int)
    pnl_total = 0.0
    pnl_count = 0
    edge_sum = 0.0
    edge_count = 0
    latency_sum = 0.0
    latency_count = 0
    by_game: dict[str, int] = defaultdict(int)

    for row in rows:
        detected_ts = _parse_ts(row[1])
        if args.since is not None and detected_ts is not None and detected_ts < args.since:
            continue
        total += 1
        game_key = str(row[0])
        by_game[game_key] += 1
        status = str(row[5])
        by_status[status] += 1
        edge_sum += float(row[3])
        edge_count += 1
        latency_sum += float(row[2])
        latency_count += 1
        if row[6] is not None:
            pnl_total += float(row[6])
            pnl_count += 1

    if total == 0:
        print("No shadow trades found.")
        return

    print(f"trades={total}")
    for status, count in sorted(by_status.items(), key=lambda x: x[0]):
        print(f"{status.lower()}={count}")
    print(f"avg_edge={edge_sum / max(1, edge_count):.5f}")
    print(f"avg_latency_ms={latency_sum / max(1, latency_count):.1f}")
    if pnl_count:
        print(f"avg_realized_pnl={pnl_total / pnl_count:.5f}")
        print(f"sum_realized_pnl={pnl_total:.5f}")
    print(f"games_seen={len(by_game)}")


if __name__ == "__main__":
    main()
