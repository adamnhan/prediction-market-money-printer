from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _update_column(conn: sqlite3.Connection, table: str, col: str, scale: float) -> int:
    sql = f"UPDATE {table} SET {col} = {col} * ? WHERE {col} IS NOT NULL"
    cur = conn.execute(sql, (scale,))
    return cur.rowcount


def _backup_db(db_path: Path, backup_path: Path) -> None:
    if backup_path.exists():
        raise FileExistsError(f"backup already exists: {backup_path}")
    with sqlite3.connect(db_path) as src, sqlite3.connect(backup_path) as dst:
        src.backup(dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescale candle price columns in a sqlite DB (e.g. cents -> dollars)."
    )
    parser.add_argument(
        "--db",
        default="data/phase1_candles.sqlite",
        help="Path to candle sqlite DB.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="Scale factor to apply to price columns.",
    )
    parser.add_argument(
        "--backup",
        default=None,
        help="Optional backup path; created before modifying.",
    )
    parser.add_argument(
        "--table",
        default="candles",
        help="Target table name.",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"missing db: {db_path}")

    if args.backup:
        _backup_db(db_path, Path(args.backup))

    with sqlite3.connect(db_path) as conn:
        columns = _existing_columns(conn, args.table)
        target_cols = [col for col in ("close", "p_open", "p_base") if col in columns]
        if not target_cols:
            raise RuntimeError(f"no target columns found in {args.table}")
        total = 0
        for col in target_cols:
            total += _update_column(conn, args.table, col, args.scale)
        conn.commit()
    print(f"updated {total} rows in {args.table} ({', '.join(target_cols)})")


if __name__ == "__main__":
    main()
