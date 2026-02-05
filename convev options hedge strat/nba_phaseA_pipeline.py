#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


@dataclass
class ParsedMarket:
    market_type: str
    team_code: str | None
    threshold_k: int | None
    threshold_raw: str | None
    k_definition: str | None


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_db_path() -> str:
    return os.path.join(_repo_root(), "data", "nba_phaseA.sqlite")


def _default_l2_path() -> str:
    return os.path.join(_repo_root(), "data", "l2_orderbook.sqlite")


def _default_enriched_path() -> str:
    return os.path.join(_repo_root(), "data", "enriched_markets.csv")


def _default_metadata_path() -> str:
    return os.path.join(_repo_root(), "data", "market_metadata.jsonl")


def _default_l1_path() -> str:
    return os.path.join(_repo_root(), "data", "l1_snapshots.jsonl")


def _default_results_path() -> str:
    return os.path.join(_repo_root(), "data", "nba_results.csv")


def init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                league TEXT,
                home_team TEXT,
                away_team TEXT,
                start_time TEXT,
                end_time TEXT,
                source_event_key TEXT,
                final_margin INTEGER,
                final_home_score INTEGER,
                final_away_score INTEGER,
                meta_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS markets (
                market_ticker TEXT PRIMARY KEY,
                event_id TEXT,
                series_ticker TEXT,
                market_type TEXT,
                threshold_k INTEGER,
                threshold_raw TEXT,
                k_definition TEXT,
                team_code TEXT,
                team_is_home INTEGER,
                polarity TEXT,
                close_time TEXT,
                tick_size REAL,
                status TEXT,
                title TEXT,
                subtitle TEXT,
                rules_text TEXT,
                meta_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS book_snapshots (
                id INTEGER PRIMARY KEY,
                ts_utc_ms INTEGER NOT NULL,
                market_ticker TEXT NOT NULL,
                yes_levels_json TEXT NOT NULL,
                no_levels_json TEXT NOT NULL,
                top_yes INTEGER,
                top_no INTEGER,
                levels_yes INTEGER NOT NULL,
                levels_no INTEGER NOT NULL,
                source TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tape_trades (
                id INTEGER PRIMARY KEY,
                ts_utc_ms INTEGER NOT NULL,
                market_ticker TEXT NOT NULL,
                side TEXT,
                price INTEGER,
                size INTEGER,
                trade_id TEXT,
                raw_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS event_ladders (
                event_id TEXT NOT NULL,
                ladder_family TEXT NOT NULL,
                team_code TEXT NOT NULL,
                thresholds_json TEXT NOT NULL,
                tickers_by_k_json TEXT NOT NULL,
                quality_flags_json TEXT NOT NULL,
                PRIMARY KEY (event_id, ladder_family, team_code)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_outcomes (
                market_ticker TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                final_margin INTEGER NOT NULL,
                outcome_yes INTEGER NOT NULL,
                outcome_rule TEXT NOT NULL,
                computed_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_markets_event ON markets(event_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_book_snapshots_ticker_ts ON book_snapshots(market_ticker, ts_utc_ms)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tape_trades_ticker_ts ON tape_trades(market_ticker, ts_utc_ms)")
        conn.commit()
    print(f"[phaseA] Initialized DB at {db_path}")


def _ensure_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
        ).fetchone()
    if row is None:
        init_db(db_path)


def _parse_event_team_codes(event_ticker: str) -> tuple[str | None, str | None]:
    tail = event_ticker.split("-")[-1]
    if not tail or not tail.isalpha():
        return None, None
    if len(tail) % 2 == 0 and 4 <= len(tail) <= 8:
        half = len(tail) // 2
        return tail[:half], tail[half:]
    return None, None


def _league_from_series(series_ticker: str | None) -> str | None:
    if not series_ticker:
        return None
    upper = series_ticker.upper()
    if "NBA" in upper:
        return "NBA"
    if "NCAAM" in upper or "NCAAB" in upper:
        return "NCAAB"
    if "NFL" in upper:
        return "NFL"
    if "MLB" in upper:
        return "MLB"
    return None


def _classify_market(series_ticker: str | None, title: str | None) -> str:
    series = (series_ticker or "").upper()
    title_u = (title or "").upper()
    if "SPREAD" in series or "SPREAD" in title_u:
        return "SPREAD"
    if "TOTAL" in series or "TOTAL" in title_u:
        return "TOTAL_POINTS"
    if "GAME" in series or "MONEYLINE" in title_u or "WINNER" in title_u:
        return "MONEYLINE"
    if "WINBY" in series or "WIN BY" in title_u:
        return "WIN_BY"
    return "OTHER"


def _parse_market(series_ticker: str | None, market_ticker: str, title: str | None) -> ParsedMarket:
    market_type = _classify_market(series_ticker, title)
    suffix = market_ticker.split("-")[-1]
    if market_type == "SPREAD":
        match = re.match(r"([A-Z]+)(-?\d+)$", suffix)
        if match:
            team = match.group(1)
            line = int(match.group(2))
            return ParsedMarket(
                market_type=market_type,
                team_code=team,
                threshold_k=line,
                threshold_raw=match.group(2),
                k_definition="spread_line",
            )
    if market_type == "TOTAL_POINTS":
        if suffix.isdigit() or (suffix.startswith("-") and suffix[1:].isdigit()):
            line = int(suffix)
            return ParsedMarket(
                market_type=market_type,
                team_code=None,
                threshold_k=line,
                threshold_raw=suffix,
                k_definition="total_points_line",
            )
    if market_type in {"MONEYLINE", "WIN_BY"}:
        if suffix.isalpha():
            return ParsedMarket(
                market_type=market_type,
                team_code=suffix,
                threshold_k=1,
                threshold_raw="1",
                k_definition="win_strict",
            )
    return ParsedMarket(
        market_type=market_type,
        team_code=None,
        threshold_k=None,
        threshold_raw=None,
        k_definition=None,
    )


def _upsert_event(
    conn: sqlite3.Connection,
    event_id: str,
    league: str | None,
    home_team: str | None,
    away_team: str | None,
    start_time: str | None,
    end_time: str | None,
    source_event_key: str | None,
    meta_json: str | None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO events (
            event_id, league, home_team, away_team, start_time, end_time,
            source_event_key, final_margin, final_home_score, final_away_score, meta_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            league,
            home_team,
            away_team,
            start_time,
            end_time,
            source_event_key,
            None,
            None,
            None,
            meta_json,
        ),
    )


def _upsert_market(
    conn: sqlite3.Connection,
    market_ticker: str,
    event_id: str | None,
    series_ticker: str | None,
    market_type: str,
    threshold_k: int | None,
    threshold_raw: str | None,
    k_definition: str | None,
    team_code: str | None,
    team_is_home: int | None,
    polarity: str | None,
    close_time: str | None,
    tick_size: float | None,
    status: str | None,
    title: str | None,
    subtitle: str | None,
    rules_text: str | None,
    meta_json: str | None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO markets (
            market_ticker, event_id, series_ticker, market_type, threshold_k,
            threshold_raw, k_definition, team_code, team_is_home, polarity,
            close_time, tick_size, status, title, subtitle, rules_text, meta_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            market_ticker,
            event_id,
            series_ticker,
            market_type,
            threshold_k,
            threshold_raw,
            k_definition,
            team_code,
            team_is_home,
            polarity,
            close_time,
            tick_size,
            status,
            title,
            subtitle,
            rules_text,
            meta_json,
        ),
    )


def ingest_enriched_markets(db_path: str, enriched_path: str) -> int:
    if not os.path.exists(enriched_path):
        raise FileNotFoundError(enriched_path)
    _ensure_db(db_path)
    inserted = 0
    with sqlite3.connect(db_path) as conn, open(enriched_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            market_ticker = row.get("ticker") or ""
            event_id = row.get("event_ticker")
            if not market_ticker or not event_id:
                continue
            series_ticker = row.get("series_ticker")
            title = row.get("title")
            parsed = _parse_market(series_ticker, market_ticker, title)
            away_code, home_code = _parse_event_team_codes(event_id)
            team_is_home = None
            if parsed.team_code and home_code and parsed.team_code == home_code:
                team_is_home = 1
            elif parsed.team_code and away_code and parsed.team_code == away_code:
                team_is_home = 0
            league = _league_from_series(series_ticker)
            _upsert_event(
                conn,
                event_id=event_id,
                league=league,
                home_team=home_code,
                away_team=away_code,
                start_time=row.get("event_time"),
                end_time=None,
                source_event_key=event_id,
                meta_json=json.dumps({"source": "enriched_markets", "row": row}, separators=(",", ":")),
            )
            _upsert_market(
                conn,
                market_ticker=market_ticker,
                event_id=event_id,
                series_ticker=series_ticker,
                market_type=parsed.market_type,
                threshold_k=parsed.threshold_k,
                threshold_raw=parsed.threshold_raw,
                k_definition=parsed.k_definition,
                team_code=parsed.team_code,
                team_is_home=team_is_home,
                polarity=parsed.team_code,
                close_time=row.get("close_time"),
                tick_size=None,
                status=row.get("status"),
                title=title,
                subtitle=row.get("subtitle"),
                rules_text=None,
                meta_json=json.dumps({"source": "enriched_markets", "row": row}, separators=(",", ":")),
            )
            inserted += 1
        conn.commit()
    print(f"[phaseA] Ingested {inserted} markets from {enriched_path}")
    return inserted


def _iter_metadata_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def ingest_metadata_jsonl(db_path: str, metadata_path: str) -> int:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(metadata_path)
    _ensure_db(db_path)
    inserted = 0
    with sqlite3.connect(db_path) as conn:
        for record in _iter_metadata_jsonl(metadata_path):
            market = record.get("market") or {}
            market_ticker = market.get("ticker") or market.get("market_ticker")
            event_id = market.get("event_ticker")
            if not market_ticker or not event_id:
                continue
            series_ticker = market.get("series_ticker")
            title = market.get("title") or market.get("event_title")
            parsed = _parse_market(series_ticker, market_ticker, title)
            away_code, home_code = _parse_event_team_codes(event_id)
            team_is_home = None
            if parsed.team_code and home_code and parsed.team_code == home_code:
                team_is_home = 1
            elif parsed.team_code and away_code and parsed.team_code == away_code:
                team_is_home = 0
            league = _league_from_series(series_ticker)
            _upsert_event(
                conn,
                event_id=event_id,
                league=league,
                home_team=home_code,
                away_team=away_code,
                start_time=market.get("event_start_time") or market.get("open_time"),
                end_time=market.get("close_time"),
                source_event_key=event_id,
                meta_json=json.dumps(
                    {"source": "market_metadata", "record": record}, separators=(",", ":")
                ),
            )
            _upsert_market(
                conn,
                market_ticker=str(market_ticker).upper(),
                event_id=event_id,
                series_ticker=series_ticker,
                market_type=parsed.market_type,
                threshold_k=parsed.threshold_k,
                threshold_raw=parsed.threshold_raw,
                k_definition=parsed.k_definition,
                team_code=parsed.team_code,
                team_is_home=team_is_home,
                polarity=parsed.team_code,
                close_time=market.get("close_time"),
                tick_size=market.get("tick_size"),
                status=market.get("status"),
                title=title,
                subtitle=market.get("subtitle"),
                rules_text=market.get("rules_text") or market.get("rules"),
                meta_json=json.dumps(
                    {"source": "market_metadata", "record": record}, separators=(",", ":")
                ),
            )
            inserted += 1
        conn.commit()
    print(f"[phaseA] Ingested {inserted} markets from {metadata_path}")
    return inserted


def ingest_l2(db_path: str, l2_path: str) -> tuple[int, int]:
    if not os.path.exists(l2_path):
        raise FileNotFoundError(l2_path)
    _ensure_db(db_path)
    snapshot_rows = 0
    trade_rows = 0
    with sqlite3.connect(db_path) as dst, sqlite3.connect(l2_path) as src:
        src.row_factory = sqlite3.Row
        for row in src.execute(
            """
            SELECT ts_utc_ms, market_ticker, bids_json, asks_json,
                   top_bid, top_ask, levels_bid, levels_ask
            FROM l2_checkpoints
            """
        ):
            dst.execute(
                """
                INSERT INTO book_snapshots (
                    ts_utc_ms, market_ticker, yes_levels_json, no_levels_json,
                    top_yes, top_no, levels_yes, levels_no, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["ts_utc_ms"],
                    row["market_ticker"],
                    row["bids_json"],
                    row["asks_json"],
                    row["top_bid"],
                    row["top_ask"],
                    row["levels_bid"],
                    row["levels_ask"],
                    "l2_checkpoints",
                ),
            )
            snapshot_rows += 1
        for row in src.execute(
            """
            SELECT ts_utc_ms, market_ticker, payload_json
            FROM l2_messages
            WHERE channel = 'trade'
            """
        ):
            payload = {}
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                payload = {}
            data = payload.get("data") if isinstance(payload, dict) else None
            data = data if isinstance(data, dict) else payload if isinstance(payload, dict) else {}
            price = data.get("price") or data.get("yes_price")
            size = data.get("size") or data.get("count") or data.get("volume")
            side = data.get("side") or data.get("action")
            trade_id = data.get("trade_id") or data.get("id")
            dst.execute(
                """
                INSERT INTO tape_trades (
                    ts_utc_ms, market_ticker, side, price, size, trade_id, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["ts_utc_ms"],
                    row["market_ticker"],
                    side,
                    price,
                    size,
                    trade_id,
                    row["payload_json"],
                ),
            )
            trade_rows += 1
        dst.commit()
    print(f"[phaseA] Ingested {snapshot_rows} book snapshots, {trade_rows} trades from {l2_path}")
    return snapshot_rows, trade_rows


def ingest_l1(db_path: str, l1_path: str) -> int:
    if not os.path.exists(l1_path):
        raise FileNotFoundError(l1_path)
    _ensure_db(db_path)
    inserted = 0
    with sqlite3.connect(db_path) as conn, open(l1_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts_utc_ms = record.get("ts_utc_ms")
            ticker = record.get("market_ticker")
            yes_levels = record.get("yes_levels") or []
            no_levels = record.get("no_levels") or []
            top_yes = record.get("top_yes")
            top_no = record.get("top_no")
            if not ts_utc_ms or not ticker:
                continue
            conn.execute(
                """
                INSERT INTO book_snapshots (
                    ts_utc_ms, market_ticker, yes_levels_json, no_levels_json,
                    top_yes, top_no, levels_yes, levels_no, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(ts_utc_ms),
                    str(ticker).upper(),
                    json.dumps(yes_levels, separators=(",", ":"), ensure_ascii=True),
                    json.dumps(no_levels, separators=(",", ":"), ensure_ascii=True),
                    top_yes,
                    top_no,
                    len(yes_levels),
                    len(no_levels),
                    "l1_snapshot",
                ),
            )
            inserted += 1
        conn.commit()
    print(f"[phaseA] Ingested {inserted} L1 snapshots from {l1_path}")
    return inserted


def build_ladders(db_path: str) -> int:
    _ensure_db(db_path)
    ladders_built = 0
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT market_ticker, event_id, market_type, threshold_k, team_code, team_is_home
            FROM markets
            WHERE threshold_k IS NOT NULL AND team_code IS NOT NULL
            """
        ).fetchall()
        grouped: dict[tuple[str, str, str], list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            if row["market_type"] in {"SPREAD", "MONEYLINE", "WIN_BY", "MARGIN_GE_K"}:
                family = "TEAM_MARGIN"
            else:
                continue
            key = (row["event_id"], family, row["team_code"])
            grouped[key].append(row)
        for (event_id, family, team_code), items in grouped.items():
            counts: dict[int, int] = defaultdict(int)
            for item in items:
                if item["threshold_k"] is None:
                    continue
                counts[int(item["threshold_k"])] += 1
            thresholds = sorted(counts.keys())
            tickers_by_k: dict[str, str] = {}
            for item in items:
                k = item["threshold_k"]
                if k is None:
                    continue
                key_k = str(int(k))
                if key_k not in tickers_by_k:
                    tickers_by_k[key_k] = item["market_ticker"]
            flags: list[str] = []
            if any(count > 1 for count in counts.values()):
                flags.append("duplicate_thresholds")
            gaps = [b - a for a, b in zip(thresholds, thresholds[1:]) if b - a > 1]
            if gaps:
                flags.append("missing_intermediate_thresholds")
            home_flags = {item["team_is_home"] for item in items}
            if len(home_flags) > 1:
                flags.append("inconsistent_team_home_flag")
            conn.execute(
                """
                INSERT OR REPLACE INTO event_ladders (
                    event_id, ladder_family, team_code, thresholds_json, tickers_by_k_json, quality_flags_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    family,
                    team_code,
                    json.dumps(thresholds, separators=(",", ":")),
                    json.dumps(tickers_by_k, separators=(",", ":")),
                    json.dumps(flags, separators=(",", ":")),
                ),
            )
            ladders_built += 1
        conn.commit()
    print(f"[phaseA] Built {ladders_built} ladders")
    return ladders_built


def ingest_results(db_path: str, results_path: str) -> int:
    if not os.path.exists(results_path):
        raise FileNotFoundError(results_path)
    _ensure_db(db_path)
    updated = 0
    with sqlite3.connect(db_path) as conn, open(results_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            event_id = row.get("event_id") or row.get("event_ticker")
            if not event_id:
                continue
            home_score = row.get("home_score") or row.get("final_home_score")
            away_score = row.get("away_score") or row.get("final_away_score")
            final_margin = row.get("final_margin")
            try:
                home_val = int(home_score) if home_score not in (None, "") else None
            except ValueError:
                home_val = None
            try:
                away_val = int(away_score) if away_score not in (None, "") else None
            except ValueError:
                away_val = None
            try:
                margin_val = int(final_margin) if final_margin not in (None, "") else None
            except ValueError:
                margin_val = None
            if margin_val is None and home_val is not None and away_val is not None:
                margin_val = home_val - away_val
            conn.execute(
                """
                UPDATE events
                SET final_margin = ?, final_home_score = ?, final_away_score = ?
                WHERE event_id = ?
                """,
                (margin_val, home_val, away_val, event_id),
            )
            updated += 1
        conn.commit()
    print(f"[phaseA] Updated {updated} events with results from {results_path}")
    return updated


def compute_labels(db_path: str) -> int:
    _ensure_db(db_path)
    inserted = 0
    now_iso = _utc_now_iso()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT m.market_ticker, m.event_id, m.threshold_k, m.team_is_home,
                   e.final_home_score, e.final_away_score
            FROM markets m
            JOIN events e ON e.event_id = m.event_id
            WHERE m.threshold_k IS NOT NULL
              AND m.team_is_home IS NOT NULL
              AND e.final_home_score IS NOT NULL
              AND e.final_away_score IS NOT NULL
            """
        ).fetchall()
        for row in rows:
            if row["team_is_home"] == 1:
                margin = int(row["final_home_score"]) - int(row["final_away_score"])
            else:
                margin = int(row["final_away_score"]) - int(row["final_home_score"])
            outcome = 1 if margin >= int(row["threshold_k"]) else 0
            conn.execute(
                """
                INSERT OR REPLACE INTO market_outcomes (
                    market_ticker, event_id, final_margin, outcome_yes, outcome_rule, computed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["market_ticker"],
                    row["event_id"],
                    margin,
                    outcome,
                    "margin_team_ge_k",
                    now_iso,
                ),
            )
            inserted += 1
        conn.commit()
    print(f"[phaseA] Computed {inserted} market outcomes")
    return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NBA Phase A pipeline (data + ladder index)")
    parser.add_argument("--db", default=_default_db_path(), help="Path to phaseA sqlite DB")
    sub = parser.add_subparsers(dest="cmd", required=True)

    init_db_cmd = sub.add_parser("init-db", help="Initialize the Phase A database")
    init_db_cmd.add_argument("--db", default=_default_db_path(), help="Path to phaseA sqlite DB")

    ingest = sub.add_parser("ingest-markets", help="Ingest market metadata")
    ingest.add_argument("--db", default=_default_db_path(), help="Path to phaseA sqlite DB")
    ingest.add_argument("--enriched", default=_default_enriched_path(), help="Path to enriched_markets.csv")
    ingest.add_argument("--metadata", default=_default_metadata_path(), help="Path to market_metadata.jsonl")
    ingest.add_argument("--skip-enriched", action="store_true", help="Skip enriched_markets.csv ingest")
    ingest.add_argument("--skip-metadata", action="store_true", help="Skip market_metadata.jsonl ingest")

    l2 = sub.add_parser("ingest-l2", help="Ingest L2 checkpoints and trades")
    l2.add_argument("--db", default=_default_db_path(), help="Path to phaseA sqlite DB")
    l2.add_argument("--l2", default=_default_l2_path(), help="Path to l2_orderbook.sqlite")

    l1 = sub.add_parser("ingest-l1", help="Ingest L1 snapshot jsonl")
    l1.add_argument("--db", default=_default_db_path(), help="Path to phaseA sqlite DB")
    l1.add_argument("--l1", default=_default_l1_path(), help="Path to l1_snapshots.jsonl")

    build = sub.add_parser("build-ladders", help="Build ladder index from markets table")
    build.add_argument("--db", default=_default_db_path(), help="Path to phaseA sqlite DB")

    results = sub.add_parser("ingest-results", help="Ingest settled game results")
    results.add_argument("--db", default=_default_db_path(), help="Path to phaseA sqlite DB")
    results.add_argument("--results", default=_default_results_path(), help="Path to nba_results.csv")
    results.add_argument("--compute-labels", action="store_true", help="Compute market outcomes")

    labels = sub.add_parser("compute-labels", help="Compute market outcomes from results")
    labels.add_argument("--db", default=_default_db_path(), help="Path to phaseA sqlite DB")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "init-db":
        init_db(args.db)
        return 0
    if args.cmd == "ingest-markets":
        if not args.skip_enriched:
            ingest_enriched_markets(args.db, args.enriched)
        if not args.skip_metadata:
            if os.path.exists(args.metadata):
                ingest_metadata_jsonl(args.db, args.metadata)
            else:
                print(f"[phaseA] Metadata file not found (skipping): {args.metadata}")
        return 0
    if args.cmd == "ingest-l2":
        ingest_l2(args.db, args.l2)
        return 0
    if args.cmd == "ingest-l1":
        ingest_l1(args.db, args.l1)
        return 0
    if args.cmd == "build-ladders":
        build_ladders(args.db)
        return 0
    if args.cmd == "ingest-results":
        ingest_results(args.db, args.results)
        if args.compute_labels:
            compute_labels(args.db)
        return 0
    if args.cmd == "compute-labels":
        compute_labels(args.db)
        return 0
    raise RuntimeError(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
