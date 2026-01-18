"""Persist deterministic match mappings."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from cross_venue_arb.storage.sqlite import connect


@dataclass(frozen=True)
class MappingRecord:
    game_key: str
    kalshi_ticker: str
    polymarket_event_id: str | None
    polymarket_market_id: str | None
    match_confidence: float
    match_method: str
    match_details: dict
    created_at: str
    as_of_date: str


@dataclass(frozen=True)
class GameMappingRecord:
    game_key: str
    kalshi_event_ticker: str
    kalshi_team_markets: list[dict]
    team_a_norm: str | None
    team_b_norm: str | None
    polymarket_event_id: str | None
    polymarket_market_id: str | None
    match_confidence: float
    match_method: str
    match_details: dict
    created_at: str
    as_of_date: str


def write_mappings(records: list[MappingRecord]) -> None:
    if not records:
        return
    conn = connect()
    _ensure_schema(conn)
    rows = []
    for record in records:
        rows.append(
            (
                record.game_key,
                record.kalshi_ticker,
                record.polymarket_event_id,
                record.polymarket_market_id,
                record.match_confidence,
                record.match_method,
                json.dumps(record.match_details, ensure_ascii=True),
                record.created_at,
                record.as_of_date,
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO match_registry (
            game_key,
            kalshi_ticker,
            polymarket_event_id,
            polymarket_market_id,
            match_confidence,
            match_method,
            match_details,
            created_at,
            as_of_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def write_game_mappings(records: list[GameMappingRecord]) -> None:
    if not records:
        return
    conn = connect()
    _ensure_game_schema(conn)
    rows = []
    for record in records:
        rows.append(
            (
                record.game_key,
                record.kalshi_event_ticker,
                json.dumps(record.kalshi_team_markets, ensure_ascii=True),
                record.team_a_norm,
                record.team_b_norm,
                record.polymarket_event_id,
                record.polymarket_market_id,
                record.match_confidence,
                record.match_method,
                json.dumps(record.match_details, ensure_ascii=True),
                record.created_at,
                record.as_of_date,
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO match_registry_games (
            game_key,
            kalshi_event_ticker,
            kalshi_team_markets,
            team_a_norm,
            team_b_norm,
            polymarket_event_id,
            polymarket_market_id,
            match_confidence,
            match_method,
            match_details,
            created_at,
            as_of_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def read_game_mappings(as_of_date: str | None = None) -> list[GameMappingRecord]:
    conn = connect()
    _ensure_game_schema(conn)
    if as_of_date is None:
        row = conn.execute(
            "SELECT MAX(as_of_date) FROM match_registry_games"
        ).fetchone()
        as_of_date = row[0] if row and row[0] else None
    if as_of_date is None:
        conn.close()
        return []
    rows = conn.execute(
        """
        SELECT game_key,
               kalshi_event_ticker,
               kalshi_team_markets,
               team_a_norm,
               team_b_norm,
               polymarket_event_id,
               polymarket_market_id,
               match_confidence,
               match_method,
               match_details,
               created_at,
               as_of_date
        FROM match_registry_games
        WHERE as_of_date = ?
        """,
        (as_of_date,),
    ).fetchall()
    conn.close()
    records: list[GameMappingRecord] = []
    for row in rows:
        records.append(
            GameMappingRecord(
                game_key=row[0],
                kalshi_event_ticker=row[1],
                kalshi_team_markets=json.loads(row[2]) if row[2] else [],
                team_a_norm=row[3],
                team_b_norm=row[4],
                polymarket_event_id=row[5],
                polymarket_market_id=row[6],
                match_confidence=float(row[7]),
                match_method=row[8],
                match_details=json.loads(row[9]) if row[9] else {},
                created_at=row[10],
                as_of_date=row[11],
            )
        )
    return records


def new_record(
    *,
    game_key: str,
    kalshi_ticker: str,
    polymarket_event_id: str | None,
    polymarket_market_id: str | None,
    match_confidence: float,
    match_method: str,
    match_details: dict,
    as_of_date: str | None = None,
) -> MappingRecord:
    created_at = datetime.now(timezone.utc).isoformat()
    if not as_of_date:
        as_of_date = datetime.now(timezone.utc).date().isoformat()
    return MappingRecord(
        game_key=game_key,
        kalshi_ticker=kalshi_ticker,
        polymarket_event_id=polymarket_event_id,
        polymarket_market_id=polymarket_market_id,
        match_confidence=match_confidence,
        match_method=match_method,
        match_details=match_details,
        created_at=created_at,
        as_of_date=as_of_date,
    )


def new_game_record(
    *,
    game_key: str,
    kalshi_event_ticker: str,
    kalshi_team_markets: list[dict],
    team_a_norm: str | None,
    team_b_norm: str | None,
    polymarket_event_id: str | None,
    polymarket_market_id: str | None,
    match_confidence: float,
    match_method: str,
    match_details: dict,
    as_of_date: str | None = None,
) -> GameMappingRecord:
    created_at = datetime.now(timezone.utc).isoformat()
    if not as_of_date:
        as_of_date = datetime.now(timezone.utc).date().isoformat()
    return GameMappingRecord(
        game_key=game_key,
        kalshi_event_ticker=kalshi_event_ticker,
        kalshi_team_markets=kalshi_team_markets,
        team_a_norm=team_a_norm,
        team_b_norm=team_b_norm,
        polymarket_event_id=polymarket_event_id,
        polymarket_market_id=polymarket_market_id,
        match_confidence=match_confidence,
        match_method=match_method,
        match_details=match_details,
        created_at=created_at,
        as_of_date=as_of_date,
    )


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS match_registry (
            game_key TEXT NOT NULL,
            kalshi_ticker TEXT NOT NULL,
            polymarket_event_id TEXT,
            polymarket_market_id TEXT,
            match_confidence REAL NOT NULL,
            match_method TEXT NOT NULL,
            match_details TEXT NOT NULL,
            created_at TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            PRIMARY KEY (kalshi_ticker, as_of_date)
        )
        """
    )


def _ensure_game_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS match_registry_games (
            game_key TEXT NOT NULL,
            kalshi_event_ticker TEXT NOT NULL,
            kalshi_team_markets TEXT NOT NULL,
            team_a_norm TEXT,
            team_b_norm TEXT,
            polymarket_event_id TEXT,
            polymarket_market_id TEXT,
            match_confidence REAL NOT NULL,
            match_method TEXT NOT NULL,
            match_details TEXT NOT NULL,
            created_at TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            PRIMARY KEY (kalshi_event_ticker, as_of_date)
        )
        """
    )
