"""Phase 5 arbitrage evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from cross_venue_arb.arb.opportunity import Leg, Opportunity
from cross_venue_arb.books.manager import BookManager
from cross_venue_arb.storage.mapping_registry import GameMappingRecord


@dataclass(frozen=True)
class Phase5Config:
    min_size: float = 10.0
    min_edge: float = 0.010
    buffer_per_contract: float = 0.005
    kalshi_fee_per_contract: float = 0.001
    polymarket_fee_per_contract: float = 0.0
    fresh_s: float = 0.25
    sync_s: float = 0.05
    emit_cooldown_s: float = 10.0
    confirm_ticks: int = 2
    fresh_ms: int = 1500
    sync_ms: int = 400


class OpportunityTracker:
    def __init__(self, cooldown_s: float) -> None:
        self._cooldown_s = cooldown_s
        self._last_emit: dict[str, float] = {}
        self._streaks: dict[str, int] = {}

    def should_emit(self, key: str, edge: float) -> bool:
        now = time.monotonic()
        last = self._last_emit.get(key)
        if last is None or now - last >= self._cooldown_s:
            self._last_emit[key] = now
            return True
        return False

    def track_streak(self, key: str, passed: bool, confirm_ticks: int) -> bool:
        if not passed:
            self._streaks[key] = 0
            return False
        current = self._streaks.get(key, 0) + 1
        self._streaks[key] = current
        return current >= confirm_ticks


def evaluate_game(
    record: GameMappingRecord,
    manager: BookManager,
    config: Phase5Config,
) -> list[Opportunity]:
    ts = time.monotonic()
    team_map = _kalshi_team_map(record.kalshi_team_markets)
    if len(team_map) < 2:
        return [Opportunity(record.game_key, ts, "invalid", 0, 0, 0, [], "missing_teams")]
    gate_reason = _sync_gate(record, manager, config, now_ts=ts)
    if gate_reason:
        return [Opportunity(record.game_key, ts, "invalid", 0, 0, 0, [], gate_reason)]

    outcomes = list(team_map.keys())[:2]
    team0, team1 = outcomes[0], outcomes[1]

    poly_outcome_map = record.match_details.get("polymarket_outcome_map", {})
    poly_asset_team0 = poly_outcome_map.get(team0)
    poly_asset_team1 = poly_outcome_map.get(team1)

    opportunities: list[Opportunity] = []
    for team_a, team_b, poly_asset in [
        (team0, team1, poly_asset_team0),
        (team1, team0, poly_asset_team1),
    ]:
        opp = _evaluate_pair(
            record,
            manager,
            config,
            team_a,
            team_b,
            poly_asset,
        )
        opportunities.append(opp)
    return opportunities


def _evaluate_pair(
    record: GameMappingRecord,
    manager: BookManager,
    config: Phase5Config,
    team_a: str,
    team_b: str,
    poly_asset: str | None,
) -> Opportunity:
    ts = time.monotonic()
    kalshi_a = _kalshi_team_map(record.kalshi_team_markets).get(team_a)
    kalshi_b = _kalshi_team_map(record.kalshi_team_markets).get(team_b)
    if kalshi_a is None or kalshi_b is None:
        return Opportunity(record.game_key, ts, "invalid", 0, 0, 0, [], "missing_kalshi_ticker")

    if not poly_asset:
        return Opportunity(record.game_key, ts, "invalid", 0, 0, 0, [], "missing_polymarket_asset")

    if not manager.is_healthy("kalshi", kalshi_a) or not manager.is_healthy("kalshi", kalshi_b):
        return Opportunity(record.game_key, ts, "invalid", 0, 0, 0, [], "stale_kalshi_book")
    if not manager.is_healthy("polymarket", poly_asset):
        return Opportunity(record.game_key, ts, "invalid", 0, 0, 0, [], "stale_polymarket_book")

    book_a = manager.get_book("kalshi", kalshi_a)
    book_b = manager.get_book("kalshi", kalshi_b)
    book_poly = manager.get_book("polymarket", poly_asset)
    if not book_a or not book_b or not book_poly:
        return Opportunity(record.game_key, ts, "invalid", 0, 0, 0, [], "missing_book")

    if not _books_synced(book_poly, book_b, ts, config):
        return Opportunity(record.game_key, ts, "invalid", 0, 0, 0, [], "unsynced_books")

    ask_a = book_a.best_ask[0] if book_a.best_ask else None
    ask_b = book_b.best_ask[0] if book_b.best_ask else None
    ask_poly = book_poly.best_ask[0] if book_poly.best_ask else None

    size_a = book_a.best_ask_size()
    size_b = book_b.best_ask_size()
    size_poly = book_poly.best_ask_size()

    if ask_a is None or ask_b is None or ask_poly is None:
        return Opportunity(record.game_key, ts, "invalid", 0, 0, 0, [], "missing_ask")

    size_max = min(size_a or 0, size_b or 0, size_poly or 0)
    if size_max < config.min_size:
        return Opportunity(record.game_key, ts, "invalid", size_max, 0, 0, [], "size_below_min")

    fee_poly = _estimate_fee("polymarket", config.polymarket_fee_per_contract)
    fee_kalshi = _estimate_fee("kalshi", config.kalshi_fee_per_contract)
    edge_per_contract = 1.0 - (ask_poly + ask_b) - fee_poly - fee_kalshi - config.buffer_per_contract

    expected_profit_max = edge_per_contract * size_max
    direction = f"POLY_{team_a}_KALSHI_{team_b}"

    legs = [
        Leg(
            venue="polymarket",
            market_id=poly_asset,
            action="BUY",
            outcome=team_a,
            limit_price=ask_poly,
            available_size=float(size_poly or 0),
        ),
        Leg(
            venue="kalshi",
            market_id=kalshi_b,
            action="BUY",
            outcome=team_b,
            limit_price=ask_b,
            available_size=float(size_b or 0),
        ),
    ]

    if edge_per_contract < config.min_edge:
        return Opportunity(
            record.game_key,
            ts,
            direction,
            size_max,
            edge_per_contract,
            expected_profit_max,
            legs,
            "edge_below_min",
        )

    return Opportunity(
        record.game_key,
        ts,
        direction,
        size_max,
        edge_per_contract,
        expected_profit_max,
        legs,
    )


def _kalshi_team_map(team_markets: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in team_markets:
        team = entry.get("team_norm")
        ticker = entry.get("ticker")
        if team and ticker:
            mapping[team] = ticker
    return mapping


def _polymarket_assets(record: GameMappingRecord) -> list[str]:
    details = record.match_details or {}
    asset_ids = details.get("polymarket_asset_ids") or []
    assets = [asset_id for asset_id in asset_ids if asset_id]
    if assets:
        return assets
    outcome_map = details.get("polymarket_outcome_map") or {}
    return [asset_id for asset_id in outcome_map.values() if asset_id]


def _sync_gate(
    record: GameMappingRecord,
    manager: BookManager,
    config: Phase5Config,
    *,
    now_ts: float,
) -> str | None:
    kalshi_tickers = [entry.get("ticker") for entry in record.kalshi_team_markets if entry.get("ticker")]
    polymarket_assets = _polymarket_assets(record)
    if not kalshi_tickers or not polymarket_assets:
        return None
    kalshi_ts = manager.latest_update_ts("kalshi", kalshi_tickers)
    poly_ts = manager.latest_update_ts("polymarket", polymarket_assets)
    if kalshi_ts is None or poly_ts is None:
        return "sync_missing_ts"
    fresh_s = config.fresh_ms / 1000.0
    if now_ts - kalshi_ts > fresh_s:
        return "stale_kalshi_leg"
    if now_ts - poly_ts > fresh_s:
        return "stale_polymarket_leg"
    if abs(kalshi_ts - poly_ts) * 1000.0 > config.sync_ms:
        return "sync_window"
    return None


def _estimate_fee(venue: str, fee_per_contract: float) -> float:
    _ = venue
    return fee_per_contract


def _books_synced(book_a: object, book_b: object, now: float, config: Phase5Config) -> bool:
    ts_a = getattr(book_a, "last_update_ts", None)
    ts_b = getattr(book_b, "last_update_ts", None)
    if ts_a is None or ts_b is None:
        return False
    if now - ts_a > config.fresh_s:
        return False
    if now - ts_b > config.fresh_s:
        return False
    if abs(ts_a - ts_b) > config.sync_s:
        return False
    return True
