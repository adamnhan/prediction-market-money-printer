from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None and str(Path(__file__).parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[2]))

from cross_venue_arb.matching.deterministic import (
    GameGroup,
    MatchGroupBuckets,
    MatchGroupResult,
    candidate_breakdown_group,
    match_groups,
)
from cross_venue_arb.matching.normalize import format_normalized_game, normalize_game, normalize_team
from cross_venue_arb.storage.mapping_registry import new_game_record, write_game_mappings


def load_games(path: Path) -> tuple[list[Any], list[Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    kalshi_games: list[Any] = []
    polymarket_games: list[Any] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        venue = str(entry.get("venue") or "")
        raw = entry.get("raw")
        if not isinstance(raw, dict):
            raw = entry
        normalized = normalize_game(raw, venue=venue or None)
        if normalized.venue == "kalshi":
            kalshi_games.append(normalized)
        elif normalized.venue == "polymarket":
            polymarket_games.append(normalized)
    return kalshi_games, polymarket_games


def group_games(kalshi_games: list[Any], polymarket_games: list[Any]) -> tuple[list[GameGroup], list[GameGroup]]:
    kalshi_groups: dict[str, GameGroup] = {}
    for game in kalshi_games:
        raw = game.raw or {}
        event_ticker = raw.get("event_ticker") or raw.get("eventTicker")
        group_id = str(event_ticker) if event_ticker else game.venue_id
        group = kalshi_groups.get(group_id)
        if group:
            members = group.members + [game]
            kalshi_groups[group_id] = GameGroup(
                group_id=group_id,
                venue="kalshi",
                representative=group.representative,
                members=members,
            )
        else:
            kalshi_groups[group_id] = GameGroup(
                group_id=group_id,
                venue="kalshi",
                representative=game,
                members=[game],
            )

    polymarket_groups: dict[str, GameGroup] = {}
    for game in polymarket_games:
        raw = game.raw or {}
        event_id = raw.get("event_id") or raw.get("eventId")
        condition_id = raw.get("conditionId")
        group_id = str(event_id or condition_id or game.venue_id)
        group = polymarket_groups.get(group_id)
        if group:
            members = group.members + [game]
            polymarket_groups[group_id] = GameGroup(
                group_id=group_id,
                venue="polymarket",
                representative=group.representative,
                members=members,
            )
        else:
            polymarket_groups[group_id] = GameGroup(
                group_id=group_id,
                venue="polymarket",
                representative=game,
                members=[game],
            )

    return list(kalshi_groups.values()), list(polymarket_groups.values())


def print_summary(buckets: MatchGroupBuckets) -> None:
    print(f"matched: {len(buckets.matched)}")
    print(f"ambiguous: {len(buckets.ambiguous)}")
    print(f"unmatched_kalshi: {len(buckets.unmatched_kalshi)}")
    print(f"unmatched_polymarket: {len(buckets.unmatched_polymarket)}")


def print_mappings(matches: list[MatchGroupResult], limit: int | None) -> None:
    if limit is not None:
        matches = matches[:limit]
    for match in matches:
        time_label = _time_label(match.kalshi.representative, match.polymarket.representative)
        matchup_label = _matchup_label(match.kalshi.representative, match.polymarket.representative)
        print(f"{matchup_label} {time_label}")
        print(
            f"  Kalshi: {match.kalshi.group_id} ({match.kalshi.representative.title})"
        )
        print(
            f"  Poly:   {_poly_label(match.polymarket.representative)} "
            f"({match.polymarket.representative.title})"
        )
        print(
            f"  score={match.score:.1f} confidence={match.confidence:.2f} method={match.method}"
        )


def build_registry_records(matches: list[MatchGroupResult]) -> list[Any]:
    records = []
    as_of_date = datetime.now(timezone.utc).date().isoformat()
    for match in matches:
        event_id, market_id = _polymarket_ids(match.polymarket.representative)
        asset_ids, outcomes, outcome_map = _polymarket_assets(match.polymarket.representative)
        team_markets, team_norms = _kalshi_team_markets(match.kalshi.members)
        match_details = dict(match.details)
        if asset_ids:
            match_details["polymarket_asset_ids"] = asset_ids
        if outcomes:
            match_details["polymarket_outcomes"] = outcomes
        if outcome_map:
            match_details["polymarket_outcome_map"] = outcome_map
        records.append(
            new_game_record(
                game_key=match.kalshi.representative.game_key,
                kalshi_event_ticker=match.kalshi.group_id,
                kalshi_team_markets=team_markets,
                team_a_norm=team_norms[0],
                team_b_norm=team_norms[1],
                polymarket_event_id=event_id,
                polymarket_market_id=market_id,
                match_confidence=match.confidence,
                match_method=match.method,
                match_details=match_details,
                as_of_date=as_of_date,
            )
        )
    return records


def _polymarket_ids(game: Any) -> tuple[str | None, str | None]:
    raw = game.raw or {}
    event_id = raw.get("event_id") or raw.get("eventId") or raw.get("conditionId")
    market_id = raw.get("market_id") or raw.get("marketId") or raw.get("id")
    event_id = str(event_id) if event_id is not None else None
    market_id = str(market_id) if market_id is not None else None
    return event_id, market_id


def _polymarket_assets(game: Any) -> tuple[list[str], list[str], dict[str, str]]:
    raw = game.raw or {}
    clob_tokens = raw.get("clobTokenIds") or raw.get("clob_token_ids")
    asset_ids: list[str] = []
    if isinstance(clob_tokens, str):
        try:
            clob_tokens = json.loads(clob_tokens)
        except json.JSONDecodeError:
            clob_tokens = None
    if isinstance(clob_tokens, list):
        asset_ids = [str(token) for token in clob_tokens if token]
    outcomes_raw = raw.get("outcomes")
    outcomes: list[str] = []
    if isinstance(outcomes_raw, str):
        try:
            outcomes_raw = json.loads(outcomes_raw)
        except json.JSONDecodeError:
            outcomes_raw = None
    if isinstance(outcomes_raw, list):
        outcomes = [str(item) for item in outcomes_raw if item]
    outcome_map: dict[str, str] = {}
    for idx, outcome in enumerate(outcomes):
        if idx < len(asset_ids):
            outcome_map[normalize_team(outcome)] = asset_ids[idx]
    return asset_ids, outcomes, outcome_map


def _poly_label(game: Any) -> str:
    event_id, market_id = _polymarket_ids(game)
    if event_id and market_id and event_id != market_id:
        return f"{event_id} ({market_id})"
    return event_id or market_id or "unknown"


def _time_label(kalshi: Any, polymarket: Any) -> str:
    ref_time = kalshi.start_time_utc or polymarket.start_time_utc
    if not ref_time:
        return "unknown"
    return ref_time.strftime("%Y-%m-%d %H:%MZ")


def _matchup_label(kalshi: Any, polymarket: Any) -> str:
    if kalshi.matchup_key:
        return kalshi.matchup_key.replace("|", " @ ")
    return format_normalized_game(kalshi).split("|")[2].strip()


def _print_titles(
    kalshi_games: list[Any],
    polymarket_games: list[Any],
    limit: int,
) -> None:
    print("Kalshi titles:")
    for game in kalshi_games[:limit]:
        print(f"- {game.title}")
    print("Polymarket titles:")
    for game in polymarket_games[:limit]:
        print(f"- {game.title}")


def _print_debug_stats(kalshi_games: list[Any], polymarket_games: list[Any]) -> None:
    def summarize(games: list[Any], label: str) -> None:
        total = len(games)
        with_matchup = sum(1 for g in games if g.matchup_key)
        with_time = sum(1 for g in games if g.start_time_utc)
        with_both = sum(1 for g in games if g.matchup_key and g.start_time_utc)
        print(
            f"{label}: total={total} matchup_key={with_matchup} "
            f"start_time={with_time} both={with_both}"
        )

    summarize(kalshi_games, "kalshi")
    summarize(polymarket_games, "polymarket")


def _print_unmatched_diagnostics(
    kalshi_games: list[Any],
    polymarket_games: list[Any],
    limit: int,
) -> None:
    polymarket_by_matchup: dict[str, list[Any]] = {}
    for group in polymarket_games:
        matchup_key = group.representative.matchup_key
        if matchup_key:
            polymarket_by_matchup.setdefault(matchup_key, []).append(group)
    print("Unmatched diagnostics:")
    for group in kalshi_games[:limit]:
        details = candidate_breakdown_group(group, polymarket_games, polymarket_by_matchup)
        print(
            f"- {group.representative.title} | matchup_key={details['matchup_key']} "
            f"matchup_candidates={details['matchup_candidates']} "
            f"matchup_after_time={details['matchup_after_time']} "
            f"fallback_candidates={details['fallback_candidates']} "
            f"fallback_after_time={details['fallback_after_time']}"
        )


def _print_ambiguous_diagnostics(ambiguous: list[Any], limit: int) -> None:
    print("Ambiguous diagnostics:")
    for entry in ambiguous[:limit]:
        kalshi = entry.kalshi
        print(f"- {kalshi.representative.title} ({kalshi.group_id})")
        for candidate in entry.candidates[:2]:
            time_reason = candidate.details.get("time_reason")
            time_delta = candidate.details.get("time_delta_min")
            print(
                f"  poly={_poly_label(candidate.polymarket.representative)} "
                f"score={candidate.score:.1f} delta_min={time_delta} time_reason={time_reason}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 deterministic matching")
    parser.add_argument(
        "--input",
        default="cross_venue_arb/data/phase1_markets.json",
        help="Path to phase1 markets JSON",
    )
    parser.add_argument("--print-limit", type=int, default=20, help="Print top N matches")
    parser.add_argument("--print-all", action="store_true", help="Print all matches")
    parser.add_argument(
        "--print-titles",
        type=int,
        default=0,
        help="Print N raw titles per venue for debugging",
    )
    parser.add_argument(
        "--debug-stats",
        action="store_true",
        help="Print matchup/time diagnostics per venue",
    )
    parser.add_argument(
        "--debug-unmatched",
        type=int,
        default=0,
        help="Print diagnostics for N unmatched Kalshi games",
    )
    parser.add_argument(
        "--debug-ambiguous",
        type=int,
        default=0,
        help="Print diagnostics for N ambiguous Kalshi games",
    )
    parser.add_argument("--write-db", action="store_true", help="Write mappings to SQLite")
    args = parser.parse_args()

    kalshi_games, polymarket_games = load_games(Path(args.input))
    kalshi_groups, polymarket_groups = group_games(kalshi_games, polymarket_games)
    if args.print_titles:
        _print_titles(kalshi_games, polymarket_games, args.print_titles)
    if args.debug_stats:
        _print_debug_stats(kalshi_games, polymarket_games)
    buckets = match_groups(kalshi_groups, polymarket_groups)
    print_summary(buckets)

    limit = None if args.print_all else args.print_limit
    print_mappings(buckets.matched, limit)
    if args.debug_unmatched:
        _print_unmatched_diagnostics(
            buckets.unmatched_kalshi, polymarket_groups, args.debug_unmatched
        )
    if args.debug_ambiguous:
        _print_ambiguous_diagnostics(buckets.ambiguous, args.debug_ambiguous)

    if args.write_db:
        records = build_registry_records(buckets.matched)
        write_game_mappings(records)
        print(f"wrote {len(records)} mappings to registry")


def _kalshi_team_markets(kalshi_games: list[Any]) -> tuple[list[dict], list[str | None]]:
    team_markets: list[dict] = []
    team_norms: list[str] = []
    for game in kalshi_games:
        raw = game.raw or {}
        yes_team = raw.get("yes_sub_title") or raw.get("no_sub_title")
        team_norm = normalize_team(str(yes_team)) if yes_team else None
        team_markets.append({"ticker": game.venue_id, "team_norm": team_norm})
        if team_norm and team_norm not in team_norms:
            team_norms.append(team_norm)
    if not team_norms:
        if game.away_team_norm:
            team_norms.append(game.away_team_norm)
        if game.home_team_norm:
            team_norms.append(game.home_team_norm)
    if len(team_norms) == 1:
        team_norms.append(None)
    while len(team_norms) < 2:
        team_norms.append(None)
    return team_markets, team_norms[:2]


if __name__ == "__main__":
    main()
