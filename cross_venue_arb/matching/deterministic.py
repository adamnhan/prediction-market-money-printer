"""Deterministic matching logic for markets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from cross_venue_arb.matching.normalize import NormalizedGame


@dataclass(frozen=True)
class MatchResult:
    kalshi: NormalizedGame
    polymarket: NormalizedGame
    score: float
    confidence: float
    method: str
    details: dict


@dataclass(frozen=True)
class AmbiguousMatch:
    kalshi: NormalizedGame
    candidates: list[MatchResult]


@dataclass(frozen=True)
class MatchBuckets:
    matched: list[MatchResult]
    ambiguous: list[AmbiguousMatch]
    unmatched_kalshi: list[NormalizedGame]
    unmatched_polymarket: list[NormalizedGame]


@dataclass(frozen=True)
class GameGroup:
    group_id: str
    venue: str
    representative: NormalizedGame
    members: list[NormalizedGame]


@dataclass(frozen=True)
class MatchGroupResult:
    kalshi: GameGroup
    polymarket: GameGroup
    score: float
    confidence: float
    method: str
    details: dict


@dataclass(frozen=True)
class AmbiguousGroupMatch:
    kalshi: GameGroup
    candidates: list[MatchGroupResult]


@dataclass(frozen=True)
class MatchGroupBuckets:
    matched: list[MatchGroupResult]
    ambiguous: list[AmbiguousGroupMatch]
    unmatched_kalshi: list[GameGroup]
    unmatched_polymarket: list[GameGroup]


def match_games(
    kalshi_games: list[NormalizedGame],
    polymarket_games: list[NormalizedGame],
) -> MatchBuckets:
    polymarket_by_matchup: dict[str, list[NormalizedGame]] = {}
    for game in polymarket_games:
        if game.matchup_key:
            polymarket_by_matchup.setdefault(game.matchup_key, []).append(game)

    used_polymarket: set[str] = set()
    matched: list[MatchResult] = []
    ambiguous: list[AmbiguousMatch] = []
    unmatched_kalshi: list[NormalizedGame] = []

    for kalshi in kalshi_games:
        if _teams_missing(kalshi):
            unmatched_kalshi.append(kalshi)
            continue

        candidates = _candidate_games(kalshi, polymarket_games, polymarket_by_matchup)
        scored = _score_candidates(kalshi, candidates)
        scored = [c for c in scored if c.polymarket.game_key not in used_polymarket]

        if not scored:
            unmatched_kalshi.append(kalshi)
            continue

        scored.sort(key=lambda c: c.score, reverse=True)
        top = scored[0]
        second = scored[1] if len(scored) > 1 else None

        if _is_confident_match(top, second):
            matched.append(top)
            used_polymarket.add(top.polymarket.game_key)
        else:
            ambiguous.append(AmbiguousMatch(kalshi=kalshi, candidates=scored[:5]))

    unmatched_polymarket = [
        game for game in polymarket_games if game.game_key not in used_polymarket
    ]

    return MatchBuckets(
        matched=matched,
        ambiguous=ambiguous,
        unmatched_kalshi=unmatched_kalshi,
        unmatched_polymarket=unmatched_polymarket,
    )


def match_groups(
    kalshi_groups: list[GameGroup],
    polymarket_groups: list[GameGroup],
) -> MatchGroupBuckets:
    polymarket_by_matchup: dict[str, list[GameGroup]] = {}
    for group in polymarket_groups:
        matchup_key = group.representative.matchup_key
        if matchup_key:
            polymarket_by_matchup.setdefault(matchup_key, []).append(group)

    used_polymarket: set[str] = set()
    matched: list[MatchGroupResult] = []
    ambiguous: list[AmbiguousGroupMatch] = []
    unmatched_kalshi: list[GameGroup] = []

    for kalshi in kalshi_groups:
        if _teams_missing(kalshi.representative):
            unmatched_kalshi.append(kalshi)
            continue

        candidates = _candidate_groups(kalshi, polymarket_groups, polymarket_by_matchup)
        scored = _score_group_candidates(kalshi, candidates)
        scored = [c for c in scored if c.polymarket.group_id not in used_polymarket]

        if not scored:
            unmatched_kalshi.append(kalshi)
            continue

        scored.sort(key=lambda c: c.score, reverse=True)
        top = scored[0]
        second = scored[1] if len(scored) > 1 else None

        if _is_confident_group_match(top, second):
            matched.append(top)
            used_polymarket.add(top.polymarket.group_id)
        else:
            ambiguous.append(AmbiguousGroupMatch(kalshi=kalshi, candidates=scored[:5]))

    unmatched_polymarket = [
        group for group in polymarket_groups if group.group_id not in used_polymarket
    ]

    return MatchGroupBuckets(
        matched=matched,
        ambiguous=ambiguous,
        unmatched_kalshi=unmatched_kalshi,
        unmatched_polymarket=unmatched_polymarket,
    )


def candidate_breakdown(
    kalshi: NormalizedGame,
    polymarket_games: list[NormalizedGame],
    polymarket_by_matchup: dict[str, list[NormalizedGame]],
) -> dict[str, object]:
    details: dict[str, object] = {
        "matchup_key": kalshi.matchup_key,
        "kalshi_time": kalshi.start_time_utc,
        "matchup_candidates": 0,
        "matchup_after_time": 0,
        "fallback_candidates": 0,
        "fallback_after_time": 0,
    }
    if kalshi.matchup_key and kalshi.matchup_key in polymarket_by_matchup:
        matchup_candidates = polymarket_by_matchup[kalshi.matchup_key]
        details["matchup_candidates"] = len(matchup_candidates)
        details["matchup_after_time"] = len(_filter_by_time(kalshi, matchup_candidates))
    fallback_candidates = _fallback_by_title(kalshi, polymarket_games)
    details["fallback_candidates"] = len(fallback_candidates)
    details["fallback_after_time"] = len(_filter_by_time(kalshi, fallback_candidates))
    return details


def candidate_breakdown_group(
    kalshi: GameGroup,
    polymarket_groups: list[GameGroup],
    polymarket_by_matchup: dict[str, list[GameGroup]],
) -> dict[str, object]:
    rep = kalshi.representative
    details: dict[str, object] = {
        "matchup_key": rep.matchup_key,
        "kalshi_time": rep.start_time_utc,
        "matchup_candidates": 0,
        "matchup_after_time": 0,
        "fallback_candidates": 0,
        "fallback_after_time": 0,
    }
    if rep.matchup_key and rep.matchup_key in polymarket_by_matchup:
        matchup_candidates = polymarket_by_matchup[rep.matchup_key]
        details["matchup_candidates"] = len(matchup_candidates)
        details["matchup_after_time"] = len(
            _filter_by_time(rep, [g.representative for g in matchup_candidates])
        )
    fallback_candidates = _fallback_by_title(rep, [g.representative for g in polymarket_groups])
    details["fallback_candidates"] = len(fallback_candidates)
    details["fallback_after_time"] = len(_filter_by_time(rep, fallback_candidates))
    return details


def match_markets(markets: Iterable[dict]) -> list[tuple[dict, dict]]:
    return []


def _candidate_games(
    kalshi: NormalizedGame,
    polymarket_games: list[NormalizedGame],
    polymarket_by_matchup: dict[str, list[NormalizedGame]],
) -> list[NormalizedGame]:
    if kalshi.matchup_key and kalshi.matchup_key in polymarket_by_matchup:
        candidates = polymarket_by_matchup[kalshi.matchup_key]
        if candidates:
            return candidates
    return _fallback_by_title(kalshi, polymarket_games)


def _candidate_groups(
    kalshi: GameGroup,
    polymarket_groups: list[GameGroup],
    polymarket_by_matchup: dict[str, list[GameGroup]],
) -> list[GameGroup]:
    rep = kalshi.representative
    if rep.matchup_key and rep.matchup_key in polymarket_by_matchup:
        candidates = polymarket_by_matchup[rep.matchup_key]
        if candidates:
            return candidates
    return _fallback_group_by_title(kalshi, polymarket_groups)


def _fallback_by_title(
    kalshi: NormalizedGame,
    polymarket_games: list[NormalizedGame],
) -> list[NormalizedGame]:
    kalshi_tokens = _title_tokens(kalshi.title)
    if not kalshi_tokens:
        return []
    candidates: list[NormalizedGame] = []
    for game in polymarket_games:
        if _teams_missing(game):
            continue
        overlap = _token_similarity(kalshi_tokens, _title_tokens(game.title))
        if overlap >= 0.6:
            candidates.append(game)
    return _filter_by_time(kalshi, candidates)


def _fallback_group_by_title(
    kalshi: GameGroup,
    polymarket_groups: list[GameGroup],
) -> list[GameGroup]:
    rep = kalshi.representative
    kalshi_tokens = _title_tokens(rep.title)
    if not kalshi_tokens:
        return []
    candidates: list[GameGroup] = []
    for group in polymarket_groups:
        game = group.representative
        if _teams_missing(game):
            continue
        overlap = _token_similarity(kalshi_tokens, _title_tokens(game.title))
        if overlap >= 0.6:
            candidates.append(group)
    return candidates


def _filter_by_time(
    kalshi: NormalizedGame,
    candidates: list[NormalizedGame],
) -> list[NormalizedGame]:
    if not kalshi.start_time_utc:
        return candidates
    filtered: list[NormalizedGame] = []
    for game in candidates:
        if not game.start_time_utc:
            filtered.append(game)
            continue
        delta = abs((kalshi.start_time_utc - game.start_time_utc).total_seconds()) / 60.0
        if delta <= 120:
            filtered.append(game)
        elif kalshi.start_time_utc.date() == game.start_time_utc.date():
            filtered.append(game)
    return filtered


def _score_candidates(
    kalshi: NormalizedGame,
    candidates: list[NormalizedGame],
) -> list[MatchResult]:
    scored: list[MatchResult] = []
    for game in candidates:
        if _teams_missing(game):
            continue
        details: dict[str, object] = {}
        score = 0.0
        if kalshi.matchup_key and kalshi.matchup_key == game.matchup_key:
            score += 10.0
            details["matchup_key"] = True
        else:
            details["matchup_key"] = False

        time_score, time_reason, time_delta = _score_time(kalshi.start_time_utc, game.start_time_utc)
        score += time_score
        details["time_reason"] = time_reason
        if time_delta is not None:
            details["time_delta_min"] = time_delta

        title_score, title_sim = _score_title(kalshi.title, game.title)
        score += title_score
        if title_sim is not None:
            details["title_similarity"] = title_sim

        confidence = _confidence(score, kalshi, game, time_delta)
        scored.append(
            MatchResult(
                kalshi=kalshi,
                polymarket=game,
                score=score,
                confidence=confidence,
                method="deterministic",
                details=details,
            )
        )
    return scored


def _score_group_candidates(
    kalshi: GameGroup,
    candidates: list[GameGroup],
) -> list[MatchGroupResult]:
    scored: list[MatchGroupResult] = []
    left = kalshi.representative
    for candidate in candidates:
        right = candidate.representative
        if _teams_missing(right):
            continue
        details: dict[str, object] = {}
        score = 0.0
        if left.matchup_key and left.matchup_key == right.matchup_key:
            score += 10.0
            details["matchup_key"] = True
        else:
            details["matchup_key"] = False

        time_score, time_reason, time_delta = _score_time(
            left.start_time_utc, right.start_time_utc
        )
        score += time_score
        details["time_reason"] = time_reason
        if time_delta is not None:
            details["time_delta_min"] = time_delta

        title_score, title_sim = _score_title(left.title, right.title)
        score += title_score
        if title_sim is not None:
            details["title_similarity"] = title_sim

        confidence = _confidence(score, left, right, time_delta)
        scored.append(
            MatchGroupResult(
                kalshi=kalshi,
                polymarket=candidate,
                score=score,
                confidence=confidence,
                method="deterministic",
                details=details,
            )
        )
    return scored


def _score_time(
    left: datetime | None,
    right: datetime | None,
) -> tuple[float, str, float | None]:
    if not left or not right:
        return 0.0, "missing_time", None
    delta = abs((left - right).total_seconds()) / 60.0
    date_delta = abs((left.date() - right.date()).days)
    if delta <= 15:
        return 6.0, "within_15_min", delta
    if delta <= 60:
        return 4.0, "within_60_min", delta
    if delta <= 120:
        return 2.0, "within_120_min", delta
    if left.date() == right.date():
        return 1.0, "same_date", delta
    if date_delta <= 1:
        return 1.0, "date_within_1_day", delta
    return -5.0, "over_120_min", delta


def _score_title(left: str, right: str) -> tuple[float, float | None]:
    left_tokens = _title_tokens(left)
    right_tokens = _title_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0, None
    similarity = _token_similarity(left_tokens, right_tokens)
    if similarity >= 0.75:
        return 1.0, similarity
    return 0.0, similarity


def _token_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _title_tokens(title: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in title)
    return {token for token in cleaned.split() if len(token) > 2}


def _is_confident_match(top: MatchResult, second: MatchResult | None) -> bool:
    if top.score < 10:
        if not _allow_same_date_match(top, second):
            return False
    if second is None:
        return True
    if (top.score - second.score) >= 3:
        return True
    return _allow_same_date_match(top, second)


def _is_confident_group_match(top: MatchGroupResult, second: MatchGroupResult | None) -> bool:
    if top.score < 10:
        if not _allow_same_date_group_match(top, second):
            return False
    if second is None:
        return True
    if (top.score - second.score) >= 3:
        return True
    return _allow_same_date_group_match(top, second)


def _confidence(
    score: float,
    kalshi: NormalizedGame,
    polymarket: NormalizedGame,
    time_delta_min: float | None,
) -> float:
    if not kalshi.matchup_key or kalshi.matchup_key != polymarket.matchup_key:
        return min(0.7, score / 20.0)
    if time_delta_min is None:
        return 0.8
    if time_delta_min <= 15:
        return 0.98
    if time_delta_min <= 60:
        return 0.93
    if time_delta_min <= 120:
        return 0.85
    if kalshi.start_time_utc and polymarket.start_time_utc:
        if abs((kalshi.start_time_utc.date() - polymarket.start_time_utc.date()).days) <= 1:
            return 0.8
    return 0.6


def _teams_missing(game: NormalizedGame) -> bool:
    return game.teams_source == "missing" and not game.matchup_key


def _allow_same_date_match(top: MatchResult, second: MatchResult | None) -> bool:
    if not top.details.get("matchup_key"):
        return False
    if top.details.get("time_reason") != "same_date":
        return False
    if second is None:
        return top.score >= 10
    return (top.score - second.score) >= 1


def _allow_same_date_group_match(
    top: MatchGroupResult, second: MatchGroupResult | None
) -> bool:
    if not top.details.get("matchup_key"):
        return False
    if top.details.get("time_reason") not in {"same_date", "date_within_1_day"}:
        return False
    if second is None:
        return top.score >= 10
    return (top.score - second.score) >= 1
