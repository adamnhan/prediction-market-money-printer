"""Normalize market metadata across venues."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable


@dataclass(frozen=True)
class NormalizedGame:
    venue: str
    venue_id: str
    league: str
    start_time_utc: datetime | None
    start_bucket_utc: datetime | None
    home_team: str | None
    away_team: str | None
    home_team_norm: str | None
    away_team_norm: str | None
    matchup_key: str | None
    game_key: str
    title: str
    raw: dict
    teams_source: str | None = None
    time_source: str | None = None
    parse_warnings: tuple[str, ...] = ()


_ALIASES = {
    "la clippers": "los angeles clippers",
    "la c": "los angeles clippers",
    "los angeles c": "los angeles clippers",
    "clippers": "los angeles clippers",
    "ny knicks": "new york knicks",
    "knicks": "new york knicks",
    "new york": "new york knicks",
    "okc thunder": "oklahoma city thunder",
    "thunder": "oklahoma city thunder",
    "oklahoma city": "oklahoma city thunder",
    "gs warriors": "golden state warriors",
    "warriors": "golden state warriors",
    "la l": "los angeles lakers",
    "los angeles l": "los angeles lakers",
    "lakers": "los angeles lakers",
    "atlanta": "atlanta hawks",
    "hawks": "atlanta hawks",
    "boston": "boston celtics",
    "celtics": "boston celtics",
    "brooklyn": "brooklyn nets",
    "nets": "brooklyn nets",
    "charlotte": "charlotte hornets",
    "hornets": "charlotte hornets",
    "chicago": "chicago bulls",
    "bulls": "chicago bulls",
    "cleveland": "cleveland cavaliers",
    "cavaliers": "cleveland cavaliers",
    "dallas": "dallas mavericks",
    "mavericks": "dallas mavericks",
    "denver": "denver nuggets",
    "nuggets": "denver nuggets",
    "detroit": "detroit pistons",
    "pistons": "detroit pistons",
    "golden state": "golden state warriors",
    "houston": "houston rockets",
    "rockets": "houston rockets",
    "indiana": "indiana pacers",
    "pacers": "indiana pacers",
    "memphis": "memphis grizzlies",
    "grizzlies": "memphis grizzlies",
    "miami": "miami heat",
    "heat": "miami heat",
    "milwaukee": "milwaukee bucks",
    "bucks": "milwaukee bucks",
    "minnesota": "minnesota timberwolves",
    "timberwolves": "minnesota timberwolves",
    "new orleans": "new orleans pelicans",
    "pelicans": "new orleans pelicans",
    "orlando": "orlando magic",
    "magic": "orlando magic",
    "philadelphia": "philadelphia 76ers",
    "76ers": "philadelphia 76ers",
    "sixers": "philadelphia 76ers",
    "phoenix": "phoenix suns",
    "suns": "phoenix suns",
    "portland": "portland trail blazers",
    "trail blazers": "portland trail blazers",
    "blazers": "portland trail blazers",
    "sacramento": "sacramento kings",
    "kings": "sacramento kings",
    "san antonio": "san antonio spurs",
    "spurs": "san antonio spurs",
    "toronto": "toronto raptors",
    "raptors": "toronto raptors",
    "utah": "utah jazz",
    "jazz": "utah jazz",
    "washington": "washington wizards",
    "wizards": "washington wizards",
}


def normalize_market(raw: dict, venue: str | None = None) -> NormalizedGame:
    return normalize_game(raw, venue=venue)


def normalize_game(raw: dict, venue: str | None = None) -> NormalizedGame:
    warnings: list[str] = []
    venue_value = _detect_venue(raw, venue)
    venue_id = _extract_venue_id(raw, venue_value)
    title = _extract_title(raw)
    league = _extract_league(raw, title)

    away_team, home_team, teams_source, team_warnings = _extract_teams(raw, title)
    warnings.extend(team_warnings)

    start_time, time_source, time_warnings = _extract_start_time(raw)
    warnings.extend(time_warnings)
    if venue_value == "kalshi":
        derived_start = _kalshi_event_ticker_start(raw)
        if derived_start:
            if start_time is None or _days_apart(start_time, derived_start) > 3:
                start_time = derived_start
                time_source = "derived"
                warnings.append("kalshi_event_ticker_date")
    start_bucket = _floor_to_minutes(start_time, minutes=5) if start_time else None

    away_norm = normalize_team(away_team) if away_team else None
    home_norm = normalize_team(home_team) if home_team else None
    matchup_key = _matchup_key(away_norm, home_norm)
    if matchup_key is None and title:
        vs_key = _matchup_key_from_vs_title(title)
        if vs_key:
            matchup_key = vs_key
            warnings.append("vs_title_matchup_only")

    key_base = _game_key_base(league, matchup_key, start_bucket, venue_value, venue_id)
    game_key = hashlib.sha1(key_base.encode("utf-8")).hexdigest()

    return NormalizedGame(
        venue=venue_value,
        venue_id=venue_id,
        league=league,
        start_time_utc=start_time,
        start_bucket_utc=start_bucket,
        home_team=home_team,
        away_team=away_team,
        home_team_norm=home_norm,
        away_team_norm=away_norm,
        matchup_key=matchup_key,
        game_key=game_key,
        title=title,
        raw=raw,
        teams_source=teams_source,
        time_source=time_source,
        parse_warnings=tuple(warnings),
    )


def format_normalized_game(game: NormalizedGame) -> str:
    if game.start_time_utc:
        start_label = game.start_time_utc.strftime("%Y-%m-%d %H:%MZ")
    else:
        start_label = "unknown"

    away = game.away_team_norm or game.away_team
    home = game.home_team_norm or game.home_team
    if away and home:
        matchup = f"{away} @ {home}"
    else:
        matchup = game.title or "unknown matchup"

    return (
        f"{game.league} | {start_label} | {matchup} | "
        f"venue={game.venue} id={game.venue_id}"
    )


def normalize_team(name: str) -> str:
    cleaned = _strip_team_suffixes(name)
    cleaned = re.sub(r"[\\.,'()\\-]", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    return _ALIASES.get(cleaned, cleaned)


def _detect_venue(raw: dict, venue: str | None) -> str:
    if venue:
        return venue
    value = raw.get("venue") or raw.get("source") or raw.get("exchange")
    if value:
        return str(value)
    if _has_any(raw, ("event_id", "eventId", "gamma_id")):
        return "polymarket"
    if _has_any(raw, ("ticker", "market_ticker", "event_ticker", "kalshi_ticker")):
        return "kalshi"
    return "unknown"


def _extract_venue_id(raw: dict, venue: str) -> str:
    if venue == "kalshi":
        return _first_value(raw, ("ticker", "market_ticker", "event_ticker", "id"), default="unknown")
    if venue == "polymarket":
        return _first_value(raw, ("event_id", "eventId", "id"), default="unknown")
    return _first_value(raw, ("id", "event_id", "eventId", "ticker"), default="unknown")


def _extract_title(raw: dict) -> str:
    return _first_value(
        raw,
        (
            "title",
            "question",
            "name",
            "event_title",
            "eventName",
            "event_name",
            "market_title",
        ),
        default="",
    )


def _extract_league(raw: dict, title: str) -> str:
    league = raw.get("league") or raw.get("sport") or raw.get("category")
    if league:
        league_value = str(league).upper()
        return league_value if league_value != "BASKETBALL" else "NBA"
    if "NBA" in title.upper():
        return "NBA"
    return "NBA"


def _extract_teams(raw: dict, title: str) -> tuple[str | None, str | None, str | None, list[str]]:
    warnings: list[str] = []
    away, home = _explicit_team_fields(raw)
    if away or home:
        return away, home, "explicit_fields", warnings

    away, home = _participants_team_fields(raw)
    if away or home:
        return away, home, "explicit_fields", warnings

    away, home, title_warnings = _parse_title_teams(title)
    warnings.extend(title_warnings)
    source = "parsed_title" if away or home else "missing"
    return away, home, source, warnings


def _explicit_team_fields(raw: dict) -> tuple[str | None, str | None]:
    away = _first_value(raw, ("away_team", "awayTeam", "away"))
    home = _first_value(raw, ("home_team", "homeTeam", "home"))
    return away or None, home or None


def _participants_team_fields(raw: dict) -> tuple[str | None, str | None]:
    for key in ("participants", "teams", "outcomes"):
        value = raw.get(key)
        if isinstance(value, list):
            away, home = _parse_participants(value)
            if away or home:
                return away, home
    return None, None


def _parse_participants(participants: Iterable[dict]) -> tuple[str | None, str | None]:
    away = None
    home = None
    for entry in participants:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("team") or entry.get("label")
        if not name:
            continue
        side = entry.get("side") or entry.get("home_away") or entry.get("designation")
        if isinstance(side, str):
            if side.lower() == "home":
                home = str(name)
                continue
            if side.lower() == "away":
                away = str(name)
                continue
        is_home = entry.get("home") if "home" in entry else entry.get("is_home")
        if isinstance(is_home, bool):
            if is_home:
                home = str(name)
            else:
                away = str(name)
    return away or None, home or None


def _parse_title_teams(title: str) -> tuple[str | None, str | None, list[str]]:
    warnings: list[str] = []
    if not title:
        return None, None, ["missing_title"]

    at_split = re.split(r"\s+@\s+", title, maxsplit=1, flags=re.IGNORECASE)
    if len(at_split) == 2:
        away = at_split[0].strip()
        home = at_split[1].strip()
        return away or None, home or None, warnings

    at_split = re.split(r"\s+at\s+", title, maxsplit=1, flags=re.IGNORECASE)
    if len(at_split) == 2:
        away = at_split[0].strip()
        home = at_split[1].strip()
        return away or None, home or None, warnings

    vs_split = re.split(r"\s+vs\.?\s+", title, maxsplit=1, flags=re.IGNORECASE)
    if len(vs_split) == 2:
        warnings.append("vs_title_no_home_away")
        return None, None, warnings

    warnings.append("unparseable_title")
    return None, None, warnings


def _extract_start_time(raw: dict) -> tuple[datetime | None, str, list[str]]:
    warnings: list[str] = []
    for key in (
        "start_time",
        "start_time_utc",
        "startTime",
        "start_ts",
        "startTimestamp",
        "start_timestamp",
        "event_start",
        "event_start_time",
        "close_time",
        "close_ts",
        "open_time",
        "open_ts",
        "scheduled_start",
        "scheduled_start_time",
        "start_date",
    ):
        if key not in raw:
            continue
        parsed = _parse_datetime(raw.get(key))
        if parsed:
            return parsed, "explicit", warnings
        warnings.append(f"invalid_time_{key}")
    return None, "missing", warnings


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return _from_timestamp(float(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if _looks_like_number(text):
            return _from_timestamp(float(text))
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _kalshi_event_ticker_start(raw: dict) -> datetime | None:
    ticker = raw.get("event_ticker") or raw.get("eventTicker") or raw.get("event")
    if not isinstance(ticker, str):
        return None
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    date_part = parts[1][:7]
    if len(date_part) != 7:
        return None
    year_part = date_part[0:2]
    month_part = date_part[2:5]
    day_part = date_part[5:7]
    if not (year_part.isdigit() and day_part.isdigit()):
        return None
    month_map = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }
    month = month_map.get(month_part.upper())
    if not month:
        return None
    year = 2000 + int(year_part)
    day = int(day_part)
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


def _days_apart(left: datetime, right: datetime) -> int:
    return abs((left.date() - right.date()).days)


def _matchup_key_from_vs_title(title: str) -> str | None:
    vs_split = re.split(r"\s+vs\.?\s+", title, maxsplit=1, flags=re.IGNORECASE)
    if len(vs_split) != 2:
        return None
    left = vs_split[0].strip()
    right = vs_split[1].strip()
    if not left or not right:
        return None
    left_norm = normalize_team(left)
    right_norm = normalize_team(right)
    return _matchup_key(left_norm, right_norm)


def _strip_team_suffixes(name: str) -> str:
    cleaned = name.lower().strip()
    cleaned = re.sub(r"\bwho\s+will\s+win\??$", "", cleaned).strip()
    cleaned = re.sub(r"\bwho\s+wins\??$", "", cleaned).strip()
    cleaned = re.sub(r"\bto\s+win\??$", "", cleaned).strip()
    cleaned = re.sub(r"\bwins\??$", "", cleaned).strip()
    cleaned = re.sub(r"\bwin\??$", "", cleaned).strip()
    cleaned = re.sub(r"\bwinner\??$", "", cleaned).strip()
    return cleaned


def _from_timestamp(value: float) -> datetime | None:
    if value > 1e12:
        value = value / 1000.0
    if value <= 0:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc)


def _looks_like_number(text: str) -> bool:
    return bool(re.fullmatch(r"\d+(\.\d+)?", text))


def _floor_to_minutes(dt: datetime, minutes: int = 5) -> datetime:
    minute_bucket = (dt.minute // minutes) * minutes
    return dt.replace(minute=minute_bucket, second=0, microsecond=0)


def _matchup_key(away_norm: str | None, home_norm: str | None) -> str | None:
    if not away_norm or not home_norm:
        return None
    return "|".join(sorted([away_norm, home_norm]))


def _game_key_base(
    league: str,
    matchup_key: str | None,
    start_bucket: datetime | None,
    venue: str,
    venue_id: str,
) -> str:
    bucket_label = start_bucket.strftime("%Y-%m-%dT%H:%MZ") if start_bucket else "unknown"
    matchup_label = matchup_key or "unknown"
    return f"{league}|{matchup_label}|{bucket_label}|{venue}|{venue_id}"


def _first_value(raw: dict, keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = raw.get(key)
        if value is not None and value != "":
            return str(value)
    return default


def _has_any(raw: dict, keys: Iterable[str]) -> bool:
    return any(key in raw for key in keys)
