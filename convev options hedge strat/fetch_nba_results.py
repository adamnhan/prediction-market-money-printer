#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import requests


ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

TEAM_ABBRS = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
}

ESPN_ABBR_MAP = {
    "NYK": "NY",
    "GSW": "GS",
    "NOP": "NO",
    "SAS": "SA",
    "UTA": "UTAH",
    "WAS": "WSH",
}


def _parse_event_key(event_key: str) -> tuple[str | None, str | None, str | None]:
    if len(event_key) < 13:
        return None, None, None
    date_part = event_key[:7]
    teams = event_key[7:]
    if len(teams) != 6:
        return None, None, None
    away = teams[:3]
    home = teams[3:]
    if away not in TEAM_ABBRS or home not in TEAM_ABBRS:
        return None, None, None
    try:
        date_dt = datetime.strptime(date_part, "%y%b%d")
    except ValueError:
        return None, None, None
    return date_dt.strftime("%Y-%m-%d"), away, home


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _read_event_rows(csv_path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get("event_key")
            if key:
                rows.append(
                    {
                        "event_key": key,
                        "event_start_time": row.get("event_start_time") or "",
                    }
                )
    return rows


def _read_event_rows_from_ledger(ledger_db: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with sqlite3.connect(ledger_db) as conn:
        keys = conn.execute("SELECT DISTINCT event_key FROM bundles").fetchall()
    for (key,) in keys:
        if key:
            rows.append({"event_key": key, "event_start_time": ""})
    return rows


def _fmt_yyyymmdd(date_str: str) -> str:
    return date_str.replace("-", "")


def _fetch_scoreboard(dates_param: str, timeout_s: float, max_retries: int, sleep_s: float) -> dict[str, Any]:
    params = {"dates": dates_param}
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(ESPN_SCOREBOARD_URL, params=params, timeout=timeout_s)
            if resp.status_code == 404:
                print(f"[results] no ESPN scoreboard for dates={dates_param} (404)")
                return {"events": []}
            if resp.status_code == 429:
                time.sleep(max(2.0, sleep_s))
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
            last_err = exc
            time.sleep(min(2 ** attempt, 10))
            continue
    raise last_err or RuntimeError("Failed to fetch scoreboard")


def _build_score_map(payload: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    events = payload.get("events") or []
    for event in events:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competition = competitions[0]
        status = (competition.get("status") or {}).get("type") or {}
        if status.get("name") != "STATUS_FINAL":
            continue
        event_date = event.get("date")
        if not event_date:
            continue
        date_key = event_date[:10]
        competitors = competition.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        home_team = home.get("team") or {}
        away_team = away.get("team") or {}
        home_abbr = home_team.get("abbreviation")
        away_abbr = away_team.get("abbreviation")
        if not home_abbr or not away_abbr:
            continue
        out[(date_key, away_abbr, home_abbr)] = {
            "game_id": event.get("id"),
            "home_score": home.get("score"),
            "away_score": away.get("score"),
        }
    return out


def fetch_results(
    csv_path: str,
    out_csv: str,
    sleep_s: float,
    timeout_s: float,
    max_retries: int,
    debug_unmatched: str | None,
    debug_espn_teams: str | None,
    rows_in: list[dict[str, str]] | None = None,
) -> int:
    if rows_in is None:
        rows_in = _read_event_rows(csv_path)
    events: dict[str, dict[str, Any]] = {}
    by_date: dict[str, list[str]] = defaultdict(list)
    seen = set()
    for row in rows_in:
        key = row["event_key"]
        if key in seen:
            continue
        seen.add(key)
        date_str, away, home = _parse_event_key(key)
        if not date_str:
            continue
        event_dates = {date_str}
        if row.get("event_start_time"):
            dt = _parse_iso(row["event_start_time"])
            if dt:
                event_date = dt.strftime("%Y-%m-%d")
                # Prefer event_key date if it differs by more than 1 day.
                try:
                    key_dt = datetime.fromisoformat(date_str)
                    ev_dt = datetime.fromisoformat(event_date)
                    if abs((ev_dt - key_dt).days) <= 1:
                        date_str = event_date
                except ValueError:
                    pass
                event_dates.add(event_date)
        events[key] = {"away": away, "home": home, "dates": event_dates}
        for d in event_dates:
            by_date[d].append(key)

    rows = 0
    unmatched_rows = []
    espn_team_rows = []
    with open(out_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "event_key",
                "game_date",
                "game_id",
                "home_team_abbr",
                "away_team_abbr",
                "home_team_id",
                "away_team_id",
                "home_score",
                "away_score",
                "final_margin",
            ]
        )
        dates = sorted(by_date.keys())
        range_days = 7
        matched_events = set()
        for idx in range(0, len(dates), range_days):
            chunk = dates[idx : idx + range_days]
            if not chunk:
                continue
            start = _fmt_yyyymmdd(chunk[0])
            end = _fmt_yyyymmdd(chunk[-1])
            dates_param = f"{start}-{end}" if start != end else start
            print(f"[results] fetching dates={dates_param} days={len(chunk)}")
            payload = _fetch_scoreboard(dates_param, timeout_s, max_retries, sleep_s)
            score_map = _build_score_map(payload)
            if debug_espn_teams:
                for (date_key, away_abbr, home_abbr), data in score_map.items():
                    espn_team_rows.append(
                        {
                            "date": date_key,
                            "away": away_abbr,
                            "home": home_abbr,
                            "game_id": data.get("game_id"),
                        }
                    )
            matched = 0
            for game_date in chunk:
                items = by_date[game_date]
                for event_key in items:
                    if event_key in matched_events:
                        continue
                    event = events.get(event_key)
                    if not event:
                        continue
                    away = event["away"]
                    home = event["home"]
                    away_candidates = [away]
                    mapped_away = ESPN_ABBR_MAP.get(away)
                    if mapped_away and mapped_away not in away_candidates:
                        away_candidates.append(mapped_away)
                    home_candidates = [home]
                    mapped_home = ESPN_ABBR_MAP.get(home)
                    if mapped_home and mapped_home not in home_candidates:
                        home_candidates.append(mapped_home)

                    match = None
                    candidate_dates = sorted(event["dates"])
                    for candidate_date in candidate_dates:
                        for away_abbr in away_candidates:
                            for home_abbr in home_candidates:
                                match = score_map.get((candidate_date, away_abbr, home_abbr))
                                if match:
                                    break
                            if match:
                                break
                        if match:
                            break
                    if not match:
                        # Try +/- 2 day fallback for timezone shifts.
                        try:
                            for candidate_date in candidate_dates:
                                dt = datetime.fromisoformat(candidate_date)
                                for delta in (-2, -1, 1, 2):
                                    alt = (dt + timedelta(days=delta)).strftime("%Y-%m-%d")
                                    for away_abbr in away_candidates:
                                        for home_abbr in home_candidates:
                                            match = score_map.get((alt, away_abbr, home_abbr))
                                            if match:
                                                break
                                        if match:
                                            break
                                    if match:
                                        break
                                if match:
                                    break
                        except ValueError:
                            match = None
                    if not match:
                        unmatched_rows.append(
                            {
                                "event_key": event_key,
                                "game_date": game_date,
                                "away": away,
                                "home": home,
                                "away_espn": mapped_away or away,
                                "home_espn": mapped_home or home,
                                "candidate_dates": ",".join(candidate_dates),
                            }
                        )
                        continue
                    home_score = match.get("home_score")
                    away_score = match.get("away_score")
                    try:
                        margin = int(home_score) - int(away_score)
                    except Exception:
                        margin = None
                    writer.writerow(
                        [
                            event_key,
                            game_date,
                            match.get("game_id"),
                            home,
                            away,
                            None,
                            None,
                            home_score,
                            away_score,
                            margin,
                        ]
                    )
                    rows += 1
                    matched += 1
                    matched_events.add(event_key)
            print(f"[results] dates={dates_param} matched={matched} total_rows={rows}")
            if sleep_s:
                time.sleep(sleep_s)
    if debug_unmatched:
        with open(debug_unmatched, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["event_key", "game_date", "away", "home", "away_espn", "home_espn", "candidate_dates"]
            )
            for row in unmatched_rows:
                writer.writerow(
                    [
                        row["event_key"],
                        row["game_date"],
                        row["away"],
                        row["home"],
                        row["away_espn"],
                        row["home_espn"],
                        row.get("candidate_dates", ""),
                    ]
                )
    if debug_espn_teams:
        with open(debug_espn_teams, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "away", "home", "game_id"])
            for row in espn_team_rows:
                writer.writerow([row["date"], row["away"], row["home"], row["game_id"]])
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NBA results from stats.nba.com scoreboard.")
    parser.add_argument("--csv", default="data/nba_historical_markets.csv")
    parser.add_argument("--out", default="data/nba_results.csv")
    parser.add_argument("--ledger-db", default="")
    parser.add_argument("--sleep-s", type=float, default=0.8)
    parser.add_argument("--timeout-s", type=float, default=40.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--debug-unmatched", default="")
    parser.add_argument("--debug-espn-teams", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    debug_unmatched = args.debug_unmatched or None
    debug_espn_teams = args.debug_espn_teams or None
    if args.ledger_db:
        rows_in = _read_event_rows_from_ledger(args.ledger_db)
        if not rows_in:
            print(f"[results] no event_keys found in ledger: {args.ledger_db}")
            return 1
        rows = fetch_results(
            args.csv,
            args.out,
            args.sleep_s,
            args.timeout_s,
            args.max_retries,
            debug_unmatched,
            debug_espn_teams,
            rows_in=rows_in,
        )
        print(f"[results] wrote {rows} rows to {args.out}")
        return 0
    rows = fetch_results(
        args.csv,
        args.out,
        args.sleep_s,
        args.timeout_s,
        args.max_retries,
        debug_unmatched,
        debug_espn_teams,
    )
    print(f"[results] wrote {rows} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
