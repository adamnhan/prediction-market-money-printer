#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from typing import Any


SPREAD_RE = re.compile(r"([A-Z]+)(-?\d+)$")


def _parse_spread_threshold(market_ticker: str) -> tuple[str | None, int | None]:
    suffix = market_ticker.split("-")[-1]
    match = SPREAD_RE.match(suffix or "")
    if not match:
        return None, None
    team = match.group(1)
    try:
        k = int(match.group(2))
    except ValueError:
        return None, None
    return team, k


def _parse_total_threshold(market_ticker: str) -> int | None:
    suffix = market_ticker.split("-")[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def build_ladders(csv_path: str) -> list[dict[str, Any]]:
    ladders: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            series = row.get("series_ticker") or ""
            ticker = row.get("market_ticker") or ""
            event_key = row.get("event_key") or ""
            if not ticker or not event_key:
                continue
            if series == "KXNBASPREAD":
                team, k = _parse_spread_threshold(ticker)
                if team is None or k is None:
                    continue
                ladders[(event_key, "SPREAD_TEAM", team)].append(
                    {"k": k, "ticker": ticker}
                )
            elif series == "KXNBATOTAL":
                k = _parse_total_threshold(ticker)
                if k is None:
                    continue
                ladders[(event_key, "TOTAL_POINTS", "TOTAL")].append(
                    {"k": k, "ticker": ticker}
                )
            elif series == "KXNBAGAME":
                ladders[(event_key, "MONEYLINE", "WIN")].append(
                    {"k": 0, "ticker": ticker}
                )
    out: list[dict[str, Any]] = []
    for (event_key, family, team_code), items in ladders.items():
        counts: dict[int, int] = defaultdict(int)
        tickers_by_k: dict[str, str] = {}
        for item in items:
            k = int(item["k"])
            counts[k] += 1
            tickers_by_k.setdefault(str(k), item["ticker"])
        thresholds = sorted(counts.keys())
        flags: list[str] = []
        if any(v > 1 for v in counts.values()):
            flags.append("duplicate_thresholds")
        gaps = [b - a for a, b in zip(thresholds, thresholds[1:]) if b - a > 1]
        if gaps:
            flags.append("missing_intermediate_thresholds")
        out.append(
            {
                "event_key": event_key,
                "ladder_family": family,
                "team_code": team_code,
                "thresholds": thresholds,
                "tickers_by_k": tickers_by_k,
                "quality_flags": flags,
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build event ladder index from historical markets.")
    parser.add_argument("--csv", default="data/nba_historical_markets.csv")
    parser.add_argument("--out-jsonl", default="data/nba_event_ladders.jsonl")
    parser.add_argument("--out-csv", default="data/nba_event_ladders.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ladders = build_ladders(args.csv)
    with open(args.out_jsonl, "w", encoding="utf-8") as handle:
        for row in ladders:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=True))
            handle.write("\n")
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["event_key", "ladder_family", "team_code", "thresholds", "tickers_by_k", "quality_flags"]
        )
        for row in ladders:
            writer.writerow(
                [
                    row["event_key"],
                    row["ladder_family"],
                    row["team_code"],
                    json.dumps(row["thresholds"], separators=(",", ":")),
                    json.dumps(row["tickers_by_k"], separators=(",", ":")),
                    json.dumps(row["quality_flags"], separators=(",", ":")),
                ]
            )
    print(f"[ladders] rows={len(ladders)} jsonl={args.out_jsonl} csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
