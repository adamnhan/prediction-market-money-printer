#!/usr/bin/env python3
"""Phase 4: Realized settlement payoff and edge diagnostics for Polymarket NHL moneyline markets."""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


def _safe_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def _parse_outcomes(val: Any) -> List[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return [str(v) for v in val if v is not None]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v is not None]
            if isinstance(parsed, str):
                try:
                    parsed2 = json.loads(parsed)
                    if isinstance(parsed2, list):
                        return [str(v) for v in parsed2 if v is not None]
                except Exception:
                    return []
        except Exception:
            return []
    return []


def _extract_outcomes_list(val: Any) -> List[Dict[str, Any]]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return [v for v in val if isinstance(v, dict)]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            if isinstance(parsed, list):
                return [v for v in parsed if isinstance(v, dict)]
        except Exception:
            return []
    return []


def _is_resolved(market: Dict[str, Any]) -> bool:
    if market.get("resolved") is True:
        return True
    if market.get("closed") is True:
        return True
    outcomes = _extract_outcomes_list(market.get("outcomes"))
    for o in outcomes:
        if o.get("winner") is True:
            return True
        if o.get("payout") == 1:
            return True
        if o.get("result") in ("WIN", "LOSE"):
            return True
    if market.get("resolvedAt") is not None:
        return True
    if market.get("resolutionTime") is not None:
        return True
    if market.get("resolvedTime") is not None:
        return True
    if market.get("resolutionDate") is not None:
        return True
    return False


def _get_winning_outcome(market: Dict[str, Any]) -> Optional[str]:
    outcomes = _extract_outcomes_list(market.get("outcomes"))
    for o in outcomes:
        if o.get("winner") is True:
            return _safe_str(o.get("name"))
        if o.get("payout") == 1:
            return _safe_str(o.get("name"))
        if o.get("result") == "WIN":
            return _safe_str(o.get("name"))
    return _safe_str(market.get("winningOutcome") or market.get("winner") or market.get("outcome"))

def _price_cents(row: pd.Series) -> Optional[float]:
    val = row.get("price_cents")
    if val is not None and not (isinstance(val, float) and pd.isna(val)):
        try:
            return float(val)
        except Exception:
            return None
    price = row.get("price")
    if price is None or (isinstance(price, float) and pd.isna(price)):
        return None
    try:
        price = float(price)
    except Exception:
        return None
    if price <= 1.0:
        return price * 100.0
    return price


def _side_norm(outcome: Optional[str], outcomes: List[str]) -> Optional[str]:
    if not outcome or len(outcomes) < 2:
        return None
    outcome_l = outcome.lower()
    a = outcomes[0].lower()
    b = outcomes[1].lower()
    if outcome_l == a:
        return "SIDE_A"
    if outcome_l == b:
        return "SIDE_B"
    return None


def _infer_trade_dir(val: Any) -> str:
    s = _safe_str(val)
    if s is None:
        return "buy"
    s = s.lower()
    if s in ("sell", "s", "short"):
        return "sell"
    if s in ("buy", "b", "long"):
        return "buy"
    return "buy"


def _request_with_backoff(
    session: requests.Session, url: str, params: Dict[str, Any], sleep_s: float, max_retries: int = 6
) -> Tuple[Optional[requests.Response], Optional[str]]:
    backoff = 1.0
    last_err = None
    for _ in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            last_err = str(exc)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue

        if resp.status_code in (429, 500, 502, 503, 504):
            last_err = f"HTTP {resp.status_code}"
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue

        time.sleep(sleep_s)
        return resp, None

    return None, last_err


def _fetch_markets_by_slug(
    session: requests.Session,
    base_url: str,
    slugs: List[str],
    sleep_s: float,
    log_fn,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, slug in enumerate(slugs):
        resp = session.get(f"{base_url}/markets", params={"slug": slug}, timeout=30)
        if resp.status_code != 200:
            log_fn(f"markets_slug_http_{resp.status_code}:{slug}")
            continue
        try:
            payload = resp.json()
        except Exception:
            log_fn(f"markets_slug_invalid_json:{slug}")
            continue
        data: List[Dict[str, Any]] = []
        if isinstance(payload, list):
            data = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            data = [payload]
        matched = 0
        for item in data:
            got_slug = _safe_str(item.get("slug"))
            if got_slug and got_slug.lower() == slug.lower():
                rows.append(item)
                matched += 1
        if matched == 0:
            log_fn(f"markets_slug_no_match:{slug}")
        if (idx + 1) % 25 == 0:
            log_fn(f"fetched markets by slug {idx + 1} of {len(slugs)}")
        time.sleep(sleep_s)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4: realized settlement payoff diagnostics")
    parser.add_argument("--phase2_path", default="./kch-strat/phase2_out/trades_enriched.parquet")
    parser.add_argument("--phase3_path", default="./kch-strat/phase3_out/event_positions.parquet")
    parser.add_argument("--outdir", default="./kch-strat/phase4_out")
    parser.add_argument("--gamma_base", default="https://gamma-api.polymarket.com")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--held_threshold", type=float, default=0.01)
    parser.add_argument("--late_window_hours", type=float, default=6)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    def log_fn(msg: str) -> None:
        print(f"[phase4] {msg}")

    trades = pd.read_parquet(args.phase2_path)

    trades = trades[
        trades["event_slug"].astype(str).str.startswith("nhl-")
        & ~trades["slug"].astype(str).str.contains("spread|total", case=False, na=False)
    ].copy()
    if trades.empty:
        raise SystemExit("no rows after filtering NHL moneyline")

    trades["event_slug"] = trades["event_slug"].apply(_safe_str)
    trades["slug"] = trades["slug"].apply(_safe_str)
    trades["market_key"] = trades["event_slug"] + "::" + trades["slug"]

    trades["outcome_raw"] = trades["outcome"].apply(_safe_str)
    trades["price_cents_norm"] = trades.apply(_price_cents, axis=1)
    trades["trade_dir"] = trades["side"].apply(_infer_trade_dir)

    outcomes_by_market: Dict[str, List[str]] = {}
    for market_key, group in trades.groupby("market_key"):
        outcomes = []
        for val in group["outcomes"]:
            outcomes = _parse_outcomes(val)
            if outcomes:
                break
        outcomes_by_market[market_key] = outcomes

    trades["side_norm"] = trades.apply(
        lambda r: _side_norm(r.get("outcome_raw"), outcomes_by_market.get(r["market_key"], [])), axis=1
    )
    trades = trades[trades["side_norm"].notna()].copy()

    condition_ids = set([c for c in trades["condition_id"].dropna().unique().tolist() if c])
    slugs = sorted(set([s for s in trades["slug"].dropna().unique().tolist() if s]))
    log_fn(f"filtered trades rows={len(trades)} unique_condition_ids={len(condition_ids)} slugs={len(slugs)}")
    session = requests.Session()
    log_fn("fetching markets from gamma (by slug)")
    markets_rows = _fetch_markets_by_slug(session, args.gamma_base, slugs, args.sleep, log_fn)
    log_fn(f"markets fetched matched={len(markets_rows)}")

    # Build resolution dim
    resolution_rows: List[Dict[str, Any]] = []
    market_by_condition = {
        _safe_str(m.get("conditionId")): m for m in markets_rows if _safe_str(m.get("conditionId")) is not None
    }

    for cid in condition_ids:
        m = market_by_condition.get(cid)
        if m is None:
            continue
        outcomes = _parse_outcomes(m.get("outcomes") or m.get("outcomeNames") or m.get("outcome") or [])
        winning_outcome = _get_winning_outcome(m)
        resolved = _is_resolved(m)
        closed = m.get("closed") or m.get("isClosed")
        resolved_time = (
            m.get("resolvedAt")
            or m.get("resolvedTime")
            or m.get("resolutionTime")
            or m.get("resolutionDate")
        )

        payout_a = None
        payout_b = None
        if winning_outcome and len(outcomes) >= 2:
            if winning_outcome.lower() == outcomes[0].lower():
                payout_a, payout_b = 100.0, 0.0
            elif winning_outcome.lower() == outcomes[1].lower():
                payout_a, payout_b = 0.0, 100.0

        resolution_rows.append(
            {
                "condition_id": cid,
                "resolved": bool(resolved) if resolved is not None else None,
                "closed": bool(closed) if closed is not None else None,
                "resolved_time": resolved_time,
                "winning_outcome": winning_outcome,
                "payout_per_share_sideA": payout_a,
                "payout_per_share_sideB": payout_b,
                "raw_json": json.dumps(m, separators=(",", ":"), sort_keys=True, ensure_ascii=True),
            }
        )

    resolution_dim = pd.DataFrame(resolution_rows)
    log_fn(
        f"resolution_dim rows={len(resolution_dim)} resolved_rate={resolution_dim['resolved'].mean() if len(resolution_dim) else 0.0}"
    )
    resolution_dim.to_parquet(os.path.join(args.outdir, "market_resolution_dim.parquet"), index=False)

    # Aggregate per market
    pnl_rows: List[Dict[str, Any]] = []
    for market_key, group in trades.groupby("market_key"):
        condition_id = _safe_str(group["condition_id"].iloc[0])
        outcomes = outcomes_by_market.get(market_key, [])

        def _net(side: str) -> float:
            buys = group[(group["side_norm"] == side) & (group["trade_dir"] == "buy")]["size"].sum()
            sells = group[(group["side_norm"] == side) & (group["trade_dir"] == "sell")]["size"].sum()
            return float(buys) - float(sells)

        net_a = _net("SIDE_A")
        net_b = _net("SIDE_B")

        held = (abs(net_a) + abs(net_b)) > args.held_threshold

        buy_cost = (
            group[group["trade_dir"] == "buy"]["price_cents_norm"]
            .fillna(0)
            .mul(group[group["trade_dir"] == "buy"]["size"])
            .sum()
        )
        sell_proceeds = (
            group[group["trade_dir"] == "sell"]["price_cents_norm"]
            .fillna(0)
            .mul(group[group["trade_dir"] == "sell"]["size"])
            .sum()
        )
        cost_total = float(buy_cost) - float(sell_proceeds)

        res = resolution_dim[resolution_dim["condition_id"] == condition_id]
        if len(res):
            payout_a = res["payout_per_share_sideA"].iloc[0]
            payout_b = res["payout_per_share_sideB"].iloc[0]
            winning_outcome = res["winning_outcome"].iloc[0]
            resolved_flag = res["resolved"].iloc[0]
        else:
            payout_a = None
            payout_b = None
            winning_outcome = None
            resolved_flag = None

        settle_value = None
        pnl = None
        roi = None
        inferred_winner = None
        inferred = False
        if payout_a is not None and payout_b is not None:
            settle_value = payout_a * net_a + payout_b * net_b
            pnl = settle_value - cost_total
            capital_deployed = float(buy_cost) if float(buy_cost) > 0 else None
            roi = pnl / capital_deployed if capital_deployed else None
        else:
            # Infer winner from last traded price (higher price = higher implied win prob)
            last_side_prices = (
                group.sort_values("timestamp")
                .groupby("side_norm")["price_cents_norm"]
                .last()
                .to_dict()
            )
            pa = last_side_prices.get("SIDE_A")
            pb = last_side_prices.get("SIDE_B")
            if pa is not None and pb is not None and pa != pb:
                inferred = True
                if pa > pb:
                    inferred_winner = outcomes[0] if len(outcomes) >= 2 else "SIDE_A"
                    payout_a, payout_b = 100.0, 0.0
                else:
                    inferred_winner = outcomes[1] if len(outcomes) >= 2 else "SIDE_B"
                    payout_a, payout_b = 0.0, 100.0
                settle_value = payout_a * net_a + payout_b * net_b
                pnl = settle_value - cost_total
                capital_deployed = float(buy_cost) if float(buy_cost) > 0 else None
                roi = pnl / capital_deployed if capital_deployed else None
            else:
                capital_deployed = float(buy_cost) if float(buy_cost) > 0 else None

        # payoff shape (pnl if A or B wins)
        pnl_if_a = None
        pnl_if_b = None
        if net_a is not None and net_b is not None:
            pnl_if_a = 100.0 * net_a + 0.0 * net_b - cost_total
            pnl_if_b = 0.0 * net_a + 100.0 * net_b - cost_total
        min_pnl = None
        max_pnl = None
        convexity_index = None
        if pnl_if_a is not None and pnl_if_b is not None:
            min_pnl = min(pnl_if_a, pnl_if_b)
            max_pnl = max(pnl_if_a, pnl_if_b)
            if min_pnl is not None and min_pnl < 0:
                convexity_index = max_pnl / abs(min_pnl) if abs(min_pnl) > 0 else None

        pnl_rows.append(
            {
                "market_key": market_key,
                "event_slug": group["event_slug"].iloc[0],
                "slug": group["slug"].iloc[0],
                "condition_id": condition_id,
                "net_shares_sideA": net_a,
                "net_shares_sideB": net_b,
                "held_to_settlement": held,
                "cost_total_cents": cost_total,
                "settle_value_cents": settle_value,
                "pnl_cents": pnl,
                "capital_deployed_cents": capital_deployed,
                "roi": roi,
                "resolved_flag": resolved_flag,
                "winning_outcome": winning_outcome,
                "inferred_winner": inferred_winner,
                "used_inferred_winner": inferred,
                "pnl_if_A_wins_cents": pnl_if_a,
                "pnl_if_B_wins_cents": pnl_if_b,
                "min_pnl_cents": min_pnl,
                "max_pnl_cents": max_pnl,
                "convexity_index": convexity_index,
            }
        )
        if len(pnl_rows) % 25 == 0:
            log_fn(f"processed markets {len(pnl_rows)}")

    pnl_df = pd.DataFrame(pnl_rows)

    # Merge phase3 features
    phase3 = pd.read_parquet(args.phase3_path)
    merged = pnl_df.merge(
        phase3[["market_key", "cheap_price_band", "size_ratio_exp_to_cheap", "hedge_delay_seconds"]],
        on="market_key",
        how="left",
    )

    merged.to_parquet(os.path.join(args.outdir, "market_pnl.parquet"), index=False)
    log_fn("wrote market_pnl.parquet")

    # Aggregations
    band_stats = (
        merged.groupby("cheap_price_band")["pnl_cents"]
        .agg(["count", "mean", "median", "sum"])
        .reset_index()
        .to_dict(orient="records")
    )
    convexity_series = merged["convexity_index"].dropna()
    convexity_median = float(convexity_series.median()) if not convexity_series.empty else None
    convexity_p10 = float(convexity_series.quantile(0.1)) if not convexity_series.empty else None
    convexity_p90 = float(convexity_series.quantile(0.9)) if not convexity_series.empty else None

    pnl_vals = merged["pnl_cents"].dropna().sort_values(ascending=False).tolist()
    pnl_sum = float(sum(pnl_vals)) if pnl_vals else 0.0
    top1 = sum(pnl_vals[:1]) / pnl_sum if pnl_sum else None
    top3 = sum(pnl_vals[:3]) / pnl_sum if pnl_sum else None
    top5 = sum(pnl_vals[:5]) / pnl_sum if pnl_sum else None

    pct_profitable = float((merged["pnl_cents"] > 0).mean()) if len(merged) else 0.0

    report = {
        "traded_markets": int(len(merged)),
        "resolved_markets": int(merged["settle_value_cents"].notna().sum()),
        "resolved_markets_inferred": int(merged["used_inferred_winner"].sum())
        if "used_inferred_winner" in merged.columns
        else 0,
        "held_to_settlement_markets": int(merged["held_to_settlement"].sum()),
        "total_pnl_cents": float(merged["pnl_cents"].sum(skipna=True)),
        "pct_profitable": pct_profitable,
        "pnl_by_cheap_price_band": band_stats,
        "convexity_index_median": convexity_median,
        "convexity_index_p10": convexity_p10,
        "convexity_index_p90": convexity_p90,
        "pnl_concentration_top1": top1,
        "pnl_concentration_top3": top3,
        "pnl_concentration_top5": top5,
    }

    with open(os.path.join(args.outdir, "phase4_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"traded_markets {report['traded_markets']}")
    print(f"resolved_markets {report['resolved_markets']}")
    print(f"held_to_settlement_markets {report['held_to_settlement_markets']}")
    print(f"total_pnl_cents {report['total_pnl_cents']}")
    print(f"pnl_by_band {band_stats}")
    print(f"convexity_index median {convexity_median} p90 {convexity_p90}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
