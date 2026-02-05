#!/usr/bin/env python3
from __future__ import annotations

import time
from typing import Any, Callable

from nba_engine.phase5 import _extract_orderbook_levels, _derive_best_prices


def _best_size(levels: list[tuple[int, int]], best: str) -> tuple[int | None, int | None]:
    if not levels:
        return None, None
    if best == "max":
        levels = sorted(levels, key=lambda x: x[0], reverse=True)
    else:
        levels = sorted(levels, key=lambda x: x[0])
    return levels[0][0], levels[0][1]


def _top_of_book(payload: dict[str, Any]) -> dict[str, Any]:
    levels = _extract_orderbook_levels(payload)
    best_prices = _derive_best_prices(levels)
    yes_bid_px, yes_bid_sz = _best_size(levels["yes_bids"], "max")
    yes_ask_px, yes_ask_sz = _best_size(levels["yes_asks"], "min")
    return {
        "yes_bid": best_prices.get("yes_bid"),
        "yes_ask": best_prices.get("yes_ask"),
        "yes_bid_sz": yes_bid_sz,
        "yes_ask_sz": yes_ask_sz,
    }


def execute_bundle(
    bundle: dict[str, Any],
    book_provider: Callable[[list[str]], dict[str, dict[str, Any]]],
    ttl_s: float = 2.0,
    poll_s: float = 0.2,
) -> dict[str, Any]:
    legs = bundle.get("legs") or []
    tickers = [leg["ticker"] for leg in legs]
    start = time.time()
    last_reasons: list[str] = []
    while time.time() - start <= ttl_s:
        payloads = book_provider(tickers)
        all_ok = True
        last_reasons = []
        fills = []
        for leg in legs:
            ticker = leg["ticker"]
            side = leg["side"]
            qty = int(leg.get("qty") or 1)
            limit_price = int(round(float(leg.get("limit_price")) * 100))
            payload = payloads.get(ticker)
            if payload is None:
                all_ok = False
                last_reasons.append(f"missing_book:{ticker}")
                continue
            top = _top_of_book(payload)
            if side == "BUY_YES":
                ask = top.get("yes_ask")
                ask_sz = top.get("yes_ask_sz") or 0
                if ask is None or ask > limit_price or ask_sz < qty:
                    all_ok = False
                    last_reasons.append(f"buy_not_fill:{ticker}")
                    continue
                fills.append({"ticker": ticker, "fill_price": ask, "fill_qty": qty})
            elif side == "SELL_YES":
                bid = top.get("yes_bid")
                bid_sz = top.get("yes_bid_sz") or 0
                if bid is None or bid < limit_price or bid_sz < qty:
                    all_ok = False
                    last_reasons.append(f"sell_not_fill:{ticker}")
                    continue
                fills.append({"ticker": ticker, "fill_price": bid, "fill_qty": qty})
            else:
                all_ok = False
                last_reasons.append(f"invalid_side:{ticker}")
        if all_ok and fills:
            return {"status": "FILLED", "reasons": [], "fills": fills}
        time.sleep(poll_s)
    return {"status": "REJECTED", "reasons": last_reasons or ["TTL_EXPIRED"], "fills": []}


def _portfolio_max_loss(legs: list[dict[str, Any]]) -> float:
    if not legs:
        return 0.0
    ks = sorted(set(int(leg["k"]) for leg in legs))
    bounds = [-10**9] + ks + [10**9]
    const_cost = 0.0
    for leg in legs:
        w = leg["w"]
        p = leg["p"]
        const_cost += w * (-p)
    vals = []
    for j in range(len(bounds) - 1):
        lo, hi = bounds[j], bounds[j + 1]
        rep = lo if lo > -10**8 else (hi - 1)
        ind_term = 0.0
        for leg in legs:
            ind_term += leg["w"] * (1.0 if rep >= leg["k"] else 0.0)
        vals.append(ind_term + const_cost)
    return float(min(vals)) if vals else 0.0


def _net_slope(legs: list[dict[str, Any]]) -> float:
    return float(sum(leg["w"] for leg in legs))


def execute_bundle_leg_in(
    bundle: dict[str, Any],
    book_provider: Callable[[list[str]], dict[str, dict[str, Any]]],
    max_unhedged_loss: float,
    max_unhedged_ttl_s: float,
    slope_cap: float,
    poll_s: float = 0.2,
) -> dict[str, Any]:
    legs = bundle.get("legs") or []
    # Normalize leg fields: k, w, p
    normalized = []
    for leg in legs:
        qty = int(leg.get("qty") or 1)
        side = leg.get("side")
        w = qty if side == "BUY_YES" else -qty
        normalized.append(
            {
                "ticker": leg["ticker"],
                "k": int(leg["k"]),
                "w": float(w),
                "p": float(leg["limit_price"]),
                "delta": float(leg.get("delta") or 0.0),
                "side": side,
                "qty": qty,
            }
        )

    remaining = normalized[:]
    filled: list[dict[str, Any]] = []
    start = time.time()
    first_fill_ts: float | None = None
    max_loss_seen: float | None = None

    while remaining and (time.time() - start) <= max_unhedged_ttl_s:
        # choose hedge-critical: leg that most reduces max loss
        current_max_loss = _portfolio_max_loss(filled)
        best = None
        best_improve = None
        for leg in remaining:
            trial = filled + [leg]
            trial_loss = _portfolio_max_loss(trial)
            improve = trial_loss - current_max_loss
            if best is None or improve > best_improve or (improve == best_improve and abs(leg["delta"]) > abs(best["delta"])):
                best = leg
                best_improve = improve
        if best is None:
            break

        # risk gates for partial
        trial = filled + [best]
        if _portfolio_max_loss(trial) < -max_unhedged_loss:
            return {
                "status": "REJECTED",
                "reasons": ["MAX_UNHEDGED_LOSS"],
                "fills": filled,
                "unhedged_max_loss": max_loss_seen,
            }
        if abs(_net_slope(trial)) > slope_cap:
            return {
                "status": "REJECTED",
                "reasons": ["SLOPE_CAP"],
                "fills": filled,
                "unhedged_max_loss": max_loss_seen,
            }

        # attempt fill for chosen leg
        payload = book_provider([best["ticker"]]).get(best["ticker"])
        if payload is None:
            time.sleep(poll_s)
            continue
        top = _top_of_book(payload)
        limit_price = int(round(best["p"] * 100))
        if best["side"] == "BUY_YES":
            ask = top.get("yes_ask")
            ask_sz = top.get("yes_ask_sz") or 0
            if ask is None or ask > limit_price or ask_sz < best["qty"]:
                time.sleep(poll_s)
                continue
            best["fill_price"] = float(ask) / 100.0
            best["fill_qty"] = best["qty"]
            best["fill_ts"] = time.time()
            filled.append(best)
            if first_fill_ts is None:
                first_fill_ts = time.time()
            max_loss_seen = (
                current_max_loss if max_loss_seen is None else min(max_loss_seen, current_max_loss)
            )
            remaining = [leg for leg in remaining if leg is not best]
        elif best["side"] == "SELL_YES":
            bid = top.get("yes_bid")
            bid_sz = top.get("yes_bid_sz") or 0
            if bid is None or bid < limit_price or bid_sz < best["qty"]:
                time.sleep(poll_s)
                continue
            best["fill_price"] = float(bid) / 100.0
            best["fill_qty"] = best["qty"]
            best["fill_ts"] = time.time()
            filled.append(best)
            if first_fill_ts is None:
                first_fill_ts = time.time()
            max_loss_seen = (
                current_max_loss if max_loss_seen is None else min(max_loss_seen, current_max_loss)
            )
            remaining = [leg for leg in remaining if leg is not best]
        else:
            return {
                "status": "REJECTED",
                "reasons": ["invalid_side"],
                "fills": filled,
                "unhedged_max_loss": max_loss_seen,
            }

    if remaining:
        unhedged_time = 0.0
        if first_fill_ts is not None:
            unhedged_time = time.time() - first_fill_ts
        return {
            "status": "PARTIAL",
            "reasons": ["PARTIAL_TIMEOUT"],
            "fills": filled,
            "unhedged_time_s": unhedged_time,
            "unhedged_max_loss": max_loss_seen,
        }
    return {
        "status": "FILLED",
        "reasons": [],
        "fills": filled,
        "unhedged_time_s": 0.0 if first_fill_ts is None else time.time() - first_fill_ts,
        "unhedged_max_loss": max_loss_seen,
    }
