from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ShadowMMConfig:
    markets_file: str = "config/markets_nba.txt"
    log_dir: str = "market_maker/logs"
    quote_cadence_s: float = 1.0
    summary_every_s: float = 60.0
    health_every_s: float = 10.0
    quote_size: int = 1
    quote_ttl_s: float = 10.0
    mark_price_mode: str = "bid"
    base_spread_cents: float = 6.0
    k_cents_per_contract: float = 0.05
    itarget: float = 10.0
    imax: int = 25
    min_spread_cents: float = 4.0
    staleness_s: float = 2.0
    vol_move_cents: float = 3.0
    vol_window_s: float = 2.0
    halt_s: float = 10.0
    fee_per_contract_cents: float = 0.0
    max_hold_s: float = 3600.0
    force_unwind_on_max_hold: bool = False
    stop_loss_cents: float = 500.0
    stop_loss_halt_s: float = 600.0
    force_unwind_on_stop_loss: bool = False
    touch_ms: int = 600
    p_fill_base: float = 0.25
    p_fill_spread_bonus_per_cent: float = 0.03
    p_fill_size_bonus_per_contract: float = 0.01
    p_fill_vol_penalty_per_cent: float = 0.05
    top_of_book_tick_cents: float = 1.0
    max_fills_per_minute: int = 20
    max_fills_per_quote_per_s: float = 1.0


_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")
_INT_RE = re.compile(r"^-?\d+$")


def _parse_value(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    if _INT_RE.match(value):
        return int(value)
    if _FLOAT_RE.match(value):
        return float(value)
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_value(value)
    return data


def load_config(path: str | Path) -> ShadowMMConfig:
    values = load_simple_yaml(path)
    return ShadowMMConfig(
        markets_file=values.get("markets_file", ShadowMMConfig.markets_file),
        log_dir=values.get("log_dir", ShadowMMConfig.log_dir),
        quote_cadence_s=float(values.get("quote_cadence_s", ShadowMMConfig.quote_cadence_s)),
        summary_every_s=float(values.get("summary_every_s", ShadowMMConfig.summary_every_s)),
        health_every_s=float(values.get("health_every_s", ShadowMMConfig.health_every_s)),
        quote_size=int(values.get("quote_size", ShadowMMConfig.quote_size)),
        quote_ttl_s=float(values.get("quote_ttl_s", ShadowMMConfig.quote_ttl_s)),
        mark_price_mode=str(values.get("mark_price_mode", ShadowMMConfig.mark_price_mode)),
        base_spread_cents=float(values.get("base_spread_cents", ShadowMMConfig.base_spread_cents)),
        k_cents_per_contract=float(values.get("k_cents_per_contract", ShadowMMConfig.k_cents_per_contract)),
        itarget=float(values.get("itarget", ShadowMMConfig.itarget)),
        imax=int(values.get("imax", ShadowMMConfig.imax)),
        min_spread_cents=float(values.get("min_spread_cents", ShadowMMConfig.min_spread_cents)),
        staleness_s=float(values.get("staleness_s", ShadowMMConfig.staleness_s)),
        vol_move_cents=float(values.get("vol_move_cents", ShadowMMConfig.vol_move_cents)),
        vol_window_s=float(values.get("vol_window_s", ShadowMMConfig.vol_window_s)),
        halt_s=float(values.get("halt_s", ShadowMMConfig.halt_s)),
        fee_per_contract_cents=float(
            values.get("fee_per_contract_cents", ShadowMMConfig.fee_per_contract_cents)
        ),
        max_hold_s=float(values.get("max_hold_s", ShadowMMConfig.max_hold_s)),
        force_unwind_on_max_hold=bool(
            values.get("force_unwind_on_max_hold", ShadowMMConfig.force_unwind_on_max_hold)
        ),
        stop_loss_cents=float(values.get("stop_loss_cents", ShadowMMConfig.stop_loss_cents)),
        stop_loss_halt_s=float(values.get("stop_loss_halt_s", ShadowMMConfig.stop_loss_halt_s)),
        force_unwind_on_stop_loss=bool(
            values.get("force_unwind_on_stop_loss", ShadowMMConfig.force_unwind_on_stop_loss)
        ),
        touch_ms=int(values.get("touch_ms", ShadowMMConfig.touch_ms)),
        p_fill_base=float(values.get("p_fill_base", ShadowMMConfig.p_fill_base)),
        p_fill_spread_bonus_per_cent=float(
            values.get("p_fill_spread_bonus_per_cent", ShadowMMConfig.p_fill_spread_bonus_per_cent)
        ),
        p_fill_size_bonus_per_contract=float(
            values.get("p_fill_size_bonus_per_contract", ShadowMMConfig.p_fill_size_bonus_per_contract)
        ),
        p_fill_vol_penalty_per_cent=float(
            values.get("p_fill_vol_penalty_per_cent", ShadowMMConfig.p_fill_vol_penalty_per_cent)
        ),
        top_of_book_tick_cents=float(
            values.get("top_of_book_tick_cents", ShadowMMConfig.top_of_book_tick_cents)
        ),
        max_fills_per_minute=int(
            values.get("max_fills_per_minute", ShadowMMConfig.max_fills_per_minute)
        ),
        max_fills_per_quote_per_s=float(
            values.get("max_fills_per_quote_per_s", ShadowMMConfig.max_fills_per_quote_per_s)
        ),
    )


def load_markets(path: str | Path) -> list[str]:
    markets: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        markets.append(line.upper())
    return markets
