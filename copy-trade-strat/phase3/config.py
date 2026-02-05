from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Phase3Config:
    target_wallet: str | None = None
    copy_types: list[str] = field(default_factory=lambda: ["BUY"])
    categories_allowlist: list[str] = field(default_factory=list)
    max_staleness_seconds: int = 8
    max_slippage_cents: int = 1
    min_my_trade_size_notional_usd: float = 0.50
    max_my_trade_size_notional_usd: float = 1.50
    max_open_notional_total_usd: float = 4.0
    max_open_notional_per_market_usd: float = 2.0
    max_daily_notional_usd: float = 6.0
    max_trades_per_hour: int = 6
    fixed_copy_size_shares: float = 1.0


def _parse_value(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_phase0_config(path: str) -> Phase3Config:
    cfg = Phase3Config()
    current_section = None
    list_key = None
    lines = Path(path).read_text().splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" "):
            if ":" in line:
                current_section = line.split(":", 1)[0].strip()
                list_key = None
            continue

        if stripped.startswith("- "):
            if current_section == "copy_scope" and list_key == "copy_types":
                cfg.copy_types.append(stripped[2:].strip().strip('"').strip("'").upper())
            if current_section == "copy_scope" and list_key == "categories_allowlist":
                cfg.categories_allowlist.append(stripped[2:].strip().strip('"').strip("'"))
            continue

        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            is_list = value.strip() == ""
            list_key = key if is_list else None
            value_parsed = _parse_value(value)

            if current_section == "identity" and key == "target_wallet":
                cfg.target_wallet = value_parsed
            if current_section == "copy_scope" and key == "copy_types" and not is_list:
                cfg.copy_types = [str(value_parsed).upper()]
            if current_section == "copy_scope" and key == "categories_allowlist" and not is_list:
                cfg.categories_allowlist = []
            if current_section == "staleness" and key == "max_staleness_seconds":
                cfg.max_staleness_seconds = int(value_parsed)
            if current_section == "price_protection" and key == "max_slippage_cents":
                cfg.max_slippage_cents = int(value_parsed)
            if current_section == "sizing" and key == "min_my_trade_size_notional_usd":
                cfg.min_my_trade_size_notional_usd = float(value_parsed)
            if current_section == "sizing" and key == "max_my_trade_size_notional_usd":
                cfg.max_my_trade_size_notional_usd = float(value_parsed)
            if current_section == "exposure_caps" and key == "max_open_notional_total_usd":
                cfg.max_open_notional_total_usd = float(value_parsed)
            if current_section == "exposure_caps" and key == "max_open_notional_per_market_usd":
                cfg.max_open_notional_per_market_usd = float(value_parsed)
            if current_section == "exposure_caps" and key == "max_daily_notional_usd":
                cfg.max_daily_notional_usd = float(value_parsed)
            if current_section == "exposure_caps" and key == "max_trades_per_hour":
                cfg.max_trades_per_hour = int(value_parsed)

    cfg.copy_types = [x for x in cfg.copy_types if x]
    return cfg
