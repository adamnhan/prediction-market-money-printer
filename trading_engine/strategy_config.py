# trading_engine/strategy_config.py

import os
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class StrategyConfig:
    """
    Static strategy configuration for the trading engine.

    NOTE: Phase 4 - this is just a data holder.
    It is not yet used to change any trading behavior.
    """

    # Position / capital limits
    max_open_positions: int = 1000
    max_capital_per_position_pct: float = 0.10   # 10% of total
    max_capital_per_market_pct: float = 0.10     # 20% of total
    max_total_exposure_pct: float = 0.50         # 50% of total across all open positions
    max_no_entry_price: Optional[float] = None    # allow NO entries at any price; refine via notebook grid search

# Optional exit rules (PnL-based)
    take_profit_pct: Optional[float] = 0.2      # hold to settlement by default; notebook can tune
    stop_loss_pct: Optional[float] = -0.15     # stop-loss off by default; notebook can tune

# Optional time-based exit (in seconds)
    max_hold_seconds: Optional[int] = None        # no time-based exit; settlement-driven

    # Circuit breakers / operator safety
    daily_loss_limit: Optional[float] = None      # e.g. 500 => stop new entries after -$500 realized today
    max_drawdown: Optional[float] = None          # max allowed peak-to-trough realized drawdown
    max_trades_per_day: Optional[int] = None      # block new entries after N closed trades today
    cooldown_minutes_after_stop: Optional[int] = None  # optional cooldown after a stop-loss exit

    @staticmethod
    def from_env(env: Callable[[str], Optional[str]] = os.getenv) -> "StrategyConfig":
        """
        Build a StrategyConfig from environment variables.
        Falls back to dataclass defaults when vars are missing or invalid.
        """
        def _opt_float(name: str, default: Optional[float]) -> Optional[float]:
            raw = env(name)
            if raw is None or raw == "":
                return default
            try:
                return float(raw)
            except Exception:
                return default

        def _opt_int(name: str, default: Optional[int]) -> Optional[int]:
            raw = env(name)
            if raw is None or raw == "":
                return default
            try:
                return int(raw)
            except Exception:
                return default

        return StrategyConfig(
            max_open_positions=_opt_int("STRATEGY_MAX_OPEN_POSITIONS", StrategyConfig.max_open_positions),
            max_capital_per_position_pct=_opt_float("STRATEGY_MAX_CAPITAL_PER_POSITION_PCT", StrategyConfig.max_capital_per_position_pct),
            max_capital_per_market_pct=_opt_float("STRATEGY_MAX_CAPITAL_PER_MARKET_PCT", StrategyConfig.max_capital_per_market_pct),
            max_total_exposure_pct=_opt_float("STRATEGY_MAX_TOTAL_EXPOSURE_PCT", StrategyConfig.max_total_exposure_pct),
            max_no_entry_price=_opt_float("STRATEGY_MAX_NO_ENTRY_PRICE", StrategyConfig.max_no_entry_price),
            take_profit_pct=_opt_float("STRATEGY_TAKE_PROFIT_PCT", StrategyConfig.take_profit_pct),
            stop_loss_pct=_opt_float("STRATEGY_STOP_LOSS_PCT", StrategyConfig.stop_loss_pct),
            max_hold_seconds=_opt_int("STRATEGY_MAX_HOLD_SECONDS", StrategyConfig.max_hold_seconds),
            daily_loss_limit=_opt_float("STRATEGY_DAILY_LOSS_LIMIT", StrategyConfig.daily_loss_limit),
            max_drawdown=_opt_float("STRATEGY_MAX_DRAWDOWN", StrategyConfig.max_drawdown),
            max_trades_per_day=_opt_int("STRATEGY_MAX_TRADES_PER_DAY", StrategyConfig.max_trades_per_day),
            cooldown_minutes_after_stop=_opt_int("STRATEGY_COOLDOWN_MINUTES_AFTER_STOP", StrategyConfig.cooldown_minutes_after_stop),
        )
