# trading_engine/strategy_config.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class StrategyConfig:
    """
    Static strategy configuration for the trading engine.

    NOTE: Phase 4 – this is just a data holder.
    It is not yet used to change any trading behavior.
    """

    # Position / capital limits
    max_open_positions: int = 10
    max_capital_per_position_pct: float = 0.10   # 10% of total
    max_capital_per_market_pct: float = 0.20     # 20% of total
    max_no_entry_price: Optional[float] = 0.5 # e.g. 0.30 means NO at 30¢ or less
   
    # Optional exit rules (PnL-based)
    take_profit_pct: Optional[float] = 0.1      # e.g. 0.20 for +20%
    stop_loss_pct: Optional[float] = -0.05        # e.g. -0.10 for -10%

    # Optional time-based exit (in seconds)
    max_hold_seconds: Optional[int] = 10000000
