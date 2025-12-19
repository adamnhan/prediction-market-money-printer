# trading_engine/strategy_helpers.py

from datetime import datetime
from typing import Optional

from trading_engine.models import Position
from trading_engine.strategy_config import StrategyConfig


def can_open_new_position(
    *,
    strategy_config: StrategyConfig,
    positions: list[Position],
    market_ticker: str,
    qty: int,
    entry_price: float,
    total_capital: float,
) -> bool:
    """
    Pure helper that checks basic strategy limits for opening a new position.
    """
    if total_capital <= 0:
        return False

    open_positions = [p for p in positions if p.status == "open"]

    # 1) Max number of open positions
    if len(open_positions) >= strategy_config.max_open_positions:
        return False

    # 2) Per-position capital limit
    position_cost = qty * entry_price
    max_per_pos = total_capital * strategy_config.max_capital_per_position_pct
    if position_cost > max_per_pos:
        return False

    # 3) Per-market capital limit
    market_ticker = market_ticker.upper()
    capital_in_market = sum(
        p.entry_price * p.qty
        for p in open_positions
        if p.market_ticker == market_ticker
    )
    max_per_market = total_capital * strategy_config.max_capital_per_market_pct
    if capital_in_market + position_cost > max_per_market:
        return False

    return True


def pnl_exit_reason(pos: Position, cfg: StrategyConfig) -> Optional[str]:
    """
    Returns:
        "pnl_take_profit"
        "pnl_stop_loss"
        or None
    """
    pnl = pos.unrealized_pnl

    # Take-profit
    if cfg.take_profit_pct is not None:
        if pnl >= cfg.take_profit_pct * (pos.entry_price * pos.qty):
            return "pnl_take_profit"

    # Stop-loss
    if cfg.stop_loss_pct is not None:
        if pnl <= cfg.stop_loss_pct * (pos.entry_price * pos.qty):
            return "pnl_stop_loss"

    return None


def time_exit_reason(pos: Position, cfg: StrategyConfig) -> Optional[str]:
    """
    Returns "time_expired" or None.
    """
    max_hold = cfg.max_hold_seconds
    if max_hold is None:
        return None

    if pos.entry_ts is None:
        return None

    age = (datetime.utcnow() - pos.entry_ts).total_seconds()
    if age >= max_hold:
        return "time_expired"

    return None


def strategy_exit_reason(pos: Position, cfg: StrategyConfig) -> Optional[str]:
    """
    Unified strategy exit decision:
        - pnl_take_profit
        - pnl_stop_loss
        - time_expired
    """
    # 1) PnL rules
    pnl_reason = pnl_exit_reason(pos, cfg)
    if pnl_reason is not None:
        return pnl_reason

    # 2) Time-based rules
    t_reason = time_exit_reason(pos, cfg)
    if t_reason is not None:
        return t_reason

    return None
