# trading_engine/capital.py

from dataclasses import dataclass
from typing import Dict, List

from trading_engine.models import Position


@dataclass
class CapitalState:
    """
    Tracks capital usage, total capital, and provides convenience
    methods for computing available capital, realized/unrealized PnL, etc.
    """

    total: float = 10000.0     # starting capital
    used: float = 0.0          # reserved capital for open positions

    def reserve(self, entry_price: float, qty: int) -> None:
        """
        Increase used capital when a position is opened.
        """
        self.used += entry_price * qty

    def release(self, entry_price: float, qty: int) -> None:
        """
        Release capital when a position is closed.
        """
        self.used -= entry_price * qty
        if self.used < 0:
            self.used = 0.0

    def compute_totals(self, positions: List[Position]) -> Dict[str, float]:
        """
        Computes total realized PnL, unrealized PnL, and capital at risk.
        """
        realized = sum(p.realized_pnl for p in positions)
        unrealized = sum(p.unrealized_pnl for p in positions if p.status == "open")
        capital_at_risk = sum(p.entry_price * p.qty for p in positions if p.status == "open")

        available = self.total - self.used + realized

        return {
            "total": self.total,
            "used": self.used,
            "available": available,
            "capital_at_risk": capital_at_risk,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
        }
