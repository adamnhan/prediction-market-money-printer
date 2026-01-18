"""Fee modeling utilities."""

from __future__ import annotations


def apply_fees(price: float, fee_bps: float) -> float:
    return price * (1 - fee_bps / 10_000)
