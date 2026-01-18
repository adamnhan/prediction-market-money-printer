"""Shadow execution simulator."""

from __future__ import annotations

from cross_venue_arb.arb.signal import ArbSignal


def simulate(signal: ArbSignal) -> dict:
    return {"market_id": signal.market_id, "edge_bps": signal.edge_bps}
