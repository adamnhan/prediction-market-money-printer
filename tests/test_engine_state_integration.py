import pytest

from trading_engine.trading_engine import TradingEngine
from trading_engine import engine_state_store as ess


def test_engine_control_state_round_trip(tmp_path, monkeypatch):
    # Isolate persistence to a temp DB
    monkeypatch.setattr(ess, "DB_PATH", tmp_path / "engine_state.sqlite")

    # Original engine: set flags and markets
    eng1 = TradingEngine()
    eng1.pause_entries = True
    eng1.pause_all = False
    eng1.add_market_for_event("EVT1", "MKT1")
    eng1.add_market_for_event("EVT2", "mkt2")
    eng1._retired_markets.add("old1")

    ess.save_state(eng1.get_control_state_snapshot())

    # New engine should restore snapshot
    eng2 = TradingEngine()
    snapshot = ess.load_state()
    eng2.apply_control_state_snapshot(snapshot)

    # Operator flags restored
    assert eng2.pause_entries is True
    assert eng2.pause_all is False

    # Attached markets restored (uppercased keys)
    assert set(eng2.markets.keys()) == {"MKT1", "MKT2"}

    # Retired markets restored/uppercased
    assert eng2._retired_markets == {"OLD1"}

    # Strategy config preserved
    assert eng2.strategy_config == eng1.strategy_config
