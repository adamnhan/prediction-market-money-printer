from copy import deepcopy

import pytest

from trading_engine import engine_state_store as ess


def test_load_state_returns_defaults_when_missing(tmp_path, monkeypatch):
    # Use an isolated DB path so we don't touch real artifacts.
    monkeypatch.setattr(ess, "DB_PATH", tmp_path / "engine_state.sqlite")

    state = ess.load_state()
    assert state == deepcopy(ess.DEFAULT_STATE)


def test_save_and_load_round_trip_normalizes(tmp_path, monkeypatch):
    monkeypatch.setattr(ess, "DB_PATH", tmp_path / "engine_state.sqlite")

    snapshot = {
        "attached_markets": ["abc", "XYZ"],
        "retired_markets": {"m1", "M2", "m1"},
        "operator_flags": {"pause_entries": True},
        "strategy_config": {"k": "v"},
        "ignored_field": 123,
    }

    ess.save_state(snapshot)
    restored = ess.load_state()

    assert restored["attached_markets"] == ["ABC", "XYZ"]
    assert restored["retired_markets"] == ["M1", "M2"]
    assert restored["operator_flags"] == {"pause_entries": True, "pause_all": False}
    assert restored["strategy_config"] == {"k": "v"}


def test_positions_and_capital_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(ess, "DB_PATH", tmp_path / "engine_state.sqlite")

    snapshot = {
        "positions": [
            {
                "id": 1,
                "event_ticker": "evt",
                "market_ticker": "mkt",
                "side": "no",
                "qty": 2,
                "entry_price": 0.45,
                "current_price": 0.4,
                "status": "open",
                "entry_ts": "2024-01-01T00:00:00",
                "exit_ts": None,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.1,
            }
        ],
        "capital": {"total": 5000, "used": 90.5},
    }

    ess.save_state(snapshot)
    restored = ess.load_state()

    assert restored["positions"][0]["market_ticker"] == "MKT"
    assert restored["positions"][0]["side"] == "NO"
    assert restored["capital"] == {"total": 5000.0, "used": 90.5}
    assert "cooldown_until" in restored
