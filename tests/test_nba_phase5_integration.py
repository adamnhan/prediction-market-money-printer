import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nba_engine.artifacts import load_artifacts
from nba_engine.phase1 import MarketState, TradeTick
from nba_engine.phase4 import CandleRow, PanicState, Position, SignalEngine, _exit_signal
from nba_engine.phase4 import MemoryPaperStore
from nba_engine.phase5 import entry_decision


def test_candle_builder_gap_and_activity():
    state = MarketState("MKT1")
    t0 = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    tick0 = TradeTick(market_ticker="MKT1", yes_price=0.6, volume=10, ts=t0)
    assert state.on_trade(tick0) == []

    tick1 = TradeTick(
        market_ticker="MKT1",
        yes_price=0.65,
        volume=5,
        ts=t0 + timedelta(minutes=2, seconds=5),
    )
    candles = state.on_trade(tick1)
    assert len(candles) == 2

    first, gap = candles
    assert first.start == t0
    assert first.open == 0.6
    assert first.close == 0.6
    assert first.volume == 10
    assert first.gap_flag == 0
    assert first.trade_active == 1

    assert gap.start == t0 + timedelta(minutes=1)
    assert gap.volume == 0
    assert gap.gap_flag == 1
    assert gap.trade_active == 0
    assert gap.active_last_3 == 1


def test_signal_engine_gating():
    artifacts = load_artifacts(Path("strategy_artifacts.json"))
    engine = SignalEngine(artifacts)
    ts = datetime(2025, 1, 1, 0, 3, tzinfo=timezone.utc)
    candle = CandleRow(
        rowid=1,
        market_ticker="MKT1",
        start_ts=ts,
        close=0.42,
        trade_active=1,
        ret_3=0.06,
        vol_10=0.9,
        vol_sum_5=80.0,
        gap_recent_5=0,
        p_open=0.3,
        p_base=0.32,
    )

    panic = engine.detect_panic(candle)
    assert panic is not None
    assert panic.direction == "UNDERDOG_UP"

    allowed, quality_score, reasons = engine.evaluate_entry(panic, candle, active_last_3=2)
    assert allowed
    assert reasons == []
    assert quality_score >= artifacts.quality_cutoff

    bad_candle = CandleRow(
        rowid=2,
        market_ticker="MKT1",
        start_ts=ts,
        close=0.42,
        trade_active=1,
        ret_3=0.06,
        vol_10=0.9,
        vol_sum_5=80.0,
        gap_recent_5=1,
        p_open=0.3,
        p_base=0.32,
    )
    allowed, _, reasons = engine.evaluate_entry(panic, bad_candle, active_last_3=2)
    assert not allowed
    assert "gap_recent_5" in reasons


def test_exit_cooldown_and_kill_switch():
    artifacts = load_artifacts(Path("strategy_artifacts.json"))
    engine = SignalEngine(artifacts)
    entry_ts = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    position = Position(
        id=1,
        market_ticker="MKT1",
        side="YES",
        entry_ts=entry_ts,
        entry_price=0.5,
    )

    reason, exit_price, pnl = _exit_signal(
        position,
        entry_ts + timedelta(minutes=1),
        0.54,
        None,
    )
    assert reason == "tp"
    assert math.isclose(exit_price, 0.54)
    assert pnl > 0

    candle = CandleRow(
        rowid=10,
        market_ticker="MKT1",
        start_ts=entry_ts + timedelta(minutes=1),
        close=0.42,
        trade_active=1,
        ret_3=0.06,
        vol_10=0.9,
        vol_sum_5=80.0,
        gap_recent_5=0,
        p_open=0.3,
        p_base=0.32,
    )
    panic = engine.detect_panic(candle)
    assert panic is not None

    store = MemoryPaperStore()
    trade_id = store.insert_entry(
        "MKT1",
        "YES",
        entry_ts,
        0.5,
        panic,
        1.2,
    )
    store.update_exit(trade_id, entry_ts, 0.52, "tp", 0.02)

    allowed, _, reasons = entry_decision(
        panic=panic,
        candle=candle,
        active_last_3=2,
        engine=engine,
        positions={},
        store=store,
        kill_switch=False,
    )
    assert not allowed
    assert "cooldown" in reasons

    clean_store = MemoryPaperStore()
    allowed, _, reasons = entry_decision(
        panic=panic,
        candle=candle,
        active_last_3=2,
        engine=engine,
        positions={},
        store=clean_store,
        kill_switch=True,
    )
    assert not allowed
    assert reasons == ["kill_switch"]
