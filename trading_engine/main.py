# main.py

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
from .trading_engine import TradingEngine
from .strategy_config import StrategyConfig
from . import engine_state_store
from .trade_ledger import (
    fetch_trades,
    compute_summary_metrics,
    compute_equity_curve,
    compute_per_market_summary,
)
from src import platform_ops
import asyncio
from src.ws_client import (
    connect_and_listen,
    request_subscribe,
    WS_STATE,
    ACTIVE_SUBSCRIPTIONS,
)


from dotenv import load_dotenv
load_dotenv()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared TradingEngine instance
strategy_config = StrategyConfig.from_env()
engine = TradingEngine(strategy_config=strategy_config)

class AddEventRequest(BaseModel):
    ticker: str

class AddMarketRequest(BaseModel):
    event_ticker: str
    market_ticker: str

class OperatorFlagsRequest(BaseModel):
    pause_entries: Optional[bool] = None
    pause_all: Optional[bool] = None


class CloseAllRequest(BaseModel):
    confirm: bool = False


class DetachAllRequest(BaseModel):
    confirm: bool = False
    force: bool = False


@app.on_event("startup")
async def _startup_ws():
    # Load persisted control state (best-effort, non-fatal on errors)
    try:
        snapshot = engine_state_store.load_state()
        engine.apply_control_state_snapshot(snapshot)
        engine.log("[ENGINE_RESUMED] control state restored")
    except Exception:
        logging.exception("Failed to load engine control state; continuing with defaults.")

    # Start WS connection in the background (no subscriptions yet)
    asyncio.create_task(connect_and_listen(engine))


@app.on_event("shutdown")
async def _shutdown():
    # Persist control state snapshot (best-effort)
    try:
        engine_state_store.save_state(engine.get_control_state_snapshot())
    except Exception:
        logging.exception("Failed to persist engine control state on shutdown.")


@app.get("/state")
def get_state():
    return engine.get_state()

@app.get("/logs")
def get_logs():
    return {"logs": engine.get_logs()}

@app.get("/trades")
def list_trades(
    limit: int = 100,
    offset: int = 0,
    market_ticker: str | None = None,
    event_ticker: str | None = None,
):
    """
    Return paginated trade records from the ledger.
    """
    # Basic safety guard on limit to avoid huge responses
    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    trades = fetch_trades(
        limit=limit,
        offset=offset,
        market_ticker=market_ticker,
        event_ticker=event_ticker,
    )
    return {"trades": trades, "limit": limit, "offset": offset}

@app.get("/metrics")
def get_metrics():
    """
    Summary performance metrics derived from closed trades.
    """
    return compute_summary_metrics()

@app.get("/equity_curve")
def equity_curve():
    """
    Realized-PnL cumulative equity curve for charting.
    """
    return {"points": compute_equity_curve()}

@app.get("/metrics/markets")
def market_metrics(limit: int = 200, sort_by: str = "total_realized_pnl"):
    """
    Per-market performance summary (realized PnL, win rate).
    """
    limit = max(1, min(limit, 1000))
    return {
        "markets": compute_per_market_summary(limit=limit, sort_by=sort_by),
        "limit": limit,
        "sort_by": sort_by,
    }

@app.post("/events")
async def add_event(request: AddEventRequest):
    try:
        await engine.add_event(request.ticker)
    except Exception as e:
        logging.exception("Error in /events")
        raise HTTPException(status_code=500, detail=str(e))

    return engine.get_state()

@app.delete("/events/{ticker}")
def remove_event(ticker: str):
    engine.remove_event(ticker)
    return {"status": "ok", "removed": ticker}

@app.post("/markets")
async def add_market(req: AddMarketRequest):
    engine.add_market_for_event(req.event_ticker, req.market_ticker)

    # Start WS subscription immediately for this market
    request_subscribe(req.market_ticker.upper())

    return {"status": "ok", "subscribed": req.market_ticker.upper()}

@app.delete("/markets/{market_ticker}")
def remove_market(market_ticker: str):
    engine.remove_market(market_ticker)
    return {"status": "ok", "removed": market_ticker.upper()}


@app.get("/health")
def health():
    """
    Lightweight health check for ops/monitoring.
    Returns 200 when WS is healthy; 500 when disconnected or stale.
    """
    last_msg = WS_STATE.get("last_message_ts")
    stale_seconds = None
    if last_msg:
        try:
            last_dt = datetime.fromisoformat(last_msg)
            now = datetime.now(timezone.utc)
            stale_seconds = (now - last_dt).total_seconds()
        except Exception:
            stale_seconds = None

    ws_connected = bool(WS_STATE.get("connected"))
    ws_stale = bool(WS_STATE.get("stale")) or bool(
        stale_seconds is not None and stale_seconds > 60
    )

    payload = {
        "engine_running": True,
        "ws_connected": ws_connected,
        "ws_last_connect_ts": WS_STATE.get("last_connect_ts"),
        "ws_last_message_ts": WS_STATE.get("last_message_ts"),
        "ws_last_error": WS_STATE.get("last_error"),
        "ws_stale": ws_stale,
        "ws_stale_seconds": stale_seconds,
        "ws_subscriptions": len(ACTIVE_SUBSCRIPTIONS),
    }

    status_code = status.HTTP_200_OK
    if not ws_connected or ws_stale:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    return JSONResponse(content=payload, status_code=status_code)


@app.post("/operator/close_all")
def close_all_positions(req: CloseAllRequest):
    """
    Operator kill switch: close all open positions (paper).
    Requires confirm=true to avoid accidental calls.
    """
    if not req.confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to close all positions.")

    closed = engine.close_all_positions()

    try:
        engine_state_store.save_state(engine.get_control_state_snapshot())
    except Exception:
        logging.exception("Failed to persist control state after close_all.")

    return {"ok": True, "closed": closed}


@app.post("/operator/detach_all")
def detach_all_markets(req: DetachAllRequest):
    """
    Operator control: detach/unsubscribe all tracked markets.
    If force=true, unsubscribes even when open positions exist.
    """
    if not req.confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to detach all markets.")

    removed = engine.detach_all_markets(force=bool(req.force))

    try:
        engine_state_store.save_state(engine.get_control_state_snapshot())
    except Exception:
        logging.exception("Failed to persist control state after detach_all.")

    return {"ok": True, "removed": removed, "force": bool(req.force)}


@app.post("/operator/flags")
def set_operator_flags(req: OperatorFlagsRequest):
    """
    Update operator pause flags (entries/all). Only provided fields are updated.
    """
    flags = engine.set_operator_flags(
        pause_entries=req.pause_entries, pause_all=req.pause_all
    )

    return {"ok": True, "operator_flags": flags}


@app.get("/events/{event_ticker}/markets")
async def list_event_markets(event_ticker: str):
    event_ticker = event_ticker.upper()

    try:
        resp = await platform_ops.get_event(event_ticker)
    except Exception as e:
        # Kalshi / network error
        raise HTTPException(status_code=502, detail=f"Error fetching event: {e}")

    if not resp or not resp.get("ok"):
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch event {event_ticker}: {resp!r}",
        )

    data = resp.get("data") or {}

    # Adjust this based on actual shape of resp["data"]
    raw_markets = data.get("markets") or data.get("markets_info") or []

    markets = []
    for m in raw_markets:
        markets.append(
            {
                "market_ticker": m.get("ticker") or m.get("id"),
                "title": m.get("title"),
                "status": m.get("status") or m.get("lifecycle_status"),
            }
        )

    return {"event_ticker": event_ticker, "markets": markets}


@app.post("/close_position/{position_id}")
async def close_position(position_id: int):
    """
    Close a paper position by ID and return the updated position snapshot.
    """
    engine.close_position(position_id)
    pos = engine.positions.get(position_id)
    return {
        "ok": True,
        "position": vars(pos) if pos is not None else None,
    }

