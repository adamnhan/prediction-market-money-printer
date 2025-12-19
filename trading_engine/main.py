# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .trading_engine import TradingEngine
from pydantic import BaseModel
import logging
from src import platform_ops
import asyncio
from src.ws_client import connect_and_listen, request_subscribe


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
engine = TradingEngine()

class AddEventRequest(BaseModel):
    ticker: str

class AddMarketRequest(BaseModel):
    event_ticker: str
    market_ticker: str


@app.on_event("startup")
async def _startup_ws():
    # Start WS connection in the background (no subscriptions yet)
    asyncio.create_task(connect_and_listen(engine))


@app.get("/state")
def get_state():
    return engine.get_state()

@app.get("/logs")
def get_logs():
    return {"logs": engine.get_logs()}

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


@app.post("/markets/{market_ticker}/refresh_and_trade")
async def refresh_and_trade(market_ticker: str):
    mt = market_ticker.upper()
    print(f"=== refresh_and_trade called for {mt} ===")

    # 1) Update metadata from Kalshi
    await engine.update_market_metadata(mt)

    # 2) Decide whether to open a paper NO position
    engine.maybe_open_no_position_for_market(mt)

    # 3) Return updated state
    return engine.get_state()

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

