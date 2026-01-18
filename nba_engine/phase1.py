from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import math
import os
import sqlite3
import statistics
import time
import random
import uuid
from typing import Iterable
from pathlib import Path
from urllib.parse import urlparse

import requests
import websockets
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from .config import Config, load_config


logger = logging.getLogger("nba_phase1")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[PHASE1] %(asctime)s %(message)s"))
    logger.addHandler(handler)


KALSHI_L2_DB_PATH = os.getenv("KALSHI_L2_DB_PATH", "data/l2_orderbook.sqlite")
KALSHI_L2_LOGGING = os.getenv("KALSHI_L2_LOGGING", "1") == "1"
KALSHI_L2_CHECKPOINT_INTERVAL_S = int(os.getenv("KALSHI_L2_CHECKPOINT_INTERVAL_S", "60"))
KALSHI_L2_LOG_EVERY_N = int(os.getenv("KALSHI_L2_LOG_EVERY_N", "1000"))
KALSHI_MARKET_REFRESH_S = int(os.getenv("KALSHI_MARKET_REFRESH_S", "86400"))


@dataclass
class TradeTick:
    market_ticker: str
    yes_price: float
    volume: float
    ts: datetime


@dataclass
class Candle:
    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    gap_flag: int
    trade_active: int
    ret_1: float
    ret_3: float
    ret_5: float
    vol_10: float
    vol_sum_5: float
    active_last_3: int
    gap_recent_5: int


class CandleStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                market_ticker TEXT NOT NULL,
                start_ts TEXT NOT NULL,
                close REAL,
                volume REAL,
                gap_flag INTEGER,
                trade_active INTEGER,
                ret_1 REAL,
                ret_3 REAL,
                ret_5 REAL,
                vol_10 REAL,
                vol_sum_5 REAL,
                active_last_3 INTEGER,
                gap_recent_5 INTEGER,
                p_open REAL,
                p_base REAL
            )
            """
        )
        self.conn.commit()

    def insert(self, ticker: str, candle: Candle, p_open: float, p_base: float) -> None:
        self.conn.execute(
            """
            INSERT INTO candles (
                market_ticker,
                start_ts,
                close,
                volume,
                gap_flag,
                trade_active,
                ret_1,
                ret_3,
                ret_5,
                vol_10,
                vol_sum_5,
                active_last_3,
                gap_recent_5,
                p_open,
                p_base
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                candle.start.isoformat(),
                candle.close,
                candle.volume,
                candle.gap_flag,
                candle.trade_active,
                candle.ret_1,
                candle.ret_3,
                candle.ret_5,
                candle.vol_10,
                candle.vol_sum_5,
                candle.active_last_3,
                candle.gap_recent_5,
                p_open,
                p_base,
            ),
        )
        self.conn.commit()

class L2Recorder:
    def __init__(self, path: Path, checkpoint_interval_s: int, log_every_n: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()
        self.session_id = uuid.uuid4().hex
        self.recv_idx = 0
        self.msg_count = 0
        self.log_every_n = max(1, log_every_n)
        self.checkpoint_interval_ms = max(1, checkpoint_interval_s) * 1000
        self.last_checkpoint_ms = 0

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS l2_messages (
                id INTEGER PRIMARY KEY,
                ts_utc_ms INTEGER NOT NULL,
                market_ticker TEXT NOT NULL,
                channel TEXT NOT NULL,
                seq INTEGER NULL,
                payload_json TEXT NOT NULL,
                session_id TEXT NOT NULL,
                recv_idx INTEGER NOT NULL
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_l2_messages_ticker_ts ON l2_messages (market_ticker, ts_utc_ms);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_l2_messages_ticker_seq ON l2_messages (market_ticker, seq);"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS l2_checkpoints (
                id INTEGER PRIMARY KEY,
                ts_utc_ms INTEGER NOT NULL,
                market_ticker TEXT NOT NULL,
                bids_json TEXT NOT NULL,
                asks_json TEXT NOT NULL,
                top_bid REAL,
                top_ask REAL,
                levels_bid INTEGER NOT NULL,
                levels_ask INTEGER NOT NULL
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_l2_checkpoints_ticker_ts ON l2_checkpoints (market_ticker, ts_utc_ms);"
        )
        self.conn.commit()

    def log_message(self, market_ticker: str, channel: str, payload: dict, seq: int | None) -> None:
        self.recv_idx += 1
        self.msg_count += 1
        ts_utc_ms = int(time.time() * 1000)
        payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        self.conn.execute(
            """
            INSERT INTO l2_messages (
                ts_utc_ms, market_ticker, channel, seq, payload_json, session_id, recv_idx
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts_utc_ms,
                market_ticker,
                channel,
                seq,
                payload_json,
                self.session_id,
                self.recv_idx,
            ),
        )
        self.conn.commit()
        if self.msg_count % self.log_every_n == 0:
            logger.info(
                "l2_log count=%d last_ticker=%s last_type=%s last_seq=%s",
                self.msg_count,
                market_ticker,
                channel,
                seq,
            )

    def maybe_checkpoint(self, books: dict[str, dict[str, dict[int, int]]]) -> None:
        now_ms = int(time.time() * 1000)
        if now_ms - self.last_checkpoint_ms < self.checkpoint_interval_ms:
            return
        for market_ticker, book in books.items():
            bids = sorted(
                [(int(p), int(sz)) for p, sz in book.get("yes", {}).items() if sz and sz > 0],
                reverse=True,
            )
            asks = sorted(
                [(int(p), int(sz)) for p, sz in book.get("no", {}).items() if sz and sz > 0]
            )
            top_bid = bids[0][0] if bids else None
            top_ask = asks[0][0] if asks else None
            self.conn.execute(
                """
                INSERT INTO l2_checkpoints (
                    ts_utc_ms, market_ticker, bids_json, asks_json, top_bid, top_ask, levels_bid, levels_ask
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_ms,
                    market_ticker,
                    json.dumps(bids, separators=(",", ":"), ensure_ascii=True),
                    json.dumps(asks, separators=(",", ":"), ensure_ascii=True),
                    top_bid,
                    top_ask,
                    len(bids),
                    len(asks),
                ),
            )
        self.conn.commit()
        self.last_checkpoint_ms = now_ms
        logger.info("l2_checkpoint ts_utc_ms=%d markets=%d", now_ms, len(books))

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

class MarketState:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.current_minute: datetime | None = None
        self.current_open: float | None = None
        self.current_high: float | None = None
        self.current_low: float | None = None
        self.current_close: float | None = None
        self.current_volume: float = 0.0
        self.last_candle_minute: datetime | None = None
        self.candles: list[Candle] = []
        self.price_samples: list[float] = []
        self.p_filled_history: list[float] = []
        self.last_p_filled: float = float("nan")
        self.p_open: float | None = None
        self.p_base: float | None = None

    def _set_price_sample(self, price: float) -> None:
        if math.isnan(price):
            return
        if self.p_open is not None:
            return
        self.price_samples.append(price)
        if len(self.price_samples) >= 5:
            first_five = self.price_samples[:5]
            self.p_open = float(statistics.median(first_five))
            self.p_base = self.p_open

    def _compute_returns(self, closes: list[float], lookback: int) -> float:
        if len(closes) <= lookback:
            return float("nan")
        current = closes[-1]
        prev = closes[-1 - lookback]
        if math.isnan(current) or math.isnan(prev) or prev == 0:
            return float("nan")
        return (current / prev) - 1.0

    def _compute_volatility(self, returns: Iterable[float]) -> float:
        values = [val for val in returns if not math.isnan(val)]
        if len(values) < 2:
            return float("nan")
        return float(statistics.pstdev(values))

    def _append_candle(
        self,
        minute_start: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        gap_flag: int,
    ) -> Candle:
        trade_active = 1 if volume > 0 else 0
        if math.isnan(close):
            p_filled = self.last_p_filled
        else:
            p_filled = close
        self.last_p_filled = p_filled
        self.p_filled_history.append(p_filled)
        closes = self.p_filled_history[:]
        volumes = [c.volume for c in self.candles] + [volume]
        gap_flags = [c.gap_flag for c in self.candles] + [gap_flag]
        trade_flags = [c.trade_active for c in self.candles] + [trade_active]

        ret_1 = self._compute_returns(closes, 1)
        ret_3 = self._compute_returns(closes, 3)
        ret_5 = self._compute_returns(closes, 5)
        recent_returns = [self._compute_returns(closes[: i + 1], 1) for i in range(len(closes))]
        vol_10 = self._compute_volatility(recent_returns[-10:])
        vol_sum_5 = float(sum(volumes[-5:]))
        active_last_3 = 1 if any(flag == 1 for flag in trade_flags[-3:]) else 0
        gap_recent_5 = 1 if any(flag == 1 for flag in gap_flags[-5:]) else 0

        candle = Candle(
            start=minute_start,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            gap_flag=gap_flag,
            trade_active=trade_active,
            ret_1=ret_1,
            ret_3=ret_3,
            ret_5=ret_5,
            vol_10=vol_10,
            vol_sum_5=vol_sum_5,
            active_last_3=active_last_3,
            gap_recent_5=gap_recent_5,
        )
        self.candles.append(candle)
        self.last_candle_minute = minute_start
        return candle

    def _finalize_current(self) -> Candle | None:
        if self.current_minute is None:
            return None
        open_price = self.current_open if self.current_open is not None else float("nan")
        high = self.current_high if self.current_high is not None else float("nan")
        low = self.current_low if self.current_low is not None else float("nan")
        close = self.current_close if self.current_close is not None else float("nan")
        volume = self.current_volume
        gap_flag = 1 if volume == 0 else 0
        candle = self._append_candle(
            minute_start=self.current_minute,
            open_price=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            gap_flag=gap_flag,
        )
        self.current_minute = None
        self.current_open = None
        self.current_high = None
        self.current_low = None
        self.current_close = None
        self.current_volume = 0.0
        return candle

    def _append_gap_candle(self, minute_start: datetime) -> Candle:
        return self._append_candle(
            minute_start=minute_start,
            open_price=float("nan"),
            high=float("nan"),
            low=float("nan"),
            close=float("nan"),
            volume=0.0,
            gap_flag=1,
        )

    def flush_to(self, target_minute: datetime) -> list[Candle]:
        candles: list[Candle] = []
        if self.current_minute is not None and self.current_minute < target_minute:
            finalized = self._finalize_current()
            if finalized:
                candles.append(finalized)

        last_minute = self.last_candle_minute
        if last_minute is None:
            return candles

        next_minute = last_minute + timedelta(minutes=1)
        while next_minute < target_minute:
            candles.append(self._append_gap_candle(next_minute))
            next_minute += timedelta(minutes=1)
        return candles

    def on_trade(self, tick: TradeTick) -> list[Candle]:
        candles: list[Candle] = []
        minute_start = tick.ts.replace(second=0, microsecond=0)
        if self.current_minute is None:
            if self.last_candle_minute is not None:
                candles.extend(self.flush_to(minute_start))
            self.current_minute = minute_start
            self.current_open = tick.yes_price
            self.current_high = tick.yes_price
            self.current_low = tick.yes_price
            self.current_close = tick.yes_price
            self.current_volume = tick.volume
            self._set_price_sample(tick.yes_price)
            return candles

        if minute_start == self.current_minute:
            self.current_high = max(self.current_high or tick.yes_price, tick.yes_price)
            self.current_low = min(self.current_low or tick.yes_price, tick.yes_price)
            self.current_close = tick.yes_price
            self.current_volume += tick.volume
            self._set_price_sample(tick.yes_price)
            return candles

        if minute_start > self.current_minute:
            candles.extend(self.flush_to(minute_start))
            self.current_minute = minute_start
            self.current_open = tick.yes_price
            self.current_high = tick.yes_price
            self.current_low = tick.yes_price
            self.current_close = tick.yes_price
            self.current_volume = tick.volume
            self._set_price_sample(tick.yes_price)
        return candles


def rest_base_from_ws(ws_url: str) -> str:
    parsed = urlparse(ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    path = parsed.path
    if "/ws/" in path:
        path = path.replace("/ws/", "/")
    if path.endswith("/ws"):
        path = path[: -len("/ws")]
    return f"{scheme}://{parsed.netloc}{path}"


def _backoff_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 20.0) -> float:
    return min(max_delay, base_delay * (2**attempt))


def fetch_markets(rest_url: str, series_ticker: str) -> list[dict]:
    markets: list[dict] = []
    cursor: str | None = None
    while True:
        params = {"series_ticker": series_ticker, "status": "open", "limit": 100}
        if cursor:
            params["cursor"] = cursor
        logger.info("fetching markets %s/markets params=%s", rest_url, params)
        response = None
        for attempt in range(5):
            try:
                response = requests.get(f"{rest_url}/markets", params=params, timeout=20)
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt >= 4:
                    raise
                delay = _backoff_delay(attempt)
                delay += delay * 0.2 * random.random()
                logger.info(
                    "market_fetch_retry series=%s attempt=%d delay=%.2f error=%s",
                    series_ticker,
                    attempt + 1,
                    delay,
                    exc,
                )
                time.sleep(delay)
        payload = response.json() if response is not None else {}
        batch = payload.get("markets") or []
        if isinstance(batch, list):
            markets.extend(batch)
        cursor = payload.get("cursor") or payload.get("next_cursor")
        if not cursor:
            break
    return markets


def is_open_market(market: dict) -> bool:
    return str(market.get("status") or "").lower() == "active"


def select_market_tickers(markets: list[dict]) -> list[str]:
    tickers: list[str] = []
    for market in markets:
        if not is_open_market(market):
            continue
        ticker = market.get("ticker") or market.get("market_ticker")
        if not ticker or ticker in tickers:
            continue
        tickers.append(str(ticker).upper())
    return tickers


def load_private_key(path: str) -> object:
    raw = Path(path).read_text(encoding="utf-8")
    return serialization.load_pem_private_key(
        raw.encode("utf-8"),
        password=None,
        backend=default_backend(),
    )


def build_ws_auth_headers(config: Config, private_key: object) -> dict[str, str]:
    now = datetime.now(timezone.utc)
    timestamp_ms = int(now.timestamp() * 1000)
    timestamp_str = str(timestamp_ms)
    method = "GET"
    parsed = urlparse(config.kalshi_ws_url)
    path = parsed.path or "/trade-api/ws/v2"
    message = f"{timestamp_str}{method}{path}".encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    signature_b64 = base64.b64encode(signature).decode("utf-8")
    return {
        "KALSHI-ACCESS-KEY": config.kalshi_key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_str,
        "KALSHI-ACCESS-SIGNATURE": signature_b64,
    }


def _parse_timestamp(value: object) -> datetime:
    if isinstance(value, (int, float)):
        if value > 1e12:
            return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
        if value > 1e10:
            return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
    return datetime.now(timezone.utc)


def _extract_trade(data: dict) -> TradeTick | None:
    msg = data.get("msg") or {}
    if not isinstance(msg, dict):
        return None

    ticker = msg.get("market_ticker") or msg.get("ticker") or msg.get("market")
    if not ticker:
        return None

    yes_price = msg.get("yes_price")
    raw_price = msg.get("price")
    side = (msg.get("side") or "").lower()
    if yes_price is None and raw_price is not None:
        if side == "no":
            yes_price = 100 - float(raw_price)
        else:
            yes_price = float(raw_price)

    if yes_price is not None and yes_price > 1.5:
        # Normalize cent-based Kalshi prices (0-100) into 0-1 probabilities.
        yes_price = float(yes_price) / 100.0

    volume = msg.get("volume")
    if volume is None:
        volume = msg.get("size") or msg.get("count") or msg.get("quantity")
    if volume is None and data.get("type") == "orderbook_delta":
        delta = msg.get("delta")
        if isinstance(delta, (int, float)):
            volume = abs(float(delta))

    if yes_price is None or volume is None:
        return None

    ts = _parse_timestamp(msg.get("ts") or msg.get("timestamp") or msg.get("time"))

    return TradeTick(
        market_ticker=str(ticker).upper(),
        yes_price=float(yes_price),
        volume=float(volume),
        ts=ts,
    )


async def run_live(series_ticker: str = "KXNBAGAME") -> None:
    config = load_config()
    started_at = datetime.now(timezone.utc)
    if not config.kalshi_rest_url:
        raise ValueError("KALSHI_REST_URL is required for market discovery")
    rest_url = config.kalshi_rest_url
    markets = fetch_markets(rest_url, series_ticker)
    tickers = select_market_tickers(markets)
    if not tickers:
        status_values = [str(m.get("status") or "") for m in markets[:10]]
        logger.info("market sample count=%d status_sample=%s", len(markets), status_values)
        raise RuntimeError(f"no markets found for series_ticker={series_ticker}")

    log_ticker = tickers[-1]
    logger.info("discovered %d markets, logging %s", len(tickers), log_ticker)

    private_key = load_private_key(str(config.kalshi_private_key_path))

    states = {ticker: MarketState(ticker) for ticker in tickers}
    logged_states = {log_ticker}
    db_path = Path(os.getenv("KALSHI_CANDLE_DB_PATH", "data/phase1_candles.sqlite"))
    store = CandleStore(db_path)
    l2_recorder = None
    l2_books: dict[str, dict[str, dict[int, int]]] = {}
    if KALSHI_L2_LOGGING:
        l2_recorder = L2Recorder(
            Path(KALSHI_L2_DB_PATH),
            KALSHI_L2_CHECKPOINT_INTERVAL_S,
            KALSHI_L2_LOG_EVERY_N,
        )
        logger.info(
            "l2_logging_enabled db=%s session_id=%s checkpoint_s=%d log_every_n=%d",
            KALSHI_L2_DB_PATH,
            l2_recorder.session_id,
            KALSHI_L2_CHECKPOINT_INTERVAL_S,
            KALSHI_L2_LOG_EVERY_N,
        )
    pending_subscribe_ids: dict[int, str] = {}
    market_to_sid: dict[str, int] = {}
    inactive_tickers: set[str] = set()
    active_tickers: set[str] = set()
    total_subscribe_requests = len(tickers)

    async def candle_timer() -> None:
        while True:
            now_minute = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            for state in states.values():
                candles = state.flush_to(now_minute)
                for candle in candles:
                    store.insert(
                        state.ticker,
                        candle,
                        state.p_open if state.p_open is not None else float("nan"),
                        state.p_base if state.p_base is not None else float("nan"),
                    )
                    if state.ticker in logged_states:
                        logger.info(
                            "candle %s start=%s close=%.2f vol=%.2f ret1=%.6f ret3=%.6f ret5=%.6f vol10=%.6f volsum5=%.2f gap=%d active=%d active3=%d gap5=%d p_open=%.2f p_base=%.2f",
                            state.ticker,
                            candle.start.isoformat(),
                            candle.close,
                            candle.volume,
                            candle.ret_1,
                            candle.ret_3,
                            candle.ret_5,
                            candle.vol_10,
                            candle.vol_sum_5,
                            candle.gap_flag,
                            candle.trade_active,
                            candle.active_last_3,
                            candle.gap_recent_5,
                            state.p_open if state.p_open is not None else float("nan"),
                            state.p_base if state.p_base is not None else float("nan"),
                        )
            await asyncio.sleep(1)

    async def subscription_status_timer() -> None:
        while True:
            pending_count = max(total_subscribe_requests - len(active_tickers), 0)
            logger.info(
                "subscription_status requested=%d active=%d pending=%d",
                total_subscribe_requests,
                len(active_tickers),
                pending_count,
            )
            await asyncio.sleep(30)

    asyncio.create_task(candle_timer())
    asyncio.create_task(subscription_status_timer())

    attempt = 0
    while True:
        pending_subscribe_ids.clear()
        market_to_sid.clear()
        try:
            headers = build_ws_auth_headers(config, private_key)
            async with websockets.connect(
                config.kalshi_ws_url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=20,
            ) as websocket:
                sub_id = 1
                for ticker in tickers:
                    payload = {
                        "id": sub_id,
                        "cmd": "subscribe",
                        "params": {
                            "channels": ["trade", "orderbook_delta", "market_lifecycle_v2"],
                            "market_ticker": ticker,
                        },
                    }
                    await websocket.send(json.dumps(payload))
                    pending_subscribe_ids[sub_id] = ticker
                    sub_id += 1

                async def market_refresh_timer() -> None:
                    nonlocal sub_id, total_subscribe_requests
                    if KALSHI_MARKET_REFRESH_S <= 0:
                        return
                    while True:
                        await asyncio.sleep(KALSHI_MARKET_REFRESH_S)
                        try:
                            refreshed = await asyncio.to_thread(fetch_markets, rest_url, series_ticker)
                            refreshed_tickers = set(select_market_tickers(refreshed))
                            known_tickers = set(states.keys()) | inactive_tickers
                            new_tickers = sorted(refreshed_tickers - known_tickers)
                            if not new_tickers:
                                continue
                            logger.info("market_refresh checked=%d new=%d", len(refreshed_tickers), len(new_tickers))
                            for ticker in new_tickers:
                                states[ticker] = MarketState(ticker)
                                payload = {
                                    "id": sub_id,
                                    "cmd": "subscribe",
                                    "params": {
                                        "channels": ["trade", "orderbook_delta", "market_lifecycle_v2"],
                                        "market_ticker": ticker,
                                    },
                                }
                                await websocket.send(json.dumps(payload))
                                pending_subscribe_ids[sub_id] = ticker
                                sub_id += 1
                            total_subscribe_requests += len(new_tickers)
                        except Exception as exc:
                            logger.info("market_refresh_error error=%s", exc)

                asyncio.create_task(market_refresh_timer())

                async for message in websocket:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    msg = data.get("msg") or {}
                    if msg_type == "subscribed":
                        cmd_id = data.get("id")
                        sid = (data.get("msg") or {}).get("sid")
                        if isinstance(cmd_id, int) and isinstance(sid, int):
                            ticker = pending_subscribe_ids.pop(cmd_id, None)
                            if ticker:
                                market_to_sid[ticker] = sid
                                logger.info("subscribed %s sid=%s", ticker, sid)
                        continue

                    if msg_type in {
                        "market_state",
                        "market_status",
                        "market_lifecycle",
                        "market_lifecycle_v2",
                    }:
                        ticker = (msg.get("market_ticker") or msg.get("ticker") or "").upper()
                        state = (
                            msg.get("state") or msg.get("status") or msg.get("event_type") or ""
                        ).lower()
                        if ticker and state in {"active", "open", "created", "activated"}:
                            if ticker in states:
                                active_tickers.add(ticker)
                        if ticker and state and state not in {"active", "open", "created", "activated"}:
                            sid = market_to_sid.get(ticker)
                            if sid is not None and ticker not in inactive_tickers:
                                payload = {"id": sub_id, "cmd": "unsubscribe", "params": {"sids": [sid]}}
                                await websocket.send(json.dumps(payload))
                                inactive_tickers.add(ticker)
                                sub_id += 1
                                logger.info("unsubscribed %s state=%s", ticker, state)
                            states.pop(ticker, None)
                        continue

                    if msg_type in {"orderbook_snapshot", "orderbook_delta"}:
                        ticker = (msg.get("market_ticker") or msg.get("ticker") or "").upper()
                        if ticker:
                            if ticker not in l2_books:
                                l2_books[ticker] = {"yes": {}, "no": {}}
                            book = l2_books[ticker]
                            if msg_type == "orderbook_snapshot":
                                book["yes"] = {int(p): int(sz) for p, sz in (msg.get("yes") or [])}
                                book["no"] = {int(p): int(sz) for p, sz in (msg.get("no") or [])}
                            else:
                                price = msg.get("price")
                                delta = msg.get("delta")
                                side = msg.get("side")
                                if isinstance(price, int) and isinstance(delta, int) and side in ("yes", "no"):
                                    prev = book[side].get(price, 0)
                                    new_sz = prev + delta
                                    if new_sz <= 0:
                                        book[side].pop(price, None)
                                    else:
                                        book[side][price] = new_sz

                        if l2_recorder:
                            seq = msg.get("seq") if isinstance(msg.get("seq"), int) else msg.get("sequence")
                            if not isinstance(seq, int):
                                seq = None
                            l2_recorder.log_message(ticker or "UNKNOWN", msg_type, data, seq)
                            l2_recorder.maybe_checkpoint(l2_books)

                    tick = _extract_trade(data)
                    if not tick:
                        continue
                    if tick.market_ticker in inactive_tickers:
                        continue
                    if tick.market_ticker in states:
                        active_tickers.add(tick.market_ticker)
                    state = states.get(tick.market_ticker)
                    if state is None:
                        continue
                    candles = state.on_trade(tick)
                    for candle in candles:
                        store.insert(
                            state.ticker,
                            candle,
                            state.p_open if state.p_open is not None else float("nan"),
                            state.p_base if state.p_base is not None else float("nan"),
                        )
                        if state.ticker == log_ticker:
                            logger.info(
                                "candle %s start=%s close=%.2f vol=%.2f ret1=%.6f ret3=%.6f ret5=%.6f vol10=%.6f volsum5=%.2f gap=%d active=%d active3=%d gap5=%d p_open=%.2f p_base=%.2f",
                                state.ticker,
                                candle.start.isoformat(),
                                candle.close,
                                candle.volume,
                                candle.ret_1,
                                candle.ret_3,
                                candle.ret_5,
                                candle.vol_10,
                                candle.vol_sum_5,
                                candle.gap_flag,
                                candle.trade_active,
                                candle.active_last_3,
                                candle.gap_recent_5,
                                state.p_open if state.p_open is not None else float("nan"),
                                state.p_base if state.p_base is not None else float("nan"),
                            )
            attempt = 0
        except Exception as exc:
            delay = _backoff_delay(attempt, base_delay=1.0, max_delay=30.0)
            delay += delay * 0.2 * random.random()
            logger.info("ws_reconnect attempt=%d delay=%.2f error=%s", attempt + 1, delay, exc)
            attempt += 1
            await asyncio.sleep(delay)
        finally:
            if l2_recorder:
                l2_recorder.close()


def main() -> None:
    asyncio.run(run_live())


if __name__ == "__main__":
    main()
