from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import random
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode, urlparse

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from .artifacts import load_artifacts
from .config import load_config
from .phase4 import (
    ActivityTracker,
    CandleRow,
    CandleStream,
    COOLDOWN,
    ENTRY_DELAY,
    KILL_SWITCH_SAMPLE,
    MemoryPaperStore,
    PaperStore,
    PanicState,
    Position,
    SignalEngine,
    _exit_signal,
    _seed_activity,
    _seed_last_prices,
)


logger = logging.getLogger("nba_phase5")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


STOP_LOSS = 0.05
RISK_PCT = 0.0025
MAX_NOTIONAL_PCT = 0.05
OVERRIDE_CAP_PCT = 0.20
CANCEL_AFTER = timedelta(seconds=60)
KALSHI_DIAG_INTERVAL_S = int(os.getenv("KALSHI_DIAG_INTERVAL_S", "1800"))
KALSHI_DIAG_MARKET_SAMPLE = int(os.getenv("KALSHI_DIAG_MARKET_SAMPLE", "5"))


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 5
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: float = 0.2


def _backoff_delay(attempt: int, policy: RetryPolicy) -> float:
    delay = min(policy.max_delay, policy.base_delay * (2**attempt))
    jitter = delay * policy.jitter * random.random()
    return delay + jitter


def _request_with_backoff(
    method: str,
    url: str,
    *,
    policy: RetryPolicy,
    timeout: float = 20,
    json_payload: dict[str, Any] | None = None,
    data_str: str | None = None,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(policy.attempts):
        try:
            response = requests.request(
                method,
                url,
                json=json_payload if data_str is None else None,
                data=data_str,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            status = getattr(exc.response, "status_code", None)
            body = getattr(exc.response, "text", None)
            if status is not None:
                logger.info(
                    "rest_error url=%s status=%s body=%s",
                    url,
                    status,
                    body[:500] if isinstance(body, str) else body,
                )
            if attempt >= policy.attempts - 1:
                break
            delay = _backoff_delay(attempt, policy)
            logger.info("rest_retry url=%s attempt=%d delay=%.2f", url, attempt + 1, delay)
            time.sleep(delay)
    raise RuntimeError(f"rest_request_failed url={url}") from last_exc


def _load_private_key(path: Path) -> object:
    raw = path.read_text(encoding="utf-8")
    return serialization.load_pem_private_key(
        raw.encode("utf-8"),
        password=None,
        backend=default_backend(),
    )


def _sign_message(private_key: object, message: str) -> str:
    signature = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


class RestClient:
    def __init__(
        self,
        rest_url: str,
        key_id: str,
        private_key_path: Path,
        order_url: str | None = None,
        policy: RetryPolicy | None = None,
    ) -> None:
        self.rest_url = rest_url.rstrip("/")
        self.order_url = order_url.rstrip("/") if order_url else None
        self.key_id = key_id
        self.private_key = _load_private_key(private_key_path)
        self.policy = policy or RetryPolicy()

    def _headers(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None,
        body_str: str | None,
        base_url: str,
    ) -> dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        body_str = body_str or ""
        base_path = urlparse(base_url).path.rstrip("/")
        if os.getenv("KALSHI_SIGN_NO_BASE") == "1":
            base_path = ""
        full_path = f"{base_path}/{path.lstrip('/')}" if base_path else path
        message = f"{timestamp}{method.upper()}{full_path}"
        if os.getenv("KALSHI_DEBUG_SIGNATURE") == "1":
            logger.info("sign_debug method=%s path=%s", method, full_path)
        signature = _sign_message(self.private_key, message)
        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        base_url: str | None = None,
        timeout: float = 20,
    ) -> dict[str, Any]:
        base = base_url.rstrip("/") if base_url else self.rest_url
        url = f"{base}/{path.lstrip('/')}"
        body_str = json.dumps(body, separators=(",", ":"), sort_keys=True) if body else None
        headers = self._headers(method, f"/{path.lstrip('/')}", params, body_str, base)
        response = _request_with_backoff(
            method,
            url,
            policy=self.policy,
            timeout=timeout,
            json_payload=None if body_str is not None else body,
            data_str=body_str,
            headers=headers,
            params=params,
        )
        return response.json()

    def get_portfolio(self) -> dict[str, Any]:
        return self.request("GET", "/portfolio/balance")

    def get_orderbook(self, market_ticker: str) -> dict[str, Any]:
        return self.request("GET", f"/markets/{market_ticker}/orderbook")

    def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/portfolio/orders", body=payload, base_url=self.order_url)

    def get_order(self, order_id: int) -> dict[str, Any]:
        return self.request("GET", f"/portfolio/orders/{order_id}", base_url=self.order_url)

    def cancel_order(self, order_id: int) -> dict[str, Any]:
        return self.request("DELETE", f"/portfolio/orders/{order_id}", base_url=self.order_url)

    def cancel_orders(self, order_ids: list[str]) -> dict[str, Any]:
        payload = {"ids": order_ids}
        return self.request("DELETE", "/portfolio/orders/batched", body=payload, base_url=self.order_url)


class OrderLedger(Protocol):
    def reserve(self, order_key: str, action: str, market_ticker: str, ts: datetime) -> bool:
        ...


class SqliteOrderLedger:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS order_actions (
                order_key TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                market_ticker TEXT NOT NULL,
                created_ts TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def reserve(self, order_key: str, action: str, market_ticker: str, ts: datetime) -> bool:
        try:
            self.conn.execute(
                """
                INSERT INTO order_actions (order_key, action, market_ticker, created_ts)
                VALUES (?, ?, ?, ?)
                """,
                (order_key, action, market_ticker, ts.isoformat()),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False


class MemoryOrderLedger:
    def __init__(self) -> None:
        self._keys: set[str] = set()

    def reserve(self, order_key: str, action: str, market_ticker: str, ts: datetime) -> bool:
        if order_key in self._keys:
            return False
        self._keys.add(order_key)
        return True


def _extract_balance(payload: dict[str, Any]) -> float | None:
    portfolio = payload.get("portfolio") if isinstance(payload, dict) else None
    if isinstance(portfolio, dict):
        for key in ("available_balance", "balance", "cash_balance", "total_balance"):
            if key in portfolio and portfolio[key] is not None:
                return float(portfolio[key]) / 100.0
    for key in ("available_balance", "balance", "cash_balance", "total_balance"):
        if key in payload and payload[key] is not None:
            return float(payload[key]) / 100.0
    return None


def _normalize_levels(raw: Any) -> list[tuple[int, int]]:
    levels: list[tuple[int, int]] = []
    if raw is None:
        return levels
    if isinstance(raw, dict):
        for price, size in raw.items():
            try:
                levels.append((int(price), int(size)))
            except (TypeError, ValueError):
                continue
        return levels
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                try:
                    levels.append((int(entry[0]), int(entry[1])))
                except (TypeError, ValueError):
                    continue
            elif isinstance(entry, dict):
                price = entry.get("price")
                size = entry.get("size") or entry.get("count") or entry.get("quantity")
                try:
                    levels.append((int(price), int(size)))
                except (TypeError, ValueError):
                    continue
    return levels


def _best_levels(levels: list[tuple[int, int]], best: str) -> list[tuple[int, int]]:
    if not levels:
        return []
    if best == "max":
        return sorted(levels, key=lambda x: x[0], reverse=True)
    return sorted(levels, key=lambda x: x[0])


def _extract_orderbook_levels(payload: dict[str, Any]) -> dict[str, list[tuple[int, int]]]:
    book = payload.get("orderbook") if isinstance(payload, dict) else None
    if isinstance(book, dict):
        payload = book
    keys = {
        "yes_bids": _normalize_levels(payload.get("yes_bids") or payload.get("yes_bid") or payload.get("yes")),
        "yes_asks": _normalize_levels(payload.get("yes_asks") or payload.get("yes_ask")),
        "no_bids": _normalize_levels(payload.get("no_bids") or payload.get("no_bid") or payload.get("no")),
        "no_asks": _normalize_levels(payload.get("no_asks") or payload.get("no_ask")),
    }
    return keys


def _derive_best_prices(levels: dict[str, list[tuple[int, int]]]) -> dict[str, int | None]:
    yes_bids = _best_levels(levels["yes_bids"], "max")
    yes_asks = _best_levels(levels["yes_asks"], "min")
    no_bids = _best_levels(levels["no_bids"], "max")
    no_asks = _best_levels(levels["no_asks"], "min")

    best_yes_bid = yes_bids[0][0] if yes_bids else None
    best_yes_ask = yes_asks[0][0] if yes_asks else None
    best_no_bid = no_bids[0][0] if no_bids else None
    best_no_ask = no_asks[0][0] if no_asks else None

    if best_yes_ask is None and best_no_bid is not None:
        best_yes_ask = 100 - best_no_bid
    if best_yes_bid is None and best_no_ask is not None:
        best_yes_bid = 100 - best_no_ask
    if best_no_ask is None and best_yes_bid is not None:
        best_no_ask = 100 - best_yes_bid
    if best_no_bid is None and best_yes_ask is not None:
        best_no_bid = 100 - best_yes_ask

    return {
        "yes_bid": best_yes_bid,
        "yes_ask": best_yes_ask,
        "no_bid": best_no_bid,
        "no_ask": best_no_ask,
    }


def _depth_check(
    levels: dict[str, list[tuple[int, int]]],
    side: str,
    action: str,
    qty: int,
    best_prices: dict[str, int | None],
) -> DepthCheck:
    side = side.upper()
    action = action.lower()
    if side not in {"YES", "NO"}:
        return DepthCheck(False, 0, 0, None, "invalid_side")
    key_prefix = "yes" if side == "YES" else "no"
    ladder = levels[f"{key_prefix}_{'asks' if action == 'buy' else 'bids'}"]
    sorted_levels = _best_levels(ladder, "min" if action == "buy" else "max")
    best_level_size = sorted_levels[0][1] if sorted_levels else 0
    top3_size = sum(size for _, size in sorted_levels[:3]) if sorted_levels else 0
    price_key = f"{key_prefix}_{'ask' if action == 'buy' else 'bid'}"
    best_price_cents = best_prices.get(price_key)
    if best_price_cents is None:
        return DepthCheck(False, best_level_size, top3_size, None, "missing_price")
    best_price = best_price_cents / 100.0
    if best_level_size < 2 * qty:
        return DepthCheck(False, best_level_size, top3_size, best_price, "best_level")
    if top3_size < 4 * qty:
        return DepthCheck(False, best_level_size, top3_size, best_price, "top3_levels")
    return DepthCheck(True, best_level_size, top3_size, best_price, None)


def _effective_capital(balance: float) -> float:
    cap = balance * OVERRIDE_CAP_PCT
    return min(balance, cap)


def _position_notional(balance: float) -> float:
    effective = _effective_capital(balance)
    risk_budget = effective * RISK_PCT
    raw_notional = risk_budget / STOP_LOSS
    return min(raw_notional, effective * MAX_NOTIONAL_PCT)


def _size_for_price(balance: float, price: float) -> int:
    if price <= 0:
        return 0
    notional = _position_notional(balance)
    qty = int(notional / price)
    return max(qty, 0)


def _pnl_for_side(side: str, entry_price: float, exit_price: float, qty: int) -> float:
    if side.upper() == "YES":
        return (exit_price - entry_price) * qty
    return (entry_price - exit_price) * qty
@dataclass(frozen=True)
class DepthCheck:
    ok: bool
    best_size: int
    top3_size: int
    best_price: float | None
    reason: str | None


@dataclass
class PendingOrder:
    order_id: int
    market_ticker: str
    side: str
    action: str
    qty: int
    price: float
    placed_ts: datetime
    panic: PanicState | None = None
    quality_score: float | None = None
    exit_reason: str | None = None
    pnl: float | None = None


@dataclass(frozen=True)
class EntryOrder:
    order_key: str
    market_ticker: str
    side: str
    qty: int
    entry_ts: datetime
    entry_price: float
    panic: PanicState
    quality_score: float


@dataclass(frozen=True)
class ExitOrder:
    order_key: str
    position: Position
    exit_ts: datetime
    exit_price: float
    exit_reason: str
    pnl: float


@dataclass(frozen=True)
class OrderResult:
    accepted: bool
    order_id: int | None
    reason: str | None


@dataclass(frozen=True)
class OrderStatus:
    status: str
    filled_qty: int
    remaining_qty: int
    avg_price: float | None


class OrderAdapter(Protocol):
    def submit_entry(self, order: EntryOrder) -> OrderResult:
        ...

    def submit_exit(self, order: ExitOrder) -> OrderResult:
        ...

    def get_order_status(self, order_id: int) -> OrderStatus:
        ...

    def cancel_order(self, order_id: int) -> None:
        ...

    def cancel_orders(self, order_ids: list[str]) -> None:
        ...


class PaperOrderAdapter:
    def __init__(self, store: PaperStore, ledger: OrderLedger) -> None:
        self.store = store
        self.ledger = ledger

    def submit_entry(self, order: EntryOrder) -> OrderResult:
        if not self.ledger.reserve(order.order_key, "entry", order.market_ticker, order.entry_ts):
            return OrderResult(False, None, "duplicate")
        position_id = self.store.insert_entry(
            order.market_ticker,
            order.side,
            order.entry_ts,
            order.entry_price,
            order.panic,
            order.quality_score,
            order.qty,
        )
        return OrderResult(True, position_id, None)

    def submit_exit(self, order: ExitOrder) -> OrderResult:
        if not self.ledger.reserve(order.order_key, "exit", order.position.market_ticker, order.exit_ts):
            return OrderResult(False, None, "duplicate")
        self.store.update_exit(
            order.position.id,
            order.exit_ts,
            order.exit_price,
            order.exit_reason,
            order.pnl,
        )
        return OrderResult(True, order.position.id, None)

    def get_order_status(self, order_id: int) -> OrderStatus:
        return OrderStatus(status="filled", filled_qty=1, remaining_qty=0, avg_price=None)

    def cancel_order(self, order_id: int) -> None:
        return None

    def cancel_orders(self, order_ids: list[str]) -> None:
        return None


class LiveOrderAdapter:
    def __init__(self, client: RestClient, ledger: OrderLedger) -> None:
        self.client = client
        self.ledger = ledger

    def submit_entry(self, order: EntryOrder) -> OrderResult:
        if not self.ledger.reserve(order.order_key, "entry", order.market_ticker, order.entry_ts):
            return OrderResult(False, None, "duplicate")
        payload = _build_order_payload(
            market_ticker=order.market_ticker,
            side=order.side,
            action="buy",
            qty=order.qty,
            price=order.entry_price,
            client_order_id=order.order_key,
        )
        response = self.client.place_order(payload)
        order_id = _extract_order_id(response)
        if order_id is None:
            _log_event(
                "order_submit_error",
                action="entry",
                market_ticker=order.market_ticker,
                response=response,
            )
            return OrderResult(False, None, "missing_order_id")
        return OrderResult(True, order_id, None)

    def submit_exit(self, order: ExitOrder) -> OrderResult:
        if not self.ledger.reserve(order.order_key, "exit", order.position.market_ticker, order.exit_ts):
            return OrderResult(False, None, "duplicate")
        payload = _build_order_payload(
            market_ticker=order.position.market_ticker,
            side=order.position.side,
            action="sell",
            qty=order.position.qty,
            price=order.exit_price,
            client_order_id=order.order_key,
        )
        response = self.client.place_order(payload)
        order_id = _extract_order_id(response)
        if order_id is None:
            _log_event(
                "order_submit_error",
                action="exit",
                market_ticker=order.position.market_ticker,
                response=response,
            )
            return OrderResult(False, None, "missing_order_id")
        return OrderResult(True, order_id, None)

    def get_order_status(self, order_id: int) -> OrderStatus:
        payload = self.client.get_order(order_id)
        return _parse_order_status(payload)

    def cancel_order(self, order_id: int) -> None:
        self.client.cancel_order(order_id)

    def cancel_orders(self, order_ids: list[str]) -> None:
        self.client.cancel_orders(order_ids)


def _entry_order_key(panic: PanicState) -> str:
    return f"entry:{panic.market_ticker}:{panic.direction}:{panic.detected_ts.isoformat()}"


def _exit_order_key(position: Position) -> str:
    return f"exit:{position.market_ticker}:{position.side}:{position.entry_ts.isoformat()}"


def _build_order_payload(
    *,
    market_ticker: str,
    side: str,
    action: str,
    qty: int,
    price: float,
    client_order_id: str,
) -> dict[str, Any]:
    price_cents = int(round(price * 100))
    payload: dict[str, Any] = {
        "ticker": market_ticker,
        "type": "limit",
        "action": action,
        "count": qty,
        "client_order_id": client_order_id,
        "side": side.lower(),
    }
    if side.upper() == "YES":
        payload["yes_price"] = price_cents
    else:
        payload["no_price"] = price_cents
    return payload


def _extract_order_id(response: dict[str, Any]) -> int | None:
    if not isinstance(response, dict):
        return None
    for key in ("order_id", "id"):
        if key in response and response[key] is not None:
            try:
                return int(response[key])
            except (TypeError, ValueError):
                return None
    order = response.get("order")
    if isinstance(order, dict):
        value = order.get("order_id") or order.get("id")
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _parse_order_status(response: dict[str, Any]) -> OrderStatus:
    order = response.get("order") if isinstance(response, dict) else None
    if not isinstance(order, dict):
        order = response if isinstance(response, dict) else {}
    status = str(order.get("status") or order.get("state") or "unknown").lower()
    filled = order.get("filled_count") or order.get("filled_qty") or 0
    remaining = order.get("remaining_count") or order.get("remaining_qty")
    count = order.get("count") or order.get("qty")
    try:
        filled_qty = int(filled)
    except (TypeError, ValueError):
        filled_qty = 0
    if remaining is None and count is not None:
        try:
            remaining_qty = max(int(count) - filled_qty, 0)
        except (TypeError, ValueError):
            remaining_qty = 0
    else:
        try:
            remaining_qty = int(remaining) if remaining is not None else 0
        except (TypeError, ValueError):
            remaining_qty = 0
    avg_price = order.get("avg_price") or order.get("average_price")
    if isinstance(avg_price, (int, float)) and avg_price > 1.0:
        avg_price = float(avg_price) / 100.0
    return OrderStatus(
        status=status,
        filled_qty=filled_qty,
        remaining_qty=remaining_qty,
        avg_price=float(avg_price) if isinstance(avg_price, (int, float)) else None,
    )
def _log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, "ts": datetime.now(timezone.utc).isoformat()}
    payload.update(fields)
    logger.info(json.dumps(payload, sort_keys=True))


def entry_decision(
    *,
    panic: PanicState,
    candle: CandleRow,
    active_last_3: int,
    engine: SignalEngine,
    positions: dict[str, Position],
    store: PaperStore | MemoryPaperStore,
    kill_switch: bool,
) -> tuple[bool, float, list[str]]:
    reasons: list[str] = []
    if kill_switch:
        reasons.append("kill_switch")
    if candle.market_ticker in positions:
        reasons.append("open_position")
    else:
        last_exit = store.last_exit_ts(candle.market_ticker)
        if last_exit and candle.start_ts < last_exit + COOLDOWN:
            reasons.append("cooldown")
    allowed, quality_score, eval_reasons = engine.evaluate_entry(panic, candle, active_last_3)
    reasons.extend(eval_reasons)
    return len(reasons) == 0, quality_score, reasons


def run_loop() -> None:
    config = load_config()
    artifacts_path = Path(os.getenv("STRATEGY_ARTIFACTS_PATH", "strategy_artifacts.json"))
    artifacts = load_artifacts(artifacts_path)

    candle_db_path = Path(os.getenv("KALSHI_CANDLE_DB_PATH", "data/phase1_candles.sqlite"))
    paper_db_path = Path(os.getenv("KALSHI_PAPER_DB_PATH", "data/paper_trades.sqlite"))

    stream = CandleStream(candle_db_path)
    store = PaperStore(paper_db_path)
    ledger = SqliteOrderLedger(store.conn)
    activity = ActivityTracker(_seed_activity(stream.conn))
    engine = SignalEngine(artifacts)

    mode = os.getenv("NBA_ENGINE_MODE", "live").lower()
    rest_client: RestClient | None = None
    if mode == "live":
        if not config.kalshi_rest_url:
            raise ValueError("KALSHI_REST_URL is required for live mode")
        rest_client = RestClient(
            config.kalshi_rest_url,
            config.kalshi_key_id,
            config.kalshi_private_key_path,
            order_url="https://demo-api.kalshi.co/trade-api/v2",
        )
        adapter = LiveOrderAdapter(rest_client, ledger)
    else:
        adapter = PaperOrderAdapter(store, ledger)

    last_rowid = stream.max_rowid()
    pending: dict[str, PanicState] = {}
    positions = store.load_open_positions()
    pending_entries: dict[str, PendingOrder] = {}
    pending_exits: dict[str, PendingOrder] = {}
    last_prices = _seed_last_prices(stream.conn, list(positions.keys()))
    kill_switch = False
    last_candle_ts: dict[str, datetime] = {}
    last_panic_ts: dict[str, datetime] = {}
    last_entry_ts: dict[str, datetime] = {}
    last_skip_reason: dict[str, str] = {}
    interval_skip_counts: Counter[str] = Counter()
    last_diag_ts = time.time()
    mean_pnl, sample_count = store.recent_mean_pnl(KILL_SWITCH_SAMPLE, artifacts.quality_cutoff)
    if sample_count == KILL_SWITCH_SAMPLE and mean_pnl is not None and mean_pnl < 0:
        kill_switch = True
        _log_event("kill_switch_triggered", mean_pnl=mean_pnl, sample=sample_count)

    force_ticker = os.getenv("FORCE_SIGNAL_TICKER")
    force_direction = os.getenv("FORCE_SIGNAL_DIRECTION", "UNDERDOG_UP").upper()
    force_used = False

    def _fetch_balance() -> float | None:
        if rest_client is None:
            return None
        payload = rest_client.get_portfolio()
        balance = _extract_balance(payload)
        _log_event("portfolio_ok", balance=balance)
        return balance

    def _fetch_depth(
        market_ticker: str,
        side: str,
        action: str,
        qty: int,
    ) -> DepthCheck:
        if rest_client is None:
            return DepthCheck(True, 0, 0, None, None)
        try:
            payload = rest_client.get_orderbook(market_ticker)
        except Exception as exc:
            _log_event(
                "orderbook_error",
                market_ticker=market_ticker,
                error=str(exc),
            )
            return DepthCheck(False, 0, 0, None, "orderbook_error")
        levels = _extract_orderbook_levels(payload)
        best_prices = _derive_best_prices(levels)
        depth = _depth_check(levels, side, action, qty, best_prices)
        _log_event(
            "orderbook_ok",
            market_ticker=market_ticker,
            side=side,
            action=action,
            qty=qty,
            yes_bid=best_prices.get("yes_bid"),
            yes_ask=best_prices.get("yes_ask"),
            no_bid=best_prices.get("no_bid"),
            no_ask=best_prices.get("no_ask"),
            best_size=depth.best_size,
            top3_size=depth.top3_size,
            depth_ok=depth.ok,
            depth_reason=depth.reason,
        )
        return depth

    def _prepare_order(
        market_ticker: str,
        side: str,
        action: str,
    ) -> tuple[int, float | None, DepthCheck, float | None]:
        if rest_client is None:
            return 1, None, DepthCheck(True, 0, 0, None, None), None
        try:
            book = rest_client.get_orderbook(market_ticker)
        except Exception as exc:
            _log_event(
                "orderbook_error",
                market_ticker=market_ticker,
                error=str(exc),
            )
            return 0, None, DepthCheck(False, 0, 0, None, "orderbook_error"), None
        levels = _extract_orderbook_levels(book)
        best_prices = _derive_best_prices(levels)
        price_key = f"{'yes' if side.upper() == 'YES' else 'no'}_{'ask' if action == 'buy' else 'bid'}"
        best_price_cents = best_prices.get(price_key)
        if best_price_cents is None:
            return 0, None, DepthCheck(False, 0, 0, None, "missing_price"), None
        best_price = best_price_cents / 100.0
        try:
            balance = _fetch_balance()
        except Exception as exc:
            _log_event(
                "portfolio_error",
                market_ticker=market_ticker,
                error=str(exc),
            )
            balance = None
        if balance is None:
            return 0, best_price, DepthCheck(False, 0, 0, best_price, "missing_balance"), None
        qty = _size_for_price(balance, best_price)
        if qty <= 0:
            return 0, best_price, DepthCheck(False, 0, 0, best_price, "size_zero"), balance
        depth = _depth_check(levels, side, action, qty, best_prices)
        return qty, best_price, depth, balance

    def _refresh_pending(now: datetime) -> None:
        nonlocal kill_switch
        if rest_client is None:
            return
        for ticker, pending_order in list(pending_entries.items()):
            status = adapter.get_order_status(pending_order.order_id)
            if status.remaining_qty == 0 and status.filled_qty > 0:
                fill_price = status.avg_price or pending_order.price
                if pending_order.panic is None or pending_order.quality_score is None:
                    pending_entries.pop(ticker, None)
                    continue
                position_id = store.insert_entry(
                    pending_order.market_ticker,
                    pending_order.side,
                    now,
                    fill_price,
                    pending_order.panic,
                    pending_order.quality_score,
                    pending_order.qty,
                )
                positions[pending_order.market_ticker] = Position(
                    id=position_id,
                    market_ticker=pending_order.market_ticker,
                    side=pending_order.side,
                    entry_ts=now,
                    entry_price=fill_price,
                    qty=pending_order.qty,
                )
                _log_event(
                    "entry_filled",
                    market_ticker=pending_order.market_ticker,
                    side=pending_order.side,
                    qty=pending_order.qty,
                    entry_ts=now.isoformat(),
                    entry_price=fill_price,
                    order_id=pending_order.order_id,
                )
                pending_entries.pop(ticker, None)
                continue
            if now - pending_order.placed_ts >= CANCEL_AFTER:
                adapter.cancel_order(pending_order.order_id)
                _log_event(
                    "entry_canceled",
                    market_ticker=pending_order.market_ticker,
                    order_id=pending_order.order_id,
                )
                pending_entries.pop(ticker, None)

        for ticker, pending_order in list(pending_exits.items()):
            status = adapter.get_order_status(pending_order.order_id)
            if status.remaining_qty == 0 and status.filled_qty > 0:
                exit_price = status.avg_price or pending_order.price
                position = positions.get(ticker)
                if position is None:
                    pending_exits.pop(ticker, None)
                    continue
                if pending_order.exit_reason is None or pending_order.pnl is None:
                    pending_exits.pop(ticker, None)
                    continue
                store.update_exit(
                    position.id,
                    now,
                    exit_price,
                    pending_order.exit_reason,
                    pending_order.pnl,
                )
                positions.pop(ticker, None)
                _log_event(
                    "exit_filled",
                    market_ticker=pending_order.market_ticker,
                    side=pending_order.side,
                    qty=pending_order.qty,
                    reason=pending_order.exit_reason,
                    exit_ts=now.isoformat(),
                    exit_price=exit_price,
                    pnl=pending_order.pnl,
                    order_id=pending_order.order_id,
                )
                if not kill_switch:
                    mean_pnl, sample_count = store.recent_mean_pnl(
                        KILL_SWITCH_SAMPLE, artifacts.quality_cutoff
                    )
                    if (
                        sample_count == KILL_SWITCH_SAMPLE
                        and mean_pnl is not None
                        and mean_pnl < 0
                    ):
                        kill_switch = True
                        _log_event(
                            "kill_switch_triggered",
                            mean_pnl=mean_pnl,
                            sample=sample_count,
                        )
                pending_exits.pop(ticker, None)
                continue
            if now - pending_order.placed_ts >= CANCEL_AFTER:
                adapter.cancel_order(pending_order.order_id)
                _log_event(
                    "exit_canceled",
                    market_ticker=pending_order.market_ticker,
                    order_id=pending_order.order_id,
                )
                pending_exits.pop(ticker, None)

    _log_event(
        "phase5_ready",
        candle_db=str(candle_db_path),
        paper_db=str(paper_db_path),
        last_rowid=last_rowid,
        open_positions=len(positions),
        artifacts=artifacts.summary(),
        mode=mode,
        force_ticker=force_ticker,
        force_direction=force_direction,
    )

    if rest_client is not None:
        sample_ticker = None
        if positions:
            sample_ticker = next(iter(positions.keys()))
        if sample_ticker is None:
            row = stream.conn.execute(
                "SELECT market_ticker FROM candles ORDER BY start_ts DESC LIMIT 1"
            ).fetchone()
            if row and row[0]:
                sample_ticker = str(row[0])
        if sample_ticker:
            try:
                payload = rest_client.get_orderbook(sample_ticker)
                levels = _extract_orderbook_levels(payload)
                best_prices = _derive_best_prices(levels)
                _log_event(
                    "orderbook_probe",
                    market_ticker=sample_ticker,
                    yes_bid=best_prices.get("yes_bid"),
                    yes_ask=best_prices.get("yes_ask"),
                    no_bid=best_prices.get("no_bid"),
                    no_ask=best_prices.get("no_ask"),
                    yes_bids=len(levels.get("yes_bids", [])),
                    yes_asks=len(levels.get("yes_asks", [])),
                    no_bids=len(levels.get("no_bids", [])),
                    no_asks=len(levels.get("no_asks", [])),
                )
            except Exception as exc:
                _log_event(
                    "orderbook_probe_error",
                    market_ticker=sample_ticker,
                    error=str(exc),
                )

    def _record_skip(market_ticker: str, reasons: list[str]) -> None:
        if not reasons:
            return
        for reason in reasons:
            interval_skip_counts[reason] += 1
        last_skip_reason[market_ticker] = ",".join(reasons)

    def _log_diag(now_ts: float) -> None:
        if KALSHI_DIAG_INTERVAL_S <= 0:
            return
        nonlocal last_diag_ts, interval_skip_counts
        if now_ts - last_diag_ts < KALSHI_DIAG_INTERVAL_S:
            return
        now = datetime.now(timezone.utc)
        markets_seen = list(last_candle_ts.keys())
        no_panic = [
            m
            for m in markets_seen
            if (m not in last_panic_ts)
            or (now - last_panic_ts[m] > timedelta(seconds=KALSHI_DIAG_INTERVAL_S))
        ]
        no_entry = [
            m
            for m in markets_seen
            if (m not in last_entry_ts)
            or (now - last_entry_ts[m] > timedelta(seconds=KALSHI_DIAG_INTERVAL_S))
        ]
        sample_markets = []
        for m in no_entry[: max(KALSHI_DIAG_MARKET_SAMPLE, 0)]:
            reason = last_skip_reason.get(m, "no_signal")
            sample_markets.append(f"{m}:{reason}")
        top_reasons = ",".join(
            f"{k}:{v}" for k, v in interval_skip_counts.most_common(5)
        )
        _log_event(
            "trade_inhibit_summary",
            markets_seen=len(markets_seen),
            no_panic=len(no_panic),
            no_entry=len(no_entry),
            top_reasons=top_reasons,
            sample_markets=";".join(sample_markets),
        )
        interval_skip_counts = Counter()
        last_diag_ts = now_ts

    while True:
        rows = stream.fetch_since(last_rowid)
        if not rows:
            time.sleep(1)
            continue
        for candle in rows:
            last_rowid = candle.rowid
            last_candle_ts[candle.market_ticker] = candle.start_ts
            _refresh_pending(datetime.now(timezone.utc))
            active_last_3 = activity.update(candle.market_ticker, candle.trade_active)
            if candle.close == candle.close:
                last_prices[candle.market_ticker] = candle.close
            _log_diag(time.time())

            position = positions.get(candle.market_ticker)
            if position:
                if candle.market_ticker in pending_exits:
                    continue
                reason, exit_price, pnl = _exit_signal(
                    position,
                    candle.start_ts,
                    candle.close,
                    last_prices.get(candle.market_ticker),
                )
                if reason and exit_price is not None and pnl is not None:
                    if rest_client is not None:
                        depth = _fetch_depth(
                            candle.market_ticker, position.side, "sell", position.qty
                        )
                        if not depth.ok:
                            _log_event(
                                "exit_depth_skip",
                                market_ticker=position.market_ticker,
                                side=position.side,
                                qty=position.qty,
                                reason=depth.reason,
                                best_size=depth.best_size,
                                top3_size=depth.top3_size,
                            )
                            continue
                        price = depth.best_price if depth.best_price is not None else exit_price
                        order_key = _exit_order_key(position)
                        result = adapter.submit_exit(
                            ExitOrder(
                                order_key=order_key,
                                position=position,
                                exit_ts=candle.start_ts,
                                exit_price=price,
                                exit_reason=reason,
                                pnl=_pnl_for_side(
                                    position.side, position.entry_price, price, position.qty
                                ),
                            )
                        )
                        if result.accepted and result.order_id is not None:
                            pending_exits[candle.market_ticker] = PendingOrder(
                                order_id=result.order_id,
                                market_ticker=position.market_ticker,
                                side=position.side,
                                action="sell",
                                qty=position.qty,
                                price=price,
                                placed_ts=candle.start_ts,
                                exit_reason=reason,
                                pnl=_pnl_for_side(
                                    position.side, position.entry_price, price, position.qty
                                ),
                            )
                            _log_event(
                                "exit_submitted",
                                market_ticker=position.market_ticker,
                                side=position.side,
                                qty=position.qty,
                                reason=reason,
                                exit_ts=candle.start_ts.isoformat(),
                                exit_price=price,
                                order_id=result.order_id,
                            )
                        else:
                            _log_event(
                                "exit_deduped",
                                market_ticker=position.market_ticker,
                                order_key=order_key,
                            )
                    else:
                        order_key = _exit_order_key(position)
                        result = adapter.submit_exit(
                            ExitOrder(
                                order_key=order_key,
                                position=position,
                                exit_ts=candle.start_ts,
                                exit_price=exit_price,
                                exit_reason=reason,
                                pnl=pnl,
                            )
                        )
                        if result.accepted:
                            positions.pop(candle.market_ticker, None)
                            _log_event(
                                "exit",
                                market_ticker=position.market_ticker,
                                side=position.side,
                                reason=reason,
                                exit_ts=candle.start_ts.isoformat(),
                                exit_price=exit_price,
                                pnl=pnl,
                                entry_ts=position.entry_ts.isoformat(),
                                entry_price=position.entry_price,
                            )
                        else:
                            _log_event(
                                "exit_deduped",
                                market_ticker=position.market_ticker,
                                order_key=order_key,
                            )
                    if not kill_switch:
                        mean_pnl, sample_count = store.recent_mean_pnl(
                            KILL_SWITCH_SAMPLE, artifacts.quality_cutoff
                        )
                        if (
                            sample_count == KILL_SWITCH_SAMPLE
                            and mean_pnl is not None
                            and mean_pnl < 0
                        ):
                            kill_switch = True
                            _log_event(
                                "kill_switch_triggered",
                                mean_pnl=mean_pnl,
                                sample=sample_count,
                            )

            if (
                force_ticker
                and not force_used
                and candle.market_ticker.upper() == force_ticker.upper()
            ):
                p_open = candle.p_open if candle.p_open == candle.p_open else candle.close
                p_base = candle.p_base if candle.p_base == candle.p_base else p_open
                pending[candle.market_ticker] = PanicState(
                    detected_ts=candle.start_ts - ENTRY_DELAY,
                    market_ticker=candle.market_ticker,
                    direction=force_direction,
                    p_open=p_open,
                    p_base=p_base,
                    ret_3=candle.ret_3,
                    vol_10=candle.vol_10,
                    vol_sum_5=candle.vol_sum_5,
                )
                last_panic_ts[candle.market_ticker] = candle.start_ts
                force_used = True
                _log_event(
                    "force_signal",
                    market_ticker=candle.market_ticker,
                    direction=force_direction,
                    ts=candle.start_ts.isoformat(),
                )

            panic = pending.get(candle.market_ticker)
            if panic:
                if candle.start_ts >= panic.detected_ts + ENTRY_DELAY:
                    reasons: list[str] = []
                    if kill_switch:
                        reasons.append("kill_switch")
                    if candle.market_ticker in positions:
                        reasons.append("open_position")
                    else:
                        last_exit = store.last_exit_ts(candle.market_ticker)
                        if last_exit and candle.start_ts < last_exit + COOLDOWN:
                            reasons.append("cooldown")
                    if candle.market_ticker in pending_entries:
                        reasons.append("pending_entry")
                    if reasons:
                        _record_skip(candle.market_ticker, reasons)
                        _log_event(
                            "skip_entry",
                            market_ticker=candle.market_ticker,
                            reasons=",".join(reasons),
                            entry_ts=candle.start_ts.isoformat(),
                            entry_price=candle.close,
                            p_base=panic.p_base,
                            active_last_3=active_last_3,
                            gap_recent_5=candle.gap_recent_5,
                            quality=-1.0,
                        )
                    else:
                        side = "NO" if panic.direction == "UNDERDOG_UP" else "YES"
                        qty = 1
                        entry_price = candle.close
                        balance = None
                        if rest_client is not None:
                            qty, entry_price, depth, balance = _prepare_order(
                                candle.market_ticker,
                                side,
                                "buy",
                            )
                            if not depth.ok:
                                _record_skip(candle.market_ticker, [f"entry_depth:{depth.reason}"])
                                _log_event(
                                    "entry_depth_skip",
                                    market_ticker=candle.market_ticker,
                                    side=side,
                                    reason=depth.reason,
                                    best_size=depth.best_size,
                                    top3_size=depth.top3_size,
                                    best_price=depth.best_price,
                                    balance=balance,
                                )
                                pending.pop(candle.market_ticker, None)
                                continue
                            if entry_price is None or qty <= 0:
                                _record_skip(candle.market_ticker, ["entry_size"])
                                _log_event(
                                    "entry_size_skip",
                                    market_ticker=candle.market_ticker,
                                    side=side,
                                    best_price=entry_price,
                                    qty=qty,
                                    balance=balance,
                                )
                                pending.pop(candle.market_ticker, None)
                                continue
                            eval_candle = CandleRow(
                                rowid=candle.rowid,
                                market_ticker=candle.market_ticker,
                                start_ts=candle.start_ts,
                                close=entry_price,
                                trade_active=candle.trade_active,
                                ret_3=candle.ret_3,
                                vol_10=candle.vol_10,
                                vol_sum_5=candle.vol_sum_5,
                                gap_recent_5=candle.gap_recent_5,
                                p_open=candle.p_open,
                                p_base=candle.p_base,
                            )
                            allowed, quality_score, eval_reasons = engine.evaluate_entry(
                                panic, eval_candle, active_last_3
                            )
                        else:
                            allowed, quality_score, eval_reasons = engine.evaluate_entry(
                                panic, candle, active_last_3
                            )
                        if eval_reasons:
                            _record_skip(candle.market_ticker, eval_reasons)
                            _log_event(
                                "skip_entry",
                                market_ticker=candle.market_ticker,
                                reasons=",".join(eval_reasons),
                                entry_ts=candle.start_ts.isoformat(),
                                entry_price=entry_price,
                                p_base=panic.p_base,
                                active_last_3=active_last_3,
                                gap_recent_5=candle.gap_recent_5,
                                quality=quality_score if quality_score == quality_score else -1.0,
                            )
                            pending.pop(candle.market_ticker, None)
                            continue
                        if not allowed:
                            _record_skip(candle.market_ticker, ["not_allowed"])
                            pending.pop(candle.market_ticker, None)
                            continue
                        order_key = _entry_order_key(panic)
                        result = adapter.submit_entry(
                            EntryOrder(
                                order_key=order_key,
                                market_ticker=candle.market_ticker,
                                side=side,
                                qty=qty,
                                entry_ts=candle.start_ts,
                                entry_price=entry_price,
                                panic=panic,
                                quality_score=quality_score,
                            )
                        )
                        if result.accepted and result.order_id is not None:
                            if rest_client is not None:
                                pending_entries[candle.market_ticker] = PendingOrder(
                                    order_id=result.order_id,
                                    market_ticker=candle.market_ticker,
                                    side=side,
                                    action="buy",
                                    qty=qty,
                                    price=entry_price,
                                    placed_ts=candle.start_ts,
                                    panic=panic,
                                    quality_score=quality_score,
                                )
                                _log_event(
                                    "entry_submitted",
                                    market_ticker=candle.market_ticker,
                                    side=side,
                                    qty=qty,
                                    entry_ts=candle.start_ts.isoformat(),
                                    entry_price=entry_price,
                                    order_id=result.order_id,
                                    balance=balance,
                                    effective_capital=_effective_capital(balance) if balance is not None else None,
                                    notional=_position_notional(balance) if balance is not None else None,
                                )
                                last_entry_ts[candle.market_ticker] = candle.start_ts
                            else:
                                new_position = Position(
                                    id=result.order_id,
                                    market_ticker=candle.market_ticker,
                                    side=side,
                                    entry_ts=candle.start_ts,
                                    entry_price=entry_price,
                                    qty=qty,
                                )
                                positions[candle.market_ticker] = new_position
                                _log_event(
                                    "entry",
                                    market_ticker=candle.market_ticker,
                                    side=side,
                                    qty=qty,
                                    entry_ts=candle.start_ts.isoformat(),
                                    entry_price=entry_price,
                                    panic_ts=panic.detected_ts.isoformat(),
                                    ret_3=panic.ret_3,
                                    vol_10=panic.vol_10,
                                    vol_sum_5=panic.vol_sum_5,
                                    quality=quality_score,
                                    p_open=panic.p_open,
                                    p_base=panic.p_base,
                                )
                                last_entry_ts[candle.market_ticker] = candle.start_ts
                        else:
                            _log_event(
                                "entry_deduped",
                                market_ticker=candle.market_ticker,
                                order_key=order_key,
                            )
                    pending.pop(candle.market_ticker, None)
                continue

            panic = engine.detect_panic(candle)
            if panic:
                pending[candle.market_ticker] = panic
                last_panic_ts[candle.market_ticker] = panic.detected_ts
                _log_event(
                    "panic_detected",
                    market_ticker=candle.market_ticker,
                    direction=panic.direction,
                    ts=panic.detected_ts.isoformat(),
                    ret_3=panic.ret_3,
                    vol_10=panic.vol_10,
                    vol_sum_5=panic.vol_sum_5,
                    p_open=panic.p_open,
                    p_base=panic.p_base,
                )


def run_replay() -> dict[str, Any]:
    artifacts_path = Path(os.getenv("STRATEGY_ARTIFACTS_PATH", "strategy_artifacts.json"))
    artifacts = load_artifacts(artifacts_path)
    candle_db_path = Path(os.getenv("KALSHI_CANDLE_DB_PATH", "data/phase1_candles.sqlite"))

    stream = CandleStream(candle_db_path)
    store = MemoryPaperStore()
    ledger = MemoryOrderLedger()
    engine = SignalEngine(artifacts)
    adapter = PaperOrderAdapter(store, ledger)
    activity = ActivityTracker()

    pending: dict[str, PanicState] = {}
    positions = store.load_open_positions()
    last_prices: dict[str, float] = {}
    kill_switch = False
    mean_pnl, sample_count = store.recent_mean_pnl(KILL_SWITCH_SAMPLE, artifacts.quality_cutoff)
    if sample_count == KILL_SWITCH_SAMPLE and mean_pnl is not None and mean_pnl < 0:
        kill_switch = True

    _log_event("replay_start", candle_db=str(candle_db_path))

    for candle in stream.fetch_all():
        active_last_3 = activity.update(candle.market_ticker, candle.trade_active)
        if candle.close == candle.close:
            last_prices[candle.market_ticker] = candle.close

        position = positions.get(candle.market_ticker)
        if position:
            reason, exit_price, pnl = _exit_signal(
                position,
                candle.start_ts,
                candle.close,
                last_prices.get(candle.market_ticker),
            )
            if reason and exit_price is not None and pnl is not None:
                result = adapter.submit_exit(
                    ExitOrder(
                        order_key=_exit_order_key(position),
                        position=position,
                        exit_ts=candle.start_ts,
                        exit_price=exit_price,
                        exit_reason=reason,
                        pnl=pnl,
                    )
                )
                if result.accepted:
                    positions.pop(candle.market_ticker, None)
                if not kill_switch:
                    mean_pnl, sample_count = store.recent_mean_pnl(
                        KILL_SWITCH_SAMPLE, artifacts.quality_cutoff
                    )
                    if (
                        sample_count == KILL_SWITCH_SAMPLE
                        and mean_pnl is not None
                        and mean_pnl < 0
                    ):
                        kill_switch = True

        panic = pending.get(candle.market_ticker)
        if panic:
            if candle.start_ts >= panic.detected_ts + ENTRY_DELAY:
                allowed, quality_score, reasons = entry_decision(
                    panic=panic,
                    candle=candle,
                    active_last_3=active_last_3,
                    engine=engine,
                    positions=positions,
                    store=store,
                    kill_switch=kill_switch,
                )
                if allowed and not reasons:
                    side = "NO" if panic.direction == "UNDERDOG_UP" else "YES"
                    result = adapter.submit_entry(
                        EntryOrder(
                            order_key=_entry_order_key(panic),
                            market_ticker=candle.market_ticker,
                            side=side,
                            qty=1,
                            entry_ts=candle.start_ts,
                            entry_price=candle.close,
                            panic=panic,
                            quality_score=quality_score,
                        )
                    )
                    if result.accepted and result.order_id is not None:
                        positions[candle.market_ticker] = Position(
                            id=result.order_id,
                            market_ticker=candle.market_ticker,
                            side=side,
                            entry_ts=candle.start_ts,
                            entry_price=candle.close,
                            qty=1,
                        )
                pending.pop(candle.market_ticker, None)
            continue

        panic = engine.detect_panic(candle)
        if panic:
            pending[candle.market_ticker] = panic

    closed_pnls = store.closed_pnls()
    total_pnl = float(sum(pnl for _, pnl in closed_pnls))
    report = {"realized_pnl": total_pnl, "closed_trades": len(closed_pnls)}
    _log_event("replay_complete", **report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA phase5 engine with live-ready hardening.")
    parser.add_argument("--replay", action="store_true", help="Replay stored candles and exit.")
    args = parser.parse_args()

    if args.replay:
        report = run_replay()
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    run_loop()


if __name__ == "__main__":
    main()
