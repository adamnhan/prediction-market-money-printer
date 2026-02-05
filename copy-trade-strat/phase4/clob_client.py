from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


_IMPORT_ERROR: Exception | None = None
try:
    from py_clob_client.client import ClobClient
except Exception as exc:  # pragma: no cover - optional dependency
    _IMPORT_ERROR = exc
    ClobClient = None

try:  # Optional enums; API can accept strings in newer versions.
    from py_clob_client.clob_types import OrderType, Side  # type: ignore
except Exception:
    OrderType = None
    Side = None


DEFAULT_CLOB_HOST = "https://clob.polymarket.com"


@dataclass
class ClobConfig:
    host: str = DEFAULT_CLOB_HOST
    chain_id: int = 137
    api_key: str | None = None
    api_secret: str | None = None
    api_passphrase: str | None = None
    private_key: str | None = None
    funder: str | None = None
    signature_type: int | None = None


def load_config_from_env() -> ClobConfig:
    return ClobConfig(
        host=os.getenv("POLYMARKET_CLOB_HOST", DEFAULT_CLOB_HOST),
        chain_id=int(os.getenv("POLYMARKET_CHAIN_ID", "137")),
        api_key=os.getenv("POLYMARKET_API_KEY"),
        api_secret=os.getenv("POLYMARKET_API_SECRET"),
        api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE"),
        private_key=os.getenv("POLYMARKET_PRIVATE_KEY"),
        funder=os.getenv("POLYMARKET_FUNDER"),
        signature_type=int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1")),
    )


class ClobWrapper:
    def __init__(self, cfg: ClobConfig) -> None:
        if ClobClient is None:
            raise RuntimeError(f"py-clob-client import failed: {_IMPORT_ERROR}")
        self.cfg = cfg
        try:
            self.client = ClobClient(
                host=cfg.host,
                chain_id=cfg.chain_id,
                private_key=cfg.private_key,
                funder=cfg.funder,
                signature_type=cfg.signature_type,
            )
        except TypeError:
            # Older py-clob-client versions use different init args.
            self.client = ClobClient(
                host=cfg.host,
                chain_id=cfg.chain_id,
            )
        if cfg.api_key and cfg.api_secret and cfg.api_passphrase:
            self.client.set_api_creds(cfg.api_key, cfg.api_secret, cfg.api_passphrase)

    def get_orderbook(self, token_id: str) -> dict[str, Any]:
        # Best-effort API naming; py-clob-client exposes get_order_book.
        if hasattr(self.client, "get_order_book"):
            return self.client.get_order_book(token_id)
        return self.client.get_orderbook(token_id)

    def place_limit_order(self, token_id: str, side: str, price: float, qty: float) -> dict[str, Any]:
        side_value: Any = side.upper()
        order_type_value: Any = "GTC"
        if Side is not None:
            side_value = Side.BUY if side.upper() == "BUY" else Side.SELL
        if OrderType is not None:
            order_type_value = OrderType.GTC
        order = self.client.create_order(
            token_id=token_id,
            price=price,
            size=qty,
            side=side_value,
            order_type=order_type_value,
        )
        return self.client.post_order(order)

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        if hasattr(self.client, "get_order"):
            return self.client.get_order(order_id)
        return None

    def cancel(self, order_id: str) -> dict[str, Any] | None:
        if hasattr(self.client, "cancel"):
            return self.client.cancel(order_id)
        if hasattr(self.client, "cancel_order"):
            return self.client.cancel_order(order_id)
        return None


class RestOrderbookClient:
    def __init__(self, host: str = DEFAULT_CLOB_HOST) -> None:
        import requests

        self.host = host.rstrip("/")
        self.session = requests.Session()

    def get_orderbook(self, token_id: str) -> dict[str, Any]:
        url = f"{self.host}/book"
        resp = self.session.get(url, params={"token_id": token_id}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def place_limit_order(self, token_id: str, side: str, price: float, qty: float) -> dict[str, Any]:
        raise RuntimeError("REST client is read-only; install py-clob-client for order placement")

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        return None

    def cancel(self, order_id: str) -> dict[str, Any] | None:
        return None
