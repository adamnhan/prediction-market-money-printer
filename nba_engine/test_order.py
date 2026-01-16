from __future__ import annotations

import argparse
import os

from .config import load_config
from .phase5 import RestClient, _extract_orderbook_levels, _derive_best_prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Place a demo limit order for endpoint testing.")
    parser.add_argument("--ticker", required=True, help="Market ticker to trade.")
    parser.add_argument("--side", choices=["YES", "NO"], required=True, help="Side to buy/sell.")
    parser.add_argument("--action", choices=["buy", "sell"], default="buy", help="Order action.")
    parser.add_argument("--qty", type=int, default=1, help="Contract count.")
    parser.add_argument("--price", type=float, default=None, help="Limit price in dollars.")
    args = parser.parse_args()

    config = load_config()
    if not config.kalshi_rest_url:
        raise ValueError("KALSHI_REST_URL is required")

    client = RestClient(
        config.kalshi_rest_url,
        config.kalshi_key_id,
        config.kalshi_private_key_path,
        order_url="https://demo-api.kalshi.co/trade-api/v2",
    )

    price = args.price
    if price is None:
        book = client.get_orderbook(args.ticker)
        levels = _extract_orderbook_levels(book)
        best_prices = _derive_best_prices(levels)
        key = f"{'yes' if args.side == 'YES' else 'no'}_{'ask' if args.action == 'buy' else 'bid'}"
        best = best_prices.get(key)
        if best is None:
            raise RuntimeError(f"no best price available for {args.ticker} {args.side} {args.action}")
        price = best / 100.0

    payload = {
        "ticker": args.ticker,
        "type": "limit",
        "action": args.action,
        "side": args.side.lower(),
        "count": args.qty,
        "client_order_id": f"manual-test-{args.ticker}-{args.side}-{args.action}",
    }
    price_cents = int(round(price * 100))
    if args.side == "YES":
        payload["yes_price"] = price_cents
    else:
        payload["no_price"] = price_cents

    response = client.place_order(payload)
    print(response)


if __name__ == "__main__":
    main()
