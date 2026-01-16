from __future__ import annotations

import argparse
import uuid

from .config import load_config
from .phase5 import RestClient, _extract_orderbook_levels, _derive_best_prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Place a demo exit order for an open position.")
    parser.add_argument("--ticker", required=True, help="Market ticker to close.")
    parser.add_argument("--side", choices=["YES", "NO"], required=True, help="Side to sell.")
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
        key = f"{'yes' if args.side == 'YES' else 'no'}_bid"
        best = best_prices.get(key)
        if best is None:
            raise RuntimeError(f"no best bid available for {args.ticker} {args.side}")
        price = best / 100.0

    payload = {
        "ticker": args.ticker,
        "type": "limit",
        "action": "sell",
        "side": args.side.lower(),
        "count": args.qty,
        "client_order_id": f"manual-exit-{args.ticker}-{args.side}-{uuid.uuid4().hex[:8]}",
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
