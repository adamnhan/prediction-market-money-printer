from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
import websockets

if __package__ is None and str(Path(__file__).parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[2]))

load_dotenv()

from cross_venue_arb.config import CONFIG
from cross_venue_arb.storage.mapping_registry import read_game_mappings


def _resolve_asset_ids(limit: int) -> list[str]:
    records = read_game_mappings()
    asset_ids: list[str] = []
    for record in records:
        details = record.match_details or {}
        tokens = details.get("polymarket_asset_ids") or []
        for token in tokens:
            if token not in asset_ids:
                asset_ids.append(token)
        if len(asset_ids) >= limit:
            break
    return asset_ids[:limit]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket WS debug")
    parser.add_argument("--asset-id", action="append", help="Asset id to subscribe to")
    parser.add_argument("--limit", type=int, default=2, help="Asset ids to load from registry")
    parser.add_argument("--messages", type=int, default=5, help="Messages to print")
    args = parser.parse_args()

    ws_url = CONFIG.polymarket.ws_url or "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    ws_url = ws_url.rstrip("/")

    asset_ids = args.asset_id or _resolve_asset_ids(args.limit)
    if not asset_ids:
        raise SystemExit("No asset ids found. Run phase3 with --write-db or pass --asset-id.")

    payloads = [
        {"type": "market", "assets_ids": asset_ids},
        {"type": "market", "asset_ids": asset_ids},
    ]

    async with websockets.connect(ws_url) as websocket:
        print(f"connected url={ws_url}")
        for payload in payloads:
            await websocket.send(json.dumps(payload))
            print(f"sent {payload}")
        for _ in range(args.messages):
            message = await websocket.recv()
            print(message)


if __name__ == "__main__":
    asyncio.run(main())
