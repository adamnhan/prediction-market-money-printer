from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

if __package__ is None and str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from cross_venue_arb.books.manager import BookManager

from market_maker.config import load_config, load_markets
from market_maker.kalshi_ws_client import WsMetrics, run as kalshi_ws_run
from market_maker.shadow_mm import quote_loop


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shadow market maker runner")
    parser.add_argument(
        "--config",
        default="config/shadow_mm.yaml",
        help="Path to shadow MM YAML config",
    )
    return parser


async def _run(config_path: str) -> None:
    config = load_config(config_path)
    tickers = load_markets(config.markets_file)
    if not tickers:
        raise SystemExit(f"No markets found in {config.markets_file}")

    manager = BookManager(stale_after_s=config.staleness_s)
    metrics = WsMetrics()

    await asyncio.gather(
        kalshi_ws_run(manager, tickers, metrics),
        quote_loop(manager, tickers, config, metrics),
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
