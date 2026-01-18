"""CLI entrypoint for cross-venue arb."""

from __future__ import annotations

import argparse

from cross_venue_arb.matching import deterministic, registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-venue arbitrage runner")
    parser.add_argument("--matcher", default="deterministic", help="Matching strategy")
    args = parser.parse_args()

    registry.register("deterministic", deterministic.match_markets)

    matcher = registry.get(args.matcher)
    matches = matcher([])
    print(f"Loaded matcher: {args.matcher} (matches={len(matches)})")


if __name__ == "__main__":
    main()
