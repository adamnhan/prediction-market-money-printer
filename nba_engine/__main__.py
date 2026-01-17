from __future__ import annotations

import argparse

from .artifacts import load_artifacts
from .config import load_config
from .phase5_launcher import main as launch_phase5


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA engine entrypoint.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print artifacts summary and exit (do not start phase1/phase5).",
    )
    args, _ = parser.parse_known_args()

    load_config()
    artifacts = load_artifacts()
    print(artifacts.summary())
    print(
        "ready "
        f"vol_quantiles={artifacts.vol_quantiles} "
        f"volsum_quantiles={artifacts.volsum_quantiles} "
        f"zscore_mean={artifacts.zscore_mean} "
        f"zscore_std={artifacts.zscore_std} "
        f"quality_cutoff={artifacts.quality_cutoff}"
    )

    if not args.summary_only:
        launch_phase5()


if __name__ == "__main__":
    main()
