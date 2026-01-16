from __future__ import annotations

from .artifacts import load_artifacts
from .config import load_config


def main() -> None:
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


if __name__ == "__main__":
    main()
