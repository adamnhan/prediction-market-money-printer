from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class Artifacts:
    vol_quantiles: dict[str, float]
    volsum_quantiles: dict[str, float]
    zscore_mean: float
    zscore_std: float
    quality_cutoff: float
    zscore_features: dict[str, dict[str, float]] | None = None

    def summary(self) -> str:
        return (
            "artifacts "
            f"vol_quantiles={self.vol_quantiles} "
            f"volsum_quantiles={self.volsum_quantiles} "
            f"zscore_mean={self.zscore_mean} "
            f"zscore_std={self.zscore_std} "
            f"quality_cutoff={self.quality_cutoff} "
            f"zscore_features={list(self.zscore_features or {})}"
        )


def _ensure_number_map(value: object, name: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict")
    parsed: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings")
        if not isinstance(raw, (int, float)):
            raise ValueError(f"{name}[{key}] must be a number")
        parsed[key] = float(raw)
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed


def load_artifacts(path: Path | None = None) -> Artifacts:
    path = path or Path("strategy_artifacts.json")
    if not path.exists():
        raise FileNotFoundError(f"missing artifacts file: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("artifacts must be a json object")

    vol_quantiles = _ensure_number_map(raw.get("vol_quantiles"), "vol_quantiles")
    volsum_quantiles = _ensure_number_map(
        raw.get("volsum_quantiles"), "volsum_quantiles"
    )
    zscore = raw.get("zscore")
    if not isinstance(zscore, dict):
        raise ValueError("zscore must be a dict")
    zscore_mean = zscore.get("mean")
    zscore_std = zscore.get("std")
    if not isinstance(zscore_mean, (int, float)):
        raise ValueError("zscore.mean must be a number")
    if not isinstance(zscore_std, (int, float)):
        raise ValueError("zscore.std must be a number")
    quality_cutoff = raw.get("quality_cutoff")
    if not isinstance(quality_cutoff, (int, float)):
        raise ValueError("quality_cutoff must be a number")

    zscore_features_raw = raw.get("zscore_features")
    zscore_features: dict[str, dict[str, float]] | None = None
    if zscore_features_raw is not None:
        if not isinstance(zscore_features_raw, dict):
            raise ValueError("zscore_features must be a dict")
        zscore_features = {}
        for key, stats in zscore_features_raw.items():
            if not isinstance(key, str):
                raise ValueError("zscore_features keys must be strings")
            if not isinstance(stats, dict):
                raise ValueError(f"zscore_features[{key}] must be a dict")
            mean = stats.get("mean")
            std = stats.get("std")
            if not isinstance(mean, (int, float)) or not isinstance(std, (int, float)):
                raise ValueError(f"zscore_features[{key}] must have mean/std numbers")
            zscore_features[key] = {"mean": float(mean), "std": float(std)}

    return Artifacts(
        vol_quantiles=vol_quantiles,
        volsum_quantiles=volsum_quantiles,
        zscore_mean=float(zscore_mean),
        zscore_std=float(zscore_std),
        quality_cutoff=float(quality_cutoff),
        zscore_features=zscore_features,
    )
