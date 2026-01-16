from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Config:
    kalshi_key_id: str
    kalshi_private_key_path: Path
    kalshi_ws_url: str
    kalshi_rest_url: str | None


def _parse_env_file(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        raise FileNotFoundError(f"missing env file: {env_path}")
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def load_config(env_path: Path | None = None) -> Config:
    env_path = env_path or Path(".env")
    file_values = _parse_env_file(env_path)
    merged = {**file_values, **os.environ}

    required = ["KALSHI_KEY_ID", "KALSHI_PRIVATE_KEY_PATH", "KALSHI_WS_URL"]
    missing = [key for key in required if not merged.get(key)]
    if missing:
        raise ValueError(f"missing required env vars: {', '.join(missing)}")

    key_path = Path(merged["KALSHI_PRIVATE_KEY_PATH"])
    if not key_path.is_absolute():
        key_path = (Path.cwd() / key_path).resolve()
    if not key_path.exists():
        raise FileNotFoundError(f"missing private key file: {key_path}")

    return Config(
        kalshi_key_id=merged["KALSHI_KEY_ID"],
        kalshi_private_key_path=key_path,
        kalshi_ws_url=merged["KALSHI_WS_URL"],
        kalshi_rest_url=merged.get("KALSHI_REST_URL"),
    )
