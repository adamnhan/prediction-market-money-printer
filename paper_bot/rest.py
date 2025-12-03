# paper_bot/rest.py

import requests
from typing import Optional, Dict, Any

from .config import KALSHI_API_BASE_URL


def request(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Minimal REST GET helper for Kalshi.
    Example: request("/markets", {"ticker": "CULTURE-XYZ"})
    """
    url = f"{KALSHI_API_BASE_URL}{endpoint}"
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()
