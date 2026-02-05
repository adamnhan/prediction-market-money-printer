# kalshi_fetcher/kalshi_client.py

import requests
from kalshi_fetcher.config import KALSHI_API_BASE_URL

DEFAULT_TIMEOUT_S = 15


def request(endpoint: str, params: dict | None = None, timeout_s: int | float = DEFAULT_TIMEOUT_S) -> dict:
    """
    Make a GET request to the Kalshi API (public endpoints, no auth).
    
    Args:
        endpoint: API path starting with '/', e.g. '/markets'
        params: Optional query parameters as a dict

    Returns:
        Parsed JSON response as a Python dict.
    """
    url = f"{KALSHI_API_BASE_URL}{endpoint}"

    response = requests.get(url, params=params, timeout=timeout_s)
    response.raise_for_status()  # throw if HTTP error

    return response.json()

def get_event_metadata(event_ticker: str) -> dict:
    """
    Fetch metadata for a single event from /events/{event_ticker}.
    Returns the 'event' object from the API response.
    """
    data = request(f"/events/{event_ticker}")
    # API shape is usually { "event": { ... } }
    return data.get("event", {})

def get_markets():
    """
    Fetch all markets from the Kalshi API and return them.
    """
    data = request("/markets")
    markets = data.get("markets", [])
    print(f"Fetched {len(markets)} markets.")
    return markets


if __name__ == "__main__":
    get_markets()
