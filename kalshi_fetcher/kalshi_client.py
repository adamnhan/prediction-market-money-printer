# kalshi_fetcher/kalshi_client.py

import requests
from config import KALSHI_API_BASE_URL


def request(endpoint: str, params: dict | None = None) -> dict:
    """
    Make a GET request to the Kalshi API (public endpoints, no auth).
    
    Args:
        endpoint: API path starting with '/', e.g. '/markets'
        params: Optional query parameters as a dict

    Returns:
        Parsed JSON response as a Python dict.
    """
    url = f"{KALSHI_API_BASE_URL}{endpoint}"

    response = requests.get(url, params=params)
    response.raise_for_status()  # throw if HTTP error

    return response.json()

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
