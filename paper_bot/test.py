# paper_bot/auth_test.py

import time
import base64
import requests

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

from .config import KALSHI_API_BASE_URL, KALSHI_KEY_ID, KALSHI_PRIVATE_KEY_PATH
from .entry_bot import _load_private_key   # reuse same loader


def _sign_pss_text(private_key, text: str) -> str:
    message = text.encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def main():
    if not KALSHI_KEY_ID:
        raise RuntimeError("KALSHI_KEY_ID is not set")

    if not KALSHI_PRIVATE_KEY_PATH:
        raise RuntimeError("KALSHI_PRIVATE_KEY_PATH is not set")

    private_key = _load_private_key()

    # We’ll hit the /api_keys endpoint to verify auth works
    # paper_bot/auth_test.py

    path = "/trade-api/v2/api_keys"
    timestamp = str(int(time.time() * 1000))
    msg_string = timestamp + "GET" + path

    signature = _sign_pss_text(private_key, msg_string)

    headers = {
        "KALSHI-ACCESS-KEY": KALSHI_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }

    # base_host from new API base URL
    base_host = KALSHI_API_BASE_URL.split("/trade-api")[0]
    url = base_host + path


    print("Testing auth against:", url)
    resp = requests.get(url, headers=headers)
    print("Status:", resp.status_code)
    print("Body:", resp.text)


if __name__ == "__main__":
    main()
