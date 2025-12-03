# paper_bot/config.py

from pathlib import Path
from dotenv import load_dotenv
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)

DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRADE_LOG_PATH = DATA_DIR / "paper_trades_open_short_no.csv"
PNL_LOG_PATH = DATA_DIR / "paper_trades_open_short_no_pnl.csv"

KALSHI_API_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"



KALSHI_KEY_ID = os.getenv("KALSHI_KEY_ID")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")

TARGET_CATEGORIES = {"culture", "politics", "economics", "mentions"}
JUST_OPEN_WINDOW_SEC = 7 * 24 * 3600
