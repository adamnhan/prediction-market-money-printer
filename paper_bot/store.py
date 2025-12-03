# paper_bot/store.py

from dataclasses import dataclass, asdict
from pathlib import Path
import csv

from .config import TRADE_LOG_PATH


@dataclass
class TradeRow:
    timestamp: str        # ISO string
    market_ticker: str
    category: str
    open_time: str        # ISO string
    entry_price_yes: float
    entry_price_no: float
    entry_side: str       # always "NO" for this strategy


def append_trade(row: TradeRow, path: Path = TRADE_LOG_PATH):
    """
    Append a single trade row to the trade log CSV.
    Creates the file and header if they don't already exist.
    """
    write_header = not path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(row).keys())
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(row))
