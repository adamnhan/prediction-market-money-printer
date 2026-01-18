CREATE TABLE IF NOT EXISTS arb_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    market_id TEXT NOT NULL,
    left_venue TEXT NOT NULL,
    right_venue TEXT NOT NULL,
    edge_bps REAL NOT NULL
);
