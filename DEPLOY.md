# Deployment Prep (Phase D1)

Canonical start command
- `python -m trading_engine` - boots uvicorn against `trading_engine.main:app`, runs the trading engine, restores control state, and starts the WebSocket loop via FastAPI startup hooks.
- Defaults: `APP_HOST=0.0.0.0`, `APP_PORT=8000`, single worker (in-memory engine/WS state must stay in one process). Use `APP_RELOAD=1` only for local dev.

Environment variables
- `KALSHI_KEY_ID` (required): Kalshi API key id for WS auth.
- `KALSHI_PRIVATE_KEY_PATH` (required): filesystem path to the PEM used for signing WS auth.
- `KALSHI_WS_URL` (optional, default `wss://api.elections.kalshi.com/trade-api/ws/v2`): override for non-elections environments.
- `APP_HOST` (optional, default `0.0.0.0`): uvicorn bind host.
- `APP_PORT` (optional, default `8000`): uvicorn port.
- `APP_RELOAD` (optional, default off): set to `1/true/yes` to enable autoreload in dev.
- `ENGINE_STATE_DB_PATH` (optional, default `./data/engine_state.sqlite`): path for engine control-state persistence.
- Strategy overrides (optional; defaults in code): `STRATEGY_MAX_OPEN_POSITIONS`, `STRATEGY_MAX_CAPITAL_PER_POSITION_PCT`, `STRATEGY_MAX_CAPITAL_PER_MARKET_PCT`, `STRATEGY_MAX_TOTAL_EXPOSURE_PCT`, `STRATEGY_MAX_NO_ENTRY_PRICE`, `STRATEGY_TAKE_PROFIT_PCT`, `STRATEGY_STOP_LOSS_PCT`, `STRATEGY_MAX_HOLD_SECONDS`, `STRATEGY_DAILY_LOSS_LIMIT`, `STRATEGY_MAX_DRAWDOWN`, `STRATEGY_MAX_TRADES_PER_DAY`, `STRATEGY_COOLDOWN_MINUTES_AFTER_STOP`.

Local run (example)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:KALSHI_KEY_ID="your_key_id"
$env:KALSHI_PRIVATE_KEY_PATH="kalshi_private_key.pem"
$env:ENGINE_STATE_DB_PATH="data/engine_state.sqlite"
# Optional strategy tuning:
# $env:STRATEGY_MAX_OPEN_POSITIONS="500"
python -m trading_engine
```
Or place the env vars in a local `.env` alongside the code.

Health check
- Endpoint: `GET /health`
- Behavior: fast, no DB access; returns HTTP 200 when WS is connected and not stale; returns HTTP 500 when disconnected or stale.
- Fields: `engine_running` (bool), `ws_connected` (bool), `ws_last_connect_ts`, `ws_last_message_ts`, `ws_last_error`, `ws_stale` (bool if last message >60s ago), `ws_stale_seconds`, `ws_subscriptions` (count).
- Suitable for Docker/systemd health probes; treat `ws_stale=true` or `ws_connected=false` as degraded.

Persistence
- Control state (operator flags, attached markets, etc.) persists to `data/engine_state.sqlite` on shutdown and resumes on startup. Ensure the process can write to `data/` in deployment.

Docker (local)
- Build: `docker build -t kalshi-bot .`
- Run: `docker run -p 8000:8000 --env-file .env kalshi-bot`
- Health: `curl -f http://localhost:8000/health` (expects 200 when WS is healthy; 500 when disconnected/stale)
- To persist control state across restarts, mount a host dir for `data/`: `docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data kalshi-bot`

Docker Compose (local)
- Up: `docker compose up --build`
- Uses `.env` for credentials/config.
- Mounts `./data` to persist `engine_state.sqlite` and `trade_ledger.sqlite` (`ENGINE_STATE_DB_PATH` and `TRADE_LEDGER_DB_PATH` set to `/data/...` in compose).
- Restart policy: unless-stopped. Healthcheck: HTTP GET `http://localhost:8000/health` (200 healthy, 500 degraded). Logs rotated via json-file (10m, 3 files).
