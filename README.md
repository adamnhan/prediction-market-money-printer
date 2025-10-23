# prediction-market-money-printer

Project scaffold for Kalshi data collection, anomaly detection, and automated trading.

Layout
- src/: core async clients and helpers (e.g., `src/platform_ops.py`)
- data-fetching-scripts/: backward-compatible scripts and shims
- tests/: unit tests

Quick TODOs
- [ ] Add integration tests for the Kalshi API (record and replay via VCR or similar)
- [ ] Implement sync wrappers for convenience in scripts that expect blocking calls
- [ ] Implement websocket-based market and order subscriptions for live trading
- [ ] Add proper CI (pytest run + linting)
- [ ] Securely manage API keys (use secrets manager or environment variables)

Usage

macOS / Linux

1. Create and activate a virtual environment (only required once):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run tests and open coverage report (single command):

```bash
# optional: install deps as part of the run
bash scripts/run_tests.sh install

# or run without installing deps
bash scripts/run_tests.sh
```

Windows (PowerShell)

1. Create and activate a virtual environment (only required once):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run tests and open coverage report (single command):

```powershell
# optional: install deps as part of the run
.\scripts\run_tests.ps1 -InstallDeps

# or run without installing deps
.\scripts\run_tests.ps1
```

Run a quick demo (async helpers are exposed in `src.platform_ops`):

```bash
python -c "import asyncio, src.platform_ops as po; print(asyncio.run(po.get_events(limit=1)))"
```

Notes

- The code currently uses `httpx` (async). If you need a synchronous implementation, add thin sync wrappers.
