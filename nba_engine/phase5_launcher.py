from __future__ import annotations

import argparse
import asyncio
import threading

import uvicorn

from . import phase1
from . import phase5
from .phase5_dashboard import app


def _run_phase1() -> None:
    asyncio.run(phase1.run_live())


def _run_phase5() -> None:
    phase5.run_loop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 launcher (phase1 + phase5 + dashboard).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8015)
    args = parser.parse_args()

    phase1_thread = threading.Thread(target=_run_phase1, daemon=True)
    phase5_thread = threading.Thread(target=_run_phase5, daemon=True)
    phase1_thread.start()
    phase5_thread.start()

    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
