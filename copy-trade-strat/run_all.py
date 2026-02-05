from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PHASES = [
    ROOT / "phase1" / "run.py",
    ROOT / "phase2" / "run.py",
    ROOT / "phase3" / "run.py",
    ROOT / "phase4" / "run.py",
]


def main() -> None:
    procs: list[subprocess.Popen] = []
    for script in PHASES:
        procs.append(subprocess.Popen([sys.executable, str(script)]))
        time.sleep(0.2)

    try:
        while True:
            time.sleep(1)
            for proc in procs:
                if proc.poll() is not None:
                    raise RuntimeError(f"phase exited: pid={proc.pid}")
    except KeyboardInterrupt:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait(timeout=5)


if __name__ == "__main__":
    main()
