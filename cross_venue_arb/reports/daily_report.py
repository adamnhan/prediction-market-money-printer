"""Daily report scaffold."""

from __future__ import annotations

from datetime import date


def render_daily_report(run_date: date | None = None) -> str:
    target = run_date or date.today()
    return f"Daily report for {target.isoformat()}"
