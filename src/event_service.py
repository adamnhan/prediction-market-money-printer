# src/event_service.py
from typing import Dict, Any
from src import platform_ops  # or: from . import platform_ops


async def fetch_event_metadata(event_ticker: str) -> Dict[str, Any]:
    """
    Fetch and normalize event metadata for a single Kalshi event.

    Uses platform_ops.get_event(), which returns:
      { "ok": bool, "status": int, "data": {...} } on success
      { "ok": False, "status": int|None, "error": "..." } on failure
    """

    resp = await platform_ops.get_event(event_ticker)  # 👈 no api_key passed

    # 1) Handle transport / API-level errors
    if not resp.get("ok"):
        return {
            "ok": False,
            "error": f"Kalshi get_event failed (status={resp.get('status')}): {resp.get('error')}",
        }

    data = resp.get("data")
    if not isinstance(data, dict):
        return {
            "ok": False,
            "error": f"Unexpected data type from get_event: {type(data)}",
        }

    # Kalshi usually returns {"event": {...}}, but fall back to data itself
    event_obj = data.get("event") or data
    if not isinstance(event_obj, dict):
        return {
            "ok": False,
            "error": f"No 'event' object in response: {data}",
        }

    normalized = {
        "ticker": event_obj.get("ticker", event_ticker),
        "title": event_obj.get("title") or event_ticker,
        "status": event_obj.get("status"),
        "category": event_obj.get("category"),
        "sub_category": event_obj.get("sub_category"),
        "start_ts": event_obj.get("start_ts"),
        "end_ts": event_obj.get("end_ts"),
    }

    return {"ok": True, "event": normalized}
