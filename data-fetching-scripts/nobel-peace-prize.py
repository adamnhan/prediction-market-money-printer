#!/usr/bin/env python3
"""
Fetch last 10 hours (t0) of Polymarket data for selected markets in an event.
Generates clearly named CSVs:
  nobel-2025_<person>_prices_yes_t0.csv
  nobel-2025_<person>_prices_no_t0.csv
  nobel-2025_<person>_trades_t0.csv
"""
import argparse, csv, json, os, re, sys, unicodedata
from datetime import datetime, timezone
from typing import Dict, List, Any
try:
    import requests
except ImportError:
    print("Install dependency first: pip install requests")
    sys.exit(1)

# --- endpoints ---
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
CLOB_PRICES_URL = "https://clob.polymarket.com/prices-history"
DATA_TRADES_URL = "https://data-api.polymarket.com/trades"

# --- helpers ---
def get_json(url, params=None):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def to_unix(ts: str) -> int:
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())

def normalize_text(s: str) -> str:
    if not s: return ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).casefold().strip()

def title_of(m: Dict[str, Any]) -> str:
    return m.get("question") or m.get("title") or m.get("slug") or ""

def slugify_name(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    clean = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    clean = clean.lower()
    clean = re.sub(r"[^a-z0-9]+", "-", clean).strip("-")
    return clean or "unknown"

def match_any_name(m: Dict[str,Any], needles: List[str]) -> bool:
    hay = normalize_text(title_of(m))
    return any(n in hay for n in needles)

def normalize_tokens(raw):
    if isinstance(raw, list): return [str(x) for x in raw]
    if isinstance(raw, str):
        try: return list(map(str, json.loads(raw)))
        except json.JSONDecodeError:
            return [s.strip().strip('"') for s in raw.strip("[]").split(",") if s.strip()]
    return []

def prices_history(token_id: str, start_ts: int, end_ts: int):
    params = {"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 1}
    data = get_json(CLOB_PRICES_URL, params=params)
    return data.get("history", [])

def page_trades_by_event(event_id: str, page_size=1000, max_rows=200000):
    out, offset = [], 0
    while True:
        params = {"eventId": event_id, "limit": min(page_size, max_rows - len(out)), "offset": offset}
        page = get_json(DATA_TRADES_URL, params=params)
        if not isinstance(page, list) or not page: break
        out.extend(page)
        if len(page) < page_size or len(out) >= max_rows: break
        offset += page_size
    for r in out:
        try: r["timestamp"] = int(r.get("timestamp", 0))
        except: r["timestamp"] = 0
    return out

def filter_trades_for_market(trades, market, start_ts, end_ts):
    cond = str(market.get("conditionId", "")).lower()
    out = []
    for r in trades:
        ts = int(r.get("timestamp", -1))
        if start_ts <= ts <= end_ts and str(r.get("market", "")).lower() == cond:
            out.append(r)
    return out

def write_prices_csv(outdir, event_slug, name_slug, token_label, rows):
    path = os.path.join(outdir, f"{event_slug}_{name_slug}_prices_{token_label}_t0.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unix","iso_utc","price"])
        for pt in rows:
            ts = int(pt["t"])
            iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            w.writerow([ts, iso, pt.get("p")])
    return path

def write_trades_csv(outdir, event_slug, name_slug, rows):
    path = os.path.join(outdir, f"{event_slug}_{name_slug}_trades_t0.csv")
    if not rows:
        return path
    cols = set()
    for r in rows: cols.update(r.keys())
    headers = sorted(cols)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader(); w.writerows(rows)
    return path

# --- main ---
def main():
    ap = argparse.ArgumentParser(description="Export last 10 hours (t0) for selected markets in a Polymarket event.")
    ap.add_argument("--event-id", required=True)
    ap.add_argument("--hours", type=int, default=10, help="Hours before close to include (default 10)")
    ap.add_argument("--names", default="maria corina machado,donald trump,greta thunberg",
                    help="Comma-separated names (accent-insensitive). Default includes Machado, Trump, Thunberg.")
    ap.add_argument("--skip-trades", action="store_true")
    args = ap.parse_args()

    needles = [normalize_text(s) for s in args.names.split(",") if s.strip()]
    outdir = f"event_{args.event_id}"
    os.makedirs(outdir, exist_ok=True)

    # Fetch event
    evt = get_json(GAMMA_EVENTS_URL, params={"id": args.event_id})
    ev = evt["data"][0] if isinstance(evt, dict) and "data" in evt else evt[0]
    event_slug = slugify_name(ev.get("title") or ev.get("slug") or f"event-{args.event_id}")
    markets = ev.get("markets", [])
    if not markets:
        print("No markets found in event."); sys.exit(1)

    sel = [m for m in markets if match_any_name(m, needles)]
    if not sel:
        print("No matching markets found."); sys.exit(1)

    all_trades = [] if args.skip_trades else page_trades_by_event(args.event_id)

    for m in sel:
        market_id = str(m.get("id"))
        title = title_of(m)
        name_slug = slugify_name(title)
        close_iso = m.get("closedTime") or m.get("endDate")
        if not close_iso:
            print(f"[skip] {market_id} missing close time"); continue

        close_unix = to_unix(close_iso)
        start_ts = close_unix - args.hours * 3600
        tokens = normalize_tokens(m.get("clobTokenIds"))

        print(f"\n=== {market_id} :: {title} ===")
        print(f"Close: {close_iso}  | Tokens: {tokens}")

        # prices
        labels = ["yes","no"] if len(tokens) >= 2 else [f"token{i}" for i in range(len(tokens))]
        for idx, token in enumerate(tokens):
            label = labels[idx] if idx < len(labels) else f"token{idx}"
            hist = prices_history(token, start_ts, close_unix)
            ppath = write_prices_csv(outdir, event_slug, name_slug, label, hist)
            print(f"  prices -> {ppath} ({len(hist)} points)")

        # trades
        if not args.skip_trades:
            mtrades = filter_trades_for_market(all_trades, m, start_ts, close_unix)
            tpath = write_trades_csv(outdir, event_slug, name_slug, mtrades)
            if mtrades:
                print(f"  trades -> {tpath} ({len(mtrades)} rows)")
            else:
                print("  trades -> none in last 10h")

    print("\nDone. Saved descriptive CSVs per market.")

if __name__ == "__main__":
    main()
