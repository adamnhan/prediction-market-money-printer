#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


class TradeLogger:
    def __init__(self, db_path: str, jsonl_path: str | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bundles (
                bundle_id TEXT PRIMARY KEY,
                run_id TEXT,
                ts_signal INTEGER,
                ts_decision INTEGER,
                event_key TEXT,
                series_ticker TEXT,
                mode TEXT,
                decision TEXT,
                reasons_json TEXT,
                mid_band_L INTEGER,
                mid_band_U INTEGER,
                max_loss REAL,
                mid_worst REAL,
                ev_raw REAL,
                ev_net_est REAL,
                snapshot_json TEXT,
                created_at INTEGER
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bundle_legs (
                bundle_id TEXT,
                leg_idx INTEGER,
                ticker TEXT,
                k INTEGER,
                side TEXT,
                qty INTEGER,
                limit_price REAL,
                px_bid REAL,
                px_ask REAL,
                px_used REAL,
                phat REAL,
                delta REAL,
                fill_qty INTEGER,
                fill_price REAL,
                fill_ts INTEGER,
                PRIMARY KEY (bundle_id, leg_idx)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events_settlement (
                event_key TEXT PRIMARY KEY,
                final_margin INTEGER,
                settle_ts INTEGER,
                source TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bundle_settlements (
                bundle_id TEXT PRIMARY KEY,
                event_key TEXT,
                final_margin INTEGER,
                pnl_realized REAL,
                computed_ts INTEGER
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bundle_skips (
                id INTEGER PRIMARY KEY,
                ts INTEGER NOT NULL,
                event_key TEXT,
                reason TEXT,
                details_json TEXT
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _emit_json(self, payload: dict[str, Any]) -> None:
        if not self.jsonl_path:
            return
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))
            handle.write("\n")

    def log_bundle(self, bundle: dict[str, Any]) -> str:
        bundle_id = bundle.get("bundle_id") or uuid.uuid4().hex
        bundle["bundle_id"] = bundle_id
        now = int(time.time())
        self.conn.execute(
            """
            INSERT OR REPLACE INTO bundles (
                bundle_id, run_id, ts_signal, ts_decision, event_key, series_ticker,
                mode, decision, reasons_json, mid_band_L, mid_band_U, max_loss, mid_worst,
                ev_raw, ev_net_est, snapshot_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bundle_id,
                bundle.get("run_id"),
                bundle.get("ts_signal"),
                bundle.get("ts_decision"),
                bundle.get("event_key"),
                bundle.get("series_ticker"),
                bundle.get("mode"),
                bundle.get("decision"),
                json.dumps(bundle.get("reasons") or [], separators=(",", ":"), ensure_ascii=True),
                bundle.get("mid_band_L"),
                bundle.get("mid_band_U"),
                bundle.get("max_loss"),
                bundle.get("mid_worst"),
                bundle.get("ev_raw"),
                bundle.get("ev_net_est"),
                json.dumps(bundle.get("snapshot") or {}, separators=(",", ":"), ensure_ascii=True),
                now,
            ),
        )
        self.conn.commit()
        self._emit_json({"type": "bundle_recorded", **bundle})
        return bundle_id

    def log_bundle_legs(self, bundle_id: str, legs: list[dict[str, Any]]) -> None:
        for idx, leg in enumerate(legs):
            self.conn.execute(
                """
                INSERT OR REPLACE INTO bundle_legs (
                    bundle_id, leg_idx, ticker, k, side, qty, limit_price,
                    px_bid, px_ask, px_used, phat, delta, fill_qty, fill_price, fill_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bundle_id,
                    idx,
                    leg.get("ticker"),
                    leg.get("k"),
                    leg.get("side"),
                    leg.get("qty"),
                    leg.get("limit_price"),
                    leg.get("px_bid"),
                    leg.get("px_ask"),
                    leg.get("px_used"),
                    leg.get("phat"),
                    leg.get("delta"),
                    leg.get("fill_qty"),
                    leg.get("fill_price"),
                    leg.get("fill_ts"),
                ),
            )
        self.conn.commit()
        self._emit_json({"type": "bundle_legs_recorded", "bundle_id": bundle_id, "legs": legs})

    def log_fills(self, bundle_id: str, event_key: str, fills: list[dict[str, Any]]) -> None:
        if not fills:
            return
        record = {
            "type": "bundle_fills",
            "bundle_id": bundle_id,
            "event_key": event_key,
            "fills": fills,
        }
        self._emit_json(record)
        print(record, flush=True)

    def log_heartbeat(self, payload: dict[str, Any]) -> None:
        record = {"type": "heartbeat", **payload}
        self._emit_json(record)
        print(record, flush=True)

    def log_error(self, payload: dict[str, Any]) -> None:
        record = {"type": "error", **payload}
        self._emit_json(record)
        print(record, flush=True)

    def log_skip(self, payload: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO bundle_skips (ts, event_key, reason, details_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                payload.get("ts"),
                payload.get("event_key"),
                payload.get("reason"),
                json.dumps(payload.get("details") or {}, separators=(",", ":"), ensure_ascii=True),
            ),
        )
        self.conn.commit()
        self._emit_json({"type": "bundle_skip", **payload})
