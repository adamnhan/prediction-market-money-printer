// ui/components/EventMarketsViewer.tsx
"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";

type EventSummary = {
  ticker: string;
  title: string;
};

type EventMarketsViewerProps = {
  events: EventSummary[];
};

type EventMarket = {
  market_ticker: string | null;
  title: string | null;
  status: string | null;
};

export function EventMarketsViewer({ events }: EventMarketsViewerProps) {
    const [selectedEvent, setSelectedEvent] = useState<string>("");
    const [markets, setMarkets] = useState<EventMarket[]>([]);
    const [loading, setLoading] = useState(false);
    const [attachingTicker, setAttachingTicker] = useState<string | null>(null);
  
    async function loadMarkets() {
        if (!selectedEvent) return;

        try {
        setLoading(true);
        const res = await fetch(
            `http://127.0.0.1:8000/events/${encodeURIComponent(
            selectedEvent
            )}/markets`
        );
        const data = await res.json();

        setMarkets(
            (data.markets ?? []).map((m: any) => ({
            market_ticker: m.market_ticker ?? null,
            title: m.title ?? null,
            status: m.status ?? null,
            }))
        );
        } catch (err) {
            console.error("Failed to load markets for event:", err);
            setMarkets([]);
        } finally {
            setLoading(false);
        }
    }

    async function attachMarket(marketTicker: string | null) {
        if (!selectedEvent || !marketTicker) return;

        try {
        setAttachingTicker(marketTicker);

        await fetch("http://127.0.0.1:8000/markets", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
            event_ticker: selectedEvent.toUpperCase(),
            market_ticker: marketTicker.toUpperCase(),
            }),
        });

        // We rely on the main page's polling / manual refresh
        // to pick up the updated engine state.
        } catch (err) {
        console.error("Failed to attach market:", err);
        } finally {
        setAttachingTicker(null);
        }
    }


  return (
    <section className="mt-4 rounded-lg border border-slate-800 bg-slate-900/40 p-4">
      <h2 className="mb-2 text-sm font-semibold text-slate-100">
        Event Markets
      </h2>
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <select
          className="flex-1 rounded-md border border-slate-800 bg-slate-950 px-2 py-1 text-sm text-slate-100 outline-none"
          value={selectedEvent}
          onChange={(e) => setSelectedEvent(e.target.value)}
        >
          <option value="">Select an event…</option>
          {events.map((ev) => (
            <option key={ev.ticker} value={ev.ticker}>
              {ev.ticker} — {ev.title}
            </option>
          ))}
        </select>
        <Button
          type="button"
          size="sm"
          className="mt-2 w-full sm:mt-0 sm:w-auto"
          onClick={loadMarkets}
          disabled={!selectedEvent || loading}
        >
          {loading ? "Loading…" : "Load markets"}
        </Button>
      </div>

      <div className="mt-3 max-h-60 overflow-auto text-sm">
        {loading && <p className="text-slate-400">Fetching markets…</p>}

        {!loading && markets.length === 0 && (
          <p className="text-xs text-slate-500">
            No markets loaded yet. Select an event and click &quot;Load
            markets&quot;.
          </p>
        )}

        {!loading && markets.length > 0 && (
          <ul className="space-y-1">
            {markets.map((m, idx) => (
              <li
                key={`${m.market_ticker ?? "unknown"}-${idx}`}
                className="flex items-start justify-between gap-2 rounded border border-slate-800 bg-slate-950 px-2 py-1"
              >
                <div>
                  <div className="font-mono text-xs text-slate-200">
                    {m.market_ticker ?? "(no ticker)"}
                  </div>
                  <div className="text-xs text-slate-300">
                    {m.title ?? "(no title)"}
                  </div>
                  <div className="text-[11px] text-slate-500">
                    status: {m.status ?? "unknown"}
                  </div>
                </div>
                <Button
                  type="button"
                  size="sm"
                  className="mt-1"
                  disabled={!m.market_ticker || attachingTicker === m.market_ticker}
                  onClick={() => attachMarket(m.market_ticker)}
                >
                  {attachingTicker === m.market_ticker ? "Attaching…" : "Attach"}
                </Button>
              </li>
            ))}

          </ul>
        )}
      </div>
    </section>
  );
}
