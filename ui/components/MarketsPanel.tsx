// ui/components/MarketsPanel.tsx
"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";

type MarketEntry = {
  event_ticker: string;
  market_ticker: string;
  status: string;
  last_price_yes: number | null;
  last_price_no: number | null;
};

type MarketsPanelProps = {
  markets: Record<string, MarketEntry>;
  loading: boolean;
};

export function MarketsPanel({ markets, loading }: MarketsPanelProps) {
  const [busyTicker, setBusyTicker] = useState<string | null>(null);

  async function handleRefreshAndTrade(marketTicker: string) {
    try {
      setBusyTicker(marketTicker);

      await fetch(
        `http://127.0.0.1:8000/markets/${encodeURIComponent(
          marketTicker
        )}/refresh_and_trade`,
        {
          method: "POST",
        }
      );

      // We rely on the main page's polling / manual Refresh button
      // to pick up updated state (positions, prices, etc.).
    } catch (err) {
      console.error("Failed to refresh & trade NO:", err);
    } finally {
      setBusyTicker(null);
    }
  }

  const marketList = Object.values(markets ?? {});

  return (
    <section className="mt-4 rounded-lg border border-slate-800 bg-slate-900/40 p-4">
      <h2 className="mb-2 text-sm font-semibold text-slate-100">
        Attached Markets
      </h2>

      {loading && (
        <p className="text-sm text-slate-400">Loading markets from engine…</p>
      )}

      {!loading && marketList.length === 0 && (
        <p className="text-xs text-slate-500">
          No markets attached yet. Use &quot;Event Markets&quot; to attach
          one.
        </p>
      )}

      {!loading && marketList.length > 0 && (
        <ul className="space-y-2 text-sm">
          {marketList.map((m) => (
            <li
              key={m.market_ticker}
              className="flex items-start justify-between gap-2 rounded border border-slate-800 bg-slate-950 px-2 py-1"
            >
              <div>
                <div className="font-mono text-xs text-slate-200">
                  {m.market_ticker}
                </div>
                <div className="text-[11px] text-slate-400">
                  event: {m.event_ticker}
                </div>
                <div className="text-[11px] text-slate-500">
                  status: {m.status ?? "unknown"} | YES:{" "}
                  {m.last_price_yes ?? "?"} | NO:{" "}
                  {m.last_price_no ?? "?"}
                </div>
              </div>
              <Button
                type="button"
                size="sm"
                className="mt-1"
                disabled={busyTicker === m.market_ticker}
                onClick={() => handleRefreshAndTrade(m.market_ticker)}
              >
                {busyTicker === m.market_ticker
                  ? "Trading…"
                  : "Refresh & Trade NO"}
              </Button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
