// ui/components/PositionsPanel.tsx
"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";

type PositionsPanelProps = {
  positions: any[];
  loading: boolean;
  onClose?: (id: number) => void;
};


export function PositionsPanel({ positions, loading, onClose }: PositionsPanelProps) {
    const [includeClosed, setIncludeClosed] = useState(false);
    
    const visiblePositions = includeClosed
    ? positions
    : positions.filter((p) => p.status === "open");
    return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
        <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-100">
            Positions
            </h2>
            <div className="flex items-center gap-3">
            <label className="flex items-center gap-1 text-xs text-slate-400">
                <input
                type="checkbox"
                className="h-3 w-3 cursor-pointer accent-slate-100"
                checked={includeClosed}
                onChange={(e) => setIncludeClosed(e.target.checked)}
                />
                <span>Include closed</span>
            </label>
            {loading && (
                <span className="text-xs text-slate-500">Loading…</span>
            )}
            </div>
      </div>


        {(!visiblePositions || visiblePositions.length === 0) && !loading ? (
        <p className="text-sm text-slate-500">
          No positions yet. Attach a market and run “Refresh &amp; Trade NO”.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[400px] text-left text-sm">
            <thead className="border-b border-slate-800 text-xs uppercase text-slate-400">
              <tr>
                <th className="py-2 pr-2">ID</th>
                <th className="py-2 pr-2">Event</th>
                <th className="py-2 pr-2">Market</th>
                <th className="py-2 pr-2">Side</th>
                <th className="py-2 pr-2 text-right">Qty</th>
                <th className="py-2 pr-2 text-right">Entry</th>
                <th className="py-2 pr-2 text-right">Current</th>
                <th className="py-2 pr-2 text-right">Unrealized PnL</th>
                <th className="py-2 pr-2">Status</th>
                <th className="py-2 pr-2 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {visiblePositions.map((p) => (
                <tr
                  key={p.id}
                  className="border-b border-slate-900/60 last:border-0"
                >
                  <td className="py-1.5 pr-2 text-xs text-slate-300">
                    {p.id}
                  </td>
                  <td className="py-1.5 pr-2 text-xs text-slate-300">
                    {p.event_ticker || "—"}
                  </td>
                  <td className="py-1.5 pr-2 text-xs text-slate-300">
                    {p.market_ticker}
                  </td>
                  <td className="py-1.5 pr-2 text-xs text-slate-300">
                    {p.side}
                  </td>
                  <td className="py-1.5 pr-2 text-right text-xs text-slate-300">
                    {p.qty}
                  </td>
                  <td className="py-1.5 pr-2 text-right text-xs text-slate-300">
                    {p.entry_price?.toFixed?.(3) ?? p.entry_price}
                  </td>
                  <td className="py-1.5 pr-2 text-right text-xs text-slate-300">
                    {p.current_price?.toFixed?.(3) ?? p.current_price}
                  </td>
                  <td className="py-1.5 pr-2 text-right text-xs">
                    <span
                      className={
                        (p.unrealized_pnl ?? 0) > 0
                          ? "text-emerald-400"
                          : (p.unrealized_pnl ?? 0) < 0
                          ? "text-rose-400"
                          : "text-slate-300"
                      }
                    >
                      {p.unrealized_pnl?.toFixed?.(3) ?? p.unrealized_pnl}
                    </span>
                  </td>
                  <td className="py-1.5 pr-2 text-xs text-slate-400">
                    {p.status}
                  </td>
                  <td className="py-1.5 pr-2 text-right text-xs">
                    {p.status === "open" && (
                    <Button
                        size="sm"
                        variant="outline"
                        className="border-slate-700 bg-slate-900 text-slate-100 hover:border-slate-500 hover:bg-slate-800"
                        onClick={() => onClose?.(p.id)}
                    >
                        Close
                    </Button>
                    )}
                </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
