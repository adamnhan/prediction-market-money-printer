// ui/app/page.tsx
"use client";

import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { RefreshCcw } from "lucide-react";

import type { TradingState, Capital } from "../types/trading";
import type { HealthState } from "../types/trading";
type Trade = {
  id: number;
  position_id: number;
  event_ticker: string;
  market_ticker: string;
  side: string;
  qty: number;
  entry_price: number;
  entry_ts?: string;
  exit_price?: number;
  exit_ts?: string;
  realized_pnl: number;
  exit_reason?: string;
};
type EquityPoint = { id: number; exit_ts?: string; cumulative_pnl: number };
type MarketSummary = {
  market_ticker: string;
  trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_realized_pnl: number;
  avg_realized_pnl: number;
};
type Metrics = {
  total_trades: number;
  win_rate: number;
  total_realized_pnl: number;
  average_win: number;
  average_loss: number;
  average_hold_seconds: number;
  open_trades: number;
  closed_trades: number;
};

import { CapitalSummary } from "../components/CapitalSummary";
import { AddEventForm } from "../components/AddEventForm";
import { EventsTable } from "../components/EventsTable";
import { LogsPanel } from "../components/LogsPanel";
import { DebugPanel } from "../components/DebugPanel";
import { Button } from "@/components/ui/button";
import { EventMarketsViewer } from "../components/EventMarketsViewer";
import { MarketsPanel } from "../components/MarketsPanel";
import { PositionsPanel } from "../components/PositionsPanel";



export default function HomePage() {
  const [state, setState] = useState<TradingState | null>(null);
  const [health, setHealth] = useState<HealthState | null>(null);
  const [newTicker, setNewTicker] = useState("");
  const [showDebug, setShowDebug] = useState(false);
  const [activeTab, setActiveTab] = useState<
    "dashboard" | "positions" | "analytics"
  >("dashboard");
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [equity, setEquity] = useState<EquityPoint[]>([]);
  const [loadingMetrics, setLoadingMetrics] = useState(false);
  const [loadingTrades, setLoadingTrades] = useState(false);
  const [loadingEquity, setLoadingEquity] = useState(false);
  const [marketSummary, setMarketSummary] = useState<MarketSummary[]>([]);
  const [loadingMarketSummary, setLoadingMarketSummary] = useState(false);
  const [marketSortBy, setMarketSortBy] = useState<
    "total_realized_pnl" | "trades" | "win_rate"
  >("total_realized_pnl");
  const [marketFilter, setMarketFilter] = useState("");
  const [tradeFilter, setTradeFilter] = useState("");
  const [tradeSideFilter, setTradeSideFilter] = useState<"all" | "no" | "yes">(
    "all"
  );
  const [tradeReasonFilter, setTradeReasonFilter] = useState("");
  const [tradesLimit, setTradesLimit] = useState(100);
  const [tradesOffset, setTradesOffset] = useState(0);
  const [hasMoreTrades, setHasMoreTrades] = useState(true);

  async function loadState() {
    try {
      const res = await fetch("http://127.0.0.1:8000/state");
      const data = await res.json();
      setState(data);
    } catch (err) {
      console.error("Failed to load state:", err);
    }
  }

  async function loadHealth() {
    try {
        const res = await fetch("http://127.0.0.1:8000/health");
        const data = await res.json();
        setHealth(data);
    } catch (err) {
        console.error("Failed to load health:", err);
    }
  }

  useEffect(() => {
    loadState();
    loadHealth();
    const id = setInterval(() => {
      loadState();
      loadHealth();
    }, 5000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (activeTab === "analytics") {
      setTradesOffset(0);
      setTrades([]);
      setHasMoreTrades(true);
    }
  }, [activeTab]);

  // Reset pagination when trade filters change
  useEffect(() => {
    if (activeTab === "analytics") {
      setTradesOffset(0);
      setTrades([]);
      setHasMoreTrades(true);
    }
  }, [activeTab, tradeFilter, tradeSideFilter, tradeReasonFilter]);

  async function loadTradesPage(offsetOverride?: number) {
    const offset = offsetOverride ?? tradesOffset;
    setLoadingTrades(true);
    try {
        const res = await fetch(
          `http://127.0.0.1:8000/trades?limit=${tradesLimit}&offset=${offset}`
        );
        const data = await res.json();
        const rows: Trade[] = data.trades || [];
        if (offset === 0) {
          setTrades(rows);
        } else {
          setTrades((prev) => {
            const seen = new Set(prev.map((t) => t.id));
            const merged = [...prev];
            for (const r of rows) {
              if (!seen.has(r.id)) {
                merged.push(r);
              }
            }
            return merged;
          });
        }
        setHasMoreTrades(rows.length === tradesLimit);
      } catch (err) {
        console.error("Failed to load trades:", err);
      } finally {
        setLoadingTrades(false);
      }
  }

  useEffect(() => {
    async function loadMetrics() {
      setLoadingMetrics(true);
      try {
        const res = await fetch("http://127.0.0.1:8000/metrics");
        const data = await res.json();
        setMetrics(data);
      } catch (err) {
        console.error("Failed to load metrics:", err);
      } finally {
        setLoadingMetrics(false);
      }
    }

    async function loadEquity() {
      setLoadingEquity(true);
      try {
        const res = await fetch("http://127.0.0.1:8000/equity_curve");
        const data = await res.json();
        setEquity(data.points || []);
      } catch (err) {
        console.error("Failed to load equity curve:", err);
      } finally {
        setLoadingEquity(false);
      }
    }

    async function loadMarketSummary() {
      setLoadingMarketSummary(true);
      try {
        const res = await fetch(
          `http://127.0.0.1:8000/metrics/markets?limit=200&sort_by=${marketSortBy}`
        );
        const data = await res.json();
        setMarketSummary((data.markets as MarketSummary[] | undefined) ?? []);
      } catch (err) {
        console.error("Failed to load market summary:", err);
      } finally {
        setLoadingMarketSummary(false);
      }
    }

    if (activeTab === "analytics") {
      loadMetrics();
      loadEquity();
      loadMarketSummary();
    }
  }, [activeTab, marketSortBy]);

  useEffect(() => {
    if (activeTab === "analytics") {
      loadTradesPage();
    }
  }, [activeTab, tradesLimit, tradesOffset]);

  async function handleAddEvent(e: FormEvent) {
    e.preventDefault();
    if (!newTicker.trim()) return;

    const upper = newTicker.trim().toUpperCase();

    try {
      await fetch("http://127.0.0.1:8000/events", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: upper }),
      });

      setNewTicker("");
      await loadState();
    } catch (err) {
      console.error("Failed to add event:", err);
    }
  }

  async function handleRemoveEvent(ticker: string) {
    try {
      await fetch(
        `http://127.0.0.1:8000/events/${encodeURIComponent(ticker)}`,
        {
          method: "DELETE",
        }
      );
      await loadState();
    } catch (err) {
      console.error("Failed to remove event:", err);
    }
  }

  async function handleClosePosition(id: number) {
    try {
      await fetch(`http://127.0.0.1:8000/close_position/${id}`, {
        method: "POST",
      });
      await loadState();
    } catch (err) {
      console.error("Failed to close position:", err);
    }
  }

  async function handleRemoveMarket(marketTicker: string) {
    try {
      await fetch(
        `http://127.0.0.1:8000/markets/${encodeURIComponent(marketTicker)}`,
        {
          method: "DELETE",
        }
      );
      await loadState();
    } catch (err) {
      console.error("Failed to remove market:", err);
    }
  }


  const capital: Capital =
    state?.capital ?? ({ total: 0, used: 0, available: 0 } as Capital);
  const strategyCounters = state?.strategy_counters;
  const circuit = state?.circuit_breakers;
  const operatorFlags = state?.operator_flags;

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-8">
        {/* HEADER */}
        <header className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">
              Kalshi Paper Trading Dashboard
            </h1>
            <p className="text-sm text-slate-400">
              Track events, capital, and bot activity in real time.
            </p>

            {/* Navbar tabs */}
            <div className="mt-3 inline-flex rounded-full border border-slate-800 bg-slate-900/60 p-1 text-xs">
              <button
                className={`rounded-full px-3 py-1 ${
                  activeTab === "dashboard"
                    ? "bg-slate-100 text-slate-900"
                    : "text-slate-400 hover:text-slate-100"
                }`}
                onClick={() => setActiveTab("dashboard")}
              >
                Dashboard
              </button>
              <button
                className={`rounded-full px-3 py-1 ${
                  activeTab === "positions"
                    ? "bg-slate-100 text-slate-900"
                    : "text-slate-400 hover:text-slate-100"
                }`}
                onClick={() => setActiveTab("positions")}
              >
                Positions
              </button>
              <button
                className={`rounded-full px-3 py-1 ${
                  activeTab === "analytics"
                    ? "bg-slate-100 text-slate-900"
                    : "text-slate-400 hover:text-slate-100"
                }`}
                onClick={() => setActiveTab("analytics")}
              >
                Analytics
              </button>
            </div>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={loadState}
            className="gap-2 border-slate-700 bg-slate-900 text-slate-100 hover:border-slate-500 hover:bg-slate-800"
          >
            <RefreshCcw className="h-4 w-4" />
            Refresh now
          </Button>
        </header>


        {activeTab === "dashboard" && (
        <>
          {/* TOP: CAPITAL + ADD EVENT + EVENTS TABLE */}
          <section className="grid gap-6 md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
            <div className="flex flex-col gap-6">
              <CapitalSummary capital={capital} />
              <SafetyPanel
                circuit={circuit}
                operatorFlags={operatorFlags}
                health={health}
              />
              <AddEventForm
                newTicker={newTicker}
                setNewTicker={setNewTicker}
                onSubmit={handleAddEvent}
              />
            </div>
            <EventsTable
              events={state?.events ?? []}
              loading={!state}
              onRemove={handleRemoveEvent}
            />
          </section>

          {/* BOTTOM: LOGS + DEBUG */}
          <section className="grid gap-6 md:grid-cols-2">
            <LogsPanel logs={state?.logs ?? []} loading={!state} />
            <DebugPanel
              state={state}
              show={showDebug}
              onToggle={() => setShowDebug((prev) => !prev)}
            />
            {strategyCounters && (
              <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-3 text-xs text-slate-200">
                <div className="mb-1 font-semibold text-slate-100">Strategy Counters</div>
                <div>Auto entries: {strategyCounters.auto_entries}</div>
                <div>Auto exits: {strategyCounters.auto_exits}</div>
                <div>Risk skips: {strategyCounters.skipped_entries_due_to_risk}</div>
                {strategyCounters.last_risk_skip_reason && (
                  <div className="text-slate-400">
                    Last skip: {strategyCounters.last_risk_skip_reason}
                  </div>
                )}
              </div>
            )}
          </section>

          <EventMarketsViewer
            events={(state?.events ?? []).map((ev) => ({
              ticker: ev.ticker,
              title: ev.title,
            }))}
          />

          <MarketsPanel
            markets={state?.markets ?? {}}
            loading={!state}
            onRemove={handleRemoveMarket}
          />
        </>
      )}

        {activeTab === "positions" && (
          <section className="mt-4">
            {/* Simple positions view; we can refine later (active vs closed, etc.) */}
            <PositionsPanel
              positions={state?.positions ?? []}
              loading={!state}
              onClose={handleClosePosition}
            />
          </section>
        )}

        {activeTab === "analytics" && (
          <section className="mt-4">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <MetricCard
                title="Total Trades"
                value={
                  loadingMetrics ? "Loading..." : metrics?.total_trades ?? "—"
                }
              />
              <MetricCard
                title="Win Rate"
                value={
                  loadingMetrics
                    ? "Loading..."
                    : metrics
                    ? `${(metrics.win_rate * 100).toFixed(1)}%`
                    : "—"
                }
              />
              <MetricCard
                title="Realized PnL"
                value={
                  loadingMetrics
                    ? "Loading..."
                    : metrics
                    ? metrics.total_realized_pnl.toFixed(2)
                    : "—"
                }
              />
              <MetricCard
                title="Avg Win"
                value={
                  loadingMetrics
                    ? "Loading..."
                    : metrics
                    ? metrics.average_win.toFixed(2)
                    : "—"
                }
              />
              <MetricCard
                title="Avg Loss"
                value={
                  loadingMetrics
                    ? "Loading..."
                    : metrics
                    ? metrics.average_loss.toFixed(2)
                    : "—"
                }
              />
              <MetricCard
                title="Avg Hold (s)"
                value={
                  loadingMetrics
                    ? "Loading..."
                    : metrics
                    ? metrics.average_hold_seconds.toFixed(0)
                    : "—"
                }
              />
            </div>

            <div className="mt-6 grid gap-6 lg:grid-cols-[2fr_1fr]">
              <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
                <div className="mb-3 text-sm font-semibold text-slate-200">
                  Equity Curve (realized)
                </div>
                <EquityCurve points={equity} loading={loadingEquity} />
              </div>

              <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
                <div className="mb-3 text-sm font-semibold text-slate-200">
                  Recent Trades
                </div>
                <div className="mb-3 flex flex-wrap items-center gap-3 text-xs text-slate-300">
                  <div className="flex items-center gap-2">
                    <span>Limit:</span>
                    <select
                      value={tradesLimit}
                      onChange={(e) => {
                        setTradesLimit(Number(e.target.value) || 100);
                        setTradesOffset(0);
                        setHasMoreTrades(true);
                        setTrades([]);
                      }}
                      className="rounded-md border border-slate-700 bg-slate-800/80 px-2 py-1 text-xs text-slate-100 outline-none focus:border-slate-500"
                    >
                      {[50, 100, 200, 500].map((n) => (
                        <option key={n} value={n}>
                          {n}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="flex items-center gap-2">
                    <span>Filter:</span>
                    <input
                      value={tradeFilter}
                      onChange={(e) => setTradeFilter(e.target.value)}
                      placeholder="Market ticker contains…"
                      className="rounded-md border border-slate-700 bg-slate-800/80 px-2 py-1 text-xs text-slate-100 outline-none focus:border-slate-500"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <span>Side:</span>
                    <select
                      value={tradeSideFilter}
                      onChange={(e) =>
                        setTradeSideFilter(
                          (e.target.value as "all" | "no" | "yes") || "all"
                        )
                      }
                      className="rounded-md border border-slate-700 bg-slate-800/80 px-2 py-1 text-xs text-slate-100 outline-none focus:border-slate-500"
                    >
                      <option value="all">All</option>
                      <option value="no">NO</option>
                      <option value="yes">YES</option>
                    </select>
                  </div>
                  <div className="flex items-center gap-2">
                    <span>Reason:</span>
                    <input
                      value={tradeReasonFilter}
                      onChange={(e) => setTradeReasonFilter(e.target.value)}
                      placeholder="Exit reason contains…"
                      className="rounded-md border border-slate-700 bg-slate-800/80 px-2 py-1 text-xs text-slate-100 outline-none focus:border-slate-500"
                    />
                  </div>
                </div>
                <TradesTable
                  trades={trades}
                  loading={loadingTrades}
                  filter={tradeFilter}
                  sideFilter={tradeSideFilter}
                  reasonFilter={tradeReasonFilter}
                />
                <div className="mt-3 flex items-center gap-3 text-xs text-slate-300">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setTradesOffset((prev) => prev + tradesLimit);
                    }}
                    disabled={loadingTrades || !hasMoreTrades}
                    className="border-slate-700 bg-slate-900 text-slate-100 hover:border-slate-500 hover:bg-slate-800"
                  >
                    {hasMoreTrades ? "Load more" : "No more trades"}
                  </Button>
                  <span className="text-slate-500">
                    Showing {trades.length} loaded (limit {tradesLimit} per page)
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-6 rounded-lg border border-slate-800 bg-slate-900/60 p-4">
              <div className="mb-3 text-sm font-semibold text-slate-200">
                Per-Market Performance
              </div>
              <div className="mb-3 flex flex-wrap items-center gap-3 text-xs text-slate-300">
                <div className="flex items-center gap-2">
                  <span>Sort by:</span>
                  <div className="inline-flex overflow-hidden rounded-md border border-slate-700 bg-slate-800/80">
                    {[
                      { label: "PnL", value: "total_realized_pnl" as const },
                      { label: "Trades", value: "trades" as const },
                      { label: "Win %", value: "win_rate" as const },
                    ].map((opt) => (
                      <button
                        key={opt.value}
                        onClick={() => setMarketSortBy(opt.value)}
                        className={`px-3 py-1 ${
                          marketSortBy === opt.value
                            ? "bg-slate-100 text-slate-900"
                            : "text-slate-300 hover:bg-slate-700"
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span>Filter:</span>
                  <input
                    value={marketFilter}
                    onChange={(e) => setMarketFilter(e.target.value)}
                    placeholder="Ticker contains…"
                    className="rounded-md border border-slate-700 bg-slate-800/80 px-2 py-1 text-xs text-slate-100 outline-none focus:border-slate-500"
                  />
                </div>
              </div>
              <MarketSummaryTable
                markets={marketSummary}
                loading={loadingMarketSummary}
                filter={marketFilter}
              />
            </div>
          </section>
        )}
      </div>
    </main>
  );
}

type MetricCardProps = {
  title: string;
  value: string | number;
};

function MetricCard({ title, value }: MetricCardProps) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-400">
        {title}
      </div>
      <div className="mt-2 text-2xl font-semibold text-slate-100">
        {value}
      </div>
    </div>
  );
}

function formatTs(ts?: string) {
  if (!ts) return null;
  try {
    const d = new Date(ts);
    if (isNaN(d.getTime())) return ts;
    return d.toLocaleString();
  } catch {
    return ts;
  }
}

function MarketSummaryTable({
  markets,
  loading,
  filter,
}: {
  markets: MarketSummary[];
  loading: boolean;
  filter: string;
}) {
  if (loading) {
    return <div className="text-sm text-slate-400">Loading markets…</div>;
  }

  const filtered = (filter || "").trim()
    ? markets.filter((m) =>
        (m.market_ticker || "")
          .toLowerCase()
          .includes(filter.trim().toLowerCase())
      )
    : markets;

  if (!filtered.length) {
    return <div className="text-sm text-slate-400">No market stats yet.</div>;
  }

  // show top 100 by chosen sort to keep UI light
  const limited = filtered.slice(0, 100);

  return (
    <div className="overflow-x-auto text-sm">
      <table className="min-w-full border-collapse">
        <thead className="text-left text-xs uppercase text-slate-400">
          <tr>
            <th className="py-1 pr-3">Market</th>
            <th className="py-1 pr-3">Trades</th>
            <th className="py-1 pr-3">Wins</th>
            <th className="py-1 pr-3">Win %</th>
            <th className="py-1 pr-3">Realized PnL</th>
            <th className="py-1 pr-3">Avg PnL</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {limited.map((m) => (
            <tr key={m.market_ticker}>
              <td className="py-1 pr-3 text-slate-100">{m.market_ticker}</td>
              <td className="py-1 pr-3">{m.trades}</td>
              <td className="py-1 pr-3">{m.wins}</td>
              <td className="py-1 pr-3">
                {(m.win_rate * 100).toFixed(1)}%
              </td>
              <td
                className={`py-1 pr-3 ${
                  m.total_realized_pnl >= 0
                    ? "text-emerald-300"
                    : "text-rose-300"
                }`}
              >
                {m.total_realized_pnl.toFixed(2)}
              </td>
              <td
                className={`py-1 pr-3 ${
                  m.avg_realized_pnl >= 0 ? "text-emerald-300" : "text-rose-300"
                }`}
              >
                {m.avg_realized_pnl.toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {markets.length > limited.length && (
        <div className="mt-2 text-xs text-slate-500">
          Showing top {limited.length} of {markets.length} markets.
        </div>
      )}
    </div>
  );
}

function TradesTable({
  trades,
  loading,
  filter,
  sideFilter,
  reasonFilter,
}: {
  trades: Trade[];
  loading: boolean;
  filter: string;
  sideFilter: "all" | "no" | "yes";
  reasonFilter: string;
}) {
  if (loading) {
    return <div className="text-sm text-slate-400">Loading trades…</div>;
  }

  const filtered = trades.filter((t) => {
    const matchesMarket = (filter || "").trim()
      ? (t.market_ticker || "")
          .toLowerCase()
          .includes(filter.trim().toLowerCase())
      : true;

    const matchesSide =
      sideFilter === "all"
        ? true
        : ((t.side || "").toLowerCase() === sideFilter);

    const matchesReason = (reasonFilter || "").trim()
      ? (t.exit_reason || "")
          .toLowerCase()
          .includes(reasonFilter.trim().toLowerCase())
      : true;

    return matchesMarket && matchesSide && matchesReason;
  });

  if (!filtered.length) {
    return (
      <div className="text-sm text-slate-400">No trades recorded yet.</div>
    );
  }

  return (
    <div className="overflow-x-auto text-sm">
      <table className="min-w-full border-collapse">
        <thead className="text-left text-xs uppercase text-slate-400">
          <tr>
            <th className="py-1 pr-3">Market</th>
            <th className="py-1 pr-3">Side</th>
            <th className="py-1 pr-3">Qty</th>
            <th className="py-1 pr-3">Entry</th>
            <th className="py-1 pr-3">Exit</th>
            <th className="py-1 pr-3">PnL</th>
            <th className="py-1 pr-3">Reason</th>
            <th className="py-1 pr-3">Closed</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {filtered.map((t) => (
            <tr key={t.id}>
              <td className="py-1 pr-3 text-slate-100">
                {t.market_ticker ?? "—"}
              </td>
              <td className="py-1 pr-3">{(t.side ?? "—").toUpperCase()}</td>
              <td className="py-1 pr-3">{t.qty ?? "—"}</td>
              <td className="py-1 pr-3">{t.entry_price?.toFixed(2) ?? "—"}</td>
              <td className="py-1 pr-3">{t.exit_price?.toFixed(2) ?? "—"}</td>
              <td
                className={`py-1 pr-3 ${
                  (t.realized_pnl ?? 0) >= 0
                    ? "text-emerald-300"
                    : "text-rose-300"
                }`}
              >
                {(t.realized_pnl ?? 0).toFixed(2)}
              </td>
              <td className="py-1 pr-3 text-slate-300">
                {t.exit_reason ?? "—"}
              </td>
              <td className="py-1 pr-3 text-slate-400">
                {formatTs(t.exit_ts) ?? "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EquityCurve({
  points,
  loading,
}: {
  points: EquityPoint[];
  loading: boolean;
}) {
  if (loading) {
    return <div className="text-sm text-slate-400">Loading equity…</div>;
  }

  if (!points.length) {
    return (
      <div className="text-sm text-slate-400">No equity data yet.</div>
    );
  }

  // Simple sparkline using SVG
  const width = 600;
  const height = 180;
  const padding = 12;

  const values = points.map((p) => p.cumulative_pnl);
  const minVal = Math.min(...values, 0);
  const maxVal = Math.max(...values, 0.01); // avoid zero span
  const span = maxVal - minVal || 1;

  const coords = points.map((p, idx) => {
    const x =
      padding + (idx / Math.max(1, points.length - 1)) * (width - padding * 2);
    const y =
      height -
      padding -
      ((p.cumulative_pnl - minVal) / span) * (height - padding * 2);
    return { x, y };
  });

  const path = coords
    .map((c, idx) => `${idx === 0 ? "M" : "L"} ${c.x.toFixed(2)} ${c.y.toFixed(2)}`)
    .join(" ");

  // Baseline for zero
  const zeroY =
    height -
    padding -
    ((0 - minVal) / span) * (height - padding * 2);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full">
      <defs>
        <linearGradient id="eq-fill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="#22c55e" stopOpacity="0.25" />
          <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0" />
        </linearGradient>
      </defs>
      <rect
        x="0"
        y="0"
        width={width}
        height={height}
        fill="url(#eq-fill)"
        opacity="0.1"
      />
      <line
        x1={padding}
        x2={width - padding}
        y1={zeroY}
        y2={zeroY}
        stroke="#64748b"
        strokeDasharray="4 4"
        strokeWidth="1"
        opacity="0.6"
      />
      <path
        d={path}
        fill="none"
        stroke="#22c55e"
        strokeWidth="2"
        strokeLinecap="round"
      />
      {coords.map((c, idx) => (
        <g key={idx}>
          <circle cx={c.x} cy={c.y} r="3" fill="#22c55e" opacity="0.9">
            <title>
              {`${formatTs(points[idx].exit_ts) || "N/A"} | PnL ${points[
                idx
              ].cumulative_pnl.toFixed(2)}`}
            </title>
          </circle>
        </g>
      ))}
    </svg>
  );
}

function SafetyPanel({
  circuit,
  operatorFlags,
  health,
}: {
  circuit: TradingState["circuit_breakers"];
  operatorFlags: TradingState["operator_flags"];
  health: HealthState | null;
}) {
  const pauseEntries = operatorFlags?.pause_entries ?? false;
  const pauseAll = operatorFlags?.pause_all ?? false;

  const todayPnl = circuit?.today_realized_pnl ?? 0;
  const todayTrades = circuit?.today_trades ?? 0;
  const maxDd = circuit?.max_drawdown ?? 0;
  const cooldownUntil = circuit?.cooldown_until;
  const limits = circuit?.limits ?? {};
  const wsConnected = health?.ws_connected ?? false;
  const wsStale = health?.ws_stale ?? false;
  const wsLastMsg = health?.ws_last_message_ts;

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4 text-sm text-slate-200">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-xs uppercase tracking-wide text-slate-400">
          Safety & Circuit Breakers
        </div>
        <div className="flex gap-2 text-[11px] font-semibold">
          <span
            className={`rounded-full px-2 py-0.5 ${
              pauseEntries
                ? "bg-amber-500/20 text-amber-200"
                : "bg-emerald-500/15 text-emerald-200"
            }`}
          >
            Entries {pauseEntries ? "Paused" : "On"}
          </span>
          <span
            className={`rounded-full px-2 py-0.5 ${
              pauseAll
                ? "bg-rose-500/20 text-rose-200"
                : "bg-emerald-500/15 text-emerald-200"
            }`}
          >
            Trading {pauseAll ? "Paused" : "On"}
          </span>
          <span
            className={`rounded-full px-2 py-0.5 ${
              wsConnected && !wsStale
                ? "bg-emerald-500/15 text-emerald-200"
                : "bg-rose-500/20 text-rose-200"
            }`}
          >
            WS {wsConnected ? (wsStale ? "Stale" : "OK") : "Down"}
          </span>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-md border border-slate-800/80 bg-slate-900/60 p-3">
          <div className="text-xs uppercase text-slate-400">Today</div>
          <div className="mt-1 text-lg font-semibold">
            PnL:{" "}
            <span
              className={
                todayPnl >= 0 ? "text-emerald-300" : "text-rose-300"
              }
            >
              {todayPnl.toFixed(2)}
            </span>
          </div>
          <div className="text-xs text-slate-400">Trades: {todayTrades}</div>
        </div>
        <div className="rounded-md border border-slate-800/80 bg-slate-900/60 p-3">
          <div className="text-xs uppercase text-slate-400">Drawdown</div>
          <div className="mt-1 text-lg font-semibold text-slate-100">
            {maxDd.toFixed(2)}
          </div>
          {limits.max_drawdown != null && (
            <div className="text-xs text-slate-400">
              Limit: {limits.max_drawdown}
            </div>
          )}
        </div>
      </div>

      <div className="mt-3 grid gap-3 text-xs text-slate-300 sm:grid-cols-2">
        <div>
          Daily loss limit:{" "}
          {limits.daily_loss_limit != null
            ? `-${limits.daily_loss_limit}`
            : "not set"}
        </div>
        <div>
          Max trades/day:{" "}
          {limits.max_trades_per_day != null
            ? limits.max_trades_per_day
            : "not set"}
        </div>
        <div>
          Cooldown after stop:{" "}
          {limits.cooldown_minutes_after_stop != null
            ? `${limits.cooldown_minutes_after_stop} min`
            : "off"}
        </div>
        <div>
          Cooldown until: {cooldownUntil ? formatTs(cooldownUntil) : "none"}
        </div>
        <div>WS last message: {wsLastMsg ? formatTs(wsLastMsg) : "unknown"}</div>
        <div>
          WS stale: {wsStale ? "Yes" : "No"} | Subs:{" "}
          {health?.ws_subscriptions ?? 0}
        </div>
      </div>
    </div>
  );
}
