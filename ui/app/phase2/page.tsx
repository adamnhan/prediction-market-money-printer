"use client";

import { useEffect, useMemo, useState } from "react";
import { RefreshCcw, AlertTriangle, TrendingUp, TrendingDown } from "lucide-react";

type SummaryRow = {
  ts_utc: string;
  fills: number;
  actions: number;
  fill_rate: number;
  net_cash_change: number;
  inv_yes: number;
  inv_no: number;
};

type DashboardMetrics = {
  ts_utc?: string | null;
  cash_cents?: number | null;
  equity_cents?: number | null;
  inv_yes?: number | null;
  inv_no?: number | null;
  tier_a?: number | null;
  tier_b?: number | null;
  tier_c?: number | null;
  day_pnl_cents?: number | null;
  worst_window_cents?: number | null;
  soft_throttle?: number | null;
};

type SummarySeriesPoint = {
  ts_utc: string;
  net_cash_change: number;
};

type MetricsResponse = {
  dashboard: DashboardMetrics | null;
  summary: {
    last: SummaryRow | null;
    series: SummarySeriesPoint[];
    recent: SummaryRow[];
  };
  paths: {
    summary: string | null;
    log: string | null;
  };
  generated_at: string;
};

type MarketRow = {
  market_ticker: string;
  tier: string;
  incentive_ev_score: number;
  spread_quality_score: number;
  adverse_risk: number;
  inventory_risk: number;
  total_ev_score: number;
  incentive_discount: number;
  fills_30m: number;
  ts_utc?: string | null;
};

type MarketsResponse = {
  markets: MarketRow[];
  total: number;
  path: string | null;
};

const currency = (cents?: number | null) => {
  if (cents === null || cents === undefined) return "—";
  const value = cents / 100;
  return value.toLocaleString(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  });
};

const number = (value?: number | null, digits = 0) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
};

const pct = (value?: number | null, digits = 1) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
};

function Sparkline({ series }: { series: SummarySeriesPoint[] }) {
  const points = useMemo(() => {
    if (!series.length) return "";
    const values = series.map((p) => p.net_cash_change);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    return values
      .map((v, i) => {
        const x = (i / Math.max(1, values.length - 1)) * 300;
        const y = 70 - ((v - min) / range) * 70;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }, [series]);

  const latest = series.at(-1)?.net_cash_change ?? 0;
  const accent =
    latest >= 0 ? "stroke-emerald-400" : "stroke-rose-400";

  return (
    <svg viewBox="0 0 300 70" className="h-24 w-full">
      <polyline
        fill="none"
        strokeWidth="2.5"
        className={`opacity-80 ${accent}`}
        points={points}
      />
    </svg>
  );
}

export default function Phase2Dashboard() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [markets, setMarkets] = useState<MarketsResponse | null>(null);
  const [loadingMetrics, setLoadingMetrics] = useState(false);
  const [loadingMarkets, setLoadingMarkets] = useState(false);
  const [marketSortBy, setMarketSortBy] = useState<
    "total_ev_score" | "fills_30m" | "adverse_risk"
  >("total_ev_score");
  const [marketFilter, setMarketFilter] = useState("");

  async function loadMetrics() {
    setLoadingMetrics(true);
    try {
      const res = await fetch("/api/phase2/metrics", { cache: "no-store" });
      const data = (await res.json()) as MetricsResponse;
      setMetrics(data);
    } catch (err) {
      console.error("Failed to load phase2 metrics:", err);
    } finally {
      setLoadingMetrics(false);
    }
  }

  async function loadMarkets() {
    setLoadingMarkets(true);
    try {
      const res = await fetch(
        `/api/phase2/markets?limit=200&sort_by=${marketSortBy}`,
        { cache: "no-store" }
      );
      const data = (await res.json()) as MarketsResponse;
      setMarkets(data);
    } catch (err) {
      console.error("Failed to load phase2 markets:", err);
    } finally {
      setLoadingMarkets(false);
    }
  }

  useEffect(() => {
    loadMetrics();
    loadMarkets();
    const id = setInterval(() => {
      loadMetrics();
      loadMarkets();
    }, 6000);
    return () => clearInterval(id);
  }, [marketSortBy]);

  const filteredMarkets = useMemo(() => {
    const rows = markets?.markets ?? [];
    const filter = marketFilter.trim().toLowerCase();
    if (!filter) return rows;
    return rows.filter((row) =>
      row.market_ticker.toLowerCase().includes(filter)
    );
  }, [markets, marketFilter]);

  const lastSummary = metrics?.summary.last;
  const dashboard = metrics?.dashboard;
  const series = metrics?.summary.series ?? [];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="pointer-events-none fixed inset-0 opacity-80">
        <div className="absolute left-0 top-0 h-72 w-72 rounded-full bg-emerald-500/20 blur-[120px]" />
        <div className="absolute right-0 top-32 h-72 w-72 rounded-full bg-sky-500/20 blur-[120px]" />
        <div className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-amber-400/10 blur-[140px]" />
      </div>

      <div className="relative mx-auto max-w-6xl px-6 pb-16 pt-12">
        <header className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-emerald-300/70">
              Phase 2 Incentive Bot
            </p>
            <h1 className="mt-3 text-3xl font-semibold tracking-tight text-slate-50 sm:text-4xl">
              Market Making Pulse
            </h1>
            <p className="mt-2 text-sm text-slate-400">
              Live snapshot from CSV + log stream.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                loadMetrics();
                loadMarkets();
              }}
              className="inline-flex items-center gap-2 rounded-full border border-slate-700/60 bg-slate-900/60 px-4 py-2 text-sm text-slate-200 transition hover:border-emerald-400/40 hover:text-white"
            >
              <RefreshCcw className="h-4 w-4" />
              Refresh
            </button>
            <div className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-2 text-xs text-emerald-200">
              {metrics?.generated_at
                ? `Updated ${new Date(metrics.generated_at).toLocaleTimeString()}`
                : "Waiting for data"}
            </div>
          </div>
        </header>

        <section className="mt-10 grid gap-6 md:grid-cols-[1.3fr_1fr]">
          <div className="rounded-3xl border border-slate-800/80 bg-slate-900/60 p-6 shadow-2xl shadow-emerald-500/10">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                  Equity drift
                </p>
                <p className="mt-2 text-2xl font-semibold text-slate-100">
                  {currency(dashboard?.equity_cents ?? null)}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                  Day PnL
                </p>
                <p
                  className={`mt-2 text-xl font-semibold ${
                    (dashboard?.day_pnl_cents ?? 0) >= 0
                      ? "text-emerald-300"
                      : "text-rose-300"
                  }`}
                >
                  {currency(dashboard?.day_pnl_cents ?? null)}
                </p>
              </div>
            </div>
            <div className="mt-6 rounded-2xl bg-slate-950/70 px-4 py-4">
              {series.length ? (
                <Sparkline series={series} />
              ) : (
                <div className="flex h-24 items-center justify-center text-sm text-slate-500">
                  {loadingMetrics ? "Loading equity series..." : "No summary data yet."}
                </div>
              )}
            </div>
            <div className="mt-5 grid grid-cols-2 gap-3 text-xs text-slate-400 sm:grid-cols-4">
              <div>
                <p className="uppercase tracking-[0.2em]">Cash</p>
                <p className="mt-1 text-sm text-slate-100">
                  {currency(dashboard?.cash_cents ?? null)}
                </p>
              </div>
              <div>
                <p className="uppercase tracking-[0.2em]">Inv YES</p>
                <p className="mt-1 text-sm text-slate-100">
                  {number(dashboard?.inv_yes ?? null)}
                </p>
              </div>
              <div>
                <p className="uppercase tracking-[0.2em]">Inv NO</p>
                <p className="mt-1 text-sm text-slate-100">
                  {number(dashboard?.inv_no ?? null)}
                </p>
              </div>
              <div>
                <p className="uppercase tracking-[0.2em]">Worst Window</p>
                <p className="mt-1 text-sm text-slate-100">
                  {currency(dashboard?.worst_window_cents ?? null)}
                </p>
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-slate-800/80 bg-slate-900/60 p-6">
            <div className="flex items-center justify-between">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                Activity + risk
              </p>
              <div className="flex items-center gap-2 text-xs text-slate-400">
                {dashboard?.soft_throttle ? (
                  <span className="inline-flex items-center gap-2 rounded-full bg-amber-500/15 px-3 py-1 text-amber-200">
                    <AlertTriangle className="h-3 w-3" />
                    Soft throttle
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-2 rounded-full bg-emerald-500/15 px-3 py-1 text-emerald-200">
                    Normal
                  </span>
                )}
              </div>
            </div>
            <div className="mt-6 grid gap-4">
              <div className="rounded-2xl border border-slate-800/60 bg-slate-950/60 px-4 py-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
                      Fill Rate
                    </p>
                    <p className="mt-2 text-2xl font-semibold text-slate-100">
                      {pct(lastSummary?.fill_rate ?? null)}
                    </p>
                  </div>
                  <div className="text-right text-xs text-slate-400">
                    <p>Fills: {number(lastSummary?.fills ?? null)}</p>
                    <p>Actions: {number(lastSummary?.actions ?? null)}</p>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-3 text-center text-xs text-slate-400">
                <div className="rounded-2xl border border-slate-800/60 bg-slate-950/60 px-3 py-4">
                  <p className="uppercase tracking-[0.2em]">Tier A</p>
                  <p className="mt-2 text-base text-slate-100">
                    {number(dashboard?.tier_a ?? null)}
                  </p>
                </div>
                <div className="rounded-2xl border border-slate-800/60 bg-slate-950/60 px-3 py-4">
                  <p className="uppercase tracking-[0.2em]">Tier B</p>
                  <p className="mt-2 text-base text-slate-100">
                    {number(dashboard?.tier_b ?? null)}
                  </p>
                </div>
                <div className="rounded-2xl border border-slate-800/60 bg-slate-950/60 px-3 py-4">
                  <p className="uppercase tracking-[0.2em]">Tier C</p>
                  <p className="mt-2 text-base text-slate-100">
                    {number(dashboard?.tier_c ?? null)}
                  </p>
                </div>
              </div>
              <div className="rounded-2xl border border-slate-800/60 bg-slate-950/60 px-4 py-4 text-xs text-slate-400">
                <p className="uppercase tracking-[0.2em] text-slate-500">Data sources</p>
                <div className="mt-2 grid gap-1">
                  <p>
                    Summary:{" "}
                    <span className="text-slate-200">
                      {metrics?.paths.summary ?? "Not found"}
                    </span>
                  </p>
                  <p>
                    Log:{" "}
                    <span className="text-slate-200">
                      {metrics?.paths.log ?? "Not found"}
                    </span>
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="mt-10 rounded-3xl border border-slate-800/80 bg-slate-900/60 p-6">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                Top markets (Phase 4 EV)
              </p>
              <p className="mt-2 text-lg font-semibold text-slate-50">
                Incentive surface
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
              <input
                value={marketFilter}
                onChange={(event) => setMarketFilter(event.target.value)}
                placeholder="Filter ticker…"
                className="rounded-full border border-slate-700/60 bg-slate-950/60 px-4 py-2 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none"
              />
              <div className="flex items-center gap-2 rounded-full border border-slate-700/60 bg-slate-950/60 px-2 py-2">
                {[
                  { label: "EV", value: "total_ev_score" },
                  { label: "Fills", value: "fills_30m" },
                  { label: "Risk", value: "adverse_risk" },
                ].map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() =>
                      setMarketSortBy(
                        opt.value as "total_ev_score" | "fills_30m" | "adverse_risk"
                      )
                    }
                    className={`rounded-full px-3 py-1 text-xs transition ${
                      marketSortBy === opt.value
                        ? "bg-emerald-400/20 text-emerald-200"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="mt-6 overflow-hidden rounded-2xl border border-slate-800/60">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-950/70 text-[10px] uppercase tracking-[0.2em] text-slate-500">
                <tr>
                  <th className="px-4 py-3">Market</th>
                  <th className="px-4 py-3">Tier</th>
                  <th className="px-4 py-3">Total EV</th>
                  <th className="px-4 py-3">Spread</th>
                  <th className="px-4 py-3">Adverse</th>
                  <th className="px-4 py-3">Inventory</th>
                  <th className="px-4 py-3">Fills 30m</th>
                </tr>
              </thead>
              <tbody>
                {loadingMarkets ? (
                  <tr>
                    <td colSpan={7} className="px-4 py-6 text-center text-slate-500">
                      Loading markets…
                    </td>
                  </tr>
                ) : filteredMarkets.length ? (
                  filteredMarkets.slice(0, 50).map((row) => (
                    <tr
                      key={row.market_ticker}
                      className="border-t border-slate-800/60 hover:bg-slate-950/70"
                    >
                      <td className="px-4 py-3 font-medium text-slate-100">
                        {row.market_ticker}
                      </td>
                      <td className="px-4 py-3 text-slate-300">
                        {row.tier || "—"}
                      </td>
                      <td className="px-4 py-3 text-slate-200">
                        {number(row.total_ev_score, 2)}
                      </td>
                      <td className="px-4 py-3 text-slate-200">
                        {number(row.spread_quality_score, 2)}
                      </td>
                      <td className="px-4 py-3 text-slate-200">
                        {number(row.adverse_risk, 2)}
                      </td>
                      <td className="px-4 py-3 text-slate-200">
                        {number(row.inventory_risk, 2)}
                      </td>
                      <td className="px-4 py-3 text-slate-200">
                        {number(row.fills_30m, 0)}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={7} className="px-4 py-6 text-center text-slate-500">
                      No market rows yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="mt-4 text-xs text-slate-500">
            {markets?.path ? `Source: ${markets.path}` : "Waiting for phase4_ev.csv"}
          </div>
        </section>

        <section className="mt-10 grid gap-6 md:grid-cols-3">
          <div className="rounded-3xl border border-slate-800/80 bg-slate-900/60 p-6">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
              Snapshot
            </p>
            <div className="mt-4 flex items-center justify-between text-sm text-slate-200">
              <span>Last summary</span>
              <span className="text-slate-400">
                {lastSummary?.ts_utc ?? "—"}
              </span>
            </div>
            <div className="mt-4 grid gap-3 text-xs text-slate-400">
              <div className="flex items-center justify-between rounded-2xl bg-slate-950/60 px-4 py-3">
                <span>Net cash change</span>
                <span className="text-slate-100">
                  {currency(lastSummary?.net_cash_change ?? null)}
                </span>
              </div>
              <div className="flex items-center justify-between rounded-2xl bg-slate-950/60 px-4 py-3">
                <span>Inv YES / NO</span>
                <span className="text-slate-100">
                  {number(lastSummary?.inv_yes ?? null)} / {number(lastSummary?.inv_no ?? null)}
                </span>
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-slate-800/80 bg-slate-900/60 p-6">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
              Momentum
            </p>
            <div className="mt-4 flex items-center gap-3 text-sm text-slate-200">
              {(dashboard?.day_pnl_cents ?? 0) >= 0 ? (
                <TrendingUp className="h-4 w-4 text-emerald-300" />
              ) : (
                <TrendingDown className="h-4 w-4 text-rose-300" />
              )}
              <span>
                {currency(dashboard?.day_pnl_cents ?? null)} today
              </span>
            </div>
            <div className="mt-6 text-xs text-slate-400">
              <p className="uppercase tracking-[0.2em] text-slate-500">
                Health
              </p>
              <p className="mt-2">
                {dashboard?.soft_throttle
                  ? "Soft throttle active."
                  : "No throttle flags."}
              </p>
            </div>
          </div>

          <div className="rounded-3xl border border-slate-800/80 bg-slate-900/60 p-6">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
              Notes
            </p>
            <ul className="mt-4 space-y-3 text-xs text-slate-400">
              <li className="rounded-2xl bg-slate-950/60 px-4 py-3">
                CSV parsing uses the latest 300 summary rows.
              </li>
              <li className="rounded-2xl bg-slate-950/60 px-4 py-3">
                Log tail scans the newest 1MB for dashboard lines.
              </li>
            </ul>
          </div>
        </section>
      </div>
    </div>
  );
}
