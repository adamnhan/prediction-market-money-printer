"use client";

import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Bell, LayoutDashboard } from "lucide-react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ReTooltip, AreaChart, Area, BarChart, Bar } from "recharts";
import { useQuery } from "@tanstack/react-query";
import { fetchKalshiCandles } from "@/lib/kalshi";
/**
 * Minimal, clean dashboard with mock data + simple visuals.
 * - Overview: KPIs + tiny chart
 * - Alerts sidebar: list of mock alerts
 * - Market Detail: price line + tiny trades table
 */
export default function DashboardMocked() {
  const [series, setSeries] = useState(makeSeedSeries());
  const [alerts, setAlerts] = useState<any[]>([]);
  const [trades, setTrades] = useState<any[]>([]);

  useEffect(() => {
    setAlerts(makeSeedAlerts());
    setTrades(makeSeedTrades());
  }, []);
  // lightweight ticker to keep things moving
  useEffect(() => {
    const id = setInterval(() => {
      setSeries((prev) => tickSeries(prev));
      // occasionally add a mock trade & maybe an alert
      if (Math.random() > 0.7) setTrades((prev) => [makeTrade(), ...prev].slice(0, 20));
      if (Math.random() > 0.92) setAlerts((prev) => [makeAlert(), ...prev].slice(0, 10));
    }, 2000);
    return () => clearInterval(id);
  }, []);

  const kpis = useMemo(() => computeKPIs(series, trades, alerts), [series, trades, alerts]);
  const { data: candles = [] } = useQuery({
    queryKey: ["candles", "FED_RATE_DEC"],
    queryFn: () => fetchKalshiCandles("FED_RATE_DEC"),
    refetchInterval: 5000,
  })

  console.log("candles length", candles.length, candles[0]);
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="sticky top-0 z-20 bg-white border-b">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-3 flex items-center gap-3">
          <LayoutDashboard className="h-5 w-5" />
          <h1 className="text-lg font-semibold">InsiderWatch · Kalshi</h1>
          <div className="ml-auto">
            <Button variant="outline" size="sm" className="gap-1"><Bell className="h-4 w-4"/> Alerts</Button>
          </div>
        </div>
      </header>

      {/* Body */}
      <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6 grid grid-cols-12 gap-4">
        {/* Alerts Sidebar */}
        <aside className="col-span-12 lg:col-span-3">
          <Card className="shadow-sm">
            <CardContent className="p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="font-medium">Alerts</h2>
                <Badge variant="secondary">{alerts.length} open</Badge>
              </div>
              <div className="space-y-2">
                {alerts.map((a) => (
                  <div key={a.id} className="rounded-xl border bg-white p-3">
                    <div className="text-xs text-gray-500">{new Date(a.ts).toLocaleTimeString()}</div>
                    <div className="text-sm font-medium">{a.market}</div>
                    <div className="text-sm text-gray-600">{a.reason}</div>
                    <div className="mt-1 text-xs"><span className="font-semibold">Score:</span> {a.score.toFixed(2)}</div>
                  </div>
                ))}
                {alerts.length === 0 && (
                  <div className="rounded-xl border border-dashed bg-white p-6 text-center text-sm text-gray-500">No alerts yet</div>
                )}
              </div>
            </CardContent>
          </Card>
        </aside>

        {/* Main Column */}
        <section className="col-span-12 lg:col-span-9 space-y-4">
          {/* Overview */}
          <Card className="shadow-sm">
            <CardContent className="p-4 space-y-4">
              <h3 className="font-medium">Overview</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <KPI label="Implied %" value={kpis.implied.toFixed(1)} suffix="%" />
                <KPI label="24h Vol" value={kpis.vol.toFixed(0)} />
                <KPI label="Alerts (24h)" value={kpis.alerts24h} />
                <KPI label="Top Move" value={(kpis.topMove*1).toFixed(1)} suffix="pts" />
              </div>
              <div className="h-36">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={candles} margin={{ left: 8, right: 8, top: 8, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="t" tickFormatter={fmtTimeMini} minTickGap={30} />
                    <YAxis domain={[0, 100]} />
                    <ReTooltip />
                    <Area type="monotone" dataKey="price" name="Implied %" fillOpacity={0.15} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Market Detail */}
          <Card className="shadow-sm">
            <CardContent className="p-4 space-y-3">
              <h3 className="font-medium">Market Detail</h3>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={candles} margin={{ left: 8, right: 8, top: 8, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="t" tickFormatter={fmtTimeMini} minTickGap={30} />
                    <YAxis domain={[0, 100]} />
                    <ReTooltip />
                    <Line type="monotone" dataKey="price" name="Implied %" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="rounded-2xl border bg-white">
                <table className="min-w-full text-sm">
                  <thead className="bg-gray-100 text-gray-600">
                    <tr>
                      <Th>Time</Th>
                      <Th>Side</Th>
                      <Th>Price</Th>
                      <Th>Size</Th>
                      <Th>Notional</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {trades.map((t) => (
                      <tr key={t.id} className="border-b last:border-0">
                        <Td>{new Date(t.ts).toLocaleTimeString()}</Td>
                        <Td>{t.side}</Td>
                        <Td>{t.price.toFixed(2)}%</Td>
                        <Td>{t.size.toFixed(0)}</Td>
                        <Td>${(t.size * (t.price/100)).toFixed(2)}</Td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </section>
      </main>
    </div>
  );
}

/* ——— subcomponents ——— */
function KPI({ label, value, suffix }: { label: string; value: string | number; suffix?: string }) {
  return (
    <div className="rounded-2xl border bg-white p-3">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-xl font-semibold">{value}{suffix ? <span className="text-gray-400 text-sm ml-1">{suffix}</span> : null}</div>
    </div>
  );
}
const Th = ({ children }: { children: React.ReactNode }) => (
  <th className="px-3 py-2 text-left text-xs font-medium uppercase tracking-wide">{children}</th>
);
const Td = ({ children }: { children: React.ReactNode }) => (
  <td className="px-3 py-2">{children}</td>
);

/* ——— mock data helpers ——— */
function makeSeedSeries() {
  const now = Date.now();
  const out: { t: string; price: number }[] = [];
  let price = 47 + Math.random() * 6;
  for (let i = 60; i >= 0; i--) {
    price += (Math.random() - 0.5) * 1.2;
    price = Math.max(0, Math.min(100, price));
    out.push({ t: new Date(now - i * 60 * 1000).toISOString(), price });
  }
  return out;
}
function tickSeries(arr: { t: string; price: number }[]) {
  const last = arr[arr.length - 1];
  const t = new Date(new Date(last.t).getTime() + 60 * 1000).toISOString();
  let price = last.price + (Math.random() - 0.5) * 1.2 + (Math.random() > 0.96 ? (Math.random() - 0.5) * 8 : 0);
  price = Math.max(0, Math.min(100, price));
  return [...arr.slice(1), { t, price }];
}

let ALERT_ID = 1;
function makeSeedAlerts() {
  return [
    { id: ALERT_ID++, ts: Date.now() - 3 * 60 * 1000, market: "FED_RATE_DEC", reason: "Δp +10 pts in 7m", score: 2.7 },
    { id: ALERT_ID++, ts: Date.now() - 20 * 60 * 1000, market: "CPI_ABOVE_4", reason: "Clustered buys (5 > 3σ)", score: 3.2 },
  ];
}
function makeAlert() {
  const reasons = ["Large notional > 4σ", "Thin ask wall broken", "Price gap vs mean", "Burst of buys in 3m"];
  const mkts = ["FED_RATE_DEC", "CPI_ABOVE_4", "EARN_XYZ"];
  return { id: ALERT_ID++, ts: Date.now(), market: pick(mkts), reason: pick(reasons), score: 2 + Math.random() * 3 };
}

let TRADE_ID = 1;
function makeSeedTrades() { return [makeTrade(), makeTrade(), makeTrade(), makeTrade(), makeTrade(), makeTrade()]; }
function makeTrade() {
  const side = Math.random() > 0.5 ? "BUY" : "SELL";
  const price = 45 + Math.random() * 10;
  const size = 50 + Math.random() * 300;
  return { id: TRADE_ID++, ts: Date.now(), side, price, size };
}

function computeKPIs(series: { price: number }[], trades: { price: number; size: number }[], alerts: { ts: number }[]) {
  const implied = series[series.length - 1]?.price ?? 0;
  const vol = trades.reduce((a, t) => a + t.size, 0);
  const alerts24h = alerts.filter((a) => Date.now() - a.ts < 24 * 3600 * 1000).length;
  const topMove = Math.max(0, Math.abs((series[series.length - 1]?.price ?? 0) - (series[0]?.price ?? 0)));
  return { implied, vol, alerts24h, topMove };
}

function fmtTimeMini(s: string) {
  const d = new Date(s);
  return d.getHours() + ":" + String(d.getMinutes()).padStart(2, "0");
}

function pick<T>(arr: T[]): T { return arr[Math.floor(Math.random() * arr.length)]; }
