// ui/app/page.tsx
"use client";

import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { RefreshCcw } from "lucide-react";

import type { TradingState, Capital } from "../types/trading";

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
  const [newTicker, setNewTicker] = useState("");
  const [showDebug, setShowDebug] = useState(false);
  const [activeTab, setActiveTab] = useState<"dashboard" | "positions">(
    "dashboard"
  );

  async function loadState() {
    try {
      const res = await fetch("http://127.0.0.1:8000/state");
      const data = await res.json();
      setState(data);
    } catch (err) {
      console.error("Failed to load state:", err);
    }
  }

  useEffect(() => {
    loadState();
    const id = setInterval(() => {
      loadState();
    }, 5000);
    return () => clearInterval(id);
  }, []);

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

  
  const capital: Capital =
    state?.capital ?? ({ total: 0, used: 0, available: 0 } as Capital);

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
      </div>
    </main>
  );
}
