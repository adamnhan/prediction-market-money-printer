"use client";

import { useState } from "react";

export default function EventsPage() {
  const [items, setItems] = useState<{ticker:string; title:string}[]>([]);
  const [loading, setLoading] = useState(false);

  async function fetchEvents() {
    setLoading(true);
    const res = await fetch("/api/kalshi/events", { cache: "no-store" });
    const data = await res.json();
    setItems(data.markets || []);
    setLoading(false);
  }

  return (
    <main className="mx-auto max-w-5xl p-6 space-y-4">
      <h1 className="text-xl font-semibold">Events Browser</h1>
      <button
        onClick={fetchEvents}
        disabled={loading}
        className="rounded-lg border px-3 py-2 text-sm bg-white"
      >
        {loading ? "Fetching…" : "Fetch events"}
      </button>

      <div className="rounded-xl border bg-white">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-100 text-gray-600">
            <tr><th className="px-3 py-2 text-left">Ticker</th><th className="px-3 py-2 text-left">Title</th></tr>
          </thead>
          <tbody>
            {items.length ? items.map((m) => (
              <tr key={m.ticker} className="border-t">
                <td className="px-3 py-2 font-mono">{m.ticker}</td>
                <td className="px-3 py-2">{m.title}</td>
              </tr>
            )) : (
              <tr><td className="px-3 py-8 text-center text-gray-500" colSpan={2}>Click “Fetch events”</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </main>
  );
}
