"use client";

import { useState } from "react";

export default function EventsPage() {
  const [titles, setTitles] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [ticker, setTicker] = useState("");
  const [error, setError] = useState<string | null>(null);

  // Fetch data for a specific market ticker
  async function handleSearch() {
    if (!ticker.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`/api/kalshi/markets?ticker=${encodeURIComponent(ticker)}`, {
        cache: "no-store",
      });
      const data = await res.json();

      if (!data.ok) {
        setError(data.error || "Market not found");
        console.error("Error:", data.error);
      } else {
        console.log("Fetched market data:", data.data); // 👈 full JSON in console
      }
    } catch (err) {
      console.error("Fetch error:", err);
      setError("Something went wrong fetching market data");
    }

    setLoading(false);
  }

  return (
    <main className="mx-auto max-w-5xl p-6 space-y-6">
      <h1 className="text-xl font-semibold">Kalshi Markets</h1>

      {/* 🔍 Search bar */}
      <div className="flex gap-2">
        <input
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          placeholder="Enter market ticker (e.g. KXINFL-MAR24)"
          className="flex-1 rounded-lg border px-3 py-2 text-sm"
        />
        <button
          onClick={handleSearch}
          disabled={loading}
          className="rounded-lg border px-4 py-2 text-sm bg-white hover:bg-gray-50"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      {error && <p className="text-red-500 text-sm">{error}</p>}

      <p className="text-gray-600 text-sm">
        Try tickers like <code>KXINFL-MAR24</code> or <code>KXJOBS-JAN25</code> to test.
      </p>

      {/* You can keep your old event table or remove it for now */}
    </main>
  );
}
