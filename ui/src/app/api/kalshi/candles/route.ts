import { NextResponse } from "next/server";

// Kalshi public market-data endpoint (no auth required for this data)
const BASE = "https://api.elections.kalshi.com/trade-api/v2";

export async function GET(req: Request) {
  const url = new URL(req.url);
  const ticker = url.searchParams.get("ticker") || "FED_RATE_DEC";
  // 2 hours of 1-min candles (adjust as you like)
  const end = Math.floor(Date.now() / 1000);
  const start = end - 2 * 60 * 60;

  const qs = new URLSearchParams({
    period_interval: "1",        // 1m candles (also supports 60 or 1440)
    start_ts: String(start),
    end_ts: String(end),
  });

  const res = await fetch(`${BASE}/markets/${encodeURIComponent(ticker)}/candlesticks?${qs.toString()}`, {
    // don't cache while developing
    cache: "no-store",
  });

  if (!res.ok) {
    const text = await res.text();
    return NextResponse.json({ error: text || "Kalshi error" }, { status: 502 });
  }

  type KalshiCandle = {
    start_ts: number; end_ts: number;
    open_price: number; high_price: number; low_price: number; close_price: number;
    volume: number;
  };

  const data = (await res.json()) as { candlesticks: KalshiCandle[] };

  // Map to your UI shape: { t, price, vol } — using close as the display price
  const candles = (data.candlesticks || []).map(c => ({
    t: new Date(c.end_ts * 1000).toISOString(),
    price: c.close_price,     // 0–100 scale on Kalshi
    vol: c.volume ?? 0,
  }));

  return NextResponse.json({ ticker, candles });
}
