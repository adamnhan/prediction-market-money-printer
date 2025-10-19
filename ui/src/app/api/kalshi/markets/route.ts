import { NextResponse } from "next/server";

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);
    const ticker = searchParams.get("ticker");

    if (!ticker) {
      return NextResponse.json({ ok: false, error: "Missing ticker" }, { status: 400 });
    }

    
    // Normalize input
    const normalizedTicker = ticker.trim().toUpperCase();

    // Fetch from the public Kalshi API
    const url = `https://api.kalshi.com/trade-api/v2/markets/${encodeURIComponent(
      normalizedTicker
    )}`;

    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      return NextResponse.json(
        { ok: false, error: `Ticker ${normalizedTicker} not found` },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json({ ok: true, ticker: normalizedTicker, data });
  } catch (err: any) {
    console.error("Kalshi market lookup error:", err);
    return NextResponse.json({ ok: false, error: err.message }, { status: 500 });
  }
}
