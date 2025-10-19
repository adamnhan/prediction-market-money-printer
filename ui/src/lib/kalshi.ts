export type Candle = { t: string; price: number; vol: number };

export async function fetchKalshiCandles(ticker: string): Promise<Candle[]> {
  const res = await fetch(`/api/kalshi/candles?ticker=${encodeURIComponent(ticker)}`, {
    next: { revalidate: 0 },
  });
  if (!res.ok) throw new Error("Failed to load candles");
  const data = await res.json();
  return data.candles as Candle[];
}

const BASE = "https://api.elections.kalshi.com/trade-api/v2/events";

export async function getEventTitles(): Promise<string[]> {
  let cursor = "";
  const titles: string[] = [];

  do {
    const url = cursor ? `${BASE}?limit=200&cursor=${encodeURIComponent(cursor)}`
                       : `${BASE}?limit=200`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(await res.text());
    const data: { cursor?: string; events?: { title: string }[] } = await res.json();
    for (const ev of data.events || []) titles.push(ev.title);
    cursor = data.cursor || "";
  } while (cursor);

  return titles;
}
