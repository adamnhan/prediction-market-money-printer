export type Candle = { t: string; price: number; vol: number };

export async function fetchKalshiCandles(ticker: string): Promise<Candle[]> {
  const res = await fetch(`/api/kalshi/candles?ticker=${encodeURIComponent(ticker)}`, {
    next: { revalidate: 0 },
  });
  if (!res.ok) throw new Error("Failed to load candles");
  const data = await res.json();
  return data.candles as Candle[];
}
