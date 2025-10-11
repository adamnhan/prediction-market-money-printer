export type Candle = { t: string; price: number; vol: number };

export async function fetchDemoCandles(): Promise<Candle[]> {
  // simple mock “API”
  const now = Date.now();
  const out: Candle[] = [];
  let p = 48 + Math.random() * 4;
  for (let i = 120; i >= 0; i--) {
    p += (Math.random() - 0.5) * 1.0;
    out.push({ t: new Date(now - i * 60_000).toISOString(), price: Math.max(0, Math.min(100, p)), vol: 50 + Math.random() * 120 });
  }
  return out;
}