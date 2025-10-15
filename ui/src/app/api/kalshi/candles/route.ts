// lib/kalshi-events.ts
export type MarketLite = {
  ticker: string;
  title: string;
  open_interest: number;
  liquidity: number;
};

export async function fetchAllEventMarketsLite(): Promise<MarketLite[]> {
  const BASE = "https://api.elections.kalshi.com/trade-api/v2/events";
  let cursor = "";
  const out: MarketLite[] = [];

  do {
    const url = cursor
      ? `${BASE}?limit=200&with_nested_markets=true&cursor=${encodeURIComponent(cursor)}`
      : `${BASE}?limit=200&with_nested_markets=true`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(await res.text());
    const data: { cursor?: string; events: any[] } = await res.json();

    for (const ev of data.events || []) {
      for (const m of ev.markets || []) {
        out.push({
          ticker: m.ticker,
          title: m.title,
          open_interest: Number(m.open_interest ?? 0),
          liquidity: Number(m.liquidity ?? 0),
        });
      }
    }
    cursor = data.cursor || "";
  } while (cursor);

  return out;
}

// convenience: sort by money already in the pool (open interest)
export function sortByOpenInterest(desc: boolean = true) {
  return (a: MarketLite, b: MarketLite) =>
    desc ? b.open_interest - a.open_interest : a.open_interest - b.open_interest;
}

const markets = await fetchAllEventMarketsLite();
const byPool = markets.sort(sortByOpenInterest(true)); // biggest “money in pool” first
