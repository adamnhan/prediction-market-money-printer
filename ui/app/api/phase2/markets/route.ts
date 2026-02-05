import { NextResponse } from "next/server";
import fs from "node:fs/promises";
import path from "node:path";

export const dynamic = "force-dynamic";

const REPO_ROOT = path.resolve(process.cwd(), "..");

const DEFAULT_EV_PATHS = ["market_maker/logs/phase4_ev.csv"];

type MarketRow = {
  market_ticker: string;
  tier: string;
  incentive_ev_score: number;
  spread_quality_score: number;
  adverse_risk: number;
  inventory_risk: number;
  total_ev_score: number;
  incentive_discount: number;
  fills_30m: number;
  ts_utc?: string | null;
};

async function firstExistingPath(paths: string[]): Promise<string | null> {
  for (const rel of paths) {
    const full = path.resolve(REPO_ROOT, rel);
    try {
      await fs.access(full);
      return full;
    } catch {
      // keep trying
    }
  }
  return null;
}

function parseEvCsv(contents: string): MarketRow[] {
  const lines = contents.trim().split(/\r?\n/);
  if (lines.length <= 1) return [];
  const rows: MarketRow[] = [];
  for (let i = 1; i < lines.length; i += 1) {
    const line = lines[i];
    if (!line) continue;
    const [
      ts_utc,
      market_ticker,
      tier,
      incentive_ev_score,
      spread_quality_score,
      adverse_risk,
      inventory_risk,
      total_ev_score,
      incentive_discount,
      fills_30m,
    ] = line.split(",");
    rows.push({
      ts_utc: ts_utc?.trim() ?? null,
      market_ticker: market_ticker?.trim() ?? "",
      tier: tier?.trim() ?? "",
      incentive_ev_score: Number(incentive_ev_score ?? 0),
      spread_quality_score: Number(spread_quality_score ?? 0),
      adverse_risk: Number(adverse_risk ?? 0),
      inventory_risk: Number(inventory_risk ?? 0),
      total_ev_score: Number(total_ev_score ?? 0),
      incentive_discount: Number(incentive_discount ?? 0),
      fills_30m: Number(fills_30m ?? 0),
    });
  }
  return rows;
}

export async function GET(request: Request): Promise<NextResponse> {
  const url = new URL(request.url);
  const limit = Math.max(10, Math.min(500, Number(url.searchParams.get("limit") ?? 200)));
  const sortBy =
    url.searchParams.get("sort_by") ?? "total_ev_score";

  const evPathOverride = process.env.PHASE2_EV_PATH
    ? [process.env.PHASE2_EV_PATH]
    : [];
  const evPath =
    (await firstExistingPath([...evPathOverride, ...DEFAULT_EV_PATHS])) ?? null;

  let rows: MarketRow[] = [];
  if (evPath) {
    const csv = await fs.readFile(evPath, "utf8");
    rows = parseEvCsv(csv);
  }

  const latestByMarket = new Map<string, MarketRow>();
  for (const row of rows) {
    if (!row.market_ticker) continue;
    latestByMarket.set(row.market_ticker, row);
  }

  const markets = Array.from(latestByMarket.values());
  markets.sort((a, b) => {
    const key = sortBy as keyof MarketRow;
    const av = typeof a[key] === "number" ? (a[key] as number) : 0;
    const bv = typeof b[key] === "number" ? (b[key] as number) : 0;
    return bv - av;
  });

  return NextResponse.json({
    markets: markets.slice(0, limit),
    total: markets.length,
    path: evPath,
  });
}
