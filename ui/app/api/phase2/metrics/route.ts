import { NextResponse } from "next/server";
import fs from "node:fs/promises";
import path from "node:path";

export const dynamic = "force-dynamic";

const REPO_ROOT = path.resolve(process.cwd(), "..");

const DEFAULT_SUMMARY_PATHS = [
  "market_maker/logs/phase2_summary.csv",
];

const DEFAULT_LOG_PATHS = [
  "phase2.nohup.out",
  "market_maker/logs/phase2.log",
  "market_maker/logs/phase2.out",
];

type SummaryRow = {
  ts_utc: string;
  fills: number;
  actions: number;
  fill_rate: number;
  net_cash_change: number;
  inv_yes: number;
  inv_no: number;
};

type DashboardMetrics = {
  ts_utc?: string | null;
  cash_cents?: number | null;
  equity_cents?: number | null;
  inv_yes?: number | null;
  inv_no?: number | null;
  tier_a?: number | null;
  tier_b?: number | null;
  tier_c?: number | null;
  day_pnl_cents?: number | null;
  worst_window_cents?: number | null;
  soft_throttle?: number | null;
};

type SummarySeriesPoint = {
  ts_utc: string;
  net_cash_change: number;
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

async function readTail(filePath: string, maxBytes: number): Promise<string> {
  const stat = await fs.stat(filePath);
  const start = Math.max(0, stat.size - maxBytes);
  const length = stat.size - start;
  const handle = await fs.open(filePath, "r");
  try {
    const buffer = Buffer.alloc(length);
    await handle.read(buffer, 0, length, start);
    return buffer.toString("utf8");
  } finally {
    await handle.close();
  }
}

function parseSummaryCsv(contents: string): SummaryRow[] {
  const lines = contents.trim().split(/\r?\n/);
  if (lines.length <= 1) return [];
  const rows: SummaryRow[] = [];
  for (let i = 1; i < lines.length; i += 1) {
    const line = lines[i];
    if (!line) continue;
    const [ts_utc, fills, actions, fill_rate, net_cash_change, inv_yes, inv_no] =
      line.split(",");
    rows.push({
      ts_utc: ts_utc?.trim() ?? "",
      fills: Number(fills ?? 0),
      actions: Number(actions ?? 0),
      fill_rate: Number(fill_rate ?? 0),
      net_cash_change: Number(net_cash_change ?? 0),
      inv_yes: Number(inv_yes ?? 0),
      inv_no: Number(inv_no ?? 0),
    });
  }
  return rows;
}

function parseDashboardLine(line: string): DashboardMetrics | null {
  if (!line.includes(" dashboard ")) return null;
  const tsMatch = line.match(/\[PHASE2\]\s+([^\s]+)\s+dashboard/);
  const tiersMatch = line.match(/tiers=A:(\d+)\s+B:(\d+)\s+C:(\d+)/);
  const getNumber = (key: string) => {
    const match = line.match(new RegExp(`${key}=(-?\\d+)`));
    return match ? Number(match[1]) : null;
  };
  return {
    ts_utc: tsMatch?.[1] ?? null,
    cash_cents: getNumber("cash_cents"),
    equity_cents: getNumber("equity_cents"),
    inv_yes: getNumber("inv_yes"),
    inv_no: getNumber("inv_no"),
    tier_a: tiersMatch ? Number(tiersMatch[1]) : null,
    tier_b: tiersMatch ? Number(tiersMatch[2]) : null,
    tier_c: tiersMatch ? Number(tiersMatch[3]) : null,
    day_pnl_cents: getNumber("day_pnl_cents"),
    worst_window_cents: getNumber("worst_window_cents"),
    soft_throttle: getNumber("soft_throttle"),
  };
}

function extractLatestDashboard(logText: string): DashboardMetrics | null {
  const lines = logText.split(/\r?\n/);
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const parsed = parseDashboardLine(lines[i]);
    if (parsed) return parsed;
  }
  return null;
}

export async function GET(): Promise<NextResponse> {
  const summaryOverride = process.env.PHASE2_SUMMARY_PATH
    ? [process.env.PHASE2_SUMMARY_PATH]
    : [];
  const logOverride = process.env.PHASE2_LOG_PATH
    ? [process.env.PHASE2_LOG_PATH]
    : [];

  const summaryPath =
    (await firstExistingPath([...summaryOverride, ...DEFAULT_SUMMARY_PATHS])) ??
    null;
  const logPath =
    (await firstExistingPath([...logOverride, ...DEFAULT_LOG_PATHS])) ?? null;

  let summaryRows: SummaryRow[] = [];
  if (summaryPath) {
    const csv = await fs.readFile(summaryPath, "utf8");
    summaryRows = parseSummaryCsv(csv);
  }

  let dashboard: DashboardMetrics | null = null;
  if (logPath) {
    const logText = await readTail(logPath, 1024 * 1024);
    dashboard = extractLatestDashboard(logText);
  }

  const summaryTail = summaryRows.slice(-300);
  const lastSummary = summaryRows.length ? summaryRows[summaryRows.length - 1] : null;
  const series: SummarySeriesPoint[] = summaryTail.map((row) => ({
    ts_utc: row.ts_utc,
    net_cash_change: row.net_cash_change,
  }));

  return NextResponse.json({
    dashboard,
    summary: {
      last: lastSummary,
      series,
      recent: summaryTail.slice(-20),
    },
    paths: {
      summary: summaryPath,
      log: logPath,
    },
    generated_at: new Date().toISOString(),
  });
}
