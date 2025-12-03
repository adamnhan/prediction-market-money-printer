// ui/components/EventsTable.tsx
import { Trash2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { EventEntry } from "../types/trading";

type Props = {
  events: EventEntry[];
  loading: boolean;
  onRemove: (ticker: string) => void;
};

export function EventsTable({ events, loading, onRemove }: Props) {
  return (
    <Card className="border-slate-800 bg-slate-900/70">
      <CardHeader className="flex flex-row items-center justify-between gap-2 pb-3">
        <CardTitle className="text-sm font-semibold uppercase tracking-wide text-slate-400">
          Tracked Events
        </CardTitle>
        <span className="text-xs text-slate-500">
          {events.length} event{events.length === 1 ? "" : "s"}
        </span>
      </CardHeader>
      <CardContent className="px-0 pb-0">
        {loading ? (
          <p className="px-4 pb-4 text-sm text-slate-500">Loading events...</p>
        ) : events.length === 0 ? (
          <p className="px-4 pb-4 text-sm text-slate-500">No events yet.</p>
        ) : (
          <div className="overflow-hidden rounded-t-lg border-t border-slate-800">
            <Table className="min-w-full text-sm">
              <TableHeader className="bg-slate-900/80">
                <TableRow>
                  <TableHead className="px-3 py-2">Ticker</TableHead>
                  <TableHead className="px-3 py-2">Title</TableHead>
                  <TableHead className="px-3 py-2">Status</TableHead>
                  <TableHead className="px-3 py-2">Category</TableHead>
                  <TableHead className="px-3 py-2 text-right">Price</TableHead>
                  <TableHead className="px-3 py-2 text-right">PnL</TableHead>
                  <TableHead className="px-3 py-2 text-right">
                    Actions
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody className="bg-slate-950/60">
                {events.map((evt) => (
                  <TableRow
                    key={evt.ticker}
                    className="hover:bg-slate-900/80"
                  >
                    <TableCell className="px-3 py-2 font-mono text-xs text-sky-300">
                      {evt.ticker}
                    </TableCell>
                    <TableCell className="px-3 py-2">
                      <div className="max-w-xs truncate text-sm">
                        {evt.title || "-"}
                      </div>
                    </TableCell>
                    <TableCell className="px-3 py-2">
                      <StatusBadge status={evt.status ?? "unknown"} />
                    </TableCell>
                    <TableCell className="px-3 py-2">
                      <div className="flex flex-col">
                        <span className="text-sm">{evt.category ?? "-"}</span>
                        {evt.sub_category && (
                          <span className="text-xs text-slate-500">
                            {evt.sub_category}
                          </span>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="px-3 py-2 text-right tabular-nums">
                      {evt.price}
                    </TableCell>
                    <TableCell
                      className={`px-3 py-2 text-right tabular-nums ${
                        evt.pnl > 0
                          ? "text-emerald-400"
                          : evt.pnl < 0
                          ? "text-rose-400"
                          : "text-slate-200"
                      }`}
                    >
                      {evt.pnl}
                    </TableCell>
                    <TableCell className="px-3 py-2 text-right">
                      <Button
                        variant="outline"
                        size="icon"
                        className="h-7 w-7 border-slate-700 text-slate-300 hover:border-rose-500 hover:text-rose-400"
                        onClick={() => onRemove(evt.ticker)}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function StatusBadge({ status }: { status: string }) {
  const normalized = status.toLowerCase();

  let variantClasses =
    "bg-slate-800 text-slate-200 border border-slate-700";

  if (normalized.includes("open")) {
    variantClasses =
      "bg-emerald-900/50 text-emerald-300 border border-emerald-700";
  } else if (normalized.includes("closed") || normalized.includes("expired")) {
    variantClasses =
      "bg-slate-900/70 text-slate-300 border border-slate-700";
  } else if (normalized.includes("pending")) {
    variantClasses =
      "bg-amber-900/50 text-amber-300 border border-amber-600";
  }

  return (
    <Badge
      className={`inline-flex max-w-full items-center truncate rounded-full px-2 py-0.5 text-xs capitalize ${variantClasses}`}
    >
      {status || "unknown"}
    </Badge>
  );
}
