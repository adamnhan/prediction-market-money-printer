// ui/components/CapitalSummary.tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Capital } from "../types/trading";

export function CapitalSummary({ capital }: { capital: Capital }) {
  const items = [
    { label: "Total Capital", value: capital.total, icon: "💰" },
    { label: "Used", value: capital.used, icon: "📌" },
    { label: "Available", value: capital.available, icon: "✅" },
  ];

  return (
    <Card className="border-slate-800 bg-slate-900/70">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold uppercase tracking-wide text-slate-400">
          Capital
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid gap-3 sm:grid-cols-3">
          {items.map((item) => (
            <div
              key={item.label}
              className="rounded-lg border border-slate-800 bg-slate-950/70 px-3 py-2"
            >
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>{item.label}</span>
                <span>{item.icon}</span>
              </div>
              <div className="mt-1 text-lg font-semibold">
                {item.value.toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
