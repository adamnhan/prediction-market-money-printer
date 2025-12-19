// ui/components/LogsPanel.tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";

type Props = {
  logs: string[];
  loading: boolean;
};

export function LogsPanel({ logs, loading }: Props) {
  return (
    <Card className="flex h-80 flex-col overflow-hidden border-slate-800 bg-slate-900/70">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold uppercase tracking-wide text-slate-400">
          Logs
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1">
        {loading ? (
          <p className="text-sm text-slate-500">Loading logs...</p>
        ) : logs.length === 0 ? (
          <p className="text-sm text-slate-500">No logs yet.</p>
        ) : (
          <div className="mt-1 flex h-64 flex-col rounded-lg border border-slate-800 bg-slate-950/60">
            <ScrollArea className="h-full p-3 text-xs font-mono text-slate-200">
              <ul className="space-y-1">
                {logs
                  .slice()
                  .reverse()
                  .map((log, idx) => (
                    <li key={idx} className="whitespace-pre-wrap">
                      {log}
                    </li>
                  ))}
              </ul>
            </ScrollArea>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
