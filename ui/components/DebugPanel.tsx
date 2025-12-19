// ui/components/DebugPanel.tsx
import { Bug } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { TradingState } from "../types/trading";

type Props = {
  state: TradingState | null;
  show: boolean;
  onToggle: () => void;
};

export function DebugPanel({ state, show, onToggle }: Props) {
  return (
    <Card className="flex h-80 flex-col overflow-hidden border-slate-800 bg-slate-900/70">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-400">
          <Bug className="h-4 w-4" />
          Raw Engine State
        </CardTitle>
        <Button
          variant="ghost"
          size="sm"
          className="h-auto px-2 py-1 text-xs text-sky-400 hover:text-sky-300"
          onClick={onToggle}
        >
          {show ? "Hide" : "Show"}
        </Button>
      </CardHeader>

      <CardContent className="flex-1">
        {show ? (
          <div className="mt-1 flex h-64 flex-col rounded-lg border border-slate-800 bg-slate-950/60">
            <ScrollArea className="h-full p-3">
              <pre className="text-xs font-mono text-slate-200 whitespace-pre">
                {state ? JSON.stringify(state, null, 2) : "Loading..."}
              </pre>
            </ScrollArea>
          </div>
        ) : (
          <p className="text-xs text-slate-500">
            Hidden for cleanliness. Click &quot;Show&quot; to inspect raw JSON.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
