// ui/components/AddEventForm.tsx
import type { FormEvent } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";

type Props = {
  newTicker: string;
  setNewTicker: (v: string) => void;
  onSubmit: (e: FormEvent) => void;
};

export function AddEventForm({ newTicker, setNewTicker, onSubmit }: Props) {
  return (
    <Card className="border-slate-800 bg-slate-900/70">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold uppercase tracking-wide text-slate-400">
          Add Event
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form
          onSubmit={onSubmit}
          className="flex flex-col gap-2 sm:flex-row sm:items-center"
        >
          <Input
            type="text"
            placeholder="Enter event ticker (e.g. KX...)"
            value={newTicker}
            onChange={(e) => setNewTicker(e.target.value)}
            className="flex-1 border-slate-700 bg-slate-950 text-sm placeholder:text-slate-500"
          />
          <Button
            type="submit"
            size="sm"
            className="gap-2 bg-sky-500 text-slate-950 hover:bg-sky-400"
            disabled={!newTicker.trim()}
          >
            <Plus className="h-4 w-4" />
            Add
          </Button>
        </form>
        <p className="mt-1 text-xs text-slate-500">
          Tickers are automatically uppercased before being sent.
        </p>
      </CardContent>
    </Card>
  );
}
