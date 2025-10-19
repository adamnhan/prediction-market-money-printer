import { NextResponse } from "next/server";
import { getEventTitles } from "@/lib/kalshi"; 

export async function GET() {
  try {
    const titles = await getEventTitles();
    return NextResponse.json({ titles });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
