// ui/types/trading.ts
export type EventEntry = {
  ticker: string;
  title: string;
  price: number;
  pnl: number;
  status?: string;
  category?: string | null;
  sub_category?: string | null;
};

export type Capital = {
  total: number;
  used: number;
  available: number;
};

export type PositionView = {
  id: number;
  event_ticker: string;
  market_ticker: string;
  side: string;
  qty: number;
  entry_price: number;
  current_price: number;
  status: string;
  entry_ts?: string | null;
  exit_ts?: string | null;
  realized_pnl: number;
  unrealized_pnl: number;
};


export type TradingState = {
  events: EventEntry[];
  logs: string[];
  positions: PositionView[];
  capital: Capital;
  markets: Record<string, MarketEntry>;
};

export type MarketEntry = {
  event_ticker: string;
  market_ticker: string;
  status: string;
  last_price_yes: number | null;
  last_price_no: number | null;
};
