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
  operator_flags?: {
    pause_entries: boolean;
    pause_all: boolean;
  };
  circuit_breakers?: {
    cooldown_until?: string | null;
    today_realized_pnl?: number;
    today_trades?: number;
    max_drawdown?: number;
    limits?: {
      daily_loss_limit?: number | null;
      max_drawdown?: number | null;
      max_trades_per_day?: number | null;
      cooldown_minutes_after_stop?: number | null;
    };
  };
  strategy_counters: {
    auto_entries: number;
    auto_exits: number;
    skipped_entries_due_to_risk: number;
    last_risk_skip_reason?: string | null;
  };
};

export type HealthState = {
  engine_running: boolean;
  ws_connected: boolean;
  ws_last_connect_ts?: string | null;
  ws_last_message_ts?: string | null;
  ws_last_error?: string | null;
  ws_stale?: boolean;
  ws_stale_seconds?: number | null;
  ws_subscriptions?: number;
};

export type MarketEntry = {
  event_ticker: string;
  market_ticker: string;
  status: string;
  last_price_yes: number | null;
  last_price_no: number | null;
  last_update_ts: string | null;
};
