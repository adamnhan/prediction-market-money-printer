# Phase F: Selection to Execution Flow (Detailed)

This document describes the current end‑to‑end logic for Phase F: how markets are discovered, how bundles are selected, and how execution is simulated.

## 1) Market Discovery

Phase F runs in a continuous loop and periodically refreshes open markets:

- `KXNBAGAME` (win markets)
- `KXNBASPREAD` (spread ladder)
- `KXNBATOTAL` (total ladder)

Each market is grouped by `event_key` (parsed from the ticker). The runner keeps a map:

```
event_key -> list of markets
```

This map is refreshed every `--poll-markets-s` seconds.

## 2) Ladder Construction

For each event:

- Build a **ladder** from available markets:
  - Spread: all `KXNBASPREAD-*` markets, parsed into `(team, k)` pairs
  - Total: all `KXNBATOTAL-*` markets, parsed into `(TOTAL, k)`
- Skip events when:
  - Ladder length < 3
  - Any book is missing for the ladder tickers

The ladder is sorted by `k` ascending.

## 3) Base Probability Ladder (phat)

Using historical results (`data/nba_results.csv`), compute:

```
phat[k] = P(margin >= k)
```

Steps:

1. Load final margins (or total points for totals).
2. Compute empirical probabilities for all `k`.
3. Apply isotonic decreasing fit to enforce monotonicity.

This produces a probability vector aligned with the ladder’s `k` values.

## 4) Optional Conditioning on Win Probability

If `--use-win-prob` is enabled:

1. Find the win market tickers for the event (from `KXNBAGAME`).
2. Pull top‑of‑book for those tickers.
3. Compute implied win probability as mid price:
   - If bid/ask both available, `(bid + ask) / 200`
   - Else fall back to whichever side is available
4. Adjust each leg’s `phat`:

```
ph_adjusted = phat + win_prob_strength * (p_win - 0.5)
```

Adjustment is applied per team:

- If leg is for home team: use `p_home`
- If leg is for away team: use `p_away`

Values are clamped to `[0, 1]`.

This is a **lightweight conditioning** intended to reduce obviously stale phat values during live play.

## 5) Candidate Enumeration

For each event, enumerate 3‑leg combinations of the ladder and **sign patterns** only:

- Buy/sell patterns are 2^3 combinations (`+/-` for each leg)
- Weight magnitudes are ignored (size is fixed to `qty=1`)

For each leg:

1. Determine action:
   - `BUY_YES` if sign > 0
   - `SELL_YES` if sign < 0
2. Price from top‑of‑book:
   - Buy: `yes_ask`
   - Sell: `yes_bid`
   - If missing, fall back to mid (counts toward `--max-mids`)
3. Compute:
   - `edge_ticks`:
     - Buy: `limit - ask`
     - Sell: `bid - limit`
   - `size_ratio`: `top_size / qty`
   - `spread`: `(ask - bid) / 100`

## 6) Fillability Filters

Each candidate bundle is filtered before scoring:

- `min_edge_ticks <= edge_ticks <= max_edge_ticks`
- `min_size_ratio` satisfied across all legs
- `mid_fallbacks <= --max-mids`

These filters prevent selecting bundles that are “good on paper” but unlikely to fill.

## 7) Scoring and Selection

Compute EV using sign‑only sizing:

```
EV = Σ sign * (phat - price)
```

Then compute a score:

```
score = EV
        - spread_penalty * max(spread)
        - size_penalty * size_shortfall
```

The candidate with the highest score is selected for that event.

## 8) Bundle Logging

Each selected bundle is logged to the ledger with:

- `event_key`, `ts_signal`, `ts_decision`
- `ev_raw`, `ev_net_est`
- full legs list with:
  - `ticker`, `k`, `side`, `qty`
  - `limit_price`, `px_bid`, `px_ask`, `px_used`
  - `phat`, `delta`

## 9) Execution (Shadow)

Two execution modes:

### A) Atomic

`execute_bundle` attempts to fill **all legs** inside `--ttl-s`:

- Buy leg fills if `ask <= limit` and `ask_size >= qty`
- Sell leg fills if `bid >= limit` and `bid_size >= qty`

If any leg fails, the bundle is rejected.

### B) Leg‑In

`execute_bundle_leg_in` attempts to build the bundle one leg at a time:

- At each step, choose the leg that most improves max loss
- Enforce risk gates:
  - `max_unhedged_loss`
  - `slope_cap`
- If TTL expires before completion:
  - status = `PARTIAL`
  - filled legs are retained
  - `unhedged_time_s` and `unhedged_max_loss` are recorded

## 10) Fill Logging

When a leg fills:

- `fill_price`, `fill_qty`, `fill_ts` are recorded
- `bundle_legs` is updated after execution

Bundle status is set to:

- `SHADOW_FILLED`
- `SHADOW_PARTIAL`
- `SHADOW_REJECT`

## 11) Settlement & Analysis

Settlement updater now writes:

- `bundle_settlements` (all attempts)
- `bundle_settlements_filled` (only filled or partial)

`guard_simulator.py` uses:

- leg‑centric settlement PnL
- actual fill data if available
- partials treated as open positions to settlement

## 12) Key Tunables (CLI)

Selection / fillability:

- `--min-ev`
- `--min-edge-ticks`, `--max-edge-ticks`
- `--min-size-ratio`
- `--spread-penalty`, `--size-penalty`

Execution:

- `--ttl-s`, `--poll-s`
- `--leg-in`
- `--max-unhedged-loss`, `--max-unhedged-ttl`, `--slope-cap`

Conditioning:

- `--use-win-prob`
- `--win-prob-strength`

## 13) Interpretation

This flow is designed to:

- avoid adverse‑selection bundles
- make partial‑leg PnL visible
- reduce noise in selection
- keep execution honest to real market constraints

If results remain poor after these changes, the next diagnosis step is to compare:

- **Signal EV** vs **realized PnL** (by EV bucket)
- **partial‑only PnL** vs **full‑bundle PnL**

