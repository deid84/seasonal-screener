# seasonal-screener — Seasonality + Volatility Screener for Options

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A toolkit that combines historical seasonality analysis, volatility assessment,
and options-specific tools to help identify and evaluate monthly options setups
on a given list of tickers.

Data source: Yahoo Finance via `yfinance`. No paid data feed required.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [screener.py — CLI screening](#screenerpy)
5. [backtest.py — walk-forward backtest](#backtestpy)
6. [server.py — HTTP dashboard](#serverpy)
7. [Docker deployment](#docker-deployment)
8. [Recommended workflow](#recommended-workflow)
9. [Limitations — read before trading](#limitations--read-before-trading)

---

## What it does

| Module | Purpose |
|---|---|
| `screener.py` | Main CLI: seasonality, HV/IV, Greeks, skew, expected move, strategy suggestion, technicals, earnings warning, watchlist |
| `seasonality.py` | Monthly return statistics, t-test with Benjamini-Hochberg correction, sub-period consistency check |
| `volatility.py` | Realized HV percentile + live ATM IV snapshot with Greeks, skew, and liquidity check |
| `technicals.py` | MA50/MA200, RSI(14), 52-week range percentile, trend bias, seasonal alignment |
| `options_analysis.py` | Expected move, strategy selector (3×3 matrix), Black-Scholes pricing and Greeks |
| `iv_archive.py` | Accumulates daily IV snapshots in SQLite to compute IV Rank / IV Percentile over time |
| `db.py` | Persists screening results, backtest results, and watchlist to SQLite |
| `backtest.py` | Walk-forward backtest of the seasonal signal — price-only or options-aware |
| `server.py` | FastAPI HTTP server + built-in APScheduler daily run; serves the web dashboard |

---

## Installation

Requires **Python 3.10+** and an internet connection. No paid data feed — all
prices and options chains are fetched from Yahoo Finance via `yfinance`.

```bash
git clone https://github.com/YOUR_USERNAME/seasonal-screener.git
cd seasonal-screener
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The SQLite database (`iv_archive.db`) is created automatically on first run.
There is nothing else to configure for local use.

---

## Quick start

Choose the path that fits your use case.

### A — CLI only (no server needed)

Screen a list of tickers and print the report to the terminal:

```bash
python screener.py --tickers GLD,XRT,EQT --years 5
```

Run the backtest for a ticker:

```bash
python backtest.py --tickers GLD --years 10
python backtest.py --tickers GLD --years 10 --strategy long-call
```

### B — Local web dashboard

Run the screener once to populate the database, then start the server:

```bash
python screener.py --tickers GLD,XRT,EQT --years 5 --output db
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```

Open `http://localhost:8000`. The server will also run the screener
automatically every weekday at 22:00 UTC from that point on.

### C — Docker on a server (persistent, automated)

See [Docker deployment](#docker-deployment) below. One container handles both
the HTTP server and the daily scheduled run. No cron required.

---

## screener.py

Screens one or more tickers and prints a detailed report for each, followed by
a comparative ranking.

```bash
# Print report to terminal
python screener.py --tickers GLD,XRT,EQT,UNG --years 5

# Skip live options chain (faster — seasonality + HV only)
python screener.py --tickers AAPL --years 10 --no-options

# Save results to SQLite for the dashboard
python screener.py --tickers GLD,XRT --years 5 --output db

# Save ranking to CSV as well
python screener.py --tickers GLD,XRT --years 5 --csv ranking.csv
```

### Watchlist

Instead of typing tickers every time, store a persistent watchlist in the database:

```bash
# Add tickers
python screener.py --add GLD,XRT,EQT,UNG

# Run screener on the whole watchlist
python screener.py --watchlist --years 5

# Show current watchlist
python screener.py --list-watchlist

# Remove a ticker
python screener.py --remove UNG
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--tickers` | — | Comma-separated list of tickers (e.g. `GLD,XRT,EQT`) |
| `--watchlist` | off | Run on the stored watchlist instead of `--tickers` |
| `--add` | — | Add tickers to the watchlist and exit |
| `--remove` | — | Remove tickers from the watchlist and exit |
| `--list-watchlist` | off | Print the current watchlist and exit |
| `--years` | `5` | Years of price history to download |
| `--no-options` | off | Skip the live options chain fetch |
| `--output` | `print` | `print` (stdout) or `db` (save to SQLite for the dashboard) |
| `--iv-archive` | `iv_archive.db` | Path to the IV history database |
| `--csv` | — | Save the final ranking table to a CSV file |

### Output sections (per ticker)

**Earnings warning** *(shown at top if applicable)*
If an earnings date falls within the next 45 days (via `yfinance` calendar), a warning
appears before any analysis. Earnings inside the DTE window cause IV to spike and
compress unpredictably — the seasonal signal becomes unreliable for options strategies.
Note: only works for single stocks. ETFs have no earnings date in yfinance.

---

**Current and next month seasonality**
Historical average return, win rate, number of observations, and two quality flags.

*Statistical significance* — one-sample t-test (H₀: avg return = 0) with
**Benjamini-Hochberg FDR correction** across all 12 simultaneous tests.
Without correction, ~0.6 false positives are expected at α=0.05 purely by chance:
- `significant (p_adj=0.021)` — pattern survives multiple-testing correction
- `marginal (p_adj=0.08)` — borderline; treat with caution
- `not significant (p_adj=0.45)` — no statistically reliable edge

*Sub-period consistency* — the historical record is split into two equal halves.
If the pattern (direction + win_rate side) holds in both halves it is flagged
`consistent`; if it reverses or contradicts, `mixed`. With only 5–6 years of
data `mixed` is common and expected — it means the pattern is not yet proven
stable, not that it is wrong.

**Full seasonality table**
All 12 months with `avg_pct`, `std_pct`, `n_obs`, `win_rate_pct`, `consistency`,
`p_value`, and `p_value_adj`. Read `n_obs`, `p_value_adj`, and `consistency`
together — a month with 4 observations and `mixed` consistency is not actionable
regardless of the p-value.

---

**Realized historical volatility (HV)**
Current 20-day realized volatility and its percentile relative to the downloaded
history. Used as a proxy for IV when the IV archive is empty — see Limitations.

**Live IV snapshot**
ATM call and put for the next 1–2 available expiries: IV, bid/ask spread,
liquidity flag, Greeks (Δ, Γ, Θ/day, V/1%IV), and skew (put IV − call IV).

*Liquidity flag* — if either leg's bid-ask spread exceeds **15% of the mid**,
or if the mid is zero, the expiry is flagged `⚠ ILLIQUID`. Spread percentages
are shown for both legs so you can judge tradability at a glance. Always verify
on your broker before placing an order — yfinance data can lag.

**IV Rank / IV Percentile** *(available after 30+ daily runs)*
Once the local archive has enough snapshots:
- **IV Rank** = (current IV − period low) / (period high − period low) × 100
- **IV Percentile** = % of stored days where IV was below today's level

More accurate than the HV proxy because it uses real market IV, not realized vol.
The archive grows automatically with every screener run that fetches options data.

---

**Options analysis**
- **Expected move** = IV × √(DTE/365), expressed as ±% and ±$ with the implied
  1-sigma price range.
- **Pricing ratio** = expected move / |seasonal avg|. Below 1: options price in
  less than the historical seasonal move (favours buyers). Above 2: options price
  in more than twice the historical move (favours sellers).
- **Strategy suggestion** from directional bias (bullish/bearish/neutral)
  × IV level (low/normal/high):

  | Bias | IV low | IV normal | IV high |
  |---|---|---|---|
  | Bullish | Long Call / Debit Spread | Short Put | Short Put / CSP |
  | Bearish | Long Put / Debit Spread | Short Call | Short Call / Spread |
  | Neutral | Long Straddle / Strangle | No clear edge | Iron Condor / Strangle |

---

**Technical context**
Calculated from the same price history already downloaded — no extra API calls.

| Indicator | What it tells you |
|---|---|
| Price vs MA50/MA200 | Current trend direction and momentum |
| RSI(14) | Overbought (>70) / oversold (<30) context |
| 52-week range percentile | Where price sits in its annual range (0 = at lows, 100 = at highs) |
| Trend bias | BULLISH / BEARISH / NEUTRAL — synthesises MA position and RSI |
| **Trend alignment** | **ALIGNED / DIVERGENT / NEUTRAL vs seasonal bias** |

The most actionable output is **trend alignment**. A seasonal bullish signal with
a `DIVERGENT` technical trend has meaningfully lower conviction than an `ALIGNED`
one — the tool flags it explicitly so you can decide whether to skip or downweight
that setup.

---

## backtest.py

Walk-forward backtest of the seasonal signal. For each test month, seasonality
and HV are computed **exclusively on data preceding that month**, eliminating
lookahead bias.

### Price-only backtest (default)

Enters long on the underlying at the first close of the month, exits at the last
close. Useful baseline to check whether the seasonal signal has an edge before
adding options complexity.

```bash
python backtest.py --tickers GLD,XRT --years 10
python backtest.py --tickers GLD --years 10 --trend-filter
```

### Options-aware backtest (`--strategy`)

Prices an ATM European option at month entry using **Black-Scholes with the
20-day realized HV as the IV proxy**, holds to month-end, and records P&L as
% of spot.

```bash
python backtest.py --tickers GLD --years 10 --strategy long-call
python backtest.py --tickers XRT --years 10 --strategy short-put --trend-filter
python backtest.py --tickers GLD,XRT --years 10 --strategy long-call --csv bt.csv
```

Supported strategies: `long-call`, `short-put`, `long-put`, `short-call`.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--tickers` | required | Comma-separated list of tickers |
| `--years` | `10` | Years of history to download |
| `--min-history` | `3` | Warm-up years before testing begins |
| `--entry-avg` | `1.0` | Min historical avg monthly return (%) to trigger entry |
| `--entry-wr` | `55` | Min historical win rate (%) to trigger entry |
| `--trend-filter` | off | Only enter when price is above its 200-day SMA |
| `--strategy` | — | If set, runs the options-aware backtest |
| `--output` | `print` | `print` (stdout) or `db` (save to SQLite) |
| `--iv-archive` | `iv_archive.db` | Path to the IV history database |
| `--csv` | — | Save the full trade log to a CSV file |

### Interpreting the options backtest output

The report shows two columns side by side: **Options P&L** and **Underlying
return** for the same signal months.

- **Avg entry premium %**: average option cost as % of spot at entry. A long
  call costing 2% needs the underlying to move more than 2% to break even.
- **Win rate (options)** is typically lower than the underlying win rate because
  theta decay means you can be right on direction but still lose if the move
  doesn't cover the premium.
- **Total return (compounded)** is the most informative single number: compare
  the options column to the underlying column to gauge the effect of leverage
  and theta cost.
- The **IV%** column in the trade detail shows the HV used as IV proxy at
  entry. Values above 30% mean expensive options and conservative long-premium P&L.

---

## server.py

FastAPI HTTP server with a built-in daily scheduler. Run it instead of (or
alongside) the CLI to get a web dashboard and automated daily screening.

```bash
# Development
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
# → http://localhost:8000
```

> **Important:** always use `--workers 1`. Multiple workers each start their
> own scheduler instance, causing duplicate screening runs.

### Dashboard

Opening `http://localhost:8000` shows a single-page dashboard with:
- Sortable ranking table (click any column header to sort)
- Detail panel per ticker: seasonality table with colour-coded p-values,
  volatility/IV section, options analysis, Greeks
- IV history chart (requires 2+ daily runs to populate)

### Automatic daily screening

The scheduler runs the equivalent of `screener.py --output db` automatically
every weekday at the configured time (default 22:00 UTC). Configure via env vars:

| Env var | Default | Description |
|---|---|---|
| `TICKERS` | `GLD,XRT,EQT` | Comma-separated tickers to screen |
| `YEARS` | `5` | Years of history |
| `SCHEDULE_HOUR` | `22` | UTC hour for the daily run |
| `SCHEDULE_MIN` | `0` | UTC minute for the daily run |
| `IV_ARCHIVE_DB` | `iv_archive.db` | Path to the SQLite database |

### Manual trigger

To run screening immediately without waiting for the schedule:

```bash
curl -X POST http://localhost:8000/api/run
```

### API endpoints

| Endpoint | Description |
|---|---|
| `GET /api/results` | Latest screening run, all tickers sorted by score |
| `GET /api/results/{ticker}` | Score history for a specific ticker (last 90 days) |
| `GET /api/iv-history/{ticker}` | Stored IV history used for the chart |
| `GET /api/backtest/{ticker}` | Most recent backtest result for a ticker |
| `POST /api/run` | Triggers an immediate screening run in the background |

---

## Docker deployment

The project ships as a single image that acts as both the HTTP server and the
daily scheduler. Add it to an existing `docker-compose.yml` using the provided
snippet.

### Files in `deploy/`

| File | Purpose |
|---|---|
| `deploy/docker-compose.snippet.yml` | Service block to copy into your compose |
| `deploy/nginx.conf` | nginx `server {}` block (subdomain or sub-path variant) |

### Adding to an existing docker-compose

1. Build the image or let compose build it:
   ```yaml
   # excerpt from deploy/docker-compose.snippet.yml
   services:
     seasonal-screener:
       build:
         context: ./seasonal-screener
         dockerfile: Dockerfile
       restart: always
       environment:
         IV_ARCHIVE_DB: /data/iv_archive.db
         TICKERS: GLD,XRT,EQT,UNG
         YEARS: "5"
         SCHEDULE_HOUR: "22"
         SCHEDULE_MIN: "0"
       volumes:
         - screener_data:/data
       networks:
         - proxy   # same network as your nginx
   ```

2. Add the named volume `screener_data` and make sure the service is on the
   same Docker network as nginx.

3. Add an nginx location or server block from `deploy/nginx.conf`. The service
   is reachable internally as `http://seasonal-screener:8000`.

4. Optionally add HTTP basic auth (recommended — the dashboard exposes portfolio
   data). See the comments in `deploy/nginx.conf`.

### First run

On first start the database is empty. The dashboard will show "No screening
results yet." Either wait for the scheduled run or trigger one immediately:

```bash
curl -X POST https://screener.yourdomain.com/api/run
```

---

## Recommended workflow

1. **Screen** (`screener.py --watchlist` or let the server run automatically).
   Identify tickers where the seasonal signal is statistically non-trivial
   (`p_adj < 0.10`), has enough history (`n_obs ≥ 7`), and shows `consistent`
   sub-period behaviour.

2. **Check the earnings warning.** If it fires, skip or wait — IV behaviour
   around earnings makes the seasonal signal unreliable for options.

3. **Check technical alignment.** An `ALIGNED` trend confirmation raises
   conviction. A `DIVERGENT` flag is a reason to downweight or skip the setup,
   especially if the RSI is extreme.

4. **Check liquidity.** If the nearest expiry shows `⚠ ILLIQUID`, look at the
   next expiry or check on your broker — yfinance spreads are indicative only.

5. **Review the strategy suggestion.** Verify that bias, IV level, and pricing
   ratio are consistent with your own view before sizing a position.

6. **Run the price-only backtest** to confirm the signal has an out-of-sample
   edge on the underlying over your desired lookback period.

7. **Run the options backtest** (`--strategy long-call` or whichever was
   suggested) to see whether the edge survives after accounting for premium and
   theta decay.

8. **Let the IV archive accumulate.** After 30+ daily runs IV Rank replaces the
   HV proxy for a more accurate read on whether options are cheap or expensive.

---

## Limitations — read before trading

**Small sample size.** With 5 years of data you have at most 5 observations per
month. Benjamini-Hochberg correction reduces false positives but cannot fix the
fundamental problem: p-values are unreliable below ~10 observations. Prefer
`--years 10` or more for any signal you intend to trade.

**Sub-period consistency is weak with short history.** Splitting 5–6 years into
two halves gives ~2–3 observations per half per month. `mixed` is almost always
expected at this sample size — it is not a negative signal per se, just an honest
reflection of insufficient data to confirm stability.

**HV ≠ IV.** Realized historical volatility is used as a proxy for implied
volatility until the IV archive accumulates 30+ observations. Real IV reflects
market expectations and can diverge significantly from HV around earnings, macro
events, or volatility regime shifts. Let the archive grow before relying on the
HV percentile for IV-level decisions.

**Options backtest uses HV as IV proxy.** Historical IV data is not freely
available. The backtest underestimates premium in high-IV regimes and
overestimates it in low-IV regimes. Treat results as directionally informative,
not exact.

**Liquidity filter is indicative.** The 15% bid-ask threshold is applied to
yfinance data, which can lag or differ from real market conditions. Always verify
on your broker before trading.

**Technical indicators are price-only.** MA50/MA200 and RSI(14) capture trend
and momentum but ignore volume, market breadth, or macro context. A `DIVERGENT`
alignment flag is a prompt to investigate further, not an automatic disqualifier.

**Earnings filter covers single stocks only.** ETFs do not have earnings dates
in yfinance. For ETFs with earnings-sensitive underlying exposure (e.g. sector
ETFs near reporting season), check manually.

**ATM strike = entry spot.** The backtest uses the exact spot price as strike.
Real discrete strikes introduce a small delta bias.

**No transaction costs.** Bid/ask spread, commissions, and slippage are not
modelled. For options with wide spreads, actual P&L will be materially worse.

**This is an informational tool, not financial advice.** Use it as a starting
point for your own research, not as sufficient reason to open a position.
