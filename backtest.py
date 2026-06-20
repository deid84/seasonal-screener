"""
backtest.py
Walk-forward backtest of the monthly seasonality signal.

For each month in the test window (after a configurable warm-up period):
  1. Computes seasonality using ONLY data preceding that month (no lookahead)
  2. If avg historical return >= entry_avg_pct AND win_rate >= entry_win_rate:
     simulates a long position entered at the first close and exited at the
     last close of that month
  3. Records the actual return and computes aggregate performance metrics

Key design choice: the seasonality for month M of year Y is always computed
on data up to the last trading day of month M-1 of year Y. This eliminates
lookahead bias and gives an honest estimate of out-of-sample performance.

USAGE:
    python backtest.py --tickers GLD,XRT,EQT --years 10
    python backtest.py --tickers AAPL --years 15 --entry-avg 1.5 --entry-wr 60
    python backtest.py --tickers GLD,XRT --years 10 --csv backtest_results.csv
    python backtest.py --tickers GLD --years 10 --min-history 5 --entry-avg 0.5
"""
import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

from seasonality import compute_monthly_seasonality, MONTH_NAMES


def fetch_history(ticker: str, years: int) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def run_backtest(
    price_df: pd.DataFrame,
    min_history_years: int = 3,
    entry_avg_pct: float = 1.0,
    entry_win_rate: float = 55.0,
    trend_filter: bool = False,
) -> dict:
    """
    Walk-forward backtest of the monthly seasonality signal.

    Parameters
    ----------
    price_df : DataFrame with a 'Close' column indexed by DatetimeIndex.
    min_history_years : warm-up period; the signal is not computed until
                        this many years of data are available.
    entry_avg_pct : minimum historical average monthly return to trigger entry.
    entry_win_rate : minimum historical win rate (%) to trigger entry.
    trend_filter : if True, only enter when the last close before the month
                   is above its 200-day SMA (computed on pre-month data only).

    Returns
    -------
    dict with keys:
      'trades'  — DataFrame, one row per evaluated month (traded or not)
      'summary' — dict of aggregate performance metrics
    """
    df = price_df.copy()
    df["year"] = df.index.year
    df["month"] = df.index.month

    # Monthly returns using the same method as seasonality.py:
    # first close vs last close of the month.
    monthly = (
        df.groupby(["year", "month"])["Close"]
        .agg(["first", "last"])
        .reset_index()
    )
    monthly["ret_pct"] = (monthly["last"] / monthly["first"] - 1) * 100
    monthly = monthly.sort_values(["year", "month"]).reset_index(drop=True)

    warmup = min_history_years * 12
    records = []

    for i in range(warmup, len(monthly)):
        row = monthly.iloc[i]
        year, month_num = int(row["year"]), int(row["month"])
        month_name = MONTH_NAMES[month_num]
        actual_ret = float(row["ret_pct"])

        # Seasonality computed exclusively on data before this month.
        cutoff = pd.Timestamp(year=year, month=month_num, day=1)
        hist = price_df[price_df.index < cutoff]
        if hist.empty:
            continue

        try:
            seas = compute_monthly_seasonality(hist)
        except Exception:
            continue

        if month_name not in seas.index:
            continue

        s = seas.loc[month_name]
        signal_avg = float(s["avg_pct"])
        signal_wr = float(s["win_rate_pct"])
        signal_n = int(s["n_obs"])

        # SMA 200 trend filter (computed on pre-month data, no lookahead)
        if len(hist) >= 200:
            sma200 = float(hist["Close"].rolling(200).mean().iloc[-1])
            above_sma200 = bool(hist["Close"].iloc[-1] > sma200)
        else:
            above_sma200 = None  # not enough data — filter not applied for this month

        traded = signal_avg >= entry_avg_pct and signal_wr >= entry_win_rate
        if trend_filter and above_sma200 is not None:
            traded = traded and above_sma200

        records.append({
            "year": year,
            "month": month_name,
            "month_num": month_num,
            "signal_avg_pct": round(signal_avg, 2),
            "signal_win_rate_pct": round(signal_wr, 1),
            "signal_n_obs": signal_n,
            "above_sma200": above_sma200,
            "actual_return_pct": round(actual_ret, 2),
            "traded": traded,
        })

    trades_df = pd.DataFrame(records)

    if trades_df.empty:
        return {
            "trades": trades_df,
            "summary": {"error": "No data remained after the warm-up period."},
        }

    n_evaluated = len(trades_df)
    baseline_avg = round(float(trades_df["actual_return_pct"].mean()), 2)
    baseline_wr = round(float((trades_df["actual_return_pct"] > 0).mean() * 100), 1)

    traded = trades_df[trades_df["traded"]].copy()
    n_traded = len(traded)

    if n_traded == 0:
        return {
            "trades": trades_df,
            "summary": {
                "n_evaluated": n_evaluated,
                "n_traded": 0,
                "signal_rate_pct": 0.0,
                "baseline_avg_return_pct": baseline_avg,
                "baseline_win_rate_pct": baseline_wr,
                "note": "No months passed the entry filter — try lowering --entry-avg or --entry-wr.",
            },
        }

    rets = traded["actual_return_pct"].values
    actual_wr = round(float((rets > 0).mean() * 100), 1)
    avg_ret = round(float(rets.mean()), 2)
    std_ret = float(rets.std())
    sharpe = round((avg_ret / std_ret * np.sqrt(12)) if std_ret > 0 else 0.0, 2)

    # Compounded equity curve and max drawdown
    equity = np.cumprod(1 + rets / 100)
    peak = np.maximum.accumulate(equity)
    max_dd = round(float(((peak - equity) / peak).max() * 100), 2)
    total_ret = round(float((equity[-1] - 1) * 100), 2)

    # How many signal months were blocked by the trend filter
    n_blocked_by_trend = 0
    if trend_filter and "above_sma200" in trades_df.columns:
        signal_months = trades_df[
            (trades_df["signal_avg_pct"] >= entry_avg_pct) &
            (trades_df["signal_win_rate_pct"] >= entry_win_rate)
        ]
        n_blocked_by_trend = int((signal_months["above_sma200"] == False).sum())

    summary = {
        "n_evaluated": n_evaluated,
        "n_traded": n_traded,
        "signal_rate_pct": round(n_traded / n_evaluated * 100, 1),
        "trend_filter_active": trend_filter,
        "n_blocked_by_trend_filter": n_blocked_by_trend,
        "actual_win_rate_pct": actual_wr,
        "avg_return_per_trade_pct": avg_ret,
        "std_return_pct": round(std_ret, 2),
        "sharpe_annualized": sharpe,
        "max_drawdown_pct": max_dd,
        "total_return_compounded_pct": total_ret,
        "baseline_avg_return_pct": baseline_avg,
        "baseline_win_rate_pct": baseline_wr,
    }

    return {"trades": trades_df, "summary": summary}


def print_backtest_report(ticker: str, result: dict, entry_avg_pct: float, entry_win_rate: float):
    s = result["summary"]
    trades_df = result["trades"]

    trend_label = "ON" if s.get("trend_filter_active") else "OFF"
    print("=" * 72)
    print(f"  BACKTEST — {ticker}")
    print(f"  Entry filter: avg >= {entry_avg_pct}%  AND  win_rate >= {entry_win_rate}%  |  SMA200 filter: {trend_label}")
    print("=" * 72)

    if "error" in s:
        print(f"\n  {s['error']}\n")
        return

    if "note" in s:
        print(f"\n  Evaluated months : {s['n_evaluated']}")
        print(f"  Signal triggered : 0 (0.0% of months)")
        print(f"\n  Baseline avg return : {s['baseline_avg_return_pct']:+.2f}%  |  win rate: {s['baseline_win_rate_pct']}%")
        print(f"\n  {s['note']}\n")
        return

    print(f"\nEvaluated months : {s['n_evaluated']}")
    blocked = s.get("n_blocked_by_trend_filter", 0)
    if s.get("trend_filter_active") and blocked:
        print(f"Signal triggered : {s['n_traded']}  ({s['signal_rate_pct']}% of months)  "
              f"[{blocked} additional blocked by SMA200 filter]")
    else:
        print(f"Signal triggered : {s['n_traded']}  ({s['signal_rate_pct']}% of months)")
    print()

    print("--- Strategy (months where signal fired) --------")
    print(f"  Actual win rate       : {s['actual_win_rate_pct']}%")
    print(f"  Avg return per trade  : {s['avg_return_per_trade_pct']:+.2f}%")
    print(f"  Std dev per trade     : {s['std_return_pct']:.2f}%")
    print(f"  Sharpe (annualized)   : {s['sharpe_annualized']:.2f}")
    print(f"  Max drawdown          : -{s['max_drawdown_pct']:.2f}%")
    print(f"  Total return (cmpd.)  : {s['total_return_compounded_pct']:+.2f}%")

    print("\n--- Baseline (all evaluated months, no filter) --")
    print(f"  Avg return  : {s['baseline_avg_return_pct']:+.2f}%")
    print(f"  Win rate    : {s['baseline_win_rate_pct']}%")

    traded = trades_df[trades_df["traded"]].copy()
    print("\n--- Traded months detail -------------------------")
    cols = ["year", "month", "signal_avg_pct", "signal_win_rate_pct", "signal_n_obs"]
    if "above_sma200" in traded.columns:
        cols.append("above_sma200")
    cols.append("actual_return_pct")
    display = traded[cols].copy()
    rename = {"signal_avg_pct": "sig_avg%", "signal_win_rate_pct": "sig_wr%",
              "signal_n_obs": "sig_n", "above_sma200": "sma200↑", "actual_return_pct": "actual%"}
    display.columns = [rename.get(c, c) for c in display.columns]
    print(display.to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the seasonality screener")
    parser.add_argument("--tickers", required=True, help="Comma-separated list of tickers, e.g.: GLD,XRT")
    parser.add_argument("--years", type=int, default=10, help="Years of history to download (default: 10)")
    parser.add_argument("--min-history", type=int, default=3,
                        help="Warm-up years before testing begins (default: 3)")
    parser.add_argument("--entry-avg", type=float, default=1.0,
                        help="Min historical avg monthly return %% to trigger entry (default: 1.0)")
    parser.add_argument("--entry-wr", type=float, default=55.0,
                        help="Min historical win rate %% to trigger entry (default: 55)")
    parser.add_argument("--trend-filter", action="store_true",
                        help="Only enter when price is above its 200-day SMA")
    parser.add_argument("--csv", help="Save the full trade log to a CSV file")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    all_trades = []

    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        try:
            df = fetch_history(ticker, args.years)
            result = run_backtest(df, args.min_history, args.entry_avg, args.entry_wr, args.trend_filter)
            print_backtest_report(ticker, result, args.entry_avg, args.entry_wr)
            if not result["trades"].empty:
                result["trades"].insert(0, "ticker", ticker)
                all_trades.append(result["trades"])
        except Exception as e:
            print(f"  ERROR on {ticker}: {e}", file=sys.stderr)

    if args.csv and all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv(args.csv, index=False)
        print(f"Trade log saved to {args.csv}")


if __name__ == "__main__":
    main()
