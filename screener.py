"""
screener.py
Command-line tool that, for a list of tickers:
  1. downloads price history (yfinance)
  2. computes monthly historical seasonality for each ticker
  3. estimates whether volatility (and therefore option cost) is
     currently low or high relative to its own history (HV proxy)
  4. fetches a live IV snapshot from the options chain
  5. produces a ticker ranking by "seasonal opportunity +
     contained option cost"

USAGE:
    source venv/bin/activate
    python screener.py --tickers GLD,XRT,EQT,UNG --years 5
    python screener.py --tickers AAPL --years 10 --no-options
    python screener.py --tickers GLD,XRT --years 5 --csv ranking.csv

Setup:
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

IMPORTANT: this script is an informational screening tool.
It does not generate trading signals and does not constitute financial advice.
See README.md for methodological limitations (in particular regarding
volatility estimation, see volatility.py).
"""
import argparse
import sys
from datetime import datetime

import pandas as pd
import yfinance as yf

from seasonality import compute_monthly_seasonality, MONTH_NAMES, reliability_flag
from volatility import realized_vol_percentile, fetch_atm_iv_snapshot
from iv_archive import save_snapshot, compute_iv_rank


def fetch_history(ticker: str, years: int) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    # yfinance may return MultiIndex columns even for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def analyze_ticker(ticker: str, years: int, fetch_options: bool = True, iv_db: str = "iv_archive.db") -> dict:
    current_month = datetime.now().month
    next_month = current_month % 12 + 1

    df = fetch_history(ticker, years)
    seas = compute_monthly_seasonality(df)
    vol = realized_vol_percentile(df)

    seas_now = seas.loc[MONTH_NAMES[current_month]] if MONTH_NAMES[current_month] in seas.index else None
    seas_next = seas.loc[MONTH_NAMES[next_month]] if MONTH_NAMES[next_month] in seas.index else None

    report = {
        "ticker": ticker,
        "years_requested": years,
        "n_years_actual_data": int(df.index.year.nunique()),
        "current_month": MONTH_NAMES[current_month],
        "next_month": MONTH_NAMES[next_month],
        "seasonality_current_month": seas_now.to_dict() if seas_now is not None else None,
        "seasonality_next_month": seas_next.to_dict() if seas_next is not None else None,
        "reliability_current_month": reliability_flag(seas_now) if seas_now is not None else None,
        "full_table": seas,
        "volatility": vol,
    }

    if fetch_options:
        try:
            snap = fetch_atm_iv_snapshot(ticker)
            report["options_snapshot"] = snap
            # persist snapshot and compute IV rank from accumulated history
            if isinstance(snap, list) and snap:
                save_snapshot(ticker, snap, iv_db)
                current_iv = (snap[0]["iv_call_pct"] + snap[0]["iv_put_pct"]) / 2
                report["iv_rank"] = compute_iv_rank(ticker, current_iv, db_path=iv_db)
            else:
                report["iv_rank"] = None
        except Exception as e:
            report["options_snapshot"] = f"unavailable ({e})"
            report["iv_rank"] = None

    return report


def score_opportunity(report: dict) -> float:
    """
    Simple explicit score, used ONLY for relative ranking among the
    input tickers. Not a buy/sell signal.

    Combines:
      - expected seasonality (current + next month, weighted by win rate)
      - volatility "discount" vs own history
        (low HV percentile = higher score = options probably
        cheaper than usual)
    """
    seas_now = report["seasonality_current_month"]
    seas_next = report["seasonality_next_month"]
    vol_pct = report["volatility"]["hv_percentile"]

    seas_score = 0.0
    for s in (seas_now, seas_next):
        if s:
            seas_score += s["avg_pct"] * (s["win_rate_pct"] / 100)

    if vol_pct is None:
        cheapness_score = 0.0
    else:
        cheapness_score = (100 - vol_pct) / 10  # 0 (vol alta) - 10 (vol ai minimi)

    return round(seas_score + cheapness_score, 2)


def print_report(report: dict):
    print("=" * 72)
    print(f"  {report['ticker']}  —  {report['n_years_actual_data']} years of actual data")
    print("=" * 72)

    print(f"\nCurrent month ({report['current_month']}):")
    s = report["seasonality_current_month"]
    if s:
        print(f"  avg historical return: {s['avg_pct']:+.2f}%  |  win rate: {s['win_rate_pct']}%  "
              f"|  observations: {int(s['n_obs'])}  |  {report['reliability_current_month']}")
    else:
        print("  insufficient data for this month")

    s2 = report["seasonality_next_month"]
    print(f"\nNext month ({report['next_month']}):")
    if s2:
        print(f"  avg historical return: {s2['avg_pct']:+.2f}%  |  win rate: {s2['win_rate_pct']}%  "
              f"|  observations: {int(s2['n_obs'])}")
    else:
        print("  insufficient data for this month")

    print("\nFull seasonality table (best to worst month historically):")
    print(report["full_table"].to_string())

    print("\nRealized historical volatility (option cost proxy — see limitations in README.md):")
    v = report["volatility"]
    if v["hv_percentile"] is not None:
        print(f"  current HV: {v['hv_current_pct']}%  |  historical percentile: {v['hv_percentile']}th"
              f"  (period range: {v['hv_min_period_pct']}% - {v['hv_max_period_pct']}%)")
        if v["hv_percentile"] < 25:
            print("  -> historically LOW volatility: options are probably cheaper than usual")
        elif v["hv_percentile"] > 75:
            print("  -> historically HIGH volatility: options are probably more expensive than usual")
        else:
            print("  -> volatility within normal range relative to own history")
    else:
        print("  insufficient data")

    if "options_snapshot" in report:
        print("\nLive IV snapshot from options chain (current market data):")
        snap = report["options_snapshot"]
        if isinstance(snap, list) and snap:
            for s3 in snap:
                print(f"  expiry {s3['expiry']}: spot {s3['spot']}, ATM strike {s3['strike_atm']}  |  "
                      f"call IV {s3['iv_call_pct']}% (bid {s3['bid_call']}/ask {s3['ask_call']})  |  "
                      f"put IV {s3['iv_put_pct']}% (bid {s3['bid_put']}/ask {s3['ask_put']})")
        elif isinstance(snap, list):
            print("  no listed options found for this ticker")
        else:
            print(f"  {snap}")

        ivr = report.get("iv_rank")
        if ivr and ivr.get("available"):
            print(f"\nIV Rank / IV Percentile (from {ivr['n_observations']} stored observations, "
                  f"period range {ivr['iv_min_period']}%–{ivr['iv_max_period']}%):")
            print(f"  IV Rank       : {ivr['iv_rank']}  "
                  f"({'LOW — options cheap vs own history' if ivr['iv_rank'] < 25 else 'HIGH — options expensive vs own history' if ivr['iv_rank'] > 75 else 'normal range'})")
            print(f"  IV Percentile : {ivr['iv_percentile']}th")
        elif ivr and not ivr.get("available"):
            print(f"\n  IV Rank: not yet available "
                  f"({ivr['n_observations']}/{ivr['min_required']} observations collected — keep running the screener daily)")

    print(f"\nCombined score for ranking (seasonality + volatility discount): {score_opportunity(report)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Seasonality + volatility screener for options")
    parser.add_argument("--tickers", required=True, help="Comma-separated list of tickers, e.g.: GLD,XRT,EQT,UNG")
    parser.add_argument("--years", type=int, default=5, help="Years of history to analyze (default 5)")
    parser.add_argument("--no-options", action="store_true", help="Skip the live options chain fetch")
    parser.add_argument("--iv-archive", default="iv_archive.db",
                        help="Path to the IV history database (default: iv_archive.db)")
    parser.add_argument("--csv", help="Save the final ranking to a CSV file")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    reports = []

    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        try:
            report = analyze_ticker(ticker, args.years, fetch_options=not args.no_options, iv_db=args.iv_archive)
            reports.append(report)
        except Exception as e:
            print(f"  ERROR on {ticker}: {e}", file=sys.stderr)

    if not reports:
        print("No analyzable data. Check the tickers and your internet connection.", file=sys.stderr)
        sys.exit(1)

    print("\n\n" + "#" * 72)
    print("#  DETAILED REPORT PER TICKER")
    print("#" * 72 + "\n")
    for r in reports:
        print_report(r)

    print("#" * 72)
    print("#  FINAL RANKING (combined score, descending)")
    print("#" * 72)
    ranking = sorted(reports, key=score_opportunity, reverse=True)
    rows = []
    for r in ranking:
        rows.append({
            "ticker": r["ticker"],
            "score": score_opportunity(r),
            "hv_percentile": r["volatility"]["hv_percentile"],
            "seasonality_current_month_pct": r["seasonality_current_month"]["avg_pct"] if r["seasonality_current_month"] else None,
            "seasonality_next_month_pct": r["seasonality_next_month"]["avg_pct"] if r["seasonality_next_month"] else None,
        })
    ranking_df = pd.DataFrame(rows)
    print(ranking_df.to_string(index=False))

    if args.csv:
        ranking_df.to_csv(args.csv, index=False)
        print(f"\nRanking saved to {args.csv}")


if __name__ == "__main__":
    main()
