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
import contextlib
import io
import os
import sys
from datetime import datetime

_DEFAULT_DB = os.environ.get("IV_ARCHIVE_DB", "iv_archive.db")

import pandas as pd
import yfinance as yf

from seasonality import compute_monthly_seasonality, MONTH_NAMES, reliability_flag
from volatility import realized_vol_percentile, fetch_atm_iv_snapshot
from iv_archive import save_snapshot, compute_iv_rank
from options_analysis import compute_expected_move, suggest_strategy
from db import save_screening_result, add_to_watchlist, remove_from_watchlist, load_watchlist
from technicals import compute_technicals, trend_alignment


def fetch_history(ticker: str, years: int) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    # yfinance may return MultiIndex columns even for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def check_earnings(ticker: str, dte: int = 45) -> dict | None:
    """
    Returns a warning dict if an earnings date falls within the next `dte` days,
    None otherwise. A known earnings inside the DTE window changes IV dynamics
    and makes the seasonal signal unreliable for options strategies.
    """
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            cal = yf.Ticker(ticker).calendar
        if not cal:
            return None
        # calendar is a dict; 'Earnings Date' may be a list or a single Timestamp
        raw = cal.get("Earnings Date") or cal.get("earnings_date")
        if raw is None:
            return None
        dates = raw if isinstance(raw, list) else [raw]
        today = datetime.now().date()
        upcoming = []
        for d in dates:
            try:
                ed = d.date() if hasattr(d, "date") else datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
                days_away = (ed - today).days
                if 0 <= days_away <= dte:
                    upcoming.append({"date": str(ed), "days_away": days_away})
            except Exception:
                continue
        if upcoming:
            nearest = min(upcoming, key=lambda x: x["days_away"])
            return {
                "warning": True,
                "nearest_date": nearest["date"],
                "days_away": nearest["days_away"],
                "message": (
                    f"EARNINGS in {nearest['days_away']} days ({nearest['date']}) — "
                    f"IV may behave differently; seasonal signal less reliable for options"
                ),
            }
    except Exception:
        pass
    return None


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

    report["earnings_warning"] = check_earnings(ticker)
    report["technicals"] = compute_technicals(df)

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

    # Options analysis: expected move + strategy suggestion
    snap = report.get("options_snapshot")
    seas_now = report.get("seasonality_current_month")
    if isinstance(snap, list) and snap and seas_now:
        atm = snap[0]
        iv_mid = (atm["iv_call_pct"] + atm["iv_put_pct"]) / 2
        em = compute_expected_move(atm["spot"], iv_mid, atm["expiry"])

        # Use IV Rank if available, fall back to HV percentile
        ivr = report.get("iv_rank")
        if ivr and ivr.get("available"):
            iv_ref = ivr["iv_rank"]
        else:
            iv_ref = report["volatility"].get("hv_percentile")

        report["options_analysis"] = suggest_strategy(
            seas_now["avg_pct"],
            seas_now["win_rate_pct"],
            iv_ref,
            expected_move=em,
        )
        report["options_analysis"]["expected_move"] = em

    # Trend alignment: technical bias vs seasonal bias
    tech = report.get("technicals", {})
    oa   = report.get("options_analysis", {})
    if tech and not tech.get("error") and oa:
        report["trend_alignment"] = trend_alignment(tech["trend_bias"], oa["bias"])

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

    ew = report.get("earnings_warning")
    if ew:
        print(f"\n  ⚠  {ew['message']}")

    print(f"\nCurrent month ({report['current_month']}):")
    s = report["seasonality_current_month"]
    if s:
        cons = s.get("consistency", "")
        cons_tag = f"  |  consistency: {cons}" if cons else ""
        print(f"  avg historical return: {s['avg_pct']:+.2f}%  |  win rate: {s['win_rate_pct']}%  "
              f"|  observations: {int(s['n_obs'])}  |  {report['reliability_current_month']}{cons_tag}")
    else:
        print("  insufficient data for this month")

    s2 = report["seasonality_next_month"]
    print(f"\nNext month ({report['next_month']}):")
    if s2:
        cons2 = s2.get("consistency", "")
        cons2_tag = f"  |  consistency: {cons2}" if cons2 else ""
        print(f"  avg historical return: {s2['avg_pct']:+.2f}%  |  win rate: {s2['win_rate_pct']}%  "
              f"|  observations: {int(s2['n_obs'])}{cons2_tag}")
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
                liq = s3.get("liquidity_ok")
                liq_tag = "" if liq else "  ⚠ ILLIQUID"
                sc = s3.get("spread_pct_call")
                sp = s3.get("spread_pct_put")
                spread_info = f"  spread call {sc}% / put {sp}%" if (sc or sp) else ""
                print(f"  expiry {s3['expiry']}: spot {s3['spot']}, ATM strike {s3['strike_atm']}  |  "
                      f"call IV {s3['iv_call_pct']}% (bid {s3['bid_call']}/ask {s3['ask_call']})  |  "
                      f"put IV {s3['iv_put_pct']}% (bid {s3['bid_put']}/ask {s3['ask_put']})"
                      f"{spread_info}{liq_tag}")
            # Greeks and skew for the nearest expiry only
            s0 = snap[0]
            gc = s0.get("greeks_call", {})
            gp = s0.get("greeks_put", {})
            if gc:
                print(f"\n  Greeks [{s0['expiry']}]  —  ATM call: "
                      f"Δ {gc['delta']}  Γ {gc['gamma']}  Θ {gc['theta']}/day  V {gc['vega']}/1%IV")
                print(f"                           ATM put:  "
                      f"Δ {gp['delta']}  Γ {gp['gamma']}  Θ {gp['theta']}/day  V {gp['vega']}/1%IV")
            skew = s0.get("skew_pct")
            if skew is not None:
                direction = "put premium over calls" if skew > 0 else "call premium over puts"
                print(f"  Skew (put IV − call IV)  : {skew:+.1f}pp  ({direction})")
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

    oa = report.get("options_analysis")
    if oa:
        print("\nOptions analysis:")
        em = oa.get("expected_move", {})
        if "expected_move_pct" in em:
            print(f"  Expected move (options pricing) : ±{em['expected_move_pct']}%"
                  f"  (±${em['expected_move_dollar']}  →  range ${em['range_low']} / ${em['range_high']})"
                  f"  [DTE: {em['dte']}, expiry: {em['expiry']}]")
        seas_avg = oa.get("seasonal_avg_pct") or (
            report["seasonality_current_month"]["avg_pct"]
            if report.get("seasonality_current_month") else None
        )
        if seas_avg is not None:
            print(f"  Seasonal avg (historical)       : {seas_avg:+.2f}%")
        if "pricing_ratio" in oa:
            print(f"  Pricing ratio (EM / seasonal)   : {oa['pricing_ratio']}×  — {oa['pricing_note']}")

        bias_label = oa["bias"].upper()
        iv_label = oa["iv_level"].upper()
        print(f"\nStrategy suggestion  [bias: {bias_label}  |  IV: {iv_label}]:")
        print(f"  {oa['strategy']}")
        print(f"  Why    : {oa['rationale']}")
        print(f"  How    : {oa['structure']}")

    tech = report.get("technicals", {})
    if tech and not tech.get("error"):
        print("\nTechnical context:")
        if tech.get("ma50") and tech.get("ma200"):
            ma50_tag  = "above" if tech["price_vs_ma50_pct"]  > 0 else "below"
            ma200_tag = "above" if tech["price_vs_ma200_pct"] > 0 else "below"
            print(f"  Price vs MA50  : {tech['price_vs_ma50_pct']:+.1f}%  ({ma50_tag})")
            print(f"  Price vs MA200 : {tech['price_vs_ma200_pct']:+.1f}%  ({ma200_tag})")
        elif tech.get("ma50"):
            print(f"  Price vs MA50  : {tech['price_vs_ma50_pct']:+.1f}%  (MA200 unavailable — insufficient history)")
        print(f"  RSI(14)        : {tech['rsi14']}  ({tech['rsi_label']})")
        print(f"  52w range      : {tech['range_52w_pct']}th percentile")
        print(f"  Trend bias     : {tech['trend_bias'].upper()}")
        alignment = report.get("trend_alignment")
        if alignment:
            alignment_note = {
                "aligned":  "technical trend confirms seasonal bias",
                "divergent": "⚠  technical trend DIVERGES from seasonal bias — lower conviction",
                "neutral":  "one or both signals are neutral",
            }.get(alignment, "")
            print(f"  Trend alignment: {alignment.upper()}  — {alignment_note}")

    print(f"\nCombined score for ranking (seasonality + volatility discount): {score_opportunity(report)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Seasonality + volatility screener for options")
    parser.add_argument("--tickers", help="Comma-separated list of tickers, e.g.: GLD,XRT,EQT,UNG")
    parser.add_argument("--watchlist", action="store_true",
                        help="Use the stored watchlist as the ticker list")
    parser.add_argument("--add", metavar="TICKER[,...]",
                        help="Add one or more tickers to the watchlist and exit")
    parser.add_argument("--remove", metavar="TICKER[,...]",
                        help="Remove one or more tickers from the watchlist and exit")
    parser.add_argument("--list-watchlist", action="store_true",
                        help="Print the current watchlist and exit")
    parser.add_argument("--years", type=int, default=5, help="Years of history to analyze (default 5)")
    parser.add_argument("--no-options", action="store_true", help="Skip the live options chain fetch")
    parser.add_argument("--iv-archive", default=_DEFAULT_DB,
                        help="Path to the IV history database (default: iv_archive.db or $IV_ARCHIVE_DB)")
    parser.add_argument("--output", choices=["print", "db"], default="print",
                        help="Output mode: print to stdout (default) or save to iv_archive.db")
    parser.add_argument("--csv", help="Save the final ranking to a CSV file")
    args = parser.parse_args()

    # Watchlist management commands — run and exit immediately
    if args.add:
        for t in args.add.split(","):
            add_to_watchlist(t.strip(), db_path=args.iv_archive)
            print(f"Added {t.strip().upper()} to watchlist.")
        sys.exit(0)
    if args.remove:
        for t in args.remove.split(","):
            remove_from_watchlist(t.strip(), db_path=args.iv_archive)
            print(f"Removed {t.strip().upper()} from watchlist.")
        sys.exit(0)
    if args.list_watchlist:
        entries = load_watchlist(db_path=args.iv_archive)
        if not entries:
            print("Watchlist is empty.")
        else:
            print(f"{'Ticker':<10} {'Added':>12}  Notes")
            print("-" * 40)
            for e in entries:
                print(f"{e['ticker']:<10} {e['added_date']:>12}  {e['notes'] or ''}")
        sys.exit(0)

    # Resolve tickers: from --tickers or --watchlist
    if args.watchlist:
        entries = load_watchlist(db_path=args.iv_archive)
        if not entries:
            print("Watchlist is empty. Add tickers with --add TICKER.", file=sys.stderr)
            sys.exit(1)
        tickers = [e["ticker"] for e in entries]
        print(f"Using watchlist: {', '.join(tickers)}")
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        print("Specify --tickers or --watchlist (or --add/--remove to manage it).", file=sys.stderr)
        sys.exit(1)

    reports = []

    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        try:
            report = analyze_ticker(ticker, args.years, fetch_options=not args.no_options, iv_db=args.iv_archive)
            reports.append(report)
            if args.output == "db":
                save_screening_result(ticker, args.years, report,
                                      score_opportunity(report), args.iv_archive)
                print(f"  Saved to database.")
        except Exception as e:
            print(f"  ERROR on {ticker}: {e}", file=sys.stderr)

    if not reports:
        print("No analyzable data. Check the tickers and your internet connection.", file=sys.stderr)
        sys.exit(1)

    if args.output == "db":
        print(f"\nAll results saved to {args.iv_archive}.")
        sys.exit(0)

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
