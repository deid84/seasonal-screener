"""
volatility.py
Computes realized historical volatility and its current percentile
(proxy for "are options historically cheap or expensive right now"),
and fetches the real IV from the live options chain via yfinance.

IMPORTANT NOTE ON THE LIMITATIONS OF THIS APPROACH:
yfinance does not provide an implied volatility (IV) history.
To estimate whether options are relatively cheap "today", this module
uses the past REALIZED volatility of the underlying (HV, computed from
historical returns) as an approximate proxy, and calculates the percentile
of today's value relative to the analyzed period.

This is NOT the same as the IV percentile shown on a broker platform
(IBKR, ToS, Tastytrade, etc.), which reflects the market's expectations
about the future, not just past price behavior. The two measures are
correlated but can diverge significantly, especially around known events
(earnings, macro data releases). Treat this as a useful approximation
for a first screening pass, not as the definitive figure on which to
base a decision.
"""
from datetime import date as _date

import numpy as np
import pandas as pd
import yfinance as yf

from options_analysis import black_scholes_greeks as _bs_greeks


def realized_vol_percentile(price_df: pd.DataFrame, window: int = 20) -> dict:
    """
    Computes the annualized realized historical volatility on a rolling
    window of `window` days, and the percentile of the most recent value
    relative to the full historical period available in the DataFrame.
    """
    df = price_df.copy()
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    rolling_vol = log_ret.rolling(window).std() * np.sqrt(252) * 100  # annualized, in %

    rolling_vol = rolling_vol.dropna()
    if rolling_vol.empty:
        return {
            "hv_current_pct": None,
            "hv_percentile": None,
            "hv_avg_period_pct": None,
            "hv_min_period_pct": None,
            "hv_max_period_pct": None,
        }

    current_vol = rolling_vol.iloc[-1]
    percentile = (rolling_vol < current_vol).mean() * 100

    return {
        "hv_current_pct": round(float(current_vol), 2),
        "hv_percentile": round(float(percentile), 1),
        "hv_avg_period_pct": round(float(rolling_vol.mean()), 2),
        "hv_min_period_pct": round(float(rolling_vol.min()), 2),
        "hv_max_period_pct": round(float(rolling_vol.max()), 2),
    }


def fetch_atm_iv_snapshot(ticker: str, max_expiries: int = 2) -> list:
    """
    Fetches the real market IV for the options closest to ATM
    (at-the-money) across the next `max_expiries` available expiries
    on the live options chain.

    Returns a list of dicts with: expiry, spot, ATM strike,
    call/put IV in %, call/put volume.

    Requires an active internet connection (yfinance queries Yahoo Finance
    in real time). Returns an empty list if the ticker has no listed options.
    """
    tk = yf.Ticker(ticker)
    expiries = tk.options[:max_expiries]
    if not expiries:
        return []

    hist = tk.history(period="1d")
    if hist.empty:
        return []
    spot = hist["Close"].iloc[-1]

    results = []
    for exp in expiries:
        try:
            chain = tk.option_chain(exp)
        except Exception:
            continue
        calls, puts = chain.calls.copy(), chain.puts.copy()
        if calls.empty or puts.empty:
            continue

        calls["dist"] = (calls["strike"] - spot).abs()
        puts["dist"] = (puts["strike"] - spot).abs()
        atm_call = calls.sort_values("dist").iloc[0]
        atm_put = puts.sort_values("dist").iloc[0]

        iv_call_pct = round(float(atm_call["impliedVolatility"]) * 100, 1)
        iv_put_pct  = round(float(atm_put["impliedVolatility"]) * 100, 1)
        strike      = float(atm_call["strike"])
        dte         = max((_date.fromisoformat(exp) - _date.today()).days, 1)
        T           = dte / 365

        bid_c, ask_c = float(atm_call["bid"]), float(atm_call["ask"])
        bid_p, ask_p = float(atm_put["bid"]),  float(atm_put["ask"])
        mid_c = (bid_c + ask_c) / 2
        mid_p = (bid_p + ask_p) / 2
        spread_pct_call = round((ask_c - bid_c) / mid_c * 100, 1) if mid_c > 0 else None
        spread_pct_put  = round((ask_p - bid_p) / mid_p * 100, 1) if mid_p > 0 else None
        # illiquid if either leg spread exceeds 15% of mid, or mid is zero
        liquidity_ok = (
            mid_c > 0 and mid_p > 0
            and (spread_pct_call or 999) <= 15
            and (spread_pct_put  or 999) <= 15
        )

        results.append({
            "expiry": exp,
            "spot": round(float(spot), 2),
            "strike_atm": strike,
            "iv_call_pct": iv_call_pct,
            "iv_put_pct":  iv_put_pct,
            "skew_pct": round(iv_put_pct - iv_call_pct, 1),
            "volume_call": int(atm_call["volume"]) if pd.notna(atm_call["volume"]) else 0,
            "volume_put":  int(atm_put["volume"])  if pd.notna(atm_put["volume"])  else 0,
            "bid_call": bid_c,
            "ask_call": ask_c,
            "bid_put":  bid_p,
            "ask_put":  ask_p,
            "spread_pct_call": spread_pct_call,
            "spread_pct_put":  spread_pct_put,
            "liquidity_ok": liquidity_ok,
            "greeks_call": _bs_greeks(float(spot), strike, T, iv_call_pct, option_type="call"),
            "greeks_put":  _bs_greeks(float(spot), strike, T, iv_put_pct,  option_type="put"),
        })
    return results
