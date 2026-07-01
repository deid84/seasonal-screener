"""
technicals.py
Lightweight technical indicators computed from the price history already
downloaded by the screener (no extra API calls).

Indicators
----------
MA50 / MA200  : simple moving averages; price position and % distance
RSI(14)       : Wilder's relative strength index
52-week range : where the current price sits in its annual high/low range (0–100)
trend_bias    : "bullish" / "bearish" / "neutral" based on MA and RSI together

These are used as a FILTER on the seasonal signal, not as independent signals.
The key question is: does the current technical context support or contradict
the seasonal directional bias?
"""
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> float:
    """Wilder's RSI. Returns the most recent value."""
    delta = close.diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # Wilder smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 1)


def compute_technicals(price_df: pd.DataFrame) -> dict:
    """
    Computes technical indicators from daily OHLCV data.

    Parameters
    ----------
    price_df : DataFrame with a 'Close' column indexed by DatetimeIndex,
               as returned by yfinance.download.

    Returns a dict with:
      ma50, ma200, price_vs_ma50_pct, price_vs_ma200_pct,
      rsi14, range_52w_pct, trend_bias, summary
    """
    close = price_df["Close"].dropna()

    if len(close) < 20:
        return {"error": "insufficient data for technical indicators"}

    current = float(close.iloc[-1])

    # Moving averages
    ma50  = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else None
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    pct_vs_ma50  = round((current / ma50  - 1) * 100, 2) if ma50  else None
    pct_vs_ma200 = round((current / ma200 - 1) * 100, 2) if ma200 else None

    # RSI(14)
    rsi = _rsi(close) if len(close) >= 20 else None

    # 52-week range percentile
    window_252 = close.iloc[-252:] if len(close) >= 252 else close
    lo_52, hi_52 = float(window_252.min()), float(window_252.max())
    range_52w_pct = round((current - lo_52) / (hi_52 - lo_52) * 100, 1) if hi_52 > lo_52 else 50.0

    # Trend bias
    bias = _trend_bias(current, ma50, ma200, rsi)

    # Human-readable summary line
    ma_str = ""
    if ma50 and ma200:
        ma_str = (f"above MA50 ({pct_vs_ma50:+.1f}%) and MA200 ({pct_vs_ma200:+.1f}%)"
                  if pct_vs_ma50 > 0 and pct_vs_ma200 > 0
                  else f"below MA50 ({pct_vs_ma50:+.1f}%) and MA200 ({pct_vs_ma200:+.1f}%)"
                  if pct_vs_ma50 < 0 and pct_vs_ma200 < 0
                  else f"MA50 {pct_vs_ma50:+.1f}% / MA200 {pct_vs_ma200:+.1f}% (mixed)")
    rsi_label = "overbought" if rsi and rsi > 70 else "oversold" if rsi and rsi < 30 else "neutral"

    return {
        "current_price": round(current, 2),
        "ma50":  round(ma50,  2) if ma50  else None,
        "ma200": round(ma200, 2) if ma200 else None,
        "price_vs_ma50_pct":  pct_vs_ma50,
        "price_vs_ma200_pct": pct_vs_ma200,
        "rsi14": rsi,
        "rsi_label": rsi_label,
        "range_52w_pct": range_52w_pct,
        "trend_bias": bias,
        "summary": f"{ma_str}  |  RSI {rsi} ({rsi_label})  |  52w range {range_52w_pct}th pct",
    }


def _trend_bias(price: float, ma50: float | None, ma200: float | None,
                rsi: float | None) -> str:
    """
    Simple trend bias classification.

    bullish : price above both MAs (or above MA50 if MA200 unavailable) AND RSI not overbought
    bearish : price below both MAs (or below MA50) AND RSI not oversold
    neutral : mixed signals or RSI extreme contradicting MA position
    """
    if ma50 is None:
        return "neutral"

    above_ma50  = price > ma50
    above_ma200 = price > ma200 if ma200 is not None else None

    if above_ma200 is not None:
        ma_bullish = above_ma50 and above_ma200
        ma_bearish = not above_ma50 and not above_ma200
    else:
        ma_bullish = above_ma50
        ma_bearish = not above_ma50

    rsi_overbought = rsi is not None and rsi > 70
    rsi_oversold   = rsi is not None and rsi < 30

    if ma_bullish and not rsi_overbought:
        return "bullish"
    if ma_bearish and not rsi_oversold:
        return "bearish"
    return "neutral"


def trend_alignment(tech_bias: str, seasonal_bias: str) -> str:
    """
    Compares the technical trend bias with the seasonal directional bias.

    Returns:
      "aligned"    — both point in the same direction
      "divergent"  — they point in opposite directions (caution)
      "neutral"    — one or both are neutral
    """
    if tech_bias == "neutral" or seasonal_bias == "neutral":
        return "neutral"
    if tech_bias == seasonal_bias:
        return "aligned"
    return "divergent"
