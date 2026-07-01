"""
seasonality.py
Computes historical seasonality statistics (average monthly return,
win rate, standard deviation) for a ticker, from daily price history.
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def _bh_correction(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction for multiple comparisons.
    NaN entries (too few observations) are left as NaN.
    Returns adjusted p-values clipped to [0, 1].
    """
    m = len(p_values)
    adj = np.full(m, np.nan)
    valid_mask = ~np.isnan(p_values)
    if not valid_mask.any():
        return adj

    idx = np.where(valid_mask)[0]
    pv = p_values[idx]
    order = np.argsort(pv)
    ranks = np.argsort(order) + 1          # 1-based rank among valid p-values
    n_valid = len(pv)
    adjusted = pv[order] * n_valid / np.arange(1, n_valid + 1)
    # enforce monotonicity: running minimum from the largest rank
    for i in range(n_valid - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    adj[idx] = np.clip(adjusted[np.argsort(order)], 0, 1)
    return adj


def compute_monthly_seasonality(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    price_df: DataFrame with a 'Close' column indexed by DatetimeIndex
    (e.g. output of yfinance.download with daily data).

    Returns a DataFrame indexed by month (Jan..Dec) with columns:
    avg_pct, std_pct, n_obs, n_positive, win_rate_pct
    Sorted from the historically best to worst month.
    """
    df = price_df.copy()
    df["year"] = df.index.year
    df["month"] = df.index.month

    # approximate monthly return: first close vs last close of the month
    monthly = df.groupby(["year", "month"])["Close"].agg(["first", "last"])
    monthly["ret_pct"] = (monthly["last"] / monthly["first"] - 1) * 100
    monthly = monthly.reset_index()

    stats = monthly.groupby("month")["ret_pct"].agg(
        avg_pct="mean",
        std_pct="std",
        n_obs="count",
    )
    stats["n_positive"] = monthly.groupby("month")["ret_pct"].apply(lambda x: (x > 0).sum())
    stats["win_rate_pct"] = (stats["n_positive"] / stats["n_obs"] * 100).round(1)
    stats["avg_pct"] = stats["avg_pct"].round(2)
    stats["std_pct"] = stats["std_pct"].round(2)

    # t-test: H0 = avg monthly return is zero for this calendar month
    def _pvalue(x):
        if len(x) < 3:
            return np.nan
        _, p = scipy_stats.ttest_1samp(x, 0)
        return round(float(p), 4)

    stats["p_value"] = monthly.groupby("month")["ret_pct"].apply(_pvalue)

    # Sub-period consistency: same directional pattern in both halves of history
    consistency = check_subperiod_consistency(price_df)
    stats["consistency"] = consistency.reindex([MONTH_NAMES[m] for m in stats.index]).values

    # Benjamini-Hochberg FDR correction across all 12 simultaneous tests.
    # Prevents ~0.6 expected false positives at α=0.05 from the 12 t-tests.
    raw = stats["p_value"].values.astype(float)
    stats["p_value_adj"] = _bh_correction(raw)

    # months missing from the dataset (e.g. ticker too recent) are dropped
    stats.index = [MONTH_NAMES[m] for m in stats.index]
    stats.index.name = "month"
    return stats.sort_values("avg_pct", ascending=False)


def check_subperiod_consistency(price_df: pd.DataFrame) -> pd.Series:
    """
    Splits the history into two equal halves and checks whether each month's
    directional pattern (avg sign + win_rate side) is consistent across both.

    Returns a Series indexed by month name with values:
      "consistent"   — same direction and win_rate > 50% in both halves
      "mixed"        — direction or win_rate diverges between halves
      "insufficient" — fewer than 2 observations in one half
    """
    df = price_df.copy()
    df["year"] = df.index.year
    df["month"] = df.index.month

    monthly = df.groupby(["year", "month"])["Close"].agg(["first", "last"])
    monthly["ret_pct"] = (monthly["last"] / monthly["first"] - 1) * 100
    monthly = monthly.reset_index()

    years = sorted(monthly["year"].unique())
    mid = len(years) // 2
    first_half_years  = set(years[:mid])
    second_half_years = set(years[mid:])

    results = {}
    for month_num, name in MONTH_NAMES.items():
        all_rows = monthly[monthly["month"] == month_num]["ret_pct"]
        h1 = monthly[monthly["year"].isin(first_half_years)  & (monthly["month"] == month_num)]["ret_pct"]
        h2 = monthly[monthly["year"].isin(second_half_years) & (monthly["month"] == month_num)]["ret_pct"]

        if len(h1) < 2 or len(h2) < 2:
            results[name] = "insufficient"
            continue

        dir1 = "pos" if h1.mean() > 0 else "neg"
        dir2 = "pos" if h2.mean() > 0 else "neg"
        wr1  = (h1 > 0).mean()
        wr2  = (h2 > 0).mean()

        # consistent: same direction and win_rate on the same side of 50% in both
        if dir1 == dir2 and ((wr1 > 0.5 and wr2 > 0.5) or (wr1 <= 0.5 and wr2 <= 0.5)):
            results[name] = "consistent"
        else:
            results[name] = "mixed"

    return pd.Series(results)


def best_and_worst_months(stats: pd.DataFrame, n: int = 3):
    """Returns (best_n, worst_n) as separate DataFrames."""
    best = stats.sort_values("avg_pct", ascending=False).head(n)
    worst = stats.sort_values("avg_pct", ascending=True).head(n)
    return best, worst


def reliability_flag(row: "pd.Series") -> str:
    """
    Qualitative flag based on the BH-adjusted p-value (H0: avg return == 0).
    Uses the adjusted p-value to account for multiple testing across 12 months.
    """
    if row["n_obs"] < 5:
        return "very small sample (n<5)"
    p_adj = row.get("p_value_adj", np.nan)
    p_raw = row.get("p_value", np.nan)
    p = p_adj if (p_adj is not None and not np.isnan(p_adj)) else p_raw
    if p is not None and not np.isnan(p):
        if p < 0.05:
            return f"significant (p_adj={p:.3f})"
        elif p < 0.10:
            return f"marginal (p_adj={p:.3f})"
        else:
            return f"not significant (p_adj={p:.3f})"
    # fallback if p-value unavailable
    if abs(row["std_pct"]) > abs(row["avg_pct"]) * 2:
        return "high noise relative to signal"
    if row["win_rate_pct"] >= 70 or row["win_rate_pct"] <= 30:
        return "consistent pattern"
    return "weak/mixed pattern"
