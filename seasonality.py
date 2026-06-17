"""
seasonality.py
Calcola statistiche di stagionalita storica (rendimento medio per mese,
win rate, deviazione standard) per un ticker, a partire dallo storico
prezzi giornaliero.
"""
import pandas as pd

MONTH_NAMES_IT = {
    1: "Gen", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mag", 6: "Giu",
    7: "Lug", 8: "Ago", 9: "Set", 10: "Ott", 11: "Nov", 12: "Dic",
}


def compute_monthly_seasonality(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    price_df: DataFrame con colonna 'Close' indicizzata da DatetimeIndex
    (es. output di yfinance.download con dati giornalieri).

    Ritorna un DataFrame indicizzato per mese (Gen..Dic) con colonne:
    media_pct, dev_std_pct, n_oss, n_positivi, win_rate_pct
    Ordinato dal mese storicamente migliore al peggiore.
    """
    df = price_df.copy()
    df["year"] = df.index.year
    df["month"] = df.index.month

    # rendimento approssimato del mese: primo close vs ultimo close del mese
    monthly = df.groupby(["year", "month"])["Close"].agg(["first", "last"])
    monthly["ret_pct"] = (monthly["last"] / monthly["first"] - 1) * 100
    monthly = monthly.reset_index()

    stats = monthly.groupby("month")["ret_pct"].agg(
        media_pct="mean",
        dev_std_pct="std",
        n_oss="count",
    )
    stats["n_positivi"] = monthly.groupby("month")["ret_pct"].apply(lambda x: (x > 0).sum())
    stats["win_rate_pct"] = (stats["n_positivi"] / stats["n_oss"] * 100).round(1)
    stats["media_pct"] = stats["media_pct"].round(2)
    stats["dev_std_pct"] = stats["dev_std_pct"].round(2)

    # mesi mancanti dal dataset (es. ticker troppo recente) vengono scartati
    stats.index = [MONTH_NAMES_IT[m] for m in stats.index]
    stats.index.name = "mese"
    return stats.sort_values("media_pct", ascending=False)


def best_and_worst_months(stats: pd.DataFrame, n: int = 3):
    """Ritorna (migliori_n, peggiori_n) come DataFrame separati."""
    best = stats.sort_values("media_pct", ascending=False).head(n)
    worst = stats.sort_values("media_pct", ascending=True).head(n)
    return best, worst


def reliability_flag(row: pd.Series) -> str:
    """
    Flag qualitativo sulla robustezza statistica del pattern stagionale
    per un singolo mese. Un campione piccolo o una deviazione standard
    superiore alla media indicano un pattern poco affidabile.
    """
    if row["n_oss"] < 5:
        return "campione molto piccolo"
    if abs(row["dev_std_pct"]) > abs(row["media_pct"]) * 2:
        return "rumore elevato rispetto al segnale"
    if row["win_rate_pct"] >= 70 or row["win_rate_pct"] <= 30:
        return "pattern consistente"
    return "pattern debole/misto"
