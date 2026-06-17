"""
volatility.py
Calcola la volatilita storica realizzata e il suo percentile attuale
(proxy per "le opzioni sono storicamente economiche o costose adesso"),
e recupera la IV reale dalla catena opzioni live via yfinance.

NOTA IMPORTANTE SUI LIMITI DI QUESTO APPROCCIO:
yfinance non fornisce uno storico della volatilita implicita (IV).
Per stimare se "oggi" le opzioni sono relativamente economiche, questo
modulo usa la volatilita storica REALIZZATA (HV, calcolata sui rendimenti
passati del sottostante) come proxy, e calcola il percentile del valore
odierno rispetto al periodo analizzato.

Questo NON e lo stesso dell'IV percentile che si vede su un broker
(IBKR, ToS, Tastytrade, ecc.), che riflette le aspettative del mercato
sul futuro, non solo il comportamento passato del prezzo. Le due misure
sono correlate ma possono divergere, specialmente attorno a eventi noti
(earnings, dati macro). Trattalo come un'approssimazione utile per un
primo screening, non come il dato definitivo su cui basare una decisione.
"""
import numpy as np
import pandas as pd
import yfinance as yf


def realized_vol_percentile(price_df: pd.DataFrame, window: int = 20) -> dict:
    """
    Calcola la volatilita storica realizzata annualizzata su una rolling
    window di `window` giorni, e il percentile del valore piu recente
    rispetto a tutto il periodo storico disponibile nel DataFrame.
    """
    df = price_df.copy()
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    rolling_vol = log_ret.rolling(window).std() * np.sqrt(252) * 100  # annualizzata, in %

    rolling_vol = rolling_vol.dropna()
    if rolling_vol.empty:
        return {
            "hv_attuale_pct": None,
            "hv_percentile_storico": None,
            "hv_media_periodo_pct": None,
            "hv_min_periodo_pct": None,
            "hv_max_periodo_pct": None,
        }

    current_vol = rolling_vol.iloc[-1]
    percentile = (rolling_vol < current_vol).mean() * 100

    return {
        "hv_attuale_pct": round(float(current_vol), 2),
        "hv_percentile_storico": round(float(percentile), 1),
        "hv_media_periodo_pct": round(float(rolling_vol.mean()), 2),
        "hv_min_periodo_pct": round(float(rolling_vol.min()), 2),
        "hv_max_periodo_pct": round(float(rolling_vol.max()), 2),
    }


def fetch_atm_iv_snapshot(ticker: str, max_expiries: int = 2) -> list:
    """
    Recupera la IV reale di mercato per le opzioni piu vicine all'ATM
    (at-the-money) sulle prossime `max_expiries` scadenze disponibili
    sulla catena opzioni live.

    Ritorna una lista di dict con: scadenza, spot, strike ATM,
    IV call/put in %, volume call/put.

    Richiede connessione internet attiva (yfinance interroga Yahoo Finance
    in tempo reale). Se il ticker non ha opzioni quotate, ritorna lista vuota.
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

        results.append({
            "scadenza": exp,
            "spot": round(float(spot), 2),
            "strike_atm": float(atm_call["strike"]),
            "iv_call_pct": round(float(atm_call["impliedVolatility"]) * 100, 1),
            "iv_put_pct": round(float(atm_put["impliedVolatility"]) * 100, 1),
            "volume_call": int(atm_call["volume"]) if pd.notna(atm_call["volume"]) else 0,
            "volume_put": int(atm_put["volume"]) if pd.notna(atm_put["volume"]) else 0,
            "bid_call": float(atm_call["bid"]),
            "ask_call": float(atm_call["ask"]),
            "bid_put": float(atm_put["bid"]),
            "ask_put": float(atm_put["ask"]),
        })
    return results
