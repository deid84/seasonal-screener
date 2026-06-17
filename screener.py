"""
screener.py
Tool da riga di comando per:
  1. scaricare lo storico prezzi di una lista di ticker (yfinance)
  2. calcolare la stagionalita mensile storica per ciascuno
  3. stimare se la volatilita (e quindi il costo delle opzioni) e
     attualmente bassa o alta rispetto al proprio storico (proxy HV)
  4. recuperare uno snapshot della IV reale dalla catena opzioni live
  5. produrre un ranking dei ticker per "opportunita stagionale +
     costo opzioni contenuto"

USO:
    python screener.py --tickers GLD,XRT,EQT,UNG --years 5
    python screener.py --tickers AAPL --years 10 --no-options
    python screener.py --tickers GLD,XRT --years 5 --csv ranking.csv

Setup:
    pip install -r requirements.txt

IMPORTANTE: questo script e uno strumento di screening informativo,
non genera segnali di trading e non costituisce consulenza finanziaria.
Vedi README.md per i limiti metodologici (in particolare sulla stima
della volatilita, vedi volatility.py).
"""
import argparse
import sys
from datetime import datetime

import pandas as pd
import yfinance as yf

from seasonality import compute_monthly_seasonality, MONTH_NAMES_IT, reliability_flag
from volatility import realized_vol_percentile, fetch_atm_iv_snapshot


def fetch_history(ticker: str, years: int) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Nessun dato trovato per {ticker}")
    # yfinance puo restituire colonne MultiIndex anche per un singolo ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def analyze_ticker(ticker: str, years: int, fetch_options: bool = True) -> dict:
    current_month = datetime.now().month
    next_month = current_month % 12 + 1

    df = fetch_history(ticker, years)
    seas = compute_monthly_seasonality(df)
    vol = realized_vol_percentile(df)

    seas_now = seas.loc[MONTH_NAMES_IT[current_month]] if MONTH_NAMES_IT[current_month] in seas.index else None
    seas_next = seas.loc[MONTH_NAMES_IT[next_month]] if MONTH_NAMES_IT[next_month] in seas.index else None

    report = {
        "ticker": ticker,
        "anni_richiesti": years,
        "n_anni_dati_effettivi": int(df.index.year.nunique()),
        "mese_corrente": MONTH_NAMES_IT[current_month],
        "mese_prossimo": MONTH_NAMES_IT[next_month],
        "stagionalita_mese_corrente": seas_now.to_dict() if seas_now is not None else None,
        "stagionalita_mese_prossimo": seas_next.to_dict() if seas_next is not None else None,
        "affidabilita_mese_corrente": reliability_flag(seas_now) if seas_now is not None else None,
        "tabella_completa": seas,
        "volatilita": vol,
    }

    if fetch_options:
        try:
            report["opzioni_snapshot"] = fetch_atm_iv_snapshot(ticker)
        except Exception as e:
            report["opzioni_snapshot"] = f"non disponibile ({e})"

    return report


def score_opportunity(report: dict) -> float:
    """
    Punteggio semplice ed esplicito, SOLO per ranking relativo tra i
    ticker passati in input. Non e un segnale di acquisto/vendita.

    Combina:
      - stagionalita attesa (mese corrente + prossimo, pesata per win rate)
      - "sconto" sulla volatilita attuale rispetto al proprio storico
        (percentile HV basso = punteggio piu alto = opzioni probabilmente
        piu economiche del solito)
    """
    seas_now = report["stagionalita_mese_corrente"]
    seas_next = report["stagionalita_mese_prossimo"]
    vol_pct = report["volatilita"]["hv_percentile_storico"]

    seas_score = 0.0
    for s in (seas_now, seas_next):
        if s:
            seas_score += s["media_pct"] * (s["win_rate_pct"] / 100)

    if vol_pct is None:
        cheapness_score = 0.0
    else:
        cheapness_score = (100 - vol_pct) / 10  # 0 (vol alta) - 10 (vol ai minimi)

    return round(seas_score + cheapness_score, 2)


def print_report(report: dict):
    print("=" * 72)
    print(f"  {report['ticker']}  —  {report['n_anni_dati_effettivi']} anni di dati effettivi")
    print("=" * 72)

    print(f"\nMese corrente ({report['mese_corrente']}):")
    s = report["stagionalita_mese_corrente"]
    if s:
        print(f"  rendimento medio storico: {s['media_pct']:+.2f}%  |  win rate: {s['win_rate_pct']}%  "
              f"|  osservazioni: {int(s['n_oss'])}  |  {report['affidabilita_mese_corrente']}")
    else:
        print("  dati insufficienti per questo mese")

    s2 = report["stagionalita_mese_prossimo"]
    print(f"\nMese prossimo ({report['mese_prossimo']}):")
    if s2:
        print(f"  rendimento medio storico: {s2['media_pct']:+.2f}%  |  win rate: {s2['win_rate_pct']}%  "
              f"|  osservazioni: {int(s2['n_oss'])}")
    else:
        print("  dati insufficienti per questo mese")

    print("\nStagionalita completa (dal mese storicamente migliore al peggiore):")
    print(report["tabella_completa"].to_string())

    print("\nVolatilita storica realizzata (proxy costo opzioni — vedi limiti in README.md):")
    v = report["volatilita"]
    if v["hv_percentile_storico"] is not None:
        print(f"  HV attuale: {v['hv_attuale_pct']}%  |  percentile storico: {v['hv_percentile_storico']}°"
              f"  (range periodo: {v['hv_min_periodo_pct']}% - {v['hv_max_periodo_pct']}%)")
        if v["hv_percentile_storico"] < 25:
            print("  -> volatilita storicamente BASSA: opzioni probabilmente piu economiche del solito")
        elif v["hv_percentile_storico"] > 75:
            print("  -> volatilita storicamente ALTA: opzioni probabilmente piu costose del solito")
        else:
            print("  -> volatilita nella norma rispetto al proprio storico")
    else:
        print("  dati insufficienti")

    if "opzioni_snapshot" in report:
        print("\nSnapshot IV reale dalla catena opzioni live (dati di mercato attuali):")
        snap = report["opzioni_snapshot"]
        if isinstance(snap, list) and snap:
            for s3 in snap:
                print(f"  scadenza {s3['scadenza']}: spot {s3['spot']}, strike ATM {s3['strike_atm']}  |  "
                      f"IV call {s3['iv_call_pct']}% (bid {s3['bid_call']}/ask {s3['ask_call']})  |  "
                      f"IV put {s3['iv_put_pct']}% (bid {s3['bid_put']}/ask {s3['ask_put']})")
        elif isinstance(snap, list):
            print("  nessuna opzione quotata trovata per questo ticker")
        else:
            print(f"  {snap}")

    print(f"\nPunteggio combinato per ranking (stagionalita + sconto volatilita): {score_opportunity(report)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Screener stagionalita + volatilita per opzioni")
    parser.add_argument("--tickers", required=True, help="Lista ticker separati da virgola, es: GLD,XRT,EQT,UNG")
    parser.add_argument("--years", type=int, default=5, help="Anni di storico da analizzare (default 5)")
    parser.add_argument("--no-options", action="store_true", help="Salta il recupero della catena opzioni live")
    parser.add_argument("--csv", help="Salva il ranking finale in un file CSV")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    reports = []

    for ticker in tickers:
        print(f"Scaricando dati per {ticker}...")
        try:
            report = analyze_ticker(ticker, args.years, fetch_options=not args.no_options)
            reports.append(report)
        except Exception as e:
            print(f"  ERRORE su {ticker}: {e}", file=sys.stderr)

    if not reports:
        print("Nessun dato analizzabile. Controlla i ticker e la connessione internet.", file=sys.stderr)
        sys.exit(1)

    print("\n\n" + "#" * 72)
    print("#  REPORT DETTAGLIATO PER TICKER")
    print("#" * 72 + "\n")
    for r in reports:
        print_report(r)

    print("#" * 72)
    print("#  RANKING FINALE (punteggio combinato, decrescente)")
    print("#" * 72)
    ranking = sorted(reports, key=score_opportunity, reverse=True)
    rows = []
    for r in ranking:
        rows.append({
            "ticker": r["ticker"],
            "punteggio": score_opportunity(r),
            "hv_percentile": r["volatilita"]["hv_percentile_storico"],
            "stagionalita_mese_corrente_pct": r["stagionalita_mese_corrente"]["media_pct"] if r["stagionalita_mese_corrente"] else None,
            "stagionalita_mese_prossimo_pct": r["stagionalita_mese_prossimo"]["media_pct"] if r["stagionalita_mese_prossimo"] else None,
        })
    ranking_df = pd.DataFrame(rows)
    print(ranking_df.to_string(index=False))

    if args.csv:
        ranking_df.to_csv(args.csv, index=False)
        print(f"\nRanking salvato in {args.csv}")


if __name__ == "__main__":
    main()
