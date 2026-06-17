# Screener Stagionalità + Volatilità per Opzioni

Tool da riga di comando che, per una lista di ticker, calcola:

1. **Stagionalità storica mensile** — rendimento medio, deviazione standard e win rate per ciascun mese, su tutto lo storico disponibile.
2. **Volatilità storica realizzata** e il suo percentile attuale, come proxy di quanto "costose" siano probabilmente le opzioni in questo momento rispetto al passato del titolo.
3. **Snapshot della IV reale** dalla catena opzioni live (call/put ATM sulle prossime scadenze disponibili).
4. Un **ranking finale** dei ticker analizzati, basato su un punteggio semplice che combina stagionalità attesa e "sconto" di volatilità.

## Installazione

```bash
pip install -r requirements.txt
```

Richiede Python 3.9+ e una connessione internet attiva (i dati vengono scaricati da Yahoo Finance tramite la libreria `yfinance`).

## Uso

```bash
python screener.py --tickers GLD,XRT,EQT,UNG --years 5
```

Opzioni disponibili:

- `--tickers` (obbligatorio): lista di ticker separati da virgola.
- `--years`: anni di storico da analizzare (default 5). Più anni = stima più robusta ma più lenta da scaricare.
- `--no-options`: salta il recupero della catena opzioni live (più rapido, utile se vuoi solo la stagionalità).
- `--csv nomefile.csv`: salva il ranking finale anche su file CSV.

Esempio più rapido, solo stagionalità su 10 anni:

```bash
python screener.py --tickers AAPL,MSFT,NVDA --years 10 --no-options
```

## Come leggere l'output

Per ogni ticker vedrai:

- La tabella di stagionalità completa, ordinata dal mese storicamente migliore al peggiore, con un flag di affidabilità sul mese corrente (es. "campione molto piccolo", "pattern consistente").
- La volatilità storica realizzata attuale e il suo percentile rispetto al periodo analizzato.
- Lo snapshot IV reale dalla catena opzioni (se disponibile per quel ticker).
- Un punteggio combinato, usato solo per ordinare i ticker tra loro nel ranking finale.

## Limiti metodologici — leggere prima di usare i risultati per investire

**Il punteggio non è un segnale di trading.** È un numero costruito ad-hoc per ordinare i ticker passati in input l'uno rispetto all'altro. Non ha alcuna validazione statistica di robustezza predittiva.

**La stagionalità è una tendenza storica, non una legge.** Con pochi anni di dati (es. 5), un singolo mese anomalo può distorcere pesantemente la media. Controlla sempre `n_oss` (numero di osservazioni) e la deviazione standard rispetto alla media: se la deviazione standard è più grande della media, il pattern è probabilmente rumore statistico, non un effetto reale.

**La "volatilità storica realizzata" NON è la volatilità implicita (IV).** Questo script non ha accesso a uno storico della IV (yfinance non lo fornisce gratuitamente). Usa la volatilità realizzata passata del sottostante come proxy approssimativo di quanto siano "a buon mercato" le opzioni oggi. La IV reale, quella che determina il prezzo effettivo delle opzioni, riflette le aspettative del mercato sul futuro e può discostarsi anche significativamente dalla volatilità storica, specialmente vicino a eventi noti come earnings o dati macro. Per il dato di IV percentile "vero" servirebbe una fonte dati professionale con storico IV (es. un broker come IBKR, o provider a pagamento).

**Lo snapshot della catena opzioni è solo fotografico.** Mostra la situazione al momento dell'esecuzione dello script, non uno storico. Verifica sempre bid/ask live sul tuo broker prima di piazzare un ordine — i dati di yfinance possono avere ritardi o piccole discrepanze rispetto al book reale.

**Nessun backtest di P&L.** Questo strumento fa solo screening descrittivo (stagionalità + volatilità), non simula l'esecuzione di una strategia con opzioni nel tempo, non calcola Sharpe ratio né drawdown. Se vuoi validare un'idea in modo più rigoroso, il passo successivo naturale sarebbe costruire un vero backtest su dati storici delle opzioni stesse, non solo del sottostante.

**Questo è uno strumento informativo, non consulenza finanziaria.** Va usato come punto di partenza per la tua ricerca, non come motivo sufficiente per aprire una posizione.

## Struttura dei file

```
options_screener/
├── screener.py        # script principale, CLI
├── seasonality.py      # calcolo stagionalità mensile
├── volatility.py        # volatilità realizzata + snapshot IV live
├── requirements.txt
└── README.md
```

## Possibili estensioni future

- Salvare uno snapshot giornaliero della IV per costruire, nel tempo, un vero storico IV personale (oggi impossibile per mancanza di dati storici gratuiti).
- Aggiungere un vero backtest delle strategie in opzioni (es. simulare l'acquisto di una call OTM ogni anno nello stesso mese e calcolare il P&L storico reale).
- Estendere il confronto di correlazione tra i ticker selezionati, per evitare di concentrare inconsapevolmente il rischio su un solo fattore macro (es. più ticker tutti legati al prezzo del gas).
