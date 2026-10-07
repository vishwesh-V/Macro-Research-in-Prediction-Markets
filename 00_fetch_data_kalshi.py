# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "pandas", "pyarrow", "requests"]
# ///
"""Download Kalshi trades and 1-minute candles for a set of series into ./data.

Outputs:
  data/kalshi_markets.parquet  one row per contract: event, outcome bucket,
                               open/close times, status, settlement result
  data/kalshi_trades.parquet   every trade: price, size, taker side, block flag
  data/kalshi_candles_1m.parquet
                               one row per contract-minute in which the quote or
                               a trade changed: best YES bid/ask OHLC, trade
                               price OHLC, volume, open interest

Run as a script to download:
  uv run --script 00_fetch_data_kalshi.py [-- --series KXFEDDECISION,KXFED --full]
Open in marimo to inspect; the download only starts when the button is clicked:
  uvx marimo edit --sandbox 00_fetch_data_kalshi.py
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import requests

    BASE = "https://api.elections.kalshi.com/trade-api/v2"
    # Per-meeting decision buckets (cut >25 / cut 25 / hold / hike 25 / hike >25),
    # one event per FOMC meeting since May 2023. Far more liquid than the KXFED
    # rate-level ladder for the next meetings.
    SERIES = ["KXFEDDECISION"]
    # Contract descriptors attached to every trade and candle row.
    KEYS = ["event_ticker", "bucket", "outcome", "strike"]
    OUT = Path(__file__).resolve().parent / "data"
    MARKETS_FILE = OUT / "kalshi_markets.parquet"
    TRADES_FILE = OUT / "kalshi_trades.parquet"
    CANDLES_FILE = OUT / "kalshi_candles_1m.parquet"

    MAX_CANDLES = 5000  # per request, enforced by the API
    WORKERS = 4
    REQS_PER_SEC = 4  # measured: ~4.5 req/s succeed on the public tier, the rest get 429
    FLUSH_EVERY = 50  # contracts between checkpoint writes

    THREAD_LOCAL = threading.local()
    LOCK = threading.Lock()
    NEXT_SLOT = [0.0]
    RATE_LIMITED = [0]  # count of 429 responses, reported in progress lines


@app.cell
def _():
    mo.md(r"""
    # 00 · Fetch Kalshi data

    Downloads every trade and every 1-minute candle for the series below into `data/`.

    | File | Contents |
    |---|---|
    | `kalshi_markets.parquet` | One row per contract: event, outcome bucket, open/close times, status, settlement result |
    | `kalshi_trades.parquet` | Every trade: price, size, taker side, block flag |
    | `kalshi_candles_1m.parquet` | One row per contract-minute where the quote or a trade changed: best YES bid/ask OHLC, trade price OHLC, volume, open interest |

    - **The candle file is sparse:** Kalshi only emits a candle when something changed, so forward-fill the quote for a regular grid. Sizes at the best bid/ask are not available historically.
    - **Two endpoints:** markets settled before Kalshi's historical cutoff are only on `/historical/*`, and trades filled before the cutoff are only on `/historical/trades`, even for contracts that are still open. Both are queried.
    - **Incremental:** re-running skips settled contracts already stored and resumes open ones from their last timestamp. Each contract costs one request per 5,000 minutes of its life, and Kalshi's public tier allows ~4.5 requests/s: about an hour for `KXFEDDECISION`, about 3 hours for the `KXFED` ladder.
    - **Known data issue:** Kalshi's own volume for `KXFEDDECISION-25JUN-H25` is 34,000 contracts higher than the trades its API serves: the 2025-05-15 23:32 UTC candle shows 68,000 contracts against four trades totalling 34,000. All other settled contracts match within 1%.
    - **Contract labels:** `bucket` is the ticker suffix (`H25`, `C26`, `TH50`, …; codes change over time) and `outcome` is Kalshi's label (`Hike 25bps`, `Cut >25bps`, …). `strike` is only set for rate-level ladders such as `KXFED` (`-T4.00` → 4.00).
    """)
    return


@app.function
def throttle():
    with LOCK:
        now = time.monotonic()
        wait = NEXT_SLOT[0] - now
        NEXT_SLOT[0] = max(now, NEXT_SLOT[0]) + 1 / REQS_PER_SEC
    if wait > 0:
        time.sleep(wait)


@app.function
def get(path, params=None, retries=8):
    if not hasattr(THREAD_LOCAL, "s"):
        THREAD_LOCAL.s = requests.Session()
    for attempt in range(retries):
        throttle()
        try:
            r = THREAD_LOCAL.s.get(BASE + path, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            return {}
        if r.status_code == 429 or r.status_code >= 500:
            RATE_LIMITED[0] += r.status_code == 429
            time.sleep(min(2 ** attempt, 10))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"failed after {retries} tries: {path} {params}")


@app.function
def paginate(path, key, params, limit=200):
    out, cursor = [], None
    while True:
        d = get(path, {**params, "limit": limit, **({"cursor": cursor} if cursor else {})})
        out += d.get(key, [])
        cursor = d.get("cursor")
        if not cursor:
            return out


@app.function
def num(x):
    return None if x in (None, "") else float(x)


@app.function
def field(d, name):
    # Live endpoints suffix keys with _dollars / _fp; historical ones don't.
    d = d or {}
    for k in (f"{name}_dollars", f"{name}_fp", name):
        if k in d:
            return num(d[k])
    return None


@app.function
def fetch_markets(series):
    rows = []
    for s in series:
        # List contracts by series, not event by event: /events omits some events whose
        # contracts the markets endpoints still serve (e.g. FEDDECISION-24JAN).
        titles = {e["event_ticker"]: e.get("title") for e in paginate("/events", "events", {"series_ticker": s})}
        n = 0
        for historical, path in ((False, "/markets"), (True, "/historical/markets")):
            for m in paginate(path, "markets", {"series_ticker": s}, limit=1000):
                n += 1
                bucket = m["ticker"].rsplit("-", 1)[-1]
                # Threshold ladders ("above k") carry the strike in floor_strike; older ones
                # (2022 CPI) only in the ticker suffix, T<k>. Decision buckets like TH50
                # ("hike >50bps") have none.
                strike = (num(m.get("floor_strike")) if m.get("strike_type") in ("greater", "greater_or_equal")
                          else None)
                if strike is None and bucket.startswith("T"):
                    try:
                        strike = float(bucket[1:])
                    except ValueError:
                        pass
                rows.append({
                    "ticker": m["ticker"], "event_ticker": m["event_ticker"], "series_ticker": s,
                    "event_title": titles.get(m["event_ticker"]), "title": m.get("title"),
                    "bucket": bucket, "outcome": m.get("yes_sub_title"),
                    "strike_type": m.get("strike_type"), "strike": strike, "status": m.get("status"),
                    "result": m.get("result") or None,
                    "open_time": m["open_time"], "close_time": m["close_time"],
                    "volume": num(m.get("volume_fp", m.get("volume"))),
                    "historical": historical,
                })
        print(f"{s}: {len(titles)} listed events, {n} contracts")
    df = pd.DataFrame(rows).drop_duplicates("ticker", keep="last")
    df["strike"] = df.strike.astype(float)  # all-None for decision series otherwise
    for c in ("open_time", "close_time"):
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    return df.sort_values(["series_ticker", "close_time", "ticker"]).reset_index(drop=True)


@app.function
def fetch_trades(ticker, since=None):
    params = {"ticker": ticker, **({"min_ts": since} if since else {})}
    raw = []
    for path in ("/historical/trades", "/markets/trades"):
        raw += paginate(path, "trades", params, limit=1000)
    return [{
        "ticker": t["ticker"],
        "created_time": t["created_time"],
        "yes_price": num(t.get("yes_price_dollars")),
        "no_price": num(t.get("no_price_dollars")),
        "count": num(t.get("count_fp", t.get("count"))),
        "taker_side": t.get("taker_side"),
        "is_block_trade": t.get("is_block_trade"),
        "trade_id": t["trade_id"],
    } for t in raw]


@app.function
def fetch_candles(m, since=None):
    path = (f"/historical/markets/{m.ticker}/candlesticks" if m.historical
            else f"/series/{m.series_ticker}/markets/{m.ticker}/candlesticks")
    start = since or int(m.open_time.timestamp())
    end = min(int(m.close_time.timestamp()) + 3600, int(time.time()))
    step = MAX_CANDLES * 60
    rows = []
    for s in range(start, end, step):
        d = get(path, {"start_ts": s, "end_ts": min(s + step - 60, end), "period_interval": 1})
        for c in d.get("candlesticks", []):
            bid, ask, px = c.get("yes_bid"), c.get("yes_ask"), c.get("price")
            rows.append({
                "ticker": m.ticker, "ts": c["end_period_ts"],
                **{f"bid_{k}": field(bid, k) for k in ("open", "high", "low", "close")},
                **{f"ask_{k}": field(ask, k) for k in ("open", "high", "low", "close")},
                **{f"price_{k}": field(px, k) for k in ("open", "high", "low", "close", "mean")},
                "volume": field(c, "volume"),
                "open_interest": field(c, "open_interest"),
            })
    return rows


@app.function
def load(path):
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@app.function
def tidy_trades(df, markets):
    df = df.drop_duplicates("trade_id", keep="last")
    df = df.drop(columns=KEYS, errors="ignore") \
        .merge(markets[["ticker", *KEYS]], on="ticker", how="left")
    return df.sort_values(["ticker", "created_time"]).reset_index(drop=True)


@app.function
def tidy_candles(df, markets):
    df = df.drop_duplicates(["ticker", "ts"], keep="last")
    df = df.drop(columns=KEYS, errors="ignore") \
        .merge(markets[["ticker", *KEYS]], on="ticker", how="left")
    return df.sort_values(["ticker", "ts"]).reset_index(drop=True)


@app.function
def download(series, full=False):
    OUT.mkdir(exist_ok=True)
    markets = fetch_markets(series)
    # Keep contracts of other series already stored, so one file covers every run.
    old_m = pd.DataFrame() if full else load(MARKETS_FILE)
    if len(old_m):
        old_m = old_m[~old_m.series_ticker.isin(series)]
        markets = pd.concat([old_m, markets], ignore_index=True)
    markets.to_parquet(MARKETS_FILE, index=False)
    print(f"{len(markets)} contracts")

    old_t = pd.DataFrame() if full else load(TRADES_FILE)
    old_c = pd.DataFrame() if full else load(CANDLES_FILE)
    last_t = old_t.groupby("ticker").created_time.max() if len(old_t) else pd.Series(dtype=object)
    last_c = old_c.groupby("ticker").ts.max() if len(old_c) else pd.Series(dtype=object)

    # Settled contracts never change, so skip them once stored. Re-fetch open ones
    # from their last stored timestamp (minus an hour of overlap, deduped later).
    done = set(markets.ticker[markets.status.isin(["finalized", "settled"])])
    todo = [m for m in markets.itertuples()
            if not (m.ticker in done and m.ticker in last_c.index and m.ticker in last_t.index)]
    print(f"{len(markets) - len(todo)} settled contracts already stored, {len(todo)} to fetch")

    def since(last, t):
        return int(last[t].timestamp()) - 3600 if t in last.index else None

    def job(m):
        return (fetch_trades(m.ticker, since(last_t, m.ticker)),
                fetch_candles(m, since(last_c, m.ticker)))

    new_t, new_c = [], []

    def flush():
        nt, nc = pd.DataFrame(new_t), pd.DataFrame(new_c)
        if len(nt):
            nt["created_time"] = pd.to_datetime(nt.created_time, utc=True, format="ISO8601")
        if len(nc):
            nc["ts"] = pd.to_datetime(nc.ts, unit="s", utc=True)
        t = pd.concat([old_t, nt], ignore_index=True)
        c = pd.concat([old_c, nc], ignore_index=True)
        if len(t):
            tidy_trades(t, markets).to_parquet(TRADES_FILE, index=False)
        if len(c):
            tidy_candles(c, markets).to_parquet(CANDLES_FILE, index=False)

    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(job, m) for m in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            trades, candles = fut.result()
            new_t += trades
            new_c += candles
            if i % FLUSH_EVERY == 0 or i == len(todo):
                flush()
                print(f"  {i}/{len(todo)} contracts, {len(new_t):,} trades, "
                      f"{len(new_c):,} candles, {RATE_LIMITED[0]} rate-limited, "
                      f"{time.time() - t0:.0f}s", flush=True)

    trades = load(TRADES_FILE)
    print(f"wrote {len(trades):,} trades and {len(load(CANDLES_FILE)):,} 1-minute candles to {OUT}")

    # Completeness check: settled contracts' trades should sum to reported volume.
    vol = trades.groupby("ticker")["count"].sum()
    chk = markets[markets.ticker.isin(done)].set_index("ticker").volume.dropna()
    ok = (vol.reindex(chk.index).fillna(0) - chk).abs() <= 0.01 * chk.clip(lower=1)
    print(f"settled contracts whose trades match reported volume (±1%): {ok.mean():.0%} of {len(chk)}")


@app.cell
def _():
    args = mo.cli_args()
    series = mo.ui.text(value=args.get("series") or ",".join(SERIES),
                        label="Series (comma-separated)")
    full = mo.ui.checkbox(value=bool(args.get("full")), label="Ignore existing files and re-download")
    go = mo.ui.run_button(label="Run download")
    mo.hstack([series, full, go], justify="start")
    return full, go, series


@app.cell
def _(full, go, series):
    # In the editor, wait for the button; as a script, run straight away.
    mo.stop(mo.running_in_notebook() and not go.value,
            mo.md("Click **Run download** to fetch. Existing files are shown below."))
    download([s.strip() for s in series.value.split(",") if s.strip()], full=full.value)
    return


@app.cell
def _():
    import pyarrow.parquet as pq

    # Peek at the files without loading the (large) candle file into memory.
    def peek(p):
        f = pq.ParquetFile(p)
        return mo.vstack([mo.md(f"**{p.name}**: {f.metadata.num_rows:,} rows"),
                          f.read_row_group(0).to_pandas().head(5)])

    mo.vstack([peek(p) for p in (MARKETS_FILE, TRADES_FILE, CANDLES_FILE) if p.exists()])
    return


if __name__ == "__main__":
    app.run()
