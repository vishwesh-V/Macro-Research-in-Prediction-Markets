"""Fetch Kalshi Fed funds rate contracts (series KXFED, incl. legacy FED-* events)
and build daily time series.

Each FOMC meeting is an event (e.g. KXFED-26OCT) with a ladder of binary markets
"Will the upper bound of the fed funds target be above X% after the meeting?".
Markets settled before the historical cutoff live under /historical/*, so both the
live and historical endpoints are queried.

Outputs (in ./data):
  markets.csv            one row per contract (metadata + settlement result)
  candles_daily.csv      one row per contract per day (bid/ask/last/volume/OI)
  implied_rate_daily.csv one row per meeting per day: market-implied expected
                         upper bound and probability distribution over outcomes.
                         `reliable` is False when >10% of probability sits beyond
                         the quoted strikes, or in gaps wider than 50bp between
                         usable strikes (e.g. freshly listed, unquoted ladders).

Usage: python3 fetch_kalshi_fed.py [--interval 1440] [--rebuild]
No API key needed: market data endpoints are public.
"""

import argparse
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXFED"
OUT = Path(__file__).resolve().parent / "data"
STEP = 0.25  # FOMC moves in 25bp increments

# python.org builds on macOS ship without CA certs; fall back to the system bundle.
try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context(
        cafile="/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else None)


def get(path, params=None, retries=5):
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30, context=SSL_CTX) as r:
                time.sleep(0.08)  # stay well under the public rate limit
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429 or e.code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise
        except urllib.error.URLError:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed: {url}")


def paginate(path, key, params):
    out, cursor = [], None
    while True:
        p = dict(params, limit=200)
        if cursor:
            p["cursor"] = cursor
        d = get(path, p) or {}
        out += d.get(key, [])
        cursor = d.get("cursor")
        if not cursor:
            return out


def to_ts(iso):
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def f(x):
    return None if x in (None, "") else float(x)


def ohlc(d, field):
    # Live endpoint uses *_dollars keys, historical uses bare keys.
    d = d or {}
    return f(d.get(f"{field}_dollars", d.get(field)))


def fetch_candles(m, historical, interval):
    start, end = to_ts(m["open_time"]), to_ts(m["close_time"]) + 86400
    path = (f"/historical/markets/{m['ticker']}/candlesticks" if historical
            else f"/series/{SERIES}/markets/{m['ticker']}/candlesticks")
    rows, chunk = [], 5000 * interval * 60 // 2
    for s in range(start, end, chunk):
        d = get(path, {"start_ts": s, "end_ts": min(s + chunk, end),
                       "period_interval": interval})
        for c in (d or {}).get("candlesticks", []):
            rows.append({
                "ticker": m["ticker"],
                "ts": c["end_period_ts"],
                "yes_bid": ohlc(c.get("yes_bid"), "close"),
                "yes_ask": ohlc(c.get("yes_ask"), "close"),
                "last": ohlc(c.get("price"), "close"),
                "mean_trade": ohlc(c.get("price"), "mean"),
                "volume": f(c.get("volume_fp", c.get("volume"))),
                "open_interest": f(c.get("open_interest_fp", c.get("open_interest"))),
            })
    return rows


def daily_grid(candles, markets):
    """Candles are only emitted on days with activity, so carry each contract's
    last quote forward to every calendar day up to its meeting."""
    out = []
    close = markets.set_index("ticker").meeting_date
    asof = candles.date.max()
    for t, g in candles.groupby("ticker"):
        g = g.drop_duplicates("date", keep="last").set_index("date")
        days = pd.date_range(g.index.min(), max(g.index.max(), min(close[t], asof)))
        g = g.reindex(days.strftime("%Y-%m-%d"))
        g[["yes_bid", "yes_ask", "last", "open_interest", "event_ticker", "strike"]] = \
            g[["yes_bid", "yes_ask", "last", "open_interest", "event_ticker", "strike"]].ffill()
        g["volume"] = g.volume.fillna(0)
        g["ticker"] = t
        out.append(g.rename_axis("date").reset_index())
    return pd.concat(out)


def pava_decreasing(y, w):
    """Weighted isotonic regression: best non-increasing fit to y."""
    blocks = []  # [value, weight, count]
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return [v for v, _, n in blocks for _ in range(n)]


def implied(candles, markets):
    """Per meeting per day: P(upper bound > k) ladder -> distribution + mean."""
    df = daily_grid(candles, markets).merge(markets[["ticker", "meeting_date"]], on="ticker")
    df = df[df.date <= df.meeting_date]
    # Probability estimate: bid/ask mid when the spread is <=20c; the last trade
    # clipped into the quote when the spread is <=40c; otherwise the strike is
    # uninformative and dropped.
    mid = (df.yes_bid + df.yes_ask) / 2
    spread = df.yes_ask - df.yes_bid
    last = df["last"].clip(lower=df.yes_bid, upper=df.yes_ask)
    df["p_above"] = mid.where(spread.between(0, 0.20),
                              last.where(spread.between(0, 0.40)))
    # Tighter quotes get more say when the ladder needs smoothing.
    df["w"] = 1 / ((df.yes_ask - df.yes_bid).clip(lower=0).fillna(1) + 0.02)
    df = df.dropna(subset=["p_above"])

    out = []
    for (ev, date), g in df.groupby(["event_ticker", "date"]):
        g = g.sort_values("strike")
        k = g.strike.to_numpy()
        # Enforce monotonicity: P(>k) must be non-increasing in k.
        p = pd.Series(pava_decreasing(g.p_above.to_numpy(), g.w.to_numpy())).clip(0, 1).to_numpy()
        # Outcome buckets: <=k0, (k0,k1], ..., >k_last. The upper bound sits on
        # the 25bp grid; a bucket spanning several grid points (a strike was
        # missing that day) is valued at its midpoint and labelled by its top.
        probs = {f"p_le_{k[0]:.2f}": 1 - p[0]}
        mean = k[0] * (1 - p[0])
        wide = 0.0
        for i in range(len(k)):
            hi = p[i + 1] if i + 1 < len(k) else 0.0
            top = k[i + 1] if i + 1 < len(k) else k[i] + STEP
            mass = p[i] - hi
            probs[f"p_{top:.2f}"] = mass
            mean += (k[i] + STEP + top) / 2 * mass
            if top - k[i] > 2 * STEP:
                wide += mass
        mode = max(probs, key=probs.get)
        # Mass the ladder can't place precisely. Below 0.25% is the zero lower
        # bound, so mass there is exact.
        lo_tail = 0.0 if k[0] <= 0.25 else 1 - p[0]
        hi_tail = p[-1]
        out.append({"event_ticker": ev, "meeting_date": g.meeting_date.iloc[0],
                    "date": date, "implied_upper_bound": round(mean, 4),
                    "most_likely": mode.split("_")[-1], "n_strikes": len(k),
                    "tail_mass": round(lo_tail + hi_tail, 4),
                    "gap_mass": round(wide, 4),
                    "reliable": bool(lo_tail <= 0.10 and hi_tail <= 0.10 and wide <= 0.10),
                    "volume": g.volume.sum(), "open_interest": g.open_interest.sum(),
                    "distribution": json.dumps({a: round(b, 4) for a, b in probs.items()})})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=1440, choices=[1, 60, 1440])
    ap.add_argument("--rebuild", action="store_true",
                    help="recompute implied rates from saved CSVs without fetching")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if args.rebuild:
        markets = pd.read_csv(OUT / "markets.csv")
        candles = pd.read_csv(OUT / "candles_daily.csv")
        rates = implied(candles, markets).sort_values(["meeting_date", "date"])
        rates.to_csv(OUT / "implied_rate_daily.csv", index=False)
        print(f"wrote {len(rates)} meeting-days ({rates.reliable.sum()} reliable)")
        return

    events = paginate("/events", "events", {"series_ticker": SERIES})
    print(f"{len(events)} events in {SERIES}")

    mrows, crows = [], []
    for e in sorted(events, key=lambda e: e["event_ticker"]):
        et = e["event_ticker"]
        live = paginate("/markets", "markets", {"event_ticker": et})
        hist = paginate("/historical/markets", "markets", {"event_ticker": et})
        for historical, ms in ((False, live), (True, hist)):
            for m in ms:
                strike = m.get("floor_strike")
                if strike is None:  # old markets: parse from ticker "...-T0.25"
                    strike = float(m["ticker"].rsplit("-T", 1)[-1])
                mrows.append({
                    "ticker": m["ticker"], "event_ticker": et,
                    "event_title": e["title"], "meeting_date": m["close_time"][:10],
                    "strike": float(strike), "status": m["status"],
                    "result": m.get("result"), "open_time": m["open_time"],
                    "close_time": m["close_time"], "volume": f(m.get("volume_fp")),
                    "historical": historical, "title": m.get("title"),
                })
                crows += fetch_candles(m, historical, args.interval)
        print(f"  {et}: {len(live)} live + {len(hist)} historical markets")

    markets = pd.DataFrame(mrows).sort_values(["meeting_date", "strike"])
    candles = pd.DataFrame(crows)
    candles["datetime"] = pd.to_datetime(candles.ts, unit="s", utc=True)
    # Daily candles end at midnight ET; label each by the ET day it covers.
    candles["date"] = (pd.to_datetime(candles.ts - 1, unit="s", utc=True)
                       .dt.tz_convert("America/New_York").dt.date.astype(str)
                       if args.interval == 1440 else candles.datetime.astype(str))
    candles = candles.merge(markets[["ticker", "event_ticker", "strike"]], on="ticker") \
        .sort_values(["event_ticker", "strike", "ts"])

    rates = implied(candles, markets).sort_values(["meeting_date", "date"])

    markets.to_csv(OUT / "markets.csv", index=False)
    candles.to_csv(OUT / "candles_daily.csv", index=False)
    rates.to_csv(OUT / "implied_rate_daily.csv", index=False)
    print(f"wrote {len(markets)} markets, {len(candles)} candles, "
          f"{len(rates)} meeting-days to {OUT}")


if __name__ == "__main__":
    main()
