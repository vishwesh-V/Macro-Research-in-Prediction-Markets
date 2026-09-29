# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "pandas", "altair", "requests"]
# ///
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import time
    from datetime import datetime

    import altair as alt
    import marimo as mo
    import pandas as pd
    import requests

    BASE = "https://api.elections.kalshi.com/trade-api/v2"
    SERIES = "KXFEDDECISION"
    return BASE, SERIES, alt, datetime, mo, pd, requests, time


@app.cell
def _(mo):
    mo.md("""
    # Kalshi Fed decision markets

    Daily prices for each outcome (cut / hold / hike) of an FOMC meeting,
    from Kalshi's public API (no key needed). A contract price of $0.45 ≈ 45% implied probability.
    """)
    return


@app.cell
def _(BASE, SERIES, pd, requests):
    def get(path, **params):
        r = requests.get(f"{BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    events = get("/events", series_ticker=SERIES, limit=200)["events"]
    events = sorted(events, key=lambda e: e["strike_date"])
    event_options = {f"{e['title']} ({e['event_ticker']})": e["event_ticker"] for e in events}
    # Default to the next upcoming meeting
    next_event = next(e for e in events if pd.Timestamp(e["strike_date"]) > pd.Timestamp.now(tz="UTC"))
    default_label = f"{next_event['title']} ({next_event['event_ticker']})"
    return default_label, event_options, get


@app.cell
def _(default_label, event_options, mo):
    event_picker = mo.ui.dropdown(options=event_options, value=default_label, label="FOMC meeting")
    event_picker
    return (event_picker,)


@app.cell
def _(SERIES, datetime, event_picker, get, mo, pd, time):
    def to_float(x):
        return float(x) if x not in (None, "") else None

    def candles(ticker, start_ts, end_ts):
        data = get(
            f"/series/{SERIES}/markets/{ticker}/candlesticks",
            start_ts=start_ts,
            end_ts=end_ts,
            period_interval=1440,  # daily
        )
        rows = []
        for c in data["candlesticks"]:
            close = to_float(c["price"].get("close_dollars"))
            bid = to_float(c["yes_bid"].get("close_dollars"))
            ask = to_float(c["yes_ask"].get("close_dollars"))
            mid = (bid + ask) / 2 if bid is not None and ask is not None else None
            rows.append(
                {
                    "date": pd.to_datetime(c["end_period_ts"], unit="s"),
                    "last_trade": close,
                    "mid": mid,
                    "volume": to_float(c["volume_fp"]),
                }
            )
        return pd.DataFrame(rows)

    markets = get("/markets", event_ticker=event_picker.value)["markets"]
    # Older meetings (2023-24) return no markets from the live API; likely archived elsewhere
    mo.stop(not markets, mo.md(f"**No markets returned for `{event_picker.value}`.** Pick a more recent meeting."))
    now = int(time.time())
    frames = []
    for m in markets:
        start = int(datetime.fromisoformat(m["open_time"].replace("Z", "+00:00")).timestamp())
        end = min(now, int(datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp()))
        df_m = candles(m["ticker"], start, end)
        df_m["outcome"] = m["yes_sub_title"]
        df_m["ticker"] = m["ticker"]
        frames.append(df_m)

    prices = pd.concat(frames, ignore_index=True)
    # Prefer bid/ask midpoint (always quoted); fall back to last trade
    prices["prob"] = prices["mid"].fillna(prices["last_trade"])
    prices
    return (prices,)


@app.cell
def _(alt, event_picker, prices):
    alt.Chart(prices.dropna(subset=["prob"])).mark_line().encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("prob:Q", title="Implied probability", axis=alt.Axis(format="%")),
        color=alt.Color("outcome:N", title="Outcome"),
        tooltip=["date:T", "outcome:N", alt.Tooltip("prob:Q", format=".0%"), "volume:Q"],
    ).properties(title=f"Kalshi: {event_picker.value}", width="container", height=380)
    return


if __name__ == "__main__":
    app.run()
