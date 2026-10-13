"""Download 1-minute bid, ask and trade bars for CME 30-day fed funds futures from Bloomberg.

Run on a machine with a logged-in Bloomberg Terminal (the Desktop API listens on localhost:8194):

    pip install pandas
    pip install blpapi --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/
    python 00b_fetch_bloomberg_ff_intraday.py                      # as far back as Bloomberg keeps
    python 00b_fetch_bloomberg_ff_intraday.py --start 2026-05-01 --end 2026-10-12

Bloomberg keeps about 140 business days of 1-minute history, so the default start (200 days back) asks for more
than exists and takes what comes back. Days with no bars are reported at the end.

Output: data/bloomberg/ff_intraday_1m.csv.gz, one row per contract, bar type and minute in which that bar type
updated:

    contract_month  YYYY-MM whose average EFFR the contract settles on
    ticker          Bloomberg ticker used, e.g. FFV6 Comdty
    event           BID, ASK or TRADE
    time_utc        start of the 1-minute bar, UTC
    open, high, low, close, volume, num_events

Bars exist only for minutes with an update, so forward-fill `close` for a regular grid; the bid/ask mid is the
price to convert (deferred months trade thinly). Contracts run from the start date's month to 13 months after
the end date's, so every day has the 13 consecutive months FF1..FF13 that the conversion in
03_fedfunds_vs_kalshi.py uses. Refresh data/KalshiProjectFedFunds.xlsx over the same dates for the EFFR and
target range.

Contract tickers are resolved per month: live contracts use a one-digit year (FFV6), expired ones a two-digit
year (FFV26). Each candidate is checked against Bloomberg's FUT_MONTH_YR so a ticker never maps to the wrong
decade.
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import blpapi
import pandas as pd

MONTH_CODES = "FGHJKMNQUVXZ"
MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
EVENTS = ["BID", "ASK", "TRADE"]
CHUNK_DAYS = 10  # days per IntradayBarRequest; long windows can time out
MONTHS_AHEAD = 13  # FF1..FF13 on the last day
OUT = Path(__file__).resolve().parent / "data" / "bloomberg" / "ff_intraday_1m.csv.gz"
COLUMNS = ["contract_month", "ticker", "event", "time_utc", "open", "high", "low", "close", "volume", "num_events"]


def contract_months(start, end):
    """(year, month) from the start date's month to MONTHS_AHEAD - 1 months after the end date's."""
    y, m = start.year, start.month
    last = end.year * 12 + end.month - 1 + MONTHS_AHEAD - 1
    while y * 12 + m - 1 <= last:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def open_session(host, port):
    options = blpapi.SessionOptions()
    options.setServerHost(host)
    options.setServerPort(port)
    session = blpapi.Session(options)
    if not session.start():
        sys.exit(f"Could not connect to Bloomberg at {host}:{port}. Is the Terminal running and logged in?")
    if not session.openService("//blp/refdata"):
        sys.exit("Could not open //blp/refdata.")
    return session, session.getService("//blp/refdata")


def responses(session, request):
    """Every message answering `request`, up to and including the final RESPONSE event."""
    session.sendRequest(request)
    while True:
        event = session.nextEvent(10000)
        for msg in event:
            yield msg
        if event.eventType() == blpapi.Event.RESPONSE:
            return


def resolve_ticker(session, service, year, month):
    """The Bloomberg ticker of the fed funds contract for (year, month), or None."""
    code = MONTH_CODES[month - 1]
    candidates = [f"FF{code}{year % 10} Comdty", f"FF{code}{year % 100:02d} Comdty"]
    expected = f"{MONTH_ABBR[month - 1]}{year % 100:02d}"
    request = service.createRequest("ReferenceDataRequest")
    for c in candidates:
        request.append("securities", c)
    request.append("fields", "FUT_MONTH_YR")
    found = {}
    for msg in responses(session, request):
        if not msg.hasElement("securityData"):
            continue
        data = msg.getElement("securityData")
        for i in range(data.numValues()):
            sec = data.getValueAsElement(i)
            fields = sec.getElement("fieldData")
            if sec.hasElement("securityError") or not fields.hasElement("FUT_MONTH_YR"):
                continue
            found[sec.getElementAsString("security")] = fields.getElementAsString("FUT_MONTH_YR")
    for c in candidates:
        if c in found and found[c].replace(" ", "").upper() == expected:
            return c
    if found:
        print(f"  {year}-{month:02d}: no candidate matched {expected}; Bloomberg returned {found}")
    return None


def bars(session, service, ticker, event, start, end):
    """1-minute bars of one bar type for one contract between two UTC datetimes."""
    request = service.createRequest("IntradayBarRequest")
    request.set("security", ticker)
    request.set("eventType", event)
    request.set("interval", 1)
    request.set("startDateTime", start.strftime("%Y-%m-%dT%H:%M:%S"))
    request.set("endDateTime", end.strftime("%Y-%m-%dT%H:%M:%S"))
    rows = []
    for msg in responses(session, request):
        if msg.hasElement("responseError"):
            print(f"  {ticker} {event} {start:%Y-%m-%d}: {msg.getElement('responseError')}")
            continue
        if not msg.hasElement("barData"):
            continue
        ticks = msg.getElement("barData").getElement("barTickData")
        for i in range(ticks.numValues()):
            b = ticks.getValueAsElement(i)
            rows.append((b.getElementAsDatetime("time"), b.getElementAsFloat("open"), b.getElementAsFloat("high"),
                         b.getElementAsFloat("low"), b.getElementAsFloat("close"), b.getElementAsInteger("volume"),
                         b.getElementAsInteger("numEvents")))
    return rows


def main():
    today = dt.date.today()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=dt.date.fromisoformat, default=today - dt.timedelta(days=200))
    parser.add_argument("--end", type=dt.date.fromisoformat, default=today)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8194)
    args = parser.parse_args()

    session, service = open_session(args.host, args.port)
    windows = []
    day = args.start
    while day <= args.end:
        last = min(day + dt.timedelta(days=CHUNK_DAYS - 1), args.end)
        windows.append((dt.datetime.combine(day, dt.time(0, 0)), dt.datetime.combine(last, dt.time(23, 59))))
        day = last + dt.timedelta(days=1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames, missing = [], []
    for year, month in contract_months(args.start, args.end):
        ticker = resolve_ticker(session, service, year, month)
        if ticker is None:
            missing.append(f"{year}-{month:02d}")
            continue
        for event in EVENTS:
            rows = [r for start, end in windows for r in bars(session, service, ticker, event, start, end)]
            frames.append(pd.DataFrame(rows, columns=COLUMNS[3:]).assign(
                contract_month=f"{year}-{month:02d}", ticker=ticker, event=event))
            print(f"{year}-{month:02d} {ticker:<14} {event:<5} {len(rows):>8,} bars", flush=True)
        # Checkpoint after every contract, so an interrupted run keeps what it has.
        pd.concat(frames)[COLUMNS].to_csv(OUT, index=False)

    data = pd.concat(frames)[COLUMNS]
    data.to_csv(OUT, index=False)
    days = pd.to_datetime(data.time_utc).dt.date
    print(f"\nWrote {len(data):,} bars to {OUT}")
    print(f"Bars from {days.min()} to {days.max()} ({days.nunique()} days with data)")
    if missing:
        print(f"No ticker found for contract months: {', '.join(missing)}")
    session.stop()


if __name__ == "__main__":
    main()
