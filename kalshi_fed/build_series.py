"""Derive summary series from data/*.csv (run after fetch_kalshi_fed.py).

Outputs:
  data/realized_target.csv    upper bound after each settled meeting
  data/meeting_summary.csv    implied rate the day before each meeting vs outcome
  data/constant_horizon.csv   per date: implied upper bound (and implied EFFR)
                              for the next, 3rd and 6th upcoming meeting, plus
                              the target in force and actual EFFR; only days
                              flagged reliable are used
  data/fred_effr.csv          EFFR and target upper bound from FRED (cached)
  data/implied_rate_daily.csv adds effr_spread and implied_effr columns
  data/chart_data.json        compact payload for the HTML chart
  fed_curve.html              interactive chart (template + payload)
"""
import argparse
import io
import json
import os
import ssl
import urllib.request
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--refresh-fred", action="store_true", help="re-download FRED data")
args = ap.parse_args()

D = Path(__file__).resolve().parent / "data"
m = pd.read_csv(D / "markets.csv")
r = pd.read_csv(D / "implied_rate_daily.csv")


def fred(series):
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context(
            cafile="/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else None)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=2021-01-01"
    d = pd.read_csv(io.StringIO(urllib.request.urlopen(url, timeout=30, context=ctx).read().decode()))
    d.columns = ["date", series]
    d[series] = pd.to_numeric(d[series], errors="coerce")
    return d


# Implied EFFR. Kalshi settles on the target's upper bound; fed funds futures
# settle on the effective rate, which trades inside the range. Add the recent
# EFFR - upper bound gap: median of the last 20 published days, so each date
# only uses information available then.
fred_path = D / "fred_effr.csv"
if args.refresh_fred or not fred_path.exists():
    fred("EFFR").merge(fred("DFEDTARU"), on="date", how="outer").sort_values("date") \
        .to_csv(fred_path, index=False)
fr = pd.read_csv(fred_path)
fr["gap"] = fr.EFFR - fr.DFEDTARU
gap = fr.dropna(subset=["gap"]).set_index("date").gap.rolling(20, min_periods=5).median()
days = pd.date_range(fr.date.min(), max(fr.date.max(), r.date.max())).strftime("%Y-%m-%d")
gap = gap.reindex(days).ffill()
effr = fr.set_index("date").EFFR.reindex(days)
r = r.drop(columns=["effr_spread", "implied_effr"], errors="ignore")
r["effr_spread"] = r.date.map(gap).round(4)
r["implied_effr"] = (r.implied_upper_bound + r.effr_spread).round(4)
r.to_csv(D / "implied_rate_daily.csv", index=False)

settled = m[m.result.isin(["yes", "no"])]
realized = settled.groupby(["event_ticker", "meeting_date"]).apply(
    lambda g: g[g.result == "yes"].strike.max() + 0.25 if (g.result == "yes").any()
    else g.strike.min(), include_groups=False).rename("upper_bound").reset_index() \
    .sort_values("meeting_date")
realized.to_csv(D / "realized_target.csv", index=False)

pre = r[r.date < r.meeting_date].sort_values("date").groupby("event_ticker").tail(1)
summ = pre[["event_ticker", "meeting_date", "date", "implied_upper_bound", "most_likely"]] \
    .merge(realized[["event_ticker", "upper_bound"]], on="event_ticker", how="left") \
    .rename(columns={"date": "last_quote_date", "upper_bound": "realized"}) \
    .sort_values("meeting_date")
summ["error_bp"] = ((summ.implied_upper_bound - summ.realized) * 100).round(1)
summ.to_csv(D / "meeting_summary.csv", index=False)

# Constant-horizon (by meeting count) series.
meetings = sorted(r.meeting_date.unique())
lookup = r[r.reliable].set_index(["meeting_date", "date"]).implied_upper_bound
rows = []
for d in sorted(r.date.unique()):
    upcoming = [x for x in meetings if x >= d]
    past = realized[realized.meeting_date < d]
    row = {"date": d, "current_target": past.upper_bound.iloc[-1] if len(past) else None,
           "effr": effr.get(d), "effr_spread": gap.get(d)}
    for name, i in (("next", 0), ("third", 2), ("sixth", 5)):
        mt = upcoming[i] if i < len(upcoming) else None
        row[f"{name}_meeting"] = mt
        row[f"{name}_implied"] = lookup.get((mt, d)) if mt else None
        row[f"{name}_implied_effr"] = row[f"{name}_implied"] + gap.get(d) \
            if row[f"{name}_implied"] is not None and pd.notna(gap.get(d)) else None
    rows.append(row)
ch = pd.DataFrame(rows)

# Hair chart: on the first trading day of each month, the implied path through
# every upcoming meeting that's reliably priced, starting from the target in force.
hairs = []
for d in ch.groupby(ch.date.str[:7]).date.min():
    snap = r[(r.date == d) & r.reliable & (r.meeting_date >= d)].sort_values("meeting_date")
    tgt = ch.loc[ch.date == d, "current_target"].iloc[0]
    if len(snap) and pd.notna(tgt):
        hairs.append({"d": d, "pts": [[d, float(tgt)]] +
                      [[a, round(b, 4)] for a, b in zip(snap.meeting_date, snap.implied_upper_bound)]})
ch.to_csv(D / "constant_horizon.csv", index=False)

def on_grid(d):
    """Spread each bucket's mass evenly over the 25bp outcomes it spans, so
    days with missing strikes share one set of outcomes. Mass at or below the
    lowest strike sits on that strike."""
    out, prev = {}, None
    for k, v in d.items():
        x = float(k.split("_")[-1])
        if k.startswith("p_le"):
            pts = [x]
        else:
            n = max(1, round((x - prev) / 0.25))
            pts = [round(prev + 0.25 * (i + 1), 2) for i in range(n)]
        for q in pts:
            out[q] = out.get(q, 0) + v / len(pts)
        prev = x
    return out


# Chart payload.
meet = {}
for ev, g in r.sort_values("date").groupby("event_ticker"):
    dist = [on_grid(json.loads(x)) for x in g.distribution]
    grid = sorted({k for d in dist for k in d})
    keys = [f"p_{k:.2f}" for k in grid]
    dist = [{f"p_{k:.2f}": v for k, v in d.items()} for d in dist]
    meet[ev] = {
        "meeting": g.meeting_date.iloc[0],
        "title": ev,
        "d": g.date.tolist(),
        "mu": g.implied_upper_bound.round(4).tolist(),
        "vol": g.volume.round(0).tolist(),
        "rel": [int(x) for x in g.reliable],
        "keys": keys,
        "p": [[round(x.get(k, 0), 3) for k in keys] for x in dist],
    }
real = realized.merge(summ[["event_ticker", "implied_upper_bound", "error_bp", "last_quote_date"]],
                      on="event_ticker", how="left")
payload = {
    "asof": r.date.max(),
    "n_contracts": len(m),
    "meetings": meet,
    "realized": real.astype(object).where(real.notna(), None).to_dict("records"),
    "hairs": hairs,
    "effr_gap": round(float(gap.get(r.date.max())), 4),
    "horizon": {c: ch[c].astype(object).where(ch[c].notna(), None).tolist() for c in
                ["date", "current_target", "next_implied", "third_implied", "sixth_implied",
                 "next_meeting", "third_meeting", "sixth_meeting"]},
}
(D / "chart_data.json").write_text(json.dumps(payload, separators=(",", ":")))

# Self-contained chart page.
here = Path(__file__).resolve().parent
html = (here / "chart_template.html").read_text().replace(
    "/*DATA*/null", json.dumps(payload, separators=(",", ":")))
(here / "fed_curve.html").write_text(html)
print(summ.tail(3).to_string(), "\n", ch.tail(3).to_string())
print("payload KB", (D / "chart_data.json").stat().st_size // 1024)
