"""Derive summary series from data/*.csv (run after fetch_kalshi_fed.py).

Outputs:
  data/realized_target.csv    upper bound after each settled meeting
  data/meeting_summary.csv    implied rate the day before each meeting vs outcome
  data/constant_horizon.csv   per date: implied upper bound for the next,
                              3rd and 6th upcoming meeting (+ current target);
                              only days flagged reliable are used
  data/chart_data.json        compact payload for the HTML chart
  fed_curve.html              interactive chart (template + payload)
"""
import json
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent / "data"
m = pd.read_csv(D / "markets.csv")
r = pd.read_csv(D / "implied_rate_daily.csv")

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
    row = {"date": d, "current_target": past.upper_bound.iloc[-1] if len(past) else None}
    for name, i in (("next", 0), ("third", 2), ("sixth", 5)):
        mt = upcoming[i] if i < len(upcoming) else None
        row[f"{name}_meeting"] = mt
        row[f"{name}_implied"] = lookup.get((mt, d)) if mt else None
    rows.append(row)
ch = pd.DataFrame(rows)
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
