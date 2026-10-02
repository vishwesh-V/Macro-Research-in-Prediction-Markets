# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "duckdb", "pandas", "pyarrow", "altair"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import re
    from pathlib import Path

    import altair as alt
    import duckdb
    import marimo as mo
    import pandas as pd

    ROOT = Path(__file__).resolve().parent


@app.cell(hide_code=True)
def intro():
    mo.md(r"""
    # 01 · EDA: Kalshi Fed decision markets

    `KXFEDDECISION`: one event per FOMC meeting (May 2023 – Jan 2028), five yes/no buckets per meeting
    (cut >25 / cut 25 / hold / hike 25 / hike >25). Data from `00_fetch_data_kalshi.py`, queried with
    DuckDB through the views in `sql/kalshi_views.sql`; the queries below are in `sql/kalshi_eda.sql`.

    Prices are probabilities. `mid` is the end-of-day bid/ask midpoint, blank when the spread is wider than 20¢.
    """)
    return


@app.cell
def data():
    con = duckdb.connect()
    con.execute((ROOT / "sql/kalshi_views.sql").read_text())

    # Named queries from kalshi_eda.sql, split on their "-- name: ..." headers.
    _parts = re.split(r"^-- name: (\w+)\s*$", (ROOT / "sql/kalshi_eda.sql").read_text(), flags=re.M)[1:]
    queries = dict(zip(_parts[::2], _parts[1::2]))

    # Diverging encoding: cuts blue, hikes red, hold gray; darker toward the extremes.
    # Hike 50 / Hike >50 only exist for the May 2023 meeting.
    MOVE_COLORS = {"Cut >25": "#1d5fb0", "Cut 25": "#6aa3e8", "Hold": "#9a9994", "Hike 25": "#ee8a88",
                   "Hike >25": "#c0302f", "Hike 50": "#c0302f", "Hike >50": "#8f1d1c"}
    MOVES = list(MOVE_COLORS)


    def move_scale(moves):
        """Color scale limited to the outcomes present, in cut-to-hike order."""
        present = [m for m in MOVES if m in set(moves)]
        return alt.Scale(domain=present, range=[MOVE_COLORS[m] for m in present])
    list(queries)
    return con, move_scale, queries


@app.cell(hide_code=True)
def meetings_md():
    mo.md("""
    ## 1 · The meetings

    One row per FOMC meeting: which bucket settled YES and how many contracts traded in total.
    Only 9 of the 28 settled meetings were not a hold.
    """)
    return


@app.cell
def meetings(con):
    meetings = con.sql("SELECT * FROM meeting_summary").df()
    mo.ui.table(meetings, selection=None, page_size=40)
    return (meetings,)


@app.cell(hide_code=True)
def liquidity_md():
    mo.md("""
    ## 2 · Liquidity over time

    Contracts traded per month across all meetings (log scale). Volume jumps by roughly 10× around May 2025:
    the 2023–24 market and the 2025–26 market are very different, so results should be split at the break.
    """)
    return


@app.cell
def liquidity(con, queries):
    liquidity = con.sql(queries["liquidity_by_month"]).df()
    liquidity["month"] = pd.to_datetime(liquidity["month"]).dt.tz_localize(None)
    # Drop the current, incomplete month: a few days of volume would read as a collapse.
    liquidity = liquidity[liquidity.month < pd.Timestamp.today().to_period("M").to_timestamp()]
    # A line, not bars: bars on a log scale have no zero baseline to grow from.
    _line = alt.Chart(liquidity).mark_line(color="#2a78d6", strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True, color="#2a78d6")).encode(
        x=alt.X("yearmonth(month):T", title=None),
        y=alt.Y("contracts:Q", scale=alt.Scale(type="log"), title="Contracts traded per month (log)"),
        tooltip=[alt.Tooltip("yearmonth(month):T", title="Month"),
                 alt.Tooltip("contracts:Q", format=",.0f", title="Contracts"),
                 alt.Tooltip("trades:Q", format=",", title="Trades"),
                 alt.Tooltip("median_trade_size:Q", format=",.0f", title="Median trade size")],
    )
    _break = alt.Chart(pd.DataFrame({"month": [pd.Timestamp("2025-05-01")]})).mark_rule(strokeDash=[4, 4], color="#52514e").encode(x="month:T")
    (_line + _break).properties(width="container", height=280, title="Monthly volume, all KXFEDDECISION meetings")
    return


@app.cell(hide_code=True)
def path_md():
    mo.md("""
    ## 3 · Probability path for one meeting

    End-of-day mid per bucket. Gaps are days where the spread was wider than 20¢.
    """)
    return


@app.cell
def meeting_picker(meetings):
    meeting_picker = mo.ui.dropdown(
        options={f"{r.meeting_date} · {r.decision or 'open'}": r.event_ticker for r in meetings.itertuples()},
        value=next(f"{r.meeting_date} · {r.decision or 'open'}" for r in meetings.itertuples() if r.event_ticker == "KXFEDDECISION-26OCT"),
        label="Meeting",
    )
    meeting_picker
    return (meeting_picker,)


@app.cell
def path(con, meeting_picker, move_scale, queries):
    path = con.execute(queries["probability_path"], {"event_ticker": meeting_picker.value}).df()
    alt.Chart(path).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("mid:Q", title="Probability (mid)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("move:N", scale=move_scale(path.move), title="Outcome"),
        tooltip=[alt.Tooltip("date:T"), "move:N", alt.Tooltip("days_to_meeting:Q", title="Days to meeting"),
                 alt.Tooltip("bid:Q", format=".2f"), alt.Tooltip("ask:Q", format=".2f"), alt.Tooltip("mid:Q", format=".3f"),
                 alt.Tooltip("volume:Q", format=",.0f")],
    ).properties(width="container", height=320, title=f"Daily mid by outcome · {meeting_picker.value}")
    return


@app.cell(hide_code=True)
def quality_md():
    mo.md("""
    ## 4 · Quote quality

    **Left:** median spread per bucket in the 60 days before each meeting. **Right:** how far the buckets of a
    meeting sum from 1 on days when every bucket has a usable quote. A sum above 1 is the market's overround.
    """)
    return


@app.cell
def quality(con, queries):
    spreads = con.sql(queries["spread_by_move"]).df()
    sum_mids = con.sql(queries["sum_of_mids"]).df()
    sum_mids = sum_mids[sum_mids.buckets_quoted == sum_mids.buckets]
    _hist = alt.Chart(sum_mids).mark_bar(color="#2a78d6", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("sum_mid:Q", bin=alt.Bin(step=0.01), title="Sum of mids across buckets"),
        y=alt.Y("count():Q", title="Meeting-days"),
        tooltip=[alt.Tooltip("sum_mid:Q", bin=alt.Bin(step=0.01), title="Sum of mids"), alt.Tooltip("count():Q", title="Meeting-days")],
    ).properties(width=380, height=240, title="Sum of mids, t in [-60, -1]")
    mo.hstack([mo.ui.table(spreads, selection=None), _hist], widths=[1, 1])
    return


@app.cell(hide_code=True)
def calibration_md():
    mo.md("""
    ## 5 · First look at calibration

    Settled meetings, daily mids in the 60 days before each meeting, grouped into 10¢ bins. Points below the
    diagonal mean the outcome happened less often than priced. This is a preliminary view: days within a
    meeting are highly correlated, so the effective sample is closer to 28 meetings than to the number of
    contract-days, and the ~1.5% overround is not yet removed.
    """)
    return


@app.cell
def calibration(con, queries):
    calibration = con.sql(queries["calibration_bins"]).df()
    _diag = alt.Chart(pd.DataFrame({"p": [0, 1]})).mark_line(color="#9a9994", strokeDash=[4, 4]).encode(x="p:Q", y="p:Q")
    _pts = alt.Chart(calibration).mark_circle(color="#2a78d6", opacity=0.9, stroke="#fcfcfb", strokeWidth=2).encode(
        x=alt.X("avg_price:Q", title="Average price (implied probability)", scale=alt.Scale(domain=[-0.03, 1.03], nice=False)),
        y=alt.Y("hit_rate:Q", title="Share that settled YES", scale=alt.Scale(domain=[-0.03, 1.03], nice=False)),
        size=alt.Size("contract_days:Q", title="Contract-days", scale=alt.Scale(range=[60, 600])),
        tooltip=[alt.Tooltip("price_bin:Q", format=".1f", title="Bin from"), alt.Tooltip("avg_price:Q", format=".3f"),
                 alt.Tooltip("hit_rate:Q", format=".3f"), alt.Tooltip("contract_days:Q", format=","), "meetings:Q"],
    )
    mo.hstack([
        (_diag + _pts).properties(width=380, height=380, title="Calibration, settled meetings"),
        mo.ui.table(calibration, selection=None),
    ], widths=[1, 1])
    return


if __name__ == "__main__":
    app.run()
