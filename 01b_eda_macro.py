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
    SERIES = {
        "KXCPI": "CPI MoM",
        "KXCPICORE": "Core CPI MoM",
        "KXCPIYOY": "CPI YoY",
        "KXPAYROLLS": "Payrolls",
    }
    SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1–4, fixed order
    UNITS = {"KXCPI": "%", "KXCPICORE": "%", "KXCPIYOY": "%", "KXPAYROLLS": " jobs"}


@app.cell(hide_code=True)
def intro():
    mo.md(r"""
    # 01b · EDA: Kalshi macro release markets

    Kalshi's data-release markets for headline CPI month over month (`KXCPI`), core CPI month over month
    (`KXCPICORE`), CPI year over year (`KXCPIYOY`) and nonfarm payrolls (`KXPAYROLLS`), the macro counterpart of
    `01_eda_kalshi.py`. Data from `00_fetch_data_kalshi.py`, queried with DuckDB through the views and queries in
    `sql/kalshi_eda_macro.sql`.

    **The contracts are ladders.** Each data release is one event with yes/no contracts "Will CPI rise more than
    $k$%?" for several strikes $k$. They are not mutually exclusive: a 0.15% print settles every strike below 0.15%
    YES. So a ladder's prices should *fall* as the strike rises (section 4), and they do not sum to 1. Every contract
    closes at 8:29 ET on the release day. Prices are probabilities; `mid` is the end-of-day bid/ask midpoint.
    """)
    return


@app.cell
def data():
    con = duckdb.connect()
    con.execute((ROOT / "sql/kalshi_views.sql").read_text())
    # The file's views come before its first "-- name: ..." header, the named queries after.
    _parts = re.split(r"^-- name: (\w+)\s*$", (ROOT / "sql/kalshi_eda_macro.sql").read_text(), flags=re.M)
    con.execute(_parts[0])
    queries = dict(zip(_parts[1::2], _parts[2::2]))
    return con, queries


@app.cell(hide_code=True)
def coverage_md():
    mo.md("""
    ## 0 · Coverage

    What Kalshi lists per series, and how much of it is downloaded: `contracts_with_data` counts contracts with at
    least one 1-minute candle in `data/`. While `00_fetch_data_kalshi.py` is still running, the sections below show
    only the releases downloaded so far.
    """)
    return


@app.cell
def coverage(con, queries):
    _cov = con.sql(queries["coverage"]).df()
    _cov.insert(0, "series", _cov.pop("series_ticker").map(SERIES))
    mo.ui.table(_cov, selection=None)
    return


@app.cell(hide_code=True)
def releases_md():
    mo.md("""
    ## 1 · The releases

    One row per data release, newest first: the ladder's strikes and where the print landed. The print lies above
    `print_above` (the highest strike that settled YES) and at or below `print_at_most` (the lowest that settled NO);
    an empty bound means it fell outside the ladder. Volume is contracts traded across the ladder.
    """)
    return


@app.cell
def releases(con, queries):
    releases = con.sql(queries["releases"]).df()
    releases.insert(0, "series", releases.series_ticker.map(SERIES))
    _summary = (releases.groupby("series", sort=False)
                .agg(releases=("event_ticker", "size"), settled=("settled", "sum"),
                     median_strikes=("strikes", "median"), median_volume=("volume", "median"),
                     print_below_ladder=("print_above", lambda s: int(s.isna().sum())))
                .reset_index())
    mo.ui.tabs({
        "Every release": mo.ui.table(releases.drop(columns="series_ticker"), selection=None, page_size=20),
        "Summary by series": mo.ui.table(_summary, selection=None),
    })
    return (releases,)


@app.cell(hide_code=True)
def liquidity_md():
    mo.md("""
    ## 2 · Liquidity over time

    Contracts traded per month in each series (log scale), from the downloaded trades. Click a series in the legend
    to highlight it.
    """)
    return


@app.cell
def liquidity(con, queries):
    _liq = con.sql(queries["liquidity_by_month"]).df()
    _liq["series"] = _liq.series_ticker.map(SERIES)
    _liq["month"] = pd.to_datetime(_liq["month"]).dt.tz_localize(None)
    # Drop the current, incomplete month: a few days of volume would read as a collapse.
    _liq = _liq[_liq.month < pd.Timestamp.today().to_period("M").to_timestamp()]
    _domain = list(SERIES.values())
    _pick = alt.selection_point(fields=["series"], bind="legend")
    _fade = alt.condition(_pick, alt.value(1), alt.value(0.12))
    # Lines, not bars: bars on a log scale have no zero baseline to grow from.
    alt.Chart(_liq).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=48, filled=True)).encode(
        x=alt.X("yearmonth(month):T", title=None),
        y=alt.Y("contracts:Q", scale=alt.Scale(type="log"), title="Contracts traded per month (log)"),
        color=alt.Color("series:N", title="Series", sort=_domain,
                        scale=alt.Scale(domain=_domain, range=SERIES_COLORS)),
        opacity=_fade,
        tooltip=["series:N", alt.Tooltip("yearmonth(month):T", title="Month"),
                 alt.Tooltip("contracts:Q", format=",.0f", title="Contracts"),
                 alt.Tooltip("trades:Q", format=",", title="Trades"),
                 alt.Tooltip("median_trade_size:Q", format=",.0f", title="Median trade size")],
    ).add_params(_pick).properties(width="container", height=300, title="Monthly volume by series")
    return


@app.cell(hide_code=True)
def path_md():
    mo.md("""
    ## 3 · Probability path for one release

    **Top:** end-of-day mid of every contract in the ladder, one line per strike, darker for higher strikes. On any
    day the lines should be stacked with the lowest strike on top. **Bottom:** the bid–ask spread behind each mid
    (log scale). Hover a line for its strike. Releases appear once their data is downloaded. Some ladders are
    listed months ahead with an empty book (a 100¢ spread and a meaningless 50% mid) for most of that time, so the
    window defaults to the last 60 days.
    """)
    return


@app.cell
def path_controls(releases):
    series_picker = mo.ui.dropdown(options={label: s for s, label in SERIES.items()}, value="CPI MoM",
                                   label="Series")
    series_picker
    return (series_picker,)


@app.cell
def release_picker_cell(con, releases, series_picker):
    _with_data = set(con.sql("SELECT DISTINCT event_ticker FROM macro_candles").df().event_ticker)
    _rel = releases[(releases.series_ticker == series_picker.value) & releases.event_ticker.isin(_with_data)]
    mo.stop(_rel.empty, mo.md("**No release of this series is downloaded yet.**"))
    _unit = UNITS[series_picker.value]


    def _label(r):
        _date = pd.Timestamp(r.release_date).strftime("%Y-%m-%d")
        if pd.isna(r.print_above) and pd.isna(r.print_at_most):
            return f"{_date} · open"
        lo = "below ladder" if pd.isna(r.print_above) else f"> {r.print_above:g}{_unit}"
        hi = "" if pd.isna(r.print_at_most) else f", ≤ {r.print_at_most:g}{_unit}"
        return f"{_date} · print {lo}{hi}"


    _options = {_label(r): r.event_ticker for r in _rel.itertuples()}
    release_picker = mo.ui.dropdown(options=_options, value=next(iter(_options)), label="Release")
    window = mo.ui.dropdown(options={"Last 60 days": 60, "Last 30 days": 30, "Since listing": None},
                            value="Last 60 days", label="Window")
    mo.hstack([release_picker, window], justify="start", gap=2)
    return release_picker, window


@app.cell
def path(con, queries, release_picker, window):
    _path = con.execute(queries["probability_path"], {"event_ticker": release_picker.value}).df()
    if window.value is not None:
        _path = _path[_path.days_to_release <= window.value]
    _x = alt.X("date:T", title=None)
    _color = alt.Color("strike:Q", title="Strike",
                       scale=alt.Scale(range=["#86b6ef", "#2a78d6", "#104281"]))
    _tooltip = [alt.Tooltip("date:T"), alt.Tooltip("outcome:N", title="Contract"),
                alt.Tooltip("days_to_release:Q", title="Days to release"),
                alt.Tooltip("bid:Q", format=".2f"), alt.Tooltip("ask:Q", format=".2f"),
                alt.Tooltip("mid:Q", format=".3f"), alt.Tooltip("spread:Q", format=".2f"),
                alt.Tooltip("volume:Q", format=",.0f")]
    _price = alt.Chart(_path).mark_line(strokeWidth=2).encode(
        x=_x, y=alt.Y("mid:Q", title="Probability (mid)", scale=alt.Scale(domain=[0, 1])),
        color=_color, detail="ticker:N", tooltip=_tooltip,
    ).properties(width=760, height=320, title=f"Daily mid by strike · {release_picker.selected_key}")
    _spread = alt.Chart(_path).mark_line(strokeWidth=1.5).encode(
        x=_x,
        y=alt.Y("spread:Q", title="Bid–ask spread", scale=alt.Scale(type="log", domain=[0.01, 1]),
                axis=alt.Axis(values=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
                              labelExpr="round(datum.value * 100) + '¢'")),
        color=_color, detail="ticker:N", tooltip=_tooltip,
    ).properties(width=760, height=160, title="End-of-day spread, log scale")
    alt.vconcat(_price, _spread).resolve_scale(x="shared", color="shared")
    return


@app.cell(hide_code=True)
def quality_md():
    mo.md("""
    ## 4 · Quote quality

    **Left:** median end-of-day spread by price level, per series, 1–60 days before the release; contracts priced
    near 0 or 1 are usually quoted tightly at the floor. The 40–60¢ bins are dominated by empty books: a 0¢ bid and
    a 100¢ ask give a mid of exactly 50¢ and a 100¢ spread, so their median spread is not a quoted market's. **Right:** ladder
    consistency. P(above $k$) must fall as $k$ rises, so a higher strike priced above a lower one is a violation:
    the share of release-days with at least one, and with one larger than 2¢ (beyond quote noise).
    """)
    return


@app.cell
def quality(con, queries):
    _spreads = con.sql(queries["spread_by_price"]).df()
    _spreads["series"] = _spreads.series_ticker.map(SERIES)
    _domain = list(SERIES.values())
    _chart = alt.Chart(_spreads).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=48, filled=True)).encode(
        x=alt.X("price_bin:Q", title="Price level (mid, 10¢ bins)", axis=alt.Axis(format="%")),
        y=alt.Y("median_spread:Q", title="Median spread", axis=alt.Axis(format=".0%")),
        color=alt.Color("series:N", title="Series", sort=_domain,
                        scale=alt.Scale(domain=_domain, range=SERIES_COLORS)),
        tooltip=["series:N", alt.Tooltip("price_bin:Q", format=".0%", title="Bin from"),
                 alt.Tooltip("median_spread:Q", format=".3f", title="Median spread"),
                 alt.Tooltip("contract_days:Q", format=",", title="Contract-days")],
    ).properties(width=380, height=280, title="Median spread by price level")
    _viol = con.sql(queries["ladder_violations"]).df()
    _viol.insert(0, "series", _viol.pop("series_ticker").map(SERIES))
    mo.hstack([_chart, mo.vstack([mo.md("**Ladder violations**"), mo.ui.table(_viol.round(3), selection=None),
                                  mo.md("**Spreads**"), mo.ui.table(_spreads.drop(columns="series_ticker").round(3),
                                                                    selection=None, page_size=10)])],
              widths=[1, 1])
    return


if __name__ == "__main__":
    app.run()
