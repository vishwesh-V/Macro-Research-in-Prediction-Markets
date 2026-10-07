# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "duckdb", "numpy", "pandas", "pyarrow", "altair"]
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
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parent
    # Macro release series with enough settled history; download them with
    # 00_fetch_data_kalshi.py --series KXCPI,KXCPICORE,KXCPIYOY,KXPAYROLLS.
    SERIES = {
        "KXCPI": "CPI MoM",
        "KXCPICORE": "Core CPI MoM",
        "KXCPIYOY": "CPI YoY",
        "KXPAYROLLS": "Payrolls",
    }
    SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1–4, fixed order
    TTMS = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60]  # days before the release
    HORIZON_GROUPS = {
        "1–21 days (main)": [h for h in TTMS if h <= 21],
        "30 days": [30],
        "45–60 days": [45, 60],
    }
    SNAPSHOT_HOUR_ET = 16
    STALE_DAYS = 7  # a quote unchanged for longer than this is stale (robustness check)
    MAX_DAYS = 60  # time-to-release buckets cover days 1..MAX_DAYS before the release
    N_TTM_BUCKETS = 5
    TTM_COLORS = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"]  # one blue, darker = closer to release


@app.function
def bin_edges(delta):
    """Bin edges in steps of delta, with the outer bins split at 1¢ and 99¢: a mid below 1¢ or from 99¢ up
    is a contract at the tick floor (0¢ bid / 1¢ ask, or no ask), not a priced forecast."""
    return np.unique(np.round(np.concatenate([np.arange(0, 1, delta), [0.01, 0.99, 1.0]]), 6))


@app.function
def prob_bin(p, delta):
    """Lower edge of the bin [edge_k, edge_k+1) that contains each probability."""
    edges = bin_edges(delta)
    # Round first: a mid of (0.29 + 0.31) / 2 is 0.30000000000000004 in floating point.
    return edges[np.searchsorted(edges, np.round(p, 9), side="right") - 1]


@app.function
def bin_label(lower, delta):
    """'[lower, upper)' for the bin that starts at `lower`."""
    edges = bin_edges(delta)
    return f"[{lower:.2f}, {edges[np.searchsorted(edges, lower, side='right')]:.2f})"


@app.function
def reliability_table(forecasts, delta):
    """Forecasts per probability bin of width `delta`: mean forecast `f` and share settled YES."""
    return (
        forecasts.assign(bin_from=prob_bin(forecasts.f, delta))
        .groupby("bin_from", as_index=False)
        .agg(forecasts=("f", "size"), contracts=("ticker", "nunique"), releases=("event_ticker", "nunique"),
             mean_p=("f", "mean"), observed=("y", "mean"))
        .assign(bin=lambda d: [bin_label(lo, delta) for lo in d.bin_from])
        [["bin", "bin_from", "forecasts", "contracts", "releases", "mean_p", "observed"]]
    )


@app.function
def group_summary(tables, field, releases):
    """One row per `field` group of binned `tables`: forecasts, releases, weighted mean forecast and share YES,
    and the binned calibration error, the weighted mean of |share YES − mean forecast| across bins."""
    rows = []
    for name, t in tables.groupby(field, sort=False):
        w = t.forecasts
        rows.append({
            field: name, "forecasts": int(w.sum()), "releases": releases[name],
            "mean_forecast": (t.mean_p * w).sum() / w.sum(),
            "share_yes": (t.observed * w).sum() / w.sum(),
            "calibration_error": ((t.observed - t.mean_p).abs() * w).sum() / w.sum(),
        })
    return pd.DataFrame(rows)


@app.function
def reliability_chart(table, title):
    """Reliability diagram: share settled YES against the mean forecast per bin, with the 45° line."""
    scale = alt.Scale(domain=[-0.03, 1.03], nice=False)
    diagonal = alt.Chart(pd.DataFrame({"p": [0, 1]})).mark_line(color="#9a9994", strokeDash=[4, 4]).encode(
        x="p:Q", y="p:Q")
    points = alt.Chart(table).mark_circle(color="#2a78d6", opacity=0.9, stroke="#fcfcfb", strokeWidth=2).encode(
        x=alt.X("mean_p:Q", title="Predicted", axis=alt.Axis(format="%"), scale=scale),
        y=alt.Y("observed:Q", title="Observed", axis=alt.Axis(format="%"), scale=scale),
        size=alt.Size("forecasts:Q", title="Forecasts", scale=alt.Scale(range=[60, 600])),
        tooltip=[alt.Tooltip("bin:N", title="Bin"), alt.Tooltip("forecasts:Q", format=",", title="Forecasts"),
                 "contracts:Q", "releases:Q",
                 alt.Tooltip("mean_p:Q", format=".3f", title="Mean forecast"),
                 alt.Tooltip("observed:Q", format=".3f", title="Share YES")],
    )
    return (diagonal + points).properties(width=380, height=380, title=title)


@app.function
def reliability_lines_chart(table, field, domain, colors, legend_title, title):
    """Reliability diagram with one line per `field` value (`domain`, in legend order, colored by `colors`).
    Click a legend entry to highlight it (shift-click for several); the others fade."""
    scale = alt.Scale(domain=[-0.03, 1.03], nice=False)
    diagonal = alt.Chart(pd.DataFrame({"p": [0, 1]})).mark_line(color="#9a9994", strokeDash=[4, 4]).encode(
        x="p:Q", y="p:Q")
    base = alt.Chart(table).encode(
        x=alt.X("mean_p:Q", title="Predicted", axis=alt.Axis(format="%"), scale=scale),
        y=alt.Y("observed:Q", title="Observed", axis=alt.Axis(format="%"), scale=scale),
        color=alt.Color(f"{field}:N", title=legend_title, sort=domain,
                        scale=alt.Scale(domain=domain, range=colors[:len(domain)])),
    )
    pick = alt.selection_point(fields=[field], bind="legend")
    fade = alt.condition(pick, alt.value(1), alt.value(0.12))
    lines = base.mark_line(strokeWidth=2).encode(order="mean_p:Q", opacity=fade).add_params(pick)
    points = base.mark_circle(size=80, stroke="#fcfcfb", strokeWidth=2).encode(
        opacity=fade,
        tooltip=[alt.Tooltip(f"{field}:N", title=legend_title), alt.Tooltip("bin:N", title="Bin"),
                 alt.Tooltip("forecasts:Q", format=",", title="Forecasts"), "contracts:Q", "releases:Q",
                 alt.Tooltip("mean_p:Q", format=".3f", title="Mean forecast"),
                 alt.Tooltip("observed:Q", format=".3f", title="Share YES")],
    )
    return (diagonal + lines + points).properties(width=380, height=380, title=title)


@app.function
def ce_heatmap(long, row, row_title, row_order, stat_title, title):
    """Heatmap of log loss: one row per `row` value, one column per horizon, darker = higher loss (log scale).
    `long` has columns `row`, `ttm`, `value` and `n` (forecasts behind the cell)."""
    long = long.assign(horizon=long.ttm.map(lambda h: f"{h}d"))
    return alt.Chart(long).mark_rect(stroke="#fcfcfb", strokeWidth=2, cornerRadius=2).encode(
        x=alt.X("horizon:O", title="Days to release",
                sort=[f"{h}d" for h in sorted(long.ttm.unique(), reverse=True)], axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{row}:O", title=row_title, sort=row_order),
        color=alt.Color("value:Q", title=f"{stat_title} log loss",
                        scale=alt.Scale(type="log", range=["#cde2fb", "#86b6ef", "#2a78d6", "#104281", "#0d366b"])),
        tooltip=[alt.Tooltip(f"{row}:O", title=row_title), alt.Tooltip("horizon:O", title="Days to release"),
                 alt.Tooltip("value:Q", format=".3f", title=f"{stat_title} log loss"),
                 alt.Tooltip("n:Q", format=",", title="Forecasts")],
    ).properties(width="container", height=alt.Step(28), title=title)


@app.function
def equal_trade_buckets(trades_by_day, n):
    """Cut days 1..max into `n` runs of whole days holding about the same number of trades: each cut falls on the
    day whose cumulative share of trades is nearest k/n. Returns {"a–b days": (a, b)}, nearest the release first."""
    days = range(1, trades_by_day.days_to_release.max() + 1)
    counts = trades_by_day.set_index("days_to_release").trades.reindex(days, fill_value=0)
    share = counts.cumsum() / counts.sum()
    cuts = [int((share - k / n).abs().idxmin()) for k in range(1, n)]
    assert cuts == sorted(set(cuts)), f"two cuts on the same day: {cuts}"
    edges = list(zip([1] + [c + 1 for c in cuts], cuts + [max(days)]))
    return {f"{lo}–{hi} days": (lo, hi) for lo, hi in edges}


@app.cell(hide_code=True)
def intro():
    mo.md(r"""
    # 03 · Calibration: Kalshi macro release markets

    The calibration of notebook 02, applied to Kalshi's macro data-release markets: headline CPI month over month
    (`KXCPI`), core CPI month over month (`KXCPICORE`), CPI year over year (`KXCPIYOY`) and nonfarm payrolls
    (`KXPAYROLLS`). US GDP and the pre-2026 unemployment markets are not served by Kalshi's API.

    **How these markets differ from the Fed decision markets.** Each data release is one event with a *ladder* of
    yes/no contracts, "Will CPI rise more than $k$%?", for several strikes $k$. The contracts are not mutually
    exclusive: a print of 0.15% settles "above −0.1%", "above 0.0%" and "above 0.1%" all YES. Every contract closes at
    8:29 ET on the release day, a minute before the 8:30 release.

    For every settled contract $i$ and horizon $h$ (days before the release), the forecast $p_i$ is the bid/ask
    midpoint at 16:00 ET $h$ days before the release, and $Y_i \in \{0, 1\}$ is whether it settled YES. Calibration
    compares the share of YES outcomes with the average forecast per probability bin, and each forecast scores log
    loss $\ell_i = -\left[Y_i \ln p_i + (1 - Y_i) \ln(1 - p_i)\right]$ and Brier $(p_i - Y_i)^2$.

    **Forecast construction**, following notebook 02's choices where they apply:

    | Choice | Main | Robustness |
    |---|---|---|
    | Price | raw mid: ladder contracts are not exclusive, so there is nothing to rescale to sum to 1 | — |
    | Tick floor (mid below 1¢ or from 99¢ up) | excluded | included |
    | Stale quotes (unchanged for more than 7 days) | kept | dropped |
    | Wide spreads | kept | — |
    | Horizons | per group: 1–21 days, 30 days, 45–60 days before the release | — |

    **Data.** `data/kalshi_candles_1m.parquet`, `data/kalshi_trades.parquet` and `data/kalshi_markets.parquet`, queries
    in `sql/kalshi_calibration_macro.sql`.
    """)
    return


@app.cell
def data():
    con = duckdb.connect()
    con.execute((ROOT / "sql/kalshi_views.sql").read_text())
    con.execute(f"CREATE VIEW candles_raw AS SELECT * FROM read_parquet('{ROOT / 'data/kalshi_candles_1m.parquet'}')")
    con.execute(f"CREATE VIEW trades_raw AS SELECT * FROM read_parquet('{ROOT / 'data/kalshi_trades.parquet'}')")
    _missing = sorted(set(SERIES) - set(con.sql("SELECT DISTINCT series_ticker FROM all_markets").df().series_ticker))
    mo.stop(_missing, mo.md(f"**Not downloaded yet: {', '.join(_missing)}.** Run "
                            "`uv run --script 00_fetch_data_kalshi.py -- --series KXCPI,KXCPICORE,KXCPIYOY,KXPAYROLLS`."))

    # Named queries from kalshi_calibration_macro.sql, split on their "-- name: ..." headers.
    _parts = re.split(r"^-- name: (\w+)\s*$", (ROOT / "sql/kalshi_calibration_macro.sql").read_text(), flags=re.M)[1:]
    queries = dict(zip(_parts[::2], _parts[1::2]))

    forecasts = con.execute(queries["forecasts"], {"series": list(SERIES), "ttms": TTMS,
                                                   "snapshot_hour": SNAPSHOT_HOUR_ET}).df()
    forecasts["series"] = forecasts.series_ticker.map(SERIES)
    # A forecast of exactly 0 or 1 that turns out wrong has infinite log loss.
    assert forecasts.p.between(0, 1, inclusive="neither").all()
    forecasts
    return con, forecasts, queries


@app.cell(hide_code=True)
def panel_md():
    mo.md("""
    ## 1 · The forecast panel

    One row per series and horizon. `releases` counts the data releases with a quote at that horizon, `contracts`
    the ladder contracts, `share_yes` the share that settled YES (ladders put several contracts on each side of the
    print, so this sits near one half), `at_floor` the share priced below 1¢ or from 99¢ up, and `stale` the share
    of quotes unchanged for more than 7 days.
    """)
    return


@app.cell
def panel(forecasts):
    _coverage = (
        forecasts.groupby(["series", "ttm"], sort=False)
        .agg(releases=("event_ticker", "nunique"), contracts=("ticker", "size"), share_yes=("y", "mean"),
             at_floor=("at_floor", "mean"), stale=("quote_age_days", lambda a: (a > STALE_DAYS).mean()))
        .reset_index()
    )
    mo.ui.table(_coverage.round(3), selection=None, page_size=44)
    return


@app.cell(hide_code=True)
def choices_md():
    mo.md("""
    ## Analysis choices

    These settings apply to every section below. The defaults are the main specification.
    """)
    return


@app.cell
def choices():
    series_pick = mo.ui.multiselect(options={label: s for s, label in SERIES.items()}, value=list(SERIES.values()),
                                    label="Series")
    include_floor = mo.ui.checkbox(value=False, label="Include tick-floor contracts (mid below 1¢ or from 99¢ up)")
    drop_stale = mo.ui.checkbox(value=False, label=f"Drop quotes unchanged for more than {STALE_DAYS} days")
    mo.hstack([series_pick, mo.vstack([include_floor, drop_stale])], justify="start", gap=3)
    return drop_stale, include_floor, series_pick


@app.cell
def sample(drop_stale, forecasts, include_floor, series_pick):
    # The sample under the chosen settings; `f` is the forecast used from here on.
    mo.stop(not series_pick.value, mo.md("**Pick at least one series.**"))
    _keep = forecasts.series_ticker.isin(series_pick.value)
    if not include_floor.value:
        _keep &= ~forecasts.at_floor
    if drop_stale.value:
        _keep &= forecasts.quote_age_days <= STALE_DAYS
    sample = forecasts[_keep].assign(f=lambda d: d.p)
    sample["log_loss"] = -(sample.y * np.log(sample.f) + (1 - sample.y) * np.log(1 - sample.f))
    sample["brier"] = (sample.f - sample.y) ** 2
    picked = [s for s in SERIES if s in series_pick.value]
    mo.md(f"**Sample:** {len(sample):,} of {len(forecasts):,} forecasts, {sample.event_ticker.nunique()} releases "
          f"across {len(picked)} series, {int(sample.y.sum())} settled YES.")
    return picked, sample


@app.cell(hide_code=True)
def reliability_md():
    mo.md(r"""
    ## 2 · Reliability diagram

    Share of contracts that settled YES against the average forecast, per bin, for the chosen horizon group. Point
    size is the number of forecasts in the bin. Forecasts of the same release are correlated (the contracts of a
    ladder share one print, and each contract appears once per snapshot day), so a bin's effective sample is closer
    to its `releases` count than to its `forecasts` count.
    """)
    return


@app.cell
def controls():
    horizon = mo.ui.dropdown(options=HORIZON_GROUPS, value="1–21 days (main)", label="Days to release")
    bin_width = mo.ui.dropdown(options={"5¢": 0.05, "10¢": 0.10, "20¢": 0.20}, value="10¢", label="Bin width δ")
    mo.hstack([horizon, bin_width], justify="start", gap=2)
    return bin_width, horizon


@app.cell
def reliability(bin_width, horizon, sample):
    _table = reliability_table(sample[sample.ttm.isin(horizon.value)], bin_width.value)
    mo.hstack([
        reliability_chart(_table, f"Reliability · {horizon.selected_key}"),
        mo.ui.table(_table.drop(columns="bin_from").round(4), selection=None),
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def series_md():
    mo.md("""
    ### By series

    The same diagram with one line per series, for the chosen horizon group, to compare how well calibrated each
    market is. Bins default to 20¢: split by series, the middle bins hold only a few releases each. Click a series
    in the legend to highlight it.
    """)
    return


@app.cell
def lines_controls():
    lines_bin_width = mo.ui.dropdown(options={"10¢": 0.10, "20¢": 0.20, "25¢": 0.25}, value="20¢",
                                     label="Bin width δ (line charts)")
    lines_bin_width
    return (lines_bin_width,)


@app.cell
def by_series(horizon, lines_bin_width, picked, sample):
    _sel = sample[sample.ttm.isin(horizon.value)]
    _by = {SERIES[s]: _sel[_sel.series_ticker == s] for s in picked}
    _by = {name: d for name, d in _by.items() if len(d)}
    _tables = pd.concat([reliability_table(d, lines_bin_width.value).assign(series=name) for name, d in _by.items()])
    _summary = group_summary(_tables, "series", {name: d.event_ticker.nunique() for name, d in _by.items()})
    _domain = list(SERIES.values())  # every series keeps its color whatever is picked
    mo.hstack([
        reliability_lines_chart(_tables, "series", _domain, SERIES_COLORS, "Series",
                                f"Reliability by series · {horizon.selected_key}"),
        mo.ui.tabs({
            "Summary by series": mo.ui.table(_summary.round(4), selection=None),
            "Every bin": mo.ui.table(_tables[["series", "bin", "forecasts", "contracts", "releases", "mean_p",
                                              "observed"]].round(4), selection=None, page_size=25),
        }),
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def ttm_md():
    mo.md(r"""
    ### By time to release

    One line per time-to-release bucket, darker closer to the release. The buckets split days 1–60 before the
    release into five runs of whole days that each hold about a fifth of the trades in the chosen series' settled
    contracts (table below), recomputed whenever the series selection changes. Each line pools the snapshot days
    that fall in its bucket. Uses the line-chart bin width above.
    """)
    return


@app.cell
def ttm_buckets_cell(con, picked, queries):
    _trades = con.execute(queries["trades_by_day"], {"series": picked, "max_days": MAX_DAYS}).df()
    ttm_buckets = equal_trade_buckets(_trades, N_TTM_BUCKETS)
    _split = pd.DataFrame([
        {"bucket": name, "days": hi - lo + 1,
         "trades": int(_trades[_trades.days_to_release.between(lo, hi)].trades.sum())}
        for name, (lo, hi) in ttm_buckets.items()
    ]).assign(share=lambda d: d.trades / d.trades.sum(),
              snapshot_days=lambda d: [", ".join(str(h) for h in TTMS if lo <= h <= hi)
                                       for lo, hi in ttm_buckets.values()])
    mo.ui.table(_split.round(3), selection=None)
    return (ttm_buckets,)


@app.cell
def ttm_reliability(lines_bin_width, sample, ttm_buckets):
    _by = {name: sample[sample.ttm.between(lo, hi)] for name, (lo, hi) in ttm_buckets.items()}
    _by = {name: d for name, d in _by.items() if len(d)}
    _tables = pd.concat([reliability_table(d, lines_bin_width.value).assign(bucket=name) for name, d in _by.items()])
    _summary = group_summary(_tables, "bucket", {name: d.event_ticker.nunique() for name, d in _by.items()})
    mo.hstack([
        reliability_lines_chart(_tables, "bucket", list(ttm_buckets), TTM_COLORS, "Days to release",
                                "Reliability by days to release"),
        mo.ui.tabs({
            "Summary by bucket": mo.ui.table(_summary.round(4), selection=None),
            "Every bin": mo.ui.table(_tables[["bucket", "bin", "forecasts", "contracts", "releases", "mean_p",
                                              "observed"]].round(4), selection=None, page_size=25),
        }),
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def ce_md():
    mo.md("""
    ## 3 · Cross-entropy

    Log loss per series (rows) and horizon (columns), as a mean or a median, as a heatmap (darker = worse; the color
    scale is logarithmic) or a table. The mean is driven by a few surprise prints; the median ignores them.
    """)
    return


@app.cell
def ce_controls():
    ce_stat = mo.ui.dropdown(options={"Mean": "mean", "Median": "median"}, value="Mean", label="Statistic")
    ce_stat
    return (ce_stat,)


@app.cell
def ce_series(ce_stat, picked, sample):
    _stat = ce_stat.value
    _order = [SERIES[s] for s in picked]
    _long = sample.groupby(["series", "ttm"]).log_loss.agg(value=_stat, n="size").reset_index()
    _all = sample.groupby("ttm").log_loss.agg(value=_stat, n="size").reset_index().assign(series="All series")
    _heat = ce_heatmap(pd.concat([_long, _all]), "series", "Series", [*_order, "All series"],
                       ce_stat.selected_key, f"{ce_stat.selected_key} log loss by series")
    _table = sample.pivot_table(index="series", columns="ttm", values="log_loss", aggfunc=_stat).reindex(_order)
    _table.loc["All series"] = sample.groupby("ttm").log_loss.agg(_stat)
    _table.columns = [f"{h}d" for h in _table.columns]
    mo.ui.tabs({"Chart": _heat, "Table": mo.ui.table(_table.round(3).reset_index(names="series"), selection=None)})
    return


@app.cell(hide_code=True)
def ce_bin_md():
    mo.md("""
    ### By probability bin

    Log loss per probability bin (rows, highest at the top; bin width from section 2) and horizon (columns), with the
    number of forecasts behind each cell in the last tab.
    """)
    return


@app.cell
def ce_bin(bin_width, ce_stat, sample):
    _delta = bin_width.value
    _binned = sample.assign(bin_from=prob_bin(sample.f, _delta))
    _long = (_binned.groupby(["bin_from", "ttm"]).log_loss.agg(value=ce_stat.value, n="size").reset_index()
             .assign(bin=lambda d: [bin_label(lo, _delta) for lo in d.bin_from]))
    _heat = ce_heatmap(_long, "bin", "Probability bin",
                       [bin_label(lo, _delta) for lo in sorted(_long.bin_from.unique(), reverse=True)],
                       ce_stat.selected_key, f"{ce_stat.selected_key} log loss by probability bin")
    _n = _binned.pivot_table(index="bin_from", columns="ttm", values="log_loss", aggfunc="count", fill_value=0)
    _n.columns = [f"{h}d" for h in _n.columns]
    _n.index = [bin_label(lo, _delta) for lo in _n.index]
    mo.ui.tabs({"Chart": _heat, "Forecasts per cell": mo.ui.table(_n.reset_index(names="probability bin"),
                                                                  selection=None, page_size=25)})
    return


@app.cell(hide_code=True)
def release_md():
    mo.md(r"""
    ## 4 · Per-release scores

    Each release scored as a whole: the average log loss and Brier score across its ladder contracts at each horizon
    (the release is the independent observation). The chart shows each release's log loss as a dot and the mean and
    median across releases as lines; releases far above the lines are the surprise prints, labeled at their worst
    horizon. The table gives both scores per series and horizon group.
    """)
    return


@app.cell
def release_scores(forecasts, sample):
    _per = sample.groupby(["series", "event_ticker", "ttm"], as_index=False).agg(
        log_loss=("log_loss", "mean"), brier=("brier", "mean"))
    _dates = forecasts.groupby("event_ticker").release_date.first()
    _per["release"] = _per.event_ticker.map(lambda e: pd.Timestamp(_dates[e]).strftime("%b %Y"))
    _lines = (_per.groupby("ttm").log_loss.agg(Mean="mean", Median="median").reset_index()
              .melt(id_vars="ttm", var_name="statistic", value_name="log_loss"))
    _worst = _per.loc[_per.groupby("event_ticker").log_loss.idxmax()]
    _worst = _worst[_worst.log_loss >= _worst.log_loss.quantile(0.97)]

    _x = alt.X("ttm:Q", title="Days to release", scale=alt.Scale(reverse=True))
    _y = alt.Y("log_loss:Q", title="Log loss per release (lower is better)")
    _dots = alt.Chart(_per).mark_circle(size=50, color="#9a9994", opacity=0.5).encode(
        x=_x, y=_y, tooltip=["series:N", "release:N", alt.Tooltip("ttm:Q", title="Days to release"),
                             alt.Tooltip("log_loss:Q", format=".3f", title="Log loss"),
                             alt.Tooltip("brier:Q", format=".3f", title="Brier")])
    _stat_lines = alt.Chart(_lines).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
        x=_x, y=_y,
        color=alt.Color("statistic:N", title="Across releases",
                        scale=alt.Scale(domain=["Mean", "Median"], range=["#2a78d6", "#eb6834"]),
                        legend=alt.Legend(orient="top-left")),
        tooltip=["statistic:N", alt.Tooltip("ttm:Q", title="Days to release"),
                 alt.Tooltip("log_loss:Q", format=".3f", title="Log loss")])
    _labels = [
        alt.Chart(_worst[side]).mark_text(align=align, dx=dx, color="#52514e").encode(
            x=_x, y=_y, text="label:N").transform_calculate(label="datum.series + ' · ' + datum.release")
        for side, align, dx in ((_worst.ttm >= 15, "left", 8), (_worst.ttm < 15, "right", -8))
    ]
    _chart = alt.layer(_dots, _stat_lines, *_labels).properties(
        width="container", height=320, title="Log loss per release, by days to release")

    _group = {h: g for g, hs in HORIZON_GROUPS.items() for h in hs}
    _table = (_per.assign(group=_per.ttm.map(_group))
              .groupby(["series", "group"], sort=False)
              .agg(releases=("event_ticker", "nunique"), log_loss_mean=("log_loss", "mean"),
                   log_loss_median=("log_loss", "median"), brier_mean=("brier", "mean"),
                   brier_median=("brier", "median"))
              .reset_index())
    mo.vstack([_chart, mo.ui.table(_table.round(4), selection=None, page_size=20)])
    return


if __name__ == "__main__":
    app.run()
