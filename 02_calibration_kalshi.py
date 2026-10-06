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
    CLEAN_CANDLES = ROOT / "data/cleaned/kalshi_candles_1m.parquet"
    TTMS = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60]  # days to meeting
    # Results are reported per group: every settled meeting is listed by 21 days out, 27 of 28 by
    # 30 days, and only the recent ones by 45 or 60 days.
    HORIZON_GROUPS = {
        "1–21 days (main)": [h for h in TTMS if h <= 21],
        "30 days": [30],
        "45–60 days": [45, 60],
    }
    SNAPSHOT_HOUR_ET = 16
    STALE_DAYS = 7  # a quote unchanged for longer than this is stale (robustness check)
    MINUTE_WINDOW_DAYS = 60  # minute-level reliability: days before the meeting


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
def reliability_chart(table, size, size_title, title):
    """Reliability diagram: share settled YES against the mean forecast per bin, with the 45° line."""
    scale = alt.Scale(domain=[-0.03, 1.03], nice=False)
    diagonal = alt.Chart(pd.DataFrame({"p": [0, 1]})).mark_line(color="#9a9994", strokeDash=[4, 4]).encode(
        x="p:Q", y="p:Q")
    points = alt.Chart(table).mark_circle(color="#2a78d6", opacity=0.9, stroke="#fcfcfb", strokeWidth=2).encode(
        x=alt.X("mean_p:Q", title="Predicted", axis=alt.Axis(format="%"), scale=scale),
        y=alt.Y("observed:Q", title="Observed", axis=alt.Axis(format="%"), scale=scale),
        size=alt.Size(f"{size}:Q", title=size_title, scale=alt.Scale(range=[60, 600])),
        tooltip=[alt.Tooltip("bin:N", title="Bin"),
                 alt.Tooltip(f"{size}:Q", format=",.0f", title=size_title), "contracts:Q", "meetings:Q",
                 alt.Tooltip("mean_p:Q", format=".3f", title="Mean forecast"),
                 alt.Tooltip("observed:Q", format=".3f", title="Share YES")],
    )
    return (diagonal + points).properties(width=380, height=380, title=title)


@app.cell(hide_code=True)
def intro():
    mo.md(r"""
    # 02 · Calibration: Kalshi Fed decision markets

    For every settled contract $i$ and horizon $h$ (days to the meeting), the forecast $p_i$ is the bid/ask
    midpoint at 16:00 ET $h$ days before the meeting (`SNAPSHOT_HOUR_ET` in the setup cell), and
    $Y_i \in \{0, 1\}$ is whether the contract settled YES.

    **Calibration.** A forecaster is calibrated if $\mathbb{P}(Y = 1 \mid \hat p = p) = p$. Since $p$ is
    continuous, forecasts are grouped into bins $B_k = [p_k, p_k + \delta)$, and the share of YES outcomes in
    each bin is compared with the bin's average forecast:

    $$\frac{1}{|B_k|}\sum_{i \in B_k} Y_i \;\approx\; \bar p_k = \frac{1}{|B_k|}\sum_{i \in B_k} p_i .$$

    The reliability diagram plots the first against the second; perfect calibration is the 45° line.

    **Cross-entropy.** Each forecast scores $\ell_i = -\left[Y_i \ln p_i + (1 - Y_i) \ln(1 - p_i)\right]$
    (in nats; lower is better), averaged by contract type and by probability bin at each horizon. Section 5
    scores each meeting's whole distribution instead.

    **Forecast construction.** The defaults below are the main specification; the other settings are robustness
    checks, switched in *Analysis choices*.

    | Choice | Main | Robustness |
    |---|---|---|
    | Price | $q_i = p_i / \sum_{j \in m} p_j$: the meeting's buckets rescaled to sum to 1 (they sum to ~1.015) | raw mid $p_i$ |
    | Tick floor (mid below 1¢ or from 99¢ up) | excluded from the contract-level results | included |
    | Stale quotes (unchanged for more than 7 days) | kept | dropped |
    | Wide spreads | kept: no quote is dropped for its spread | — |
    | Horizons | reported per group: 1–21 days (all meetings), 30 days, 45–60 days | — |

    **Data.** Quotes from `data/cleaned/kalshi_candles_1m.parquet`, outcomes from `data/kalshi_markets.parquet`,
    queries in `sql/kalshi_calibration.sql`. No quote is removed at the candle level, so the cleaned candles are
    an unmodified copy of the raw file; the choices above are applied to the forecasts.
    """)
    return


@app.cell
def data():
    mo.stop(
        not CLEAN_CANDLES.exists(),
        mo.md(f"**Missing `{CLEAN_CANDLES.relative_to(ROOT)}`.** Until the cleaning step exists, "
              "copy `data/kalshi_candles_1m.parquet` there."),
    )
    con = duckdb.connect()
    con.execute((ROOT / "sql/kalshi_views.sql").read_text())
    con.execute(f"CREATE VIEW candles_clean AS SELECT * FROM read_parquet('{CLEAN_CANDLES}')")

    # Named queries from kalshi_calibration.sql, split on their "-- name: ..." headers.
    _parts = re.split(r"^-- name: (\w+)\s*$", (ROOT / "sql/kalshi_calibration.sql").read_text(), flags=re.M)[1:]
    queries = dict(zip(_parts[::2], _parts[1::2]))

    forecasts = con.execute(queries["forecasts"], {"ttms": TTMS, "snapshot_hour": SNAPSHOT_HOUR_ET}).df()
    # A forecast of exactly 0 or 1 that turns out wrong has infinite log loss.
    assert forecasts.p.between(0, 1, inclusive="neither").all()
    assert forecasts.q.dropna().between(0, 1, inclusive="neither").all()
    forecasts
    return con, forecasts, queries


@app.cell(hide_code=True)
def panel_md():
    mo.md("""
    ## 1 · The forecast panel

    One row per settled contract and horizon; contracts not yet listed at a horizon drop out. Most meetings
    were listed only a few weeks ahead, so the 45- and 60-day horizons cover far fewer meetings, mostly the
    recent, more liquid ones: compare them with the shorter horizons with care, which is why results are
    reported per horizon group. `no_bid` is the share of contracts with no bid, whose mid sits at the 0.5¢
    floor; `at_floor` also counts mids from 99¢ up. `sum_p` is the median sum of the raw mids across a meeting's
    buckets, the overround that $q$ removes; `complete` is the share of meeting-snapshots with every bucket
    quoted (only those get a $q$). `stale` is the share of quotes unchanged for more than 7 days.
    """)
    return


@app.cell
def panel(forecasts):
    _group = {h: g for g, hs in HORIZON_GROUPS.items() for h in hs}
    # One row per meeting-snapshot; q is NULL for every bucket of an incomplete snapshot.
    _snapshots = forecasts.groupby(["ttm", "event_ticker"]).agg(sum_p=("sum_p", "first"),
                                                                 complete=("q", lambda q: q.notna().all()))
    _coverage = (
        forecasts.groupby("ttm")
        .agg(meetings=("event_ticker", "nunique"), contracts=("ticker", "size"), settled_yes=("y", "sum"),
             no_bid=("bid", lambda b: (b == 0).mean()), at_floor=("at_floor", "mean"),
             stale=("quote_age_days", lambda a: (a > STALE_DAYS).mean()))
        .join(_snapshots.groupby("ttm").agg(sum_p=("sum_p", "median"), complete=("complete", "mean")))
        .reset_index()
    )
    _coverage.insert(0, "group", _coverage.ttm.map(_group))
    mo.ui.table(_coverage.round(3), selection=None)
    return


@app.cell(hide_code=True)
def choices_md():
    mo.md(r"""
    ## Analysis choices

    These settings apply to sections 2–5. The defaults are the main specification; the others are the
    robustness checks. The tick-floor setting applies to the contract-level sections (2–4) only: section 5 scores
    each meeting's full distribution, so it always keeps every bucket.
    """)
    return


@app.cell
def choices():
    price = mo.ui.radio(options={"Normalized: q = p / Σp (main)": "q", "Raw mid p": "p"},
                        value="Normalized: q = p / Σp (main)", label="Price")
    include_floor = mo.ui.checkbox(value=False, label="Include tick-floor contracts (mid below 1¢ or from 99¢ up)")
    drop_stale = mo.ui.checkbox(value=False, label=f"Drop quotes unchanged for more than {STALE_DAYS} days")
    mo.hstack([price, mo.vstack([include_floor, drop_stale])], justify="start", gap=3)
    return drop_stale, include_floor, price


@app.cell
def sample(drop_stale, forecasts, include_floor, price):
    # The contract-level sample under the chosen settings; `f` is the forecast used from here on.
    _keep = forecasts[price.value].notna()
    if not include_floor.value:
        _keep &= ~forecasts.at_floor
    if drop_stale.value:
        _keep &= forecasts.quote_age_days <= STALE_DAYS
    sample = forecasts[_keep].assign(f=lambda d: d[price.value])
    sample["log_loss"] = -(sample.y * np.log(sample.f) + (1 - sample.y) * np.log(1 - sample.f))
    mo.md(f"**Sample:** {len(sample):,} of {len(forecasts):,} forecasts, "
          f"{sample.event_ticker.nunique()} meetings, {int(sample.y.sum())} settled YES.")
    return (sample,)


@app.cell(hide_code=True)
def reliability_md():
    mo.md(r"""
    ## 2 · Reliability diagram

    Share of contracts that settled YES against the average forecast, per bin. Point size is the number of
    forecasts in the bin. Forecasts of the same meeting are correlated, so a bin's effective sample is closer
    to its `meetings` count than to its `forecasts` count.

    Bins are $[p_k, p_k + \delta)$ except at the ends, which are split at 1¢ and 99¢: below 1¢ and from 99¢ up are
    contracts at the tick floor (0¢ bid / 1¢ ask, or no ask), which say little about beliefs, so they get bins of
    their own and the priced longshots and favorites sit in $[0.01, \delta)$ and $[1 - \delta, 0.99)$. In the main
    specification the floor contracts are excluded, so those two bins are empty.
    """)
    return


@app.cell
def controls():
    horizon = mo.ui.dropdown(
        options={**HORIZON_GROUPS, **{f"{h} day{'s' if h > 1 else ''}": [h] for h in TTMS}},
        value="1–21 days (main)",
        label="Days to meeting",
    )
    bin_width = mo.ui.dropdown(options={"5¢": 0.05, "10¢": 0.10, "20¢": 0.20}, value="10¢", label="Bin width δ")
    mo.hstack([horizon, bin_width], justify="start", gap=2)
    return bin_width, horizon


@app.cell
def reliability(bin_width, horizon, sample):
    _sel = sample[sample.ttm.isin(horizon.value)]
    reliability = (
        _sel.assign(bin_from=prob_bin(_sel.f, bin_width.value))
        .groupby("bin_from", as_index=False)
        .agg(forecasts=("f", "size"), contracts=("ticker", "nunique"), meetings=("event_ticker", "nunique"),
             mean_p=("f", "mean"), observed=("y", "mean"))
        .assign(bin=lambda d: [bin_label(lo, bin_width.value) for lo in d.bin_from])
        [["bin", "bin_from", "forecasts", "contracts", "meetings", "mean_p", "observed"]]
    )
    mo.hstack([
        reliability_chart(reliability, "forecasts", "Forecasts", f"Reliability · {horizon.selected_key}"),
        mo.ui.table(reliability.drop(columns="bin_from").round(4), selection=None),
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def minutes_md():
    mo.md("""
    ### Every minute, 0–60 days before the meeting (descriptive)

    **Descriptive only, not part of the tests.** This view includes the meeting day itself up to the contract's
    close, when prices converge to 0 or 1, and it uses raw mids: the analysis choices above do not apply to it.

    The same diagram from the full 1-minute data: each quote counts for as long as it was in force, from midnight ET
    60 days before the meeting until the contract closes (`MINUTE_WINDOW_DAYS` in the setup cell). Point size is
    contract-hours. Days 31–60 exist only for meetings listed that far ahead.

    More minutes are not more independent evidence: each contract still has a single outcome, so read each bin's
    `contracts` and `meetings`, not its hours. As everywhere else, quotes are kept whatever their spread. The
    checkbox drops those with a spread above 20¢: an empty book (0¢ bid, 99¢ or 100¢ ask) has a mid near 50% that
    is not a forecast, and fills the middle bins with contracts that were never in doubt.
    """)
    return


@app.cell
def minutes(con, queries):
    minute_quotes = con.execute(queries["minute_quotes"], {"days_before": MINUTE_WINDOW_DAYS}).df()
    return (minute_quotes,)


@app.cell
def minute_controls():
    drop_wide = mo.ui.checkbox(value=False, label="Drop quotes with a spread above 20¢")
    drop_wide
    return (drop_wide,)


@app.cell
def minute_reliability(bin_width, drop_wide, minute_quotes):
    _q = minute_quotes[minute_quotes.spread <= 0.20] if drop_wide.value else minute_quotes
    reliability_minutes = (
        _q.assign(bin_from=prob_bin(_q.p, bin_width.value), p_s=_q.p * _q.seconds, y_s=_q.y * _q.seconds)
        .groupby("bin_from", as_index=False)
        .agg(seconds=("seconds", "sum"), p_s=("p_s", "sum"), y_s=("y_s", "sum"),
             contracts=("ticker", "nunique"), meetings=("event_ticker", "nunique"))
        .assign(contract_hours=lambda d: d.seconds / 3600, mean_p=lambda d: d.p_s / d.seconds,
                observed=lambda d: d.y_s / d.seconds)
        .assign(bin=lambda d: [bin_label(lo, bin_width.value) for lo in d.bin_from])
        [["bin", "bin_from", "contract_hours", "contracts", "meetings", "mean_p", "observed"]]
    )
    mo.hstack([
        reliability_chart(reliability_minutes, "contract_hours", "Contract-hours",
                          f"Reliability · every minute, 0–{MINUTE_WINDOW_DAYS} days"),
        mo.ui.table(reliability_minutes.drop(columns="bin_from").round(4), selection=None),
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def ce_type_md():
    mo.md(r"""
    ## 3 · Cross-entropy by contract type

    Log loss per contract type (rows) and horizon (columns), as a mean or a median (control below). `settled YES`
    counts the meetings in which that type won. A type that rarely or never wins scores well simply by being priced
    near zero, so read each row next to that count.

    The mean is driven by a few surprises: a single meeting can double a horizon's mean. The median ignores them,
    and for types that rarely win it only returns the loss of a contract at the 0.5¢ floor, $-\ln(0.995) \approx 0.005$.
    In the main specification those floor contracts are excluded, so types that sat at the floor throughout drop out.
    The last row averages within each meeting first and then takes the mean or median across meetings, which treats
    the meeting as the independent unit.
    """)
    return


@app.cell
def ce_controls():
    ce_stat = mo.ui.dropdown(options={"Mean": "mean", "Median": "median"}, value="Mean", label="Statistic")
    ce_stat
    return (ce_stat,)


@app.cell
def ce_type(ce_stat, sample):
    _stat = ce_stat.value
    _order = sample.sort_values("move_bps").move.unique()
    _by_type = sample.pivot_table(index="move", columns="ttm", values="log_loss", aggfunc=_stat).reindex(_order)
    _by_type.loc["All contracts"] = sample.groupby("ttm").log_loss.agg(_stat)
    _by_type.loc["All contracts, per meeting"] = (
        sample.groupby(["ttm", "event_ticker"]).log_loss.mean().groupby("ttm").agg(_stat)
    )
    _by_type.columns = [f"{h}d" for h in _by_type.columns]
    _yes = sample[sample.y == 1].groupby("move").event_ticker.nunique()
    _by_type.insert(0, "settled YES", _yes.reindex(_by_type.index).fillna(0).astype(int))
    _by_type.loc[["All contracts", "All contracts, per meeting"], "settled YES"] = sample.event_ticker.nunique()
    mo.ui.table(_by_type.round(3).reset_index(names="contract type"), selection=None)
    return


@app.cell(hide_code=True)
def ce_bucket_md():
    mo.md("""
    ## 4 · Cross-entropy by probability bin

    Log loss per probability bin (rows, bin width from the control in section 2) and horizon (columns), as the mean
    or median chosen in section 3, with the number of forecasts behind each cell in the second tab. As in section 2,
    the bins below 1¢ and from 99¢ up hold the contracts at the tick floor.
    """)
    return


@app.cell
def ce_bucket(bin_width, ce_stat, sample):
    _delta = bin_width.value
    _binned = sample.assign(bin_from=prob_bin(sample.f, _delta))
    _ce = _binned.pivot_table(index="bin_from", columns="ttm", values="log_loss", aggfunc=ce_stat.value)
    _n = _binned.pivot_table(index="bin_from", columns="ttm", values="log_loss", aggfunc="count", fill_value=0)


    def _table(t, decimals):
        t = t.round(decimals)
        t.columns = [f"{h}d" for h in t.columns]
        t.index = [bin_label(lo, _delta) for lo in t.index]
        return mo.ui.table(t.reset_index(names="probability bin"), selection=None, page_size=25)


    mo.ui.tabs({f"{ce_stat.selected_key} cross-entropy": _table(_ce, 3), "Forecasts per cell": _table(_n, 0)})
    return


@app.cell(hide_code=True)
def meeting_md():
    mo.md(r"""
    ## 5 · Per-meeting scores

    Sections 2–4 score each contract on its own. Here each meeting's whole distribution is one forecast: for meeting
    $m$ at horizon $h$, with buckets $k$ and $k^*$ the bucket that settled YES,

    $$\text{log score}_{m,h} = -\ln f_{m,k^*,h}, \qquad \text{Brier}_{m,h} = \sum_k \left(f_{m,k,h} - Y_{m,k}\right)^2,$$

    both lower is better (log score in nats; Brier from 0 to 2). $f$ is the price chosen in *Analysis choices*:
    $q$, which sums to 1, or the raw mid. Every bucket is kept, floor included: a distribution cannot be scored with
    some of its buckets removed. The stale-quote setting does not apply either. Only meeting-snapshots with every
    bucket quoted are scored. This is the unit for the comparison with WIRP, and the meeting is the independent
    observation, so the `meetings` column is the sample size.
    """)
    return


@app.cell
def meeting_scores(forecasts, price):
    _f = forecasts[forecasts.q.notna()].assign(f=lambda d: d[price.value])
    meeting_scores = (
        _f.assign(sq=(_f.f - _f.y) ** 2)
        .groupby(["ttm", "event_ticker"], as_index=False).agg(brier=("sq", "sum"))
        .merge(_f[_f.y == 1].assign(log_score=lambda d: -np.log(d.f))[["ttm", "event_ticker", "log_score"]],
               on=["ttm", "event_ticker"])
    )
    _group = {h: g for g, hs in HORIZON_GROUPS.items() for h in hs}


    def _summary(grouped):
        return grouped.agg(
            meetings=("event_ticker", "nunique"),
            log_score_mean=("log_score", "mean"), log_score_median=("log_score", "median"),
            brier_mean=("brier", "mean"), brier_median=("brier", "median"),
        )


    _by_ttm = _summary(meeting_scores.groupby("ttm")).reset_index()
    _by_ttm.insert(0, "horizon", _by_ttm.pop("ttm").map(lambda h: f"{h}d"))
    _by_group = _summary(meeting_scores.assign(group=meeting_scores.ttm.map(_group)).groupby("group", sort=False))
    _by_group = _by_group.reindex(list(HORIZON_GROUPS)).reset_index(names="horizon")

    _long = _by_ttm.assign(ttm=_by_ttm.horizon.str.rstrip("d").astype(int)).melt(
        id_vars=["ttm", "meetings"], value_vars=["log_score_mean", "brier_mean"], var_name="score")
    _long["score"] = _long.score.map({"log_score_mean": "Log score", "brier_mean": "Brier"})
    _chart = alt.Chart(_long).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("ttm:Q", title="Days to meeting", scale=alt.Scale(reverse=True)),
        y=alt.Y("value:Q", title="Mean score per meeting (lower is better)"),
        color=alt.Color("score:N", scale=alt.Scale(range=["#2a78d6", "#c0302f"]), title=None),
        tooltip=[alt.Tooltip("ttm:Q", title="Days"), "score:N", alt.Tooltip("value:Q", format=".3f"),
                 alt.Tooltip("meetings:Q", title="Meetings")],
    ).properties(width="container", height=260, title="Per-meeting scores by horizon")
    mo.vstack([
        _chart,
        mo.ui.tabs({"By horizon group": mo.ui.table(_by_group.round(3), selection=None),
                    "By horizon": mo.ui.table(_by_ttm.round(3), selection=None)}),
    ])
    return (meeting_scores,)


if __name__ == "__main__":
    app.run()
