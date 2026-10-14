# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "duckdb", "numpy", "pandas", "pyarrow", "altair"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math
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
    # Time-to-maturity buckets for the reliability-by-horizon diagrams: days 1–60 before the meeting cut into this
    # many runs of whole days with about the same number of trades each (cut in the `ttm_buckets` cell).
    N_TTM_BUCKETS = 5
    TTM_COLORS = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"]  # one blue, darker = closer to the meeting
    SNAPSHOT_HOUR_ET = 16
    STALE_DAYS = 7  # a quote unchanged for longer than this is stale (robustness check)
    MINUTE_WINDOW_DAYS = 60  # minute-level reliability: days before the meeting
    # Kalshi's liquidity break: the first meeting after it (May 2025) traded about 15 times the contracts of the
    # one before (March 2025), and volume stayed there. Regimes are split by meeting date.
    LIQUIDITY_BREAK = pd.Timestamp("2025-05-01")
    REGIMES = ["Before May 2025", "From May 2025"]


@app.function(hide_code=True)
def bin_edges(delta):
    """Bin edges in steps of delta, with the outer bins split at 1¢ and 99¢: a mid below 1¢ or from 99¢ up
    is a contract at the tick floor (0¢ bid / 1¢ ask, or no ask), not a priced forecast."""
    return np.unique(np.round(np.concatenate([np.arange(0, 1, delta), [0.01, 0.99, 1.0]]), 6))


@app.function(hide_code=True)
def prob_bin(p, delta):
    """Lower edge of the bin [edge_k, edge_k+1) that contains each probability."""
    edges = bin_edges(delta)
    # Round first: a mid of (0.29 + 0.31) / 2 is 0.30000000000000004 in floating point.
    return edges[np.searchsorted(edges, np.round(p, 9), side="right") - 1]


@app.function(hide_code=True)
def bin_label(lower, delta):
    """'[lower, upper)' for the bin that starts at `lower`."""
    edges = bin_edges(delta)
    return f"[{lower:.2f}, {edges[np.searchsorted(edges, lower, side='right')]:.2f})"


@app.function(hide_code=True)
def reliability_table(forecasts, delta):
    """Forecasts per probability bin of width `delta`: mean forecast `f` and share settled YES."""
    return (
        forecasts.assign(bin_from=prob_bin(forecasts.f, delta))
        .groupby("bin_from", as_index=False)
        .agg(forecasts=("f", "size"), contracts=("ticker", "nunique"), meetings=("event_ticker", "nunique"),
             mean_p=("f", "mean"), observed=("y", "mean"))
        .assign(bin=lambda d: [bin_label(lo, delta) for lo in d.bin_from])
        [["bin", "bin_from", "forecasts", "contracts", "meetings", "mean_p", "observed"]]
    )


@app.function(hide_code=True)
def minute_reliability_table(quotes, delta):
    """Time-weighted bins of width `delta`: each quote counts for the seconds it was in force."""
    return (
        quotes.assign(bin_from=prob_bin(quotes.p, delta), p_s=quotes.p * quotes.seconds, y_s=quotes.y * quotes.seconds)
        .groupby("bin_from", as_index=False)
        .agg(seconds=("seconds", "sum"), p_s=("p_s", "sum"), y_s=("y_s", "sum"),
             contracts=("ticker", "nunique"), meetings=("event_ticker", "nunique"))
        .assign(contract_hours=lambda d: d.seconds / 3600, mean_p=lambda d: d.p_s / d.seconds,
                observed=lambda d: d.y_s / d.seconds)
        .assign(bin=lambda d: [bin_label(lo, delta) for lo in d.bin_from])
        [["bin", "bin_from", "contract_hours", "contracts", "meetings", "mean_p", "observed"]]
    )


@app.function(hide_code=True)
def bucket_summary(tables, weight, meetings):
    """One row per time-to-maturity bucket of binned `tables`: total `weight`, meetings, weighted mean forecast and
    share YES, and the binned calibration error, the weighted mean of |share YES − mean forecast| across bins."""
    rows = []
    for name, t in tables.groupby("bucket", sort=False):
        w = t[weight]
        rows.append({
            "bucket": name, weight: w.sum(), "meetings": meetings[name],
            "mean_forecast": (t.mean_p * w).sum() / w.sum(),
            "share_yes": (t.observed * w).sum() / w.sum(),
            "calibration_error": ((t.observed - t.mean_p).abs() * w).sum() / w.sum(),
        })
    return pd.DataFrame(rows)


@app.function(hide_code=True)
def reliability_by_bucket_chart(table, buckets, size, size_title, title):
    """Reliability diagram with one line per time-to-maturity bucket (`buckets`, nearest the meeting first),
    darker = closer to the meeting."""
    scale = alt.Scale(domain=[-0.03, 1.03], nice=False)
    diagonal = alt.Chart(pd.DataFrame({"p": [0, 1]})).mark_line(color="#9a9994", strokeDash=[4, 4]).encode(
        x="p:Q", y="p:Q")
    base = alt.Chart(table).encode(
        x=alt.X("mean_p:Q", title="Predicted", axis=alt.Axis(format="%"), scale=scale),
        y=alt.Y("observed:Q", title="Observed", axis=alt.Axis(format="%"), scale=scale),
        color=alt.Color("bucket:N", title="Days to meeting", sort=buckets,
                        scale=alt.Scale(domain=buckets, range=TTM_COLORS[:len(buckets)])),
    )
    # Click a bucket in the legend to highlight it (shift-click for several); the others fade.
    pick = alt.selection_point(fields=["bucket"], bind="legend")
    fade = alt.condition(pick, alt.value(1), alt.value(0.12))
    lines = base.mark_line(strokeWidth=2).encode(order="mean_p:Q", opacity=fade).add_params(pick)
    points = base.mark_circle(size=80, stroke="#fcfcfb", strokeWidth=2).encode(
        opacity=fade,
        tooltip=[alt.Tooltip("bucket:N", title="Days to meeting"), alt.Tooltip("bin:N", title="Bin"),
                 alt.Tooltip(f"{size}:Q", format=",.0f", title=size_title), "contracts:Q", "meetings:Q",
                 alt.Tooltip("mean_p:Q", format=".3f", title="Mean forecast"),
                 alt.Tooltip("observed:Q", format=".3f", title="Share YES")],
    )
    return (diagonal + lines + points).properties(width=380, height=380, title=title)


@app.function(hide_code=True)
def ce_heatmap(long, row, row_title, row_order, stat_title, title):
    """Heatmap of log loss: one row per `row` value, one column per horizon, darker = higher loss (log scale).
    `long` has columns `row`, `ttm`, `value` and `n` (forecasts behind the cell)."""
    long = long.assign(horizon=long.ttm.map(lambda h: f"{h}d"))
    return alt.Chart(long).mark_rect(stroke="#fcfcfb", strokeWidth=2, cornerRadius=2).encode(
        x=alt.X("horizon:O", title="Days to meeting", sort=[f"{h}d" for h in sorted(long.ttm.unique(), reverse=True)],
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{row}:O", title=row_title, sort=row_order),
        color=alt.Color("value:Q", title=f"{stat_title} log loss",
                        scale=alt.Scale(type="log", range=["#cde2fb", "#86b6ef", "#2a78d6", "#104281", "#0d366b"])),
        tooltip=[alt.Tooltip(f"{row}:O", title=row_title), alt.Tooltip("horizon:O", title="Days to meeting"),
                 alt.Tooltip("value:Q", format=".3f", title=f"{stat_title} log loss"),
                 alt.Tooltip("n:Q", format=",", title="Forecasts")],
    ).properties(width="container", height=alt.Step(28), title=title)


@app.function(hide_code=True)
def equal_trade_buckets(trades_by_day, n):
    """Cut days 1..max into `n` runs of whole days holding about the same number of trades: each cut falls on the
    day whose cumulative share of trades is nearest k/n. Returns {"a–b days": (a, b)}, nearest the meeting first."""
    days = range(1, trades_by_day.days_to_meeting.max() + 1)
    counts = trades_by_day.set_index("days_to_meeting").trades.reindex(days, fill_value=0)
    share = counts.cumsum() / counts.sum()
    cuts = [int((share - k / n).abs().idxmin()) for k in range(1, n)]
    assert cuts == sorted(set(cuts)), f"two cuts on the same day: {cuts}"
    edges = list(zip([1] + [c + 1 for c in cuts], cuts + [max(days)]))
    return {f"{lo}–{hi} days": (lo, hi) for lo, hi in edges}


@app.function(hide_code=True)
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


@app.function(hide_code=True)
def fit_logit(X, y):
    """Logistic regression by Newton-Raphson, started at the calibrated curve (0, 1, 0, ...). Returns
    (coefficients, fitted probabilities, Hessian), or None when there is no finite solution (perfect separation)."""
    b = np.zeros(X.shape[1])
    b[1] = 1.0
    for _ in range(100):
        mu = 1 / (1 + np.exp(-(X @ b)))
        H = X.T @ (X * (mu * (1 - mu))[:, None])
        try:
            step = np.linalg.solve(H, X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return None
        b = b + step
        if not np.isfinite(b).all() or np.abs(b).max() > 50:
            return None
        if np.abs(step).max() < 1e-10:
            return b, mu, H
    return None


@app.function(hide_code=True)
def cluster_covariance(X, y, mu, H, g):
    """Cluster-robust (CR1) covariance of logistic regression coefficients; g holds cluster codes 0..G-1."""
    n_clusters = g.max() + 1
    S = np.zeros((n_clusters, X.shape[1]))
    np.add.at(S, g, X * (y - mu)[:, None])  # score per cluster
    H_inv = np.linalg.inv(H)
    return n_clusters / (n_clusters - 1) * H_inv @ S.T @ S @ H_inv


@app.function(hide_code=True)
def bootstrap_joint_p(boots, estimate, null):
    """Share of bootstrap draws farther from the estimate than the null is, in the metric of the bootstrap
    covariance: a joint test that does not rely on many clusters."""
    V_inv = np.linalg.inv(np.atleast_2d(np.cov(boots.T)))
    dist = np.einsum("ij,jk,ik->i", boots - estimate, V_inv, boots - estimate)
    d = estimate - null
    return (np.sum(dist >= d @ V_inv @ d) + 1) / (len(boots) + 1)


@app.function(hide_code=True)
def cluster_bootstrap(X, y, g, n_boot, seed):
    """Coefficients of logistic regressions on whole clusters resampled with replacement (pairs cluster
    bootstrap). Resamples with no finite estimate (perfect separation) are skipped."""
    rows_of = [np.flatnonzero(g == k) for k in range(g.max() + 1)]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        rows = np.concatenate([rows_of[k] for k in rng.integers(0, len(rows_of), len(rows_of))])
        r = fit_logit(X[rows], y[rows])
        if r is not None:
            boots.append(r[0])
    return np.array(boots)


@app.function(hide_code=True)
def calibration_regression(y, f, clusters, n_boot=999, seed=0):
    """Logistic calibration regression P(Y = 1 | f) = Λ(α + β·logit f), fitted by maximum likelihood.

    Calibrated forecasts have α = 0 and β = 1. Tests use cluster-robust (CR1) standard errors. With few
    clusters those are unreliable, so whole clusters are also resampled (pairs cluster bootstrap) for percentile
    intervals and a joint test. A resample in which the outcome is perfectly separated has no finite estimate
    and is skipped; `boot_ok` is the share kept.
    """
    X = np.column_stack([np.ones(len(f)), np.log(f / (1 - f))])
    y = np.asarray(y, dtype=float)
    g = pd.factorize(pd.Series(clusters))[0]
    b, mu, H = fit_logit(X, y)
    V = cluster_covariance(X, y, mu, H, g)
    d = b - np.array([0.0, 1.0])
    se = np.sqrt(np.diag(V))

    boots = cluster_bootstrap(X, y, g, n_boot, seed)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return dict(
        forecasts=len(y), clusters=g.max() + 1,
        alpha=b[0], alpha_se=se[0], alpha_ci=f"[{lo[0]:.2f}, {hi[0]:.2f}]",
        beta=b[1], beta_se=se[1], beta_ci=f"[{lo[1]:.2f}, {hi[1]:.2f}]",
        p_alpha_0=math.erfc(abs(d[0] / se[0]) / math.sqrt(2)),
        p_beta_1=math.erfc(abs(d[1] / se[1]) / math.sqrt(2)),
        p_joint=math.exp(-(d @ np.linalg.solve(V, d)) / 2),  # chi-squared with 2 degrees of freedom
        p_joint_boot=bootstrap_joint_p(boots, b, np.array([0.0, 1.0])),
        boot_ok=len(boots) / n_boot,
    )


@app.function(hide_code=True)
def regime_difference(y, f, clusters, after, n_boot=999, seed=0):
    """Does the calibration curve change at a break? Fits
    P(Y = 1 | f) = Λ(α + β·logit f + Δα·D + Δβ·D·logit f), with D = 1 after the break, so Δα and Δβ are the
    changes in intercept and slope; Δα = Δβ = 0 means one curve fits both regimes. Every cluster must lie in one
    regime. Tests as in `calibration_regression`; the bootstrap resamples clusters within each regime, so each
    keeps its number of clusters.
    """
    logit = np.log(f / (1 - f))
    D = np.asarray(after, dtype=float)
    X = np.column_stack([np.ones(len(f)), logit, D, D * logit])
    y = np.asarray(y, dtype=float)
    g = pd.factorize(pd.Series(clusters))[0]
    assert (pd.Series(D).groupby(g).nunique() == 1).all(), "a cluster spans both regimes"
    b, mu, H = fit_logit(X, y)
    V = cluster_covariance(X, y, mu, H, g)
    d, V_d = b[2:], V[2:, 2:]
    se = np.sqrt(np.diag(V_d))

    rows_of = [np.flatnonzero(g == k) for k in range(g.max() + 1)]
    strata = [[k for k in range(len(rows_of)) if D[rows_of[k][0]] == v] for v in (0.0, 1.0)]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        picks = [k for s in strata for k in rng.choice(s, len(s))]
        rows = np.concatenate([rows_of[k] for k in picks])
        r = fit_logit(X[rows], y[rows])
        if r is not None:
            boots.append(r[0][2:])
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return dict(
        delta_alpha=d[0], delta_alpha_ci=f"[{lo[0]:.2f}, {hi[0]:.2f}]",
        delta_beta=d[1], delta_beta_ci=f"[{lo[1]:.2f}, {hi[1]:.2f}]",
        p_delta_alpha=math.erfc(abs(d[0] / se[0]) / math.sqrt(2)),
        p_delta_beta=math.erfc(abs(d[1] / se[1]) / math.sqrt(2)),
        p_joint=math.exp(-(d @ np.linalg.solve(V_d, d)) / 2),  # chi-squared with 2 degrees of freedom
        p_joint_boot=bootstrap_joint_p(boots, d, np.zeros(2)),
        boot_ok=len(boots) / n_boot,
    )


@app.function(hide_code=True)
def calibration_curves_chart(results, field, domain, colors, legend_title, title):
    """Fitted calibration curves Λ(α + β·logit p), one per row of `results`, against the 45° line."""
    _p = np.linspace(0.01, 0.99, 99)
    _curves = pd.concat([
        pd.DataFrame({field: getattr(r, field), "p": _p,
                      "fitted": 1 / (1 + np.exp(-(r.alpha + r.beta * np.log(_p / (1 - _p)))))})
        for r in results.itertuples()
    ])
    _diagonal = alt.Chart(pd.DataFrame({"p": [0, 1]})).mark_line(color="#9a9994", strokeDash=[4, 4]).encode(
        x="p:Q", y="p:Q")
    _lines = alt.Chart(_curves).mark_line(strokeWidth=2).encode(
        x=alt.X("p:Q", title="Forecast", axis=alt.Axis(format="%")),
        y=alt.Y("fitted:Q", title="Fitted P(YES)", axis=alt.Axis(format="%")),
        color=alt.Color(f"{field}:N", scale=alt.Scale(domain=domain, range=colors), title=legend_title),
        tooltip=[f"{field}:N", alt.Tooltip("p:Q", format=".2f", title="Forecast"),
                 alt.Tooltip("fitted:Q", format=".3f", title="Fitted P(YES)")],
    )
    return (_diagonal + _lines).properties(width=380, height=380, title=title)


@app.function(hide_code=True)
def rolling_calibration(forecasts, window, n_boot=499, seed=0):
    """The calibration regression Λ(α + β·logit f) on every run of `window` consecutive meetings, ordered by meeting
    date. Point estimates by maximum likelihood, 95% percentile intervals from the meeting bootstrap of
    `calibration_regression`. A window in which the outcome is perfectly separated has no finite estimate; its
    row has NaN coefficients."""
    meetings = forecasts.groupby("event_ticker").meeting_date.first().sort_values()
    rows = []
    for i in range(len(meetings) - window + 1):
        _in = meetings.index[i:i + window]
        x = forecasts[forecasts.event_ticker.isin(_in)]
        X = np.column_stack([np.ones(len(x)), np.log(x.f / (1 - x.f))])
        y = x.y.to_numpy(dtype=float)
        row = dict(first_meeting=meetings.iloc[i], last_meeting=meetings.iloc[i + window - 1],
                   meetings=window, forecasts=len(x),
                   after_break=(meetings.iloc[i:i + window] >= LIQUIDITY_BREAK).mean())
        fit = fit_logit(X, y)
        if fit is None:
            rows.append(row)
            continue
        boots = cluster_bootstrap(X, y, pd.factorize(x.event_ticker)[0], n_boot, seed)
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        rows.append({**row, "alpha": fit[0][0], "alpha_lo": lo[0], "alpha_hi": hi[0],
                     "beta": fit[0][1], "beta_lo": lo[1], "beta_hi": hi[1], "boot_ok": len(boots) / n_boot})
    return pd.DataFrame(rows, columns=["first_meeting", "last_meeting", "meetings", "forecasts", "after_break",
                                       "alpha", "alpha_lo", "alpha_hi", "beta", "beta_lo", "beta_hi", "boot_ok"])


@app.function(hide_code=True)
def rolling_coefficient_chart(rolling, coef, null, title):
    """One coefficient of the rolling calibration regression against the last meeting in each window, with its 95%
    bootstrap band, the calibrated value `null`, and the liquidity break."""
    _base = alt.Chart(rolling.dropna(subset=[coef])).encode(
        x=alt.X("last_meeting:T", title="Last meeting in window"))
    _band = _base.mark_area(color="#86b6ef", opacity=0.35).encode(
        y=alt.Y(f"{coef}_lo:Q", title=coef), y2=f"{coef}_hi:Q")
    _line = _base.mark_line(color="#104281", strokeWidth=2).encode(y=f"{coef}:Q")
    _points = _base.mark_circle(color="#104281", size=64, opacity=1).encode(
        y=f"{coef}:Q",
        tooltip=[alt.Tooltip("first_meeting:T", title="First meeting"),
                 alt.Tooltip("last_meeting:T", title="Last meeting"),
                 alt.Tooltip(f"{coef}:Q", format=".2f"),
                 alt.Tooltip(f"{coef}_lo:Q", format=".2f", title="95% low"),
                 alt.Tooltip(f"{coef}_hi:Q", format=".2f", title="95% high"),
                 alt.Tooltip("forecasts:Q", title="Forecasts"),
                 alt.Tooltip("after_break:Q", format=".0%", title="Meetings from May 2025"),
                 alt.Tooltip("boot_ok:Q", format=".2f")],
    )
    _null = alt.Chart(pd.DataFrame({"y": [null]})).mark_rule(color="#9a9994", strokeDash=[4, 4]).encode(y="y:Q")
    _break = alt.Chart(pd.DataFrame({"x": [LIQUIDITY_BREAK], "label": ["Liquidity break"]}))
    _break = (_break.mark_rule(color="#9a9994").encode(x="x:T")
              + _break.mark_text(align="left", dx=4, dy=-6, color="#6b6a66").encode(
                  x="x:T", y=alt.value(0), text="label:N"))
    return (_band + _null + _break + _line + _points).properties(width=560, height=220, title=title)


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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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

    These settings apply to sections 2–7. The defaults are the main specification; the others are the
    robustness checks. The tick-floor setting applies to the contract-level sections (2–4, 6 and 7) only: section 5
    scores each meeting's full distribution, so it always keeps every bucket.
    """)
    return


@app.cell(hide_code=True)
def choices():
    price = mo.ui.radio(options={"Normalized: q = p / Σp (main)": "q", "Raw mid p": "p"},
                        value="Normalized: q = p / Σp (main)", label="Price")
    include_floor = mo.ui.checkbox(value=False, label="Include tick-floor contracts (mid below 1¢ or from 99¢ up)")
    drop_stale = mo.ui.checkbox(value=False, label=f"Drop quotes unchanged for more than {STALE_DAYS} days")
    mo.hstack([price, mo.vstack([include_floor, drop_stale])], justify="start", gap=3)
    return drop_stale, include_floor, price


@app.cell(hide_code=True)
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

    The forecasts are 4pm snapshots taken 1, 2, 3, 5, 7, 10, 14, 21, 30, 45 and 60 days before each meeting
    (`TTMS` in the setup cell), closer together near the meeting, where prices move most. A horizon group pools
    its snapshots, so each contract appears once per day in the group: 1–21 days counts every contract up to
    eight times, with the same outcome each time.

    Bins are $[p_k, p_k + \delta)$ except at the ends, which are split at 1¢ and 99¢: below 1¢ and from 99¢ up are
    contracts at the tick floor (0¢ bid / 1¢ ask, or no ask), which say little about beliefs, so they get bins of
    their own and the priced longshots and favorites sit in $[0.01, \delta)$ and $[1 - \delta, 0.99)$. In the main
    specification the floor contracts are excluded, so those two bins hold only contracts priced at exactly 1¢ or
    99¢ that the rescaling to $q$ moved just across the edge.
    """)
    return


@app.cell(hide_code=True)
def controls():
    horizon = mo.ui.dropdown(options=HORIZON_GROUPS, value="1–21 days (main)", label="Days to meeting")
    bin_width = mo.ui.dropdown(options={"5¢": 0.05, "10¢": 0.10, "20¢": 0.20}, value="10¢", label="Bin width δ")
    mo.hstack([horizon, bin_width], justify="start", gap=2)
    return bin_width, horizon


@app.cell(hide_code=True)
def reliability(bin_width, horizon, sample):
    reliability = reliability_table(sample[sample.ttm.isin(horizon.value)], bin_width.value)
    mo.hstack([
        reliability_chart(reliability, "forecasts", "Forecasts", f"Reliability · {horizon.selected_key}"),
        mo.ui.table(reliability.drop(columns="bin_from").round(4), selection=None),
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def ttm_md():
    mo.md(r"""
    ### By time to maturity

    The same diagram with one line per time-to-maturity bucket, darker closer to the meeting, to see whether the
    pattern changes as the meeting approaches. The buckets split days 1–60 before the meeting into five runs of whole
    days that each hold about a fifth of all trades in settled contracts (table below), so they are narrow near the
    meeting, where trading is heaviest, and wide further out. They are recomputed from the trade data on each run.
    Each daily line pools the snapshot days (`TTMS`) that fall in its bucket, listed in the table's `snapshot_days`.
    The analysis choices apply. Bins default to 20¢, wider than above: split five ways, the middle bins hold only a
    few meetings each, and at 10¢ single outcomes swing them between 0% and 100%. Points are not sized: hover for a
    bin's forecasts and meetings. Click a bucket in the legend to highlight it (shift-click for several).

    The summary's **calibration error** is the forecast-weighted average gap between the share settled YES and the
    mean forecast across bins, $\sum_k \frac{n_k}{N}\,\lvert \bar y_k - \bar p_k \rvert$ (0 = on the diagonal). It
    depends on the bin width and is descriptive only; the test comes later.
    """)
    return


@app.cell(hide_code=True)
def ttm_buckets_cell(con, queries):
    _trades = con.execute(queries["trades_by_day"], {"max_days": MINUTE_WINDOW_DAYS}).df()
    ttm_buckets = equal_trade_buckets(_trades, N_TTM_BUCKETS)
    _split = pd.DataFrame([
        {"bucket": name, "days": hi - lo + 1,
         "trades": int(_trades[_trades.days_to_meeting.between(lo, hi)].trades.sum())}
        for name, (lo, hi) in ttm_buckets.items()
    ]).assign(share=lambda d: d.trades / d.trades.sum(),
              snapshot_days=lambda d: [", ".join(str(h) for h in TTMS if lo <= h <= hi)
                                       for lo, hi in ttm_buckets.values()])
    mo.ui.table(_split.round(3), selection=None)
    return (ttm_buckets,)


@app.cell(hide_code=True)
def ttm_controls():
    ttm_bin_width = mo.ui.dropdown(options={"10¢": 0.10, "20¢": 0.20, "25¢": 0.25}, value="20¢",
                                   label="Bin width δ (time-to-maturity charts)")
    ttm_bin_width
    return (ttm_bin_width,)


@app.cell(hide_code=True)
def ttm_reliability(sample, ttm_bin_width, ttm_buckets):
    _by_bucket = {name: sample[sample.ttm.between(lo, hi)] for name, (lo, hi) in ttm_buckets.items()}
    _tables = pd.concat([reliability_table(d, ttm_bin_width.value).assign(bucket=name)
                         for name, d in _by_bucket.items()])
    _summary = bucket_summary(_tables, "forecasts", {name: d.event_ticker.nunique() for name, d in _by_bucket.items()})
    mo.hstack([
        reliability_by_bucket_chart(_tables, list(ttm_buckets), "forecasts", "Forecasts",
                                    "Reliability by days to meeting"),
        mo.ui.tabs({
            "Summary by bucket": mo.ui.table(_summary.round(4), selection=None),
            "Every bin": mo.ui.table(_tables[["bucket", "bin", "forecasts", "contracts", "meetings", "mean_p",
                                              "observed"]].round(4), selection=None, page_size=25),
        }),
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


@app.cell(hide_code=True)
def minutes(con, queries, ttm_buckets):
    minute_quotes = con.execute(queries["minute_quotes"], {"days_before": MINUTE_WINDOW_DAYS}).df()
    minute_quotes_by_bucket = con.execute(queries["minute_quotes_by_bucket"], {
        "buckets": list(ttm_buckets),
        "lo": [lo for lo, _ in ttm_buckets.values()],
        "hi": [hi for _, hi in ttm_buckets.values()],
    }).df()
    return minute_quotes, minute_quotes_by_bucket


@app.cell(hide_code=True)
def minute_controls():
    drop_wide = mo.ui.checkbox(value=False, label="Drop quotes with a spread above 20¢")
    drop_wide
    return (drop_wide,)


@app.cell(hide_code=True)
def minute_reliability(bin_width, drop_wide, minute_quotes):
    _q = minute_quotes[minute_quotes.spread <= 0.20] if drop_wide.value else minute_quotes
    reliability_minutes = minute_reliability_table(_q, bin_width.value)
    mo.hstack([
        reliability_chart(reliability_minutes, "contract_hours", "Contract-hours",
                          f"Reliability · every minute, 0–{MINUTE_WINDOW_DAYS} days"),
        mo.ui.table(reliability_minutes.drop(columns="bin_from").round(4), selection=None),
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def minute_ttm_md():
    mo.md("""
    #### Every minute, by time to maturity (descriptive)

    The minute data split into the same time-to-maturity buckets as above, by the ET calendar day each quote was in
    force: a quote that spans a bucket boundary counts toward both, for the time it spent in each. The meeting day
    itself is in no bucket, so unlike the chart above this view stops at midnight before the meeting. The
    time-to-maturity bin width and the spread checkbox apply.
    """)
    return


@app.cell(hide_code=True)
def minute_ttm_reliability(drop_wide, minute_quotes_by_bucket, ttm_bin_width, ttm_buckets):
    _q = minute_quotes_by_bucket[minute_quotes_by_bucket.spread <= 0.20] if drop_wide.value else minute_quotes_by_bucket
    _by_bucket = {name: _q[_q.bucket == name] for name in ttm_buckets}
    _tables = pd.concat([minute_reliability_table(d, ttm_bin_width.value).assign(bucket=name)
                         for name, d in _by_bucket.items()])
    _summary = bucket_summary(_tables, "contract_hours",
                              {name: d.event_ticker.nunique() for name, d in _by_bucket.items()})
    mo.hstack([
        reliability_by_bucket_chart(_tables, list(ttm_buckets), "contract_hours", "Contract-hours",
                                    "Reliability by days to meeting · every minute"),
        mo.ui.tabs({
            "Summary by bucket": mo.ui.table(_summary.round(4), selection=None),
            "Every bin": mo.ui.table(_tables[["bucket", "bin", "contract_hours", "contracts", "meetings", "mean_p",
                                              "observed"]].round(4), selection=None, page_size=25),
        }),
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def ce_type_md():
    mo.md(r"""
    ## 3 · Cross-entropy by contract type

    Log loss per contract type (rows) and horizon (columns), as a mean or a median (control below), as a heatmap
    (darker = worse; the color scale is logarithmic, since losses run from 0.005 to above 1) or a table. `settled YES`
    counts the meetings in which that type won. A type that rarely or never wins scores well simply by being priced
    near zero, so read each row next to that count.

    The mean is driven by a few surprises: a single meeting can double a horizon's mean. The median ignores them,
    and for types that rarely win it only returns the loss of a contract at the 0.5¢ floor, $-\ln(0.995) \approx 0.005$.
    In the main specification those floor contracts are excluded, so types that sat at the floor throughout drop out.
    The last row averages within each meeting first and then takes the mean or median across meetings, which treats
    the meeting as the independent unit.
    """)
    return


@app.cell(hide_code=True)
def ce_controls():
    ce_stat = mo.ui.dropdown(options={"Mean": "mean", "Median": "median"}, value="Mean", label="Statistic")
    ce_stat
    return (ce_stat,)


@app.cell(hide_code=True)
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

    _types = sample.groupby(["move", "ttm"]).log_loss.agg(value=_stat, n="size").reset_index()
    _all = sample.groupby("ttm").log_loss.agg(value=_stat, n="size").reset_index().assign(move="All contracts")
    _heat = ce_heatmap(pd.concat([_types, _all]), "move", "Contract type", [*_order, "All contracts"],
                       ce_stat.selected_key, f"{ce_stat.selected_key} log loss by contract type")
    mo.ui.tabs({"Chart": _heat, "Table": mo.ui.table(_by_type.round(3).reset_index(names="contract type"),
                                                      selection=None)})
    return


@app.cell(hide_code=True)
def ce_meetings_md():
    mo.md("""
    ### Which meetings drive the score

    Each dot is one meeting's average log loss across its contracts at one horizon; the lines are the mean and the
    median across meetings (the last row of the table above). Where the mean sits well above the median, a few
    surprise meetings are pulling it up; they are labeled at their worst horizon. Hover for the meeting and the
    decision.
    """)
    return


@app.cell(hide_code=True)
def ce_meetings(forecasts, sample):
    _decision = forecasts[forecasts.y == 1].groupby("event_ticker").move.first()
    _dates = forecasts.groupby("event_ticker").meeting_date.first()
    _per_meeting = sample.groupby(["ttm", "event_ticker"], as_index=False).log_loss.mean()
    _per_meeting["meeting"] = _per_meeting.event_ticker.map(lambda e: pd.Timestamp(_dates[e]).strftime("%b %Y"))
    _per_meeting["decision"] = _per_meeting.event_ticker.map(_decision)
    _lines = (_per_meeting.groupby("ttm").log_loss.agg(Mean="mean", Median="median").reset_index()
              .melt(id_vars="ttm", var_name="statistic", value_name="log_loss"))
    # Label each surprise meeting once, at its worst horizon.
    _worst = _per_meeting.loc[_per_meeting.groupby("event_ticker").log_loss.idxmax()]
    _worst = _worst[_worst.log_loss > 0.5]

    _x = alt.X("ttm:Q", title="Days to meeting", scale=alt.Scale(reverse=True))
    _y = alt.Y("log_loss:Q", title="Log loss per meeting (lower is better)")
    _dots = alt.Chart(_per_meeting).mark_circle(size=60, color="#9a9994", opacity=0.6).encode(
        x=_x, y=_y,
        tooltip=["meeting:N", "decision:N", alt.Tooltip("ttm:Q", title="Days to meeting"),
                 alt.Tooltip("log_loss:Q", format=".3f", title="Log loss")])
    _stat_lines = alt.Chart(_lines).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
        x=_x, y=_y,
        color=alt.Color("statistic:N", title="Across meetings",
                        scale=alt.Scale(domain=["Mean", "Median"], range=["#2a78d6", "#eb6834"]),
                        legend=alt.Legend(orient="top-left")),
        tooltip=["statistic:N", alt.Tooltip("ttm:Q", title="Days to meeting"),
                 alt.Tooltip("log_loss:Q", format=".3f", title="Log loss")])
    # Labels sit right of their dot, or left of it near the meeting so they stay inside the plot.
    _labels = [
        alt.Chart(_worst[side]).mark_text(align=align, dx=dx, color="#52514e").encode(
            x=_x, y=_y, text="label:N").transform_calculate(label="datum.meeting + ' · ' + datum.decision")
        for side, align, dx in ((_worst.ttm >= 15, "left", 8), (_worst.ttm < 15, "right", -8))
    ]
    alt.layer(_dots, _stat_lines, *_labels).properties(
        width="container", height=320, title="Log loss per meeting, by days to meeting")
    return


@app.cell(hide_code=True)
def ce_bucket_md():
    mo.md("""
    ## 4 · Cross-entropy by probability bin

    Log loss per probability bin (rows, bin width from the control in section 2) and horizon (columns), as the mean
    or median chosen in section 3, as a heatmap (darker = worse, logarithmic color scale; highest bins at the top) or a
    table, with the number of forecasts behind each cell in the last tab. As in section 2,
    the bins below 1¢ and from 99¢ up hold the contracts at the tick floor.
    """)
    return


@app.cell(hide_code=True)
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


    _long = (_binned.groupby(["bin_from", "ttm"]).log_loss.agg(value=ce_stat.value, n="size").reset_index()
             .assign(bin=lambda d: [bin_label(lo, _delta) for lo in d.bin_from]))
    _heat = ce_heatmap(_long, "bin", "Probability bin",
                       [bin_label(lo, _delta) for lo in sorted(_long.bin_from.unique(), reverse=True)],
                       ce_stat.selected_key, f"{ce_stat.selected_key} log loss by probability bin")
    mo.ui.tabs({"Chart": _heat, f"{ce_stat.selected_key} cross-entropy": _table(_ce, 3),
                "Forecasts per cell": _table(_n, 0)})
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def regression_md():
    mo.md(r"""
    ## 6 · Calibration regression

    A test of calibration without bins. For each horizon group, the outcome is regressed on the forecast in log
    odds,

    $$\mathbb{P}(Y_i = 1 \mid f_i) = \Lambda\big(\alpha + \beta \operatorname{logit} f_i\big), \qquad \Lambda(z) = \frac{1}{1 + e^{-z}},$$

    by maximum likelihood. Calibrated forecasts have $\alpha = 0$ and $\beta = 1$, and the regression then returns
    the forecast itself.

    - $\beta > 1$: the market is **under-confident**. Prices are too close to 50%: favorites win more often than
      priced and longshots less often (the favorite-longshot pattern of section 2).
    - $\beta < 1$: **over-confident**: prices are too far from 50%.
    - $\alpha \ne 0$: at a 50% price, YES happens more ($\alpha > 0$) or less ($\alpha < 0$) often than half the time.

    The sample is the one set in *Analysis choices*. Forecasts of one meeting share its outcome and each contract
    appears once per snapshot day, so standard errors are clustered by meeting. With 18 to 28 meetings the clustered
    tests can be too optimistic, so meetings are also resampled with replacement (999 times) for the 95% intervals
    and a second joint test, `p_joint_boot`. Where the two joint tests disagree, the bootstrap is the safer one.
    `boot_ok` is the share of resamples with a finite estimate: one in which some forecast level perfectly
    separates YES from NO has none and is skipped, which pulls the interval for $\beta$ down. Read rows with
    `boot_ok` well below 1 (the 45–60 day group) with care.
    """)
    return


@app.cell(hide_code=True)
def regression(sample):
    _rows = []
    for _name, _hs in HORIZON_GROUPS.items():
        _x = sample[sample.ttm.isin(_hs)]
        _rows.append({"horizon": _name, **calibration_regression(_x.y, _x.f.to_numpy(), _x.event_ticker)})
    calibration_fit = pd.DataFrame(_rows)
    mo.vstack([
        calibration_curves_chart(calibration_fit, "horizon", list(HORIZON_GROUPS), TTM_COLORS[::2], "Days to meeting",
                                 "Fitted calibration curves"),
        mo.ui.table(calibration_fit.round(3), selection=None),
    ])
    return


@app.cell(hide_code=True)
def regime_md():
    mo.md(r"""
    ## 7 · Before and after the liquidity break

    Kalshi's Fed markets changed size in spring 2025: the May 2025 meeting traded about 41 million contracts against
    2.8 million for March 2025, and every meeting since has traded 20 to 100 million. The calibration regression of
    section 6 is fitted separately on the meetings before May 2025 (16 meetings, May 2023 to March 2025) and from May
    2025 (12 meetings), in each horizon group.

    **The test.** One regression on both regimes,

    $$\mathbb{P}(Y_i = 1 \mid f_i) = \Lambda\big(\alpha + \beta \operatorname{logit} f_i + \Delta\alpha\, D_i + \Delta\beta\, D_i \operatorname{logit} f_i\big),$$

    with $D_i = 1$ for meetings from May 2025. $\Delta\beta$ is the change in slope and $\Delta\alpha$ the change in
    intercept; $\Delta\alpha = \Delta\beta = 0$ means one calibration curve fits both regimes (`p_joint`). Standard
    errors are clustered by meeting, and the bootstrap resamples meetings within each regime.

    **Liquidity is not the only difference.** The two regimes are also different stretches of monetary policy: mostly
    holds before May 2025, a cutting cycle after. A change here is consistent with liquidity, not proof of it. The
    fed funds futures, whose liquidity did not change, are the control: `03_fedfunds_vs_kalshi.py` runs the same
    split on both sources.

    With 12 to 16 meetings per regime the tests have little power: a large p-value means no detectable change, not no
    change. The 45–60 day group has almost no meetings before May 2025 (they were listed only weeks ahead).
    """)
    return


@app.cell(hide_code=True)
def regime(sample):
    _regime = np.where(sample.meeting_date >= LIQUIDITY_BREAK, REGIMES[1], REGIMES[0])
    _fits, _diffs = [], []
    for _name, _hs in HORIZON_GROUPS.items():
        _in = sample.ttm.isin(_hs).to_numpy()
        for _r in REGIMES:
            _x = sample[_in & (_regime == _r)]
            if _x.event_ticker.nunique() >= 5:
                _fits.append({"horizon": _name, "regime": _r,
                              **calibration_regression(_x.y, _x.f.to_numpy(), _x.event_ticker)})
        _x = sample[_in]
        if (pd.Series(_regime[_in]).groupby(_x.event_ticker.to_numpy()).first().value_counts() >= 5).sum() == 2:
            _diffs.append({"horizon": _name, **regime_difference(_x.y, _x.f.to_numpy(), _x.event_ticker,
                                                                  _regime[_in] == REGIMES[1])})
    regime_fit = pd.DataFrame(_fits)
    regime_diff = pd.DataFrame(_diffs)
    mo.vstack([
        calibration_curves_chart(regime_fit[regime_fit.horizon == "1–21 days (main)"], "regime", REGIMES,
                                 ["#86b6ef", "#104281"], "Meetings", "Fitted calibration curves by regime · 1–21 days"),
        mo.md("**Calibration in each regime** (groups with fewer than 5 meetings in a regime are left out)"),
        mo.ui.table(regime_fit.round(3), selection=None),
        mo.md("**Change from before to after May 2025**"),
        mo.ui.table(regime_diff.round(3), selection=None),
    ])
    return


@app.cell(hide_code=True)
def rolling_md():
    mo.md(r"""
    ## 8 · Rolling calibration regression (descriptive)

    Section 7 splits the meetings at one date chosen in advance. Here the regression of section 6 is refitted on
    every run of consecutive meetings (ordered by meeting date) to show how $\alpha$ and $\beta$ move over time,
    without choosing a break. Each point is one window, plotted at its last meeting, so it uses only meetings held
    by that date. The band is the 95% meeting-bootstrap interval (499 resamples); the dashed line is the
    calibrated value ($\beta = 1$, $\alpha = 0$).

    **Read it as a picture, not a test.** Neighbouring windows share all but one meeting, so the points are
    strongly dependent, and a step in the line is usually one meeting entering or leaving the window: hover a
    point to see the window, and compare with *Which meetings drive the score* in section 3. With 8 to 16 meetings
    per window a single surprise meeting is a large share of the sample. A window in which some forecast level
    perfectly separates YES from NO has no finite estimate and is left out (listed below the charts), and
    `boot_ok` well below 1 means the band is pulled down, as in section 6.
    """)
    return


@app.cell(hide_code=True)
def rolling_controls():
    rolling_horizon = mo.ui.dropdown(options=list(HORIZON_GROUPS), value="1–21 days (main)", label="Days to meeting")
    rolling_window = mo.ui.slider(8, 16, value=10, label="Meetings per window", show_value=True)
    mo.hstack([rolling_horizon, rolling_window], justify="start", gap=2)
    return rolling_horizon, rolling_window


@app.cell(hide_code=True)
def rolling(rolling_horizon, rolling_window, sample):
    _x = sample[sample.ttm.isin(HORIZON_GROUPS[rolling_horizon.value])]
    mo.stop(_x.event_ticker.nunique() < rolling_window.value,
            mo.md(f"**{_x.event_ticker.nunique()} meetings in this group**, fewer than one window."))
    rolling_fit = rolling_calibration(_x, rolling_window.value)
    _separated = rolling_fit[rolling_fit.beta.isna()]
    mo.vstack([
        rolling_coefficient_chart(rolling_fit, "beta", 1.0,
                                  f"Slope β · {rolling_window.value}-meeting windows · {rolling_horizon.value}"),
        rolling_coefficient_chart(rolling_fit, "alpha", 0.0,
                                  f"Intercept α · {rolling_window.value}-meeting windows · {rolling_horizon.value}"),
        mo.md(f"**No finite estimate (perfect separation):** {len(_separated)} of {len(rolling_fit)} windows"
              + "".join(f"<br>{r.first_meeting:%b %Y} – {r.last_meeting:%b %Y}" for r in _separated.itertuples())),
        mo.ui.table(rolling_fit.round(3), selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()
