# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "duckdb", "numpy", "pandas", "pyarrow", "altair", "openpyxl"]
#
# [tool.uv]
# python-preference = "only-managed"  # Anaconda's Python misreports macOS as 10.16, so uv rejects modern wheels
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import calendar
    import math
    import re
    from pathlib import Path

    import altair as alt
    import duckdb
    import marimo as mo
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parent
    FF_XLSX = ROOT / "data/KalshiProjectFedFunds.xlsx"
    CLEAN_CANDLES = ROOT / "data/cleaned/kalshi_candles_1m.parquet"
    # 1-minute bid/ask bars from data/KalshiProjectFFIntraday_template.xlsx, refreshed on a Bloomberg Terminal
    # (the data of 04_leadlag_fedfunds_kalshi.py); only the minute-level reliability diagram uses them.
    INTRADAY_XLSX = ROOT / "data/bloomberg/ff_intraday.xlsx"
    INTRADAY_CACHE = ROOT / "data/bloomberg/ff_intraday_1m.parquet"  # rebuilt when the workbook is newer
    TZ = "America/New_York"  # the Terminal's time zone
    DECISION_HOUR = 14  # FOMC statements are released at 14:00 ET
    FF_COLS = [f"FF{n}" for n in range(1, 14)]  # FFn is the contract for month n - 1 relative to the date
    SAMPLE_START = pd.Timestamp("2021-01-01")  # the guide's sample; earlier far contracts have stale prints
    STEP = 0.25  # one policy move, in percentage points
    SPREAD_WINDOW = 10  # business days in the rolling EFFR-minus-midpoint median (decision days excluded)
    TTMS = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60]  # days to meeting, as in 02_calibration_kalshi.py
    SNAPSHOT_HOUR_ET = 15  # futures settle around 15:00 ET, so Kalshi is read at the same time
    P_FLOOR = 0.005  # scores clip both sources to [0.5¢, 99.5¢], the floor Kalshi's mid already has
    # Kalshi's liquidity break: the first meeting after it (May 2025) traded about 15 times the contracts of the
    # one before (March 2025), and volume stayed there. Regimes are split by meeting date.
    LIQUIDITY_BREAK = pd.Timestamp("2025-05-01")
    REGIMES = ["Before May 2025", "From May 2025"]

    # Same encoding as 01_eda_kalshi.py: cuts blue, hikes red, hold gray; darker toward the extremes.
    MOVE_COLORS = {"Cut >25": "#1d5fb0", "Cut 25": "#6aa3e8", "Hold": "#9a9994", "Hike 25": "#ee8a88",
                   "Hike >25": "#c0302f", "Hike 50": "#c0302f", "Hike >50": "#8f1d1c"}
    MOVES = list(MOVE_COLORS)
    # Range of 25bp steps each Kalshi bucket pays on (open-ended buckets run to infinity).
    MOVE_STEPS = {"Cut >25": (-np.inf, -2), "Cut 25": (-1, -1), "Hold": (0, 0), "Hike 25": (1, 1),
                  "Hike >25": (2, np.inf), "Hike 50": (2, 2), "Hike >50": (3, np.inf)}
    # Buckets the futures can price at every horizon; the rest are 0 unless more than one full step is priced.
    CORE_MOVES = ["Cut 25", "Hold", "Hike 25"]

    alt.data_transformers.disable_max_rows()


@app.function(hide_code=True)
def load_futures(path):
    """Fed funds futures 1-13, EFFR and target range from the Bloomberg pull, cleaned as in Step 0 of the guide.

    Rows whose 13 prices all repeat the previous row are dropped: those are exchange holidays carried
    forward by the pull. On a holiday that opens a month (1 Jan, Labor Day) the copied row holds last
    month's contracts under a new-month date, which would shift every contract by one month.
    One month before SAMPLE_START is kept so the first month's realized EFFR days are known.
    """
    raw = pd.read_excel(path, "Data", header=0, skiprows=[1])  # row 2 holds descriptions, not data
    raw.columns = ["date", *FF_COLS, "effr", "upper", "lower"]
    raw["date"] = pd.to_datetime(raw.date)
    raw = raw[raw.date >= SAMPLE_START - pd.DateOffset(months=1)].reset_index(drop=True)
    raw[raw.columns[1:]] = raw[raw.columns[1:]].apply(pd.to_numeric, errors="coerce")  # #N/A -> NaN
    assert raw[FF_COLS + ["effr", "upper", "lower"]].notna().all().all(), "missing prices after SAMPLE_START"
    stale = raw[FF_COLS].diff().abs().sum(axis=1).eq(0)
    return raw[~stale].reset_index(drop=True), raw.loc[stale, "date"]


@app.function(hide_code=True)
def load_fomc(path):
    """FOMC decision dates from the workbook, with the decision in bp for past meetings."""
    f = pd.read_excel(path, "FOMC_Dates")
    f.columns = ["meeting_date", "effective_date", "scheduled", "decision_bp", "upper_after", "lower_after", "notes"]
    f = f[f.meeting_date != "Start level"].copy()
    f["meeting_date"] = pd.to_datetime(f.meeting_date)
    return f.sort_values("meeting_date").reset_index(drop=True)


@app.function(hide_code=True)
def add_starting_rate(futures, meeting_dates):
    """Starting rate r0 (Step 6): today's range midpoint plus the rolling median EFFR spread.

    The guide suggests a median of recent EFFR prints; that mixes two regimes for two weeks after every
    move. The spread to the midpoint is stable across moves, so its median is taken instead. Decision
    days are skipped: the range moves that day, the EFFR only the next.
    """
    midpoint = (futures.upper + futures.lower) / 2
    spread = (futures.effr - midpoint).where(~futures.date.isin(meeting_dates))
    spread_median = spread.rolling(SPREAD_WINDOW, min_periods=3).median().ffill().bfill()
    return futures.assign(spread=futures.effr - midpoint, spread_median=spread_median, r0=midpoint + spread_median)


@app.function(hide_code=True)
def months_ahead(t, d):
    """Calendar months from date t to date d (0 = same month)."""
    return (d.year - t.year) * 12 + d.month - t.month


@app.function(hide_code=True)
def implied_moves(futures, meeting_dates):
    """Steps 1-5 of the guide for every date: one row per (date, upcoming meeting) with the expected
    move x in 25bp steps. Meetings are chained, so each one starts from the previous one's rate.

    Method B (the month after the meeting has no meeting: its average is the post-meeting rate) is
    preferred; Method A (un-blend the meeting month) amplifies price noise by N / (N - d). For a meeting
    on the last day of its month Method A is undefined, since that month says nothing about the new
    rate; if Method B is not available either, that meeting and all later ones are left out that day.
    """
    effr_daily = futures.set_index("date").effr.resample("D").ffill()  # weekends carry Friday's rate
    rows = []
    for r in futures[futures.date >= SAMPLE_START].itertuples():
        t = r.date
        rbar = 100 - np.array([getattr(r, c) for c in FF_COLS])  # implied monthly averages (Step 2)
        upcoming = [d for d in meeting_dates if d > t and months_ahead(t, d) <= 12]
        months = [months_ahead(t, d) for d in upcoming]
        assert len(set(months)) == len(months), f"two meetings in one month seen on {t:%Y-%m-%d}"
        r_before, cum = r.r0, 0.0
        for d, j in zip(upcoming, months):
            n_days = calendar.monthrange(d.year, d.month)[1]
            if d.day == n_days:
                r_a = np.nan
            elif j == 0:  # current month: days before t are realized EFFR prints (eq. 8)
                realized = effr_daily[t.replace(day=1): t - pd.Timedelta(days=1)]
                assert len(realized) == t.day - 1, f"gap in EFFR before {t:%Y-%m-%d}"
                r_a = (n_days * rbar[0] - realized.sum() - (d.day - len(realized)) * r_before) / (n_days - d.day)
            else:  # eq. 9
                r_a = (n_days * rbar[j] - d.day * r_before) / (n_days - d.day)
            r_b = rbar[j + 1] if j + 1 <= 12 and j + 1 not in months else np.nan  # eq. 10
            r_after = r_a if np.isnan(r_b) else r_b
            if np.isnan(r_after):
                break
            x = (r_after - r_before) / STEP
            cum += x
            rows.append(dict(date=t, meeting_date=d, months_ahead=j, method="A" if np.isnan(r_b) else "B",
                             r_before=r_before, r_after=r_after, x=x, cum_x=cum, gap_a_b_bp=(r_a - r_b) * 100))
            r_before = r_after
    return pd.DataFrame(rows)


@app.function(hide_code=True)
def bucket_prob(x, lo, hi):
    """Probability that the move lands in [lo, hi] steps when x splits over the two nearest steps (Step 7)."""
    n = np.floor(x)
    p = x - n
    return np.where((lo <= n) & (n <= hi), 1 - p, 0.0) + np.where((lo <= n + 1) & (n + 1 <= hi), p, 0.0)


@app.function(hide_code=True)
def move_scale(moves):
    """Color scale limited to the outcomes present, in cut-to-hike order."""
    present = [m for m in MOVES if m in set(moves)]
    return alt.Scale(domain=present, range=[MOVE_COLORS[m] for m in present])


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

    rows_of = [np.flatnonzero(g == k) for k in range(g.max() + 1)]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        rows = np.concatenate([rows_of[k] for k in rng.integers(0, len(rows_of), len(rows_of))])
        r = fit_logit(X[rows], y[rows])
        if r is not None:
            boots.append(r[0])
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return dict(
        forecasts=len(y), clusters=len(rows_of),
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
def source_reliability_chart(rel, size, size_title, title):
    """Reliability diagram with one color per source (Kalshi, Futures); point size is `size`."""
    scale = alt.Scale(domain=[-0.03, 1.03], nice=False)
    diagonal = alt.Chart(pd.DataFrame({"p": [0, 1]})).mark_line(color="#9a9994", strokeDash=[4, 4]).encode(
        x="p:Q", y="p:Q")
    points = alt.Chart(rel).mark_circle(opacity=0.9, stroke="#fcfcfb", strokeWidth=2).encode(
        x=alt.X("mean_p:Q", title="Predicted", axis=alt.Axis(format="%"), scale=scale),
        y=alt.Y("observed:Q", title="Observed", axis=alt.Axis(format="%"), scale=scale),
        color=alt.Color("source:N", scale=alt.Scale(domain=["Kalshi", "Futures"], range=["#2a78d6", "#eb6834"]),
                        title="Source"),
        size=alt.Size(f"{size}:Q", title=size_title, scale=alt.Scale(range=[60, 600])),
        tooltip=["source:N", alt.Tooltip("bin_from:Q", format=".1f", title="Bin from"),
                 alt.Tooltip(f"{size}:Q", format=",.0f", title=size_title), "meetings:Q",
                 alt.Tooltip("mean_p:Q", format=".3f", title="Mean forecast"),
                 alt.Tooltip("observed:Q", format=".3f", title="Share YES")],
    )
    return (diagonal + points).properties(width=380, height=380, title=title)


@app.function(hide_code=True)
def load_intraday_bars(path, cache):
    """Long table of the Bloomberg 1-minute bars: contract_month, side (bid/ask), ts (end of the bar, ET), price.

    Bloomberg labels a bar by the minute it starts and Kalshi labels a candle by the minute it ends, so the
    Bloomberg label is moved one minute later, as in 04_leadlag_fedfunds_kalshi.py.
    """
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        bars = pd.read_parquet(cache)
    else:
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        parts = []
        for name in wb.sheetnames[2:]:  # README and Settings first, then one sheet per contract month
            rows = list(wb[name].iter_rows(min_row=6, max_col=5, values_only=True))
            for side, (i, j) in {"bid": (0, 1), "ask": (3, 4)}.items():
                d = pd.DataFrame([(r[i], r[j]) for r in rows if r[i] is not None], columns=["ts", "price"])
                parts.append(d.assign(contract_month=name, side=side))
        bars = pd.concat(parts, ignore_index=True)
        bars["price"] = pd.to_numeric(bars.price, errors="coerce")
        bars.dropna().to_parquet(cache)
    bars["ts"] = pd.to_datetime(bars.ts).dt.tz_localize(TZ) + pd.Timedelta(minutes=1)
    return bars


@app.function(hide_code=True)
def cme_minutes(start, end):
    """Every minute CME Globex trades fed funds futures: Sunday 18:00 to Friday 17:00 ET, paused 17:00-18:00."""
    grid = pd.date_range(start.ceil("min"), end, freq="min", tz=TZ)
    wd, hr = grid.dayofweek, grid.hour
    return grid[~((wd == 5) | ((wd == 4) & (hr >= 17)) | ((wd == 6) & (hr < 18)) | (hr == 17))]


@app.function(hide_code=True)
def next_meeting_move(prices, times, sessions, meetings, daily):
    """Futures-implied expected move (bp) at the next FOMC meeting, one value per row: Steps 1-5 of the guide
    applied to whatever prices the rows hold. Same as in 04_leadlag_fedfunds_kalshi.py.

    prices: one column per contract month ("YYYY-MM"), quoted as 100 minus the rate.
    times: naive ET timestamps; the next meeting is the first decision (14:00 ET) after each.
    sessions: the date each row's r0 and realized EFFR are taken from.
    daily: effr and r0 by calendar date, forward-filled.
    """
    meeting_idx = pd.DatetimeIndex(meetings)
    decision = meeting_idx + pd.Timedelta(hours=DECISION_HOUR)
    D = pd.DatetimeIndex(meeting_idx[np.searchsorted(decision, times, side="right")])
    sessions = pd.DatetimeIndex(sessions)
    r_before = daily.r0.reindex(sessions).to_numpy()
    cum_effr = daily.effr.cumsum()
    with_meeting = {(d.year, d.month) for d in meetings}
    move = np.full(len(times), np.nan)
    method = np.full(len(times), "", dtype=object)
    for d in D.unique():
        sel = np.flatnonzero(D == d)
        after = d + pd.offsets.MonthBegin(1)
        after_key, own_key = f"{after.year}-{after.month:02d}", f"{d.year}-{d.month:02d}"
        if (after.year, after.month) not in with_meeting and after_key in prices:
            r_after = 100 - prices[after_key].to_numpy()[sel]
            method[sel] = "B"
        elif own_key in prices:
            n_days = calendar.monthrange(d.year, d.month)[1]
            rbar = 100 - prices[own_key].to_numpy()[sel]
            s = sessions[sel]
            same = (s.year == d.year) & (s.month == d.month)
            month_start = pd.Timestamp(d.year, d.month, 1)
            # EFFR already printed this month: days 1 .. session day - 1 (eq. 8 of the guide)
            before_start = cum_effr.get(month_start - pd.Timedelta(days=1), 0.0)
            realized = np.where(same, cum_effr.reindex(s - pd.Timedelta(days=1)).to_numpy() - before_start, 0.0)
            realized = np.where(same & (s.day == 1), 0.0, realized)
            n_real = np.where(same, s.day - 1, 0)
            r_after = (n_days * rbar - realized - (d.day - n_real) * r_before[sel]) / (n_days - d.day)
            method[sel] = "A"
        else:
            continue
        move[sel] = (r_after - r_before[sel]) * 100
    return pd.DataFrame({"meeting": D, "method": method, "futures_bp": move})


@app.cell(hide_code=True)
def intro():
    mo.md(r"""
    # 03 · Fed funds futures vs Kalshi

    Market-implied probabilities of each FOMC outcome from fed funds futures, put on the same contracts as
    Kalshi's `KXFEDDECISION` buckets, following `Literature/fed_funds_implied_probabilities.pdf` ("the guide").

    **Method in one paragraph.** Contract FF$n$ settles on the average EFFR of month $n-1$, so
    $\bar r_j = 100 - P_{j+1}$. The rate can only jump at meetings, so each month is a day-weighted blend of the
    rates before and after its meeting; un-blending gives the expected rate after each meeting and the expected
    move $x_k$ in 25bp steps. One number per meeting identifies one probability: with $n = \lfloor x \rfloor$ and
    $p = x - n$, the move is $n$ steps with probability $1 - p$ and $n + 1$ steps with probability $p$.
    Each Kalshi bucket then gets the mass on the steps it pays on.

    **What the futures cannot do.** Every bucket other than the two around $x$ gets exactly 0. "Cut >25" is
    non-zero only when more than one full cut is priced, so the >25bp buckets are 0 *by construction* on most
    days. Compare Hold and ±25 directly; read the tails as Kalshi against a model without tails.

    **Data.** Futures, EFFR, target range and FOMC dates from `data/KalshiProjectFedFunds.xlsx` (Bloomberg,
    daily settlements). Kalshi contracts from `data/kalshi_markets.parquet`, quotes from
    `data/cleaned/kalshi_candles_1m.parquet`, queries in `sql/fedfunds_vs_kalshi.sql`.
    """)
    return


@app.cell(hide_code=True)
def futures_data():
    mo.stop(not FF_XLSX.exists(), mo.md(f"**Missing `{FF_XLSX.relative_to(ROOT)}`.**"))
    fomc = load_fomc(FF_XLSX)
    meeting_dates = fomc.meeting_date.tolist()
    futures, stale_dates = load_futures(FF_XLSX)
    futures = add_starting_rate(futures, meeting_dates)
    implied = implied_moves(futures, meeting_dates)
    implied
    return fomc, futures, implied, meeting_dates, stale_dates


@app.cell(hide_code=True)
def kalshi_data():
    mo.stop(
        not CLEAN_CANDLES.exists(),
        mo.md(f"**Missing `{CLEAN_CANDLES.relative_to(ROOT)}`.** Until the cleaning step exists, "
              "copy `data/kalshi_candles_1m.parquet` there."),
    )
    con = duckdb.connect()
    # The views read 'data/...' relative to the repo root; anchor them so the notebook runs from any directory.
    con.execute((ROOT / "sql/kalshi_views.sql").read_text().replace("'data/", f"'{ROOT}/data/"))
    con.execute(f"CREATE VIEW candles_clean AS SELECT * FROM read_parquet('{CLEAN_CANDLES}')")

    # Named queries from fedfunds_vs_kalshi.sql, split on their "-- name: ..." headers.
    _parts = re.split(r"^-- name: (\w+)\s*$", (ROOT / "sql/fedfunds_vs_kalshi.sql").read_text(), flags=re.M)[1:]
    queries = dict(zip(_parts[::2], _parts[1::2]))

    contracts = con.sql("""
        SELECT ticker, event_ticker, meeting_date, move, move_bps, status, result,
               CAST(open_time AT TIME ZONE 'America/New_York' AS DATE) AS open_date
        FROM markets
        -- FEDDECISION-24JAN is an untraded duplicate of FEDDECISION-24JAN31 that closed on Jan 26.
        QUALIFY sum(volume) OVER (PARTITION BY event_ticker) > 0
    """).df()
    contracts["meeting_date"] = pd.to_datetime(contracts.meeting_date)
    contracts["open_date"] = pd.to_datetime(contracts.open_date)
    contracts["y"] = (contracts.result == "yes").astype(int)
    assert contracts.move.isin(MOVE_STEPS).all(), "a Kalshi bucket with no step range"
    events_with_quotes = set(con.sql("SELECT DISTINCT event_ticker FROM candles_clean").df().event_ticker)
    return con, contracts, events_with_quotes, queries


@app.cell(hide_code=True)
def lines(contracts, implied, meeting_dates):
    _missing = set(contracts.meeting_date) - set(meeting_dates)
    assert not _missing, f"Kalshi meetings missing from FOMC_Dates: {sorted(_missing)}"

    # One row per Kalshi contract per futures trading day, from the day the contract was listed to the last
    # trading day before the decision (on decision day the futures settle after the announcement).
    futures_lines = contracts.merge(implied, on="meeting_date")
    futures_lines = futures_lines[futures_lines.date >= futures_lines.open_date].copy()
    _lo, _hi = zip(*futures_lines.move.map(MOVE_STEPS))
    futures_lines["p_ff"] = bucket_prob(futures_lines.x.to_numpy(), np.array(_lo), np.array(_hi))
    futures_lines["zero_by_construction"] = futures_lines.p_ff == 0
    futures_lines["days_to_meeting"] = (futures_lines.meeting_date - futures_lines.date).dt.days
    futures_lines = futures_lines[["ticker", "event_ticker", "meeting_date", "move", "move_bps", "status", "y",
                                   "date", "days_to_meeting", "method", "x", "p_ff", "zero_by_construction"]]
    futures_lines
    return (futures_lines,)


@app.cell(hide_code=True)
def checks_md():
    mo.md(r"""
    ## 1 · Checks on the futures side

    The validation list of the guide (Section 14), run on the whole 2021+ sample.

    - **Cleaning.** Rows dropped as holiday copies, and the roll check: on the first trading day of each month,
      FF1 should be close to the previous day's FF2 (both are the same contract).
    - **Worked example.** 15 September 2026 against the guide's own numbers.
    - **Probabilities.** On every date the buckets of a meeting sum to 1.
    - **Day before.** The probability given to the decision that happened, on the last trading day before each
      meeting.
    - **Method A vs B.** Where both are available they should agree within a few bp; the gap shows how much
      un-blending amplifies noise when the meeting is late in the month.
    """)
    return


@app.cell(hide_code=True)
def cleaning(futures, stale_dates):
    _prev = futures.shift(1)
    _new_month = futures.date.dt.to_period("M") != _prev.date.dt.to_period("M")
    _roll = pd.DataFrame({
        "first trading day": futures.date[_new_month],
        "FF1": futures.FF1[_new_month],
        "previous FF2": _prev.FF2[_new_month],
    }).dropna()
    _roll["gap (bp)"] = ((_roll.FF1 - _roll["previous FF2"]) * 100).round(2)
    mo.vstack([
        mo.md(f"**{len(stale_dates)}** rows dropped as copies of the previous day (holidays carried forward). "
              f"Largest roll gap after cleaning: **{_roll['gap (bp)'].abs().max():.1f} bp** "
              f"over {len(_roll)} month starts."),
        mo.ui.table(_roll.sort_values("gap (bp)", key=abs, ascending=False).head(10), selection=None),
    ])
    return


@app.cell(hide_code=True)
def worked_example(implied):
    _guide = pd.DataFrame({"meeting_date": pd.to_datetime(["2026-09-16", "2026-10-28", "2026-12-09"]),
                           "guide x": [0.921, 0.479, 0.705], "guide method": ["A", "B", "A"]})
    _ours = implied[implied.date == "2026-09-15"]
    mo.vstack([
        mo.md("**Worked example, 15 Sep 2026.** The guide assumes every September day printed 3.63 and a flat "
              "starting rate; small differences come from the actual EFFR prints and the rolling spread."),
        mo.ui.table(_guide.merge(_ours[["meeting_date", "method", "r_before", "r_after", "x"]], on="meeting_date")
                    .round(4), selection=None),
    ])
    return


@app.cell(hide_code=True)
def sums_and_day_before(fomc, futures_lines, implied):
    _sums = futures_lines.groupby(["event_ticker", "date"]).p_ff.sum()
    _bad_sums = (_sums - 1).abs().gt(1e-9).sum()

    _last = implied.loc[implied.groupby("meeting_date").date.idxmax()].merge(
        fomc[["meeting_date", "decision_bp"]], on="meeting_date")
    _last = _last[_last.decision_bp.notna()].copy()
    _k = _last.decision_bp / 25
    _last["p_decision"] = bucket_prob(_last.x.to_numpy(), _k.to_numpy(), _k.to_numpy())
    mo.vstack([
        mo.md(f"**Bucket sums.** {len(_sums):,} (meeting, date) pairs; {_bad_sums} do not sum to 1."),
        mo.md(f"**Day before.** Over {len(_last)} meetings since 2021 the futures gave the actual decision a median "
              f"probability of **{_last.p_decision.median():.2f}**; {(_last.p_decision >= 0.8).sum()} meetings "
              f"at 0.8 or more. Lowest five:"),
        mo.ui.table(_last.nsmallest(5, "p_decision")[["date", "meeting_date", "method", "x", "decision_bp",
                                                      "p_decision"]].round(3), selection=None),
    ])
    return


@app.cell(hide_code=True)
def method_gap(implied):
    _both = implied.dropna(subset=["gap_a_b_bp"]).copy()
    _both["day of month"] = _both.meeting_date.dt.day
    _q = _both.gap_a_b_bp.abs().quantile([0.5, 0.9, 0.99])
    _chart = alt.Chart(_both).mark_circle(size=40, opacity=0.4, color="#2a78d6").encode(
        x=alt.X("day of month:Q", title="Day of month of the meeting", scale=alt.Scale(domain=[1, 31])),
        y=alt.Y("gap_a_b_bp:Q", title="Method A − Method B (bp)"),
        tooltip=["date:T", "meeting_date:T", alt.Tooltip("gap_a_b_bp:Q", format=".1f", title="Gap (bp)")],
    ).properties(width="container", height=260, title="Method A vs B: the gap grows as the meeting moves late in the month")
    mo.vstack([
        mo.md(f"**Method A vs B.** {len(_both):,} (date, meeting) pairs where both apply. Absolute gap: median "
              f"{_q[0.5]:.1f} bp, 90th percentile {_q[0.9]:.1f} bp, 99th {_q[0.99]:.1f} bp. Method B is used for "
              f"all of these."),
        _chart,
    ])
    return


@app.cell(hide_code=True)
def path_md():
    mo.md("""
    ## 2 · Probability path for one meeting

    Solid lines: Kalshi, the bid/ask mid at the top of every hour. Dots on dashed steps: the futures, one value
    per trading day at settlement (about 15:00 ET), from the day the contract was listed to the day before the
    decision. Meetings after September 2025 have futures lines only until the Kalshi candles are refetched.
    A wide Kalshi book has a mid that says little (a 0¢/37¢ book gives 18.5¢); tick the box to drop those hours.
    """)
    return


@app.cell(hide_code=True)
def path_controls(events_with_quotes, fomc, futures_lines):
    _events = (futures_lines.groupby("event_ticker").meeting_date.first().sort_values().reset_index()
               .merge(fomc[["meeting_date", "decision_bp"]], on="meeting_date", how="left"))

    def _label(r):
        decision = "open" if pd.isna(r.decision_bp) else f"{r.decision_bp:+.0f}bp"
        quotes = "" if r.event_ticker in events_with_quotes else " · futures only"
        return f"{r.meeting_date:%Y-%m-%d} · {decision}{quotes}"

    _options = {_label(r): r.event_ticker for r in _events.itertuples()}
    meeting_picker = mo.ui.dropdown(
        options=_options,
        value=next(k for k, v in _options.items() if v == "FEDDECISION-24SEP"),
        label="Meeting",
    )
    drop_wide_path = mo.ui.checkbox(value=False, label="Drop Kalshi hours with a spread above 20¢")
    mo.hstack([meeting_picker, drop_wide_path], justify="start", gap=2)
    return drop_wide_path, meeting_picker


@app.cell(hide_code=True)
def path(con, drop_wide_path, futures_lines, meeting_picker, queries):
    _kalshi = con.execute(queries["hourly_path"], {"event_ticker": meeting_picker.value}).df()
    _kalshi["ts"] = pd.to_datetime(_kalshi.ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    if drop_wide_path.value:
        _kalshi = _kalshi[_kalshi.spread <= 0.20]
    _kalshi["source"] = "Kalshi"
    _ff = futures_lines[futures_lines.event_ticker == meeting_picker.value].copy()
    _ff["ts"] = _ff.date + pd.Timedelta(hours=SNAPSHOT_HOUR_ET)
    _ff["source"] = "Fed funds futures"

    _hover = alt.selection_point(fields=["ts"], nearest=True, on="pointerover", clear="pointerout", empty=False)
    _color = alt.Color("move:N", scale=move_scale(_ff.move), title="Outcome")
    _x = alt.X("ts:T", title=None)
    _yscale = alt.Scale(domain=[0, 1])
    _kalshi_lines = alt.Chart(_kalshi).mark_line(strokeWidth=2).encode(
        x=_x, y=alt.Y("mid:Q", scale=_yscale, title="Probability"), color=_color, detail="ticker:N")
    _ff_base = alt.Chart(_ff).encode(x=_x, y=alt.Y("p_ff:Q", scale=_yscale, title="Probability"), color=_color,
                                     detail="ticker:N")
    _ff_steps = _ff_base.mark_line(strokeWidth=2, strokeDash=[4, 4], interpolate="step-after")
    _ff_dots = _ff_base.mark_point(filled=True, size=64, stroke="#fcfcfb", strokeWidth=2).encode(
        tooltip=[alt.Tooltip("date:T", title="Date"), "move:N",
                 alt.Tooltip("days_to_meeting:Q", title="Days to meeting"),
                 alt.Tooltip("x:Q", format=".2f", title="Expected move (steps)"),
                 alt.Tooltip("p_ff:Q", format=".3f", title="Futures probability"),
                 alt.Tooltip("method:N", title="Method")],
    ).add_params(_hover)
    _rule = alt.Chart(_ff).mark_rule(color="#9a9994").encode(x=_x).transform_filter(_hover)
    (_kalshi_lines + _ff_steps + _ff_dots + _rule).properties(
        width="container", height=380,
        title=f"{meeting_picker.selected_key} · solid: Kalshi hourly mid · dashed with dots: futures, daily",
    )
    return


@app.cell(hide_code=True)
def matched_md():
    mo.md("""
    ## 3 · Matched daily panel

    Every futures day of every Kalshi contract, with the Kalshi mid in force at 15:00 ET that day, the time the
    futures settle. Rows before a contract's first quote drop out. `zero by construction` is the share of rows
    where the futures give the bucket exactly 0; for the >25bp buckets that is almost every row.
    """)
    return


@app.cell(hide_code=True)
def matched(con, futures_lines, queries):
    con.register("snapshot_grid", futures_lines[["ticker", "date"]].assign(date=lambda d: d.date.dt.date))
    _snaps = con.execute(queries["snapshots"], {"snapshot_hour": SNAPSHOT_HOUR_ET}).df()
    _snaps["date"] = pd.to_datetime(_snaps.date)
    matched = futures_lines.merge(_snaps, on=["ticker", "date"])
    matched["diff"] = matched.kalshi_p - matched.p_ff
    matched["horizon"] = pd.cut(matched.days_to_meeting, [0, 1, 7, 30, 60, 120, 1000],
                                labels=["1", "2–7", "8–30", "31–60", "61–120", "121+"])
    matched["group"] = np.where(matched.move.isin(CORE_MOVES), "Hold and ±25", "Tails (>25bp)")

    _summary = (
        matched.groupby(["group", "horizon"], observed=True)
        .agg(rows=("diff", "size"), meetings=("event_ticker", "nunique"),
             mean_abs_diff=("diff", lambda d: d.abs().mean()), mean_diff=("diff", "mean"),
             corr=("diff", lambda d: matched.loc[d.index, ["kalshi_p", "p_ff"]].corr().iloc[0, 1]),
             zero_by_construction=("zero_by_construction", "mean"))
        .reset_index()
    )
    mo.vstack([
        mo.md(f"**{len(matched):,}** matched contract-days over **{matched.event_ticker.nunique()}** meetings. "
              "`mean_diff` is Kalshi minus futures: positive means Kalshi prices the bucket higher."),
        mo.ui.table(_summary.round(3), selection=None),
    ])
    return (matched,)


@app.cell(hide_code=True)
def scores_md():
    mo.md(r"""
    ## 4 · Scores at fixed horizons

    The horizons of `02_calibration_kalshi.py`: for each settled contract and horizon $h$, the last futures
    trading day on or before the meeting minus $h$ days, and the Kalshi mid at 15:00 ET that same day. Only
    contracts with both forecasts are scored, so the two sources are always scored on the same rows. Futures
    trade on weekdays only: for a Wednesday meeting the 3- and 5-day horizons both fall on the Friday before,
    so those two columns repeat each other (02 reads Kalshi on weekends too, so its numbers differ there).

    Brier $(p - Y)^2$ and log loss $-[Y \ln p + (1 - Y)\ln(1 - p)]$, lower is better. Both sources are clipped to
    $[0.005, 0.995]$ first: the futures give many buckets exactly 0, and one of them settling YES would make the
    futures' log loss infinite. Kalshi's mid already sits inside that range. `futures clipped` counts the futures
    forecasts the floor changed.
    """)
    return


@app.cell(hide_code=True)
def score_controls():
    score_group = mo.ui.dropdown(options=["Hold and ±25", "Tails (>25bp)", "All buckets"],
                                 value="Hold and ±25", label="Buckets")
    score_group
    return (score_group,)


@app.cell(hide_code=True)
def scores(matched, score_group):
    _settled = matched[matched.status == "finalized"].sort_values("date")
    _grid = (_settled[["ticker", "meeting_date"]].drop_duplicates()
             .merge(pd.DataFrame({"ttm": TTMS}), how="cross"))
    _grid["target"] = _grid.meeting_date - pd.to_timedelta(_grid.ttm, unit="D")
    scored = pd.merge_asof(_grid.sort_values("target"), _settled.drop(columns="meeting_date"),
                           left_on="target", right_on="date", by="ticker", direction="backward")
    # A target on a weekend or holiday takes the last trading day before it; a longer gap means the contract
    # had no Kalshi quote yet at that horizon.
    scored = scored[(scored.target - scored.date).dt.days.between(0, 4)].copy()
    if score_group.value != "All buckets":
        scored = scored[scored.group == score_group.value]

    _long = []
    for _name, _col in [("Kalshi", "kalshi_p"), ("Futures", "p_ff")]:
        _p = scored[_col].clip(P_FLOOR, 1 - P_FLOOR)
        _long.append(scored.assign(
            source=_name, p=_p, clipped=_p != scored[_col],
            brier=(_p - scored.y) ** 2, log_loss=-(scored.y * np.log(_p) + (1 - scored.y) * np.log(1 - _p))))
    scores_long = pd.concat(_long)

    def _table(metric):
        t = scores_long.pivot_table(index="source", columns="ttm", values=metric, aggfunc="mean")
        t.columns = [f"{h}d" for h in t.columns]
        return mo.ui.table(t.round(4).reset_index(), selection=None)

    _counts = scored.groupby("ttm").agg(forecasts=("ticker", "size"), meetings=("event_ticker", "nunique"),
                                         settled_yes=("y", "sum"))
    _counts["futures clipped"] = scores_long[scores_long.source == "Futures"].groupby("ttm").clipped.sum()
    _counts.index = [f"{h}d" for h in _counts.index]
    mo.ui.tabs({
        "Brier": _table("brier"),
        "Log loss": _table("log_loss"),
        "Forecasts per horizon": mo.ui.table(_counts.T.reset_index(names=""), selection=None),
    })
    return (scores_long,)


@app.cell(hide_code=True)
def reliability_md():
    mo.md("""
    ### Reliability, both sources on the same forecasts

    Share settled YES against the mean forecast in 10¢ bins, horizons 1–30 days pooled, after the clip above.
    Futures forecasts at exactly 0 or 1 all land in the end bins. Forecasts of the same meeting are correlated,
    so read each point's `meetings` count, not its forecast count.
    """)
    return


@app.cell(hide_code=True)
def reliability(scores_long):
    _s = scores_long[scores_long.ttm <= 30]
    _rel = (
        _s.assign(bin_from=np.minimum(np.floor(_s.p * 10) / 10, 0.9))
        .groupby(["source", "bin_from"], as_index=False)
        .agg(forecasts=("p", "size"), meetings=("event_ticker", "nunique"), mean_p=("p", "mean"),
             observed=("y", "mean"))
    )
    mo.hstack([
        source_reliability_chart(_rel, "forecasts", "Forecasts", "Reliability · 1–30 days, pooled"),
        mo.ui.table(_rel.round(4), selection=None),
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def minute_reliability_md():
    mo.md("""
    ### Reliability every minute, both sources (descriptive)

    **Descriptive only, not part of the tests.** The same diagram from 1-minute data instead of one 15:00 ET
    snapshot a day, over the window of the Bloomberg intraday bars (`data/bloomberg/ff_intraday.xlsx`, the data of
    `04_leadlag_fedfunds_kalshi.py`; Bloomberg keeps about 140 business days of them). Every CME minute (Sunday
    18:00 to Friday 17:00 ET, paused 17:00–18:00), each settled contract of the **next** meeting gets two forecasts:

    - **Futures:** the expected move at the next meeting from the bid/ask mids of the contracts in force that
      minute (the method of 04, the guide's Steps 1–5), split over the buckets as above.
    - **Kalshi:** the contract's raw bid/ask mid in force at the end of that minute (candles carried forward).

    Each minute in which both exist counts once, so point size is contract-hours. The window runs up to the 2pm
    decision, so the morning of meeting day is included. The clip and the buckets picked above apply; 10¢ bins.
    Only the few meetings decided inside the window are scored: more minutes are not more independent evidence,
    so read each bin's `meetings`, not its hours.
    """)
    return


@app.cell(hide_code=True)
def minute_panel(con, contracts, futures, meeting_dates, queries):
    mo.stop(not INTRADAY_XLSX.exists(),
            mo.md(f"**Missing `{INTRADAY_XLSX.relative_to(ROOT)}`.** Refresh "
                  "`data/KalshiProjectFFIntraday_template.xlsx` on a Bloomberg Terminal and save it there."))
    _bars = load_intraday_bars(INTRADAY_XLSX, INTRADAY_CACHE)
    _days = pd.date_range(futures.date.min(), _bars.ts.max().tz_localize(None).normalize(), freq="D")
    # Weekends and the days after the workbook's last row carry the last EFFR and r0 forward.
    _daily = futures.set_index("date")[["effr", "r0"]].reindex(_days).ffill()
    _grid = cme_minutes(_bars.ts.min(), _bars.ts.max())
    _wide = _bars.pivot_table(index="ts", columns=["contract_month", "side"], values="price", aggfunc="last")
    _wide = _wide.reindex(_wide.index.union(_grid)).sort_index().ffill().reindex(_grid)
    _prices = pd.DataFrame({m: (_wide[(m, "bid")] + _wide[(m, "ask")]) / 2 for m in _wide.columns.levels[0]})
    # A session runs 18:00 to 17:00 ET and is dated by the day it ends.
    _session = (_grid + pd.Timedelta(hours=6)).tz_localize(None).normalize()
    _move = next_meeting_move(_prices, _grid.tz_localize(None), _session, meeting_dates, _daily)

    _settled = contracts[(contracts.status == "finalized") & contracts.meeting_date.isin(_move.meeting.unique())]
    _k = con.execute(queries["minute_mids"], {"tickers": _settled.ticker.tolist()}).df()
    _k["ts"] = pd.to_datetime(_k.ts, utc=True).dt.tz_convert(TZ)
    _mids = _k.pivot_table(index="ts", columns="ticker", values="mid", aggfunc="last")
    _mids = _mids.reindex(_mids.index.union(_grid)).sort_index().ffill().reindex(_grid)

    _parts = []
    for _c in _settled.itertuples():
        _sel = (_move.meeting == _c.meeting_date).to_numpy()
        _lo, _hi = MOVE_STEPS[_c.move]
        _parts.append(pd.DataFrame({
            "ts": _grid[_sel], "ticker": _c.ticker, "event_ticker": _c.event_ticker,
            "meeting_date": _c.meeting_date, "move": _c.move, "y": _c.y,
            "p_ff": bucket_prob(_move.futures_bp.to_numpy()[_sel] / 25, _lo, _hi),
            "kalshi_p": _mids[_c.ticker].to_numpy()[_sel] if _c.ticker in _mids else np.nan,
        }))
    minute_panel = pd.concat(_parts, ignore_index=True).dropna(subset=["p_ff", "kalshi_p"])
    minute_panel["group"] = np.where(minute_panel.move.isin(CORE_MOVES), "Hold and ±25", "Tails (>25bp)")
    return (minute_panel,)


@app.cell(hide_code=True)
def minute_reliability(minute_panel, score_group):
    _m = minute_panel if score_group.value == "All buckets" else minute_panel[minute_panel.group == score_group.value]
    _long = []
    for _name, _col in [("Kalshi", "kalshi_p"), ("Futures", "p_ff")]:
        _p = _m[_col].clip(P_FLOOR, 1 - P_FLOOR)
        _long.append(_m.assign(source=_name, p=_p, brier=(_p - _m.y) ** 2,
                               log_loss=-(_m.y * np.log(_p) + (1 - _m.y) * np.log(1 - _p))))
    _long = pd.concat(_long)
    _rel = (
        _long.assign(bin_from=np.minimum(np.floor(_long.p * 10) / 10, 0.9))
        .groupby(["source", "bin_from"], as_index=False)
        .agg(minutes=("p", "size"), contracts=("ticker", "nunique"), meetings=("event_ticker", "nunique"),
             mean_p=("p", "mean"), observed=("y", "mean"))
        .assign(contract_hours=lambda d: d.minutes / 60)
        [["source", "bin_from", "contract_hours", "contracts", "meetings", "mean_p", "observed"]]
    )
    _scores = (_long.groupby("source", sort=False)
               .agg(minutes=("p", "size"), meetings=("event_ticker", "nunique"),
                    brier=("brier", "mean"), log_loss=("log_loss", "mean"))
               .assign(contract_hours=lambda d: d.minutes / 60).drop(columns="minutes").reset_index())
    _meetings = ", ".join(f"{d:%Y-%m-%d}" for d in sorted(_m.meeting_date.unique()))
    mo.vstack([
        mo.md(f"**{len(_m):,}** contract-minutes, {_m.ts.min():%Y-%m-%d %H:%M} to {_m.ts.max():%Y-%m-%d %H:%M} ET; "
              f"meetings {_meetings}. Scores are averaged over minutes."),
        mo.hstack([
            source_reliability_chart(_rel, "contract_hours", "Contract-hours", "Reliability · every minute"),
            mo.ui.tabs({
                "Every bin": mo.ui.table(_rel.round(4), selection=None),
                "Scores": mo.ui.table(_scores.round(4), selection=None),
            }),
        ], widths=[1, 1]),
    ])
    return


@app.cell(hide_code=True)
def regression_md():
    mo.md(r"""
    ### Calibration regression, both sources on the same forecasts

    The test of section 6 of `02_calibration_kalshi.py`, run on each source: the outcome is regressed on the forecast
    in log odds, $\mathbb{P}(Y = 1 \mid p) = \Lambda(\alpha + \beta \operatorname{logit} p)$, and calibrated forecasts
    have $\alpha = 0$ and $\beta = 1$ ($\beta > 1$: under-confident, prices too close to 50%; $\beta < 1$:
    over-confident). The rows are the scored forecasts above, for the buckets picked there, after the clip to
    $[0.005, 0.995]$, grouped into the horizon groups of notebook 02. Standard errors are clustered by meeting, and
    meetings are resampled (999 times) for the 95% intervals and `p_joint_boot`.

    Futures forecasts clipped from exactly 0 or 1 sit at the clip with the outcome they predicted, so they barely move
    the fit; dropping them leaves the estimates unchanged. The Kalshi price here is the raw mid at 15:00 ET, not
    notebook 02's normalized $q$ at 16:00, so its estimates differ somewhat from 02's.
    """)
    return


@app.cell(hide_code=True)
def regression(scores_long):
    _groups = {"1–21 days (main)": [h for h in TTMS if h <= 21], "30 days": [30], "45–60 days": [45, 60]}
    _rows = []
    for _source in ["Kalshi", "Futures"]:
        for _group, _hs in _groups.items():
            _x = scores_long[(scores_long.source == _source) & scores_long.ttm.isin(_hs)]
            _rows.append({"source": _source, "horizon": _group,
                          **calibration_regression(_x.y, _x.p.to_numpy(), _x.event_ticker)})
    calibration_fit = pd.DataFrame(_rows)
    mo.vstack([
        calibration_curves_chart(calibration_fit[calibration_fit.horizon == "1–21 days (main)"], "source",
                                 ["Kalshi", "Futures"], ["#2a78d6", "#eb6834"], "Source",
                                 "Fitted calibration curves · 1–21 days"),
        mo.ui.table(calibration_fit.round(3), selection=None),
    ])
    return


@app.cell(hide_code=True)
def regime_md():
    mo.md(r"""
    ### Before and after Kalshi's liquidity break, both sources

    Section 7 of `02_calibration_kalshi.py` finds how Kalshi's calibration changed when its volume jumped about 15-fold
    in May 2025. The two regimes are also different policy periods (mostly holds, then a cutting cycle), so a change
    there could come from the meetings rather than from liquidity. The futures, whose liquidity did not change, are
    the control: fitted on the same forecasts, a change in Kalshi's curve that the futures do not share points to
    Kalshi's market; a change both share points to the meetings. 1–21 days, the buckets picked above; standard errors
    clustered by meeting, bootstrap within each regime. 16 meetings before, 12 from May 2025.
    """)
    return


@app.cell(hide_code=True)
def regime(scores_long):
    _s = scores_long[scores_long.ttm <= 21]
    _regime = np.where(_s.meeting_date >= LIQUIDITY_BREAK, REGIMES[1], REGIMES[0])
    _fits, _diffs = [], []
    for _source in ["Kalshi", "Futures"]:
        _src = (_s.source == _source).to_numpy()
        for _r in REGIMES:
            _x = _s[_src & (_regime == _r)]
            _fits.append({"source": _source, "regime": _r,
                          **calibration_regression(_x.y, _x.p.to_numpy(), _x.event_ticker)})
        _x = _s[_src]
        _diffs.append({"source": _source, **regime_difference(_x.y, _x.p.to_numpy(), _x.event_ticker,
                                                              _regime[_src] == REGIMES[1])})
    mo.vstack([
        mo.md("**Calibration in each regime**"),
        mo.ui.table(pd.DataFrame(_fits).round(3), selection=None),
        mo.md("**Change from before to after May 2025**"),
        mo.ui.table(pd.DataFrame(_diffs).round(3), selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()
