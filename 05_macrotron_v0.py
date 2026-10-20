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
    import importlib.util
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
    MARKETS_FILE = ROOT / "data/kalshi_markets.parquet"
    TRADES_FILE = ROOT / "data/kalshi_trades.parquet"
    RAW_CANDLES = ROOT / "data/kalshi_candles_1m.parquet"
    CLEAN_CANDLES = ROOT / "data/cleaned/kalshi_candles_1m.parquet"

    SNAPSHOT_HOUR_ET = 15  # futures settle around 15:00 ET; Kalshi is read at the same time (as in 03)
    STEP_BP = 25  # one policy move
    WIDE_SPREAD = 0.20  # a bucket quoted wider than this has a mid that says little
    HORIZONS = [1, 5]  # trading days ahead for the targets
    LIQUIDITY_BREAK = pd.Timestamp("2025-05-01")  # same split as 02 and 03

    # Feature construction, fixed 2026-10-19 with the feature list (section 4), before any results were seen.
    NEWS_TAU_DAYS = 5  # a release's effect decays as exp(-days since / tau): the few-day catch-up in the hypothesis
    MIN_QUOTED_STRIKES = 3  # fewer quoted strikes the evening before a release: no usable implied distribution
    MAX_TRADE_AGE_H = 720  # hours since the last trade are capped at 30 days (and used when nothing traded yet)

    # Validation.
    MIN_TRAIN_MEETINGS = 8  # the expanding window starts testing once this many meetings are in the past
    TUNE_MEETINGS = 3  # the last meetings of each training window choose ridge's alpha
    ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]  # widened in dev run 2: run 1 always picked an end
    FREE = ("gap_bp",)  # never shrunk, so heavy shrinkage pulls ridge toward AR(1), the baseline, not toward 0
    N_BOOT = 999  # meeting-cluster bootstrap draws
    P_CUT = 0.05  # significance level for the feature verdicts
    SIGN_STABILITY = 0.90  # share of bootstrap fits that must agree with the predicted sign

    # Final test set, fixed 2026-10-19 before any results were seen: every meeting from here on (5 settled, Mar-Sep
    # 2026, plus all later ones). Everything in this notebook uses only earlier meetings until SCORE_HOLDOUT is True.
    # Flip it once, after the feature list, alpha grid and decay rules are frozen, and report what comes out.
    HOLDOUT_FROM = pd.Timestamp("2026-03-01")
    SCORE_HOLDOUT = False

    alt.data_transformers.disable_max_rows()

    def load_notebook(stem):
        """Import another notebook's @app.function helpers without copying them, so fixes upstream flow here.
        Filenames start with a digit, so plain `import` can't load them."""
        spec = importlib.util.spec_from_file_location(f"nb_{stem}", ROOT / f"{stem}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    nb03 = load_notebook("03_fedfunds_vs_kalshi")  # Panos: futures-implied moves (the guide's Steps 1-5)


@app.function(hide_code=True)
def read_queries(path):
    """Named queries of a .sql file, split on `-- name:` lines; the text before the first one is under ''."""
    parts = re.split(r"^-- name: (\w+)\s*$", Path(path).read_text(), flags=re.M)
    return {"": parts[0], **dict(zip(parts[1::2], parts[2::2]))}


@app.function(hide_code=True)
def ols(y, X):
    """Least squares with an intercept. Returns coefficients [intercept, slopes...]."""
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


@app.function(hide_code=True)
def ols_cluster(y, X, clusters):
    """OLS with an intercept and standard errors clustered by `clusters` (with the usual small-sample factor).
    Returns (coefficients, standard errors), intercept first."""
    A = np.column_stack([np.ones(len(X)), X])
    bread = np.linalg.inv(A.T @ A)
    beta = bread @ A.T @ y
    u = y - A @ beta
    scores = pd.DataFrame(A * u[:, None]).groupby(np.asarray(clusters)).sum().to_numpy()
    n, k, g = len(y), A.shape[1], len(scores)
    V = bread @ (scores.T @ scores) @ bread * g / (g - 1) * (n - 1) / (n - k)
    return beta, np.sqrt(np.diag(V))


@app.function(hide_code=True)
def cluster_bootstrap(df, stat, cluster="event_ticker", n_boot=N_BOOT, seed=0):
    """Resample whole meetings with replacement and recompute `stat(df)`; returns an array of draws.
    Days of one meeting are strongly correlated, so the meeting, not the day, is the independent unit."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = list(groups)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        draws.append(stat(pd.concat([groups[k] for k in pick], ignore_index=True)))
    return np.array(draws)


@app.function(hide_code=True)
def boot_linear(df, features, target, alpha=0.0, cluster="event_ticker", n_boot=N_BOOT, seed=0):
    """Linear fit (OLS at alpha=0, ridge otherwise; intercept and FREE not penalized) and meeting-bootstrap draws.
    Each meeting's cross-products are summed once, so a draw only adds up resampled meetings: fast and exact."""
    X = np.column_stack([np.ones(len(df)), df[features].to_numpy()])
    y = df[target].to_numpy()
    codes = pd.factorize(df[cluster])[0]
    G, k = codes.max() + 1, X.shape[1]
    XtX, Xty = np.zeros((G, k, k)), np.zeros((G, k))
    for g in range(G):
        Xg = X[codes == g]
        XtX[g], Xty[g] = Xg.T @ Xg, Xg.T @ y[codes == g]
    P = alpha * np.diag([0.0] + [0.0 if f in FREE else 1.0 for f in features])
    point = np.linalg.solve(XtX.sum(0) + P, Xty.sum(0))
    idx = np.random.default_rng(seed).integers(0, G, (n_boot, G))
    draws = np.array([np.linalg.solve(XtX[i].sum(0) + P, Xty[i].sum(0)) for i in idx])
    return point, draws


@app.function(hide_code=True)
def ridge_fit_predict(train, test, features, target, alpha):
    """Ridge in closed form on features standardized with the training rows. Neither the intercept nor the FREE
    features are penalized, so as alpha grows the fit tends to AR(1) on the gap, the baseline to beat."""
    mu, sd = train[features].mean(), train[features].std().replace(0, 1)
    Xtr, Xte = ((train[features] - mu) / sd).to_numpy(), ((test[features] - mu) / sd).to_numpy()
    ybar = train[target].mean()
    P = alpha * np.diag([0.0 if f in FREE else 1.0 for f in features])
    beta = np.linalg.solve(Xtr.T @ Xtr + P, Xtr.T @ (train[target] - ybar).to_numpy())
    return ybar + Xte @ beta


@app.function(hide_code=True)
def ridge_tuned(train, test, features, target, target_date):
    """Ridge with alpha chosen on the last TUNE_MEETINGS training meetings, then refit on all training meetings.
    The inner split respects time the same way the outer one does, so the test meeting never picks alpha."""
    order = train.groupby("event_ticker").meeting_date.first().sort_values()
    tune = train[train.event_ticker.isin(order.index[-TUNE_MEETINGS:])]
    inner = train[train.event_ticker.isin(order.index[:-TUNE_MEETINGS]) & (train[target_date] < tune.date.min())]
    sse = [((tune[target] - ridge_fit_predict(inner, tune, features, target, a)) ** 2).sum() for a in ALPHAS]
    alpha = ALPHAS[int(np.argmin(sse))]
    return ridge_fit_predict(train, test, features, target, alpha), alpha


@app.function(hide_code=True)
def clark_west(actual, pred_small, pred_big, clusters):
    """Clark-West test that a bigger model forecasts better than a smaller model nested in it. Diebold-Mariano is
    biased toward the smaller model in that case; Clark-West adds back the noise the bigger model must estimate.
    Meeting-clustered standard error, one-sided normal p-value (generous with ~15 test meetings)."""
    f = (actual - pred_small) ** 2 - ((actual - pred_big) ** 2 - (pred_small - pred_big) ** 2)
    s = pd.Series(f - f.mean()).groupby(np.asarray(clusters)).sum()
    t = f.mean() / (np.sqrt((s ** 2).sum()) / len(f))
    return t, 0.5 * math.erfc(t / math.sqrt(2))


@app.function(hide_code=True)
def holm(p):
    """Holm adjustment of p-values for testing several features at once."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(max=1)
    out = np.empty_like(adj)
    out[order] = adj
    return out


@app.function(hide_code=True)
def ladder_moments(strikes, mids, step):
    """Mean and standard deviation of an "above k" ladder. The mids are P(print > k); forcing them to fall with k
    gives a survival curve, and each bin between strikes is represented by its midpoint (the tails by half a step
    beyond the end strikes), the same grid `bracket_value` uses for the print."""
    k = np.asarray(strikes, dtype=float)
    s = np.minimum.accumulate(np.clip(np.asarray(mids, dtype=float), 0, 1))
    values = np.r_[k[0] - step / 2, (k[:-1] + k[1:]) / 2, k[-1] + step / 2]
    mass = np.r_[1 - s[0], s[:-1] - s[1:], s[-1]]
    mean = mass @ values
    return mean, np.sqrt(mass @ (values - mean) ** 2)


@app.function(hide_code=True)
def bracket_value(strikes, results, step):
    """The settled print on the ladder's grid: midpoint of (highest strike settled YES, lowest settled NO]."""
    k, r = np.asarray(strikes, dtype=float), np.asarray(results)
    yes, no = k[r == "yes"], k[r == "no"]
    if len(yes) and len(no):
        return (yes.max() + no.min()) / 2
    return yes.max() + step / 2 if len(yes) else no.min() - step / 2


@app.cell(hide_code=True)
def intro():
    mo.md(r"""
    # 05 · MacroTron v0: when Kalshi and fed funds futures disagree

    **Question.** For each upcoming FOMC meeting and each day, futures and Kalshi each imply an expected move at
    that meeting. When they disagree, what explains the gap, and which market moves toward the other?

    **Four statements to test:**

    1. The gap is persistent and has a sign (section 2).
    2. The gap closes mostly by Kalshi moving toward futures, as the minute-level lead–lag in `04` suggests
       (section 3).
    3. Macro news predicts how the gap changes, beyond gap dynamics, microstructure and futures (section 6).
    4. The gap's change is predictable out of sample, beating no-change and mean reversion (section 5).

    The verdicts are collected in section 7.

    **Sources reused, not copied.** Futures-implied moves come from `03_fedfunds_vs_kalshi.py`
    (`implied_moves`, the guide's Steps 1–5); Kalshi quotes at 15:00 ET use 03's `snapshots` query in
    `sql/fedfunds_vs_kalshi.sql`. New queries are in `sql/macrotron_v0.sql`. The method is explained in
    `05_macrotron_v0_guide.md`.
    """)
    return


@app.cell(hide_code=True)
def dev_log():
    mo.md(r"""
    **Development log.** Every change made after seeing development results, so the paper can report how many
    versions were tried. None of them touched the holdout.

    1. *2026-10-20, run 1 → 2.* `effr_drift` had ±25–50bp spikes the day after every decision (the range moves
       on decision day, the EFFR a day later): decision days are now blanked. Ridge always chose an end of the alpha
       grid, and shrank the gap coefficient toward 0, i.e. toward "no change" rather than the AR(1) baseline: the
       grid is wider and `gap_bp` is no longer penalized (`FREE`).
    """)
    return


@app.cell(hide_code=True)
def futures_side():
    mo.stop(not FF_XLSX.exists(), mo.md(f"**Missing `{FF_XLSX.relative_to(ROOT)}`.**"))
    fomc = nb03.load_fomc(FF_XLSX)
    meeting_dates = fomc.meeting_date.tolist()
    futures, _ = nb03.load_futures(FF_XLSX)
    futures = nb03.add_starting_rate(futures, meeting_dates)
    implied = nb03.implied_moves(futures, meeting_dates)
    return futures, implied


@app.cell(hide_code=True)
def kalshi_side():
    candles_path = CLEAN_CANDLES if CLEAN_CANDLES.exists() else RAW_CANDLES
    mo.stop(
        not (MARKETS_FILE.exists() and TRADES_FILE.exists() and candles_path.exists()),
        mo.md("**Missing Kalshi data.** Run `00_fetch_data_kalshi.py` first (it writes `data/kalshi_*.parquet`)."),
    )
    con = duckdb.connect()
    con.execute((ROOT / "sql/kalshi_views.sql").read_text().replace("'data/", f"'{ROOT}/data/"))
    con.execute(read_queries(ROOT / "sql/kalshi_eda_macro.sql")[""].replace("'data/", f"'{ROOT}/data/"))
    con.execute(f"CREATE VIEW candles_clean AS SELECT * FROM read_parquet('{candles_path}')")
    queries = read_queries(ROOT / "sql/fedfunds_vs_kalshi.sql") | read_queries(ROOT / "sql/macrotron_v0.sql")

    contracts = con.sql("""
        SELECT ticker, event_ticker, meeting_date, move, move_bps, status, result,
               CAST(open_time AT TIME ZONE 'America/New_York' AS DATE) AS open_date
        FROM markets
        QUALIFY sum(volume) OVER (PARTITION BY event_ticker) > 0  -- drops the untraded duplicate event
    """).df()
    contracts["meeting_date"] = pd.to_datetime(contracts.meeting_date)
    contracts["open_date"] = pd.to_datetime(contracts.open_date)
    mo.md(f"Kalshi: **{contracts.event_ticker.nunique()}** meetings, **{len(contracts)}** contracts; "
          f"quotes from `{candles_path.relative_to(ROOT)}`.")
    return con, contracts, queries


@app.cell(hide_code=True)
def panel_md():
    mo.md(r"""
    ## 1 · The panel

    One row per (meeting, futures trading day), from the day the meeting was listed on Kalshi to the last trading
    day before the decision.

    - **Futures expected move** $x^{FF}$, in 25bp steps, from 03.
    - **Kalshi expected move** $x^{K} = \sum_i q_i \cdot \text{move}_i / 25$, with each bucket's 15:00 ET mid rescaled
      to sum to 1 ($q_i$). Open-ended buckets count as the next step (cut >25 as −50bp, hike >25 as +50bp), as in
      `sql/kalshi_views.sql`, which biases $x^K$ toward zero when a large move is likely.
    - **Gap** $= (x^{FF} - x^{K}) \times 25$ bp. Positive: futures price more tightening than Kalshi.
    - Rows where not every bucket is quoted are flagged `complete = False` and left out of the analysis.
    """)
    return


@app.cell(hide_code=True)
def panel_cell(con, contracts, implied, queries):
    _grid = contracts.merge(implied[["date", "meeting_date", "x", "method"]], on="meeting_date")
    _grid = _grid[(_grid.date >= _grid.open_date) & (_grid.date < _grid.meeting_date)]
    con.register("snapshot_grid", _grid[["ticker", "date"]].drop_duplicates().assign(date=lambda d: d.date.dt.date))
    _snaps = con.execute(queries["snapshots"], {"snapshot_hour": SNAPSHOT_HOUR_ET}).df()
    _snaps["date"] = pd.to_datetime(_snaps.date)

    quotes = _grid.merge(_snaps, on=["ticker", "date"], how="left")  # one row per contract and day
    quotes["w"] = quotes.kalshi_p * quotes.move_bps
    _n_buckets = contracts.groupby("event_ticker").ticker.nunique().rename("n_buckets")
    panel = (quotes.groupby(["event_ticker", "meeting_date", "date"], as_index=False)
             .agg(x_ff=("x", "first"), method=("method", "first"), n_quoted=("kalshi_p", "count"),
                  sum_mid=("kalshi_p", "sum"), w=("w", "sum"), max_spread=("spread", "max"))
             .merge(_n_buckets, on="event_ticker"))
    panel["complete"] = panel.n_quoted == panel.n_buckets
    panel["x_kalshi"] = panel.w / panel.sum_mid / STEP_BP
    panel["gap_bp"] = (panel.x_ff - panel.x_kalshi) * STEP_BP
    panel["days_to_meeting"] = (panel.meeting_date - panel.date).dt.days
    panel["wide"] = panel.max_spread > WIDE_SPREAD
    panel["regime"] = np.where(panel.meeting_date >= LIQUIDITY_BREAK, "From May 2025", "Before May 2025")
    panel["holdout"] = panel.meeting_date >= HOLDOUT_FROM
    panel = panel.drop(columns="w").sort_values(["event_ticker", "date"]).reset_index(drop=True)

    _n_hold = panel[panel.holdout].event_ticker.nunique()
    mo.vstack([
        mo.callout(mo.md("**Holdout is being scored.** These results are the final test; don't change anything "
                         "after reading them."), kind="danger") if SCORE_HOLDOUT else
        mo.callout(mo.md(f"**Holdout locked.** {_n_hold} meetings from {HOLDOUT_FROM:%Y-%m-%d} are left out of "
                         "every section below."), kind="info"),
        mo.md(f"**{len(panel):,}** meeting-days over **{panel.event_ticker.nunique()}** meetings; "
              f"**{panel.complete.mean():.0%}** complete, **{panel[panel.complete].wide.mean():.0%}** of those "
              f"with a bucket wider than {WIDE_SPREAD * 100:.0f}¢."),
        mo.ui.table(panel[~panel.holdout].head(50), selection=None),
    ])
    return panel, quotes


@app.cell(hide_code=True)
def describe_md():
    mo.md(r"""
    ## 2 · Statement 1: is the gap persistent, and does it have a sign?

    Each faint line is one meeting; the thick line is the median across meetings by days to the meeting.
    The table splits by horizon and by liquidity regime. `share_pos` is the fraction of days on which futures price
    more tightening than Kalshi.
    """)
    return


@app.cell(hide_code=True)
def describe(panel):
    _p = panel[panel.complete & (SCORE_HOLDOUT | ~panel.holdout)].copy()
    _p["lag_gap"] = _p.groupby("event_ticker").gap_bp.shift(1)
    _p["horizon"] = pd.cut(_p.days_to_meeting, [0, 7, 30, 60, 120, 1000],
                           labels=["1–7d", "8–30d", "31–60d", "61–120d", "120d+"])
    _table = (_p.groupby(["regime", "horizon"], observed=True)
              .agg(rows=("gap_bp", "size"), meetings=("event_ticker", "nunique"), mean_bp=("gap_bp", "mean"),
                   median_bp=("gap_bp", "median"), sd_bp=("gap_bp", "std"),
                   share_pos=("gap_bp", lambda g: (g > 0).mean()))
              .round(2).reset_index())
    _ac = _p.dropna(subset=["lag_gap"])[["gap_bp", "lag_gap"]].corr().iloc[0, 1]
    _mean_draws = cluster_bootstrap(_p, lambda df: df.gap_bp.mean())
    gap_summary = {"mean": _p.gap_bp.mean(), "ci": np.percentile(_mean_draws, [2.5, 97.5]),
                   "share_pos": (_p.gap_bp > 0).mean(), "autocorr": _ac, "meetings": _p.event_ticker.nunique()}

    _lines = alt.Chart(_p).mark_line(opacity=0.25, strokeWidth=1).encode(
        x=alt.X("days_to_meeting:Q", scale=alt.Scale(reverse=True), title="Days to meeting"),
        y=alt.Y("gap_bp:Q", title="Gap: futures − Kalshi (bp)"), detail="event_ticker:N",
        color=alt.Color("regime:N", title=None))
    _median = alt.Chart(_p).mark_line(color="#11151c", strokeWidth=2.5).encode(
        x="days_to_meeting:Q", y="median(gap_bp):Q")
    _zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9a9994").encode(y="y:Q")
    mo.vstack([
        (_zero + _lines + _median).properties(width="container", height=320),
        mo.md(f"Mean gap **{gap_summary['mean']:.2f}bp** (95% CI {gap_summary['ci'][0]:.2f} to "
              f"{gap_summary['ci'][1]:.2f}, resampling meetings); lag-1 autocorrelation **{_ac:.2f}**."),
        mo.ui.table(_table, selection=None),
    ])
    return (gap_summary,)


@app.cell(hide_code=True)
def convergence_md():
    mo.md(r"""
    ## 3 · Statement 2: who moves to close the gap?

    For horizon $h$, regress each market's change on today's gap, all meetings pooled:

    $$\Delta x^{K}_{t,t+h} = a + b \cdot \text{gap}_t, \qquad \Delta x^{FF}_{t,t+h} = c + d \cdot \text{gap}_t .$$

    If Kalshi moves toward futures, $b > 0$; if futures move toward Kalshi, $d < 0$. The share of the closing done by
    Kalshi is $b / (b - d)$. Confidence intervals resample whole meetings.
    """)
    return


@app.cell(hide_code=True)
def targets(panel):
    # Changes h complete rows ahead within a meeting; `target_date` lets the validation drop overlapping rows.
    data = panel[panel.complete].sort_values(["event_ticker", "date"]).reset_index(drop=True)
    _g = data.groupby("event_ticker")
    for _h in HORIZONS:
        data[f"d_kalshi_{_h}"] = (_g.x_kalshi.shift(-_h) - data.x_kalshi) * STEP_BP
        data[f"d_ff_{_h}"] = (_g.x_ff.shift(-_h) - data.x_ff) * STEP_BP
        data[f"d_gap_{_h}"] = _g.gap_bp.shift(-_h) - data.gap_bp
        data[f"target_date_{_h}"] = _g.date.shift(-_h)
    return (data,)


@app.cell(hide_code=True)
def convergence(data):
    _rows = []
    for _h in HORIZONS:
        _d = data[SCORE_HOLDOUT | ~data.holdout].dropna(subset=[f"d_kalshi_{_h}", f"d_ff_{_h}"])

        def _stat(df, h=_h):
            b = ols(df[f"d_kalshi_{h}"].to_numpy(), df[["gap_bp"]].to_numpy())[1]
            d = ols(df[f"d_ff_{h}"].to_numpy(), df[["gap_bp"]].to_numpy())[1]
            return np.array([b, d, b / (b - d) if b != d else np.nan])

        _est = _stat(_d)
        _boot = cluster_bootstrap(_d, _stat)
        _lo, _hi = np.nanpercentile(_boot, [2.5, 97.5], axis=0)
        for _i, _name in enumerate(["b: Kalshi on gap", "d: futures on gap", "Kalshi share of closing"]):
            _rows.append({"h (days)": _h, "estimate": _name, "value": _est[_i], "95% CI low": _lo[_i],
                          "95% CI high": _hi[_i], "rows": len(_d), "meetings": _d.event_ticker.nunique()})
    closing = pd.DataFrame(_rows).round(3)
    mo.ui.table(closing, selection=None)
    return (closing,)


@app.cell(hide_code=True)
def features_md():
    mo.md(r"""
    ## 4 · Features (frozen 2026-10-19, before any results)

    Every feature is **known at 15:00 ET on day t**. Each has a one-line hypothesis and a predicted sign for its
    effect on the target, the change in the gap (−1: it predicts the gap shrinking; 0: no prediction).

    Features that say *how fast* a gap should close, not *which way*, are multiplied by the gap: on their own they
    have no sign, so a linear model couldn't use them.

    **Macro surprises** are z-scores against Kalshi's own distribution:
    $z = (\text{print} - \text{implied mean}) / \text{implied sd}$, both read from the ladder the evening before
    the release. The print is the midpoint of the bracket the ladder settled in, on the same grid as the implied
    mean. On panel day $t$, the latest release on or before $t$ counts (CPI and payrolls come out at 8:30 ET, before
    the 15:00 snapshot), decayed by $e^{-\text{days since}/\tau}$ with $\tau$ = `NEWS_TAU_DAYS`.

    **Left out on purpose**, as robustness checks rather than main features: core CPI (close to headline CPI, and
    one inflation surprise is enough for ~20 meetings), order flow, overround, days to the meeting, the regime
    dummy, and TIPS breakevens (not in our data).
    """)
    return


@app.cell(hide_code=True)
def feature_registry():
    FEATURES = {
        # name: (group, predicted sign on the change in the gap, hypothesis)
        "gap_bp": ("gap dynamics", -1, "Gaps close: the larger today's gap, the more it shrinks."),
        "gap_chg_5": ("gap dynamics", -1, "A gap that just opened is the one most likely to close."),
        "gap_x_spread": ("Kalshi microstructure", -1,
                         "Gap × widest bucket spread: a gap seen through a wide book is partly noise and closes faster."),
        "gap_x_trade_age": ("Kalshi microstructure", -1,
                            "Gap × log hours since Kalshi last traded: a stale price catches up once someone trades."),
        "cpi_surprise": ("macro", -1,
                         "Hawkish CPI surprise: futures reprice that morning, Kalshi catches up over days, gap falls."),
        "payroll_surprise": ("macro", -1, "Same for payrolls."),
        "gap_x_news": ("macro", -1, "Gap × recency of the last CPI or payroll release: news-driven gaps close faster."),
        "effr_drift": ("macro", 0,
                       "EFFR spread to the range midpoint minus its 10-day median, previous day (bp): drift that moves "
                       "futures but not Kalshi."),
        "x_ff": ("market expectations", 0, "Futures-implied move at this meeting: is a move priced at all?"),
    }
    GROUPS = list(dict.fromkeys(g for g, _s, _n in FEATURES.values()))
    mo.ui.table(pd.DataFrame([{"feature": k, "group": g, "predicted sign": s, "hypothesis": n}
                              for k, (g, s, n) in FEATURES.items()]), selection=None)
    return FEATURES, GROUPS


@app.cell(hide_code=True)
def macro_releases(con, queries):
    _lad = con.execute(queries["macro_ladders"]).df()
    _lad["release_date"] = pd.to_datetime(_lad.release_date)
    _rows = []
    for (_series, _event, _date), _g in _lad.groupby(["series_ticker", "event_ticker", "release_date"]):
        _k = np.sort(_g.strike.unique())
        _step = np.median(np.diff(_k)) if len(_k) > 1 else np.nan
        _q = _g[_g.mid.notna() & (_g.spread <= WIDE_SPREAD)].sort_values("strike")
        _settled = _g.result.isin(["yes", "no"]).all()
        _row = {"series": _series, "event_ticker": _event, "release_date": _date, "strikes": len(_k),
                "quoted": len(_q), "settled": _settled}
        if _settled and len(_q) >= MIN_QUOTED_STRIKES:
            _mean, _sd = ladder_moments(_q.strike, _q.mid, _step)
            _print = bracket_value(_g.strike, _g.result, _step)
            _row |= {"print": _print, "implied_mean": _mean, "implied_sd": _sd,
                     "z": (_print - _mean) / max(_sd, _step / 2)}
        _rows.append(_row)
    # A release listed twice keeps the listing with the most quoted strikes.
    releases = (pd.DataFrame(_rows).sort_values("quoted").drop_duplicates(["series", "release_date"], keep="last")
                .sort_values(["series", "release_date"]).reset_index(drop=True))

    _show = releases[(releases.release_date < HOLDOUT_FROM) | SCORE_HOLDOUT]
    _usable = _show.groupby("series").agg(releases=("z", "size"), with_surprise=("z", "count"))
    _chart = alt.Chart(_show.dropna(subset=["z"])).mark_bar().encode(
        x=alt.X("release_date:T", title=None), y=alt.Y("z:Q", title="Surprise (z vs Kalshi)"),
        color=alt.Color("series:N", title=None),
        tooltip=["event_ticker", "release_date:T", "print", alt.Tooltip("implied_mean:Q", format=".3f"),
                 alt.Tooltip("z:Q", format=".2f")]).properties(width="container", height=220)
    mo.vstack([
        mo.md("**Macro surprises.** Releases without a settled ladder or with fewer than "
              f"{MIN_QUOTED_STRIKES} strikes quoted within {WIDE_SPREAD * 100:.0f}¢ the evening before have no "
              "surprise and count as no news."),
        mo.ui.table(_usable.reset_index(), selection=None),
        _chart,
        mo.ui.table(_show.round(3), selection=None, page_size=10),
    ])
    return (releases,)


@app.cell(hide_code=True)
def trade_ages(con, queries, quotes):
    con.register("snapshot_grid", quotes[["ticker", "date"]].drop_duplicates().assign(date=lambda d: d.date.dt.date))
    _lt = con.execute(queries["last_trades"], {"snapshot_hour": SNAPSHOT_HOUR_ET}).df()
    _lt["date"] = pd.to_datetime(_lt.date)
    _x = quotes[["ticker", "event_ticker", "date", "kalshi_p"]].merge(_lt, on=["ticker", "date"], how="left")
    _x["hours"] = _x.hours_since_trade.fillna(MAX_TRADE_AGE_H).clip(upper=MAX_TRADE_AGE_H)
    # Weighted by each bucket's probability: a stale 1¢ tail matters less than a stale favourite.
    _x["wh"] = _x.kalshi_p * _x.hours
    trade_age = _x.groupby(["event_ticker", "date"], as_index=False).agg(wh=("wh", "sum"), p=("kalshi_p", "sum"))
    trade_age["trade_age"] = np.log1p(trade_age.wh / trade_age.p)
    trade_age = trade_age[["event_ticker", "date", "trade_age"]]
    return (trade_age,)


@app.cell(hide_code=True)
def build_features(FEATURES, data, futures, releases, trade_age):
    feat = data.merge(trade_age, on=["event_ticker", "date"], how="left").sort_values("date").reset_index(drop=True)

    def _since(rel):
        """Latest release on or before each row's date: (z, days since)."""
        m = pd.merge_asof(feat[["date"]].astype("datetime64[ns]"),
                          rel.astype({"release_date": "datetime64[ns]"}).sort_values("release_date"),
                          left_on="date", right_on="release_date")
        return m.z if "z" in m else None, (m.date - m.release_date).dt.days

    for _series, _name in [("KXCPI", "cpi_surprise"), ("KXPAYROLLS", "payroll_surprise")]:
        _z, _days = _since(releases[(releases.series == _series) & releases.z.notna()][["release_date", "z"]])
        feat[_name] = (_z * np.exp(-_days / NEWS_TAU_DAYS)).fillna(0)
    _, _days = _since(releases[["release_date"]].drop_duplicates())
    feat["news_weight"] = np.exp(-_days / NEWS_TAU_DAYS).fillna(0)

    _f = futures[["date", "upper", "spread", "spread_median"]].sort_values("date")
    # On a decision day the range moves but the EFFR follows only the next day: that step is not drift.
    _f["spread"] = _f.spread.where(_f.upper.eq(_f.upper.shift()))
    _f = _f.assign(effr_drift=((_f.spread - _f.spread_median) * 100).shift(1))  # percent to bp, previous day
    _f["effr_drift"] = _f.effr_drift.fillna(0)
    feat = feat.merge(_f[["date", "effr_drift"]], on="date", how="left")

    feat = feat.sort_values(["event_ticker", "date"]).reset_index(drop=True)
    feat["gap_chg_5"] = feat.gap_bp - feat.groupby("event_ticker").gap_bp.shift(5)
    feat["gap_x_spread"] = feat.gap_bp * feat.max_spread
    feat["gap_x_trade_age"] = feat.gap_bp * feat.trade_age
    feat["gap_x_news"] = feat.gap_bp * feat.news_weight
    ALL = list(FEATURES)

    _dev = feat[SCORE_HOLDOUT | ~feat.holdout]
    _summary = pd.DataFrame({
        "group": [FEATURES[k][0] for k in ALL], "mean": _dev[ALL].mean().values, "sd": _dev[ALL].std().values,
        "share non-zero": (_dev[ALL] != 0).mean().values, "missing": _dev[ALL].isna().sum().values,
    }, index=ALL).round(3)
    mo.vstack([mo.md("**Feature check** (step 3 of the guide): spread, how often each feature is non-zero, and "
                     "missing values, which drop the row. `gap_chg_5` is missing for each meeting's first 5 days."),
               mo.ui.table(_summary.reset_index(names="feature"), selection=None)])
    return ALL, feat


@app.cell(hide_code=True)
def models_md():
    mo.md(r"""
    ## 5 · Statement 4: is the change in the gap predictable out of sample?

    **Target:** the change in the gap over $h$ trading days, `d_gap_h`.

    **Validation: expanding window by meeting.** For each test meeting $m$ (in date order, once
    `MIN_TRAIN_MEETINGS` meetings are past), train on meetings decided **before** $m$, using only rows whose target
    window ends **before** $m$'s first row. No meeting is split between train and test.

    **Three tiers of meetings.** *Tuning*: the last `TUNE_MEETINGS` meetings of each training window choose ridge's
    alpha from `ALPHAS`. *Validation*: the expanding-window test meetings before `HOLDOUT_FROM`. *Final test*:
    meetings from `HOLDOUT_FROM` on, scored once, by setting `SCORE_HOLDOUT = True` after everything is frozen.

    **Models:** no change (predict 0); mean reversion, AR(1), `d_gap ~ gap`, the baseline to beat; ridge on all
    nine features. Every ridge is also refit without each group (section 6) and without each feature.

    **Scores:** out-of-sample $R^2$ relative to no change and relative to AR(1), the share of days on which the
    direction is right, and a Clark–West test of ridge against AR(1). CIs resample whole test meetings.
    """)
    return


@app.cell(hide_code=True)
def validation(ALL, FEATURES, GROUPS, feat):
    _specs = {"Ridge: all": ALL}
    _specs |= {f"Ridge: without {g}": [k for k in ALL if FEATURES[k][0] != g] for g in GROUPS}
    _specs |= {f"Ridge: without {k}": [j for j in ALL if j != k] for k in ALL}

    preds, _alphas = [], []
    for _h in HORIZONS:
        _target, _tdate = f"d_gap_{_h}", f"target_date_{_h}"
        _d = feat.dropna(subset=[_target, *ALL])  # the same rows for every model, so scores compare
        _order = _d.groupby("event_ticker").meeting_date.first().sort_values()
        # Development scores the pre-holdout meetings; the final run scores only the holdout, still expanding.
        _tests = _order.index[MIN_TRAIN_MEETINGS:]
        _tests = _tests[(_order[_tests] >= HOLDOUT_FROM).to_numpy() == SCORE_HOLDOUT]
        for _m in _tests:
            _test = _d[_d.event_ticker == _m]
            _train = _d[(_d.meeting_date < _order[_m]) & (_d[_tdate] < _test.date.min())]
            if _train.event_ticker.nunique() < MIN_TRAIN_MEETINGS:
                continue
            _out = _test[["event_ticker", "meeting_date", "date", _target]].rename(columns={_target: "actual"})
            _out = _out.assign(h=_h)
            _a, _b = ols(_train[_target].to_numpy(), _train[["gap_bp"]].to_numpy())
            _out["No change"] = 0.0
            _out["Mean reversion (AR1)"] = _a + _b * _test.gap_bp.to_numpy()
            for _name, _cols in _specs.items():
                _out[_name], _alpha = ridge_tuned(_train, _test, _cols, _target, _tdate)
                if _name == "Ridge: all":
                    _alphas.append({"h": _h, "test meeting": _m, "alpha": _alpha})
            preds.append(_out)
    preds = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    alphas = pd.DataFrame(_alphas)
    return alphas, preds


@app.cell(hide_code=True)
def scores(preds):
    mo.stop(preds.empty, mo.md("Not enough meetings to start the expanding window."))

    def _boot_sse(df, models):
        """Per-meeting sums of squared errors, and their meeting-bootstrap totals."""
        se = pd.DataFrame({m: (df.actual - df[m]) ** 2 for m in models}).groupby(df.event_ticker.to_numpy()).sum()
        idx = np.random.default_rng(0).integers(0, len(se), (N_BOOT, len(se)))
        return se.sum(), pd.DataFrame(se.to_numpy()[idx].sum(axis=1), columns=models)

    model_cols = [c for c in preds.columns if c in ("No change", "Mean reversion (AR1)") or c.startswith("Ridge")]
    _rows = []
    oos = {}
    for _h, _g in preds.groupby("h"):
        _tot, _boot = _boot_sse(_g, model_cols)
        oos[_h] = (_g, _tot, _boot)
        for _model in ["Mean reversion (AR1)", "Ridge: all"]:
            _r0 = 1 - _tot[_model] / _tot["No change"]
            _r1 = 1 - _tot[_model] / _tot["Mean reversion (AR1)"]
            _b0 = np.percentile(1 - _boot[_model] / _boot["No change"], [2.5, 97.5])
            _b1 = np.percentile(1 - _boot[_model] / _boot["Mean reversion (AR1)"], [2.5, 97.5])
            _small = "No change" if _model == "Mean reversion (AR1)" else "Mean reversion (AR1)"
            _t, _p = clark_west(_g.actual.to_numpy(), _g[_small].to_numpy(), _g[_model].to_numpy(), _g.event_ticker)
            _nz = _g.actual != 0
            _rows.append({"h": _h, "model": _model, "R² vs no change": round(_r0, 3),
                          "95% CI": f"[{_b0[0]:.3f}, {_b0[1]:.3f}]", "R² vs AR1": round(_r1, 3),
                          "95% CI ": f"[{_b1[0]:.3f}, {_b1[1]:.3f}]",
                          "direction hit rate": round((np.sign(_g[_model]) == np.sign(_g.actual))[_nz].mean(), 3),
                          "Clark–West p vs": f"{_p:.3f} ({_small})",
                          "test meetings": _g.event_ticker.nunique(), "rows": len(_g)})
    main_scores = pd.DataFrame(_rows)
    mo.vstack([
        mo.ui.table(main_scores, selection=None),
        mo.md("R² vs AR1 for the AR1 row is 0 by construction. Ridge passes statement 4 if its R² vs AR1 is "
              "positive with a CI above zero and the Clark–West p-value is below 0.05."),
    ])
    return main_scores, oos


@app.cell(hide_code=True)
def alpha_chart(alphas):
    mo.stop(alphas.empty)
    mo.vstack([mo.md("Alpha chosen in each window (full ridge). A choice stuck at either end of `ALPHAS` means "
                     "the grid should be wider."),
               mo.ui.table(alphas.groupby(["h", "alpha"]).size().rename("windows").reset_index(), selection=None)])
    return


@app.cell(hide_code=True)
def ablation_md():
    mo.md(r"""
    ## 6 · Statement 3: which groups matter? (layer 1, the main test)

    Each row refits ridge without one group. **Loss from dropping** = $R^2_\text{all} - R^2_\text{without}$, both
    relative to AR(1): how much out-of-sample accuracy the group adds. The CI resamples test meetings, paired
    across the two models. The Clark–West test asks whether the full model beats the model without the group.

    Statement 3 holds if dropping **macro** costs accuracy, with a CI above zero and Clark–West p below 0.05.
    """)
    return


@app.cell(hide_code=True)
def ablation(GROUPS, oos):
    _rows = []
    for _h, (_g, _tot, _boot) in oos.items():
        for _grp in GROUPS:
            _col = f"Ridge: without {_grp}"
            _loss = (_tot[_col] - _tot["Ridge: all"]) / _tot["Mean reversion (AR1)"]
            _bl = np.percentile((_boot[_col] - _boot["Ridge: all"]) / _boot["Mean reversion (AR1)"], [2.5, 97.5])
            _t, _p = clark_west(_g.actual.to_numpy(), _g[_col].to_numpy(), _g["Ridge: all"].to_numpy(), _g.event_ticker)
            _rows.append({"h": _h, "group dropped": _grp,
                          "R² vs AR1 without it": round(1 - _tot[_col] / _tot["Mean reversion (AR1)"], 3),
                          "loss from dropping": round(_loss, 4), "95% CI": f"[{_bl[0]:.4f}, {_bl[1]:.4f}]",
                          "Clark–West t": round(_t, 2), "p": round(_p, 3)})
    group_tests = pd.DataFrame(_rows)
    mo.ui.table(group_tests, selection=None)
    return (group_tests,)


@app.cell(hide_code=True)
def features_test_md():
    mo.md(r"""
    ## 6b · Which features inside the groups? (layer 2, the explanation)

    Four checks per feature, on the development meetings. Features are standardized, so coefficients compare.

    - **Coef / clustered t / Holm p:** OLS of the target on all nine features, standard errors clustered by
      meeting, p-values adjusted for testing nine features at once (Holm).
    - **Sign stability:** share of meeting-bootstrap refits whose coefficient has the predicted sign.
    - **Out-of-sample loss from dropping it**, in units of R² vs AR(1), and the Clark–West p-value.

    **Verdict:** *effect* if the sign is as predicted, Holm p < `P_CUT`, sign stability ≥ `SIGN_STABILITY`, and
    dropping it costs accuracy with Clark–West p < `P_CUT`. *Weak* if some of these hold, *none* if none do.
    For features with no predicted sign, only significance and out-of-sample loss count. These verdicts explain
    the group result; they don't replace it.
    """)
    return


@app.cell(hide_code=True)
def feature_tests(ALL, FEATURES, feat, oos):
    _rows = []
    for _h in HORIZONS:
        _target = f"d_gap_{_h}"
        _d = feat[SCORE_HOLDOUT | ~feat.holdout].dropna(subset=[_target, *ALL]).copy()
        _d[ALL] = (_d[ALL] - _d[ALL].mean()) / _d[ALL].std().replace(0, 1)
        _beta, _se = ols_cluster(_d[_target].to_numpy(), _d[ALL].to_numpy(), _d.event_ticker)
        _t = _beta[1:] / _se[1:]
        _p = np.array([math.erfc(abs(t) / math.sqrt(2)) for t in _t])
        _ph = holm(_p)
        _, _draws = boot_linear(_d, ALL, _target)
        _g, _tot, _ = oos[_h] if _h in oos else (None, None, None)
        for _i, _k in enumerate(ALL):
            _sign = FEATURES[_k][1]
            _stab = (np.sign(_draws[:, _i + 1]) == _sign).mean() if _sign else np.nan
            _row = {"h": _h, "feature": _k, "group": FEATURES[_k][0], "predicted sign": _sign,
                    "coef": round(_beta[_i + 1], 3), "clustered t": round(_t[_i], 2), "Holm p": round(_ph[_i], 3),
                    "sign stability": round(_stab, 3)}
            if _g is not None:
                _col = f"Ridge: without {_k}"
                _row["OOS loss from dropping"] = round((_tot[_col] - _tot["Ridge: all"]) / _tot["Mean reversion (AR1)"], 4)
                _row["Clark–West p"] = round(clark_west(_g.actual.to_numpy(), _g[_col].to_numpy(),
                                                        _g["Ridge: all"].to_numpy(), _g.event_ticker)[1], 3)
            _checks = [_ph[_i] < P_CUT, _row.get("Clark–West p", 1) < P_CUT]
            if _sign:
                _checks += [np.sign(_beta[_i + 1]) == _sign, _stab >= SIGN_STABILITY]
            _row["verdict"] = "effect" if all(_checks) else ("weak" if any(_checks) else "none")
            _rows.append(_row)
    feature_table = pd.DataFrame(_rows)
    mo.ui.table(feature_table, selection=None, page_size=20)
    return (feature_table,)


@app.cell(hide_code=True)
def coef_plot(ALL, FEATURES, alphas, feat):
    # Ridge on all development meetings, standardized, at the alpha chosen most often; whiskers resample meetings.
    _h = HORIZONS[-1]
    _target = f"d_gap_{_h}"
    _d = feat[SCORE_HOLDOUT | ~feat.holdout].dropna(subset=[_target, *ALL]).copy()
    _d[ALL] = (_d[ALL] - _d[ALL].mean()) / _d[ALL].std().replace(0, 1)
    _alpha = alphas[alphas.h == _h].alpha.mode().iloc[0] if not alphas.empty else 10.0
    _point, _draws = boot_linear(_d, ALL, _target, alpha=_alpha)
    _lo, _hi = np.percentile(_draws[:, 1:], [2.5, 97.5], axis=0)
    _c = pd.DataFrame({"feature": ALL, "group": [FEATURES[k][0] for k in ALL], "coef": _point[1:],
                       "lo": _lo, "hi": _hi})
    _base = alt.Chart(_c).encode(y=alt.Y("feature:N", sort=ALL, title=None))
    _zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#9a9994").encode(x="x:Q")
    mo.vstack([
        mo.md(f"**Ridge coefficients, h = {_h}** (alpha {_alpha:g}; bp of gap change per 1 sd of the feature; "
              "95% meeting-bootstrap intervals)."),
        (_zero + _base.mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", title="Coefficient"), x2="hi:Q",
                                                       color=alt.Color("group:N", title=None))
         + _base.mark_point(filled=True, size=70).encode(x="coef:Q", color="group:N"))
        .properties(width="container", height=260),
    ])
    return


@app.cell(hide_code=True)
def verdicts(closing, gap_summary, group_tests, main_scores):
    def _row(df, **kw):
        m = np.logical_and.reduce([df[k] == v for k, v in kw.items()])
        return df[m].iloc[0] if m.any() else None

    _h = HORIZONS[-1]
    _share = _row(closing, **{"h (days)": _h, "estimate": "Kalshi share of closing"})
    _ridge = _row(main_scores, h=_h, model="Ridge: all")
    _macro = _row(group_tests, h=_h, **{"group dropped": "macro"})
    _ci = gap_summary["ci"]
    _lines = [
        f"1. **Gap persistent and signed?** Mean {gap_summary['mean']:.2f}bp (CI {_ci[0]:.2f} to {_ci[1]:.2f}), "
        f"positive on {gap_summary['share_pos']:.0%} of days, lag-1 autocorrelation {gap_summary['autocorr']:.2f}. "
        f"Signed if the CI excludes 0.",
    ]
    if _share is not None:
        _lines.append(f"2. **Kalshi closes the gap?** Kalshi's share of closing over {_h} days: "
                      f"{_share['value']:.2f} (CI {_share['95% CI low']:.2f} to {_share['95% CI high']:.2f}). "
                      "Supported if the CI lies above 0.5.")
    if _macro is not None:
        _lines.append(f"3. **Macro adds beyond the rest?** Loss from dropping macro at h = {_h}: "
                      f"{_macro['loss from dropping']} {_macro['95% CI']}, Clark–West p = {_macro['p']}.")
    if _ridge is not None:
        _lines.append(f"4. **Predictable out of sample?** Ridge R² vs AR1 at h = {_h}: {_ridge['R² vs AR1']} "
                      f"{_ridge['95% CI ']}, Clark–West p {_ridge['Clark–West p vs']}.")
    mo.md("## 7 · Verdicts (" + ("final test" if SCORE_HOLDOUT else "development meetings only") + ")\n\n"
          + "\n".join(_lines)
          + "\n\nA statement that fails is still a result: write it up with the number that decides it.")
    return


if __name__ == "__main__":
    app.run()
