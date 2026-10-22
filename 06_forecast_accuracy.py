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
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parent
    FF_XLSX = ROOT / "data/KalshiProjectFedFunds.xlsx"
    RAW_CANDLES = ROOT / "data/kalshi_candles_1m.parquet"
    CLEAN_CANDLES = ROOT / "data/cleaned/kalshi_candles_1m.parquet"

    # Fixed 2026-10-22, before any result of this notebook was seen (v0's results were known; see the intro).
    HORIZON_BINS = [0, 7, 30, 90, 10_000]
    HORIZON_LABELS = ["1–7d", "8–30d", "31–90d", "90d+"]
    N_BOOT = 999
    CONFIRM_FROM = pd.Timestamp("2026-10-22")  # meetings decided after this date confirm the results once settled

    alt.data_transformers.disable_max_rows()

    def load_notebook(stem):
        """Import another notebook's @app.function helpers (filenames start with a digit, so `import` can't)."""
        spec = importlib.util.spec_from_file_location(f"nb_{stem}", ROOT / f"{stem}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    nb03 = load_notebook("03_fedfunds_vs_kalshi")  # Panos: futures-implied moves
    nb05 = load_notebook("05_macrotron_v0")  # MacroTron v0: the Kalshi panel


@app.function(hide_code=True)
def meeting_boot(per_meeting, n_boot=N_BOOT, seed=0):
    """Mean across meetings of a per-meeting number, with a 95% interval from resampling meetings and a t-test.
    Each meeting counts once however many days it was listed, so long-listed meetings don't dominate."""
    v = np.asarray(per_meeting, dtype=float)
    draws = np.random.default_rng(seed).choice(v, (n_boot, len(v))).mean(axis=1)
    t = v.mean() / (v.std(ddof=1) / math.sqrt(len(v)))
    return v.mean(), np.percentile(draws, [2.5, 97.5]), t, math.erfc(abs(t) / math.sqrt(2))


@app.function(hide_code=True)
def encompassing(df, kalshi="f_k"):
    """Forecast-encompassing regression: realized - futures = a + w (Kalshi - futures). w is the weight the best
    combination puts on Kalshi: w > 0 means Kalshi adds information futures miss; w = 1 means futures add nothing.
    Meeting-clustered standard errors; returns (w, its se, two-sided p)."""
    beta, se = nb05.ols_cluster((df.realized - df.f_ff).to_numpy(), (df[kalshi] - df.f_ff).to_numpy()[:, None],
                                df.event_ticker)
    return beta[1], se[1], math.erfc(abs(beta[1] / se[1]) / math.sqrt(2))


@app.cell(hide_code=True)
def intro():
    mo.md(r"""
    # 06 · Does Kalshi forecast FOMC decisions better than fed funds futures?

    MacroTron v0 compared the two markets with **each other**: they agree within 1–2bp, and which one adjusts is
    unstable. This notebook scores both against **what the Fed actually did**.

    For every decided meeting and every day it was listed, each market's expected move (bp) is a forecast of the
    realized decision (bp). Same panel, same usable rows and same 15:00 ET snapshot as v0.

    **Written down 2026-10-22, before any result here was seen** (v0's results were known):

    1. **Accuracy by horizon** (1–7, 8–30, 31–90, 90+ days): mean squared error of each market, and the difference
       Kalshi − futures. Each meeting counts once; 95% intervals resample meetings; Diebold–Mariano-type t-test on
       per-meeting loss differences. Negative = Kalshi more accurate.
    2. **Encompassing by horizon:** realized − futures = a + w·(Kalshi − futures). w > 0: Kalshi adds information
       futures miss. Meeting-clustered errors.
    3. **Liquidity:** the encompassing weight w by tercile of Kalshi's widest spread. Hypothesis from v0: Kalshi
       adds information only when its book is tight.
    4. **Robustness:** open-ended buckets scored at ±62.5bp.

    No holdout is left from v0's split; instead, meetings decided after `CONFIRM_FROM` (October and December 2026,
    then 2027) are an untouched confirmation set once they settle. Rerun then without changing anything.
    """)
    return


@app.cell(hide_code=True)
def data():
    candles = CLEAN_CANDLES if CLEAN_CANDLES.exists() else RAW_CANDLES
    fomc = nb03.load_fomc(FF_XLSX)
    _futures, _ = nb03.load_futures(FF_XLSX)
    _futures = nb03.add_starting_rate(_futures, fomc.meeting_date.tolist())
    _implied = nb03.implied_moves(_futures, fomc.meeting_date.tolist())
    _con, _contracts, _queries = nb05.connect_kalshi(candles)
    _panel, _ = nb05.build_panel(_con, _contracts, _implied, _queries)

    scored = _panel[_panel.usable].merge(fomc[["meeting_date", "decision_bp"]], on="meeting_date", how="inner")
    scored = scored[scored.decision_bp.notna()].copy()
    scored["realized"] = scored.decision_bp
    scored["f_ff"] = scored.x_ff * nb05.STEP_BP
    scored["f_k"] = scored.x_kalshi * nb05.STEP_BP
    scored["f_k_alt"] = scored.x_kalshi_alt * nb05.STEP_BP
    scored["horizon"] = pd.cut(scored.days_to_meeting, HORIZON_BINS, labels=HORIZON_LABELS)
    for _f in ["f_ff", "f_k", "f_k_alt"]:
        scored[f"se_{_f}"] = (scored.realized - scored[_f]) ** 2
    scored["confirm"] = scored.meeting_date > CONFIRM_FROM

    _m = scored.groupby("event_ticker").agg(meeting=("meeting_date", "first"), decision=("realized", "first"))
    mo.vstack([
        mo.md(f"**{len(scored):,}** usable meeting-days over **{scored.event_ticker.nunique()}** decided meetings "
              f"({_m.meeting.min():%b %Y} – {_m.meeting.max():%b %Y}). Decisions: "
              + ", ".join(f"{int(k):+d}bp × {v}" for k, v in _m.decision.value_counts().sort_index().items()) + "."),
    ])
    return (scored,)


@app.cell(hide_code=True)
def accuracy_md():
    mo.md(r"""
    ## 1 · Accuracy by horizon

    Per meeting, the mean squared error of each market's forecast over its days in the horizon bin; then the mean
    across meetings. **Δ MSE** = Kalshi − futures: negative means Kalshi was closer to the decision.
    """)
    return


@app.cell(hide_code=True)
def accuracy(scored):
    def _table(df, kalshi="se_f_k", label="Kalshi"):
        rows = []
        for h, g in df.groupby("horizon", observed=True):
            per = g.groupby("event_ticker")[["se_f_ff", kalshi]].mean()
            diff, ci, t, p = meeting_boot(per[kalshi] - per.se_f_ff)
            rows.append({"horizon": h, "meetings": len(per), "rows": len(g),
                         "RMSE futures (bp)": round(math.sqrt(per.se_f_ff.mean()), 2),
                         f"RMSE {label} (bp)": round(math.sqrt(per[kalshi].mean()), 2),
                         "Δ MSE (Kalshi − futures)": round(diff, 2), "95% CI": f"[{ci[0]:.2f}, {ci[1]:.2f}]",
                         "t": round(t, 2), "p": round(p, 3)})
        return pd.DataFrame(rows)

    accuracy_table = _table(scored)
    accuracy_alt = _table(scored, "se_f_k_alt", "Kalshi ±62.5")

    _long = accuracy_table.melt(id_vars="horizon", value_vars=["RMSE futures (bp)", "RMSE Kalshi (bp)"],
                                var_name="market", value_name="RMSE")
    _chart = alt.Chart(_long).mark_line(point=True).encode(
        x=alt.X("horizon:N", sort=HORIZON_LABELS, title="Days to meeting"),
        y=alt.Y("RMSE:Q", title="Forecast error, RMSE (bp)"), color=alt.Color("market:N", title=None),
    ).properties(width="container", height=260)
    mo.vstack([mo.ui.table(accuracy_table, selection=None), _chart,
               mo.md("**Robustness: open-ended buckets at ±62.5bp**"), mo.ui.table(accuracy_alt, selection=None)])
    return accuracy_alt, accuracy_table


@app.cell(hide_code=True)
def encompassing_md():
    mo.md(r"""
    ## 2 · Does Kalshi add information futures miss?

    $\text{realized} - f^{FF} = a + w\,(f^{K} - f^{FF})$, by horizon. **w = 0**: futures already contain everything
    Kalshi knows. **w > 0**: the best forecast leans toward Kalshi. **w = 1**: futures add nothing beyond Kalshi.
    """)
    return


@app.cell(hide_code=True)
def encompass(scored):
    _rows = []
    for _h, _g in scored.groupby("horizon", observed=True):
        for _col, _label in [("f_k", "baseline"), ("f_k_alt", "open-ended ±62.5")]:
            _w, _se, _p = encompassing(_g, _col)
            _rows.append({"horizon": _h, "Kalshi scoring": _label, "w (weight on Kalshi)": round(_w, 3),
                          "95% CI": f"[{_w - 1.96 * _se:.2f}, {_w + 1.96 * _se:.2f}]", "p (w = 0)": round(_p, 3),
                          "meetings": _g.event_ticker.nunique(), "rows": len(_g)})
    encompass_table = pd.DataFrame(_rows)
    mo.ui.table(encompass_table, selection=None)
    return (encompass_table,)


@app.cell(hide_code=True)
def liquidity_md():
    mo.md(r"""
    ## 3 · Is Kalshi's information tied to its liquidity?

    The same encompassing weight, by tercile of Kalshi's widest bucket spread that day (all horizons pooled, then
    1–30 days only, where Kalshi is most liquid).
    """)
    return


@app.cell(hide_code=True)
def liquidity(scored):
    _s = scored.copy()
    _s["liquidity"] = pd.qcut(_s.max_spread.rank(method="first"), 3, labels=["tight", "middle", "wide"])
    _rows = []
    for _sample, _d in [("all horizons", _s), ("1–30 days", _s[_s.days_to_meeting <= 30])]:
        for _liq, _g in _d.groupby("liquidity", observed=True):
            _w, _se, _p = encompassing(_g)
            _rows.append({"sample": _sample, "Kalshi spread": _liq, "max spread range":
                          f"{_g.max_spread.min():.2f}–{_g.max_spread.max():.2f}", "w": round(_w, 3),
                          "95% CI": f"[{_w - 1.96 * _se:.2f}, {_w + 1.96 * _se:.2f}]", "p (w = 0)": round(_p, 3),
                          "meetings": _g.event_ticker.nunique(), "rows": len(_g)})
    liquidity_table = pd.DataFrame(_rows)
    mo.ui.table(liquidity_table, selection=None)
    return (liquidity_table,)


@app.cell(hide_code=True)
def verdict(accuracy_table, encompass_table):
    _better = accuracy_table[(accuracy_table.p < 0.05)]
    _enc = encompass_table[(encompass_table["Kalshi scoring"] == "baseline") & (encompass_table["p (w = 0)"] < 0.05)]
    mo.md("## 4 · Verdict\n\n"
          + ("- Kalshi's accuracy differs from futures (p < 0.05) at: "
             + ", ".join(f"{r.horizon} (Δ MSE {r['Δ MSE (Kalshi − futures)']})" for _, r in _better.iterrows())
             if len(_better) else "- No horizon where the two markets' accuracy differs at p < 0.05.")
          + "\n"
          + ("- Kalshi adds information beyond futures (w > 0, p < 0.05) at: "
             + ", ".join(f"{r.horizon} (w = {r['w (weight on Kalshi)']})" for _, r in _enc.iterrows())
             if len(_enc) else "- No horizon where Kalshi adds information beyond futures at p < 0.05.")
          + "\n\nWith ~28 decided meetings, a non-significant difference means the data can't tell the two apart, "
          "not that they are equal. Report the CIs.")
    return


if __name__ == "__main__":
    app.run()
