# /// script
# dependencies = ["duckdb", "pandas", "pyarrow"]
# ///
"""Write the chart data files in presentation/data/ that are computed from raw data.

The other files in presentation/data/ were copied by hand from the saved outputs of the
marimo notebooks (02_calibration_kalshi, 03_calibration_macro, 03_fedfunds_vs_kalshi).

Run from the repository root:  uv run presentation/build_data.py
"""
import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else ROOT
OUT = ROOT / "presentation" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def decimal_year(s):
    t = pd.to_datetime(pd.Series(s).reset_index(drop=True))
    y = t.dt.year
    start = pd.to_datetime(y.astype(str) + "-01-01")
    days = (pd.to_datetime((y + 1).astype(str) + "-01-01") - start).dt.days
    return (y + (t - start).dt.days / days).values


# Monthly contracts traded on the Fed decision markets (Oct 2026 is partial and dropped)
con = duckdb.connect()
vol = con.execute(f"""
    SELECT date_trunc('month', created_time)::DATE AS month, sum(count) / 1e6 AS mcontracts
    FROM read_parquet('{REPO}/data/kalshi_trades.parquet')
    WHERE ticker LIKE 'KXFEDDECISION%' OR ticker LIKE 'FEDDECISION%'
    GROUP BY 1 ORDER BY 1
""").df()
vol = vol[pd.to_datetime(vol.month) < "2026-10-01"]
vol["x"] = decimal_year(vol.month.astype(str)) + 1 / 24
vol[["x", "mcontracts"]].to_csv(OUT / "fed_monthly_volume.dat", sep=" ", index=False, float_format="%.4f")

# DKW replication: mean absolute error of the implied rate by days to meeting
kf = REPO / "kalshi_fed" / "data"
curves = json.load(open(kf / "validation.json"))["curves"]
pd.DataFrame({"h": curves["h"], "ours": curves["ours_mean"], "dkw": curves["dkw_mean"]}).to_csv(
    OUT / "dkw_mae_curve.dat", sep=" ", index=False)

# Hair chart: implied upper-bound paths (one per start date) and the realized target
chart = json.load(open(kf / "chart_data.json"))
rows = []
for hair in chart["hairs"]:
    pts = pd.DataFrame(hair["pts"], columns=["date", "rate"])
    pts["x"] = decimal_year(pts.date)
    rows += [f"{x:.4f} {r:.4f}" for x, r in zip(pts.x, pts.rate)] + ["nan nan"]
(OUT / "hairs.dat").write_text("x rate\n" + "\n".join(rows) + "\n")
real = pd.DataFrame(chart["realized"])
real["x"] = decimal_year(real.meeting_date)
real[["x", "upper_bound"]].to_csv(OUT / "realized_target.dat", sep=" ", index=False, float_format="%.4f")

# Sep 2024 meeting: Kalshi (hourly) vs futures-implied (daily, 15:00 ET) probabilities,
# read from the chart saved in the 03_fedfunds_vs_kalshi session
sess = json.load(open(REPO / "__marimo__" / "session" / "03_fedfunds_vs_kalshi.py.json"))
for cell in sess["cells"]:
    for o in cell.get("outputs", []):
        spec = o.get("data", {}).get("application/vnd.vegalite.v6+json")
        if spec is None:
            continue
        spec = json.loads(spec) if isinstance(spec, str) else spec
        for data in spec.get("datasets", {}).values():
            df = pd.DataFrame(data)
            if "p_ff" in df:
                ff = df
            elif "mid" in df:
                ks = df
meeting = pd.Timestamp("2024-09-18")
for df in (ff, ks):
    df["t"] = (pd.to_datetime(df.ts) - meeting).dt.total_seconds() / 86400
for move, tag in [("Cut 25", "c25"), ("Cut >25", "c50")]:
    f = ff[ff.move == move].sort_values("t")
    f[["t", "p_ff"]].to_csv(OUT / f"sep24_ff_{tag}.dat", sep=" ", index=False, float_format="%.4f")
    k = ks[(ks.move == move) & (ks.spread <= 0.20) & (ks.t <= 0)].sort_values("t")
    k = k[pd.to_datetime(k.ts).dt.hour.isin([3, 9, 15, 21])]  # four points a day
    k[["t", "mid"]].to_csv(OUT / f"sep24_kalshi_{tag}.dat", sep=" ", index=False, float_format="%.4f")

print("wrote", sorted(p.name for p in OUT.iterdir()))
