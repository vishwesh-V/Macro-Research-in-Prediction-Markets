# /// script
# dependencies = ["vl-convert-python"]
# ///
"""Export charts from the saved marimo sessions to PDF for the deck, restyled for slides.

Charts are read from __marimo__/session/<notebook>.py.json (top-level outputs and charts nested in
tabs), recolored to the deck palette (Kalshi = MIT red, futures = charcoal), resized, and written
to presentation/figures/. Rerun a notebook, then rerun this script to refresh the figures.

Run from the repository root:  uv run presentation/export_figures.py [repo_root]
"""
import html
import json
import re
import sys
from pathlib import Path

import vl_convert as vlc

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
OUT = ROOT / "presentation" / "figures"

# output name: (notebook, cell id, chart index in cell, width, height)
FIGURES = {
    "reliability_fed":        ("02_calibration_kalshi", "aLJB", 0, 250, 250),
    "fitted_by_regime":       ("02_calibration_kalshi", "mWxS", 0, 250, 250),
    "fitted_by_horizon":      ("02_calibration_kalshi", "urSm", 0, 250, 250),
    "reliability_minute":     ("02_calibration_kalshi", "TXez", 0, 250, 250),
    "scores_by_horizon":      ("02_calibration_kalshi", "fwwy", 0, 460, 220),
    "logloss_per_meeting":    ("02_calibration_kalshi", "dGlV", 0, 460, 220),
    "monthly_volume":         ("01_eda_kalshi", "SFPL", 0, 460, 220),
    "macro_by_series":        ("03_calibration_macro", "AjVT", 0, 250, 250),
    "macro_fitted":           ("03_calibration_macro", "dGlV", 0, 250, 250),
    "ff_vs_kalshi_fitted":    ("03_fedfunds_vs_kalshi", "TXez", 0, 250, 250),
    "ff_vs_kalshi_reliab":    ("03_fedfunds_vs_kalshi", "aqbW", 0, 250, 250),
    "leadlag_xcorr":          ("04_leadlag_fedfunds_kalshi", "ulZA", 0, 340, 230),
    "leadlag_after_futures":  ("04_leadlag_fedfunds_kalshi", "Pvdt", 0, 250, 170),
    "leadlag_after_kalshi":   ("04_leadlag_fedfunds_kalshi", "Pvdt", 1, 250, 170),
    "leadlag_sep26_path":     ("04_leadlag_fedfunds_kalshi", "qnkX", 0, 520, 220),
}

COLORS = {
    "#2a78d6": "#A31F34",  # Kalshi blue -> MIT red
    "#eb6834": "#4D5359",  # futures orange -> charcoal
    "#c0302f": "#4D5359",
    "#104281": "#6E1423", "#1c5cab": "#8A1A2C", "#5598e7": "#C0606F", "#86b6ef": "#DDA3AD",
}

CONFIG = {
    "font": "Helvetica, Arial, sans-serif",
    "view": {"stroke": None},
    "axis": {"labelFontSize": 15, "titleFontSize": 16, "titleFontWeight": "normal",
             "labelColor": "#55595E", "titleColor": "#1E1E1E", "gridColor": "#E4E5E6",
             "domainColor": "#8A8B8C", "tickColor": "#8A8B8C"},
    "legend": {"labelFontSize": 15, "titleFontSize": 15, "titleFontWeight": "normal",
               "labelColor": "#1E1E1E", "titleColor": "#55595E"},
    "header": {"labelFontSize": 15, "titleFontSize": 15},
    "text": {"fontSize": 14},
}


def specs_in(output):
    for mime, v in output.get("data", {}).items():
        if "vegalite" in mime:
            yield json.loads(v) if isinstance(v, str) else v
        elif mime == "text/html" and isinstance(v, str):
            for m in re.finditer(r"<marimo-mime-renderer([^>]*)>", v):
                attrs = dict(re.findall(r"data-([a-z-]+)='([^']*)'", m.group(1)))
                if "vegalite" not in html.unescape(attrs.get("mime", "")):
                    continue
                data = json.loads(html.unescape(attrs["data"]))
                yield json.loads(data) if isinstance(data, str) else data


def load(notebook, cell_id, k):
    cells = json.load(open(REPO / "__marimo__" / "session" / f"{notebook}.py.json"))["cells"]
    for c in cells:
        if c.get("id") == cell_id:
            charts = [s for o in c.get("outputs", []) for s in specs_in(o)]
            return charts[k]
    raise KeyError(f"{notebook}:{cell_id}")


def restyle(spec, width, height):
    datasets = spec.pop("datasets", {})
    s = json.dumps(spec)
    for old, new in COLORS.items():
        s = re.sub(old, new, s, flags=re.I)
    spec = json.loads(s)
    spec["datasets"] = datasets
    spec.pop("title", None)  # the slide title carries the message
    spec["width"], spec["height"] = width, height
    spec["config"] = {**spec.get("config", {}), **CONFIG}
    spec["background"] = "white"
    return spec


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (nb, cell, k, w, h) in FIGURES.items():
        spec = restyle(load(nb, cell, k), w, h)
        if name.startswith("leadlag_after"):  # shown side by side: legend below to save width
            spec["config"]["legend"] = {**CONFIG["legend"], "orient": "bottom", "title": None}
        (OUT / f"{name}.pdf").write_bytes(vlc.vegalite_to_pdf(spec))
        print("wrote", name)
