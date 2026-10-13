# Progress-update deck (Beamer, MIT Sloan style)

6 main slides plus an appendix. Every chart is drawn with pgfplots from the small text files in `data/`,
so numbers and labels can be edited directly in Overleaf.

| File | What it is |
|---|---|
| `main.tex` | The slides. Main deck first, appendix after `\sloanappendix`. |
| `beamerthemesloan.sty` | Theme: colors, section bar at the top, footer with source and progress bar. |
| `data/*.dat` | Chart data. |
| `build_data.py` | Regenerates the data files computed from raw data (see below). |
| `main.pdf` | Compiled deck. |

## Open in Overleaf

1. Zip this folder (or use `presentation-overleaf.zip` if you were given one).
2. In Overleaf: **New Project → Upload Project** and choose the zip.
3. Compiler: pdfLaTeX (the default). `main.tex` is the main document.

## Editing

- Each slide's `\source{...}` puts the notebook it comes from in the footer.
- `\section{...}` names drive the navigation bar at the top. Keep main-deck sections short.
- Colors are defined once at the top of `beamerthemesloan.sty` (`sloanred`, `bench`, ...).
- Author names are on line 8 of `main.tex`. Check them.

## Where the numbers come from

- Computed from raw data by `build_data.py`: `fed_monthly_volume`, `dkw_mae_curve`, `hairs`,
  `realized_target`, `sep24_*`. Run from the repository root: `uv run presentation/build_data.py`.
- Copied from the last saved run of the marimo notebooks (`__marimo__/session/*.json`): all other files.
  `02_calibration_kalshi.py` → `fed_*`; `03_calibration_macro.py` → `macro_*`;
  `03_fedfunds_vs_kalshi.py` → `ff_vs_kalshi_*`; `kalshi_fed/data/validation.json` → `dkw_horizon_bars`.
  Rerunning a notebook does not update these; copy the new values in by hand.
