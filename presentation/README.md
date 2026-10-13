# Progress-update deck (Beamer, MIT Sloan style)

6 main slides (replication, calibration ×2, fed funds futures, Kalshi vs. futures, lead–lag) plus an appendix.

| File | What it is |
|---|---|
| `main.tex` | The slides. Main deck first, appendix after `\sloanappendix`. |
| `beamerthemesloan.sty` | Theme: colors, section bar at the top, footer with source and progress bar. |
| `figures/*.pdf` | Charts exported from the marimo notebooks by `export_figures.py`. |
| `data/*.dat` | Data for the charts drawn in LaTeX (pgfplots): replication, Sep 2024 example, futures gap. |
| `export_figures.py` | Exports the marimo charts, recolored to the deck palette (Kalshi red, futures charcoal). |
| `build_data.py` | Regenerates the computed `data/*.dat` files. |
| `main.pdf` | Compiled deck. |

## Open in Overleaf

1. Zip this folder (or use `presentation-overleaf.zip` if you were given one).
2. In Overleaf: **New Project → Upload Project** and choose the zip.
3. Compiler: pdfLaTeX (the default). `main.tex` is the main document.

## Editing

- Each slide's `\source{...}` puts the notebook it comes from in the footer.
- `\section{...}` names drive the navigation bar at the top. Keep main-deck sections short.
- Colors are defined once at the top of `beamerthemesloan.sty` (`sloanred`, `bench`, ...).
- Author names are on line 10 of `main.tex`. Check them.
- Numbers in the text (key figures, tables) were copied from the notebooks' saved outputs; update them
  by hand after a rerun.

## Refreshing the charts after rerunning a notebook

Charts are read from the saved session of each notebook (`__marimo__/session/<notebook>.py.json`),
so rerun and save the notebook in marimo first. Then, from the repository root:

```
uv run presentation/export_figures.py
uv run presentation/build_data.py
```

`export_figures.py` lists which notebook cell each figure comes from (`FIGURES`). If a cell id changes,
update it there.
