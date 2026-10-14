# Redraw the lead-lag cross-correlation for the deck so an outside viewer can read it unaided.
# Values from 04_leadlag_fedfunds_kalshi.py (5-minute changes, all CME hours), recomputed with xcorr.py.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAGS = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
CORR = [-0.0147, 0.0147, 0.0196, 0.0137, -0.015, -0.031, 0.2193, 0.1127, 0.0136, 0.0456, 0.034, 0.0344, -0.017]
CRIMSON, SLATE, LIGHT, INK, MUTED = "#A31F34", "#4B5058", "#C3C7CD", "#1E2126", "#6B7079"

plt.rcParams.update({"font.family": "Arial", "font.size": 13})
fig, ax = plt.subplots(figsize=(7.9, 4.0), dpi=220)
colors = [LIGHT if k < 0 else SLATE if k == 0 else CRIMSON for k in LAGS]
ax.bar(LAGS, CORR, width=4.2, color=colors, zorder=3)
ax.axhline(0, color=INK, lw=0.8, zorder=4)
ax.axvline(-2.5, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=2)
ax.axvline(2.5, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=2)

ax.set_xticks(LAGS, [f"{k:+d}" if k else "0" for k in LAGS])
ax.set_xlim(-33, 33)
ax.set_ylim(-0.05, 0.27)
ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
ax.set_ylabel("Correlation", color=INK)
ax.set_xlabel("Minutes from the futures' 5-minute change to Kalshi's", color=INK)
ax.tick_params(colors=MUTED, length=0)
ax.grid(axis="y", color="#E6E8EB", lw=0.8, zorder=0)
for s in ax.spines.values():
    s.set_visible(False)

# Region labels make the sign convention explicit.
ax.text(-17.5, 0.262, "Kalshi moved first", ha="center", va="center", color=MUTED, fontsize=12, fontweight="bold")
ax.text(17.5, 0.262, "Futures moved first", ha="center", va="center", color=CRIMSON, fontsize=12, fontweight="bold")
ax.annotate("Same 5-minute window: 0.22", (-2.1, 0.2193), xytext=(-6, 0.205), ha="right", va="center",
            color=SLATE, fontsize=11, fontweight="bold", arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.8))
ax.annotate("0.11: Kalshi still\ncatching up 5 min later", (5, 0.1127), xytext=(8, 0.15), ha="left", va="center",
            color=CRIMSON, fontsize=11, arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=0.8))
ax.text(-17.5, -0.042, "all near zero", ha="center", va="center", color=MUTED, fontsize=10.5, style="italic")

fig.tight_layout(pad=0.4)
fig.savefig("leadlag_v2.png")
