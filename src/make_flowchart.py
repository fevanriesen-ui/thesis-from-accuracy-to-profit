"""
Generate flowchart_methodology.pdf in an academic black-and-white style.

Standalone (not part of main.py): the methodology flowchart is a static
diagram of the experimental design, not a data-dependent artefact, so it
is generated separately.

Usage:
    python src/make_flowchart.py   # writes flowchart_methodology.pdf at project root
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT_ROOT, "flowchart_methodology.pdf")

# Academic style: serif font, thin black borders, near-white boxes,
# light grey for the central data product and the evaluation step.
plt.rcParams.update({
    "font.family":   "serif",
    "font.serif":    ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":     10,
    "pdf.fonttype":  42,
    "ps.fonttype":   42,
})

EDGE       = "black"
LW_BOX     = 0.7
LW_ARROW   = 0.7
COL_WHITE  = "white"
COL_GREY   = "#EDEDED"
COL_PARENT = "#FAFAFA"


def box(ax, cx, cy, w, h, lines, face=COL_WHITE,
        font_first=10, font_rest=9, bold_first=True,
        rounding=0.04, lw=LW_BOX, edge=EDGE):
    """Draw an almost-rectangular box with multi-line text."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(patch)
    n = len(lines)
    spacing = 0.34
    total_h = (n - 1) * spacing
    start_y = cy + total_h / 2
    for i, line in enumerate(lines):
        weight = "bold" if (i == 0 and bold_first) else "normal"
        size = font_first if (i == 0 and bold_first) else font_rest
        ax.text(cx, start_y - i * spacing, line,
                ha="center", va="center",
                fontsize=size, fontweight=weight, color="black", zorder=3)


def arrow(ax, x1, y1, x2, y2, connectionstyle="arc3,rad=0"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=10,
        linewidth=LW_ARROW, color="black", zorder=1,
        connectionstyle=connectionstyle,
        shrinkA=2, shrinkB=3,
    )
    ax.add_patch(a)


def label(ax, x, y, text, rotation=0):
    ax.text(x, y, text, fontsize=8.5, color="black",
            ha="center", va="center", rotation=rotation,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.5),
            zorder=2)


# Figure ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 20)
ax.set_aspect("equal")
ax.axis("off")

# Row 1 — three raw data sources (three equal-width boxes side-by-side)
box(ax, 2.5, 18.4, 4.0, 1.7, [
    "TenneT Transparency Portal",
    "Measured Infeed (target)",
    "Settlement prices & regulation state",
], font_first=9.5, font_rest=8)
box(ax, 7.0, 18.4, 4.0, 1.7, [
    "KNMI station De Bilt (260)",
    "Air temperature",
    "Wind speed",
], font_first=9.5, font_rest=8)
box(ax, 11.5, 18.4, 4.0, 1.7, [
    "holidays Python package",
    "Dutch public holidays",
    "(is_holiday feature)",
], font_first=9.5, font_rest=8)

# Row 2 — merge & features
box(ax, 7.0, 15.7, 10.5, 1.7, [
    "Merge & feature engineering",
    "Inner-join on PTU timestamp  ·  DST de-duplication",
    "3 autoregressive lags + 6 calendar + 2 weather = 11 predictors",
])

# Row 3 — dataset (light grey emphasis)
box(ax, 7.0, 13.4, 7.2, 1.1, [
    "Final dataset: 139,568 PTUs  ×  19 columns",
], face=COL_GREY)

# Row 4 — splits (with en-dash for date range, matching thesis text)
box(ax, 2.6, 10.7, 3.8, 1.7, [
    "Training",
    "2022–2023",
    "n = 69,400",
])
box(ax, 7.0, 10.7, 3.8, 1.7, [
    "Validation",
    "2024",
    "n = 35,132",
])
box(ax, 11.4, 10.7, 3.8, 1.7, [
    "Test (held-out)",
    "2025",
    "n = 35,036",
])

# Row 5 — parent group around two model boxes
# Title uses a colon (no em-dash) to match global thesis style.
box(ax, 7.0, 7.3, 10.0, 2.4, [
    "Two XGBoost models: identical except the loss function",
    "Same features, hyperparameters, training data, early-stopping criterion",
    "",
    "",
], face=COL_PARENT, font_first=10, font_rest=8.5)

box(ax, 4.7, 6.55, 3.4, 0.85, [
    "MSE baseline",
], face=COL_WHITE, font_first=9, bold_first=True)
box(ax, 9.3, 6.55, 4.0, 0.85, [
    "Cost-sensitive asymmetric",
], face=COL_WHITE, font_first=9, bold_first=True)

# Row 6 — evaluation (light grey emphasis)
box(ax, 7.0, 3.4, 10.0, 2.4, [
    "Predict on 2025 test  &  evaluate",
    "Realised imbalance cost (RQ1)",
    "Forecast bias & regime-conditional analysis (RQ2)",
    "MAE/RMSE trade-off  ·  Wilcoxon & paired t-test (RQ3)",
], face=COL_GREY, font_first=10, font_rest=9)

# Arrows ----------------------------------------------------------------------
# Three sources → merge (outer two angle inward, middle goes straight down)
arrow(ax, 2.5, 17.55, 4.5, 16.55)
arrow(ax, 7.0, 17.55, 7.0, 16.55)
arrow(ax, 11.5, 17.55, 9.5, 16.55)
# Merge → dataset
arrow(ax, 7.0, 14.85, 7.0, 13.95)
# Dataset → splits
arrow(ax, 7.0, 12.85, 2.6, 11.55)
arrow(ax, 7.0, 12.85, 7.0, 11.55)
arrow(ax, 7.0, 12.85, 11.4, 11.55)
# Train → models
arrow(ax, 2.6, 9.85, 3.8, 8.5)
label(ax, 2.95, 9.15, "fit")
# Val → models
arrow(ax, 7.0, 9.85, 7.0, 8.5)
label(ax, 7.0, 9.2, "early-stop on RMSE")
# Models → eval
arrow(ax, 7.0, 6.1, 7.0, 4.6)
# Test → eval (route around the right side of the Models box)
test_to_eval = FancyArrowPatch(
    (13.0, 9.85),
    (12.0, 3.4),
    arrowstyle="-|>", mutation_scale=10,
    linewidth=LW_ARROW, color="black", zorder=1,
    connectionstyle="angle,angleA=-90,angleB=0,rad=6",
    shrinkA=2, shrinkB=3,
)
ax.add_patch(test_to_eval)
label(ax, 13.3, 7.0, "evaluate", rotation=90)

plt.tight_layout(pad=0.2)
plt.savefig(OUT, format="pdf", bbox_inches="tight", pad_inches=0.1)
plt.close()
print(f"Saved: {OUT}")
