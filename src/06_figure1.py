"""Figure 1: industry composition of F1 sponsorship, 2018-2025.

Output: output/figure1_composition.svg and .png (one figure in two formats).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import OUTPUT, load_sponsors

SERIES = {
    "Automotive": "#8C1C13",
    "Industrial & Aerospace": "#BF6B04",
    "Technology": "#1B5E20",
    "Financial Services": "#14375E",
}
LABELS = {"Industrial & Aerospace": "Industrial/Aerospace"}
# Two series end within a point of each other in 2025; offset their labels.
LABEL_OFFSET = {"Financial Services": 7, "Industrial & Aerospace": -7}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 11,
})


def main():
    sp = load_sponsors()
    mix = pd.crosstab(sp["Season"], sp["Industry"], normalize="index") * 100
    years = sorted(mix.index)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, colour in SERIES.items():
        vals = [mix.loc[y, name] for y in years]
        ax.plot(years, vals, color=colour, linewidth=1.9, marker="o", markersize=4, zorder=3)
        ax.annotate(LABELS.get(name, name), xy=(years[-1], vals[-1]),
                    xytext=(6, LABEL_OFFSET.get(name, 0)), textcoords="offset points",
                    color=colour, fontsize=9.5, va="center")

    ax.set_xlabel("Season", labelpad=8)
    ax.set_title("Selected industries in recorded F1 sponsorships", fontsize=12, pad=14)
    ax.set_ylabel("Share of all recorded sponsorships (%)", labelpad=8)
    ax.set_xlim(years[0] - 0.3, years[-1] + 1.9)
    ax.set_ylim(0, 30)
    ax.set_xticks(years)
    ax.grid(axis="y", color="#CCCCCC", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout(pad=1.4)
    OUTPUT.mkdir(exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(OUTPUT / f"figure1_composition.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
