"""Shared plotting style.

Figures are static PNG/PDF for a written report, so a single validated
light-surface palette is used throughout. Categorical hues are assigned in a
fixed order and never cycled; charts with a single series carry no legend
because the title names the series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette, light surface, fixed slot order.
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#84837c"
GRID = "#e4e3de"

# Sequential ramp (one hue, light -> dark) for magnitude encodings.
SEQUENTIAL = ["#d6e5f7", "#a7c8ee", "#6fa4e2", "#2a78d6", "#1c5296", "#123561"]

STATUS_GOOD = "#1baf7a"
STATUS_CRITICAL = "#e34948"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "font.family": "sans-serif",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.labelsize": 10,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": TEXT_SECONDARY,
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
        }
    )


def clean_axes(ax, x_grid: bool = False, y_grid: bool = True) -> None:
    """Recessive axes: no top/right spines, grid on one direction only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(x_grid)
    ax.yaxis.grid(y_grid)


def stat_tile(
    ax, value: str, label: str, sub: str = "", color: str = SERIES[0], size: int = 26
) -> None:
    """A hero number. Used where the data has no distribution worth plotting."""
    ax.axis("off")
    ax.text(0.5, 0.66, value, ha="center", va="center", fontsize=size,
            fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.42, label, ha="center", va="center", fontsize=10,
            color=TEXT_PRIMARY, transform=ax.transAxes, linespacing=1.4)
    if sub:
        ax.text(0.5, 0.20, sub, ha="center", va="center", fontsize=8.5,
                color=TEXT_MUTED, transform=ax.transAxes, linespacing=1.5)


def suptitle(fig, text: str) -> None:
    # Trailing newline reserves a gap so the figure title never collides with
    # the first row of axes titles under constrained layout.
    fig.suptitle(text + "\n", fontsize=12.5, fontweight="bold", color=TEXT_PRIMARY)


def caption(fig, text: str) -> None:
    """A figure-level caption stating scope (full dataset vs subset)."""
    import textwrap

    fig.supxlabel("\n".join(textwrap.wrap(text, 150)), fontsize=8.5,
                  color=TEXT_MUTED, ha="center")


def save(fig, path: str | Path, also_pdf: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[figure] {path.name}")
    return path


def thousands(value, _pos=None) -> str:
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"
