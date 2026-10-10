#!/usr/bin/env python3
"""Generate the three primary figures from frozen machine-readable artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "reproduced" / "figures"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GRAY = "#5F6368"


def load(name: str) -> dict:
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / f"{stem}.pdf",
        metadata={"Creator": "scripts/manuscript_figures.py", "CreationDate": None},
    )
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)


def rounded_box(ax, xy, width, height, text, *, face, edge, fontsize=7.2):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.0,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end, *, color=GRAY):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, linewidth=1.0, color=color)
    )


def figure_workflow() -> None:
    fig = plt.figure(figsize=(7.15, 3.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.44, 0.56], wspace=0.08)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # Panel a: layered geometry and radial observation.
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    ax0.text(0.01, 0.955, "a", fontweight="bold", va="top", fontsize=10)
    ax0.text(0.11, 0.955, "Seven-layer forward model", fontweight="bold", va="top", fontsize=9.0)

    layers = [
        ("SC", "20 um", "#F6D7B0"),
        ("Epidermis", "80 um", "#F4B183"),
        ("Papillary dermis", "150 um", "#F8CBAD"),
        ("Upper plexus", "80 um", "#C00000"),
        ("Reticular dermis", "1500 um", "#F4CCCC"),
        ("Deep plexus", "80 um", "#7F0000"),
        ("Subcutis", "6090 um", "#FFE699"),
    ]
    x0, width = 0.12, 0.58
    y_top, h = 0.72, 0.075
    for i, (name, thickness, color) in enumerate(layers):
        y = y_top - i * h
        ax0.add_patch(Rectangle((x0, y), width, h, facecolor=color, edgecolor="white", linewidth=0.8))
        text_color = "white" if i in (3, 5) else "black"
        ax0.text(x0 + 0.02, y + h / 2, name, va="center", color=text_color, fontsize=6.5)
        ax0.text(x0 + width - 0.02, y + h / 2, thickness, va="center", ha="right", color=text_color, fontsize=6.3)
    ax0.text(x0 + width / 2, 0.155, "Nominal depths; drawing not to scale", ha="center", fontsize=6.5, color=GRAY)

    # Keep the source and detector-ring arcs clear of the left-aligned SC label.
    source_x = x0 + 0.18
    ax0.add_patch(Circle((source_x, 0.785), 0.016, facecolor=ORANGE, edgecolor="black", linewidth=0.7))
    arrow(ax0, (source_x, 0.77), (source_x, 0.735), color=ORANGE)
    ax0.text(source_x, 0.825, "540, 580, 633, 800 nm", ha="center", fontsize=6.5)
    for radius in (0.045, 0.08, 0.12):
        # The ring centres sit below the source so no arc passes through the marker.
        arc = plt.Circle((source_x, 0.715), radius, fill=False, edgecolor=BLUE, linewidth=0.8, alpha=0.9)
        arc.set_clip_path(Rectangle((0, 0.75), 1, 0.14, transform=ax0.transData))
        ax0.add_patch(arc)
    ax0.text(0.70, 0.76, "30 annular bins\n0-1.5 cm", va="center", fontsize=6.6, color=BLUE)
    ax0.text(0.12, 0.07, r"$\psi=[S_U,f_U,S_D,f_D,\mu_{a,\mathrm{DermR}},\mu_{a,\mathrm{SubQ}}]^\mathsf{T}$", fontsize=6.4)

    # Panel b: analysis flow and evidence boundaries.
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.text(0.01, 0.955, "b", fontweight="bold", va="top", fontsize=10)
    ax1.text(0.10, 0.955, "Frozen production analysis", fontweight="bold", va="top", fontsize=9.0)

    rounded_box(ax1, (0.05, 0.73), 0.29, 0.13, "32 streams\n6.4M photons\nper wavelength", face="#EAF2F8", edge=BLUE, fontsize=5.7)
    rounded_box(ax1, (0.39, 0.73), 0.24, 0.13, "$R,\\ J_\\psi$\n120 bins", face="#E8F6F3", edge=GREEN, fontsize=5.9)
    rounded_box(ax1, (0.68, 0.73), 0.25, 0.13, "$J_\\psi D$\nscaled coordinates", face="#FDEDEC", edge=RED, fontsize=5.9)
    arrow(ax1, (0.34, 0.795), (0.39, 0.795))
    arrow(ax1, (0.63, 0.795), (0.68, 0.795))

    branch_y = [0.49, 0.34, 0.19]
    labels = [
        ("Raw geometry", "SVD only", "#F2F3F4", GRAY),
        ("Diagonal Poisson", "likelihood sensitivity", "#FFF4D6", ORANGE),
        ("Fixed-launch multinomial", "prespecified headline", "#EAF2F8", BLUE),
    ]
    for y, (title, subtitle, face, edge) in zip(branch_y, labels):
        rounded_box(ax1, (0.10, y), 0.38, 0.105, f"{title}\n{subtitle}", face=face, edge=edge, fontsize=6.7)
        rounded_box(ax1, (0.60, y), 0.31, 0.105, "Singular spectrum\nand weakest direction" if y == branch_y[0] else "Fisher information\nand CRLB", face="white", edge=edge, fontsize=6.7)
        arrow(ax1, (0.47, y + 0.052), (0.61, y + 0.052), color=edge)
    arrow(ax1, (0.805, 0.73), (0.30, 0.595))
    ax1.text(0.06, 0.055, "Separate estimator audit: AD vs replay pMC vs independent reference", fontsize=6.5, color=PURPLE)
    ax1.plot([0.06, 0.94], [0.09, 0.09], color=PURPLE, linewidth=1.0)

    save(fig, "fig1_model_workflow")


def figure_spectrum() -> None:
    data = load("production_crb.json")["branches"]
    branches = ["raw", "diagonal_poisson", "multinomial_fixed_launch"]
    labels = ["Raw geometry", "Diagonal Poisson", "Fixed-launch multinomial"]
    colors = [GRAY, ORANGE, BLUE]
    markers = ["o", "s", "^"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.15, 3.25), gridspec_kw={"width_ratios": [0.47, 0.53]})
    fig.subplots_adjust(wspace=0.30)

    ranks = np.arange(1, 7)
    for key, label, color, marker in zip(branches, labels, colors, markers):
        s = np.asarray(data[key]["geometry"]["singular_values"], dtype=float)
        ax0.semilogy(ranks, s / s[0], marker=marker, color=color, linewidth=1.5, markersize=4.5, label=label)
    ax0.set_xlabel("Singular-value index")
    ax0.set_ylabel(r"Normalized singular value $s_i/s_1$")
    ax0.set_xticks(ranks)
    ax0.grid(axis="y", which="both", color="#D9D9D9", linewidth=0.6)
    ax0.legend(frameon=False, loc="lower left")
    ax0.set_title("a   Spectrum", loc="left", fontweight="bold")

    params = [r"$S_U$", r"$f_U$", r"$S_D$", r"$f_D$", r"$\mu_{a,DermR}$", r"$\mu_{a,SubQ}$"]
    x = np.arange(len(params))
    width = 0.24
    for i, (key, label, color) in enumerate(zip(branches, labels, colors)):
        near_null = data[key]["geometry"]["near_null_direction_z_from_svd"]
        vals = [near_null[k] for k in ("StO2_U", "f_U", "StO2_D", "f_D", "mua_DermR", "mua_SubQ")]
        ax1.bar(x + (i - 1) * width, vals, width, label=label, color=color, edgecolor="white", linewidth=0.4)
    ax1.axhline(0, color="black", linewidth=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(params, rotation=28, ha="right")
    ax1.set_ylabel(r"Weakest right-singular-vector coefficient in $z$")
    ax1.set_ylim(-0.23, 1.08)
    ax1.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax1.set_title("b   Dominant weak direction", loc="left", fontweight="bold")
    ax1.text(2, 1.015, "deep saturation", ha="center", va="bottom", color=RED, fontsize=7.4)

    save(fig, "fig2_identifiability_spectrum")


def figure_estimator_audit() -> None:
    targets = load("jacobian_validation_pilot.json")["targets"]
    wavelengths = [540, 580, 633, 800]
    categories = [(target, wl) for target in ("f_U", "StO2_D") for wl in wavelengths]
    # Explicit indexing keeps the wavelength-major order visible.
    z_ad = np.asarray(
        targets["f_U"]["inference"]["z_AD"] + targets["StO2_D"]["inference"]["z_AD"], dtype=float
    )
    z_pmc = np.asarray(
        targets["f_U"]["inference"]["z_pMC"] + targets["StO2_D"]["inference"]["z_pMC"], dtype=float
    )
    separation = np.abs(
        np.asarray(
            targets["f_U"]["inference"]["separation_z"]
            + targets["StO2_D"]["inference"]["separation_z"],
            dtype=float,
        )
    )
    x = np.arange(8)
    ticklabels = [str(w) for _target in ("f_U", "StO2_D") for w in wavelengths]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.15, 4.65), sharex=True, gridspec_kw={"hspace": 0.27})
    ax0.axhspan(-3, 3, color="#E8F6F3", alpha=0.9, zorder=0)
    ax0.axhline(3, color=RED, linestyle="--", linewidth=1.0)
    ax0.axhline(-3, color=RED, linestyle="--", linewidth=1.0)
    ax0.axhline(0, color="black", linewidth=0.6)
    ax0.plot(x - 0.08, z_ad, "o", color=BLUE, markersize=5, label="AD - independent reference")
    ax0.plot(x + 0.08, z_pmc, "s", color=ORANGE, markersize=4.7, label="Replay pMC - independent reference")
    ax0.set_ylabel("Paired score z")
    ax0.set_ylim(-3.35, 3.35)
    ax0.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax0.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )
    ax0.set_title(
        "a   Neither estimator is preferred against the independent reference",
        loc="left",
        fontweight="bold",
        pad=38,
    )
    ax0.text(7.45, 2.72, "frozen threshold", ha="right", va="top", color=RED, fontsize=7.2)

    ax1.semilogy(x, separation, "D", color=PURPLE, markersize=5)
    ax1.axhline(3, color=RED, linestyle="--", linewidth=1.0)
    ax1.set_ylabel(r"$|z|$ for AD - replay pMC")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ticklabels)
    ax1.set_ylim(2.5, 650)
    ax1.grid(axis="y", which="both", color="#E5E5E5", linewidth=0.6)
    ax1.set_title("b   The two same-stream estimators are detectably separated", loc="left", fontweight="bold", pad=8)

    for boundary in (3.5,):
        ax0.axvline(boundary, color="#9E9E9E", linewidth=0.8)
        ax1.axvline(boundary, color="#9E9E9E", linewidth=0.8)
    ax1.text(1.5, 0.91, r"$f_U$", transform=ax1.get_xaxis_transform(), ha="center", va="top", fontweight="bold")
    ax1.text(5.5, 0.91, r"$S_D$", transform=ax1.get_xaxis_transform(), ha="center", va="top", fontweight="bold")

    save(fig, "fig3_estimator_audit")


def main() -> int:
    style()
    figure_workflow()
    figure_spectrum()
    figure_estimator_audit()
    print(f"wrote manuscript figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
