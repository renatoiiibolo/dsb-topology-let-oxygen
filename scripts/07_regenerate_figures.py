#!/usr/bin/env python3
"""
================================================================================
07_regenerate_figures.py
================================================================================
Regenerates ALL analysis figures from saved output files produced by the
updated pipeline:
  02_ph_topology_analysis.py    →  analysis/ph/
  03_compute_features.py
  04_build_feature_matrix.py    →  analysis/feature_matrix.csv
  05_random_forest.py           →  analysis/rf/
  06_additional_analyses.py     →  analysis/additional/

No re-running of any ML or PH pipeline is required.  Edit the CONFIGURATION
section at the top to change colours, fonts, DPI, or figure sizes.

DATA REQUIRED
-------------
  analysis/feature_matrix.csv
  analysis/rf/results_summary.json
  analysis/rf/ablation/ablation_results.json       (optional)
  analysis/ph/ph_summary.json
  analysis/ph/wasserstein_H0.npy                   (for UMAP — optional)
  analysis/ph/wasserstein_H1.npy                   (for UMAP — optional)
  analysis/ph/diagrams/*.npz                       (for landscapes — optional)
  analysis/additional/cross_modality_correlation.json
  analysis/additional/single_modality_o2_accuracy.json
  analysis/additional/effect_sizes.json
  analysis/additional/condition_pca.json
  analysis/additional/sobp_effect_sizes.json

OUTPUTS  (written to analysis/figures/)
-------
  confusion/
    task1_o2_{particle_key}_cm.png             7 files
    task2_particle_cm.png
    task3_joint_cm.png                         49×49 (annotations suppressed)
    task4_sobp_{species}_cm.png                3 files
  importance/
    task1_o2_{particle_key}_importance.png     7 files
    task2_particle_importance.png
    task3_joint_importance.png
    task4_sobp_{species}_importance.png        3 files
  ablation/
    modality_ablation.png
  summary/
    task1_o2_balanced_accuracy.png
    task1_o2_per_class_recall.png
    task4_sobp_summary.png
  ph/
    umap_H{0,1}.png                            optional
    within_between_H{0,1}.png
    condition_heatmap_H{0,1}.png
    landscape_H{0,1}.png                       optional
  additional/
    cross_modality_correlation.png             Stage 2
    single_modality_o2_accuracy.png            Stage 3
    let_o2_modality_profiles.png               Stage 4
    feature_o2_effect_sizes.png                Stage 5a
    feature_particle_effect_sizes.png          Stage 5b
    condition_pca.png                          Stage 6
    condition_pca_scree.png                    Stage 6
    o2_gradient_profiles.png                   Stage 7
    sobp_effect_sizes.png                      Stage 8
    condition_dendrogram.png                   Stage 9

USAGE
-----
  python 07_regenerate_figures.py
  python 07_regenerate_figures.py --basedir /path/to/project
  python 07_regenerate_figures.py --skip-umap
  python 07_regenerate_figures.py --skip-landscapes
  python 07_regenerate_figures.py --skip-additional
  python 07_regenerate_figures.py --top-n 30

DEPENDENCIES
------------
  numpy, pandas, scipy, scikit-learn, matplotlib
  umap-learn    (optional — UMAP figures only)
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, FancyBboxPatch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import umap as umap_module
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        CONFIGURATION                                    ║
# ║  Edit this block freely. Nothing below needs to change for aesthetics.  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Output ───────────────────────────────────────────────────────────────────
DPI        = 600
OUT_FORMAT = "png"   # "png" or "pdf"

# ── Font ─────────────────────────────────────────────────────────────────────
import matplotlib.font_manager as fm

_HN_DIR = Path.home() / ".local" / "share" / "fonts" / "HelveticaNeue"
if _HN_DIR.exists():
    for _ttf in sorted(_HN_DIR.glob("*.ttf")):
        fm.fontManager.addfont(str(_ttf))

FONT_FAMILY     = "sans-serif"
FONT_SANS_SERIF = ["Helvetica Neue", "Helvetica", "Arial",
                   "Liberation Sans", "DejaVu Sans"]
FONT_SIZE       = 9
TITLE_SIZE      = 11
SUBTITLE_SIZE   = 8
LABEL_SIZE      = 9
TICK_SIZE       = 8
LEGEND_SIZE     = 8

# ── Colour palette — "Amalfi Coast at 2pm in July" ───────────────────────────
#
# The scene: standing at the waterline on an Amalfi beach.
#   Behind:   chili-red limestone cliffs crumbling into warm sable sand.
#   Feet:     piscine turquoise shallows.
#   Horizon:  marine deep-blue open Tyrrhenian sea.
#   Overhead: the sun burning lemon-gold through coastal haze.
#   Cliff:    maquis scrub — rosemary, oregano, sun-bleached sage.
#   Beach:    warm sable sand between the cliff base and the tideline.
#   Towel:    a cut melon, its flesh sun-warmed.
#
# Particle configurations (7)
#   electron mono  → lemon-gold      #F09714  afternoon sun burning on the water
#   proton  pSOBP  → marine deep     #37657E  open Tyrrhenian sea beyond the bay
#   proton  dSOBP  → marine dusk     #2A5060  same sea, hour later
#   helium  pSOBP  → maquis sage     #6B8C5A  rosemary & oregano on the cliff path
#   helium  dSOBP  → sun-dried maquis stalks  #B5956A
#   carbon  pSOBP  → chili cliff     #CD5F00  sun-baked Amalfi limestone
#   carbon  dSOBP  → bougainvillea   #9B5878  flowers cascading on the villa wall
#
# O₂ gradient (sea depth: deep offshore → shallows → pale seafoam — 7 stops)
#   21.0%   → deep offshore  #1D4E63
#    5.0%   → near offshore  #2A6070
#    2.1%   → marine         #37657E
#    0.5%   → piscine        #508799
#    0.1%   → pale piscine   #6FA3AE
#   0.021%  → seafoam light  #8DC0C9
#   0.005%  → seafoam pale   #A8D4E0
#
# Modalities (seven distinct Amalfi landscape elements)
#   Spatial Distribution      → marine        #37657E
#   Radial Track Structure    → piscine        #508799
#   Local Energy Heterogeneity→ maquis sage   #6B8C5A
#   Dose Distribution         → deep offshore  #1D4E63
#   Genomic Location          → sable sand    #C2A387
#   Damage Complexity         → lemon-gold    #F09714
#   Topological Summaries     → chili cliff   #CD5F00
#
# Heatmap gradient: seafoam white → sable sand → chili cliff → deep marine
# Chance-level reference line: melon flesh #D4845A

PARTICLE_COLORS: Dict[str, str] = {
    "electron_mono": "#F09714",
    "proton_psobp":  "#37657E",
    "proton_dsobp":  "#2A5060",
    "helium_psobp":  "#6B8C5A",
    "helium_dsobp":  "#B5956A",
    "carbon_psobp":  "#CD5F00",
    "carbon_dsobp":  "#9B5878",
}

SPECIES_COLORS: Dict[str, str] = {
    "electron": "#F09714",
    "proton":   "#37657E",
    "helium":   "#6B8C5A",
    "carbon":   "#CD5F00",
}

O2_COLORS: List[str] = [
    "#1D4E63",   # 21.0%
    "#2A6070",   #  5.0%
    "#37657E",   #  2.1%
    "#508799",   #  0.5%
    "#6FA3AE",   #  0.1%
    "#8DC0C9",   # 0.021%
    "#A8D4E0",   # 0.005%
]

MODALITY_COLORS: Dict[str, str] = {
    "Spatial Distribution":       "#37657E",
    "Radial Track Structure":     "#508799",
    "Local Energy Heterogeneity": "#6B8C5A",
    "Dose Distribution":          "#1D4E63",
    "Genomic Location":           "#C2A387",
    "Damage Complexity":          "#F09714",
    "Topological Summaries":      "#CD5F00",
}

COLOR_WITHIN  = "#508799"
COLOR_BETWEEN = "#CD5F00"
COLOR_CHANCE  = "#D4845A"

CM_CMAP = LinearSegmentedColormap.from_list(
    "amalfi_cm",
    ["#FFFFFF", "#C2A387", "#CD5F00", "#1D4E63"],
    N=256,
)

# Diverging correlation map: deep marine ← 0 → chili cliff
CORR_CMAP = LinearSegmentedColormap.from_list(
    "amalfi_corr",
    ["#1D4E63", "#FFFFFF", "#CD5F00"],
    N=256,
)

# Effect-size colourmap (low → high): seafoam pale → chili cliff
ETA_CMAP = LinearSegmentedColormap.from_list(
    "amalfi_eta",
    ["#A8D4E0", "#508799", "#CD5F00", "#1D4E63"],
    N=256,
)

STRIP_FILL = "#E8DDD1"
STRIP_TEXT = "#1A1A1A"

# ── Axis / grid style ─────────────────────────────────────────────────────────
GRID_COLOR  = "#EBEBEB"
SPINE_COLOR = "#BBBBBB"
TICK_COLOR  = "#555555"
TEXT_COLOR  = "#1A1A1A"
SUB_COLOR   = "#666666"

# ── Figure sizes (width × height, inches) ────────────────────────────────────
SIZE_CM_7      = (7.5,  7.0)
SIZE_CM_2      = (4.5,  4.0)
SIZE_CM_49     = (22.0, 20.0)
SIZE_IMP_W     = 9.5
SIZE_ABL_PANEL = (4.2, 5.8)
SIZE_SUMMARY   = (8.0,  5.0)
SIZE_SOBP_SUM  = (7.5,  4.8)
SIZE_RECALL_W  = 3.8
SIZE_RECALL_H  = 5.0
SIZE_UMAP      = (13.0, 5.5)
SIZE_VIOLIN    = (5.5,  4.8)
SIZE_HEATMAP   = (14.0, 12.5)
SIZE_LAND_W    = 3.8
SIZE_LAND_H    = 4.2
# Additional analyses figure sizes
SIZE_CORR_HEATMAP  = (8.0, 7.0)    # 7×7 cross-modality correlation
SIZE_ACC_MATRIX    = (9.0, 6.5)    # 7-modality × 7-particle accuracy matrix
SIZE_LET_PROFILES  = (14.0, 7.0)   # LET–O₂ modality profile panel
SIZE_ETA_BAR_W     = 10.0          # effect-size ranked bar chart width
SIZE_PCA           = (12.0, 5.5)   # PCA scatter + scree side by side
SIZE_O2_GRADIENT   = (14.0, 8.5)   # O₂ gradient profile 7-panel
SIZE_SOBP_EFFECT   = (13.0, 5.5)   # SOBP effect size 3-panel
SIZE_DENDROGRAM    = (14.0, 6.0)   # condition dendrogram

# ── Importance plot ───────────────────────────────────────────────────────────
TOP_N = 25

# ── UMAP ─────────────────────────────────────────────────────────────────────
UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.1
UMAP_RANDOM_STATE = 42

# ── Persistence landscape ─────────────────────────────────────────────────────
PH_MAX_DIST     = 9.3
LANDSCAPE_N_T   = 200
LANDSCAPE_K_MAX = 3

CM_ANNOTATE_MAX_N = 15


# ══════════════════════════════════════════════════════════════════════════════
# DERIVED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PARTICLE_KEY_ORDER: List[str] = [
    "electron_mono",
    "proton_psobp",
    "proton_dsobp",
    "helium_psobp",
    "helium_dsobp",
    "carbon_psobp",
    "carbon_dsobp",
]

PARTICLE_LABELS: Dict[str, str] = {
    "electron_mono": "Electron mono (0.2 keV/µm)",
    "proton_psobp":  "Proton pSOBP (4.6 keV/µm)",
    "proton_dsobp":  "Proton dSOBP (8.1 keV/µm)",
    "helium_psobp":  "Helium pSOBP (10 keV/µm)",
    "helium_dsobp":  "Helium dSOBP (30 keV/µm)",
    "carbon_psobp":  "Carbon pSOBP (40.9 keV/µm)",
    "carbon_dsobp":  "Carbon dSOBP (70.7 keV/µm)",
}

PARTICLE_SHORT: Dict[str, str] = {
    "electron_mono": "e⁻",
    "proton_psobp":  "p⁺p",
    "proton_dsobp":  "p⁺d",
    "helium_psobp":  "Hep",
    "helium_dsobp":  "Hed",
    "carbon_psobp":  "Cp",
    "carbon_dsobp":  "Cd",
}

PARTICLE_SPECIES: Dict[str, str] = {
    "electron_mono": "electron",
    "proton_psobp":  "proton",
    "proton_dsobp":  "proton",
    "helium_psobp":  "helium",
    "helium_dsobp":  "helium",
    "carbon_psobp":  "carbon",
    "carbon_dsobp":  "carbon",
}

PARTICLE_LET: Dict[str, float] = {
    "electron_mono": 0.2,
    "proton_psobp":  4.6,
    "proton_dsobp":  8.1,
    "helium_psobp":  10.0,
    "helium_dsobp":  30.0,
    "carbon_psobp":  40.9,
    "carbon_dsobp":  70.7,
}

DIR_TO_PARTICLE_KEY: Dict[str, str] = {
    "electron_0.2":  "electron_mono",
    "proton_4.6":    "proton_psobp",
    "proton_8.1":    "proton_dsobp",
    "helium_10.0":   "helium_psobp",
    "helium_30.0":   "helium_dsobp",
    "carbon_40.9":   "carbon_psobp",
    "carbon_70.7":   "carbon_dsobp",
}

O2_ORDERED: List[str] = [
    "21.0", "5.0", "2.1", "0.5", "0.1", "0.021", "0.005"
]

O2_LABELS: Dict[str, str] = {
    "21.0":  "21.0% (Normoxic)",
    "5.0":   "5.0% (T. Norm.)",
    "2.1":   "2.1% (Mild)",
    "0.5":   "0.5%",
    "0.1":   "0.1% (Severe)",
    "0.021": "0.021% (Anoxic)",
    "0.005": "0.005% (True Anox.)",
}
O2_SHORT: Dict[str, str] = {
    "21.0":  "21.0",
    "5.0":   "5.0",
    "2.1":   "2.1",
    "0.5":   "0.5",
    "0.1":   "0.1",
    "0.021": "0.021",
    "0.005": "0.005",
}

O2_COLOR_MAP: Dict[str, str] = dict(zip(O2_ORDERED, O2_COLORS))

MODALITY_ORDER: List[str] = list(MODALITY_COLORS.keys())
MODALITY_SHORT: List[str] = ["m1", "m2", "m3", "m4", "m5", "m6", "m7"]
MODALITY_PREFIX: Dict[str, str] = {
    "m1_": "Spatial Distribution",
    "m2_": "Radial Track Structure",
    "m3_": "Local Energy Heterogeneity",
    "m4_": "Dose Distribution",
    "m5_": "Genomic Location",
    "m6_": "Damage Complexity",
    "m7_": "Topological Summaries",
}

CONDITIONS: List[Tuple[str, str]] = [
    (pk, o2)
    for pk in PARTICLE_KEY_ORDER
    for o2 in O2_ORDERED
]

CONDITION_LABELS_SHORT: List[str] = [
    f"{PARTICLE_SHORT[pk]} {O2_SHORT[o2]}"
    for pk, o2 in CONDITIONS
]

SOBP_SPECIES: List[str] = ["proton", "helium", "carbon"]

CHANCE_O2:       float = 1.0 / len(O2_ORDERED)
CHANCE_PARTICLE: float = 1.0 / len(PARTICLE_KEY_ORDER)
CHANCE_JOINT:    float = 1.0 / len(CONDITIONS)
CHANCE_SOBP:     float = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL MATPLOTLIB STYLE
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":           FONT_FAMILY,
    "font.sans-serif":       FONT_SANS_SERIF,
    "font.size":             FONT_SIZE,
    "text.color":            TEXT_COLOR,
    "axes.titlesize":        TITLE_SIZE,
    "axes.titleweight":      "bold",
    "axes.titlecolor":       TEXT_COLOR,
    "axes.labelsize":        LABEL_SIZE,
    "axes.labelcolor":       TEXT_COLOR,
    "xtick.labelsize":       TICK_SIZE,
    "ytick.labelsize":       TICK_SIZE,
    "xtick.color":           TICK_COLOR,
    "ytick.color":           TICK_COLOR,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "xtick.major.size":      3.5,
    "ytick.major.size":      3.5,
    "xtick.major.width":     0.7,
    "ytick.major.width":     0.7,
    "legend.fontsize":       LEGEND_SIZE,
    "legend.title_fontsize": LEGEND_SIZE + 0.5,
    "legend.framealpha":     0.95,
    "legend.edgecolor":      "#CCCCCC",
    "figure.facecolor":      "white",
    "axes.facecolor":        "white",
    "savefig.facecolor":     "white",
    "axes.grid":             True,
    "axes.grid.which":       "major",
    "grid.color":            GRID_COLOR,
    "grid.linewidth":        0.6,
    "grid.linestyle":        "-",
    "axes.axisbelow":        True,
    "axes.edgecolor":        SPINE_COLOR,
    "axes.linewidth":        0.7,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "figure.dpi":            150,
    "savefig.dpi":           DPI,
})


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=TICK_COLOR, length=3.5, width=0.7)


def _title_subtitle(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold",
                 color=TEXT_COLOR, loc="left", pad=16)
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes,
                fontsize=SUBTITLE_SIZE, color=SUB_COLOR,
                style="italic", ha="left", va="bottom")


def _strip_header(ax: plt.Axes, label: str) -> None:
    ax.add_patch(FancyBboxPatch(
        (0, 1.02), 1, 0.10,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=STRIP_FILL, zorder=5, clip_on=False,
        transform=ax.transAxes,
    ))
    ax.text(0.5, 1.07, label, transform=ax.transAxes,
            ha="center", va="center",
            fontsize=FONT_SIZE + 0.5, fontweight="bold",
            color=STRIP_TEXT, zorder=6)


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {path.name}")


def _feat_color(feat: str) -> str:
    for pfx, name in MODALITY_PREFIX.items():
        if feat.startswith(pfx):
            return MODALITY_COLORS[name]
    return "#AAAAAA"


def _modality_of(feat: str) -> str:
    for pfx, name in MODALITY_PREFIX.items():
        if feat.startswith(pfx):
            return name
    return "Unknown"


def _norm_o2(v) -> str:
    try:
        f = float(v)
        for s in O2_ORDERED:
            if abs(f - float(s)) < 1e-9:
                return s
    except (ValueError, TypeError):
        pass
    return str(v)


def _pk_sort_key(pk: str) -> int:
    try:
        return PARTICLE_KEY_ORDER.index(pk)
    except ValueError:
        return 99


def _o2_sort_key(o2: str) -> int:
    try:
        return O2_ORDERED.index(str(o2))
    except ValueError:
        return 99


def _run_sort_key(s: pd.Series) -> pd.Series:
    def _k(x):
        if x in PARTICLE_KEY_ORDER:
            return PARTICLE_KEY_ORDER.index(x)
        if x in O2_ORDERED:
            return O2_ORDERED.index(x)
        try:
            return int(x)
        except (ValueError, TypeError):
            return 0
    return s.map(_k)


def _load_json_optional(path: Path) -> Optional[Dict]:
    """Load a JSON file if it exists; return None with a warning otherwise."""
    if not path.exists():
        logger.warning(f"  Not found (skipping): {path.name}")
        return None
    with open(path) as fh:
        return json.load(fh)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(base_dir: Path):
    rf_dir  = base_dir / "analysis" / "rf"
    ph_dir  = base_dir / "analysis" / "ph"
    add_dir = base_dir / "analysis" / "additional"

    # ── RF results ────────────────────────────────────────────────────────────
    results_path = rf_dir / "results_summary.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"results_summary.json not found at {results_path}\n"
            "  Run 05_random_forest.py first."
        )
    with open(results_path) as fh:
        results = json.load(fh)

    ablation: Dict = results.get("ablation", {})
    abl_path = rf_dir / "ablation" / "ablation_results.json"
    if not ablation and abl_path.exists():
        with open(abl_path) as fh:
            ablation = json.load(fh)

    # ── Feature matrix ────────────────────────────────────────────────────────
    fm_path = base_dir / "analysis" / "feature_matrix.csv"
    if not fm_path.exists():
        raise FileNotFoundError(
            f"feature_matrix.csv not found at {fm_path}\n"
            "  Run 04_build_feature_matrix.py first."
        )
    fm = pd.read_csv(fm_path)
    fm["o2"] = fm["o2"].apply(_norm_o2)
    feat_cols = [c for c in fm.columns
                 if any(c.startswith(pfx) for pfx in MODALITY_PREFIX)]
    logger.info(f"  {len(feat_cols)} feature columns, "
                f"{len(fm)} rows loaded from feature_matrix.csv")

    # ── PH summary ────────────────────────────────────────────────────────────
    ph_path = ph_dir / "ph_summary.json"
    if not ph_path.exists():
        raise FileNotFoundError(
            f"ph_summary.json not found at {ph_path}\n"
            "  Run 02_ph_topology_analysis.py first."
        )
    with open(ph_path) as fh:
        ph_summary = json.load(fh)

    # ── Wasserstein matrices (optional) ───────────────────────────────────────
    wass: Dict[int, np.ndarray] = {}
    for deg in [0, 1]:
        p = ph_dir / f"wasserstein_H{deg}.npy"
        if p.exists():
            wass[deg] = np.load(p).astype(float)
            logger.info(f"  wasserstein_H{deg}.npy loaded: {wass[deg].shape}")

    # ── Additional analyses JSONs (optional) ──────────────────────────────────
    additional: Dict[str, Optional[Dict]] = {}
    json_files = {
        "cross_modality_correlation": add_dir / "cross_modality_correlation.json",
        "single_modality_o2_accuracy": add_dir / "single_modality_o2_accuracy.json",
        "effect_sizes": add_dir / "effect_sizes.json",
        "condition_pca": add_dir / "condition_pca.json",
        "sobp_effect_sizes": add_dir / "sobp_effect_sizes.json",
    }
    for key, path in json_files.items():
        additional[key] = _load_json_optional(path)

    return results, ablation, feat_cols, fm, ph_summary, wass, ph_dir, additional


# ══════════════════════════════════════════════════════════════════════════════
# SHARED CONFUSION MATRIX PLOTTER
# ══════════════════════════════════════════════════════════════════════════════

def _plot_cm(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    subtitle: str,
    figsize: Tuple[float, float],
    annotate: bool = True,
    cell_fontsize: float = TICK_SIZE + 0.5,
    label_fontsize: float = TICK_SIZE,
    label_rotation: float = 35.0,
) -> plt.Figure:
    n = len(labels)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    im = ax.imshow(cm, vmin=0, vmax=1, cmap=CM_CMAP, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("Recall (row-normalised)", fontsize=FONT_SIZE, color=TICK_COLOR)
    cbar.ax.tick_params(labelsize=TICK_SIZE - 0.5, colors=TICK_COLOR)
    cbar.outline.set_edgecolor("#CCCCCC")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=label_rotation,
                       ha="right", fontsize=label_fontsize)
    ax.set_yticklabels(labels, fontsize=label_fontsize)
    ax.set_xlabel("Predicted", labelpad=6)
    ax.set_ylabel("True", labelpad=6)
    ax.grid(False)

    if annotate:
        for i in range(n):
            for j in range(n):
                v = cm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=cell_fontsize, fontweight="bold",
                        color="white" if v > 0.55 else TEXT_COLOR)

    _style_ax(ax)
    _title_subtitle(ax, title, subtitle)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 1 — CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def make_confusion(results: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _W = "confusion_matrix not found for {task}.\n  Re-run 05_random_forest.py."

    t1 = results.get("task1_o2_per_particle", {})

    for pk in PARTICLE_KEY_ORDER:
        d = t1.get(pk, {})
        if "confusion_matrix" not in d:
            logger.warning(_W.format(task=f"task1_o2/{pk}")); continue
        fig = _plot_cm(
            np.array(d["confusion_matrix"]),
            [O2_LABELS[o] for o in O2_ORDERED],
            f"O₂ Classification — {PARTICLE_LABELS[pk]}",
            (f"Balanced Accuracy = {d['bal_acc_mean']:.3f} "
             f"± {d['bal_acc_std']:.3f}   ·   "
             f"Macro-F1 = {d['f1_mean']:.3f}   ·   "
             f"Chance = {CHANCE_O2:.3f}"),
            SIZE_CM_7,
        )
        _savefig(fig, out_dir / f"task1_o2_{pk}_cm.{OUT_FORMAT}")

    d2 = results.get("task2_particle", {})
    if "confusion_matrix" in d2:
        fig2 = _plot_cm(
            np.array(d2["confusion_matrix"]),
            [PARTICLE_LABELS[pk] for pk in PARTICLE_KEY_ORDER],
            "Particle / LET Classification",
            (f"Balanced Accuracy = {d2['bal_acc_mean']:.3f} "
             f"± {d2['bal_acc_std']:.3f}   ·   "
             f"Macro-F1 = {d2['f1_mean']:.3f}   ·   "
             f"Chance = {CHANCE_PARTICLE:.3f}"),
            SIZE_CM_7,
        )
        _savefig(fig2, out_dir / f"task2_particle_cm.{OUT_FORMAT}")

    d3 = results.get("task3_joint_49class", {})
    if "confusion_matrix" in d3:
        cm49 = np.array(d3["confusion_matrix"])
        fig3 = _plot_cm(
            cm49,
            CONDITION_LABELS_SHORT,
            "Joint 49-Class Classification",
            (f"Balanced Accuracy = {d3['bal_acc_mean']:.3f} "
             f"± {d3['bal_acc_std']:.3f}   ·   "
             f"Macro-F1 = {d3['f1_mean']:.3f}   ·   "
             f"Chance = {CHANCE_JOINT:.4f}"),
            SIZE_CM_49,
            annotate=False,
            label_fontsize=5.5,
            label_rotation=55.0,
        )
        ax49 = fig3.axes[0]
        for pos in range(7, 49, 7):
            ax49.axhline(pos - 0.5, color="white", lw=1.4, zorder=5)
            ax49.axvline(pos - 0.5, color="white", lw=1.4, zorder=5)
        _savefig(fig3, out_dir / f"task3_joint_cm.{OUT_FORMAT}")

    t4 = results.get("task4_sobp_position", {})
    for sp in SOBP_SPECIES:
        d4 = t4.get(sp, {})
        if "confusion_matrix" not in d4:
            logger.warning(_W.format(task=f"task4_sobp/{sp}")); continue
        fig4 = _plot_cm(
            np.array(d4["confusion_matrix"]),
            ["pSOBP", "dSOBP"],
            f"SOBP Position — {sp.capitalize()}",
            (f"Balanced Accuracy = {d4['bal_acc_mean']:.3f} "
             f"± {d4['bal_acc_std']:.3f}   ·   "
             f"Macro-F1 = {d4['f1_mean']:.3f}   ·   "
             f"Chance = {CHANCE_SOBP:.2f}"),
            SIZE_CM_2,
        )
        _savefig(fig4, out_dir / f"task4_sobp_{sp}_cm.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 2 — PERMUTATION IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def _plot_importance(
    importances: np.ndarray,
    feat_cols: List[str],
    title: str,
    subtitle: str,
    top_n: int,
) -> plt.Figure:
    idx    = np.argsort(importances)[::-1][:top_n]
    vals   = importances[idx]
    names  = [feat_cols[i].replace("_", " ") for i in idx]
    colors = [_feat_color(feat_cols[i]) for i in idx]

    fig, ax = plt.subplots(figsize=(SIZE_IMP_W, max(5.5, top_n * 0.31)))
    fig.patch.set_facecolor("white")
    y = np.arange(top_n)
    ax.barh(y[::-1], vals, color=colors,
            edgecolor="white", linewidth=0.25, height=0.74)
    ax.set_yticks(y[::-1])
    ax.set_yticklabels(names, fontsize=TICK_SIZE)
    ax.set_xlabel("Mean permutation importance (Δ balanced accuracy)", labelpad=6)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    handles = [Patch(facecolor=MODALITY_COLORS[m], label=m, edgecolor="none")
               for m in MODALITY_ORDER]
    leg = ax.legend(handles=handles, title="Modality",
                    fontsize=LEGEND_SIZE, title_fontsize=LEGEND_SIZE + 0.5,
                    loc="lower right", framealpha=0.95, edgecolor="#CCCCCC")
    leg.get_title().set_fontweight("bold")
    _style_ax(ax)
    _title_subtitle(ax, title, subtitle)
    fig.tight_layout()
    return fig


def make_importance(
    results: Dict,
    feat_cols: List[str],
    out_dir: Path,
    top_n: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = (f"Top {top_n} features  ·  mean permutation importance over 50 folds")
    t1 = results.get("task1_o2_per_particle", {})
    t4 = results.get("task4_sobp_position", {})

    tasks = []
    for pk in PARTICLE_KEY_ORDER:
        tasks.append((f"task1_o2_{pk}", t1.get(pk, {}),
                      f"Feature Importance — O₂ / {PARTICLE_LABELS[pk]}"))
    tasks.append(("task2_particle", results.get("task2_particle", {}),
                  "Feature Importance — Particle / LET Classification"))
    tasks.append(("task3_joint", results.get("task3_joint_49class", {}),
                  "Feature Importance — Joint 49-Class Classification"))
    for sp in SOBP_SPECIES:
        tasks.append((f"task4_sobp_{sp}", t4.get(sp, {}),
                      f"Feature Importance — SOBP Position / {sp.capitalize()}"))

    for key, d, title in tasks:
        if "feature_importances" not in d:
            continue
        imp = np.array(d["feature_importances"])
        if len(imp) != len(feat_cols):
            logger.warning(f"  Length mismatch for {key} — skipping."); continue
        fig = _plot_importance(imp, feat_cols, title, sub, top_n)
        _savefig(fig, out_dir / f"{key}_importance.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 3 — MODALITY ABLATION
# ══════════════════════════════════════════════════════════════════════════════

def make_ablation(ablation: Dict, out_dir: Path) -> None:
    if not ablation:
        logger.warning("No ablation data found — skipping."); return
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_keys   = [f"t1_{pk}" for pk in PARTICLE_KEY_ORDER] + ["t2_particle"]
    panel_titles = {f"t1_{pk}": f"O₂ / {PARTICLE_SHORT[pk]}"
                    for pk in PARTICLE_KEY_ORDER}
    panel_titles["t2_particle"] = "Particle / LET"

    available = [k for k in panel_keys if k in ablation]
    if not available:
        logger.warning("No ablation task keys recognised — skipping."); return

    n_cols = 4
    n_rows = (len(available) + n_cols - 1) // n_cols
    pw, ph = SIZE_ABL_PANEL
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(pw * n_cols, ph * n_rows),
                             sharey=True)
    if n_rows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]
    fig.patch.set_facecolor("white")

    for idx, key in enumerate(available):
        ax    = axes[idx]
        drops = [ablation[key].get(m, 0.0) for m in MODALITY_ORDER]
        colors = [MODALITY_COLORS[m] for m in MODALITY_ORDER]
        y = np.arange(len(MODALITY_ORDER))
        ax.barh(y, drops, color=colors, edgecolor="white",
                linewidth=0.25, height=0.72)
        ax.axvline(0, color="#888888", lw=0.8, ls="--", zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(MODALITY_ORDER, fontsize=TICK_SIZE - 0.5)
        ax.set_xlabel("Δ balanced accuracy", labelpad=4, fontsize=LABEL_SIZE - 0.5)
        for yi, val in zip(y, drops):
            ax.text(val + (0.0005 if val >= 0 else -0.0005), yi,
                    f"{val:+.3f}", va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=TICK_SIZE - 1.5, color=TEXT_COLOR)
        _style_ax(ax)
        ax.set_title(panel_titles.get(key, key),
                     fontsize=TITLE_SIZE - 1, fontweight="bold",
                     color=TEXT_COLOR, loc="left")

    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Modality Ablation", fontsize=TITLE_SIZE + 1, fontweight="bold",
                 color=TEXT_COLOR, x=0.01, ha="left", y=1.01)
    fig.text(0.01, 0.985,
             "Drop = baseline balanced accuracy − ablated balanced accuracy  "
             "·  positive = modality contributes",
             fontsize=SUBTITLE_SIZE, color=SUB_COLOR, style="italic", va="top")
    fig.tight_layout()
    _savefig(fig, out_dir / f"modality_ablation.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 4 — O2 SUMMARY (Task 1)
# ══════════════════════════════════════════════════════════════════════════════

def make_o2_summary(results: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t1 = results.get("task1_o2_per_particle", {})
    cv = results.get("cv_config", {})

    particles = [pk for pk in PARTICLE_KEY_ORDER if pk in t1]
    if not particles:
        logger.warning("No Task 1 results — skipping O2 summary."); return

    means = [t1[pk]["bal_acc_mean"] for pk in particles]
    stds  = [t1[pk]["bal_acc_std"]  for pk in particles]

    fig, ax = plt.subplots(figsize=SIZE_SUMMARY)
    fig.patch.set_facecolor("white")
    x = np.arange(len(particles))
    ax.bar(x, means, yerr=stds,
           color=[PARTICLE_COLORS[pk] for pk in particles],
           capsize=5, edgecolor="white", linewidth=0.3, width=0.60,
           error_kw={"ecolor": "#555555", "lw": 1.1, "capsize": 5, "capthick": 1.1},
           zorder=3)
    ax.axhline(CHANCE_O2, color=COLOR_CHANCE, ls="--", lw=1.2, zorder=4,
               label=f"Chance ({CHANCE_O2:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([PARTICLE_SHORT[pk] for pk in particles],
                       fontsize=TICK_SIZE + 0.5)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Balanced Accuracy", labelpad=6)
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + s + 0.025, f"{m:.3f}", ha="center", va="bottom",
                fontsize=TICK_SIZE, fontweight="bold", color=TEXT_COLOR)
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.95, edgecolor="#CCCCCC")
    _style_ax(ax)
    n_folds = cv.get("n_folds", "?")
    _title_subtitle(ax, "O₂ Level Classification — Balanced Accuracy",
                    (f"7-class  ·  {cv.get('n_splits', '?')}-fold × "
                     f"{cv.get('n_repeats', '?')} repeats ({n_folds} folds)  ·  mean ± 1 SD"))
    fig.tight_layout()
    _savefig(fig, out_dir / f"task1_o2_balanced_accuracy.{OUT_FORMAT}")

    # Per-class recall faceted
    n_p = len(particles)
    n_c = 4
    n_r = (n_p + n_c - 1) // n_c
    fig2, axes2 = plt.subplots(n_r, n_c,
                               figsize=(SIZE_RECALL_W * n_c, SIZE_RECALL_H * n_r),
                               sharey=True)
    axes2_flat = list(axes2) if n_r == 1 else [ax for row in axes2 for ax in row]
    fig2.patch.set_facecolor("white")

    for panel_idx, pk in enumerate(particles):
        ax = axes2_flat[panel_idx]
        recalls = [t1[pk]["per_class_recall_o2"].get(o, 0.0) for o in O2_ORDERED]
        x2 = np.arange(len(O2_ORDERED))
        bars = ax.bar(x2, recalls, color=O2_COLORS, edgecolor="white",
                      linewidth=0.3, width=0.60, zorder=3)
        ax.axhline(CHANCE_O2, color=COLOR_CHANCE, ls="--", lw=1.0, zorder=4)
        ax.axhline(1.0, color="#CCCCCC", ls=":", lw=0.7, zorder=2)
        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.022, f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=TICK_SIZE - 1, fontweight="bold", color=TEXT_COLOR)
        ax.set_xticks(x2)
        ax.set_xticklabels([O2_SHORT[o] for o in O2_ORDERED],
                           fontsize=TICK_SIZE - 1, rotation=30, ha="right")
        ax.set_ylim(0, 1.32)
        if panel_idx % n_c == 0:
            ax.set_ylabel("Per-class Recall", fontsize=LABEL_SIZE)
        _style_ax(ax)
        _strip_header(ax, PARTICLE_LABELS[pk])

    for idx in range(len(particles), len(axes2_flat)):
        axes2_flat[idx].set_visible(False)

    fig2.text(0.01, 0.975, "O₂ Level Classification — Per-class Recall",
              fontsize=TITLE_SIZE + 1, fontweight="bold", color=TEXT_COLOR, va="top")
    fig2.text(0.01, 0.930,
              (f"7-class  ·  {cv.get('n_splits', '?')}-fold × "
               f"{cv.get('n_repeats', '?')} repeats  ·  "
               f"dashed = chance ({CHANCE_O2:.3f})"),
              fontsize=SUBTITLE_SIZE, color=SUB_COLOR, style="italic", va="top")
    handles_o2 = [Patch(facecolor=O2_COLORS[i],
                        label=f"{O2_LABELS[o].split(' ')[0]}",
                        edgecolor="none")
                  for i, o in enumerate(O2_ORDERED)]
    handles_o2.append(plt.Line2D([0], [0], color=COLOR_CHANCE, ls="--", lw=1.0,
                                 label=f"Chance ({CHANCE_O2:.3f})"))
    fig2.legend(handles=handles_o2, title="O₂ level",
                title_fontsize=LEGEND_SIZE + 0.5, fontsize=LEGEND_SIZE,
                loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.03),
                framealpha=0.95, edgecolor="#CCCCCC")
    fig2.tight_layout(rect=[0, 0.06, 1, 0.91])
    _savefig(fig2, out_dir / f"task1_o2_per_class_recall.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 5 — SOBP SUMMARY (Task 4)
# ══════════════════════════════════════════════════════════════════════════════

def make_sobp_summary(results: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t4 = results.get("task4_sobp_position", {})
    cv = results.get("cv_config", {})

    species = [sp for sp in SOBP_SPECIES if sp in t4]
    if not species:
        logger.warning("No Task 4 results — skipping SOBP summary."); return

    fig, ax = plt.subplots(figsize=SIZE_SOBP_SUM)
    fig.patch.set_facecolor("white")
    x     = np.arange(len(species))
    means = [t4[sp]["bal_acc_mean"] for sp in species]
    stds  = [t4[sp]["bal_acc_std"]  for sp in species]
    cols  = [SPECIES_COLORS[sp] for sp in species]

    ax.bar(x, means, yerr=stds, color=cols, capsize=5,
           edgecolor="white", linewidth=0.3, width=0.52,
           error_kw={"ecolor": "#555555", "lw": 1.1, "capsize": 5, "capthick": 1.1},
           zorder=3)
    ax.axhline(CHANCE_SOBP, color=COLOR_CHANCE, ls="--", lw=1.2, zorder=4,
               label=f"Chance ({CHANCE_SOBP:.2f})")
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + s + 0.02, f"{m:.3f}", ha="center", va="bottom",
                fontsize=TICK_SIZE, fontweight="bold", color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([sp.capitalize() for sp in species], fontsize=TICK_SIZE + 0.5)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Balanced Accuracy", labelpad=6)
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.95, edgecolor="#CCCCCC")
    _style_ax(ax)
    _title_subtitle(ax, "SOBP Position Classification (pSOBP vs. dSOBP)",
                    (f"2-class  ·  {cv.get('n_splits', '?')}-fold × "
                     f"{cv.get('n_repeats', '?')} repeats  ·  pooled over 7 O₂ levels  ·  mean ± 1 SD"))
    fig.tight_layout()
    _savefig(fig, out_dir / f"task4_sobp_summary.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 6 — UMAP (optional)
# ══════════════════════════════════════════════════════════════════════════════

def make_umap(wass: Dict[int, np.ndarray], fm: pd.DataFrame, out_dir: Path) -> None:
    if not HAS_UMAP:
        logger.warning("umap-learn not installed — skipping UMAP.\n  pip install umap-learn")
        return
    if not wass:
        logger.warning("No Wasserstein matrices — skipping UMAP.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_df = (fm[["particle_key", "o2", "run_id"]]
               .copy()
               .sort_values(["particle_key", "o2", "run_id"], key=_run_sort_key)
               .reset_index(drop=True))

    for deg, D in wass.items():
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)
        logger.info(f"  Fitting UMAP H{deg}  ({D.shape[0]} nuclei)…")
        emb = umap_module.UMAP(
            n_components=2, metric="precomputed",
            n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
            random_state=UMAP_RANDOM_STATE, n_jobs=1,
        ).fit_transform(D)

        fig, (ax_pk, ax_o2) = plt.subplots(1, 2, figsize=SIZE_UMAP)
        fig.patch.set_facecolor("white")

        for pk in PARTICLE_KEY_ORDER:
            mask = runs_df["particle_key"].values == pk
            ax_pk.scatter(emb[mask, 0], emb[mask, 1],
                          c=PARTICLE_COLORS[pk], label=PARTICLE_SHORT[pk],
                          s=16, alpha=0.82, linewidths=0, zorder=3)
        ax_pk.set_xlabel("UMAP 1"); ax_pk.set_ylabel("UMAP 2")
        leg = ax_pk.legend(title="Particle config", fontsize=LEGEND_SIZE,
                           title_fontsize=LEGEND_SIZE + 0.5,
                           framealpha=0.95, edgecolor="#CCCCCC", ncol=2)
        leg.get_title().set_fontweight("bold")
        _style_ax(ax_pk)
        _title_subtitle(ax_pk, f"H{deg} Diagram Space — by Particle Config",
                        f"Wasserstein-2  ·  n = {len(runs_df)} nuclei")

        for i, o2 in enumerate(O2_ORDERED):
            mask = runs_df["o2"].values == o2
            ax_o2.scatter(emb[mask, 0], emb[mask, 1],
                          c=O2_COLORS[i], label=f"{o2}%",
                          s=16, alpha=0.82, linewidths=0, zorder=3)
        ax_o2.set_xlabel("UMAP 1"); ax_o2.set_ylabel("UMAP 2")
        leg = ax_o2.legend(title="O₂ level", fontsize=LEGEND_SIZE,
                           title_fontsize=LEGEND_SIZE + 0.5,
                           framealpha=0.95, edgecolor="#CCCCCC", ncol=2)
        leg.get_title().set_fontweight("bold")
        _style_ax(ax_o2)
        _title_subtitle(ax_o2, f"H{deg} Diagram Space — by O₂ Level",
                        f"Wasserstein-2  ·  n = {len(runs_df)} nuclei")

        fig.tight_layout()
        _savefig(fig, out_dir / f"umap_H{deg}.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 7 — WITHIN / BETWEEN VIOLIN
# ══════════════════════════════════════════════════════════════════════════════

def make_violin(
    wass: Dict[int, np.ndarray],
    fm: pd.DataFrame,
    ph_summary: Dict,
    out_dir: Path,
) -> None:
    if not wass:
        logger.warning("No Wasserstein matrices — skipping violins."); return
    out_dir.mkdir(parents=True, exist_ok=True)

    all_conds = [f"{pk}_{o2}" for pk, o2 in CONDITIONS]
    runs_df = (fm[["particle_key", "o2", "run_id"]]
               .sort_values(["particle_key", "o2", "run_id"], key=_run_sort_key)
               .reset_index(drop=True))
    cond_idx = np.array([
        all_conds.index(f"{r.particle_key}_{r.o2}")
        for _, r in runs_df.iterrows()
    ])

    for deg, D in wass.items():
        n = len(D)
        within, between = [], []
        for i in range(n):
            for j in range(i + 1, n):
                (within if cond_idx[i] == cond_idx[j]
                 else between).append(float(D[i, j]))

        s = ph_summary.get(f"h{deg}", {})
        fig, ax = plt.subplots(figsize=SIZE_VIOLIN)
        fig.patch.set_facecolor("white")
        vp = ax.violinplot([np.array(within), np.array(between)],
                           positions=[0, 1], showmedians=True, showextrema=True)
        for pc, c in zip(vp["bodies"], [COLOR_WITHIN, COLOR_BETWEEN]):
            pc.set_facecolor(c); pc.set_alpha(0.65); pc.set_edgecolor("white")
        for k in ["cmedians", "cbars", "cmins", "cmaxes"]:
            vp[k].set_color("#444444"); vp[k].set_linewidth(1.2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Within condition", "Between conditions"],
                           fontsize=TICK_SIZE + 0.5)
        ax.set_ylabel(f"Wasserstein-2 distance (H{deg})")
        _style_ax(ax)
        _title_subtitle(ax, f"H{deg} Diagram Separability",
                        (f"Within median = {s.get('within_median', float('nan')):.3f}   ·   "
                         f"Between median = {s.get('between_median', float('nan')):.3f}   ·   "
                         f"Ratio = {s.get('separation_ratio', float('nan')):.2f}"))
        fig.tight_layout()
        _savefig(fig, out_dir / f"within_between_H{deg}.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 8 — CONDITION HEATMAPS (49×49 Wasserstein)
# ══════════════════════════════════════════════════════════════════════════════

def make_heatmaps(ph_summary: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for deg in [0, 1]:
        s = ph_summary.get(f"h{deg}", {})
        if "condition_mean_matrix" not in s:
            logger.warning(f"  condition_mean_matrix missing for H{deg} — skipping.")
            continue
        cmat = np.array(s["condition_mean_matrix"])
        n    = cmat.shape[0]

        fig, ax = plt.subplots(figsize=SIZE_HEATMAP)
        fig.patch.set_facecolor("white")
        im   = ax.imshow(cmat, cmap=CM_CMAP, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
        cbar.set_label("Mean Wasserstein-2 distance",
                       fontsize=FONT_SIZE, color=TICK_COLOR)
        cbar.ax.tick_params(labelsize=TICK_SIZE - 0.5, colors=TICK_COLOR)
        cbar.outline.set_edgecolor("#CCCCCC")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(CONDITION_LABELS_SHORT[:n],
                           rotation=55, ha="right", fontsize=5.5)
        ax.set_yticklabels(CONDITION_LABELS_SHORT[:n], fontsize=5.5)
        ax.grid(False)
        for pos in range(7, n, 7):
            ax.axhline(pos - 0.5, color="white", lw=1.5, zorder=5)
            ax.axvline(pos - 0.5, color="white", lw=1.5, zorder=5)
        _style_ax(ax)
        _title_subtitle(ax, f"H{deg} Pairwise Wasserstein Distances",
                        f"{n} conditions × 50 runs  ·  mean over inter-condition pairs  ·  "
                        "white lines: particle-config boundaries")
        fig.tight_layout()
        _savefig(fig, out_dir / f"condition_heatmap_H{deg}.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 9 — PERSISTENCE LANDSCAPES (optional)
# ══════════════════════════════════════════════════════════════════════════════

def _dgm_to_landscape(dgm: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    k_max = LANDSCAPE_K_MAX
    n_t   = len(t_vals)
    if len(dgm) == 0:
        return np.zeros((k_max, n_t))
    tents = np.maximum(0.0, np.minimum(
        t_vals[None, :] - dgm[:, 0, None],
        dgm[:, 1, None] - t_vals[None, :],
    ))
    st = np.sort(tents, axis=0)[::-1]
    if st.shape[0] < k_max:
        st = np.vstack([st, np.zeros((k_max - st.shape[0], n_t))])
    return st[:k_max]


def make_landscapes(ph_dir: Path, out_dir: Path) -> None:
    diag_dir = ph_dir / "diagrams"
    if not diag_dir.exists():
        logger.warning("Diagrams directory not found — skipping landscapes."); return
    all_npz = sorted(diag_dir.glob("*_diagram.npz"))
    if not all_npz:
        logger.warning("No .npz files in diagrams/ — skipping landscapes."); return
    out_dir.mkdir(parents=True, exist_ok=True)

    t_vals    = np.linspace(0, PH_MAX_DIST / 2, LANDSCAPE_N_T)
    all_conds = {(pk, o2): [] for pk in PARTICLE_KEY_ORDER for o2 in O2_ORDERED}

    for npz_path in all_npz:
        stem       = npz_path.stem.replace("_diagram", "")
        matched_pk = None; matched_o2 = None
        for dir_nm, pk in DIR_TO_PARTICLE_KEY.items():
            if stem.startswith(dir_nm + "_"):
                remainder = stem[len(dir_nm) + 1:]
                parts     = remainder.rsplit("_", 1)
                if len(parts) == 2:
                    o2_str = _norm_o2(parts[0])
                    if o2_str in O2_ORDERED:
                        matched_pk = pk; matched_o2 = o2_str
                break
        if matched_pk is None:
            continue
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                h0 = np.array(data.get("h0", np.empty((0, 2))))
                h1 = np.array(data.get("h1", np.empty((0, 2))))
        except Exception:
            continue
        all_conds[(matched_pk, matched_o2)].append({"h0": h0, "h1": h1})

    for deg in [0, 1]:
        deg_key = f"h{deg}"
        cond_means = {}
        for (pk, o2), entries in all_conds.items():
            landscapes = []
            for arrays in entries:
                dgm = arrays.get(deg_key, np.empty((0, 2)))
                if dgm is None or not isinstance(dgm, np.ndarray):
                    dgm = np.empty((0, 2))
                if len(dgm) > 0:
                    dgm = dgm[np.isfinite(dgm[:, 1])]
                landscapes.append(_dgm_to_landscape(dgm, t_vals))
            if landscapes:
                cond_means[(pk, o2)] = np.mean(landscapes, axis=0)
        if not cond_means:
            logger.warning(f"  No landscape data for H{deg} — skipping."); continue

        n_cols = 4; n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(SIZE_LAND_W * n_cols, SIZE_LAND_H * n_rows),
                                 sharey=True)
        axes_flat = [ax for row in axes for ax in row]
        fig.patch.set_facecolor("white")

        for panel_idx, pk in enumerate(PARTICLE_KEY_ORDER):
            ax = axes_flat[panel_idx]
            for i, o2 in enumerate(O2_ORDERED):
                if (pk, o2) not in cond_means:
                    continue
                ax.plot(t_vals, cond_means[(pk, o2)][0],
                        color=O2_COLORS[i], label=f"{O2_SHORT[o2]}%",
                        linewidth=1.8, alpha=0.92, zorder=3)
            ax.set_xlabel("Filtration value (µm)", labelpad=5)
            if panel_idx % n_cols == 0:
                ax.set_ylabel(f"λ₁(t)  [H{deg}]", labelpad=5)
            leg = ax.legend(title="O₂", fontsize=LEGEND_SIZE - 0.5,
                            title_fontsize=LEGEND_SIZE, framealpha=0.95,
                            edgecolor="#CCCCCC", ncol=2)
            leg.get_title().set_fontweight("bold")
            _style_ax(ax)
            _strip_header(ax, PARTICLE_LABELS[pk])

        axes_flat[7].set_visible(False)
        fig.text(0.01, 0.975, f"Mean Persistence Landscape λ₁ — H{deg}",
                 fontsize=TITLE_SIZE + 1, fontweight="bold", color=TEXT_COLOR, va="top")
        fig.text(0.01, 0.930,
                 "First landscape function  ·  mean over 50 runs per condition  ·  coloured by O₂ level",
                 fontsize=SUBTITLE_SIZE, color=SUB_COLOR, style="italic", va="top")
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        _savefig(fig, out_dir / f"landscape_H{deg}.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 10 — CROSS-MODALITY CORRELATION (Stage 2)
# ══════════════════════════════════════════════════════════════════════════════

def make_cross_modality_correlation(data: Optional[Dict], out_dir: Path) -> None:
    if data is None:
        logger.warning("cross_modality_correlation.json not found — skipping."); return
    out_dir.mkdir(parents=True, exist_ok=True)

    corr = np.array(data["correlation_matrix"])
    names = data["modality_names"]
    n_conds = data.get("n_conditions", 49)

    fig, ax = plt.subplots(figsize=SIZE_CORR_HEATMAP)
    fig.patch.set_facecolor("white")

    im = ax.imshow(corr, vmin=-1, vmax=1, cmap=CORR_CMAP, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Pearson r (condition-mean vectors)", fontsize=FONT_SIZE, color=TICK_COLOR)
    cbar.ax.tick_params(labelsize=TICK_SIZE - 0.5, colors=TICK_COLOR)
    cbar.outline.set_edgecolor("#CCCCCC")

    n = len(names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [f"m{i+1}" for i in range(n)]
    ax.set_xticklabels(short, fontsize=TICK_SIZE + 1)
    ax.set_yticklabels(short, fontsize=TICK_SIZE + 1)
    ax.grid(False)

    # Annotate each cell
    for i in range(n):
        for j in range(n):
            v = corr[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=TICK_SIZE, fontweight="bold",
                    color="white" if abs(v) > 0.65 else TEXT_COLOR)

    # Modality name legend below
    legend_text = "  ".join([f"m{i+1} = {nm}" for i, nm in enumerate(names)])
    fig.text(0.01, -0.04, legend_text, fontsize=FONT_SIZE - 1.5,
             color=SUB_COLOR, style="italic", va="top")

    # Highlight {m1, m5, m7} oxygen-sensitive cluster with a bracket border
    cluster_idx = [0, 4, 6]  # m1, m5, m7 (0-indexed)
    for ci in cluster_idx:
        for cj in cluster_idx:
            ax.add_patch(plt.Rectangle(
                (cj - 0.5, ci - 0.5), 1, 1,
                fill=False, edgecolor="#CD5F00", linewidth=1.6, zorder=6,
            ))

    _style_ax(ax)
    _title_subtitle(
        ax,
        "Cross-Modality Correlation",
        (f"Pearson r of condition-mean modality vectors  ·  {n_conds} conditions  ·  "
         "orange borders: {m1, m5, m7} oxygen-sensitive cluster"),
    )
    fig.tight_layout()
    _savefig(fig, out_dir / f"cross_modality_correlation.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 11 — SINGLE-MODALITY O₂ ACCURACY MATRIX (Stage 3)
# ══════════════════════════════════════════════════════════════════════════════

def make_single_modality_o2_accuracy(data: Optional[Dict], out_dir: Path) -> None:
    if data is None:
        logger.warning("single_modality_o2_accuracy.json not found — skipping."); return
    out_dir.mkdir(parents=True, exist_ok=True)

    acc   = np.array(data["accuracy_matrix"])   # shape (7 modalities, 7 particles)
    mod_names = data["modality_names"]
    pk_keys   = data["particle_keys"]
    pk_lets   = data["particle_lets"]
    chance    = data.get("chance_level", 1.0 / 7)

    # acc[mod_idx, pk_idx]
    fig, ax = plt.subplots(figsize=SIZE_ACC_MATRIX)
    fig.patch.set_facecolor("white")

    im = ax.imshow(acc, vmin=0, vmax=max(acc.max(), 0.7),
                   cmap=CM_CMAP, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("7-class O₂ balanced accuracy", fontsize=FONT_SIZE, color=TICK_COLOR)
    cbar.ax.tick_params(labelsize=TICK_SIZE - 0.5, colors=TICK_COLOR)
    cbar.outline.set_edgecolor("#CCCCCC")

    n_mod = len(mod_names)
    n_pk  = len(pk_keys)
    ax.set_xticks(range(n_pk))
    ax.set_yticks(range(n_mod))
    pk_xlabels = [f"{PARTICLE_SHORT.get(pk, pk)}\n{let:.1f}" for pk, let in zip(pk_keys, pk_lets)]
    ax.set_xticklabels(pk_xlabels, fontsize=TICK_SIZE)
    ax.set_yticklabels([f"m{i+1}  {nm}" for i, nm in enumerate(mod_names)],
                       fontsize=TICK_SIZE)
    ax.set_xlabel("Particle config  (LET, keV/µm)", labelpad=6)
    ax.grid(False)

    for i in range(n_mod):
        for j in range(n_pk):
            v = acc[i, j]
            marker = "*" if v == acc[:, j].max() else ""
            ax.text(j, i, f"{v:.2f}{marker}", ha="center", va="center",
                    fontsize=TICK_SIZE - 0.5, fontweight="bold",
                    color="white" if v > 0.45 else TEXT_COLOR)

    # Draw chance level contour line
    ax.axhline(-0.5, color="none")   # dummy to fix extent
    ax.contour(acc, levels=[chance], colors=[COLOR_CHANCE],
               linewidths=1.2, linestyles="--", zorder=6)

    # Add modality colour bands on y-axis
    for i, mod in enumerate(MODALITY_ORDER):
        ax.add_patch(plt.Rectangle(
            (-0.52, i - 0.5), 0.06, 1.0,
            transform=ax.transData, clip_on=False,
            facecolor=MODALITY_COLORS[mod], zorder=7,
        ))

    _style_ax(ax)
    _title_subtitle(
        ax,
        "Single-Modality O₂ Classification Accuracy",
        (f"7-class balanced accuracy  ·  one RF per modality per particle  ·  "
         f"chance = {chance:.3f}  ·  * = best modality per particle  ·  "
         "dashed contour = chance level"),
    )
    fig.tight_layout()
    _savefig(fig, out_dir / f"single_modality_o2_accuracy.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 12 — LET–O₂ MODALITY ENCODING PROFILES (Stage 4)
# ══════════════════════════════════════════════════════════════════════════════

def make_let_o2_profiles(data: Optional[Dict], out_dir: Path) -> None:
    """One line per modality showing 7-class O₂ accuracy vs. LET."""
    if data is None:
        logger.warning("single_modality_o2_accuracy.json not found — skipping LET profiles."); return
    out_dir.mkdir(parents=True, exist_ok=True)

    acc     = np.array(data["accuracy_matrix"])   # (n_mod, n_pk)
    pk_keys = data["particle_keys"]
    pk_lets = data["particle_lets"]
    mod_names = data["modality_names"]
    chance  = data.get("chance_level", 1.0 / 7)

    lets = np.array(pk_lets)
    log_lets = np.log10(lets)

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    fig.patch.set_facecolor("white")

    for i, (mod_name, color) in enumerate(MODALITY_COLORS.items()):
        vals = acc[i, :]
        ax.plot(log_lets, vals, color=color, linewidth=2.0,
                marker="o", markersize=6, label=f"m{i+1} {mod_name}",
                alpha=0.9, zorder=3 + i)

    ax.axhline(chance, color=COLOR_CHANCE, ls="--", lw=1.2, zorder=2,
               label=f"Chance ({chance:.3f})")
    ax.set_xlabel("LET (keV/µm)  [log₁₀ scale]", labelpad=6)
    ax.set_ylabel("7-class O₂ balanced accuracy", labelpad=6)
    ax.set_xticks(log_lets)
    ax.set_xticklabels([f"{l:.1f}" for l in lets], fontsize=TICK_SIZE)
    ax.set_ylim(bottom=0)

    leg = ax.legend(title="Modality", fontsize=LEGEND_SIZE,
                    title_fontsize=LEGEND_SIZE + 0.5,
                    loc="upper right", framealpha=0.95, edgecolor="#CCCCCC",
                    ncol=1)
    leg.get_title().set_fontweight("bold")
    _style_ax(ax)
    _title_subtitle(
        ax,
        "LET-Dependent O₂ Encoding — Single-Modality Accuracy",
        "7-class O₂ balanced accuracy per modality across the LET axis  ·  pooled over 7 O₂ levels",
    )
    fig.tight_layout()
    _savefig(fig, out_dir / f"let_o2_modality_profiles.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 13 — EFFECT SIZES (η²) — O₂ and PARTICLE (Stage 5)
# ══════════════════════════════════════════════════════════════════════════════

def _effect_size_bar(
    features: List[str],
    eta2: List[float],
    title: str,
    subtitle: str,
    out_path: Path,
    top_n: int = 30,
) -> None:
    idx    = np.argsort(eta2)[::-1][:top_n]
    vals   = [eta2[i] for i in idx]
    names  = [features[i].replace("_", " ") for i in idx]
    colors = [_feat_color(features[i]) for i in idx]

    fig, ax = plt.subplots(figsize=(SIZE_ETA_BAR_W, max(5.5, top_n * 0.31)))
    fig.patch.set_facecolor("white")
    y = np.arange(top_n)
    ax.barh(y[::-1], vals, color=colors, edgecolor="white",
            linewidth=0.25, height=0.74)
    ax.set_yticks(y[::-1])
    ax.set_yticklabels(names, fontsize=TICK_SIZE)
    ax.set_xlabel("η² (one-way ANOVA effect size)", labelpad=6)
    ax.set_xlim(0, min(1.05, max(vals) * 1.12))

    handles = [Patch(facecolor=MODALITY_COLORS[m], label=m, edgecolor="none")
               for m in MODALITY_ORDER]
    leg = ax.legend(handles=handles, title="Modality",
                    fontsize=LEGEND_SIZE, title_fontsize=LEGEND_SIZE + 0.5,
                    loc="lower right", framealpha=0.95, edgecolor="#CCCCCC")
    leg.get_title().set_fontweight("bold")
    _style_ax(ax)
    _title_subtitle(ax, title, subtitle)
    fig.tight_layout()
    _savefig(fig, out_path)


def make_effect_sizes(data: Optional[Dict], out_dir: Path, top_n: int = 30) -> None:
    if data is None:
        logger.warning("effect_sizes.json not found — skipping."); return
    out_dir.mkdir(parents=True, exist_ok=True)

    features    = data["features"]
    eta2_o2     = data["eta2_o2"]
    eta2_part   = data["eta2_particle"]

    _effect_size_bar(
        features, eta2_o2,
        "Feature Effect Sizes — O₂ Axis",
        f"η² (one-way ANOVA, O₂ grouping)  ·  top {top_n} features  ·  coloured by modality",
        out_dir / f"feature_o2_effect_sizes.{OUT_FORMAT}",
        top_n=top_n,
    )

    _effect_size_bar(
        features, eta2_part,
        "Feature Effect Sizes — Particle / LET Axis",
        f"η² (one-way ANOVA, particle grouping)  ·  top {top_n} features  ·  coloured by modality",
        out_dir / f"feature_particle_effect_sizes.{OUT_FORMAT}",
        top_n=top_n,
    )

    # ── Dual-axis scatter: η²_O₂ vs η²_particle, coloured by modality ────────
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    fig.patch.set_facecolor("white")

    for feat, e_o2, e_pk in zip(features, eta2_o2, eta2_part):
        col = _feat_color(feat)
        ax.scatter(e_o2, e_pk, color=col, s=28, alpha=0.72,
                   linewidths=0, zorder=3)

    handles = [Patch(facecolor=MODALITY_COLORS[m], label=m, edgecolor="none")
               for m in MODALITY_ORDER]
    leg = ax.legend(handles=handles, title="Modality",
                    fontsize=LEGEND_SIZE, title_fontsize=LEGEND_SIZE + 0.5,
                    loc="upper right", framealpha=0.95, edgecolor="#CCCCCC")
    leg.get_title().set_fontweight("bold")

    # Annotate notable outliers
    notable = [
        ("m1_n_dsbs", "m1_n_dsbs"),
        ("m7_h0_persistent_entropy", "m7_h0_persistent_entropy"),
        ("m7_h1_persistent_entropy", "m7_h1_persistent_entropy"),
    ]
    for feat_key, label in notable:
        if feat_key in features:
            fi = features.index(feat_key)
            ax.annotate(
                label.replace("_", " "), (eta2_o2[fi], eta2_part[fi]),
                textcoords="offset points", xytext=(5, 4),
                fontsize=TICK_SIZE - 1.5, color=TEXT_COLOR,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=SPINE_COLOR),
            )

    ax.set_xlabel("η²  (O₂ axis)", labelpad=6)
    ax.set_ylabel("η²  (Particle axis)", labelpad=6)
    ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.02, 1.05)
    ax.axhline(0, color=SPINE_COLOR, lw=0.5, zorder=1)
    ax.axvline(0, color=SPINE_COLOR, lw=0.5, zorder=1)
    _style_ax(ax)
    _title_subtitle(
        ax,
        "Dual-Axis Effect Size — O₂ vs. Particle",
        "Each point = one feature  ·  top-right = dual-signal; "
        "right edge = particle-only; top edge = O₂-only",
    )
    fig.tight_layout()
    _savefig(fig, out_dir / f"effect_sizes_scatter.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 14 — CONDITION PCA (Stage 6)
# ══════════════════════════════════════════════════════════════════════════════

def make_condition_pca(pca_data: Optional[Dict], fm: pd.DataFrame,
                       feat_cols: List[str], out_dir: Path) -> None:
    if pca_data is None:
        logger.warning("condition_pca.json not found — skipping PCA."); return
    out_dir.mkdir(parents=True, exist_ok=True)

    var_exp   = pca_data["variance_explained"]
    cum_90_pcs = pca_data["cumulative_var_90_pcs"]
    pc12_var  = pca_data["pc1_pc2_variance"]

    # ── Compute PCA from feature matrix (condition means) ─────────────────────
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    fm2 = fm.copy()
    fm2["o2"] = fm2["o2"].apply(_norm_o2)
    cond_means = fm2.groupby(["particle_key", "o2"])[feat_cols].mean().reset_index()

    X = StandardScaler().fit_transform(cond_means[feat_cols].values)
    pca = PCA(n_components=min(len(X), len(feat_cols)))
    Xp  = pca.fit_transform(X)

    pk_col = cond_means["particle_key"].values
    o2_col = cond_means["o2"].values

    # ── Figure A: PC1 vs PC2 scatter ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=SIZE_PCA)
    fig.patch.set_facecolor("white")

    ax_pca = axes[0]
    for pk in PARTICLE_KEY_ORDER:
        mask = pk_col == pk
        ax_pca.scatter(Xp[mask, 0], Xp[mask, 1],
                       c=PARTICLE_COLORS.get(pk, "#AAAAAA"),
                       label=PARTICLE_SHORT[pk],
                       s=80, alpha=0.88, linewidths=0, zorder=3)
        # Label each point with its O₂ value
        for xi, o2v in zip(Xp[mask], o2_col[mask]):
            ax_pca.text(xi[0] + 0.05, xi[1] + 0.05, O2_SHORT.get(o2v, o2v),
                        fontsize=5.5, color=TEXT_COLOR, alpha=0.7)

    ax_pca.set_xlabel(f"PC1  ({var_exp[0]*100:.1f}% var.)", labelpad=6)
    ax_pca.set_ylabel(f"PC2  ({var_exp[1]*100:.1f}% var.)", labelpad=6)
    leg = ax_pca.legend(title="Particle", fontsize=LEGEND_SIZE,
                        title_fontsize=LEGEND_SIZE + 0.5,
                        framealpha=0.95, edgecolor="#CCCCCC", ncol=1)
    leg.get_title().set_fontweight("bold")
    _style_ax(ax_pca)
    _title_subtitle(ax_pca, "Condition PCA — PC1 vs PC2",
                    f"49 conditions  ·  107 features  ·  coloured by particle  ·  labelled by O₂")

    # ── Figure B: scree plot ──────────────────────────────────────────────────
    ax_scr = axes[1]
    n_show = min(10, len(var_exp))
    xs     = np.arange(1, n_show + 1)
    ax_scr.bar(xs, [v * 100 for v in var_exp[:n_show]],
               color=PARTICLE_COLORS["proton_psobp"],
               edgecolor="white", linewidth=0.3, width=0.6, zorder=3)
    cum = np.cumsum(var_exp[:n_show]) * 100
    ax_scr2 = ax_scr.twinx()
    ax_scr2.plot(xs, cum, color=PARTICLE_COLORS["carbon_dsobp"],
                 marker="o", markersize=5, linewidth=1.6, zorder=4,
                 label="Cumulative")
    ax_scr2.axhline(90, color=COLOR_CHANCE, ls="--", lw=1.0, zorder=5,
                    label="90% threshold")
    ax_scr2.set_ylabel("Cumulative variance (%)", labelpad=6,
                       color=PARTICLE_COLORS["carbon_dsobp"])
    ax_scr2.tick_params(colors=TICK_COLOR)
    ax_scr2.set_ylim(0, 105)
    # Annotate the 90% PC
    ax_scr.axvline(cum_90_pcs + 0.5, color=COLOR_CHANCE, ls="--", lw=1.0, zorder=5)
    ax_scr.text(cum_90_pcs + 0.55, 0.5,
                f"{cum_90_pcs} PCs\n→ 90% var.",
                fontsize=TICK_SIZE - 1, color=COLOR_CHANCE, va="bottom")
    ax_scr.set_xlabel("Principal component", labelpad=6)
    ax_scr.set_ylabel("Variance explained (%)", labelpad=6)
    ax_scr.set_xticks(xs)
    ax_scr.set_ylim(bottom=0)
    _style_ax(ax_scr)
    _title_subtitle(ax_scr, "Condition PCA — Scree Plot",
                    f"{cum_90_pcs} PCs explain 90% of condition variance  ·  "
                    f"PC1+PC2 = {pc12_var:.1f}%")

    fig.tight_layout()
    _savefig(fig, out_dir / f"condition_pca.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 15 — SOBP EFFECT SIZES (Stage 8)
# ══════════════════════════════════════════════════════════════════════════════

def make_sobp_effect_sizes(data: Optional[Dict], out_dir: Path,
                            top_n: int = 20) -> None:
    if data is None:
        logger.warning("sobp_effect_sizes.json not found — skipping."); return
    out_dir.mkdir(parents=True, exist_ok=True)

    species_list = [sp for sp in SOBP_SPECIES if sp in data]
    if not species_list:
        logger.warning("No SOBP species found in data — skipping."); return

    n_sp = len(species_list)
    panel_w = 4.5
    fig, axes = plt.subplots(1, n_sp, figsize=(panel_w * n_sp, 7.0), sharey=False)
    if n_sp == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, sp in zip(axes, species_list):
        sp_data   = data[sp]
        feats_raw = sp_data["top_features_by_r"]
        # Sort by abs(rank_biserial_r) descending
        feats_sorted = sorted(feats_raw, key=lambda x: abs(x["rank_biserial_r"]),
                              reverse=True)[:top_n]
        names  = [f["feature"].replace("_", " ") for f in feats_sorted]
        r_vals = [f["rank_biserial_r"] for f in feats_sorted]
        colors = [_feat_color(f["feature"]) for f in feats_sorted]

        y = np.arange(len(names))
        ax.barh(y[::-1], r_vals, color=colors,
                edgecolor="white", linewidth=0.25, height=0.72)
        ax.axvline(0, color="#888888", lw=0.8, ls="--", zorder=4)
        ax.set_yticks(y[::-1])
        ax.set_yticklabels(names, fontsize=TICK_SIZE - 1)
        ax.set_xlabel("Rank-biserial r\n(pSOBP vs. dSOBP)", labelpad=4,
                      fontsize=LABEL_SIZE - 0.5)
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(
            (f"{sp.capitalize()}  "
             f"({sp_data['let_proximal']:.1f} vs. {sp_data['let_distal']:.1f} keV/µm)"),
            fontsize=TITLE_SIZE - 1, fontweight="bold",
            color=TEXT_COLOR, loc="left",
        )
        _style_ax(ax)

    handles = [Patch(facecolor=MODALITY_COLORS[m], label=m, edgecolor="none")
               for m in MODALITY_ORDER]
    fig.legend(handles=handles, title="Modality",
               fontsize=LEGEND_SIZE, title_fontsize=LEGEND_SIZE + 0.5,
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06),
               framealpha=0.95, edgecolor="#CCCCCC")

    fig.suptitle("SOBP Position Effect Sizes (Rank-Biserial r)",
                 fontsize=TITLE_SIZE + 1, fontweight="bold",
                 color=TEXT_COLOR, x=0.01, ha="left")
    fig.text(0.01, 0.96,
             f"Top {top_n} features by |r|  ·  positive = higher in dSOBP  ·  Mann-Whitney U",
             fontsize=SUBTITLE_SIZE, color=SUB_COLOR, style="italic", va="top")
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    _savefig(fig, out_dir / f"sobp_effect_sizes.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 16 — O₂ GRADIENT PROFILES (Stage 7)
# ══════════════════════════════════════════════════════════════════════════════

def make_o2_gradient_profiles(fm: pd.DataFrame, feat_cols: List[str],
                               effect_data: Optional[Dict], out_dir: Path) -> None:
    """
    For the top-η²(O₂) feature in each modality, plot mean ± SD across
    the 7 O₂ levels for each of the 7 particle configurations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Identify top-η²(O₂) feature per modality
    if effect_data is not None:
        features  = effect_data["features"]
        eta2_o2   = effect_data["eta2_o2"]
        top_feats: Dict[str, str] = {}
        for mod_name, prefix in zip(MODALITY_ORDER, MODALITY_PREFIX):
            pfx = [k for k, v in MODALITY_PREFIX.items() if v == mod_name][0]
            cands = [(f, e) for f, e in zip(features, eta2_o2) if f.startswith(pfx)]
            if cands:
                top_feats[mod_name] = max(cands, key=lambda x: x[1])[0]
    else:
        # Fallback: hard-code known top features
        top_feats = {
            "Spatial Distribution":       "m1_n_dsbs",
            "Radial Track Structure":     "m2_frac_r_lt_0.5um",
            "Local Energy Heterogeneity": "m3_energy_cv",
            "Dose Distribution":          "m4_dose_cv",
            "Genomic Location":           "m5_dsbs_per_chrom_std",
            "Damage Complexity":          "m6_complexity_score_mean",
            "Topological Summaries":      "m7_h0_persistent_entropy",
        }

    fm2 = fm.copy()
    fm2["o2"] = fm2["o2"].apply(_norm_o2)

    n_mod = len(MODALITY_ORDER)
    n_col = 4
    n_row = (n_mod + n_col - 1) // n_col
    fig, axes = plt.subplots(n_row, n_col,
                             figsize=SIZE_O2_GRADIENT, sharey=False)
    axes_flat = [ax for row in axes for ax in row] if n_row > 1 else list(axes)
    fig.patch.set_facecolor("white")

    for panel_idx, mod_name in enumerate(MODALITY_ORDER):
        ax   = axes_flat[panel_idx]
        feat = top_feats.get(mod_name)
        if feat is None or feat not in fm2.columns:
            ax.set_visible(False); continue

        for pk in PARTICLE_KEY_ORDER:
            sub  = fm2[fm2["particle_key"] == pk]
            means = [sub[sub["o2"] == o][feat].mean() for o in O2_ORDERED]
            stds  = [sub[sub["o2"] == o][feat].std() for o in O2_ORDERED]
            xs    = np.arange(len(O2_ORDERED))
            ax.plot(xs, means, color=PARTICLE_COLORS[pk],
                    linewidth=1.6, marker="o", markersize=4,
                    label=PARTICLE_SHORT[pk], zorder=3)
            ax.fill_between(xs,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=PARTICLE_COLORS[pk], alpha=0.12, zorder=2)

        ax.set_xticks(np.arange(len(O2_ORDERED)))
        ax.set_xticklabels([O2_SHORT[o] for o in O2_ORDERED],
                           fontsize=TICK_SIZE - 1.5, rotation=30, ha="right")
        ax.set_ylabel(feat.replace("_", " "), fontsize=TICK_SIZE - 1, labelpad=4)
        _style_ax(ax)
        _strip_header(ax, mod_name)

    # Hide unused panels
    for idx in range(n_mod, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared legend
    handles_pk = [Patch(facecolor=PARTICLE_COLORS[pk],
                        label=PARTICLE_SHORT[pk], edgecolor="none")
                  for pk in PARTICLE_KEY_ORDER]
    fig.legend(handles=handles_pk, title="Particle config",
               title_fontsize=LEGEND_SIZE + 0.5, fontsize=LEGEND_SIZE,
               loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.04),
               framealpha=0.95, edgecolor="#CCCCCC")

    fig.suptitle("O₂ Gradient Profiles — Top Feature per Modality",
                 fontsize=TITLE_SIZE + 1, fontweight="bold",
                 color=TEXT_COLOR, x=0.01, ha="left")
    fig.text(0.01, 0.96,
             "Mean ± 1 SD across 50 nuclei  ·  highest-η²(O₂) feature shown per modality  ·  coloured by particle",
             fontsize=SUBTITLE_SIZE, color=SUB_COLOR, style="italic", va="top")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    _savefig(fig, out_dir / f"o2_gradient_profiles.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SET 17 — CONDITION DENDROGRAM (Stage 9)
# ══════════════════════════════════════════════════════════════════════════════

def make_condition_dendrogram(fm: pd.DataFrame, feat_cols: List[str],
                              out_dir: Path) -> None:
    from sklearn.preprocessing import StandardScaler

    out_dir.mkdir(parents=True, exist_ok=True)

    fm2 = fm.copy()
    fm2["o2"] = fm2["o2"].apply(_norm_o2)
    grp = fm2.groupby(["particle_key", "o2"])[feat_cols].mean().reset_index()

    # Sort into canonical order
    grp["_pk_ord"] = grp["particle_key"].map(lambda x: PARTICLE_KEY_ORDER.index(x)
                                             if x in PARTICLE_KEY_ORDER else 99)
    grp["_o2_ord"] = grp["o2"].map(lambda x: O2_ORDERED.index(x)
                                   if x in O2_ORDERED else 99)
    grp = grp.sort_values(["_pk_ord", "_o2_ord"]).reset_index(drop=True)

    X = StandardScaler().fit_transform(grp[feat_cols].values)
    D = pdist(X, metric="euclidean")
    Z = linkage(D, method="complete")

    cond_labels = [
        f"{PARTICLE_SHORT.get(r.particle_key, r.particle_key)} {O2_SHORT.get(r.o2, r.o2)}"
        for _, r in grp.iterrows()
    ]
    # Colour by particle species
    leaf_colors = [PARTICLE_COLORS.get(r.particle_key, "#AAAAAA")
                   for _, r in grp.iterrows()]

    fig, ax = plt.subplots(figsize=SIZE_DENDROGRAM)
    fig.patch.set_facecolor("white")

    dend = dendrogram(
        Z,
        labels=cond_labels,
        ax=ax,
        leaf_rotation=70,
        leaf_font_size=6.5,
        color_threshold=0,   # all black structure lines; we colour labels
        above_threshold_color="#AAAAAA",
    )
    # Colour x-tick labels by particle
    for tick_label, color in zip(ax.get_xticklabels(), leaf_colors):
        tick_label.set_color(color)
        tick_label.set_fontsize(6.0)

    ax.set_ylabel("Euclidean distance (standardised features)", labelpad=6)
    ax.set_xlabel("")
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    _style_ax(ax)

    handles_sp = [Patch(facecolor=SPECIES_COLORS[sp],
                        label=sp.capitalize(), edgecolor="none")
                  for sp in ["electron", "proton", "helium", "carbon"]]
    leg = ax.legend(handles=handles_sp, title="Species",
                    fontsize=LEGEND_SIZE, title_fontsize=LEGEND_SIZE + 0.5,
                    loc="upper right", framealpha=0.95, edgecolor="#CCCCCC")
    leg.get_title().set_fontweight("bold")

    _title_subtitle(
        ax,
        "Condition Hierarchical Clustering",
        "Complete-linkage dendrogram  ·  49 conditions  ·  standardised 107-feature means  ·  labels coloured by particle",
    )
    fig.tight_layout()
    _savefig(fig, out_dir / f"condition_dendrogram.{OUT_FORMAT}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    # Raise file-descriptor soft limit (macOS default 256 is insufficient)
    try:
        _fd_soft, _fd_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if _fd_soft < _fd_hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (_fd_hard, _fd_hard))
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description=(
            "Regenerate all publication figures from saved pipeline outputs "
            "(05_random_forest.py + 02_ph_topology_analysis.py + 06_additional_analyses.py)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--basedir", type=Path, default=Path("."),
                        help="Project root directory (default: current directory).")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Override output root (default: <basedir>/analysis/figures/).")
    parser.add_argument("--top-n", type=int, default=TOP_N,
                        help=f"Top-N features in importance/effect-size charts (default: {TOP_N}).")
    parser.add_argument("--skip-umap", action="store_true",
                        help="Skip UMAP embedding figures.")
    parser.add_argument("--skip-landscapes", action="store_true",
                        help="Skip persistence landscape figures.")
    parser.add_argument("--skip-additional", action="store_true",
                        help="Skip additional-analyses figures (Figure Sets 10–17).")
    args = parser.parse_args()

    base_dir = args.basedir.resolve()
    fig_dir  = (args.outdir.resolve() if args.outdir
                else base_dir / "analysis" / "figures")

    logger.info("=" * 64)
    logger.info("07_regenerate_figures.py")
    logger.info(f"  Base dir  : {base_dir}")
    logger.info(f"  Figures   : {fig_dir}")
    logger.info(f"  DPI       : {DPI}")
    logger.info(f"  Top-N     : {args.top_n}")
    logger.info("=" * 64)

    try:
        results, ablation, feat_cols, fm, ph_summary, wass, ph_dir, additional = \
            load_data(base_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc)); return 1

    # ── Figure Set 1 — Confusion matrices ────────────────────────────────────
    logger.info("─" * 64)
    logger.info("=== Figure Set 1: Confusion matrices ===")
    make_confusion(results, fig_dir / "confusion")

    # ── Figure Set 2 — Feature importance ────────────────────────────────────
    logger.info("─" * 64)
    logger.info("=== Figure Set 2: Feature importance ===")
    make_importance(results, feat_cols, fig_dir / "importance", args.top_n)

    # ── Figure Set 3 — Modality ablation ─────────────────────────────────────
    logger.info("─" * 64)
    logger.info("=== Figure Set 3: Modality ablation ===")
    make_ablation(ablation, fig_dir / "ablation")

    # ── Figure Set 4 — O₂ summary (Task 1) ───────────────────────────────────
    logger.info("─" * 64)
    logger.info("=== Figure Set 4: O₂ summary (Task 1) ===")
    make_o2_summary(results, fig_dir / "summary")

    # ── Figure Set 5 — SOBP summary (Task 4) ─────────────────────────────────
    logger.info("─" * 64)
    logger.info("=== Figure Set 5: SOBP summary (Task 4) ===")
    make_sobp_summary(results, fig_dir / "summary")

    # ── Figure Set 6 — UMAP ──────────────────────────────────────────────────
    logger.info("─" * 64)
    logger.info("=== Figure Set 6: UMAP embeddings ===")
    if args.skip_umap:
        logger.info("  Skipped (--skip-umap).")
    else:
        make_umap(wass, fm, fig_dir / "ph")

    # ── Figure Set 7 — Within/between violins ────────────────────────────────
    logger.info("─" * 64)
    logger.info("=== Figure Set 7: Within/between violins ===")
    make_violin(wass, fm, ph_summary, fig_dir / "ph")

    # ── Figure Set 8 — Condition heatmaps ────────────────────────────────────
    logger.info("─" * 64)
    logger.info("=== Figure Set 8: Condition Wasserstein heatmaps ===")
    make_heatmaps(ph_summary, fig_dir / "ph")

    # ── Figure Set 9 — Persistence landscapes ────────────────────────────────
    logger.info("─" * 64)
    logger.info("=== Figure Set 9: Persistence landscapes ===")
    if args.skip_landscapes:
        logger.info("  Skipped (--skip-landscapes).")
    else:
        make_landscapes(ph_dir, fig_dir / "ph")

    # ── Figure Sets 10–17 — Additional analyses ───────────────────────────────
    if args.skip_additional:
        logger.info("─" * 64)
        logger.info("=== Figure Sets 10–17: Skipped (--skip-additional). ===")
    else:
        add_out = fig_dir / "additional"

        logger.info("─" * 64)
        logger.info("=== Figure Set 10: Cross-modality correlation ===")
        make_cross_modality_correlation(
            additional.get("cross_modality_correlation"), add_out)

        logger.info("─" * 64)
        logger.info("=== Figure Set 11: Single-modality O₂ accuracy matrix ===")
        make_single_modality_o2_accuracy(
            additional.get("single_modality_o2_accuracy"), add_out)

        logger.info("─" * 64)
        logger.info("=== Figure Set 12: LET–O₂ encoding profiles ===")
        make_let_o2_profiles(
            additional.get("single_modality_o2_accuracy"), add_out)

        logger.info("─" * 64)
        logger.info("=== Figure Set 13: Effect sizes (η²) ===")
        make_effect_sizes(
            additional.get("effect_sizes"), add_out, top_n=args.top_n)

        logger.info("─" * 64)
        logger.info("=== Figure Set 14: Condition PCA ===")
        make_condition_pca(
            additional.get("condition_pca"), fm, feat_cols, add_out)

        logger.info("─" * 64)
        logger.info("=== Figure Set 15: SOBP effect sizes ===")
        make_sobp_effect_sizes(
            additional.get("sobp_effect_sizes"), add_out, top_n=args.top_n)

        logger.info("─" * 64)
        logger.info("=== Figure Set 16: O₂ gradient profiles ===")
        make_o2_gradient_profiles(
            fm, feat_cols, additional.get("effect_sizes"), add_out)

        logger.info("─" * 64)
        logger.info("=== Figure Set 17: Condition dendrogram ===")
        make_condition_dendrogram(fm, feat_cols, add_out)

    logger.info("=" * 64)
    logger.info(f"All figures written to: {fig_dir}")
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
