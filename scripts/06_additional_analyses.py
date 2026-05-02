#!/usr/bin/env python3
"""
================================================================================
06_additional_analyses.py
================================================================================
Supplementary analyses bridging [2]'s Random Forest findings to the sheaf
neural network ([3]) and digital twin ([4]) downstream projects.

PIPELINE POSITION
-----------------
  01  extract_dsb.py
  02  ph_topology_analysis.py    →  analysis/ph/
  03  compute_features.py        →  analysis/features/
  04  build_feature_matrix.py    →  analysis/feature_matrix.csv
  05  random_forest.py           →  analysis/rf/
  06  additional_analyses.py     →  analysis/additional/      ← THIS
  07  regenerate_figures.py      →  analysis/figures/  (600 DPI)

NOTE: Script 00 (parse_sdd_particle_history.py) is no longer part of the
pipeline. It existed solely for the old Event Attribution modality (m7).
With that modality retired in favour of Topological Summaries, script 00
can be deleted.

PURPOSE
-------
Each stage addresses a specific question that [3] and [4] require answered:

  Stage 1 — Condition feature statistics
      Per-condition mean ± SD for all 107 features across all 49 conditions.
      Primary export consumed by [3] as context for interpreting consistency
      score variation and by [4] for GBDT surrogate feature ranges.

  Stage 2 — Cross-modality Pearson correlation analysis
      7×7 correlation matrix between modality-level mean feature vectors
      across all 49 conditions. Directly anticipates [3]'s sheaf edge
      structure: which modality pairs are globally coherent and which are
      globally decoupled across the LET × pO₂ landscape?

  Stage 3 — Single-modality O2 classification
      For each of the 7 particle configurations × 7 modalities, train a
      7-class O2 Random Forest using only that modality's features. The
      resulting 7×7 accuracy matrix is the empirical basis for [3]'s
      §E.4 Prediction 2 (Δc_v sign pattern matching RF importance loss).
      Computationally lighter than 05 (200 trees, 5-fold × 5-repeat).

  Stage 4 — LET-dependent O2 encoding profile
      From Stage 3, for each modality, plot 7-class O2 balanced accuracy
      vs mean LET of each particle configuration. Shows the decoupling
      gradient that motivates [3]'s directed edge asymmetry and the
      expected consistency score suppression at high LET.

  Stage 5 — Feature effect sizes (η²)
      One-way ANOVA η² for (a) O2 effect across 7 levels and (b) particle
      effect across 7 configs, for every feature. Top-N features by each
      effect type, coloured by modality. Formal basis for predicting which
      sheaf nodes carry the O2 vs. particle signal.

  Stage 6 — Condition PCA
      PCA of the 49-condition × 107-feature matrix (condition means,
      standardised). Shows the effective dimensionality of the condition
      space, how conditions cluster by particle vs. O2, and which features
      dominate the first two principal components.

  Stage 7 — O2 gradient profiles
      For the highest-η²(O2) feature in each modality, plot mean ± SD
      value across the 7 O2 levels for each of the 7 particle configs.
      Annotates the nearest sampled O₂ transition boundary (0.5%), validating that the
      O2 sampling design densely covers the biologically informative region.

  Stage 8 — SOBP position effect
      For each species with both SOBP positions (proton, helium, carbon),
      Mann-Whitney U test + rank-biserial effect size r for every feature.
      Top features by |r| per species, coloured by modality. Validates
      Task 4 of 05 and supports [3]'s SOBP position analysis.

  Stage 9 — Condition hierarchical clustering
      Complete-linkage hierarchical clustering of the 49 conditions on
      standardised feature means. Dendrogram coloured by particle species.
      Shows whether unsupervised structure is dominated by particle (LET)
      or O2 axes.

OUTPUTS  (→ analysis/additional/)
-------
  condition_feature_means.csv          Stage 1 — 49 × 107 mean matrix
  condition_feature_stats.json         Stage 1 — full mean/SD/Q25/Q75
  cross_modality_correlation.png       Stage 2
  cross_modality_correlation.json      Stage 2
  single_modality_o2_accuracy.png      Stage 3
  single_modality_o2_accuracy.json     Stage 3
  let_o2_modality_profiles.png         Stage 4
  feature_o2_effect_sizes.png          Stage 5a
  feature_particle_effect_sizes.png    Stage 5b
  effect_sizes.json                    Stage 5
  condition_pca.png                    Stage 6
  condition_pca.json                   Stage 6
  o2_gradient_profiles.png             Stage 7
  sobp_effect_sizes.png                Stage 8
  sobp_effect_sizes.json               Stage 8
  condition_dendrogram.png             Stage 9
  additional_analyses_summary.json     All stages — meta-summary

USAGE
-----
  python 06_additional_analyses.py
  python 06_additional_analyses.py --basedir /path/to/project
  python 06_additional_analyses.py --skip-single-modality-rf
  python 06_additional_analyses.py --top-n 20

DEPENDENCIES
------------
  numpy, pandas, scipy, scikit-learn, matplotlib
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — keep in sync with 02, 03, 04, 05                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PARTICLE_KEY_ORDER: List[str] = [
    "electron_mono",
    "proton_psobp",
    "proton_dsobp",
    "helium_psobp",
    "helium_dsobp",
    "carbon_psobp",
    "carbon_dsobp",
]

PARTICLE_LET: Dict[str, float] = {
    "electron_mono":  0.2,
    "proton_psobp":   4.6,
    "proton_dsobp":   8.1,
    "helium_psobp":  10.0,
    "helium_dsobp":  30.0,
    "carbon_psobp":  40.9,
    "carbon_dsobp":  70.7,
}

PARTICLE_LABELS: Dict[str, str] = {
    "electron_mono":  "Electron mono (0.2 keV/µm)",
    "proton_psobp":   "Proton pSOBP (4.6 keV/µm)",
    "proton_dsobp":   "Proton dSOBP (8.1 keV/µm)",
    "helium_psobp":   "Helium pSOBP (10 keV/µm)",
    "helium_dsobp":   "Helium dSOBP (30 keV/µm)",
    "carbon_psobp":   "Carbon pSOBP (40.9 keV/µm)",
    "carbon_dsobp":   "Carbon dSOBP (70.7 keV/µm)",
}

PARTICLE_SHORT: Dict[str, str] = {
    "electron_mono":  "e⁻",
    "proton_psobp":   "p⁺ p",
    "proton_dsobp":   "p⁺ d",
    "helium_psobp":   "He p",
    "helium_dsobp":   "He d",
    "carbon_psobp":   "C p",
    "carbon_dsobp":   "C d",
}

PARTICLE_SPECIES: Dict[str, str] = {
    "electron_mono":  "electron",
    "proton_psobp":   "proton",
    "proton_dsobp":   "proton",
    "helium_psobp":   "helium",
    "helium_dsobp":   "helium",
    "carbon_psobp":   "carbon",
    "carbon_dsobp":   "carbon",
}

SOBP_SPECIES_KEYS: Dict[str, Tuple[str, str]] = {
    "proton":  ("proton_psobp",  "proton_dsobp"),
    "helium":  ("helium_psobp",  "helium_dsobp"),
    "carbon":  ("carbon_psobp",  "carbon_dsobp"),
}

O2_ORDERED: List[str] = [
    "21.0", "5.0", "2.1", "0.5", "0.1", "0.021", "0.005"
]
O2_LABELS: Dict[str, str] = {
    "21.0":  "21.0% (Norm.)",
    "5.0":   "5.0% (T.Norm.)",
    "2.1":   "2.1% (Mild)",
    "0.5":   "0.5%",
    "0.1":   "0.1% (Severe)",
    "0.021": "0.021% (Anoxic)",
    "0.005": "0.005% (True Anox.)",
}
O2_K_HALF: str = "0.5"     # nearest sampled level to VOxA K½ (0.265% not in design)

MODALITIES: Dict[str, str] = {
    "m1_": "Spatial Distribution",
    "m2_": "Radial Track Structure",
    "m3_": "Local Energy Heterogeneity",
    "m4_": "Dose Distribution",
    "m5_": "Genomic Location",
    "m6_": "Damage Complexity",
    "m7_": "Topological Summaries",
}
MODALITY_KEYS: List[str] = list(MODALITIES.keys())
MODALITY_NAMES: List[str] = list(MODALITIES.values())

# ── Local/global grouping (for [3] interpretation) ────────────────────────────
MODALITY_GROUP: Dict[str, str] = {
    "m1_": "track-local",
    "m2_": "track-local",
    "m3_": "intermediate",
    "m4_": "global",
    "m5_": "global",
    "m6_": "global",
    "m7_": "intermediate",
}

# ── RF parameters for Stage 3 (lighter than 05) ───────────────────────────────
RF_N_TREES:   int = 200
RF_N_SPLITS:  int = 5
RF_N_REPEATS: int = 5      # 25 folds — adequate for ranking, fast enough for 49 models
CHANCE_O2: float = 1.0 / len(O2_ORDERED)   # 1/7 ≈ 0.143

# ── Effect size ───────────────────────────────────────────────────────────────
TOP_N_FEATURES: int = 20


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  COLOR PALETTE — Amalfi Coast (consistent with 02, 05)                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PARTICLE_COLORS: Dict[str, str] = {
    "electron_mono":  "#37657E",
    "proton_psobp":   "#F09714",
    "proton_dsobp":   "#C97F0E",
    "helium_psobp":   "#6B8C5A",
    "helium_dsobp":   "#B5956A",
    "carbon_psobp":   "#CD5F00",
    "carbon_dsobp":   "#9B5878",
}

SPECIES_COLORS: Dict[str, str] = {
    "electron": "#37657E",
    "proton":   "#F09714",
    "helium":   "#6B8C5A",
    "carbon":   "#CD5F00",
}

O2_COLORS: List[str] = [
    "#1D4E63",  # 21.0  — deep offshore
    "#2A6070",  # 5.0   — near offshore
    "#37657E",  # 2.1   — marine
    "#508799",  # 0.5   — moderate hypoxia
    "#6FA3AE",  # 0.1   — shallow piscine
    "#8DC0C9",  # 0.021 — pale seafoam
    "#A8D4E0",  # 0.005 — seafoam
]
O2_COLOR_MAP: Dict[str, str] = dict(zip(O2_ORDERED, O2_COLORS))

MODALITY_COLORS: Dict[str, str] = {
    "m1_": "#37657E",
    "m2_": "#CD5F00",
    "m3_": "#C4922A",
    "m4_": "#1D4E63",
    "m5_": "#9B5878",
    "m6_": "#4A6B3A",
    "m7_": "#6B8C5A",
}
MODALITY_COLOR_NAME: Dict[str, str] = {
    MODALITIES[k]: MODALITY_COLORS[k] for k in MODALITY_KEYS
}

GROUP_COLORS: Dict[str, str] = {
    "track-local":  "#CD5F00",
    "intermediate": "#C4922A",
    "global":       "#1D4E63",
}

# Correlation heatmap: negative (chili) → zero (cream) → positive (marine)
_CORR_CMAP = LinearSegmentedColormap.from_list(
    "corr", ["#CD5F00", "#F5EFE7", "#1D4E63"], N=256
)

# Heatmap for accuracy / effect size: cream → marine
_ACC_CMAP = LinearSegmentedColormap.from_list(
    "acc", ["#F5EFE7", "#508799", "#1D4E63"], N=256
)

STRIP_FILL: str = "#E8DDD1"
STRIP_TEXT: str = "#1A1A1A"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GLOBAL MATPLOTLIB STYLE                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_HELVETICA = ["Helvetica", "Helvetica Neue", "Arial",
              "Liberation Sans", "DejaVu Sans"]

plt.rcParams.update({
    "font.family":           "sans-serif",
    "font.sans-serif":       _HELVETICA,
    "font.size":             9,
    "text.color":            "#1A1A1A",
    "axes.titlesize":        11,
    "axes.titleweight":      "bold",
    "axes.titlecolor":       "#1A1A1A",
    "axes.labelsize":        9,
    "axes.labelcolor":       "#1A1A1A",
    "xtick.labelsize":       8,
    "ytick.labelsize":       8,
    "xtick.color":           "#555555",
    "ytick.color":           "#555555",
    "legend.fontsize":       8,
    "legend.title_fontsize": 8.5,
    "legend.framealpha":     0.95,
    "legend.edgecolor":      "#CCCCCC",
    "figure.facecolor":      "white",
    "axes.facecolor":        "white",
    "savefig.facecolor":     "white",
    "axes.grid":             True,
    "axes.grid.which":       "major",
    "grid.color":            "#EBEBEB",
    "grid.linewidth":        0.6,
    "grid.linestyle":        "-",
    "axes.axisbelow":        True,
    "axes.edgecolor":        "#BBBBBB",
    "axes.linewidth":        0.7,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "xtick.major.size":      3.5,
    "ytick.major.size":      3.5,
    "xtick.major.width":     0.7,
    "ytick.major.width":     0.7,
    "figure.dpi":            150,
    "savefig.dpi":           150,
})


# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#555555", length=3.5, width=0.7)


def _title_sub(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color="#1A1A1A", loc="left", pad=4)
    if subtitle:
        ax.text(0.0, 1.035, subtitle, transform=ax.transAxes,
                fontsize=8, color="#666666", style="italic",
                ha="left", va="bottom")


def _strip(ax: plt.Axes, label: str) -> None:
    ax.add_patch(FancyBboxPatch(
        (0, 1.02), 1, 0.10,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=STRIP_FILL, zorder=5, clip_on=False,
        transform=ax.transAxes,
    ))
    ax.text(0.5, 1.07, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=8,
            fontweight="bold", color=STRIP_TEXT, zorder=6)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {path.name}")


def _norm_o2(v) -> str:
    try:
        f = float(v)
        for s in O2_ORDERED:
            if abs(f - float(s)) < 1e-9:
                return s
    except (ValueError, TypeError):
        pass
    return str(v)


def _modality_color(feat: str) -> str:
    for pfx, col in MODALITY_COLORS.items():
        if feat.startswith(pfx):
            return col
    return "#AAAAAA"


def _modality_name(feat: str) -> str:
    for pfx, name in MODALITIES.items():
        if feat.startswith(pfx):
            return name
    return "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_feature_matrix(analysis_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    path = analysis_dir / "feature_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"feature_matrix.csv not found at {path}\n"
            "  Run 04_build_feature_matrix.py first."
        )
    df = pd.read_csv(path)
    df["o2"] = df["o2"].apply(_norm_o2)

    if "particle_key" not in df.columns:
        raise ValueError(
            '"particle_key" column missing — re-run 03 and 04.'
        )

    feat_cols = [c for c in df.columns if c[:3] in MODALITIES]
    logger.info(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} cols "
                f"({len(feat_cols)} features)")
    return df, feat_cols


def load_rf_summary(analysis_dir: Path) -> Optional[Dict]:
    path = analysis_dir / "rf" / "results_summary.json"
    if not path.exists():
        logger.warning("  rf/results_summary.json not found — "
                       "Stage 3 baseline comparison unavailable.")
        return None
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — CONDITION FEATURE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def stage1_condition_stats(
    df: pd.DataFrame,
    feat_cols: List[str],
    out_dir: Path,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Per-condition mean ± SD / Q25 / Q75 for every feature.

    Returns:
        means_df   — DataFrame(49 rows × 107 cols), condition means
        stats_dict — nested dict for JSON export
    """
    logger.info("Stage 1: Condition feature statistics")
    groups = (
        df.groupby("particle_key", sort=False)[feat_cols]
        .apply(lambda g: g)
    )

    records = []
    stats_dict: Dict = {}

    for pk in PARTICLE_KEY_ORDER:
        if pk not in df["particle_key"].values:
            logger.warning(f"  particle_key '{pk}' absent from data — skipping.")
            continue
        sub_pk = df[df["particle_key"] == pk]
        stats_dict[pk] = {}

        for o2 in O2_ORDERED:
            sub = sub_pk[sub_pk["o2"] == o2]
            if len(sub) == 0:
                continue
            vals = sub[feat_cols].values.astype(float)
            rec = {"particle_key": pk, "o2": o2,
                   "n_runs": int(len(sub))}
            cstats: Dict = {"n_runs": int(len(sub))}
            for j, col in enumerate(feat_cols):
                v = vals[:, j]
                finite = v[np.isfinite(v)]
                if len(finite) == 0:
                    rec[col] = np.nan
                    cstats[col] = {"mean": None, "sd": None,
                                   "q25": None, "q75": None}
                else:
                    rec[col] = float(np.mean(finite))
                    cstats[col] = {
                        "mean": float(np.mean(finite)),
                        "sd":   float(np.std(finite, ddof=1)),
                        "q25":  float(np.percentile(finite, 25)),
                        "q75":  float(np.percentile(finite, 75)),
                    }
            records.append(rec)
            stats_dict[pk][o2] = cstats

    means_df = pd.DataFrame(records)
    means_df.to_csv(out_dir / "condition_feature_means.csv", index=False)
    logger.info(f"  condition_feature_means.csv: {means_df.shape}")

    with open(out_dir / "condition_feature_stats.json", "w") as fh:
        json.dump(stats_dict, fh, indent=2, allow_nan=True)
    logger.info("  condition_feature_stats.json saved.")

    return means_df, stats_dict


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — CROSS-MODALITY PEARSON CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def stage2_cross_modality_correlation(
    means_df: pd.DataFrame,
    feat_cols: List[str],
    out_dir: Path,
) -> np.ndarray:
    """
    Compute and plot the 7×7 modality-level Pearson correlation matrix.

    For each condition (row of means_df), compute the mean of all features
    within each modality → a 49×7 matrix of modality centroids. Then
    compute the 7×7 Pearson R matrix across the 49 conditions.

    High positive r: modalities co-vary across conditions (coherent signal).
    Near-zero r:     modalities are informationally independent.
    Negative r:      modalities carry opposing information across conditions.
    """
    logger.info("Stage 2: Cross-modality Pearson correlation")

    feat_only = means_df[feat_cols].values.astype(float)
    valid_rows = np.isfinite(feat_only).all(axis=1)
    feat_clean = feat_only[valid_rows]

    # Build 49 × 7 modality-centroid matrix
    mod_centroids = np.zeros((feat_clean.shape[0], len(MODALITY_KEYS)))
    for j, pfx in enumerate(MODALITY_KEYS):
        cols_j = [i for i, c in enumerate(feat_cols) if c.startswith(pfx)]
        if cols_j:
            mod_centroids[:, j] = np.nanmean(feat_clean[:, cols_j], axis=1)

    # 7×7 Pearson R
    corr_mat = np.full((7, 7), np.nan)
    for a in range(7):
        for b in range(7):
            r, _ = stats.pearsonr(mod_centroids[:, a], mod_centroids[:, b])
            corr_mat[a, b] = float(r)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    fig.patch.set_facecolor("white")

    im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap=_CORR_CMAP, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
    cbar.set_label("Pearson r  (across 49 conditions)", fontsize=8.5,
                   color="#555555")
    cbar.ax.tick_params(labelsize=7.5, colors="#555555")
    cbar.outline.set_edgecolor("#CCCCCC")

    ticks = list(range(7))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    labels = [f"{MODALITY_NAMES[i]}\n({MODALITY_GROUP[MODALITY_KEYS[i]]})"
              for i in range(7)]
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.grid(False)

    # Annotate each cell
    for a in range(7):
        for b in range(7):
            v = corr_mat[a, b]
            ax.text(b, a, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, fontweight="bold",
                    color="white" if abs(v) > 0.65 else "#1A1A1A")

    # Draw group-boundary lines (track-local | intermediate | global)
    # m1, m2 = track-local (0,1);  m3, m7 = intermediate (2, 6);
    # m4, m5, m6 = global (3,4,5)
    # Draw separator after m2 (between idx 1 and 2) and after m3 (between 2 and 3)
    # and after m6 (between 5 and 6) based on grouping order in MODALITY_KEYS
    # (order: m1,m2,m3,m4,m5,m6,m7)
    for pos in [1.5, 2.5, 5.5]:
        ax.axhline(pos, color="white", lw=1.8, zorder=5)
        ax.axvline(pos, color="white", lw=1.8, zorder=5)

    _style_ax(ax)
    _title_sub(
        ax,
        "Cross-Modality Pearson Correlation",
        "Modality centroids across 49 conditions (7 particle configs × 7 O₂ levels)  ·  "
        "white lines: local | intermediate | global boundaries",
    )
    fig.tight_layout()
    _save(fig, out_dir / "cross_modality_correlation.png")

    # ── JSON export ───────────────────────────────────────────────────────────
    corr_json = {
        "modality_names": MODALITY_NAMES,
        "correlation_matrix": corr_mat.tolist(),
        "n_conditions": int(feat_clean.shape[0]),
    }
    with open(out_dir / "cross_modality_correlation.json", "w") as fh:
        json.dump(corr_json, fh, indent=2)

    return corr_mat


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — SINGLE-MODALITY O2 CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def _run_single_modality_rf(
    X: np.ndarray,
    y: np.ndarray,
    task_name: str,
) -> float:
    """5-fold × 5-repeat RF, returns mean balanced accuracy."""
    rskf = RepeatedStratifiedKFold(
        n_splits=RF_N_SPLITS, n_repeats=RF_N_REPEATS, random_state=42
    )
    accs = []
    for tr, te in rskf.split(X, y):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr])
        Xte = imp.transform(X[te])
        rf  = RandomForestClassifier(
            n_estimators=RF_N_TREES, class_weight="balanced",
            max_features="sqrt", n_jobs=-1, random_state=42,
        )
        rf.fit(Xtr, y[tr])
        accs.append(balanced_accuracy_score(y[te], rf.predict(Xte)))
    return float(np.mean(accs))


def stage3_single_modality_o2(
    df: pd.DataFrame,
    feat_cols: List[str],
    out_dir: Path,
    rf_summary: Optional[Dict],
    skip: bool = False,
) -> np.ndarray:
    """
    For each of 7 particle configs × 7 modalities:
    7-class O2 RF balanced accuracy using only that modality's features.

    Returns acc_matrix of shape (7_modalities, 7_particle_configs).
    """
    if skip:
        logger.info("Stage 3: Single-modality O2 RF — skipped (--skip-single-modality-rf).")
        return np.full((len(MODALITY_KEYS), len(PARTICLE_KEY_ORDER)), np.nan)

    logger.info("Stage 3: Single-modality O2 classification")
    logger.info(f"  RF: {RF_N_TREES} trees, "
                f"{RF_N_SPLITS}-fold × {RF_N_REPEATS} repeats "
                f"= {RF_N_SPLITS * RF_N_REPEATS} folds per model")
    logger.info(f"  Total models: {len(MODALITY_KEYS)} modalities × "
                f"{len(PARTICLE_KEY_ORDER)} configs = "
                f"{len(MODALITY_KEYS) * len(PARTICLE_KEY_ORDER)}")

    o2_map = {o: i for i, o in enumerate(O2_ORDERED)}
    acc_matrix = np.full((len(MODALITY_KEYS), len(PARTICLE_KEY_ORDER)), np.nan)

    for j, pk in enumerate(PARTICLE_KEY_ORDER):
        sub = df[df["particle_key"] == pk].copy()
        if len(sub) < 10:
            logger.warning(f"  Skipping {pk}: only {len(sub)} rows.")
            continue
        y = sub["o2"].map(o2_map).values
        valid = ~np.isnan(y.astype(float))
        sub, y = sub[valid], y[valid].astype(int)

        for i, pfx in enumerate(MODALITY_KEYS):
            mod_cols = [c for c in feat_cols if c.startswith(pfx)]
            if not mod_cols:
                continue
            X = sub[mod_cols].values.astype(float)
            acc = _run_single_modality_rf(X, y, f"{pfx}_{pk}")
            acc_matrix[i, j] = acc
            logger.info(f"    {MODALITY_NAMES[i]:28s} | {PARTICLE_SHORT[pk]:6s} | "
                        f"bal_acc = {acc:.4f}")

    # ── Figure — annotated accuracy heatmap ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    fig.patch.set_facecolor("white")

    im = ax.imshow(acc_matrix, vmin=CHANCE_O2, vmax=1.0,
                   cmap=_ACC_CMAP, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("7-class O₂ balanced accuracy", fontsize=8.5, color="#555555")
    cbar.ax.tick_params(labelsize=7.5, colors="#555555")
    cbar.outline.set_edgecolor("#CCCCCC")

    ax.set_xticks(range(len(PARTICLE_KEY_ORDER)))
    ax.set_yticks(range(len(MODALITY_KEYS)))
    ax.set_xticklabels([PARTICLE_SHORT[pk] for pk in PARTICLE_KEY_ORDER],
                       fontsize=8.5)
    ax.set_yticklabels(
        [f"{MODALITY_NAMES[i]}  [{MODALITY_GROUP[MODALITY_KEYS[i]]}]"
         for i in range(len(MODALITY_KEYS))],
        fontsize=8,
    )
    ax.set_xlabel("Particle configuration  →  LET (keV/µm)", labelpad=6)
    ax.set_ylabel("Modality", labelpad=6)
    ax.grid(False)

    for i in range(len(MODALITY_KEYS)):
        for j in range(len(PARTICLE_KEY_ORDER)):
            v = acc_matrix[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=7, fontweight="bold",
                    color="white" if v > 0.60 else "#1A1A1A")

    # Vertical dashed line separating SOBP pairs
    ax.axvline(0.5,  color="white", lw=1.2, ls="--", zorder=5)
    ax.axvline(2.5,  color="white", lw=1.2, ls="--", zorder=5)
    ax.axvline(4.5,  color="white", lw=1.2, ls="--", zorder=5)

    # Chance reference
    ax.text(1.01, -0.07, f"Chance = {CHANCE_O2:.3f}",
            transform=ax.transAxes, fontsize=7.5, color="#555555",
            ha="right", va="top", style="italic")

    _style_ax(ax)
    _title_sub(
        ax,
        "Single-Modality O₂ Classification — Balanced Accuracy",
        f"7-class  ·  {RF_N_SPLITS}-fold × {RF_N_REPEATS} repeats  ·  "
        f"RF {RF_N_TREES} trees  ·  chance = {CHANCE_O2:.3f}  ·  "
        "vertical dashes: species boundaries",
    )
    fig.tight_layout()
    _save(fig, out_dir / "single_modality_o2_accuracy.png")

    # ── JSON export ───────────────────────────────────────────────────────────
    acc_json = {
        "modality_names":  MODALITY_NAMES,
        "modality_keys":   MODALITY_KEYS,
        "particle_keys":   PARTICLE_KEY_ORDER,
        "particle_shorts": [PARTICLE_SHORT[pk] for pk in PARTICLE_KEY_ORDER],
        "particle_lets":   [PARTICLE_LET[pk] for pk in PARTICLE_KEY_ORDER],
        "accuracy_matrix": acc_matrix.tolist(),
        "chance_level":    CHANCE_O2,
        "cv_config": {
            "n_splits":  RF_N_SPLITS,
            "n_repeats": RF_N_REPEATS,
            "n_trees":   RF_N_TREES,
        },
    }
    with open(out_dir / "single_modality_o2_accuracy.json", "w") as fh:
        json.dump(acc_json, fh, indent=2)

    return acc_matrix


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — LET-DEPENDENT O2 ENCODING PROFILE
# ══════════════════════════════════════════════════════════════════════════════

def stage4_let_o2_profiles(
    acc_matrix: np.ndarray,
    out_dir: Path,
) -> None:
    """
    For each modality, plot 7-class O2 balanced accuracy vs. mean LET of
    each particle configuration.  Shows the decoupling gradient that motivates
    [3]'s directed edge asymmetry hypothesis.
    """
    logger.info("Stage 4: LET-dependent O2 encoding profiles")

    if np.all(~np.isfinite(acc_matrix)):
        logger.warning("  All NaN (Stage 3 skipped) — skipping Stage 4.")
        return

    let_values = np.array([PARTICLE_LET[pk] for pk in PARTICLE_KEY_ORDER])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.patch.set_facecolor("white")

    for i, (pfx, name) in enumerate(MODALITIES.items()):
        row = acc_matrix[i, :]
        valid = np.isfinite(row)
        if valid.sum() < 2:
            continue
        group = MODALITY_GROUP[pfx]
        ls = "-" if group == "track-local" else \
             "--" if group == "global" else "-."
        ax.plot(let_values[valid], row[valid],
                color=MODALITY_COLORS[pfx], linewidth=2.0, linestyle=ls,
                marker="o", markersize=6, alpha=0.9, label=name, zorder=3)

    ax.axhline(CHANCE_O2, color="#888888", ls=":", lw=1.2, zorder=2,
               label=f"Chance ({CHANCE_O2:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("Mean LET (keV/µm)  [log scale]", labelpad=6)
    ax.set_ylabel("7-class O₂ balanced accuracy", labelpad=6)
    ax.set_ylim(0, 1.05)

    # Annotate particle configs on x-axis
    for pk in PARTICLE_KEY_ORDER:
        ax.axvline(PARTICLE_LET[pk], color="#EBEBEB", lw=0.7, zorder=1)
        ax.text(PARTICLE_LET[pk], 0.005, PARTICLE_SHORT[pk],
                ha="center", va="bottom", fontsize=7, color="#888888",
                rotation=90)

    # Legend: solid = track-local, dashed = global, dash-dot = intermediate
    from matplotlib.lines import Line2D
    line_handles = [
        Line2D([0], [0], color="#888888", lw=1.5, ls="-",
               label="track-local (m1, m2)"),
        Line2D([0], [0], color="#888888", lw=1.5, ls="--",
               label="global (m4, m5, m6)"),
        Line2D([0], [0], color="#888888", lw=1.5, ls="-.",
               label="intermediate (m3, m7)"),
    ]
    mod_handles = [
        Patch(facecolor=MODALITY_COLORS[pfx], label=MODALITIES[pfx],
              edgecolor="none")
        for pfx in MODALITY_KEYS
    ]
    leg1 = ax.legend(handles=mod_handles, title="Modality",
                     title_fontsize=8.5, fontsize=8,
                     loc="upper right", framealpha=0.95, edgecolor="#CCCCCC")
    ax.add_artist(leg1)
    ax.legend(handles=line_handles, title="Group",
              title_fontsize=8, fontsize=7.5,
              loc="center right", framealpha=0.95, edgecolor="#CCCCCC")

    _style_ax(ax)
    _title_sub(
        ax,
        "LET-Dependent O₂ Encoding: Single-Modality Accuracy",
        "Motivates [3]'s directed edge asymmetry — modalities that decouple at "
        "high LET show the largest accuracy drop",
    )
    fig.tight_layout()
    _save(fig, out_dir / "let_o2_modality_profiles.png")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — FEATURE EFFECT SIZES (η²)
# ══════════════════════════════════════════════════════════════════════════════

def _eta_squared(groups: List[np.ndarray]) -> float:
    """One-way ANOVA η² = SS_between / SS_total. Returns NaN on failure."""
    if len(groups) < 2:
        return np.nan
    all_vals = np.concatenate(groups)
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) < len(groups) + 1:
        return np.nan
    grand_mean = all_vals.mean()
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    if ss_total == 0:
        return 0.0
    ss_between = sum(
        len(g[np.isfinite(g)]) * (np.nanmean(g) - grand_mean) ** 2
        for g in groups if len(g[np.isfinite(g)]) > 0
    )
    return float(ss_between / ss_total)


def stage5_effect_sizes(
    df: pd.DataFrame,
    feat_cols: List[str],
    out_dir: Path,
    top_n: int = TOP_N_FEATURES,
) -> Dict:
    """
    For every feature, compute:
      η²(O2)       — one-way ANOVA across 7 O2 levels (pooling all particle configs)
      η²(particle) — one-way ANOVA across 7 particle configs (pooling all O2 levels)
    """
    logger.info("Stage 5: Feature effect sizes (η²)")

    eta_o2  = np.zeros(len(feat_cols))
    eta_pk  = np.zeros(len(feat_cols))

    for k, col in enumerate(feat_cols):
        # O2 effect
        groups_o2 = [
            df[df["o2"] == o2][col].values.astype(float)
            for o2 in O2_ORDERED
        ]
        eta_o2[k] = _eta_squared(groups_o2)

        # Particle effect
        groups_pk = [
            df[df["particle_key"] == pk][col].values.astype(float)
            for pk in PARTICLE_KEY_ORDER
        ]
        eta_pk[k] = _eta_squared(groups_pk)

    # ── Helper: horizontal bar chart ──────────────────────────────────────────
    def _eta_bar(eta_vals: np.ndarray, title: str,
                 subtitle: str, out_path: Path) -> None:
        idx   = np.argsort(eta_vals)[::-1][:top_n]
        vals  = eta_vals[idx]
        names = [feat_cols[i].replace("_", " ") for i in idx]
        cols  = [_modality_color(feat_cols[i]) for i in idx]

        fig, ax = plt.subplots(figsize=(8.5, max(5.0, top_n * 0.32)))
        fig.patch.set_facecolor("white")

        y = np.arange(top_n)
        ax.barh(y[::-1], vals, color=cols,
                edgecolor="white", linewidth=0.25, height=0.74)
        ax.set_yticks(y[::-1])
        ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlabel("η² (proportion of variance explained)", labelpad=6)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        handles = [
            Patch(facecolor=MODALITY_COLORS[pfx], label=MODALITIES[pfx],
                  edgecolor="none")
            for pfx in MODALITY_KEYS
        ]
        leg = ax.legend(handles=handles, title="Modality",
                        title_fontsize=8.5, fontsize=8,
                        loc="lower right", framealpha=0.95, edgecolor="#CCCCCC")
        leg.get_title().set_fontweight("bold")

        _style_ax(ax)
        _title_sub(ax, title, subtitle)
        fig.tight_layout()
        _save(fig, out_path)

    _eta_bar(
        eta_o2,
        "Top Features by O₂ Effect Size (η²)",
        "One-way ANOVA across 7 O₂ levels  ·  pooled over all particle configs",
        out_dir / "feature_o2_effect_sizes.png",
    )
    _eta_bar(
        eta_pk,
        "Top Features by Particle Effect Size (η²)",
        "One-way ANOVA across 7 particle configs  ·  pooled over all O₂ levels",
        out_dir / "feature_particle_effect_sizes.png",
    )

    # ── JSON export ───────────────────────────────────────────────────────────
    result = {
        "features":        feat_cols,
        "eta2_o2":         [float(v) for v in eta_o2],
        "eta2_particle":   [float(v) for v in eta_pk],
        "top_o2_features": [feat_cols[i] for i in np.argsort(eta_o2)[::-1][:top_n]],
        "top_pk_features": [feat_cols[i] for i in np.argsort(eta_pk)[::-1][:top_n]],
    }
    with open(out_dir / "effect_sizes.json", "w") as fh:
        json.dump(result, fh, indent=2)

    # Log top-5 per modality for O2 effect
    logger.info("  Top η²(O2) feature per modality:")
    for pfx, name in MODALITIES.items():
        mod_idx = [k for k, c in enumerate(feat_cols) if c.startswith(pfx)]
        if not mod_idx:
            continue
        best = max(mod_idx, key=lambda k: eta_o2[k])
        logger.info(f"    {name:28s}: {feat_cols[best]:35s}  η² = {eta_o2[best]:.4f}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — CONDITION PCA
# ══════════════════════════════════════════════════════════════════════════════

def stage6_condition_pca(
    means_df: pd.DataFrame,
    feat_cols: List[str],
    out_dir: Path,
) -> Dict:
    """
    PCA on the 49-condition × 107-feature matrix (standardised condition means).
    Visualises condition clustering in PC1-PC2 space.
    """
    logger.info("Stage 6: Condition PCA")

    # Build ordered condition matrix
    rows, labels_pk, labels_o2 = [], [], []
    for pk in PARTICLE_KEY_ORDER:
        for o2 in O2_ORDERED:
            sub = means_df[
                (means_df["particle_key"] == pk) & (means_df["o2"] == o2)
            ]
            if len(sub) == 0:
                continue
            rows.append(sub[feat_cols].values[0])
            labels_pk.append(pk)
            labels_o2.append(o2)

    X  = np.array(rows, dtype=float)
    # Impute any remaining NaNs with column mean before PCA
    col_means = np.nanmean(X, axis=0)
    nan_mask  = ~np.isfinite(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    pca    = PCA(n_components=min(10, Xs.shape[0], Xs.shape[1]))
    coords = pca.fit_transform(Xs)
    var_exp = pca.explained_variance_ratio_

    n_conds = len(labels_pk)

    # ── Figure: 2 panels ──────────────────────────────────────────────────────
    fig, (ax_pk, ax_o2) = plt.subplots(1, 2, figsize=(13.0, 5.8))
    fig.patch.set_facecolor("white")

    # Panel A: colour by particle config
    for pk in PARTICLE_KEY_ORDER:
        mask = [i for i, p in enumerate(labels_pk) if p == pk]
        ax_pk.scatter(coords[mask, 0], coords[mask, 1],
                      c=PARTICLE_COLORS[pk], label=PARTICLE_SHORT[pk],
                      s=55, alpha=0.9, linewidths=0, zorder=3)
        # Label centroid
        if mask:
            cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
            ax_pk.text(cx, cy, PARTICLE_SHORT[pk],
                       ha="center", va="center", fontsize=7,
                       fontweight="bold", color="white",
                       bbox=dict(boxstyle="round,pad=0.1",
                                 facecolor=PARTICLE_COLORS[pk],
                                 edgecolor="none", alpha=0.8),
                       zorder=4)

    ax_pk.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", labelpad=6)
    ax_pk.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)", labelpad=6)
    leg = ax_pk.legend(title="Particle config", title_fontsize=8.5,
                       fontsize=8, framealpha=0.95, edgecolor="#CCCCCC",
                       ncol=2)
    leg.get_title().set_fontweight("bold")
    _style_ax(ax_pk)
    _title_sub(ax_pk, "Condition PCA — by Particle Config",
               f"n = {n_conds} conditions  ·  "
               f"PC1+PC2 = {(var_exp[0]+var_exp[1])*100:.1f}% variance")

    # Panel B: colour by O2 level
    for o2 in O2_ORDERED:
        mask = [i for i, o in enumerate(labels_o2) if o == o2]
        if not mask:
            continue
        ax_o2.scatter(coords[mask, 0], coords[mask, 1],
                      c=O2_COLOR_MAP[o2],
                      label=O2_LABELS[o2],
                      s=55, alpha=0.9, linewidths=0, zorder=3)

    ax_o2.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", labelpad=6)
    ax_o2.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)", labelpad=6)
    leg2 = ax_o2.legend(title="O₂ level", title_fontsize=8.5,
                        fontsize=7.5, framealpha=0.95, edgecolor="#CCCCCC",
                        ncol=1)
    leg2.get_title().set_fontweight("bold")
    _style_ax(ax_o2)
    _title_sub(ax_o2, "Condition PCA — by O₂ Level",
               f"0.5% O₂ (nearest sampled to VOxA K½) shown in piscine (#508799)")

    fig.tight_layout()
    _save(fig, out_dir / "condition_pca.png")

    # ── Scree plot ────────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
    fig2.patch.set_facecolor("white")
    n_pc = len(var_exp)
    ax2.bar(range(1, n_pc + 1), var_exp * 100,
            color="#508799", edgecolor="white", linewidth=0.4, width=0.72)
    ax2.plot(range(1, n_pc + 1), np.cumsum(var_exp) * 100,
             color="#CD5F00", marker="o", markersize=5, lw=1.5,
             label="Cumulative variance")
    ax2.axhline(90, color="#CCCCCC", ls=":", lw=0.9)
    ax2.set_xlabel("Principal component", labelpad=6)
    ax2.set_ylabel("Variance explained (%)", labelpad=6)
    ax2.legend(fontsize=8, framealpha=0.95, edgecolor="#CCCCCC")
    _style_ax(ax2)
    _title_sub(ax2, "Condition PCA — Scree Plot",
               f"{n_conds} conditions × {len(feat_cols)} features (standardised)")
    fig2.tight_layout()
    _save(fig2, out_dir / "condition_pca_scree.png")

    result = {
        "n_conditions":       n_conds,
        "n_features":         len(feat_cols),
        "variance_explained": [float(v) for v in var_exp],
        "cumulative_var_90_pcs": int(np.searchsorted(
            np.cumsum(var_exp), 0.90) + 1),
        "pc1_pc2_variance":   float((var_exp[0] + var_exp[1]) * 100),
    }
    with open(out_dir / "condition_pca.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — O2 GRADIENT PROFILES
# ══════════════════════════════════════════════════════════════════════════════

def stage7_o2_gradient_profiles(
    df: pd.DataFrame,
    feat_cols: List[str],
    effect_sizes: Dict,
    out_dir: Path,
) -> None:
    """
    For the highest-η²(O2) feature in each modality, plot mean ± SD across
    the 7 O2 levels for each of the 7 particle configs.  Annotates the
    nearest sampled O₂ transition boundary (0.5%).
    """
    logger.info("Stage 7: O2 gradient profiles for top features")

    eta_o2  = np.array(effect_sizes["eta2_o2"])
    best_feat: Dict[str, str] = {}
    for pfx in MODALITY_KEYS:
        idx = [k for k, c in enumerate(feat_cols) if c.startswith(pfx)]
        if idx:
            best_k = max(idx, key=lambda k: eta_o2[k])
            best_feat[pfx] = feat_cols[best_k]

    n_mod = len(best_feat)
    if n_mod == 0:
        logger.warning("  No features found — skipping Stage 7.")
        return

    n_cols = 4
    n_rows = (n_mod + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    o2_x = np.arange(len(O2_ORDERED))
    k_half_idx = O2_ORDERED.index(O2_K_HALF)

    for panel_idx, (pfx, feat) in enumerate(best_feat.items()):
        row_i = panel_idx // n_cols
        col_i = panel_idx % n_cols
        ax    = axes[row_i][col_i]

        for pk in PARTICLE_KEY_ORDER:
            sub_pk = df[df["particle_key"] == pk]
            means  = []
            sds    = []
            for o2 in O2_ORDERED:
                vals = sub_pk[sub_pk["o2"] == o2][feat].values.astype(float)
                finite = vals[np.isfinite(vals)]
                means.append(float(np.mean(finite)) if len(finite) else np.nan)
                sds.append(float(np.std(finite, ddof=1))
                           if len(finite) > 1 else 0.0)

            means_a = np.array(means)
            sds_a   = np.array(sds)
            valid   = np.isfinite(means_a)
            if not valid.any():
                continue

            ax.plot(o2_x[valid], means_a[valid],
                    color=PARTICLE_COLORS[pk], linewidth=1.8,
                    marker="o", markersize=4.5, alpha=0.9, zorder=3,
                    label=PARTICLE_SHORT[pk])
            ax.fill_between(
                o2_x[valid],
                (means_a - sds_a)[valid],
                (means_a + sds_a)[valid],
                color=PARTICLE_COLORS[pk], alpha=0.10, zorder=2,
            )

        # K½ annotation
        ax.axvline(k_half_idx, color="#508799", ls="--",
                   lw=1.1, zorder=4, alpha=0.8)
        ax.text(k_half_idx + 0.05, ax.get_ylim()[1],
                "0.5%", color="#508799", fontsize=7, va="top",
                fontweight="bold")

        ax.set_xticks(o2_x)
        ax.set_xticklabels(
            [O2_LABELS[o].replace(" (", "\n(") for o in O2_ORDERED],
            fontsize=6, rotation=25, ha="right",
        )
        ax.set_ylabel(feat.replace("_", " "), fontsize=7.5, labelpad=4)

        name = MODALITIES[pfx]
        eta  = eta_o2[feat_cols.index(feat)]
        _strip(ax, f"{name}  (η²={eta:.3f})")
        _style_ax(ax)

        if panel_idx == 0:
            ax.legend(title="Particle", fontsize=7, title_fontsize=7.5,
                      framealpha=0.95, edgecolor="#CCCCCC",
                      loc="upper right", ncol=2)

    # Hide unused panels
    for panel_idx in range(n_mod, n_rows * n_cols):
        axes[panel_idx // n_cols][panel_idx % n_cols].set_visible(False)

    fig.text(0.01, 0.99,
             "O₂ Gradient Profiles — Highest η²(O₂) Feature per Modality",
             fontsize=12, fontweight="bold", color="#1A1A1A", va="top")
    fig.text(0.01, 0.966,
             "Mean ± 1 SD across 50 runs  ·  dashed = 0.5% O₂  ·  "
             "7 particle configs coloured by Amalfi palette",
             fontsize=8.5, color="#666666", style="italic", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir / "o2_gradient_profiles.png")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — SOBP POSITION EFFECT
# ══════════════════════════════════════════════════════════════════════════════

def stage8_sobp_effect(
    df: pd.DataFrame,
    feat_cols: List[str],
    out_dir: Path,
    top_n: int = TOP_N_FEATURES,
) -> Dict:
    """
    Mann-Whitney U + rank-biserial r between pSOBP and dSOBP for each species.
    """
    logger.info("Stage 8: SOBP position effect sizes")

    sobp_results: Dict[str, Dict] = {}
    species_order = ["proton", "helium", "carbon"]

    for sp in species_order:
        pk_p, pk_d = SOBP_SPECIES_KEYS[sp]
        sub_p = df[df["particle_key"] == pk_p]
        sub_d = df[df["particle_key"] == pk_d]
        if len(sub_p) == 0 or len(sub_d) == 0:
            logger.warning(f"  Missing data for {sp} SOBP pair — skipping.")
            continue

        r_vals: Dict[str, float] = {}
        u_pvals: Dict[str, float] = {}

        for col in feat_cols:
            a = sub_p[col].values.astype(float)
            b = sub_d[col].values.astype(float)
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            if len(a) < 3 or len(b) < 3:
                r_vals[col]  = np.nan
                u_pvals[col] = np.nan
                continue
            u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
            n1, n2 = len(a), len(b)
            # rank-biserial: r = 1 - 2U/(n1*n2)
            r_rb = float(1.0 - 2.0 * u_stat / (n1 * n2))
            r_vals[col]  = r_rb
            u_pvals[col] = float(p_val)

        sobp_results[sp] = {
            "pk_proximal":   pk_p,
            "pk_distal":     pk_d,
            "let_proximal":  PARTICLE_LET[pk_p],
            "let_distal":    PARTICLE_LET[pk_d],
            "rank_biserial": r_vals,
            "mannwhitney_p": u_pvals,
        }
        top_by_r = sorted(
            [(c, abs(r_vals[c])) for c in feat_cols if np.isfinite(r_vals.get(c, np.nan))],
            key=lambda x: x[1], reverse=True,
        )[:top_n]
        top_feats = [c for c, _ in top_by_r]
        logger.info(f"  {sp.capitalize()}: top feature "
                    f"= {top_feats[0] if top_feats else 'none'}  "
                    f"|r| = {abs(r_vals.get(top_feats[0], 0)):.4f}" if top_feats else "no valid features")

    # ── Figure: one panel per species ─────────────────────────────────────────
    n_sp   = len([sp for sp in species_order if sp in sobp_results])
    if n_sp == 0:
        logger.warning("  No SOBP data — skipping figure.")
        return sobp_results

    fig, axes = plt.subplots(1, n_sp, figsize=(5.5 * n_sp, 6.5), sharey=False)
    if n_sp == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, sp in zip(axes, [s for s in species_order if s in sobp_results]):
        r_dict = sobp_results[sp]["rank_biserial"]
        valid  = [(c, r_dict[c]) for c in feat_cols
                  if np.isfinite(r_dict.get(c, np.nan))]
        # Top N by |r|
        top = sorted(valid, key=lambda x: abs(x[1]), reverse=True)[:top_n]
        feats  = [c for c, _ in top]
        r_vals = np.array([r for _, r in top])
        colors = [_modality_color(c) for c in feats]
        names  = [c.replace("_", " ") for c in feats]

        y = np.arange(top_n)
        ax.barh(y[::-1], r_vals, color=colors,
                edgecolor="white", linewidth=0.25, height=0.74)
        ax.axvline(0, color="#888888", lw=0.8, ls="--", zorder=4)

        for yi, val in zip(y[::-1], r_vals):
            ax.text(val + (0.005 if val >= 0 else -0.005), yi,
                    f"{val:+.3f}", va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=7, color="#333333")

        ax.set_yticks(y[::-1])
        ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlabel("Rank-biserial r  (pSOBP vs. dSOBP)", labelpad=6)
        ax.set_xlim(-1.1, 1.1)

        pk_p = SOBP_SPECIES_KEYS[sp][0]
        pk_d = SOBP_SPECIES_KEYS[sp][1]
        _style_ax(ax)
        _title_sub(
            ax,
            f"SOBP Effect — {sp.capitalize()}",
            f"{PARTICLE_SHORT[pk_p]} ({PARTICLE_LET[pk_p]} keV/µm) vs. "
            f"{PARTICLE_SHORT[pk_d]} ({PARTICLE_LET[pk_d]} keV/µm)  ·  "
            f"Mann-Whitney r",
        )

    handles = [Patch(facecolor=MODALITY_COLORS[pfx], label=MODALITIES[pfx],
                     edgecolor="none") for pfx in MODALITY_KEYS]
    fig.legend(handles=handles, title="Modality", title_fontsize=8.5,
               fontsize=8, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.95, edgecolor="#CCCCCC")
    fig.tight_layout(rect=[0, 0.06, 1, 1.0])
    _save(fig, out_dir / "sobp_effect_sizes.png")

    # ── JSON export ───────────────────────────────────────────────────────────
    # Flatten for JSON serialisation
    export = {}
    for sp, res in sobp_results.items():
        export[sp] = {
            "pk_proximal":  res["pk_proximal"],
            "pk_distal":    res["pk_distal"],
            "let_proximal": res["let_proximal"],
            "let_distal":   res["let_distal"],
            "top_features_by_r": [
                {"feature": c, "rank_biserial_r": float(res["rank_biserial"][c]),
                 "mannwhitney_p":  float(res["mannwhitney_p"][c])}
                for c in feat_cols
                if np.isfinite(res["rank_biserial"].get(c, np.nan))
            ][:top_n],
        }
    with open(out_dir / "sobp_effect_sizes.json", "w") as fh:
        json.dump(export, fh, indent=2)

    return sobp_results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 9 — CONDITION HIERARCHICAL CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def stage9_condition_clustering(
    means_df: pd.DataFrame,
    feat_cols: List[str],
    out_dir: Path,
) -> None:
    """
    Complete-linkage hierarchical clustering of 49 conditions on standardised
    feature means.  Dendrogram coloured by particle species.
    """
    logger.info("Stage 9: Condition hierarchical clustering")

    rows, cond_labels, cond_pks = [], [], []
    for pk in PARTICLE_KEY_ORDER:
        for o2 in O2_ORDERED:
            sub = means_df[
                (means_df["particle_key"] == pk) & (means_df["o2"] == o2)
            ]
            if len(sub) == 0:
                continue
            rows.append(sub[feat_cols].values[0])
            cond_labels.append(f"{PARTICLE_SHORT[pk]}\n{o2}%")
            cond_pks.append(pk)

    if len(rows) < 2:
        logger.warning("  Fewer than 2 conditions — skipping clustering.")
        return

    X = np.array(rows, dtype=float)
    col_means = np.nanmean(X, axis=0)
    X[~np.isfinite(X)] = np.take(col_means, np.where(~np.isfinite(X))[1])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Z = linkage(Xs, method="complete", metric="euclidean")

    # Assign leaf colors by species
    leaf_order   = leaves_list(Z)
    leaf_species = [PARTICLE_SPECIES[cond_pks[i]] for i in leaf_order]
    leaf_cols    = [SPECIES_COLORS[sp] for sp in leaf_species]

    fig, ax = plt.subplots(figsize=(max(14.0, len(rows) * 0.35), 5.5))
    fig.patch.set_facecolor("white")

    ordered_labels = [cond_labels[i] for i in leaf_order]
    dn = dendrogram(
        Z,
        labels=ordered_labels,
        ax=ax,
        leaf_rotation=75,
        leaf_font_size=6.5,
        color_threshold=0,
        above_threshold_color="#BBBBBB",
    )

    # Colour leaf labels by species
    x_ticks = ax.get_xticklabels()
    for tick, col in zip(x_ticks, leaf_cols):
        tick.set_color(col)
        tick.set_fontweight("bold")

    ax.set_ylabel("Euclidean distance\n(complete linkage, standardised features)",
                  labelpad=6)
    ax.grid(axis="x", visible=False)
    _style_ax(ax)
    _title_sub(
        ax,
        "Condition Hierarchical Clustering",
        f"{len(rows)} conditions  ·  {len(feat_cols)} features (standardised)  ·  "
        "complete linkage  ·  leaf labels coloured by particle species",
    )

    handles = [Patch(facecolor=SPECIES_COLORS[sp],
                     label=sp.capitalize(), edgecolor="none")
               for sp in ["electron", "proton", "helium", "carbon"]]
    ax.legend(handles=handles, title="Species", title_fontsize=8.5,
              fontsize=8, loc="upper right", framealpha=0.95, edgecolor="#CCCCCC")

    fig.tight_layout()
    _save(fig, out_dir / "condition_dendrogram.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Supplementary analyses bridging [2] RF findings to [3] and [4].",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--basedir", type=Path, default=Path("."),
        help="Project root directory (default: current directory).",
    )
    parser.add_argument(
        "--top-n", type=int, default=TOP_N_FEATURES,
        help=f"Number of top features in effect-size / SOBP charts "
             f"(default: {TOP_N_FEATURES}).",
    )
    parser.add_argument(
        "--skip-single-modality-rf", action="store_true",
        help="Skip Stage 3 (single-modality O2 RF). Fastest option for "
             "iterating on other stages. Stages 4 will produce an empty plot.",
    )
    args = parser.parse_args()

    base_dir     = args.basedir.resolve()
    analysis_dir = base_dir / "analysis"
    out_dir      = analysis_dir / "additional"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 64)
    logger.info("06_additional_analyses.py")
    logger.info(f"  Base dir     : {base_dir}")
    logger.info(f"  Output dir   : {out_dir}")
    logger.info(f"  Top-N        : {args.top_n}")
    logger.info(f"  Single-mod RF: "
                f"{'skip' if args.skip_single_modality_rf else 'run'}")
    logger.info("=" * 64)

    # ── Load inputs ───────────────────────────────────────────────────────────
    try:
        df, feat_cols = load_feature_matrix(analysis_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    rf_summary = load_rf_summary(analysis_dir)

    logger.info(f"  Particle configs present : "
                f"{sorted(df['particle_key'].unique())}")
    logger.info(f"  O2 levels present        : "
                f"{sorted(df['o2'].unique(), key=lambda x: O2_ORDERED.index(x) if x in O2_ORDERED else 99)}")
    logger.info(f"  Feature columns          : {len(feat_cols)}")
    logger.info(f"  Total runs               : {len(df)}")

    # ── Run stages ────────────────────────────────────────────────────────────
    logger.info("─" * 64)
    means_df, stats_dict = stage1_condition_stats(df, feat_cols, out_dir)

    logger.info("─" * 64)
    corr_mat = stage2_cross_modality_correlation(means_df, feat_cols, out_dir)

    logger.info("─" * 64)
    acc_matrix = stage3_single_modality_o2(
        df, feat_cols, out_dir, rf_summary,
        skip=args.skip_single_modality_rf,
    )

    logger.info("─" * 64)
    stage4_let_o2_profiles(acc_matrix, out_dir)

    logger.info("─" * 64)
    effect_sizes = stage5_effect_sizes(
        df, feat_cols, out_dir, top_n=args.top_n
    )

    logger.info("─" * 64)
    pca_result = stage6_condition_pca(means_df, feat_cols, out_dir)

    logger.info("─" * 64)
    stage7_o2_gradient_profiles(df, feat_cols, effect_sizes, out_dir)

    logger.info("─" * 64)
    sobp_results = stage8_sobp_effect(
        df, feat_cols, out_dir, top_n=args.top_n
    )

    logger.info("─" * 64)
    stage9_condition_clustering(means_df, feat_cols, out_dir)

    # ── Write meta-summary ────────────────────────────────────────────────────
    logger.info("─" * 64)
    summary = {
        "pipeline_position":   "06",
        "input":               str(analysis_dir / "feature_matrix.csv"),
        "output_dir":          str(out_dir),
        "n_conditions":        int(len(means_df)),
        "n_features":          int(len(feat_cols)),
        "n_runs":              int(len(df)),
        "stage1_conditions":   sorted(means_df["particle_key"].unique().tolist()),
        "stage2_max_off_diag_r": float(
            np.nanmax(np.abs(corr_mat) - np.eye(7) * 10)
        ),
        "stage3_skipped":      bool(args.skip_single_modality_rf),
        "stage3_mean_acc_matrix": acc_matrix.tolist(),
        "stage5_top3_o2_features": effect_sizes["top_o2_features"][:3],
        "stage5_top3_pk_features": effect_sizes["top_pk_features"][:3],
        "stage6_pc1_pc2_var_pct":  pca_result["pc1_pc2_variance"],
        "stage6_n_pcs_90pct_var":  pca_result["cumulative_var_90_pcs"],
        "stage8_species_analysed": list(sobp_results.keys()),
    }
    with open(out_dir / "additional_analyses_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("  additional_analyses_summary.json saved.")

    # ── Final console summary ─────────────────────────────────────────────────
    logger.info("=" * 64)
    logger.info("SUMMARY")
    logger.info("=" * 64)
    logger.info(f"  Conditions analysed      : {len(means_df)}")
    logger.info(f"  Features                 : {len(feat_cols)}")
    logger.info(
        f"  Cross-modality: max |r|  : "
        f"{summary['stage2_max_off_diag_r']:.4f}"
    )
    if not args.skip_single_modality_rf:
        finite = acc_matrix[np.isfinite(acc_matrix)]
        if len(finite):
            logger.info(
                f"  Single-mod O2 acc range  : "
                f"{finite.min():.4f} – {finite.max():.4f}  "
                f"(chance = {CHANCE_O2:.3f})"
            )
    logger.info(
        f"  PCA: PC1+PC2 variance    : "
        f"{pca_result['pc1_pc2_variance']:.1f}%"
    )
    logger.info(
        f"  PCA: PCs for 90% var     : "
        f"{pca_result['cumulative_var_90_pcs']}"
    )
    logger.info(f"  Top η²(O2) feature       : "
                f"{effect_sizes['top_o2_features'][0]}")
    logger.info(f"  Top η²(particle) feature : "
                f"{effect_sizes['top_pk_features'][0]}")
    logger.info(f"\n  Outputs : {out_dir}")
    logger.info(f"  Next    : run 07_regenerate_figures.py")
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
