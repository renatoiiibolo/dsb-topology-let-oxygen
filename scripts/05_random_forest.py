#!/usr/bin/env python3
"""
================================================================================
05_random_forest.py
================================================================================
Random Forest classification of radiation conditions from multimodal DSB
topology features (m1–m7).

PIPELINE POSITION
-----------------
  01  extract_dsb.py
  02  ph_topology_analysis.py    →  analysis/ph/m7_topological_features.json
  03  compute_features.py        →  analysis/features/{prefix}_features.json
  04  build_feature_matrix.py    →  analysis/feature_matrix.csv
  05  random_forest.py           →  analysis/rf/                    ← THIS
  06  additional_analyses.py
  07  regenerate_figures.py

NOTE: Script 00 (parse_sdd_particle_history.py) is no longer part of the
pipeline. It existed solely for the old Event Attribution modality (m7).
With that modality retired, script 00 can be deleted.

STUDY DESIGN
------------
  7 particle configurations × 7 O2 levels × 50 runs = 2,450 rows
  107 features: 97 (m1–m6) + 10 (m7 Topological Summaries)

CLASSIFICATION TASKS
--------------------
  Task 1 — O2 level classification  (7-class, within each particle config)
      Classify 7 O2 levels using features from a single particle config.
      Run separately for each of the 7 particle configurations.
      Chance level = 1/7 ≈ 0.143.
      Primary task: asks whether topology + damage features encode oxygen
      microenvironment independently of beam type.

  Task 2 — Particle / LET classification  (7-class)
      Classify all 7 particle configs across all 2,450 runs.
      Chance level = 1/7 ≈ 0.143.
      Validation task: LET differences should be strongly encoded.

  Task 3 — Joint 49-class classification
      Classify all 49 (particle_key × O2) conditions simultaneously.
      Chance level = 1/49 ≈ 0.020.
      Upper bound on total information in the feature set.

  Task 4 — SOBP position classification  (2-class, within species)
      For proton, helium, and carbon — each of which has both a pSOBP and
      a dSOBP configuration — classify SOBP position using features pooled
      across all 7 O2 levels.  Electron is skipped (mono only).
      Chance level = 0.5.
      Novel task: tests whether within-species LET variation along the SOBP
      is topologically distinguishable. Helium is the most interesting case
      because its pSOBP→dSOBP LET ratio (10→30 keV/µm) places it in the
      transitional topology regime.

METHODOLOGY
-----------
  • RepeatedStratifiedKFold(n_splits=5, n_repeats=10) = 50 folds per task
  • RandomForestClassifier: 500 trees, class_weight="balanced",
    max_features="sqrt", n_jobs=-1
  • Median imputation within fold (fit on train, apply to test; no leakage)
  • Metrics: balanced accuracy, macro-F1, per-class recall
  • Permutation importance aggregated across all folds
  • Modality ablation: leave-one-modality-out, reported as accuracy drop

FRAMING NOTE
------------
  RF classifiers serve as a validation tool to confirm that each modality
  encodes distinguishable information — they are not the primary scientific
  contribution. The primary contribution is the topological characterisation
  (PH, Wasserstein distances, landscape analysis) in 02. See study_description
  for the full framing.

OUTPUTS  (→ analysis/rf/)
-------
  results_summary.json
  confusion_matrices/   task1_o2_{particle_key}_cm.png
                        task2_particle_cm.png
                        task3_joint_cm.png
                        task4_sobp_{species}_cm.png
  feature_importance/   task1_o2_{particle_key}_importance.png
                        task2_particle_importance.png
                        task3_joint_importance.png
                        task4_sobp_{species}_importance.png
  ablation/             modality_ablation.png
                        ablation_results.json
  task1_o2_per_class_recall.png
  task1_o2_balanced_accuracy.png
  task4_sobp_summary.png

USAGE
-----
  python 05_random_forest.py
  python 05_random_forest.py --basedir /path/to/project
  python 05_random_forest.py --skip-ablation
  python 05_random_forest.py --n-splits 5 --n-repeats 10 --n-trees 500

DEPENDENCIES
------------
  numpy, pandas, scikit-learn, matplotlib
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, FancyBboxPatch

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                              confusion_matrix)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — keep in sync with 02, 03, 04                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Particle configurations (LET-ascending) ───────────────────────────────────
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
    "electron_mono":  "Electron mono (0.2 keV/µm)",
    "proton_psobp":   "Proton pSOBP (4.6 keV/µm)",
    "proton_dsobp":   "Proton dSOBP (8.1 keV/µm)",
    "helium_psobp":   "Helium pSOBP (10 keV/µm)",
    "helium_dsobp":   "Helium dSOBP (30 keV/µm)",
    "carbon_psobp":   "Carbon pSOBP (40.9 keV/µm)",
    "carbon_dsobp":   "Carbon dSOBP (70.7 keV/µm)",
}

# Short labels for axes / confusion matrices
PARTICLE_SHORT: Dict[str, str] = {
    "electron_mono":  "e⁻",
    "proton_psobp":   "p⁺ p",
    "proton_dsobp":   "p⁺ d",
    "helium_psobp":   "He p",
    "helium_dsobp":   "He d",
    "carbon_psobp":   "C p",
    "carbon_dsobp":   "C d",
}

# Species with both SOBP positions — eligible for Task 4
SOBP_SPECIES: List[str] = ["proton", "helium", "carbon"]
SOBP_SPECIES_KEYS: Dict[str, Tuple[str, str]] = {
    "proton":  ("proton_psobp",  "proton_dsobp"),
    "helium":  ("helium_psobp",  "helium_dsobp"),
    "carbon":  ("carbon_psobp",  "carbon_dsobp"),
}

# ── O2 levels ─────────────────────────────────────────────────────────────────
O2_ORDERED: List[str] = [
    "21.0", "5.0", "2.1", "0.5", "0.1", "0.021", "0.005"
]
O2_LABELS: Dict[str, str] = {
    "21.0":  "21.0%\n(Norm.)",
    "5.0":   "5.0%\n(T.Norm.)",
    "2.1":   "2.1%\n(Mild)",
    "0.5":   "0.5%",
    "0.1":   "0.1%\n(Severe)",
    "0.021": "0.021%\n(Anoxic)",
    "0.005": "0.005%\n(True Anox.)",
}
O2_LABELS_LONG: Dict[str, str] = {
    "21.0":  "21.0% — Atmospheric normoxia",
    "5.0":   "5.0%  — Tumour normoxia",
    "2.1":   "2.1%  — Mild hypoxia",
    "0.5":   "0.5% — Moderate hypoxia",
    "0.1":   "0.1%  — Severe hypoxia / HIF-1α",
    "0.021": "0.021% — Radiobiological anoxia",
    "0.005": "0.005% — True anoxia",
}

# ── Modality tags ─────────────────────────────────────────────────────────────
MODALITIES: Dict[str, str] = {
    "m1_": "Spatial Distribution",
    "m2_": "Radial Track Structure",
    "m3_": "Local Energy Heterogeneity",
    "m4_": "Dose Distribution",
    "m5_": "Genomic Location",
    "m6_": "Damage Complexity",
    "m7_": "Topological Summaries",
}

# ── Chance levels ─────────────────────────────────────────────────────────────
CHANCE_O2:       float = 1.0 / len(O2_ORDERED)            # 1/7 ≈ 0.143
CHANCE_PARTICLE: float = 1.0 / len(PARTICLE_KEY_ORDER)    # 1/7 ≈ 0.143
CHANCE_JOINT:    float = 1.0 / (
    len(PARTICLE_KEY_ORDER) * len(O2_ORDERED))             # 1/49 ≈ 0.020
CHANCE_SOBP:     float = 0.5                               # 2-class

# ── Expected run count ────────────────────────────────────────────────────────
N_RUNS_EXPECTED: int = 50


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  COLOR PALETTE — "Amalfi Coast at 2 pm in July"                         ║
# ║  Consistent with 02_ph_topology_analysis.py                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PARTICLE_COLORS: Dict[str, str] = {
    "electron_mono":  "#37657E",   # marine deep
    "proton_psobp":   "#F09714",   # lemon-gold
    "proton_dsobp":   "#C97F0E",   # deep amber
    "helium_psobp":   "#6B8C5A",   # maquis sage
    "helium_dsobp":   "#B5956A",   # sun-dried maquis stalks
    "carbon_psobp":   "#CD5F00",   # chili cliff
    "carbon_dsobp":   "#9B5878",   # bougainvillea
}

# 7-stop O2 gradient: deep offshore (normoxic) → seafoam (anoxic)
O2_COLORS: List[str] = [
    "#1D4E63",   # 21.0  — deep offshore
    "#2A6070",   # 5.0   — near offshore
    "#37657E",   # 2.1   — marine
    "#508799",   # 0.5   — moderate hypoxia
    "#6FA3AE",   # 0.1   — shallow piscine
    "#8DC0C9",   # 0.021 — pale seafoam
    "#A8D4E0",   # 0.005 — seafoam
]
O2_COLOR_MAP: Dict[str, str] = dict(zip(O2_ORDERED, O2_COLORS))

# SOBP position colors (matches 02)
SOBP_COLORS: Dict[str, str] = {
    "psobp": "#508799",
    "dsobp": "#1D4E63",
    "mono":  "#A8D4E0",
}

# Modality colors — Amalfi-harmonious
MODALITY_COLORS: Dict[str, str] = {
    "m1_": "#37657E",   # spatial        — marine deep
    "m2_": "#CD5F00",   # radial         — chili cliff
    "m3_": "#C4922A",   # energy         — warm amber
    "m4_": "#1D4E63",   # dose           — deep offshore
    "m5_": "#9B5878",   # genomic        — bougainvillea
    "m6_": "#4A6B3A",   # complexity     — deep maquis
    "m7_": "#6B8C5A",   # topological    — maquis sage
}

# Confusion matrix: limestone → piscine → deep offshore
_CM_CMAP = LinearSegmentedColormap.from_list(
    "amalfi_cm",
    ["#FFFFFF", "#C2B8A3", "#508799", "#1D4E63"],
    N=256,
)

STRIP_FILL: str = "#E8DDD1"   # limestone dust
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


def _strip(ax: plt.Axes, label: str, n_bars: int = 1) -> None:
    """Facet-strip header above an axes panel."""
    ax.add_patch(FancyBboxPatch(
        (0, 1.02), 1, 0.10,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=STRIP_FILL, zorder=5, clip_on=False,
        transform=ax.transAxes,
    ))
    ax.text(0.5, 1.07, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=STRIP_TEXT, zorder=6)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {path.name}")


def _norm_o2(v) -> str:
    """Map float or string to nearest canonical O2 string."""
    try:
        f = float(v)
        for s in O2_ORDERED:
            if abs(f - float(s)) < 1e-9:
                return s
    except (ValueError, TypeError):
        pass
    return str(v)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_feature_matrix(analysis_dir: Path) -> pd.DataFrame:
    """Load analysis/feature_matrix.csv produced by 04_build_feature_matrix.py."""
    path = analysis_dir / "feature_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"feature_matrix.csv not found at {path}\n"
            "  Run 04_build_feature_matrix.py first."
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded feature matrix: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return all columns whose name starts with a known modality tag."""
    return [c for c in df.columns if c[:3] in MODALITIES]


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION CORE
# ══════════════════════════════════════════════════════════════════════════════

def make_rf(n_trees: int = 500, seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_trees,
        class_weight="balanced",
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )


def run_cv(
    X:          np.ndarray,
    y:          np.ndarray,
    labels:     List[str],
    n_splits:   int,
    n_repeats:  int,
    n_trees:    int,
    task_name:  str,
    log_every:  int = 10,
) -> Dict:
    """
    RepeatedStratifiedKFold cross-validation of a Random Forest classifier.

    Within each fold:
      1. Median imputation fit on train, applied to test (no leakage).
      2. RF trained on imputed train set.
      3. Balanced accuracy, macro-F1, row-normalised CM, permutation importance.

    Returns aggregated statistics across all folds.
    """
    rskf        = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )
    n_classes   = len(labels)
    fold_total  = n_splits * n_repeats
    bal_accs: List[float] = []
    f1s:      List[float] = []
    cms:      List[np.ndarray] = []
    imps:     List[np.ndarray] = []

    for fold_idx, (tr, te) in enumerate(rskf.split(X, y)):
        if (fold_idx + 1) % log_every == 0:
            logger.info(f"    {task_name}: fold {fold_idx + 1}/{fold_total}")

        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr])
        Xte = imp.transform(X[te])
        ytr, yte = y[tr], y[te]

        rf  = make_rf(n_trees)
        rf.fit(Xtr, ytr)
        yp  = rf.predict(Xte)

        bal_accs.append(balanced_accuracy_score(yte, yp))
        f1s.append(f1_score(yte, yp, average="macro", zero_division=0))

        cm = confusion_matrix(yte, yp, labels=list(range(n_classes)))
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cms.append(cm.astype(float) / row_sums)

        pi = permutation_importance(
            rf, Xte, yte,
            scoring="balanced_accuracy",
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        imps.append(pi.importances_mean)

    mean_cm = np.mean(cms, axis=0)
    return {
        "bal_acc_mean":        float(np.mean(bal_accs)),
        "bal_acc_std":         float(np.std(bal_accs)),
        "f1_mean":             float(np.mean(f1s)),
        "f1_std":              float(np.std(f1s)),
        "confusion_matrix":    mean_cm.tolist(),
        "per_class_recall":    mean_cm.diagonal().tolist(),
        "feature_importances": np.mean(imps, axis=0).tolist(),
        "all_bal_accs":        [float(v) for v in bal_accs],
    }


def run_ablation(
    X_full:    np.ndarray,
    y:         np.ndarray,
    feat_cols: List[str],
    labels:    List[str],
    baseline:  float,
    n_splits:  int,
    n_repeats: int,
    n_trees:   int,
    task_name: str,
) -> Dict[str, float]:
    """
    Leave-one-modality-out ablation study.
    Returns {modality_name: accuracy_drop} for each modality present in X_full.
    """
    results: Dict[str, float] = {}
    for prefix, name in MODALITIES.items():
        keep = [i for i, c in enumerate(feat_cols) if not c.startswith(prefix)]
        n_dropped = len(feat_cols) - len(keep)
        if n_dropped == 0:
            continue   # modality absent from this feature set
        logger.info(
            f"  Ablation [{task_name}] −{name} ({n_dropped} features removed)"
        )
        res  = run_cv(X_full[:, keep], y, labels,
                      n_splits, n_repeats, n_trees, f"abl_{prefix}")
        drop = baseline - res["bal_acc_mean"]
        results[name] = float(drop)
        logger.info(f"    drop = {drop:+.4f}  "
                    f"(ablated = {res['bal_acc_mean']:.4f})")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    cm:           np.ndarray,
    class_labels: List[str],
    title:        str,
    subtitle:     str,
    out_path:     Path,
    bal_acc:      float,
    f1:           float,
    chance:       float,
    cell_fontsize: float = 8.5,
) -> None:
    """Row-normalised confusion matrix heatmap."""
    n   = len(class_labels)
    fw  = max(5.5, n * 1.15)
    fh  = max(4.8, n * 1.00)
    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor("white")

    im   = ax.imshow(cm, vmin=0, vmax=1, cmap=_CM_CMAP, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
    cbar.set_label("Recall (row-normalised)", fontsize=8, color="#555555")
    cbar.ax.tick_params(labelsize=7.5, colors="#555555")
    cbar.outline.set_edgecolor("#CCCCCC")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_labels, rotation=40, ha="right",
                       fontsize=max(5.5, min(8.5, 72 / n)))
    ax.set_yticklabels(class_labels, fontsize=max(5.5, min(8.5, 72 / n)))
    ax.set_xlabel("Predicted", labelpad=7)
    ax.set_ylabel("True", labelpad=7)
    ax.grid(False)

    # Only annotate cells if the matrix is small enough to be readable
    if n <= 15:
        for i in range(n):
            for j in range(n):
                v = cm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=cell_fontsize, fontweight="bold",
                        color="white" if v > 0.50 else "#1A1A1A")

    _style_ax(ax)
    _title_sub(
        ax, title,
        f"Bal. Acc. = {bal_acc:.3f}   Macro-F1 = {f1:.3f}"
        f"   Chance = {chance:.3f}",
    )
    fig.tight_layout()
    _save(fig, out_path)


def plot_permutation_importance(
    importances: np.ndarray,
    feat_cols:   List[str],
    title:       str,
    subtitle:    str,
    out_path:    Path,
    top_n:       int = 25,
) -> None:
    """Horizontal bar chart of top-N permutation importances, coloured by modality."""
    idx    = np.argsort(importances)[::-1][:top_n]
    vals   = importances[idx]
    names  = [feat_cols[i].replace("_", " ") for i in idx]
    colors = []
    for i in idx:
        c = "#AAAAAA"
        for pfx, col in MODALITY_COLORS.items():
            if feat_cols[i].startswith(pfx):
                c = col
                break
        colors.append(c)

    fig, ax = plt.subplots(figsize=(8.5, max(5.0, top_n * 0.31)))
    fig.patch.set_facecolor("white")

    y_pos = np.arange(top_n)
    ax.barh(y_pos[::-1], vals, color=colors,
            edgecolor="white", linewidth=0.3, height=0.74)
    ax.set_yticks(y_pos[::-1])
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel(
        "Mean permutation importance (Δ balanced accuracy)", labelpad=6
    )

    handles = [
        Patch(facecolor=MODALITY_COLORS[pfx], label=MODALITIES[pfx],
              edgecolor="none")
        for pfx in MODALITY_COLORS
        if any(feat_cols[i].startswith(pfx) for i in idx)
    ]
    leg = ax.legend(handles=handles, title="Modality", title_fontsize=8.5,
                    fontsize=8, loc="lower right",
                    framealpha=0.95, edgecolor="#CCCCCC")
    leg.get_title().set_fontweight("bold")

    _style_ax(ax)
    _title_sub(ax, title, subtitle)
    fig.tight_layout()
    _save(fig, out_path)


def plot_ablation(
    ablation_results: Dict[str, Dict[str, float]],
    task_titles:      Dict[str, str],
    out_path:         Path,
) -> None:
    """Horizontal bar ablation chart, one panel per task."""
    tasks    = list(ablation_results.keys())
    mods     = list(MODALITIES.values())
    n_tasks  = len(tasks)

    fig, axes = plt.subplots(1, n_tasks,
                             figsize=(4.8 * n_tasks, 5.8), sharey=True)
    if n_tasks == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, task in zip(axes, tasks):
        drops    = [ablation_results[task].get(m, np.nan) for m in mods]
        mod_cols = [MODALITY_COLORS[pfx] for pfx in MODALITY_COLORS]
        y_pos    = np.arange(len(mods))

        for y, val, col in zip(y_pos, drops, mod_cols):
            if not np.isfinite(val):
                continue
            ax.barh(y, val, color=col,
                    edgecolor="white", linewidth=0.3, height=0.72)

        ax.axvline(0, color="#888888", lw=0.85, ls="--", zorder=4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(mods, fontsize=8)
        ax.set_xlabel("Accuracy drop (Δ balanced accuracy)", labelpad=6)

        for y, val in zip(y_pos, drops):
            if not np.isfinite(val):
                continue
            offset = 0.001 if val >= 0 else -0.001
            ax.text(val + offset, y, f"{val:+.3f}",
                    va="center", ha="left" if val >= 0 else "right",
                    fontsize=7, color="#333333")

        _style_ax(ax)
        ax.set_title(task_titles.get(task, task), fontsize=10,
                     fontweight="bold", color="#1A1A1A", loc="left")

    fig.text(0.01, 0.975, "Modality Ablation (leave-one-out)",
             fontsize=12, fontweight="bold", color="#1A1A1A", va="top")
    fig.text(0.01, 0.940,
             "Drop = baseline balanced accuracy − accuracy without modality  "
             "·  positive = modality helps",
             fontsize=8.5, color="#666666", style="italic", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, out_path)


def plot_o2_per_class_recall(
    o2_results:  Dict[str, Dict],
    n_splits:    int,
    n_repeats:   int,
    out_path:    Path,
) -> None:
    """
    Faceted bar chart: per-class recall for each particle config (Task 1).
    One panel per particle config (7 panels), 7 bars per panel.
    """
    pkeys = [pk for pk in PARTICLE_KEY_ORDER if pk in o2_results]
    n_p   = len(pkeys)
    fig, axes = plt.subplots(1, n_p, figsize=(3.2 * n_p, 5.0), sharey=True)
    if n_p == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, pk in zip(axes, pkeys):
        res    = o2_results[pk]
        recalls = res["per_class_recall"]
        x       = np.arange(len(O2_ORDERED))

        bars = ax.bar(x, recalls, color=O2_COLORS,
                      edgecolor="white", linewidth=0.4,
                      width=0.62, zorder=3)
        ax.axhline(CHANCE_O2, color="#CD5F00", ls="--", lw=1.1, zorder=4)
        ax.axhline(1.0, color="#CCCCCC", ls=":", lw=0.8, zorder=2)

        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.025, f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="#1A1A1A")

        ax.set_xticks(x)
        ax.set_xticklabels([O2_LABELS[o] for o in O2_ORDERED], fontsize=6.5)
        ax.set_ylim(0, 1.28)
        if ax is axes[0]:
            ax.set_ylabel("Per-class Recall", labelpad=6)
        _style_ax(ax)
        _strip(ax, PARTICLE_SHORT[pk])

    # Shared legend
    handles  = [Patch(facecolor=c, label=O2_LABELS_LONG[o], edgecolor="none")
                for c, o in zip(O2_COLORS, O2_ORDERED)]
    handles += [plt.Line2D([0], [0], color="#CD5F00", ls="--", lw=1.1,
                            label=f"Chance ({CHANCE_O2:.3f})")]
    fig.legend(handles=handles, title="O\u2082 Level",
               title_fontsize=8.5, fontsize=7.5,
               loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.05),
               framealpha=0.95, edgecolor="#CCCCCC")

    fig.text(0.01, 0.975,
             "Task 1 — O\u2082 Classification: Per-class Recall",
             fontsize=12, fontweight="bold", color="#1A1A1A", va="top")
    fig.text(0.01, 0.935,
             f"7-class  ·  {n_splits}-fold × {n_repeats} repeats "
             f"({n_splits * n_repeats} folds)  ·  "
             f"RF 500 trees (balanced)  ·  chance = {CHANCE_O2:.3f}",
             fontsize=8.5, color="#666666", style="italic", va="top")
    fig.tight_layout(rect=[0, 0.08, 1, 0.91])
    _save(fig, out_path)


def plot_o2_balanced_accuracy(
    o2_results: Dict[str, Dict],
    n_splits:   int,
    n_repeats:  int,
    out_path:   Path,
) -> None:
    """
    Grouped bar chart: mean balanced accuracy ± 1 SD per particle config (Task 1).
    """
    pkeys  = [pk for pk in PARTICLE_KEY_ORDER if pk in o2_results]
    means  = [o2_results[pk]["bal_acc_mean"] for pk in pkeys]
    stds   = [o2_results[pk]["bal_acc_std"]  for pk in pkeys]
    colors = [PARTICLE_COLORS[pk] for pk in pkeys]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    fig.patch.set_facecolor("white")

    x    = np.arange(len(pkeys))
    bars = ax.bar(x, means, yerr=stds, color=colors,
                  capsize=5, edgecolor="white", linewidth=0.5,
                  width=0.58,
                  error_kw={"ecolor": "#555555", "lw": 1.2,
                             "capsize": 5, "capthick": 1.2},
                  zorder=3)
    ax.axhline(CHANCE_O2, color="#CD5F00", ls="--", lw=1.1, zorder=4,
               label=f"Chance ({CHANCE_O2:.3f})")

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + s + 0.022, f"{m:.3f}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#1A1A1A")

    ax.set_xticks(x)
    ax.set_xticklabels([PARTICLE_LABELS[pk] for pk in pkeys],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Balanced Accuracy")
    leg = ax.legend(fontsize=8, framealpha=0.95, edgecolor="#CCCCCC")
    _style_ax(ax)
    _title_sub(
        ax,
        "Task 1 — O\u2082 Classification: Balanced Accuracy",
        f"7-class  ·  {n_splits}-fold × {n_repeats} repeats  ·  mean ± 1 SD",
    )
    fig.tight_layout()
    _save(fig, out_path)


def plot_sobp_summary(
    sobp_results: Dict[str, Dict],
    n_splits:     int,
    n_repeats:    int,
    out_path:     Path,
) -> None:
    """
    Grouped bar chart: balanced accuracy ± 1 SD for Task 4 SOBP classification,
    one bar per species.
    """
    species = [sp for sp in SOBP_SPECIES if sp in sobp_results]
    means   = [sobp_results[sp]["bal_acc_mean"] for sp in species]
    stds    = [sobp_results[sp]["bal_acc_std"]  for sp in species]
    # Use psobp color of each species
    colors  = [PARTICLE_COLORS[SOBP_SPECIES_KEYS[sp][0]] for sp in species]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("white")

    x    = np.arange(len(species))
    bars = ax.bar(x, means, yerr=stds, color=colors,
                  capsize=5, edgecolor="white", linewidth=0.5,
                  width=0.52,
                  error_kw={"ecolor": "#555555", "lw": 1.2,
                             "capsize": 5, "capthick": 1.2},
                  zorder=3)
    ax.axhline(CHANCE_SOBP, color="#CD5F00", ls="--", lw=1.1, zorder=4,
               label=f"Chance ({CHANCE_SOBP:.2f})")

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + s + 0.022, f"{m:.3f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#1A1A1A")

    ax.set_xticks(x)
    ax.set_xticklabels([sp.capitalize() for sp in species], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Balanced Accuracy")
    leg = ax.legend(fontsize=8, framealpha=0.95, edgecolor="#CCCCCC")
    _style_ax(ax)
    _title_sub(
        ax,
        "Task 4 — SOBP Position Classification",
        f"2-class (pSOBP vs dSOBP)  ·  {n_splits}-fold × {n_repeats} repeats  "
        f"·  mean ± 1 SD",
    )
    fig.tight_layout()
    _save(fig, out_path)


# ══════════════════════════════════════════════════════════════════════════════
# TASK RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def task1_o2_per_particle(
    df:        pd.DataFrame,
    feat_cols: List[str],
    args:      argparse.Namespace,
    out_dir:   Path,
) -> Dict[str, Dict]:
    """
    Task 1: 7-class O2 classification within each particle configuration.
    One separate RF per particle_key.
    """
    logger.info("=" * 60)
    logger.info("TASK 1 — O2 Classification (7-class, within particle_key)")
    logger.info(f"  Chance level = {CHANCE_O2:.4f}  (1/7)")
    logger.info("=" * 60)

    o2_map  = {o: i for i, o in enumerate(O2_ORDERED)}
    results = {}

    for pk in PARTICLE_KEY_ORDER:
        sub = df[df["particle_key"] == pk].copy()
        if len(sub) == 0:
            logger.warning(f"  No runs for {pk} — skipping.")
            continue
        logger.info(f"  {pk}  ({len(sub)} runs)")

        X      = sub[feat_cols].values.astype(float)
        y      = sub["o2"].map(o2_map).values
        labels = [O2_LABELS[o] for o in O2_ORDERED]

        # Drop O2 levels absent from this particle_key's rows
        present_o2 = sorted(sub["o2"].unique(), key=lambda o: O2_ORDERED.index(o)
                            if o in O2_ORDERED else 999)
        if len(present_o2) < 2:
            logger.warning(f"  Fewer than 2 O2 levels for {pk} — skipping.")
            continue

        res = run_cv(X, y, labels,
                     args.n_splits, args.n_repeats, args.n_trees,
                     f"t1_{pk}")
        results[pk] = res
        logger.info(
            f"    Bal. Acc. = {res['bal_acc_mean']:.4f} ± "
            f"{res['bal_acc_std']:.4f}   "
            f"F1 = {res['f1_mean']:.4f}   "
            f"Δchance = {res['bal_acc_mean'] - CHANCE_O2:+.4f}"
        )

        # Short labels for CM (no newlines)
        lbl_flat = [O2_LABELS_LONG[o] for o in O2_ORDERED]
        plot_confusion_matrix(
            np.array(res["confusion_matrix"]), lbl_flat,
            f"Task 1 — O\u2082 Classification: {PARTICLE_LABELS[pk]}",
            f"Bal. Acc. = {res['bal_acc_mean']:.3f}  ±  "
            f"{res['bal_acc_std']:.3f}",
            out_dir / "confusion_matrices" / f"task1_o2_{pk}_cm.png",
            res["bal_acc_mean"], res["f1_mean"], CHANCE_O2,
        )
        plot_permutation_importance(
            np.array(res["feature_importances"]), feat_cols,
            f"Task 1 Importance — {PARTICLE_LABELS[pk]}",
            "Top 25 features · mean permutation importance across "
            f"{args.n_splits * args.n_repeats} folds",
            out_dir / "feature_importance" / f"task1_o2_{pk}_importance.png",
        )

    plot_o2_per_class_recall(
        results, args.n_splits, args.n_repeats,
        out_dir / "task1_o2_per_class_recall.png",
    )
    plot_o2_balanced_accuracy(
        results, args.n_splits, args.n_repeats,
        out_dir / "task1_o2_balanced_accuracy.png",
    )
    return results


def task2_particle(
    df:        pd.DataFrame,
    feat_cols: List[str],
    args:      argparse.Namespace,
    out_dir:   Path,
) -> Dict:
    """
    Task 2: 7-class particle_key classification across all runs.
    """
    logger.info("=" * 60)
    logger.info("TASK 2 — Particle / LET Classification (7-class)")
    logger.info(f"  Chance level = {CHANCE_PARTICLE:.4f}  (1/7)")
    logger.info("=" * 60)

    pk_map = {pk: i for i, pk in enumerate(PARTICLE_KEY_ORDER)}
    X      = df[feat_cols].values.astype(float)
    y      = df["particle_key"].map(pk_map).values
    labels = [PARTICLE_LABELS[pk] for pk in PARTICLE_KEY_ORDER]

    res = run_cv(X, y, labels,
                 args.n_splits, args.n_repeats, args.n_trees, "t2_particle")
    logger.info(
        f"  Bal. Acc. = {res['bal_acc_mean']:.4f} ± "
        f"{res['bal_acc_std']:.4f}   "
        f"F1 = {res['f1_mean']:.4f}   "
        f"Δchance = {res['bal_acc_mean'] - CHANCE_PARTICLE:+.4f}"
    )

    plot_confusion_matrix(
        np.array(res["confusion_matrix"]), labels,
        "Task 2 — Particle / LET Classification",
        f"7-class  ·  Bal. Acc. = {res['bal_acc_mean']:.3f}  ±  "
        f"{res['bal_acc_std']:.3f}",
        out_dir / "confusion_matrices" / "task2_particle_cm.png",
        res["bal_acc_mean"], res["f1_mean"], CHANCE_PARTICLE,
    )
    plot_permutation_importance(
        np.array(res["feature_importances"]), feat_cols,
        "Task 2 — Feature Importance: Particle Classification",
        "Top 25 features · mean permutation importance across "
        f"{args.n_splits * args.n_repeats} folds",
        out_dir / "feature_importance" / "task2_particle_importance.png",
    )
    return res


def task3_joint(
    df:        pd.DataFrame,
    feat_cols: List[str],
    args:      argparse.Namespace,
    out_dir:   Path,
) -> Dict:
    """
    Task 3: 49-class joint (particle_key × O2) classification.
    """
    logger.info("=" * 60)
    logger.info("TASK 3 — Joint 49-Class Classification")
    logger.info(f"  Chance level = {CHANCE_JOINT:.4f}  (1/49)")
    logger.info("=" * 60)

    conditions  = [(pk, o) for pk in PARTICLE_KEY_ORDER for o in O2_ORDERED]
    cond_map    = {f"{pk}_{o}": i for i, (pk, o) in enumerate(conditions)}
    cond_labels = [f"{PARTICLE_SHORT[pk]} {o}%" for pk, o in conditions]

    df2               = df.copy()
    df2["_condition"] = df2["particle_key"] + "_" + df2["o2"]
    X = df2[feat_cols].values.astype(float)
    y = df2["_condition"].map(cond_map).values

    # Drop any rows whose condition is not in cond_map (shouldn't happen)
    valid = ~np.isnan(y.astype(float))
    X, y  = X[valid], y[valid].astype(int)

    res = run_cv(X, y, cond_labels,
                 args.n_splits, args.n_repeats, args.n_trees, "t3_joint",
                 log_every=5)
    logger.info(
        f"  Bal. Acc. = {res['bal_acc_mean']:.4f} ± "
        f"{res['bal_acc_std']:.4f}   "
        f"F1 = {res['f1_mean']:.4f}   "
        f"Δchance = {res['bal_acc_mean'] - CHANCE_JOINT:+.4f}"
    )

    # 49×49 CM — small cell font, large figure
    plot_confusion_matrix(
        np.array(res["confusion_matrix"]), cond_labels,
        "Task 3 — Joint 49-Class Classification",
        f"Bal. Acc. = {res['bal_acc_mean']:.3f}  ·  "
        f"F1 = {res['f1_mean']:.3f}  ·  "
        f"Chance = {CHANCE_JOINT:.3f}",
        out_dir / "confusion_matrices" / "task3_joint_cm.png",
        res["bal_acc_mean"], res["f1_mean"], CHANCE_JOINT,
        cell_fontsize=3.5,
    )
    plot_permutation_importance(
        np.array(res["feature_importances"]), feat_cols,
        "Task 3 — Feature Importance: Joint 49-Class",
        "Top 25 features · mean permutation importance across "
        f"{args.n_splits * args.n_repeats} folds",
        out_dir / "feature_importance" / "task3_joint_importance.png",
    )
    return res


def task4_sobp(
    df:        pd.DataFrame,
    feat_cols: List[str],
    args:      argparse.Namespace,
    out_dir:   Path,
) -> Dict[str, Dict]:
    """
    Task 4: 2-class SOBP position classification (pSOBP vs dSOBP)
    within each species that has both positions.
    Pooled across all 7 O2 levels; ~700 runs per species.
    """
    logger.info("=" * 60)
    logger.info("TASK 4 — SOBP Position Classification (2-class)")
    logger.info(f"  Species: {SOBP_SPECIES}")
    logger.info(f"  Chance level = {CHANCE_SOBP:.4f}")
    logger.info("=" * 60)

    sobp_map = {"psobp": 0, "dsobp": 1}
    results: Dict[str, Dict] = {}

    for sp in SOBP_SPECIES:
        pk_p, pk_d = SOBP_SPECIES_KEYS[sp]
        sub = df[df["particle"].str.lower() == sp].copy()
        # Keep only psobp and dsobp rows (exclude mono if any)
        sub = sub[sub["sobp"].isin(["psobp", "dsobp"])]

        if len(sub) == 0:
            logger.warning(f"  No runs for species {sp} — skipping.")
            continue
        n_p = (sub["sobp"] == "psobp").sum()
        n_d = (sub["sobp"] == "dsobp").sum()
        logger.info(f"  {sp.capitalize()}: pSOBP={n_p}, dSOBP={n_d}")

        X      = sub[feat_cols].values.astype(float)
        y      = sub["sobp"].map(sobp_map).values
        labels = [
            PARTICLE_LABELS.get(pk_p, f"{sp} pSOBP"),
            PARTICLE_LABELS.get(pk_d, f"{sp} dSOBP"),
        ]

        res = run_cv(X, y, labels,
                     args.n_splits, args.n_repeats, args.n_trees,
                     f"t4_{sp}")
        results[sp] = res
        logger.info(
            f"    Bal. Acc. = {res['bal_acc_mean']:.4f} ± "
            f"{res['bal_acc_std']:.4f}   "
            f"F1 = {res['f1_mean']:.4f}   "
            f"Δchance = {res['bal_acc_mean'] - CHANCE_SOBP:+.4f}"
        )

        plot_confusion_matrix(
            np.array(res["confusion_matrix"]), labels,
            f"Task 4 — SOBP Position: {sp.capitalize()}",
            f"2-class (pSOBP vs dSOBP)  ·  "
            f"Bal. Acc. = {res['bal_acc_mean']:.3f}  ±  "
            f"{res['bal_acc_std']:.3f}",
            out_dir / "confusion_matrices" / f"task4_sobp_{sp}_cm.png",
            res["bal_acc_mean"], res["f1_mean"], CHANCE_SOBP,
        )
        plot_permutation_importance(
            np.array(res["feature_importances"]), feat_cols,
            f"Task 4 — Feature Importance: {sp.capitalize()} SOBP",
            "Top 25 features · mean permutation importance across "
            f"{args.n_splits * args.n_repeats} folds",
            out_dir / "feature_importance" / f"task4_sobp_{sp}_importance.png",
        )

    if results:
        plot_sobp_summary(
            results, args.n_splits, args.n_repeats,
            out_dir / "task4_sobp_summary.png",
        )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RF classification of DSB topology features (4 tasks).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--basedir", type=Path, default=Path("."),
        help="Project root directory (default: current directory).",
    )
    parser.add_argument(
        "--n-splits", type=int, default=5,
        help="CV splits (default: 5).",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=10,
        help="CV repeats (default: 10).",
    )
    parser.add_argument(
        "--n-trees", type=int, default=500,
        help="RF trees (default: 500).",
    )
    parser.add_argument(
        "--skip-ablation", action="store_true",
        help="Skip modality ablation study (saves ~70%% of runtime).",
    )
    parser.add_argument(
        "--skip-task4", action="store_true",
        help="Skip Task 4 (SOBP position classification).",
    )
    args = parser.parse_args()

    base_dir     = args.basedir.resolve()
    analysis_dir = base_dir / "analysis"
    out_dir      = analysis_dir / "rf"

    for sub in ["confusion_matrices", "feature_importance", "ablation"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 62)
    logger.info("05_random_forest.py")
    logger.info(f"  Base dir     : {base_dir}")
    logger.info(f"  Output dir   : {out_dir}")
    logger.info(f"  CV           : {args.n_splits}-fold × {args.n_repeats} "
                f"= {args.n_splits * args.n_repeats} folds per task")
    logger.info(f"  RF trees     : {args.n_trees}")
    logger.info(f"  Ablation     : {'skip' if args.skip_ablation else 'yes'}")
    logger.info(f"  Task 4 SOBP  : {'skip' if args.skip_task4 else 'yes'}")
    logger.info("=" * 62)

    # ── Load feature matrix ───────────────────────────────────────────────
    try:
        df = load_feature_matrix(analysis_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    # Normalise o2 values to canonical strings
    df["o2"] = df["o2"].apply(_norm_o2)

    # Ensure particle_key column present (required for Tasks 1, 2, 3, 4)
    if "particle_key" not in df.columns:
        logger.error(
            '"particle_key" column missing from feature_matrix.csv.\n'
            "  Re-run 03_compute_features.py and 04_build_feature_matrix.py."
        )
        return 1

    feat_cols = get_feature_cols(df)
    if not feat_cols:
        logger.error("No feature columns (m1_–m7_) found in feature matrix.")
        return 1

    n_total = len(df)
    logger.info(f"Feature columns  : {len(feat_cols)}")
    logger.info(f"Runs loaded      : {n_total}")
    logger.info(f"Particle configs : "
                f"{sorted(df['particle_key'].unique())}")
    logger.info(f"O2 levels        : {sorted(df['o2'].unique())}")

    # ── Run all tasks ─────────────────────────────────────────────────────
    t1_results = task1_o2_per_particle(df, feat_cols, args, out_dir)
    t2_results = task2_particle(df, feat_cols, args, out_dir)
    t3_results = task3_joint(df, feat_cols, args, out_dir)
    t4_results = task4_sobp(df, feat_cols, args, out_dir) \
        if not args.skip_task4 else {}

    # ── Modality ablation ─────────────────────────────────────────────────
    ablation_all: Dict[str, Dict[str, float]] = {}
    if not args.skip_ablation:
        logger.info("=" * 62)
        logger.info("MODALITY ABLATION")
        logger.info("=" * 62)
        o2_map = {o: i for i, o in enumerate(O2_ORDERED)}
        pk_map = {pk: i for i, pk in enumerate(PARTICLE_KEY_ORDER)}

        # One ablation per particle_key for Task 1
        for pk in PARTICLE_KEY_ORDER:
            if pk not in t1_results:
                continue
            sub = df[df["particle_key"] == pk]
            X   = sub[feat_cols].values.astype(float)
            y   = sub["o2"].map(o2_map).values
            ablation_all[f"t1_{pk}"] = run_ablation(
                X, y, feat_cols,
                [O2_LABELS[o] for o in O2_ORDERED],
                t1_results[pk]["bal_acc_mean"],
                args.n_splits, args.n_repeats, args.n_trees,
                f"t1_{pk}",
            )

        # Task 2 ablation
        X = df[feat_cols].values.astype(float)
        y = df["particle_key"].map(pk_map).values
        ablation_all["t2_particle"] = run_ablation(
            X, y, feat_cols,
            [PARTICLE_LABELS[pk] for pk in PARTICLE_KEY_ORDER],
            t2_results["bal_acc_mean"],
            args.n_splits, args.n_repeats, args.n_trees,
            "t2_particle",
        )

        # Build task title dict for the ablation plot
        task_titles = {
            f"t1_{pk}": f"O\u2082 / {PARTICLE_SHORT[pk]}"
            for pk in PARTICLE_KEY_ORDER if f"t1_{pk}" in ablation_all
        }
        task_titles["t2_particle"] = "Particle LET"

        plot_ablation(
            ablation_all, task_titles,
            out_dir / "ablation" / "modality_ablation.png",
        )
        with open(out_dir / "ablation" / "ablation_results.json", "w") as fh:
            json.dump(ablation_all, fh, indent=2)
        logger.info("  Saved: ablation_results.json")

    # ── Save results_summary.json ─────────────────────────────────────────
    def _fmt_t1(r: Dict) -> Dict:
        return {
            "bal_acc_mean":         r["bal_acc_mean"],
            "bal_acc_std":          r["bal_acc_std"],
            "f1_mean":              r["f1_mean"],
            "f1_std":               r["f1_std"],
            "per_class_recall_o2":  dict(zip(O2_ORDERED,
                                              r["per_class_recall"])),
            "confusion_matrix":     r["confusion_matrix"],
            "feature_importances":  r["feature_importances"],
        }

    def _fmt_t2(r: Dict) -> Dict:
        return {
            "bal_acc_mean":              r["bal_acc_mean"],
            "bal_acc_std":               r["bal_acc_std"],
            "f1_mean":                   r["f1_mean"],
            "f1_std":                    r["f1_std"],
            "per_class_recall_particle": dict(zip(PARTICLE_KEY_ORDER,
                                                   r["per_class_recall"])),
            "confusion_matrix":          r["confusion_matrix"],
            "feature_importances":       r["feature_importances"],
        }

    def _fmt_t4(r: Dict, sp: str) -> Dict:
        pk_p, pk_d = SOBP_SPECIES_KEYS[sp]
        return {
            "bal_acc_mean":    r["bal_acc_mean"],
            "bal_acc_std":     r["bal_acc_std"],
            "f1_mean":         r["f1_mean"],
            "f1_std":          r["f1_std"],
            "per_class_recall": {
                "psobp": r["per_class_recall"][0],
                "dsobp": r["per_class_recall"][1],
            },
            "confusion_matrix":    r["confusion_matrix"],
            "feature_importances": r["feature_importances"],
        }

    summary = {
        "cv_config": {
            "n_splits":  args.n_splits,
            "n_repeats": args.n_repeats,
            "n_folds":   args.n_splits * args.n_repeats,
            "n_trees":   args.n_trees,
        },
        "chance_levels": {
            "task1_o2":       CHANCE_O2,
            "task2_particle": CHANCE_PARTICLE,
            "task3_joint":    CHANCE_JOINT,
            "task4_sobp":     CHANCE_SOBP,
        },
        "task1_o2_per_particle":  {pk: _fmt_t1(r)
                                   for pk, r in t1_results.items()},
        "task2_particle":         _fmt_t2(t2_results),
        "task3_joint_49class": {
            "bal_acc_mean":        t3_results["bal_acc_mean"],
            "bal_acc_std":         t3_results["bal_acc_std"],
            "f1_mean":             t3_results["f1_mean"],
            "f1_std":              t3_results["f1_std"],
            "confusion_matrix":    t3_results["confusion_matrix"],
            "feature_importances": t3_results["feature_importances"],
        },
        "task4_sobp_position":    {sp: _fmt_t4(r, sp)
                                   for sp, r in t4_results.items()},
        "ablation":               ablation_all,
    }

    with open(out_dir / "results_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("results_summary.json saved.")

    # ── Final console summary ─────────────────────────────────────────────
    logger.info("=" * 62)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 62)

    logger.info(f"Task 1 — O\u2082 Classification "
                f"(7-class, chance = {CHANCE_O2:.3f}):")
    for pk, r in t1_results.items():
        logger.info(
            f"  {PARTICLE_SHORT[pk]:6s}: "
            f"{r['bal_acc_mean']:.4f} ± {r['bal_acc_std']:.4f}  "
            f"(Δchance = {r['bal_acc_mean'] - CHANCE_O2:+.4f})"
        )

    logger.info(
        f"\nTask 2 — Particle LET "
        f"(7-class, chance = {CHANCE_PARTICLE:.3f}):\n"
        f"  {t2_results['bal_acc_mean']:.4f} ± "
        f"{t2_results['bal_acc_std']:.4f}  "
        f"(Δchance = {t2_results['bal_acc_mean'] - CHANCE_PARTICLE:+.4f})"
    )

    logger.info(
        f"\nTask 3 — Joint 49-class "
        f"(chance = {CHANCE_JOINT:.4f}):\n"
        f"  {t3_results['bal_acc_mean']:.4f} ± "
        f"{t3_results['bal_acc_std']:.4f}  "
        f"(Δchance = {t3_results['bal_acc_mean'] - CHANCE_JOINT:+.4f})"
    )

    if t4_results:
        logger.info(
            f"\nTask 4 — SOBP Position "
            f"(2-class, chance = {CHANCE_SOBP:.2f}):"
        )
        for sp, r in t4_results.items():
            logger.info(
                f"  {sp.capitalize():8s}: "
                f"{r['bal_acc_mean']:.4f} ± {r['bal_acc_std']:.4f}  "
                f"(Δchance = {r['bal_acc_mean'] - CHANCE_SOBP:+.4f})"
            )

    logger.info(f"\n  Outputs : {out_dir}")
    logger.info(f"  Next    : run 06_additional_analyses.py")
    logger.info("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
