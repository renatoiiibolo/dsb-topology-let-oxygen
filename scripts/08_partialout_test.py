#!/usr/bin/env python3
"""
================================================================================
08_partialout_test.py
================================================================================
Partial-out test: does m7 (Topological Summaries) carry oxygen information
independent of DSB count, or does its signal collapse when count-proportional
features are removed?

PIPELINE POSITION
-----------------
  01  extract_dsb.py
  02  ph_topology_analysis.py    →  analysis/ph/
  03  compute_features.py        →  analysis/features/
  04  build_feature_matrix.py    →  analysis/feature_matrix.csv
  05  random_forest.py           →  analysis/rf/
  06  additional_analyses.py     →  analysis/additional/
  07  regenerate_figures.py      →  analysis/figures/
  08  partialout_test.py         →  analysis/partialout/   ← THIS

SCIENTIFIC QUESTION
-------------------
Project [2] identifies m7 (Topological Summaries) as the dominant
oxygen-information modality. However, m1_n_dsbs has η²_O₂ = 0.924 —
the highest effect size in the 107-feature set — and is effectively a
direct readout of VOxA's calibrated P_fix([O₂]) curve. If m7's oxygen
signal is primarily driven by DSB count (i.e., higher count → more complex
topology), then removing the count-proportional signal should cause m7 to
collapse toward chance. If m7 carries structurally distinct information
(spatial organisation of retained DSBs beyond count), it should survive.

The test is conducted on three complementary levels:
  (A) η² partial-out — regress m1_n_dsbs out of every m7 feature and
      recompute η²_O₂ on the residuals. Did the O₂ effect size survive
      count removal?
  (B) RF classification under exclusion conditions — train per-particle 7-class
      O₂ classifiers under five feature-set conditions:
        0. Full 107 features (baseline)
        1. Minus m1_n_dsbs (106 features)
        2. Minus entire m1 modality (74 features)
        3. m7 only, raw (10 features)
        4. m7 only, residualised — each m7 feature regressed against
           m1_n_dsbs; use residuals as features (10 features)
      The key comparison is condition 3 vs condition 4: does m7's O₂
      accuracy survive partial-out of count?
  (C) Cross-condition η² change plot — for each m7 feature, compare raw η²_O₂
      vs residualised η²_O₂. Points on the diagonal survive; points below
      collapse.

INTERPRETATION GUIDE
--------------------
  • If m7 BA (condition 4) ≈ m7 BA (condition 3):
        m7 carries oxygen information orthogonal to DSB count.
        The topological organisation of DSBs encodes O₂ independently.
        Claim is defensible.

  • If m7 BA (condition 4) collapses toward chance:
        m7's oxygen signal is largely mediated by DSB count.
        The "first nuclear-scale PH analysis" claim is methodologically
        sound; the "novel oxygen signal" claim needs to be qualified as
        "consistent with VOxA kinetics, not topologically independent."

  • Intermediate result:
        Some m7 features survive (e.g. h0_betti0_var, h1_landscape_integral),
        others do not (e.g. h0_persistent_entropy). This would support a
        nuanced claim: topology partly adds new signal, partly echoes count.

OUTPUTS  (→ analysis/partialout/)
-------
  eta2_partialout.json              Stage A — raw vs. residualised η² for all features
  rf_exclusion_results.json         Stage B — BA per particle × condition
  fig_eta2_scatter.png              Stage C1 — η² before/after scatter, m7 highlighted
  fig_eta2_m7_bars.png              Stage C2 — per-feature η² bars (raw vs. residualised)
  fig_rf_ba_comparison.png          Stage B1 — BA per particle across 5 conditions
  fig_rf_ba_heatmap.png             Stage B2 — 7-particle × 5-condition BA heatmap
  partialout_summary.json           Meta-summary

USAGE
-----
  python 08_partialout_test.py
  python 08_partialout_test.py --basedir /path/to/project
  python 08_partialout_test.py --top-n 20

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
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, FancyBboxPatch
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
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
# ║  CONFIGURATION — keep in sync with 06, 07                               ║
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

O2_ORDERED: List[str] = [
    "21.0", "5.0", "2.1", "0.5", "0.1", "0.021", "0.005"
]

MODALITY_PREFIX: Dict[str, str] = {
    "m1_": "Spatial Distribution",
    "m2_": "Radial Track Structure",
    "m3_": "Local Energy Heterogeneity",
    "m4_": "Dose Distribution",
    "m5_": "Genomic Location",
    "m6_": "Damage Complexity",
    "m7_": "Topological Summaries",
}

# The count feature being partialled out
COUNT_FEATURE: str = "m1_n_dsbs"

# RF parameters — match 06
# Match 05_random_forest.py exactly
RF_N_TREES:   int   = 500
RF_N_SPLITS:  int   = 5
RF_N_REPEATS: int   = 10
CHANCE_O2:    float = 1.0 / len(O2_ORDERED)   # 1/7 ≈ 0.143

# Condition labels for the five exclusion conditions
CONDITION_LABELS: List[str] = [
    "Full (107)",
    "−n_dsbs (106)",
    "−m1 (74)",
    "m7 raw (10)",
    "m7 resid. (10)",
]
CONDITION_KEYS: List[str] = [
    "full",
    "minus_ndsbs",
    "minus_m1",
    "m7_raw",
    "m7_residualised",
]
N_CONDITIONS: int = len(CONDITION_KEYS)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  COLOUR PALETTE — Amalfi Coast (consistent with 07)                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_HN_DIR = Path.home() / ".local" / "share" / "fonts" / "HelveticaNeue"
if _HN_DIR.exists():
    for _ttf in sorted(_HN_DIR.glob("*.ttf")):
        fm.fontManager.addfont(str(_ttf))

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

MODALITY_COLORS: Dict[str, str] = {
    "Spatial Distribution":       "#37657E",
    "Radial Track Structure":     "#508799",
    "Local Energy Heterogeneity": "#6B8C5A",
    "Dose Distribution":          "#1D4E63",
    "Genomic Location":           "#C2A387",
    "Damage Complexity":          "#F09714",
    "Topological Summaries":      "#CD5F00",
}

# Five exclusion-condition colours — light to dark marine/chili arc
CONDITION_COLORS: List[str] = [
    "#1D4E63",   # Full        — deep offshore
    "#508799",   # −n_dsbs     — piscine
    "#6B8C5A",   # −m1         — maquis sage
    "#CD5F00",   # m7 raw      — chili cliff
    "#9B5878",   # m7 residual — bougainvillea
]

COLOR_CHANCE  = "#D4845A"   # melon flesh
STRIP_FILL    = "#E8DDD1"
STRIP_TEXT    = "#1A1A1A"

# Heatmap: seafoam → chili → deep marine
_BA_CMAP = LinearSegmentedColormap.from_list(
    "amalfi_ba",
    ["#A8D4E0", "#C2A387", "#CD5F00", "#1D4E63"],
    N=256,
)

# Scatter cmap for non-m7 features
_ETA_GRAY = "#CCCCCC"

# ── Global style ──────────────────────────────────────────────────────────────
_FONT_SANS = ["Helvetica Neue", "Helvetica", "Arial",
              "Liberation Sans", "DejaVu Sans"]
DPI = 600

plt.rcParams.update({
    "font.family":           "sans-serif",
    "font.sans-serif":       _FONT_SANS,
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
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "xtick.major.size":      3.5,
    "ytick.major.size":      3.5,
    "xtick.major.width":     0.7,
    "ytick.major.width":     0.7,
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
    "figure.dpi":            150,
    "savefig.dpi":           DPI,
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
                 color="#1A1A1A", loc="left", pad=16)
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes,
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
            ha="center", va="center", fontsize=9, fontweight="bold",
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

    # Select feature columns by modality prefix — identical to 05_random_forest.py.
    # This is robust to any number of extra metadata columns (prefix, particle,
    # sobp, let, dir_name, is_normoxic, condition_id, etc.) in the CSV.
    MODALITY_PREFIXES = {"m1_", "m2_", "m3_", "m4_", "m5_", "m6_", "m7_"}
    feat_cols = [c for c in df.columns if c[:3] in MODALITY_PREFIXES]
    logger.info(f"  Loaded feature matrix: {len(df)} rows x {len(feat_cols)} features")
    logger.info(f"  Meta columns present : "
                f"{[c for c in df.columns if c[:3] not in MODALITY_PREFIXES]}")
    return df, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# RESIDUALISATION UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def residualise(X: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """
    For each column of X, regress out `covariate` (1-D) via OLS and return
    the residuals. Shapes: X is (n_samples, n_features), covariate is (n_samples,).
    Each column is independently residualised.
    """
    cov = covariate.reshape(-1, 1)
    # Fit a single intercept+slope model for each feature
    reg = LinearRegression(fit_intercept=True)
    residuals = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        y = X[:, j]
        reg.fit(cov, y)
        residuals[:, j] = y - reg.predict(cov)
    return residuals


# ══════════════════════════════════════════════════════════════════════════════
# STAGE A — ETA² PARTIAL-OUT
# ══════════════════════════════════════════════════════════════════════════════

def _eta2_oneway(values: np.ndarray, groups: np.ndarray) -> float:
    """
    One-way ANOVA η² = SS_between / SS_total.
    Returns NaN if fewer than 2 groups or SS_total == 0.
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return float("nan")
    grand_mean = np.mean(values)
    ss_total = np.sum((values - grand_mean) ** 2)
    if ss_total < 1e-15:
        return float("nan")
    ss_between = sum(
        np.sum(groups == g) * (np.mean(values[groups == g]) - grand_mean) ** 2
        for g in unique_groups
    )
    return float(np.clip(ss_between / ss_total, 0.0, 1.0))


def stage_A_eta2_partialout(
    df: pd.DataFrame,
    feat_cols: List[str],
    out_dir: Path,
) -> Dict:
    """
    Compute η²_O₂ and η²_particle for every feature, both raw and after
    residualising against m1_n_dsbs. Save to eta2_partialout.json.
    Returns the results dict.
    """
    logger.info("Stage A — η² partial-out")

    if COUNT_FEATURE not in df.columns:
        logger.error(f"  Count feature '{COUNT_FEATURE}' not found in matrix.")
        return {}

    o2_labels   = df["o2"].values
    pk_labels   = df["particle_key"].values
    covariate   = df[COUNT_FEATURE].values.astype(float)

    results: Dict = {}

    for feat in feat_cols:
        vals_raw = df[feat].values.astype(float)
        mask = np.isfinite(vals_raw) & np.isfinite(covariate)
        if mask.sum() < 20:
            continue

        # Raw η²
        eta2_o2_raw  = _eta2_oneway(vals_raw[mask], o2_labels[mask])
        eta2_pk_raw  = _eta2_oneway(vals_raw[mask], pk_labels[mask])

        # Residualised η²
        resid = residualise(
            vals_raw[mask].reshape(-1, 1),
            covariate[mask],
        ).ravel()
        eta2_o2_resid = _eta2_oneway(resid, o2_labels[mask])
        eta2_pk_resid = _eta2_oneway(resid, pk_labels[mask])

        results[feat] = {
            "modality":       _modality_of(feat),
            "eta2_o2_raw":    round(eta2_o2_raw,  4),
            "eta2_pk_raw":    round(eta2_pk_raw,  4),
            "eta2_o2_resid":  round(eta2_o2_resid, 4),
            "eta2_pk_resid":  round(eta2_pk_resid, 4),
            "delta_o2":       round(eta2_o2_resid - eta2_o2_raw, 4),
            "survival_ratio": round(
                eta2_o2_resid / eta2_o2_raw
                if eta2_o2_raw > 0.01 else float("nan"), 4
            ),
        }

    with open(out_dir / "eta2_partialout.json", "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info(f"  eta2_partialout.json: {len(results)} features")

    # Log m7 summary
    m7_feats = [f for f in results if f.startswith("m7_")]
    logger.info("  m7 η²_O₂ raw vs. residualised:")
    for f in sorted(m7_feats):
        r = results[f]
        logger.info(
            f"    {f:<40s}  raw={r['eta2_o2_raw']:.4f}  "
            f"resid={r['eta2_o2_resid']:.4f}  "
            f"survival={r['survival_ratio']}"
        )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE B — RF EXCLUSION CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════

def _build_feature_sets(
    df: pd.DataFrame,
    feat_cols: List[str],
) -> Dict[str, Tuple[List[str], Optional[np.ndarray]]]:
    """
    Build the five (feature_name_list, residualised_array_or_None) pairs.
    For conditions 0–3, residualised_array is None — use df[cols] directly.
    For condition 4, residualised_array is the residualised m7 matrix.
    """
    m1_cols = [c for c in feat_cols if c.startswith("m1_")]
    m7_cols = [c for c in feat_cols if c.startswith("m7_")]

    sets: Dict[str, Tuple[List[str], Optional[np.ndarray]]] = {}

    # 0. Full
    sets["full"] = (feat_cols, None)

    # 1. Minus m1_n_dsbs
    minus_ndsbs = [c for c in feat_cols if c != COUNT_FEATURE]
    sets["minus_ndsbs"] = (minus_ndsbs, None)

    # 2. Minus entire m1 modality
    minus_m1 = [c for c in feat_cols if not c.startswith("m1_")]
    sets["minus_m1"] = (minus_m1, None)

    # 3. m7 raw
    sets["m7_raw"] = (m7_cols, None)

    # 4. m7 residualised — regress m1_n_dsbs out of each m7 feature
    covariate = df[COUNT_FEATURE].values.astype(float)
    X_m7 = df[m7_cols].values.astype(float)
    X_m7_resid = residualise(X_m7, covariate)
    sets["m7_residualised"] = (m7_cols, X_m7_resid)

    return sets


def _run_rf_condition(
    df_particle: pd.DataFrame,
    feat_cols: List[str],
    resid_array: Optional[np.ndarray],
    df_full: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Train a 7-class O₂ RF on df_particle rows with a given feature set.
    Returns (mean_ba, std_ba) over RepeatedStratifiedKFold.

    Imputation: within-fold median (fit on train split, applied to test
    split without refitting), matching the protocol of 05_random_forest.py
    exactly and preventing any leakage of test-set statistics.

    resid_array: if not None, use this pre-computed residual matrix instead
    of reading from df_particle; rows must align with df_particle.
    """
    y = np.array([O2_ORDERED.index(o) for o in df_particle["o2"].values])

    if resid_array is not None:
        # Align residual rows to this particle's rows in df_full
        X = resid_array  # already subsetted to this particle's rows in stage_B
    else:
        X = df_particle[feat_cols].values.astype(float)

    rskf = RepeatedStratifiedKFold(
        n_splits=RF_N_SPLITS, n_repeats=RF_N_REPEATS, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=RF_N_TREES,
        class_weight="balanced",   # match 05_random_forest.py
        random_state=42,
        n_jobs=-1,
    )

    bas: List[float] = []
    for train_idx, test_idx in rskf.split(X, y):
        # Within-fold median imputation: fit on train, apply to test (no leakage)
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[train_idx])
        Xte = imp.transform(X[test_idx])
        rf.fit(Xtr, y[train_idx])
        y_pred = rf.predict(Xte)
        bas.append(balanced_accuracy_score(y[test_idx], y_pred))

    return float(np.mean(bas)), float(np.std(bas))


def stage_B_rf_exclusion(
    df: pd.DataFrame,
    feat_cols: List[str],
    out_dir: Path,
) -> Dict:
    """
    For each of the 7 particle configurations × 5 exclusion conditions,
    train a 7-class O₂ RF classifier and record balanced accuracy.
    """
    logger.info("Stage B — RF exclusion conditions")
    logger.info(
        f"  RF: {RF_N_TREES} trees, "
        f"{RF_N_SPLITS}-fold × {RF_N_REPEATS}-repeat "
        f"= {RF_N_SPLITS * RF_N_REPEATS} folds per model"
    )
    logger.info(
        f"  Total models: "
        f"{len(PARTICLE_KEY_ORDER)} particles × {N_CONDITIONS} conditions "
        f"= {len(PARTICLE_KEY_ORDER) * N_CONDITIONS}"
    )

    # Build feature sets once (residualised array is global)
    feat_sets = _build_feature_sets(df, feat_cols)

    results: Dict = {}

    for pk in PARTICLE_KEY_ORDER:
        sub = df[df["particle_key"] == pk].copy().reset_index(drop=False)
        # Store original index mapping for residual alignment
        orig_idx = sub["index"].values
        sub = sub.set_index("index")

        results[pk] = {}
        logger.info(f"  Particle: {pk}  ({len(sub)} runs)")

        for ckey, clabel in zip(CONDITION_KEYS, CONDITION_LABELS):
            feat_list, resid_arr = feat_sets[ckey]

            if resid_arr is not None:
                # For residualised condition: extract this particle's rows
                resid_sub = resid_arr[orig_idx]
                ba_mean, ba_std = _run_rf_condition(
                    sub, feat_list, resid_sub, df
                )
            else:
                ba_mean, ba_std = _run_rf_condition(
                    sub, feat_list, None, df
                )

            results[pk][ckey] = {
                "label":    clabel,
                "n_features": len(feat_list),
                "ba_mean":  round(ba_mean, 4),
                "ba_std":   round(ba_std,  4),
            }
            logger.info(
                f"    {clabel:<20s}  n_feat={len(feat_list):3d}  "
                f"BA = {ba_mean:.4f} ± {ba_std:.4f}"
            )

    with open(out_dir / "rf_exclusion_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("  rf_exclusion_results.json saved.")

    # ── Print summary table ───────────────────────────────────────────────────
    logger.info("")
    logger.info("  BA summary table  (mean ± std):")
    header = f"  {'Particle':<20s}" + "".join(
        f"  {lbl:<22s}" for lbl in CONDITION_LABELS
    )
    logger.info(header)
    logger.info("  " + "─" * (20 + 24 * N_CONDITIONS))
    for pk in PARTICLE_KEY_ORDER:
        row = f"  {PARTICLE_SHORT[pk]:<20s}"
        for ckey in CONDITION_KEYS:
            r = results[pk][ckey]
            row += f"  {r['ba_mean']:.4f} ± {r['ba_std']:.4f}    "
        logger.info(row)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE C — FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_eta2_scatter(
    eta2_results: Dict,
    out_dir: Path,
) -> None:
    """
    Stage C1 — Scatter plot of η²_O₂ raw (x) vs. residualised (y).
    Each point is a feature. m7 features highlighted in chili cliff.
    All other features shown in light grey.
    Diagonal y = x drawn as reference.
    """
    logger.info("  Figure: fig_eta2_scatter.png")

    feats_all  = list(eta2_results.keys())
    x_raw      = np.array([eta2_results[f]["eta2_o2_raw"]   for f in feats_all])
    y_resid    = np.array([eta2_results[f]["eta2_o2_resid"] for f in feats_all])
    is_m7      = np.array([f.startswith("m7_") for f in feats_all])
    is_count   = np.array([f == COUNT_FEATURE    for f in feats_all])

    # Axis limits: driven by max of *plotted* data, not n_dsbs (which is
    # excluded from the scatter since its residualised y = NaN).
    valid_x = x_raw[np.isfinite(x_raw) & ~is_count]
    valid_y = y_resid[np.isfinite(y_resid)]
    lim_max = max(valid_x.max(), valid_y.max()) * 1.08

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    fig.patch.set_facecolor("white")

    # Non-m7, non-count background
    mask_bg = ~is_m7 & ~is_count
    valid_bg = mask_bg & np.isfinite(x_raw) & np.isfinite(y_resid)
    ax.scatter(x_raw[valid_bg], y_resid[valid_bg],
               c=_ETA_GRAY, s=20, alpha=0.5, linewidths=0,
               label="Other features (n=97)", zorder=2)

    # m7 features — chili cliff, larger; numbered 1–10 to avoid label overlap
    m7_x = x_raw[is_m7]
    m7_y = y_resid[is_m7]
    m7_names = [f for f, flag in zip(feats_all, is_m7) if flag]
    m7_short  = [f.replace("m7_", "") for f in m7_names]

    ax.scatter(m7_x, m7_y,
               c=MODALITY_COLORS["Topological Summaries"],
               s=70, alpha=0.92, linewidths=0.6,
               edgecolors="#1A1A1A",
               label="m7 Topological Summaries (n=10)", zorder=4)

    # Number labels on each m7 point (1-character, no overlap)
    for idx, (xi, yi) in enumerate(zip(m7_x, m7_y)):
        ax.text(xi, yi, str(idx + 1),
                ha="center", va="center",
                fontsize=5.5, color="white", fontweight="bold", zorder=5)

    # COUNT_FEATURE: mark at its raw x value, y=0 (residualised against itself → 0)
    if is_count.any():
        cx = x_raw[is_count][0]
        ax.scatter([cx], [0.0],
                   marker="D", c="#D4845A", s=90, linewidths=0.7,
                   edgecolors="#1A1A1A", zorder=5,
                   label=f"{COUNT_FEATURE}")
        ax.annotate(COUNT_FEATURE.replace("m1_", ""),
                    (cx, 0.0), xytext=(-6, 8), textcoords="offset points",
                    fontsize=6.5, color="#D4845A", ha="right")

    # Diagonal y = x
    ax.plot([0, lim_max], [0, lim_max], "--",
            color="#BBBBBB", lw=1.0, zorder=1, label="y = x (no change)")

    ax.set_xlim(-0.01, lim_max)
    ax.set_ylim(-0.005, lim_max * 0.08)   # y range tight: max resid ≈ 0.027
    ax.set_xlabel("η²(O₂) — raw feature")
    ax.set_ylabel("η²(O₂) — after partialling out m1_n_dsbs")

    _title_sub(
        ax,
        "η²(O₂): Raw vs. Residualised",
        f"Residualised = regress {COUNT_FEATURE} out of each feature via OLS  ·  "
        f"numbers = m7 features (key below)",
    )
    _style_ax(ax)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.95)

    # Reference table for numbered m7 features (two-column text box)
    n_cols = 2
    n_rows = int(np.ceil(len(m7_short) / n_cols))
    lines = []
    for row in range(n_rows):
        parts = []
        for col in range(n_cols):
            i = col * n_rows + row
            if i < len(m7_short):
                parts.append(f"{i+1}: {m7_short[i]}")
        lines.append("    ".join(parts))
    ref_text = "\n".join(lines)
    ax.text(0.98, 0.02, ref_text,
            transform=ax.transAxes, fontsize=6.5,
            va="bottom", ha="right",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5EFE7",
                      edgecolor="#CCCCCC", alpha=0.95))

    fig.tight_layout()
    _savefig(fig, out_dir / "fig_eta2_scatter.png")


def plot_eta2_m7_bars(
    eta2_results: Dict,
    out_dir: Path,
) -> None:
    """
    Stage C2 — Grouped bar chart for the 10 m7 features:
    raw η²_O₂ vs. residualised η²_O₂, side by side.
    Survival ratio annotated above each pair.
    """
    logger.info("  Figure: fig_eta2_m7_bars.png")

    m7_feats   = sorted([f for f in eta2_results if f.startswith("m7_")])
    short_names = [f.replace("m7_", "") for f in m7_feats]
    raw_vals    = [eta2_results[f]["eta2_o2_raw"]   for f in m7_feats]
    resid_vals  = [eta2_results[f]["eta2_o2_resid"] for f in m7_feats]
    surv_ratios = [eta2_results[f]["survival_ratio"] for f in m7_feats]

    n = len(m7_feats)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    fig.patch.set_facecolor("white")

    col_raw   = MODALITY_COLORS["Topological Summaries"]  # chili cliff
    col_resid = "#9B5878"                                   # bougainvillea

    bars_raw   = ax.bar(x - width / 2, raw_vals,   width, color=col_raw,
                        alpha=0.88, label="Raw η²(O₂)", zorder=3)
    bars_resid = ax.bar(x + width / 2, resid_vals, width, color=col_resid,
                        alpha=0.88, label="Residualised η²(O₂)", zorder=3)

    # Annotate survival ratio above each pair
    for i, (rv, sv) in enumerate(zip(resid_vals, surv_ratios)):
        if sv is not None and not (isinstance(sv, float) and np.isnan(sv)):
            ax.text(
                x[i], max(raw_vals[i], rv) + 0.015,
                f"{sv:.2f}×",
                ha="center", va="bottom", fontsize=7, color="#444444",
            )

    # Chance reference
    ax.axhline(CHANCE_O2, color=COLOR_CHANCE, lw=1.0, linestyle="--",
               label=f"Chance (1/7 = {CHANCE_O2:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("η²(O₂)")
    ax.set_ylim(0, max(raw_vals) * 1.25)

    _title_sub(
        ax,
        "m7 Topological Summaries — η²(O₂) Before and After Partial-Out",
        f"Survival ratio (×) = residualised / raw  ·  "
        f"covariate removed: {COUNT_FEATURE}",
    )
    _style_ax(ax)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _savefig(fig, out_dir / "fig_eta2_m7_bars.png")


def plot_rf_ba_comparison(
    rf_results: Dict,
    out_dir: Path,
) -> None:
    """
    Stage B1 — Line plot of O₂ BA vs. LET for each of the 5 conditions.
    One line per condition, x-axis ordered by PARTICLE_KEY_ORDER.
    """
    logger.info("  Figure: fig_rf_ba_comparison.png")

    n_particles = len(PARTICLE_KEY_ORDER)
    x = np.arange(n_particles)
    x_labels = [PARTICLE_SHORT[pk] for pk in PARTICLE_KEY_ORDER]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    fig.patch.set_facecolor("white")

    for ci, (ckey, clabel) in enumerate(zip(CONDITION_KEYS, CONDITION_LABELS)):
        ba_vals  = [rf_results[pk][ckey]["ba_mean"] for pk in PARTICLE_KEY_ORDER]
        ba_stds  = [rf_results[pk][ckey]["ba_std"]  for pk in PARTICLE_KEY_ORDER]
        col      = CONDITION_COLORS[ci]

        # m7_residualised gets special styling
        lw     = 2.0 if "m7" in ckey else 1.5
        ls     = "-"  if "m7" in ckey else ("-" if ci < 3 else "--")
        zorder = 5    if "m7" in ckey else 3
        marker = "o"  if "m7" in ckey else "s"
        ms     = 7    if "m7" in ckey else 5

        ax.plot(x, ba_vals, color=col, lw=lw, ls=ls, marker=marker,
                markersize=ms, label=clabel, zorder=zorder)
        ax.fill_between(
            x,
            [v - s for v, s in zip(ba_vals, ba_stds)],
            [v + s for v, s in zip(ba_vals, ba_stds)],
            color=col, alpha=0.12, zorder=zorder - 1,
        )

    ax.axhline(CHANCE_O2, color=COLOR_CHANCE, lw=1.0, linestyle="--",
               zorder=2, label=f"Chance (1/7 ≈ {CHANCE_O2:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Balanced accuracy (O₂ 7-class)")
    ax.set_ylim(0, 0.75)

    # Annotate LET on secondary x axis info
    ax.set_xlabel("Particle configuration  (increasing LET $\\rightarrow$)")
    _title_sub(
        ax,
        "O₂ Classification BA: Five Exclusion Conditions",
        "Key comparison: m7 raw vs. m7 residualised (n_dsbs regressed out)  ·  "
        "shaded bands = ±1 SD across folds",
    )
    _style_ax(ax)
    ax.legend(loc="upper right", fontsize=7.5, ncol=1, framealpha=0.95)
    fig.tight_layout()
    _savefig(fig, out_dir / "fig_rf_ba_comparison.png")


def plot_rf_ba_heatmap(
    rf_results: Dict,
    out_dir: Path,
) -> None:
    """
    Stage B2 — Heatmap of BA values: rows = conditions, columns = particles.
    Annotated with BA values. Chance level contoured.
    """
    logger.info("  Figure: fig_rf_ba_heatmap.png")

    ba_matrix = np.array([
        [rf_results[pk][ckey]["ba_mean"] for pk in PARTICLE_KEY_ORDER]
        for ckey in CONDITION_KEYS
    ])

    col_labels = [PARTICLE_SHORT[pk] for pk in PARTICLE_KEY_ORDER]
    row_labels = CONDITION_LABELS

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    fig.patch.set_facecolor("white")

    im = ax.imshow(ba_matrix, cmap=_BA_CMAP, aspect="auto",
                   vmin=0.0, vmax=max(0.6, ba_matrix.max() * 1.05))
    ax.grid(False)  # suppress rcParams global grid over imshow cells

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True)

    # Annotate cells
    for ri in range(ba_matrix.shape[0]):
        for ci in range(ba_matrix.shape[1]):
            val = ba_matrix[ri, ci]
            text_col = "white" if val > 0.4 else "#1A1A1A"
            ax.text(ci, ri, f"{val:.3f}", ha="center", va="center",
                    fontsize=7.5, color=text_col, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Balanced accuracy", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    # Highlight m7 rows
    for ri, ckey in enumerate(CONDITION_KEYS):
        if "m7" in ckey:
            ax.add_patch(plt.Rectangle(
                (-0.5, ri - 0.5), len(col_labels), 1,
                fill=False, edgecolor=MODALITY_COLORS["Topological Summaries"],
                lw=2.0, zorder=5,
            ))

    ax.set_xlabel("Particle configuration (increasing LET $\\rightarrow$)")
    ax.set_ylabel("Feature-set condition")

    _title_sub(
        ax,
        "O₂ Classification BA — Feature Exclusion Heatmap",
        "Boxed rows = m7-only conditions  ·  key: row 4 (m7 resid.) vs. row 3 (m7 raw)",
    )
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    _savefig(fig, out_dir / "fig_rf_ba_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Partial-out test: does m7 O₂ signal survive removal of "
            "DSB count-proportional variance (m1_n_dsbs)?"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--basedir", type=Path, default=Path("."),
        help="Project root directory (default: current directory).",
    )
    args = parser.parse_args()

    base_dir     = args.basedir.resolve()
    analysis_dir = base_dir / "analysis"
    out_dir      = analysis_dir / "partialout"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 64)
    logger.info("08_partialout_test.py")
    logger.info(f"  Base dir     : {base_dir}")
    logger.info(f"  Output dir   : {out_dir}")
    logger.info(f"  Count feature: {COUNT_FEATURE}")
    logger.info(f"  Conditions   : {CONDITION_LABELS}")
    logger.info("=" * 64)

    # ── Load inputs ───────────────────────────────────────────────────────────
    try:
        df, feat_cols = load_feature_matrix(analysis_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    logger.info(f"  Particle configs : {sorted(df['particle_key'].unique())}")
    logger.info(f"  O₂ levels        : "
                f"{sorted(df['o2'].unique(), key=lambda v: O2_ORDERED.index(v) if v in O2_ORDERED else 99)}")
    logger.info(f"  Feature columns  : {len(feat_cols)}")
    logger.info(f"  Total runs       : {len(df)}")

    # ── Stage A: η² partial-out ───────────────────────────────────────────────
    logger.info("─" * 64)
    eta2_results = stage_A_eta2_partialout(df, feat_cols, out_dir)

    # ── Stage B: RF exclusion conditions ─────────────────────────────────────
    logger.info("─" * 64)
    rf_results = stage_B_rf_exclusion(df, feat_cols, out_dir)

    # ── Stage C: Figures ──────────────────────────────────────────────────────
    logger.info("─" * 64)
    logger.info("Stage C — Figures")
    if eta2_results:
        plot_eta2_scatter(eta2_results, out_dir)
        plot_eta2_m7_bars(eta2_results, out_dir)
    if rf_results:
        plot_rf_ba_comparison(rf_results, out_dir)
        plot_rf_ba_heatmap(rf_results, out_dir)

    # ── Meta-summary ──────────────────────────────────────────────────────────
    logger.info("─" * 64)

    # Key verdict: m7_raw BA vs. m7_resid BA, mean across particles
    if rf_results:
        m7_raw_bas   = [rf_results[pk]["m7_raw"]["ba_mean"]         for pk in PARTICLE_KEY_ORDER]
        m7_resid_bas = [rf_results[pk]["m7_residualised"]["ba_mean"] for pk in PARTICLE_KEY_ORDER]
        mean_raw     = float(np.mean(m7_raw_bas))
        mean_resid   = float(np.mean(m7_resid_bas))
        survival_ba  = mean_resid / mean_raw if mean_raw > 0 else float("nan")
    else:
        mean_raw = mean_resid = survival_ba = float("nan")

    # m7 η² survival summary
    if eta2_results:
        m7_feats        = [f for f in eta2_results if f.startswith("m7_")]
        mean_raw_eta2   = float(np.nanmean([eta2_results[f]["eta2_o2_raw"]   for f in m7_feats]))
        mean_resid_eta2 = float(np.nanmean([eta2_results[f]["eta2_o2_resid"] for f in m7_feats]))
        eta2_survival   = mean_resid_eta2 / mean_raw_eta2 if mean_raw_eta2 > 0 else float("nan")
    else:
        mean_raw_eta2 = mean_resid_eta2 = eta2_survival = float("nan")

    summary = {
        "pipeline_position":           "08",
        "count_feature_removed":       COUNT_FEATURE,
        "conditions":                  CONDITION_LABELS,
        "m7_mean_ba_raw":              round(mean_raw,   4),
        "m7_mean_ba_residualised":     round(mean_resid, 4),
        "m7_ba_survival_ratio":        round(survival_ba, 4),
        "m7_mean_eta2_o2_raw":         round(mean_raw_eta2,   4),
        "m7_mean_eta2_o2_residualised":round(mean_resid_eta2, 4),
        "m7_eta2_survival_ratio":      round(eta2_survival,  4),
        "interpretation": (
            "m7 signal largely SURVIVES count removal (ratio > 0.7): "
            "topological organisation encodes O₂ beyond DSB count."
            if survival_ba > 0.7 else
            "m7 signal COLLAPSES after count removal (ratio < 0.5): "
            "m7 O₂ encoding is primarily mediated by DSB count."
            if survival_ba < 0.5 else
            "m7 signal PARTIALLY SURVIVES (0.5 ≤ ratio ≤ 0.7): "
            "some features are count-independent, others are not."
        ),
        "per_particle_ba": {
            pk: {
                ckey: rf_results[pk][ckey]["ba_mean"]
                for ckey in CONDITION_KEYS
            }
            for pk in PARTICLE_KEY_ORDER
        } if rf_results else {},
    }

    with open(out_dir / "partialout_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("  partialout_summary.json saved.")

    # ── Final console summary ─────────────────────────────────────────────────
    logger.info("=" * 64)
    logger.info("SUMMARY")
    logger.info("=" * 64)
    logger.info(f"  Covariate removed        : {COUNT_FEATURE}")
    logger.info(f"  m7 BA (raw, mean)        : {mean_raw:.4f}")
    logger.info(f"  m7 BA (residualised, mean): {mean_resid:.4f}")
    logger.info(f"  m7 BA survival ratio      : {survival_ba:.4f}")
    logger.info(f"  m7 η²_O₂ (raw, mean)     : {mean_raw_eta2:.4f}")
    logger.info(f"  m7 η²_O₂ (resid, mean)   : {mean_resid_eta2:.4f}")
    logger.info(f"  m7 η² survival ratio      : {eta2_survival:.4f}")
    logger.info(f"  Verdict: {summary['interpretation']}")
    logger.info(f"\n  Outputs : {out_dir}")
    logger.info(f"  Next    : check partialout_summary.json for verdict.")
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())