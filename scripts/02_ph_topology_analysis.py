#!/usr/bin/env python3
"""
================================================================================
02_ph_topology_analysis.py
================================================================================
Persistent Homology (PH) topology analysis of DSB point clouds.

PIPELINE POSITION
-----------------
  01  extract_dsb.py                 →  *_dsb_complexity.csv per run  ← INPUT
  02  ph_topology_analysis.py        →  diagrams, Wasserstein matrices,
                                        m7_topological_features.json  ← THIS
  03  compute_features.py            →  analysis/features/*.json (m1–m6)
  04  build_feature_matrix.py        →  feature_matrix.csv (m1–m7)
  05  random_forest.py
  06  additional_analyses.py
  07  regenerate_figures.py

NOTE: Script 00 (parse_sdd_particle_history.py) is no longer part of the
pipeline. It existed solely to produce event_summary CSVs for the old Event
Attribution modality (m7). With that modality retired and replaced by
Topological Summaries computed here, script 00 serves no purpose and can be
deleted.

DIRECTORY LAYOUT
----------------
Scripts live at the same level as the particle directories. The full directory
name (particle + LET) propagates through ALL subdirectory levels and file
prefixes — the LET tag appears at every level.

  project_root/
  ├── 02_ph_topology_analysis.py
  ├── carbon_40.9/                       ← {dir_name}
  │   └── carbon_40.9_21.0/              ← {dir_name}_{o2}
  │       └── carbon_40.9_21.0_01/       ← {dir_name}_{o2}_{run_id}
  │           └── carbon_40.9_21.0_01_dsb_complexity.csv
  ├── carbon_70.7/                       ← carbon dSOBP
  ├── electron_0.2/
  ├── helium_10.0/                         ← helium pSOBP
  ├── helium_30.0/                         ← helium dSOBP
  ├── proton_4.6/                       ← proton pSOBP
  └── proton_8.1/                       ← proton dSOBP

ADDING NEW RUNS
---------------
  Update "dir_name" and "let" in PARTICLE_CONFIGS below.
  Update O2_ORDERED if you add new oxygen levels.
  Nothing else changes.

WHAT THIS SCRIPT COMPUTES
--------------------------
  Stage 1  Vietoris-Rips persistence diagrams (ripser, H0 + H1).
           Cached per run as .npz; reused on re-runs unless --overwrite.

  Stage 2  Pairwise Wasserstein-2 distance matrices (persim).
           Parallelised by row; checkpointed every CHECKPOINT_ROWS rows.
           Full corpus: up to 2,450 × 2,450 (7 configs × 7 O2 × 50 runs).

  Stage 3  m7 Topological Summary features — 10 per run:
             H0/H1 landscape integrals, H0/H1 landscape peak locations,
             H0/H1 persistent entropy, β₀ mean/var, β₁ mean/var.
           Saved to m7_topological_features.json for script 04 to merge.

  Stage 4  UMAP embeddings (metric = "precomputed").
           Three panels: by particle/SOBP, by O₂ level, by SOBP position.

  Stage 5  Within / between condition comparison.
           Violin plot + N_cond × N_cond condition-mean heatmap (49 × 49).

  Stage 6  Persistence landscapes.
           Mean λ₁(t) per condition, one panel per particle species.
           pSOBP = solid line, dSOBP = dashed.

ALL SCALAR RESULTS + EMBEDDINGS saved to ph_summary.json so that
07_regenerate_figures.py can rebuild every figure without recomputation.

OUTPUTS  (→ analysis/ph/)
-------
  diagrams/                  {prefix}_diagram.npz       per run
  wasserstein_H0.npy         wasserstein_H1.npy         full pairwise matrices
  m7_topological_features.json                          per-run m7 features
  run_index.json                                        ordered run metadata
  ph_summary.json                                       all results + embeddings
  [figures — preview DPI only; 07 regenerates at 600 DPI]

USAGE
-----
  python 02_ph_topology_analysis.py
  python 02_ph_topology_analysis.py --basedir /path/to/project
  python 02_ph_topology_analysis.py --workers 8
  python 02_ph_topology_analysis.py --overwrite
  python 02_ph_topology_analysis.py --skip-figures
  python 02_ph_topology_analysis.py --skip-landscapes

DEPENDENCIES
------------
  numpy  pandas  ripser  persim  umap-learn  matplotlib  joblib
  pip install ripser persim umap-learn joblib
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
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from ripser import ripser as _ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

try:
    from persim import wasserstein as _persim_wass
    HAS_PERSIM = True
except ImportError:
    HAS_PERSIM = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — edit here when adding new simulation runs              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── O2 levels ─────────────────────────────────────────────────────────────────
# Seven levels spanning the full OER curve:
#   Plateau       : 21.0, 5.0
#   Transition    : 2.1, 0.5
#   Hypoxic flank : 0.1, 0.021, 0.005
O2_ORDERED: List[str] = [
    "21.0", "5.0", "2.1", "0.5", "0.1", "0.021", "0.005"
]
O2_LABELS: Dict[str, str] = {
    "21.0":  "21.0% — Atmospheric normoxia",
    "5.0":   "5.0%  — Tumour normoxia",
    "2.1":   "2.1%  — Mild hypoxia",
    "0.5":   "0.5%  — Moderate hypoxia",
    "0.1":   "0.1%  — Severe hypoxia / HIF-1α threshold",
    "0.021": "0.021% — Radiobiological anoxia",
    "0.005": "0.005% — True anoxia (maximum OER)",
}

# ── Runs per condition ────────────────────────────────────────────────────────
N_RUNS: int = 50

# ── Ripser / filtration ───────────────────────────────────────────────────────
PH_MAX_DIST: float = 9.3   # µm — nuclear diameter upper bound
PH_MIN_PTS:  int   = 3     # skip runs with fewer DSBs than this

# ── Persistence landscape parameters ─────────────────────────────────────────
LANDSCAPE_N_T:   int = 200   # filtration grid resolution
LANDSCAPE_K_MAX: int = 3     # number of landscape layers

# ── Betti curve resolution (for m7 features) ─────────────────────────────────
BETTI_N_T: int = 200

# ── Wasserstein checkpointing ─────────────────────────────────────────────────
CHECKPOINT_ROWS: int = 200   # save partial matrix every this many rows

# ── Particle / SOBP configurations ───────────────────────────────────────────
# "dir_name" must exactly match the top-level directory next to the scripts.
# LET values are from TOPAS-nBio simulation metadata.
PARTICLE_CONFIGS: List[Dict] = [
    {
        "key":      "electron_mono",
        "dir_name": "electron_0.2",
        "particle": "electron",
        "sobp":     "mono",
        "let":      0.2,
        "label":    "Electron mono (0.2 keV/µm)",
    },
    {
        "key":      "proton_psobp",
        "dir_name": "proton_4.6",
        "particle": "proton",
        "sobp":     "psobp",
        "let":      4.6,
        "label":    "Proton pSOBP (4.6 keV/µm)",
    },
    {
        "key":      "proton_dsobp",
        "dir_name": "proton_8.1",
        "particle": "proton",
        "sobp":     "dsobp",
        "let":      8.1,
        "label":    "Proton dSOBP (8.1 keV/µm)",
    },
    {
        "key":      "helium_psobp",
        "dir_name": "helium_10.0",
        "particle": "helium",
        "sobp":     "psobp",
        "let":      10.0,
        "label":    "Helium pSOBP (10 keV/µm)",
    },
    {
        "key":      "helium_dsobp",
        "dir_name": "helium_30.0",
        "particle": "helium",
        "sobp":     "dsobp",
        "let":      30.0,
        "label":    "Helium dSOBP (30 keV/µm)",
    },
    {
        "key":      "carbon_psobp",
        "dir_name": "carbon_40.9",
        "particle": "carbon",
        "sobp":     "psobp",
        "let":      40.9,
        "label":    "Carbon pSOBP (40.9 keV/µm)",
    },
    {
        "key":      "carbon_dsobp",
        "dir_name": "carbon_70.7",
        "particle": "carbon",
        "sobp":     "dsobp",
        "let":      70.7,
        "label":    "Carbon dSOBP (70.7 keV/µm)",
    },
]

# ── Derived lookups — do not edit ─────────────────────────────────────────────
PARTICLE_KEY_ORDER: List[str]       = [c["key"]      for c in PARTICLE_CONFIGS]
PCONF_BY_KEY:       Dict[str, Dict]  = {c["key"]:      c for c in PARTICLE_CONFIGS}
PCONF_BY_DIRNAME:   Dict[str, str]   = {c["dir_name"]: c["key"]
                                         for c in PARTICLE_CONFIGS}

# 49 conditions: 7 particle configs × 7 O2 levels
CONDITIONS: List[Tuple[str, str]] = [
    (pk, o) for pk in PARTICLE_KEY_ORDER for o in O2_ORDERED
]
N_CONDITIONS: int = len(CONDITIONS)    # 49

# Compact tick labels for the 49 × 49 heatmap
CONDITION_LABELS: List[str] = [
    (
        f"{PCONF_BY_KEY[pk]['particle'][:2].upper()}"
        f"{'p' if PCONF_BY_KEY[pk]['sobp'] == 'psobp' else 'd' if PCONF_BY_KEY[pk]['sobp'] == 'dsobp' else 'm'}"
        f"\n{o}%"
    )
    for pk, o in CONDITIONS
]


# ── Color palette — "Amalfi Coast at 2 pm in July" ───────────────────────────
# Particle colors: each species has a pSOBP (brighter) / dSOBP (deeper) pair.
PARTICLE_COLORS: Dict[str, str] = {
    "electron_mono":  "#37657E",   # marine deep
    "proton_psobp":   "#F09714",   # lemon-gold
    "proton_dsobp":   "#C97F0E",   # deep amber
    "helium_psobp":   "#6B8C5A",   # maquis sage
    "helium_dsobp":   "#4A6B3A",   # deep maquis
    "carbon_psobp":   "#CD5F00",   # chili cliff
    "carbon_dsobp":   "#9B5878",   # bougainvillea
}

# SOBP-position colors and markers
SOBP_COLORS: Dict[str, str] = {
    "psobp": "#508799",
    "dsobp": "#1D4E63",
    "mono":  "#A8D4E0",
}
SOBP_MARKERS: Dict[str, str] = {
    "psobp": "o",
    "dsobp": "^",
    "mono":  "s",
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

# Heatmap: white → sable → chili cliff → deep marine
_HEAT_CMAP = LinearSegmentedColormap.from_list(
    "amalfi_heat",
    ["#FFFFFF", "#C2A387", "#CD5F00", "#1D4E63"],
    N=256,
)

STRIP_FILL: str = "#E8DDD1"
STRIP_TEXT: str = "#1A1A1A"

# Preview DPI — 07_regenerate_figures.py writes final 600 DPI outputs.
_PREVIEW_DPI: int = 150

# ── Global matplotlib style ───────────────────────────────────────────────────
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
    "figure.dpi":            _PREVIEW_DPI,
    "savefig.dpi":           _PREVIEW_DPI,
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
    fig.savefig(path, dpi=_PREVIEW_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# O2 NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _norm_o2(v) -> str:
    """Map a float or string O2 value to the nearest canonical O2_ORDERED key."""
    try:
        f = float(v)
        for s in O2_ORDERED:
            if abs(f - float(s)) < 1e-9:
                return s
    except (ValueError, TypeError):
        pass
    return str(v)


# ══════════════════════════════════════════════════════════════════════════════
# RUN DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def discover_runs(base_dir: Path) -> List[Dict]:
    """
    Locate all per-run DSB complexity files.

    The {dir_name} (e.g. "carbon_40.9") propagates through every path level:

        <base_dir>/
        ├── carbon_40.9/                  ← {dir_name}         matches PARTICLE_CONFIGS
        │   └── carbon_40.9_21.0/         ← {dir_name}_{o2}
        │       └── carbon_40.9_21.0_01/  ← {dir_name}_{o2}_{run_id}
        │           └── carbon_40.9_21.0_01_dsb_complexity.csv
        └── helium_10.0/ ...

    Directories not matching any "dir_name" in PARTICLE_CONFIGS are silently
    skipped, so scripts, analysis/, and any other folders are ignored.
    """
    runs: List[Dict] = []

    for pdir in sorted(base_dir.iterdir()):
        if not pdir.is_dir():
            continue
        dname = pdir.name
        if dname not in PCONF_BY_DIRNAME:
            continue

        canon_key = PCONF_BY_DIRNAME[dname]
        pconf     = PCONF_BY_KEY[canon_key]

        for o2_raw in O2_ORDERED:
            o2_norm = _norm_o2(o2_raw)

            # Level 2: {dir_name}_{o2}   e.g. carbon_40.9_21.0
            o2_dir = pdir / f"{dname}_{o2_raw}"
            if not o2_dir.exists():
                continue

            for run_num in range(1, N_RUNS + 1):
                run_id = f"{run_num:02d}"

                # Level 3 prefix: {dir_name}_{o2}_{run_id}  e.g. carbon_40.9_21.0_01
                prefix   = f"{dname}_{o2_raw}_{run_id}"
                run_dir  = o2_dir / prefix
                dsb_file = run_dir / f"{prefix}_dsb_complexity.csv"

                if not dsb_file.exists():
                    continue

                try:
                    cond_idx = CONDITIONS.index((canon_key, o2_norm))
                except ValueError:
                    logger.warning(
                        f"Condition ({canon_key}, {o2_norm}) not in CONDITIONS "
                        f"— skipping {prefix}."
                    )
                    continue

                runs.append({
                    "particle_key": canon_key,
                    "particle":     pconf["particle"],
                    "sobp":         pconf["sobp"],
                    "let":          pconf["let"],
                    "dir_name":     dname,
                    "o2":           o2_norm,
                    "run_id":       run_id,
                    "prefix":       prefix,
                    "dsb_file":     str(dsb_file),
                    "condition":    f"{canon_key}_{o2_norm}",
                    "cond_idx":     cond_idx,
                })

    runs.sort(key=lambda r: (
        PARTICLE_KEY_ORDER.index(r["particle_key"]),
        O2_ORDERED.index(r["o2"]),
        r["run_id"],
    ))
    found_keys = sorted(set(r["particle_key"] for r in runs))
    logger.info(
        f"Discovered {len(runs)} runs across "
        f"{len(found_keys)} particle configuration(s): {found_keys}"
    )
    return runs


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — PERSISTENCE DIAGRAMS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_diagram(run_info: Dict) -> Optional[Dict]:
    """Compute H0 + H1 Vietoris-Rips persistence diagram for one run."""
    if not HAS_RIPSER:
        return None
    try:
        df     = pd.read_csv(run_info["dsb_file"])
        coords = df[["x_um", "y_um", "z_um"]].dropna().values.astype(float)
        if len(coords) < PH_MIN_PTS:
            return {
                "prefix": run_info["prefix"],
                "h0": np.empty((0, 2)),
                "h1": np.empty((0, 2)),
            }
        res  = _ripser(coords, maxdim=1, thresh=PH_MAX_DIST)
        dgms = res["dgms"]
        h0   = dgms[0]; h0 = h0[h0[:, 1] < np.inf]
        h1   = dgms[1]; h1 = h1[h1[:, 1] < np.inf]
        return {"prefix": run_info["prefix"], "h0": h0, "h1": h1}
    except Exception as exc:
        logger.warning(f"  Diagram failed {run_info['prefix']}: {exc}")
        return None


def stage1_diagrams(
    runs:      List[Dict],
    out_dir:   Path,
    workers:   int,
    overwrite: bool,
) -> Dict[str, Dict]:
    """Compute (or load cached) H0 + H1 diagrams for all runs."""
    diag_dir = out_dir / "diagrams"
    diag_dir.mkdir(parents=True, exist_ok=True)

    diagrams:   Dict[str, Dict] = {}
    to_compute: List[Dict]      = []

    for r in runs:
        npz = diag_dir / f"{r['prefix']}_diagram.npz"
        if npz.exists() and not overwrite:
            d = np.load(npz, allow_pickle=True)
            diagrams[r["prefix"]] = {"h0": d["h0"], "h1": d["h1"]}
        else:
            to_compute.append(r)

    if to_compute:
        logger.info(
            f"Stage 1: computing {len(to_compute)} diagrams "
            f"({len(diagrams)} cached)."
        )

        def _save_result(res: Optional[Dict]) -> None:
            if res:
                p = res["prefix"]
                diagrams[p] = {"h0": res["h0"], "h1": res["h1"]}
                np.savez(diag_dir / f"{p}_diagram.npz",
                         h0=res["h0"], h1=res["h1"])

        if workers > 1 and HAS_JOBLIB:
            results = Parallel(n_jobs=workers, prefer="processes")(
                delayed(_compute_diagram)(r) for r in to_compute
            )
            for res in results:
                _save_result(res)
        else:
            for i, r in enumerate(to_compute):
                _save_result(_compute_diagram(r))
                if (i + 1) % 100 == 0:
                    logger.info(f"  {i + 1}/{len(to_compute)} diagrams done")

    logger.info(f"Stage 1 complete: {len(diagrams)} diagrams available.")
    return diagrams


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — WASSERSTEIN DISTANCE MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def _wass_row(
    i:    int,
    pfxs: List[str],
    dgms: Dict[str, Dict],
    key:  str,
) -> np.ndarray:
    """Compute one row of the upper-triangular Wasserstein distance matrix."""
    n   = len(pfxs)
    row = np.zeros(n, dtype=np.float32)
    di  = dgms.get(pfxs[i], {}).get(key, np.empty((0, 2)))
    if len(di) == 0:
        di = np.array([[0.0, 0.0]])

    for j in range(i + 1, n):
        dj = dgms.get(pfxs[j], {}).get(key, np.empty((0, 2)))
        if len(dj) == 0:
            dj = np.array([[0.0, 0.0]])

        if HAS_PERSIM:
            row[j] = float(_persim_wass(di, dj, matching=False))
        else:
            # Fallback: sorted-lifetime L2 (approximation only)
            def _lt(d: np.ndarray) -> np.ndarray:
                return np.sort(d[:, 1] - d[:, 0])[::-1]
            la, lb = _lt(di), _lt(dj)
            m = max(len(la), len(lb))
            la = np.pad(la, (0, m - len(la)))
            lb = np.pad(lb, (0, m - len(lb)))
            row[j] = float(np.sqrt(np.sum((la - lb) ** 2)))
    return row


def stage2_wasserstein(
    runs:      List[Dict],
    diagrams:  Dict[str, Dict],
    out_dir:   Path,
    degree:    int,
    workers:   int,
    overwrite: bool,
) -> np.ndarray:
    """Compute (or load) the full pairwise Wasserstein-2 distance matrix."""
    cache = out_dir / f"wasserstein_H{degree}.npy"
    ckpt  = out_dir / f"wasserstein_H{degree}_checkpoint.npy"

    if cache.exists() and not overwrite:
        logger.info(f"Stage 2 H{degree}: loading cached matrix.")
        return np.load(cache).astype(float)

    key   = f"h{degree}"
    n     = len(runs)
    pfxs  = [r["prefix"] for r in runs]
    D     = np.zeros((n, n), dtype=np.float32)
    start_row = 0

    # Resume from checkpoint if available
    if ckpt.exists() and not overwrite:
        D_ckpt = np.load(ckpt)
        if D_ckpt.shape == (n, n):
            D = D_ckpt.astype(np.float32)
            for r_idx in range(n):
                if np.any(D[r_idx, r_idx + 1:] > 0) or r_idx == n - 1:
                    start_row = r_idx + 1
                else:
                    break
            logger.info(f"Stage 2 H{degree}: resuming from row {start_row}.")

    total_pairs = n * (n - 1) // 2
    logger.info(
        f"Stage 2: H{degree}, {n} runs, "
        f"{total_pairs:,} pairs, workers={workers}."
    )

    rows_todo = list(range(start_row, n))

    if workers > 1 and HAS_JOBLIB:
        results = Parallel(n_jobs=workers, prefer="processes", verbose=5)(
            delayed(_wass_row)(i, pfxs, diagrams, key) for i in rows_todo
        )
        for idx, row in zip(rows_todo, results):
            D[idx, idx + 1:] = row[idx + 1:]
            D[idx + 1:, idx] = row[idx + 1:]
            if (idx + 1) % CHECKPOINT_ROWS == 0:
                np.save(ckpt, D)
    else:
        for batch_s in range(start_row, n, CHECKPOINT_ROWS):
            for i in range(batch_s, min(batch_s + CHECKPOINT_ROWS, n)):
                if i % 50 == 0:
                    logger.info(f"  H{degree} row {i}/{n}")
                row = _wass_row(i, pfxs, diagrams, key)
                D[i, i + 1:] = row[i + 1:]
                D[i + 1:, i] = row[i + 1:]
            np.save(ckpt, D)

    np.save(cache, D)
    if ckpt.exists():
        ckpt.unlink()
    logger.info(f"Stage 2 H{degree}: matrix saved.")
    return D.astype(float)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — m7 TOPOLOGICAL SUMMARY FEATURES  (10 per run)
#
# Feature inventory (prefix: m7_):
#   m7_h0_landscape_integral    ∫λ₁(t) dt  H0  [µm²]
#   m7_h1_landscape_integral    ∫λ₁(t) dt  H1  [µm²]
#   m7_h0_landscape_peak_t      argmax λ₁(t)  H0  [µm]
#   m7_h1_landscape_peak_t      argmax λ₁(t)  H1  [µm]
#   m7_h0_persistent_entropy    −Σ(lᵢ/L) log(lᵢ/L)  H0
#   m7_h1_persistent_entropy    −Σ(lᵢ/L) log(lᵢ/L)  H1
#   m7_h0_betti0_mean           mean β₀(r) over filtration grid
#   m7_h0_betti0_var            variance β₀(r)
#   m7_h1_betti1_mean           mean β₁(r) over filtration grid
#   m7_h1_betti1_var            variance β₁(r)
# ══════════════════════════════════════════════════════════════════════════════

def _dgm_to_landscape(dgm: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    """
    Convert a persistence diagram to landscape functions.
    Returns shape (LANDSCAPE_K_MAX, len(t_vals)); zero-pads if needed.
    """
    k_max = LANDSCAPE_K_MAX
    n_t   = len(t_vals)
    if len(dgm) == 0:
        return np.zeros((k_max, n_t))
    tents = np.maximum(0.0, np.minimum(
        t_vals[None, :] - dgm[:, 0, None],
        dgm[:, 1, None] - t_vals[None, :],
    ))
    st  = np.sort(tents, axis=0)[::-1]
    n_b = st.shape[0]
    if n_b < k_max:
        st = np.vstack([st, np.zeros((k_max - n_b, n_t))])
    return st[:k_max]


def _persistent_entropy(dgm: np.ndarray) -> float:
    """Shannon entropy of the bar-lifetime distribution."""
    if len(dgm) == 0:
        return 0.0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    L = lifetimes.sum()
    p = lifetimes / L
    return float(-np.sum(p * np.log(p + 1e-300)))


def _betti_curve(dgm: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    """β(r) = number of bars alive at each filtration value r."""
    if len(dgm) == 0:
        return np.zeros(len(t_vals))
    births = dgm[:, 0][:, None]
    deaths = dgm[:, 1][:, None]
    return ((births <= t_vals[None, :]) & (t_vals[None, :] < deaths)).sum(
        axis=0
    ).astype(float)


def compute_m7_features(
    run_info: Dict,
    diagrams: Dict[str, Dict],
) -> Dict[str, float]:
    """Return the 10 m7 Topological Summary features for one run."""
    t_land  = np.linspace(0, PH_MAX_DIST / 2, LANDSCAPE_N_T)
    t_betti = np.linspace(0, PH_MAX_DIST,      BETTI_N_T)
    dt      = t_land[1] - t_land[0]

    dgm_dict = diagrams.get(run_info["prefix"], {})
    feats: Dict[str, float] = {}

    for deg, key in [(0, "h0"), (1, "h1")]:
        dgm  = dgm_dict.get(key, np.empty((0, 2)))
        land = _dgm_to_landscape(dgm, t_land)
        lam1 = land[0]

        feats[f"m7_{key}_landscape_integral"] = float(np.trapezoid(lam1, dx=dt))
        feats[f"m7_{key}_landscape_peak_t"]   = (
            float(t_land[np.argmax(lam1)]) if lam1.max() > 0 else 0.0
        )
        feats[f"m7_{key}_persistent_entropy"] = _persistent_entropy(dgm)

        bk   = f"betti{deg}"
        bc   = _betti_curve(dgm, t_betti)
        feats[f"m7_{key}_{bk}_mean"] = float(bc.mean())
        feats[f"m7_{key}_{bk}_var"]  = float(bc.var())

    return feats


def stage3_m7_features(
    runs:     List[Dict],
    diagrams: Dict[str, Dict],
    out_dir:  Path,
) -> Dict[str, Dict]:
    """Compute m7 features for all runs and save to JSON."""
    logger.info("Stage 3: computing m7 Topological Summary features...")
    m7_all: Dict[str, Dict] = {
        r["prefix"]: compute_m7_features(r, diagrams) for r in runs
    }
    m7_path = out_dir / "m7_topological_features.json"
    with open(m7_path, "w") as fh:
        json.dump(m7_all, fh, indent=2)
    logger.info(f"  Saved: {m7_path.name}  ({len(m7_all)} runs)")
    return m7_all


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — UMAP EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def stage4_umap(
    D:            np.ndarray,
    runs:         List[Dict],
    degree:       int,
    out_dir:      Path,
    save_figures: bool,
) -> np.ndarray:
    """Fit UMAP on the pre-computed Wasserstein distance matrix."""
    if not HAS_UMAP:
        logger.warning("umap-learn not installed — skipping UMAP.")
        return np.zeros((len(runs), 2))

    Ds = (D + D.T) / 2
    np.fill_diagonal(Ds, 0.0)
    logger.info(f"  Fitting UMAP H{degree} ({len(runs)} points)…")
    emb = umap.UMAP(
        n_components=2, metric="precomputed",
        n_neighbors=15, min_dist=0.1,
        random_state=42, n_jobs=1,
    ).fit_transform(Ds)

    if not save_figures:
        return emb

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor("white")

    # Panel A — particle × SOBP position
    ax = axes[0]
    for pk in PARTICLE_KEY_ORDER:
        pc   = PCONF_BY_KEY[pk]
        mask = np.array([r["particle_key"] == pk for r in runs])
        if not mask.any():
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=PARTICLE_COLORS[pk], marker=SOBP_MARKERS[pc["sobp"]],
                   label=pc["label"], s=20, alpha=0.80, linewidths=0, zorder=3)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    leg = ax.legend(title="Particle / SOBP", fontsize=6.5, title_fontsize=7.5,
                    framealpha=0.95, edgecolor="#CCCCCC", ncol=1)
    leg.get_title().set_fontweight("bold")
    _style_ax(ax)
    _title_sub(ax, f"H{degree} — by Particle",
               f"Wasserstein-2  ·  n={len(runs)}  ·  ○ pSOBP  △ dSOBP  □ mono")

    # Panel B — O2 level
    ax = axes[1]
    for i, o2 in enumerate(O2_ORDERED):
        mask = np.array([r["o2"] == o2 for r in runs])
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=O2_COLORS[i], label=f"{o2}%",
                   s=20, alpha=0.80, linewidths=0, zorder=3)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    leg = ax.legend(title="O\u2082 Level", fontsize=7.5, title_fontsize=8,
                    framealpha=0.95, edgecolor="#CCCCCC")
    leg.get_title().set_fontweight("bold")
    _style_ax(ax)
    _title_sub(ax, f"H{degree} — by O\u2082 Level",
               f"Wasserstein-2  ·  n={len(runs)}")

    # Panel C — SOBP position
    ax = axes[2]
    for sobp, color in SOBP_COLORS.items():
        mask = np.array([r["sobp"] == sobp for r in runs])
        if not mask.any():
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=color, marker=SOBP_MARKERS[sobp], label=sobp.upper(),
                   s=20, alpha=0.80, linewidths=0, zorder=3)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    leg = ax.legend(title="SOBP Position", fontsize=8, title_fontsize=8.5,
                    framealpha=0.95, edgecolor="#CCCCCC")
    leg.get_title().set_fontweight("bold")
    _style_ax(ax)
    _title_sub(ax, f"H{degree} — by SOBP Position",
               f"Wasserstein-2  ·  n={len(runs)}")

    fig.tight_layout()
    _save(fig, out_dir / f"umap_H{degree}.png")
    return emb


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — WITHIN / BETWEEN CONDITION COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def stage5_within_between(
    D:            np.ndarray,
    runs:         List[Dict],
    degree:       int,
    out_dir:      Path,
    save_figures: bool,
) -> Dict:
    n         = len(runs)
    cond_idxs = np.array([r["cond_idx"] for r in runs])
    cond_sum  = np.zeros((N_CONDITIONS, N_CONDITIONS), dtype=float)
    cond_cnt  = np.zeros((N_CONDITIONS, N_CONDITIONS), dtype=int)
    within: List[float] = []
    between: List[float] = []

    for i in range(n):
        for j in range(i + 1, n):
            d  = float(D[i, j])
            ci = int(cond_idxs[i])
            cj = int(cond_idxs[j])
            cond_sum[ci, cj] += d; cond_sum[cj, ci] += d
            cond_cnt[ci, cj] += 1; cond_cnt[cj, ci] += 1
            (within if ci == cj else between).append(d)

    with np.errstate(invalid="ignore"):
        cond_mean = np.where(cond_cnt > 0, cond_sum / cond_cnt, 0.0)

    wa  = np.array(within)
    ba  = np.array(between)
    sep = (float(np.median(ba)) / float(np.median(wa))
           if np.median(wa) > 0 else float("nan"))

    result = {
        "within_mean":           float(np.mean(wa)),
        "within_median":         float(np.median(wa)),
        "within_std":            float(np.std(wa)),
        "between_mean":          float(np.mean(ba)),
        "between_median":        float(np.median(ba)),
        "between_std":           float(np.std(ba)),
        "separation_ratio":      sep,
        "condition_mean_matrix": cond_mean.tolist(),
    }

    if not save_figures:
        return result

    # ── Violin ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    fig.patch.set_facecolor("white")
    vp = ax.violinplot([wa, ba], positions=[0, 1],
                       showmedians=True, showextrema=True)
    for body, c in zip(vp["bodies"],
                       [SOBP_COLORS["dsobp"], PARTICLE_COLORS["carbon_psobp"]]):
        body.set_facecolor(c); body.set_alpha(0.65); body.set_edgecolor("white")
    for k in ["cmedians", "cbars", "cmins", "cmaxes"]:
        vp[k].set_color("#444444"); vp[k].set_linewidth(1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Within condition", "Between conditions"], fontsize=9)
    ax.set_ylabel(f"Wasserstein-2 distance (H{degree})")
    _style_ax(ax)
    _title_sub(ax, f"H{degree} Diagram Separability",
               (f"Within median = {result['within_median']:.3f}   |   "
                f"Between median = {result['between_median']:.3f}   |   "
                f"Ratio = {sep:.2f}"))
    fig.tight_layout()
    _save(fig, out_dir / f"within_between_H{degree}.png")

    # ── 49 × 49 condition-mean heatmap ────────────────────────────────────────
    n_per = len(O2_ORDERED)    # 7
    divider_pos = [n_per * k - 0.5 for k in range(1, len(PARTICLE_CONFIGS))]

    fig, ax = plt.subplots(figsize=(16.0, 15.0))
    fig.patch.set_facecolor("white")
    im   = ax.imshow(cond_mean, cmap=_HEAT_CMAP, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean Wasserstein-2 distance", fontsize=8.5, color="#555555")
    cbar.ax.tick_params(labelsize=7, colors="#555555")
    cbar.outline.set_edgecolor("#CCCCCC")
    ax.set_xticks(range(N_CONDITIONS))
    ax.set_yticks(range(N_CONDITIONS))
    ax.set_xticklabels(CONDITION_LABELS, rotation=45, ha="right", fontsize=5.5)
    ax.set_yticklabels(CONDITION_LABELS, fontsize=5.5)
    ax.grid(False)
    for pos in divider_pos:
        ax.axhline(pos, color="white", lw=1.6)
        ax.axvline(pos, color="white", lw=1.6)
    _style_ax(ax)
    _title_sub(ax, f"H{degree} Pairwise Wasserstein Distances",
               (f"{N_CONDITIONS} conditions × {N_RUNS} runs  ·  "
                "mean over all inter-condition pairs"))
    fig.tight_layout()
    _save(fig, out_dir / f"condition_heatmap_H{degree}.png")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — PERSISTENCE LANDSCAPES
# ══════════════════════════════════════════════════════════════════════════════

def stage6_landscapes(
    runs:         List[Dict],
    diagrams:     Dict[str, Dict],
    degree:       int,
    out_dir:      Path,
    save_figures: bool,
) -> Dict[str, List]:
    """Mean λ₁(t) landscape per condition, one panel per particle species."""
    key    = f"h{degree}"
    t_vals = np.linspace(0, PH_MAX_DIST / 2, LANDSCAPE_N_T)

    acc: Dict[str, List] = {
        f"{pk}_{o}": [] for pk in PARTICLE_KEY_ORDER for o in O2_ORDERED
    }
    for r in runs:
        dgm  = diagrams.get(r["prefix"], {}).get(key, np.empty((0, 2)))
        acc[r["condition"]].append(_dgm_to_landscape(dgm, t_vals))

    means = {c: np.mean(v, axis=0) for c, v in acc.items() if v}

    if not save_figures:
        return {c: m.tolist() for c, m in means.items()}

    # One panel per distinct particle species (electron, proton, helium, carbon)
    species = list(dict.fromkeys(
        PCONF_BY_KEY[pk]["particle"] for pk in PARTICLE_KEY_ORDER
    ))
    n_sp = len(species)
    fig, axes = plt.subplots(1, n_sp, figsize=(4.8 * n_sp, 5.0), sharey=True)
    if n_sp == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, sp in zip(axes, species):
        # Collect all SOBP variants for this species
        sp_keys = [pk for pk in PARTICLE_KEY_ORDER
                   if PCONF_BY_KEY[pk]["particle"] == sp]
        multi_sobp = len(sp_keys) > 1

        for pk in sp_keys:
            pc = PCONF_BY_KEY[pk]
            ls = "-" if pc["sobp"] in ("psobp", "mono") else "--"
            lw = 2.0 if ls == "-" else 1.3

            for i, o2 in enumerate(O2_ORDERED):
                cond = f"{pk}_{o2}"
                if cond not in means:
                    continue
                lam1  = means[cond][0]
                label = (f"{o2}% ({pc['sobp'].upper()})"
                         if multi_sobp else f"{o2}% O\u2082")
                ax.plot(t_vals, lam1,
                        color=O2_COLORS[i], linestyle=ls, linewidth=lw,
                        alpha=0.92, label=label, zorder=3)

        ax.set_xlabel("Filtration value (\u00b5m)", labelpad=6)
        if ax is axes[0]:
            ax.set_ylabel(f"\u03bb\u2081(t)  [H{degree}]", labelpad=6)
        leg = ax.legend(title="O\u2082 / SOBP", fontsize=6.5,
                        title_fontsize=7.5, framealpha=0.95, edgecolor="#CCCCCC")
        leg.get_title().set_fontweight("bold")
        _style_ax(ax)
        _strip(ax, sp.capitalize())

    fig.text(0.01, 0.975,
             f"Mean Persistence Landscape \u03bb\u2081 \u2014 H{degree}",
             fontsize=12, fontweight="bold", color="#1A1A1A", va="top")
    fig.text(0.01, 0.925,
             "Solid = pSOBP/mono  ·  Dashed = dSOBP  ·  mean over 50 runs per condition",
             fontsize=8.5, color="#666666", style="italic", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, out_dir / f"landscape_H{degree}.png")

    return {c: m.tolist() for c, m in means.items()}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="PH topology analysis of DSB point clouds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--basedir", type=Path, default=Path("."),
        help="Project root directory (default: current directory).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers for diagram computation and Wasserstein rows.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Ignore cached diagrams and distance matrices; recompute all.",
    )
    parser.add_argument(
        "--skip-figures", action="store_true",
        help=(
            "Write JSON outputs only; skip all matplotlib figures. "
            "07_regenerate_figures.py handles final 600 DPI outputs."
        ),
    )
    parser.add_argument("--skip-landscapes", action="store_true")
    args = parser.parse_args()

    # ── Dependency checks ─────────────────────────────────────────────────
    if not HAS_RIPSER:
        logger.error("ripser not installed.  pip install ripser")
        return 1
    if not HAS_PERSIM:
        logger.warning(
            "persim not installed — falling back to approximate Wasserstein.  "
            "pip install persim"
        )
    if not HAS_UMAP:
        logger.warning(
            "umap-learn not installed — UMAP stage skipped.  "
            "pip install umap-learn"
        )
    if not HAS_JOBLIB and args.workers > 1:
        logger.warning(
            "joblib not installed — running single-threaded.  "
            "pip install joblib"
        )

    base_dir = args.basedir.resolve()
    out_dir  = base_dir / "analysis" / "ph"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 62)
    logger.info("02_ph_topology_analysis.py")
    logger.info(f"  Base dir      : {base_dir}")
    logger.info(f"  Output dir    : {out_dir}")
    logger.info(f"  Workers       : {args.workers}")
    logger.info(
        f"  Design        : {len(PARTICLE_CONFIGS)} particles × "
        f"{len(O2_ORDERED)} O2 levels × {N_RUNS} runs  "
        f"= {N_CONDITIONS * N_RUNS:,} max runs"
    )
    logger.info("=" * 62)

    # ── Discover runs ─────────────────────────────────────────────────────
    runs = discover_runs(base_dir)
    if not runs:
        logger.error(
            "No run directories found.\n"
            "  Expected layout: <base_dir>/<dir_name>/<dir_name>_<o2>/"
            "<dir_name>_<o2>_<run_id>/<prefix>_dsb_complexity.csv"
        )
        return 1

    # Save run index for downstream scripts (especially 07)
    run_index = [
        {k: v for k, v in r.items() if k != "dsb_file"} for r in runs
    ]
    with open(out_dir / "run_index.json", "w") as fh:
        json.dump(run_index, fh, indent=2)
    logger.info(f"Run index saved ({len(runs)} runs).")

    # ── Stage 1: persistence diagrams ────────────────────────────────────
    logger.info("=" * 62)
    logger.info("STAGE 1 — Persistence diagrams")
    logger.info("=" * 62)
    diagrams = stage1_diagrams(runs, out_dir, args.workers, args.overwrite)
    if not diagrams:
        logger.error("No diagrams computed — aborting.")
        return 1

    # ── Stage 3: m7 features (depends only on diagrams, not D) ───────────
    logger.info("=" * 62)
    logger.info("STAGE 3 — m7 Topological Summary features")
    logger.info("=" * 62)
    m7_all = stage3_m7_features(runs, diagrams, out_dir)

    # ── Stages 2, 4, 5, 6: once per homology degree ──────────────────────
    summary: Dict = {
        "n_runs":           len(runs),
        "n_conditions":     N_CONDITIONS,
        "o2_levels":        O2_ORDERED,
        "particle_configs": [
            {k: v for k, v in pc.items() if k != "dir_name"}
            for pc in PARTICLE_CONFIGS
            if any(r["particle_key"] == pc["key"] for r in runs)
        ],
    }
    umap_embeddings: Dict[str, List] = {}
    landscape_means: Dict[str, Dict] = {}

    for degree in [0, 1]:
        logger.info("=" * 62)
        logger.info(f"STAGES 2 / 4 / 5 / 6 — H{degree}")
        logger.info("=" * 62)

        D = stage2_wasserstein(
            runs, diagrams, out_dir, degree, args.workers, args.overwrite
        )

        emb = stage4_umap(
            D, runs, degree, out_dir,
            save_figures=not args.skip_figures,
        )
        umap_embeddings[str(degree)] = emb.tolist()

        wb = stage5_within_between(
            D, runs, degree, out_dir,
            save_figures=not args.skip_figures,
        )
        summary[f"h{degree}"] = wb
        logger.info(
            f"  H{degree}  within median = {wb['within_median']:.4f}  "
            f"between median = {wb['between_median']:.4f}  "
            f"ratio = {wb['separation_ratio']:.3f}"
        )

        if not args.skip_landscapes:
            lm = stage6_landscapes(
                runs, diagrams, degree, out_dir,
                save_figures=not args.skip_figures,
            )
            landscape_means[str(degree)] = lm

    # ── Save comprehensive ph_summary.json for 07 ─────────────────────────
    summary["umap_embeddings"] = umap_embeddings
    summary["landscape_means"] = landscape_means
    summary["condition_order"] = [
        {
            "condition":    f"{pk}_{o}",
            "particle_key": pk,
            "o2":           o,
            "particle":     PCONF_BY_KEY[pk]["particle"],
            "sobp":         PCONF_BY_KEY[pk]["sobp"],
            "let":          PCONF_BY_KEY[pk]["let"],
        }
        for pk, o in CONDITIONS
        if any(r["condition"] == f"{pk}_{o}" for r in runs)
    ]

    with open(out_dir / "ph_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("ph_summary.json saved.")

    # ── Final summary ─────────────────────────────────────────────────────
    logger.info("=" * 62)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 62)
    for deg in [0, 1]:
        s = summary[f"h{deg}"]
        logger.info(
            f"  H{deg}  within: {s['within_median']:.4f}  "
            f"between: {s['between_median']:.4f}  "
            f"ratio: {s['separation_ratio']:.3f}"
        )
    logger.info(f"  m7 features : 10 per run × {len(m7_all)} runs")
    logger.info(f"  Outputs     : {out_dir}")
    logger.info("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
