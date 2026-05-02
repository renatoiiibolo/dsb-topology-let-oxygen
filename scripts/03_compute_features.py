#!/usr/bin/env python3
"""
================================================================================
03_compute_features.py
================================================================================
Extract multimodal DSB topology features (Modalities 1–6) for every simulation
run, writing one JSON file per run to analysis/features/.

PIPELINE POSITION
-----------------
  01  extract_dsb.py                    →  *_dsb_complexity.csv  ← INPUT
  02  ph_topology_analysis.py           →  m7_topological_features.json
  03  compute_features.py               →  analysis/features/*.json  ← THIS
  04  build_feature_matrix.py           →  feature_matrix.csv (m1–m7)
  05  random_forest.py
  06  additional_analyses.py
  07  regenerate_figures.py

NOTE: Script 00 (parse_sdd_particle_history.py) is no longer part of the
pipeline. It existed solely for the old Event Attribution modality (m7).
With that modality retired, script 00 can be deleted.

WHY MODALITY 7 IS NOT HERE
---------------------------
The original Event Attribution modality relied on event_id provenance — a
privileged simulation tag with no experimental counterpart (you cannot identify
which primary particle caused which γH2AX focus under a microscope). It has
been retired and replaced by Topological Summaries (new m7), computed in
02_ph_topology_analysis.py from the per-run persistence diagrams. Script 04
merges the m1–m6 features produced here with the m7 features from 02.

DIRECTORY LAYOUT
----------------
Scripts live at the same level as the particle directories. The full directory
name (particle + LET) propagates through ALL subdirectory levels and file
prefixes — the LET tag appears at every level:

  project_root/
  ├── 03_compute_features.py
  ├── carbon_40.9/                       ← {dir_name}
  │   └── carbon_40.9_21.0/              ← {dir_name}_{o2}
  │       └── carbon_40.9_21.0_01/       ← {dir_name}_{o2}_{run_id}
  │           ├── carbon_40.9_21.0_01_dsb_complexity.csv   ← required
  │           ├── carbon_40.9_21.0_01_EnergyDeposit.csv    ← optional (m3)
  │           └── carbon_40.9_21.0_01_Dose.csv             ← optional (m4)
  ├── carbon_70.7/                       ← carbon dSOBP
  ├── electron_0.2/
  ├── helium_10.0/                         ← helium pSOBP
  ├── helium_30.0/                         ← helium dSOBP
  ├── proton_4.6/                        ← proton pSOBP
  └── proton_8.1/                        ← proton dSOBP

MODALITIES COMPUTED (m1–m6)
---------------------------
  m1_*   Spatial Distribution        — 3D DSB point cloud, NN, Ripley K/L, PH
  m2_*   Radial Track Structure      — transverse distance from beam axis
  m3_*   Local Energy Heterogeneity  — EnergyDeposit.csv voxel statistics
  m4_*   Dose Distribution           — Dose.csv voxel DVH statistics
  m5_*   Genomic Location            — chromosomal distribution + 1D PH
  m6_*   Damage Complexity           — DSB / DSB+ / DSB++ breakdown

OUTPUTS
-------
  analysis/features/{prefix}_features.json    one file per run
  prefix = {dir_name}_{o2}_{run_id}, e.g. carbon_40.9_21.0_01

USAGE
-----
  python 03_compute_features.py
  python 03_compute_features.py --basedir /path/to/project
  python 03_compute_features.py --particle carbon_psobp --o2 21.0
  python 03_compute_features.py --workers 4 --overwrite

ADDING NEW SOBP RUNS
--------------------
  Set "dir_name" and "let" in PARTICLE_CONFIGS to match your new directory.
  Set O2_ORDERED if you add new oxygen levels. Nothing else needs changing.

DEPENDENCIES
------------
  numpy, pandas, scipy
  ripser       (optional — m1 PH features; pip install ripser)
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from ripser import ripser as _ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        CONFIGURATION                                    ║
# ║  The only section that ever needs editing for new SOBP runs.            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

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

# ── O2 levels ─────────────────────────────────────────────────────────────────
# Seven levels spanning the full OER curve:
#   Plateau       : 21.0, 5.0
#   Transition    : 2.1, 0.5
#   Hypoxic flank : 0.1, 0.021, 0.005
O2_ORDERED:  List[str] = ["21.0", "5.0", "2.1", "0.5", "0.1", "0.021", "0.005"]
O2_NORMOXIC: str       = "21.0"

# ── Runs per condition ────────────────────────────────────────────────────────
N_RUNS: int = 50

# ── Nuclear geometry ──────────────────────────────────────────────────────────
NUCLEUS_RADIUS: float = 4.65                                          # µm
NUCLEAR_VOLUME: float = (4.0 / 3.0) * np.pi * NUCLEUS_RADIUS ** 3   # µm³

# ── Voxel grid (60³) ─────────────────────────────────────────────────────────
VOXEL_N:    int   = 60
VOXEL_HALF: float = 9.28055               # µm (half-span of grid)
VOXEL_SIZE: float = 2.0 * VOXEL_HALF / VOXEL_N    # ~0.309352 µm

# ── Chromosome sizes (Mbp) ────────────────────────────────────────────────────
# Diploid human genome: chromosomes 1–46 (1–22 × 2, X × 2, Y, extra Y).
# Genomic distance = chromosome_position (0–1) × CHROM_SIZES_MBP[chrom].
CHROM_SIZES_MBP: Dict[int, float] = {
     1: 252.823,  2: 252.823,
     3: 202.118,  4: 202.118,
     5: 183.511,  6: 183.511,
     7: 162.767,  8: 162.767,
     9: 144.886, 10: 144.886,
    11: 137.544, 12: 137.544,
    13: 117.473, 14: 117.473,
    15: 104.123, 16: 104.123,
    17:  83.534, 18:  83.534,
    19:  64.444, 20:  64.444,
    21:  48.129, 22:  48.129,
    23: 156.040, 24: 156.040,   # X homologs
    25: 252.823, 26: 252.823,
    27: 202.118, 28: 202.118,
    29: 183.511, 30: 183.511,
    31: 162.767, 32: 162.767,
    33: 144.886, 34: 144.886,
    35: 137.544, 36: 137.544,
    37: 117.473, 38: 117.473,
    39: 104.123, 40: 104.123,
    41:  83.534, 42:  83.534,
    43:  64.444, 44:  64.444,
    45: 156.040,
    46:  57.227,
}

# ── Spatial / clustering thresholds ──────────────────────────────────────────
SPATIAL_RADII:   List[float] = [0.5, 1.0, 2.0]      # µm (NN fraction)
RIPLEY_RADII_3D: List[float] = [0.5, 1.0, 1.5, 2.0] # µm (Ripley K/L)
RIPLEY_RADII_2D: List[float] = [0.5, 1.0]            # µm (transverse K)
RADIAL_THRESH:   List[float] = [0.5, 1.0, 1.5, 2.0] # µm (radial fractions)

# ── Genomic distance thresholds ───────────────────────────────────────────────
GENOMIC_THRESH_MBP: List[float] = [1.0, 5.0, 10.0]

# ── PH parameters (Modality 1 only) ──────────────────────────────────────────
PH_H0_THRESH: float = 0.5                 # µm minimum H0 lifetime to count
PH_H1_THRESH: float = 0.3                 # µm minimum H1 persistence to count
PH_MAX_DIST:  float = 2.0 * NUCLEUS_RADIUS  # µm filtration upper bound (9.3 µm)

# ══════════════════════════════════════════════════════════════════════════════
# DERIVED LOOKUPS  (do not edit)
# ══════════════════════════════════════════════════════════════════════════════

PCONF_BY_KEY:     Dict[str, Dict] = {c["key"]:      c for c in PARTICLE_CONFIGS}
PCONF_BY_DIRNAME: Dict[str, str]  = {c["dir_name"]: c["key"]
                                      for c in PARTICLE_CONFIGS}

# ══════════════════════════════════════════════════════════════════════════════
# RUN DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def _norm_o2(v) -> str:
    """Normalise floating-point o2 string to canonical form."""
    try:
        f = float(v)
        for s in O2_ORDERED:
            if abs(f - float(s)) < 1e-9:
                return s
    except (ValueError, TypeError):
        pass
    return str(v)


def discover_runs(
    base_dir:        Path,
    particle_filter: Optional[str] = None,   # canonical key, e.g. "carbon_psobp"
    o2_filter:       Optional[str] = None,   # e.g. "21.0"
) -> List[Dict]:
    """
    Walk <base_dir> for per-run DSB complexity files.

    The {dir_name} (e.g. "carbon_40.9") propagates through every path level:

        <base_dir>/
        ├── carbon_40.9/                  ← dir_name
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
        if particle_filter and canon_key != particle_filter:
            continue

        pconf = PCONF_BY_KEY[canon_key]

        for o2_raw in O2_ORDERED:
            o2_norm = _norm_o2(o2_raw)
            if o2_filter and o2_norm != _norm_o2(o2_filter):
                continue

            # Level 2: {dir_name}_{o2}   e.g. carbon_40.9_21.0
            o2_dir = pdir / f"{dname}_{o2_raw}"
            if not o2_dir.exists():
                continue

            for run_num in range(1, N_RUNS + 1):
                run_id   = f"{run_num:02d}"
                # Level 3 prefix: {dir_name}_{o2}_{run_id}  e.g. carbon_40.9_21.0_01
                prefix   = f"{dname}_{o2_raw}_{run_id}"
                run_dir  = o2_dir / prefix
                dsb_file = run_dir / f"{prefix}_dsb_complexity.csv"

                if not dsb_file.exists():
                    logger.debug(f"Missing dsb_complexity.csv: {prefix}")
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
                    "run_dir":      run_dir,
                    "is_normoxic":  (o2_norm == O2_NORMOXIC),
                })

    runs.sort(key=lambda r: (
        [c["key"] for c in PARTICLE_CONFIGS].index(r["particle_key"]),
        O2_ORDERED.index(r["o2"]),
        r["run_id"],
    ))
    found_keys = sorted(set(r["particle_key"] for r in runs))
    logger.info(f"Discovered {len(runs)} valid runs "
                f"across {len(found_keys)} configuration(s): {found_keys}")
    return runs

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient (0 = uniform, 1 = fully concentrated)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0 or v.sum() == 0:
        return 0.0
    v = np.sort(v)
    n = len(v)
    return float(
        (2.0 * np.dot(np.arange(1, n + 1), v) / (n * v.sum())) - (n + 1) / n
    )


def shannon_entropy(values: np.ndarray, n_bins: int = 20) -> float:
    """Shannon entropy (nats) estimated via histogram."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    n_bins = max(2, min(n_bins, len(v)))
    counts, _ = np.histogram(v, bins=n_bins)
    counts     = counts[counts > 0]
    p          = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-15)))


def chrom_pos_to_mbp(chromosome: int, chromosome_position: float) -> float:
    """Convert fractional chromosome_position (0–1) to absolute Mbp."""
    return float(chromosome_position) * CHROM_SIZES_MBP.get(int(chromosome), 100.0)


def ripley_k_3d(
    coords: np.ndarray,
    radii:  List[float],
    volume: float,
) -> Dict[str, float]:
    """
    Ripley's K and Besag's L for a 3D point pattern (no edge correction).

    K(r) = (V / n²) × 2 × |{pairs with distance < r}|
    L(r) = (K(r) × 3 / (4π))^(1/3) − r
         = 0 under CSR; > 0 indicates clustering.
    """
    result: Dict[str, float] = {}
    n = len(coords)
    if n < 2:
        for r in radii:
            result[f"ripley_K_{r}um"] = np.nan
            result[f"ripley_L_{r}um"] = np.nan
        return result

    dists = pdist(coords)
    for r in radii:
        count = int(np.sum(dists < r))
        K = (volume / n ** 2) * 2.0 * count
        L = (K * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) - r
        result[f"ripley_K_{r}um"] = float(K)
        result[f"ripley_L_{r}um"] = float(L)
    return result


def ph_features_3d(coords: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """
    H0 + H1 persistent homology of a 3D DSB point cloud (Vietoris-Rips).

    H0 lifetimes encode inter-cluster gap distances; H1 persistences encode
    loop scales. Requires ripser; returns NaN for all keys if unavailable.
    """
    keys = [
        "h0_n_long_components", "h0_max_lifetime_um",
        "h0_total_persistence_um", "h0_mean_lifetime_um",
        "h1_n_loops", "h1_max_persistence_um",
        "h1_total_persistence_um", "h1_mean_persistence_um",
    ]
    feats = {f"{prefix}{k}": np.nan for k in keys}
    if not HAS_RIPSER or len(coords) < 3:
        return feats
    try:
        result = _ripser(coords, maxdim=1, thresh=PH_MAX_DIST)
        dgms   = result["dgms"]

        # H0
        h0        = dgms[0]
        h0_finite = h0[h0[:, 1] < np.inf]
        lt_h0     = h0_finite[:, 1] - h0_finite[:, 0]
        feats[f"{prefix}h0_n_long_components"]    = float(np.sum(lt_h0 > PH_H0_THRESH))
        feats[f"{prefix}h0_max_lifetime_um"]      = float(np.max(lt_h0))  if len(lt_h0) else 0.0
        feats[f"{prefix}h0_total_persistence_um"] = float(np.sum(lt_h0))
        feats[f"{prefix}h0_mean_lifetime_um"]     = float(np.mean(lt_h0)) if len(lt_h0) else 0.0

        # H1
        h1        = dgms[1]
        h1_finite = h1[h1[:, 1] < np.inf]
        if len(h1_finite):
            lt_h1 = h1_finite[:, 1] - h1_finite[:, 0]
            feats[f"{prefix}h1_n_loops"]              = float(np.sum(lt_h1 > PH_H1_THRESH))
            feats[f"{prefix}h1_max_persistence_um"]   = float(np.max(lt_h1))
            feats[f"{prefix}h1_total_persistence_um"] = float(np.sum(lt_h1))
            feats[f"{prefix}h1_mean_persistence_um"]  = float(np.mean(lt_h1))
        else:
            for k in ["h1_n_loops", "h1_max_persistence_um",
                      "h1_total_persistence_um", "h1_mean_persistence_um"]:
                feats[f"{prefix}{k}"] = 0.0

    except Exception as exc:
        logger.debug(f"PH computation failed: {exc}")
    return feats


def load_voxel_grid(filepath: Path) -> Optional[np.ndarray]:
    """
    Load a 60³ voxel grid from EnergyDeposit.csv or Dose.csv.

    Expected format: rows of  ix,iy,iz,value  (no header; # lines skipped).
    Returns a (60,60,60) float32 array, or None if absent/unreadable.
    """
    if not filepath.exists():
        return None
    try:
        df   = pd.read_csv(filepath, comment="#", header=None,
                           names=["ix", "iy", "iz", "value"])
        grid = np.zeros((VOXEL_N, VOXEL_N, VOXEL_N), dtype=np.float32)
        ix   = df["ix"].values.astype(int)
        iy   = df["iy"].values.astype(int)
        iz   = df["iz"].values.astype(int)
        val  = df["value"].values.astype(np.float32)
        mask = (
            (ix >= 0) & (ix < VOXEL_N) &
            (iy >= 0) & (iy < VOXEL_N) &
            (iz >= 0) & (iz < VOXEL_N)
        )
        grid[ix[mask], iy[mask], iz[mask]] = val[mask]
        return grid
    except Exception as exc:
        logger.warning(f"Could not load voxel grid {filepath.name}: {exc}")
        return None


def coords_to_voxel_idx(x: float, y: float, z: float) -> Tuple[int, int, int]:
    """Convert µm coordinates to clamped 60³ voxel indices."""
    ix = int(np.clip(np.floor((x + VOXEL_HALF) / VOXEL_SIZE), 0, VOXEL_N - 1))
    iy = int(np.clip(np.floor((y + VOXEL_HALF) / VOXEL_SIZE), 0, VOXEL_N - 1))
    iz = int(np.clip(np.floor((z + VOXEL_HALF) / VOXEL_SIZE), 0, VOXEL_N - 1))
    return ix, iy, iz


def _clean(v):
    """Serialise numpy scalars; map non-finite floats to None for JSON."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if not np.isfinite(float(v)) else float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v

# ══════════════════════════════════════════════════════════════════════════════
# MODALITY 1 — SPATIAL DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_modality_1_spatial(dsb_df: pd.DataFrame) -> Dict[str, float]:
    """
    3D DSB point cloud statistics: centroid, per-axis SD, radius of gyration,
    radial distance from nuclear centre, nearest-neighbour distances and
    clustering fractions, Ripley's K/L (3D), and Vietoris-Rips PH (H0, H1).

    Carbon ions produce a tight axial column (small σ_r, large σ_z); electrons
    are isotropic. PH lifetime profiles differ qualitatively across LET.
    Ripley's L > 0 indicates clustering above complete spatial randomness.
    """
    f   = {}
    tag = "m1_"
    cc  = ["x_um", "y_um", "z_um"]

    if not all(c in dsb_df.columns for c in cc):
        f[f"{tag}n_dsbs"] = 0.0
        return f

    coords = dsb_df[cc].values.astype(float)
    n      = len(coords)
    f[f"{tag}n_dsbs"] = float(n)
    if n == 0:
        return f

    # Centroid
    cen = coords.mean(axis=0)
    f[f"{tag}centroid_x_um"] = float(cen[0])
    f[f"{tag}centroid_y_um"] = float(cen[1])
    f[f"{tag}centroid_z_um"] = float(cen[2])

    # Per-axis standard deviation
    f[f"{tag}std_x_um"] = float(np.std(coords[:, 0]))
    f[f"{tag}std_y_um"] = float(np.std(coords[:, 1]))
    f[f"{tag}std_z_um"] = float(np.std(coords[:, 2]))

    # Radius of gyration
    f[f"{tag}radius_of_gyration_um"] = float(
        np.sqrt(np.mean(np.sum((coords - cen) ** 2, axis=1)))
    )

    # Radial distance from nuclear centre (origin)
    r_cen = np.linalg.norm(coords, axis=1)
    f[f"{tag}r_from_center_mean_um"] = float(np.mean(r_cen))
    f[f"{tag}r_from_center_std_um"]  = float(np.std(r_cen))
    f[f"{tag}r_from_center_skew"]    = float(stats.skew(r_cen)) if n > 2 else np.nan

    # Nearest-neighbour distances
    if n > 1:
        dmat = pdist(coords)
        from scipy.spatial.distance import squareform
        dmat_sq = squareform(dmat)
        np.fill_diagonal(dmat_sq, np.inf)
        nn = dmat_sq.min(axis=1)
        f[f"{tag}nn_mean_um"]   = float(np.mean(nn))
        f[f"{tag}nn_median_um"] = float(np.median(nn))
        f[f"{tag}nn_std_um"]    = float(np.std(nn))
        for r in SPATIAL_RADII:
            f[f"{tag}frac_nn_within_{r}um"] = float(np.mean(nn < r))
    else:
        f[f"{tag}nn_mean_um"]   = np.nan
        f[f"{tag}nn_median_um"] = np.nan
        f[f"{tag}nn_std_um"]    = np.nan
        for r in SPATIAL_RADII:
            f[f"{tag}frac_nn_within_{r}um"] = np.nan

    # Ripley's K and L (3D)
    rip = ripley_k_3d(coords, RIPLEY_RADII_3D, NUCLEAR_VOLUME)
    f.update({f"{tag}{k}": v for k, v in rip.items()})

    # Persistent Homology (H0 + H1, Vietoris-Rips)
    f.update(ph_features_3d(coords, prefix=tag))

    return f

# ══════════════════════════════════════════════════════════════════════════════
# MODALITY 2 — RADIAL TRACK STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def extract_modality_2_radial(dsb_df: pd.DataFrame) -> Dict[str, float]:
    """
    Transverse (radial) distance from the beam axis: r = √(x² + y²).

    Carbon: tight core (small mean r, large σ_z / σ_r).
    Proton: intermediate penumbra.
    Electron: isotropic (σ_z / σ_r ≈ 1).

    Also includes Ripley's K in the xy transverse plane and a Spearman
    correlation between radial distance and DSB complexity score.
    """
    f   = {}
    tag = "m2_"
    cc  = ["x_um", "y_um", "z_um"]
    if not all(c in dsb_df.columns for c in cc):
        return f

    x = dsb_df["x_um"].values.astype(float)
    y = dsb_df["y_um"].values.astype(float)
    z = dsb_df["z_um"].values.astype(float)
    r = np.sqrt(x ** 2 + y ** 2)
    n = len(r)
    if n == 0:
        return f

    f[f"{tag}r_mean_um"]  = float(np.mean(r))
    f[f"{tag}r_std_um"]   = float(np.std(r))
    f[f"{tag}r_skew"]     = float(stats.skew(r))     if n > 2 else np.nan
    f[f"{tag}r_kurtosis"] = float(stats.kurtosis(r)) if n > 2 else np.nan
    f[f"{tag}r_entropy"]  = shannon_entropy(r, n_bins=10)

    for thresh in RADIAL_THRESH:
        f[f"{tag}frac_r_lt_{thresh}um"] = float(np.mean(r < thresh))

    f[f"{tag}z_std_um"] = float(np.std(z))
    sigma_r = np.std(r)
    sigma_z = np.std(z)
    f[f"{tag}axial_radial_ratio"] = (
        float(sigma_z / sigma_r) if sigma_r > 1e-9 else np.nan
    )

    # Ripley's K in xy-plane (2D transverse cross-section)
    area_2d = np.pi * NUCLEUS_RADIUS ** 2
    xy = np.column_stack([x, y])
    if n > 1:
        d2d = pdist(xy)
        for r_k in RIPLEY_RADII_2D:
            cnt = int(np.sum(d2d < r_k))
            f[f"{tag}ripley_K2D_{r_k}um"] = float((area_2d / n ** 2) * 2.0 * cnt)
    else:
        for r_k in RIPLEY_RADII_2D:
            f[f"{tag}ripley_K2D_{r_k}um"] = np.nan

    # Spearman: radial distance vs. complexity score
    if "complexity" in dsb_df.columns and n > 2:
        cmap    = {"DSB": 1, "DSB+": 2, "DSB++": 3}
        cscores = np.array([cmap.get(c, np.nan) for c in dsb_df["complexity"].values])
        valid   = np.isfinite(cscores)
        if valid.sum() > 2:
            rho, _ = stats.spearmanr(r[valid], cscores[valid])
            f[f"{tag}complexity_r_spearman"] = float(rho)
        else:
            f[f"{tag}complexity_r_spearman"] = np.nan
    else:
        f[f"{tag}complexity_r_spearman"] = np.nan

    return f

# ══════════════════════════════════════════════════════════════════════════════
# MODALITY 3 — LOCAL ENERGY HETEROGENEITY
# ══════════════════════════════════════════════════════════════════════════════

def extract_modality_3_energy(
    energy_grid: Optional[np.ndarray],
    dsb_df:      pd.DataFrame,
) -> Dict[str, float]:
    """
    Spatial heterogeneity of energy deposition (60³ EnergyDeposit.csv grid).

    Gini, top-fraction and entropy metrics are LET-sensitive. DSB-energy
    enrichment tests whether DSBs co-localise with high-energy voxels.
    """
    f   = {}
    tag = "m3_"
    if energy_grid is None:
        return f

    flat    = energy_grid.flatten()
    nonzero = flat[flat > 0.0]
    if len(nonzero) == 0:
        return f

    total_E = float(nonzero.sum())
    mu_E    = float(np.mean(nonzero))
    sd_E    = float(np.std(nonzero))

    f[f"{tag}energy_mean_MeV"]  = mu_E
    f[f"{tag}energy_std_MeV"]   = sd_E
    f[f"{tag}energy_cv"]        = float(sd_E / mu_E) if mu_E > 0 else np.nan
    f[f"{tag}energy_gini"]      = gini_coefficient(nonzero)
    f[f"{tag}energy_entropy"]   = shannon_entropy(nonzero)
    f[f"{tag}energy_n_nonzero"] = float(len(nonzero))

    sorted_desc = np.sort(nonzero)[::-1]
    for frac, label in [(0.05, "top5pct"), (0.10, "top10pct"), (0.25, "top25pct")]:
        k = max(1, int(frac * len(sorted_desc)))
        f[f"{tag}energy_{label}_frac"] = float(sorted_desc[:k].sum() / total_E)

    f[f"{tag}energy_hot_spots"] = float(np.sum(nonzero > mu_E + 2.0 * sd_E))

    # DSB-voxel energy coupling
    cc = ["x_um", "y_um", "z_um"]
    if len(dsb_df) > 0 and all(c in dsb_df.columns for c in cc):
        dsb_e = np.array([
            float(energy_grid[coords_to_voxel_idx(
                float(row["x_um"]), float(row["y_um"]), float(row["z_um"])
            )])
            for _, row in dsb_df.iterrows()
        ])
        f[f"{tag}dsb_voxel_energy_mean_MeV"] = float(np.mean(dsb_e))
        f[f"{tag}dsb_voxel_energy_std_MeV"]  = float(np.std(dsb_e))
        f[f"{tag}dsb_energy_enrichment"]      = (
            float(np.mean(dsb_e) / mu_E) if mu_E > 0 else np.nan
        )
    else:
        f[f"{tag}dsb_voxel_energy_mean_MeV"] = np.nan
        f[f"{tag}dsb_voxel_energy_std_MeV"]  = np.nan
        f[f"{tag}dsb_energy_enrichment"]      = np.nan

    return f

# ══════════════════════════════════════════════════════════════════════════════
# MODALITY 4 — DOSE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_modality_4_dose(dose_grid: Optional[np.ndarray]) -> Dict[str, float]:
    """
    Dose distribution within the nucleus (60³ Dose.csv grid).

    DVH percentiles (D10, D50, D90) and CV capture macroscopic delivery
    heterogeneity. High-LET beams produce extreme D10/D90 ratios and high CV.
    """
    f   = {}
    tag = "m4_"
    if dose_grid is None:
        return f

    flat    = dose_grid.flatten()
    nonzero = flat[flat > 0.0]
    if len(nonzero) == 0:
        return f

    mu_d = float(np.mean(nonzero))
    sd_d = float(np.std(nonzero))

    f[f"{tag}dose_mean_Gy"]  = mu_d
    f[f"{tag}dose_std_Gy"]   = sd_d
    f[f"{tag}dose_cv"]       = float(sd_d / mu_d) if mu_d > 0 else np.nan
    f[f"{tag}dose_entropy"]  = shannon_entropy(nonzero)

    for pct, label in [(10, "D10"), (50, "D50"), (90, "D90")]:
        f[f"{tag}{label}_Gy"] = float(np.percentile(nonzero, 100 - pct))

    f[f"{tag}frac_above_mean"]     = float(np.mean(nonzero > mu_d))
    f[f"{tag}frac_above_mean_1sd"] = float(np.mean(nonzero > mu_d + sd_d))
    f[f"{tag}frac_above_mean_2sd"] = float(np.mean(nonzero > mu_d + 2.0 * sd_d))

    return f

# ══════════════════════════════════════════════════════════════════════════════
# MODALITY 5 — GENOMIC LOCATION
# ══════════════════════════════════════════════════════════════════════════════

def extract_modality_5_genomic(dsb_df: pd.DataFrame) -> Dict[str, float]:
    """
    Chromosomal distribution of DSBs + 1D persistent homology on genomic positions.

    Absolute position (Mbp) = chromosome_position (0–1) × CHROM_SIZES_MBP[chrom].
    centroid_bp is the midpoint within the 10 bp SDD site window and is NOT
    used here — it describes intra-cluster geometry, not chromosomal position.

    Two DSBs on the same chromosome at close genomic positions preferentially
    form deletions / inversions; 1D PH gap statistics quantify this.
    """
    f   = {}
    tag = "m5_"

    if "chromosome" not in dsb_df.columns or len(dsb_df) == 0:
        return f

    n        = len(dsb_df)
    n_chroms = 46

    # Per-chromosome DSB count distribution
    chrom_counts = dsb_df["chromosome"].value_counts()
    counts_all   = np.array([chrom_counts.get(c, 0) for c in range(1, n_chroms + 1)])

    f[f"{tag}n_chroms_with_ge1_dsb"]  = float(np.sum(counts_all >= 1))
    f[f"{tag}n_chroms_with_ge2_dsbs"] = float(np.sum(counts_all >= 2))
    f[f"{tag}max_dsbs_per_chrom"]     = float(np.max(counts_all))
    f[f"{tag}gini_per_chrom"]         = gini_coefficient(counts_all)
    f[f"{tag}dsbs_per_chrom_std"]     = float(np.std(counts_all))

    occupied = counts_all[counts_all > 0]
    f[f"{tag}dsbs_per_occupied_chrom_mean"] = (
        float(np.mean(occupied)) if len(occupied) else np.nan
    )

    # Intra-chromosomal pairwise genomic distances (Mbp)
    intra_dists_mbp: List[float] = []
    total_pairs = n * (n - 1) / 2.0

    if "chromosome_position" in dsb_df.columns:
        for chrom, grp in dsb_df.groupby("chromosome"):
            if len(grp) < 2:
                continue
            abs_pos = np.array([
                chrom_pos_to_mbp(chrom, float(p))
                for p in grp["chromosome_position"].values
            ])
            intra_dists_mbp.extend(pdist(abs_pos.reshape(-1, 1)).tolist())

    if intra_dists_mbp and total_pairs > 0:
        ida = np.array(intra_dists_mbp)
        f[f"{tag}frac_intra_chrom_pairs"] = float(len(ida) / total_pairs)
        f[f"{tag}intra_dist_mean_Mbp"]    = float(np.mean(ida))
        f[f"{tag}intra_dist_std_Mbp"]     = float(np.std(ida))
        for thresh in GENOMIC_THRESH_MBP:
            f[f"{tag}frac_intra_lt_{thresh}Mbp"] = float(np.mean(ida < thresh))
    else:
        f[f"{tag}frac_intra_chrom_pairs"] = 0.0
        f[f"{tag}intra_dist_mean_Mbp"]    = np.nan
        f[f"{tag}intra_dist_std_Mbp"]     = np.nan
        for thresh in GENOMIC_THRESH_MBP:
            f[f"{tag}frac_intra_lt_{thresh}Mbp"] = np.nan

    # 1D PH on per-chromosome sorted absolute positions
    # Gap between consecutive DSBs on a chromosome ≡ H0 bar lifetime (Mbp)
    all_gaps:      List[float] = []
    max_gaps:      List[float] = []
    n_chroms_1dph: int         = 0

    if "chromosome_position" in dsb_df.columns:
        for chrom, grp in dsb_df.groupby("chromosome"):
            if len(grp) < 2:
                continue
            abs_pos = np.sort([
                chrom_pos_to_mbp(chrom, float(p))
                for p in grp["chromosome_position"].values
            ])
            gaps = np.diff(abs_pos)
            all_gaps.extend(gaps.tolist())
            max_gaps.append(float(np.max(gaps)))
            n_chroms_1dph += 1

    f[f"{tag}ph1d_n_chroms_analysed"] = float(n_chroms_1dph)
    f[f"{tag}ph1d_mean_gap_Mbp"]      = float(np.mean(all_gaps)) if all_gaps else np.nan
    f[f"{tag}ph1d_max_gap_Mbp"]       = float(np.mean(max_gaps)) if max_gaps else np.nan
    f[f"{tag}ph1d_gini_gaps"]         = (
        gini_coefficient(np.array(all_gaps)) if all_gaps else np.nan
    )

    return f

# ══════════════════════════════════════════════════════════════════════════════
# MODALITY 6 — DAMAGE COMPLEXITY
# ══════════════════════════════════════════════════════════════════════════════

def extract_modality_6_complexity(dsb_df: pd.DataFrame) -> Dict[str, float]:
    """
    DSB / DSB+ / DSB++ breakdown and per-lesion complexity statistics.

    Higher complexity implies harder repair and higher lethality. This modality
    is LET-sensitive (more DSB++ at high LET through direct mechanisms) and
    oxygen-sensitive for low-LET radiation (fewer co-localised radical-mediated
    lesions under hypoxia reduce DSB++ fraction for electrons).
    """
    f   = {}
    tag = "m6_"

    if "complexity" not in dsb_df.columns or len(dsb_df) == 0:
        return f

    n      = len(dsb_df)
    labels = dsb_df["complexity"].values
    cmap   = {"DSB": 1, "DSB+": 2, "DSB++": 3}
    scores = np.array([cmap.get(c, np.nan) for c in labels], dtype=float)

    f[f"{tag}frac_DSB"]     = float(np.mean(labels == "DSB"))
    f[f"{tag}frac_DSBplus"] = float(np.mean(labels == "DSB+"))
    f[f"{tag}frac_DSBpp"]   = float(np.mean(labels == "DSB++"))

    f[f"{tag}complexity_score_mean"] = float(np.nanmean(scores))
    f[f"{tag}complexity_score_var"]  = float(np.nanvar(scores))

    if "n_additional_backbone" in dsb_df.columns:
        v = dsb_df["n_additional_backbone"].values.astype(float)
        f[f"{tag}n_extra_backbone_mean"] = float(np.nanmean(v))
        f[f"{tag}n_extra_backbone_std"]  = float(np.nanstd(v))
    else:
        f[f"{tag}n_extra_backbone_mean"] = np.nan
        f[f"{tag}n_extra_backbone_std"]  = np.nan

    if "n_base_damage_in_cluster" in dsb_df.columns:
        v = dsb_df["n_base_damage_in_cluster"].values.astype(float)
        f[f"{tag}n_base_damage_mean"] = float(np.nanmean(v))
        f[f"{tag}n_base_damage_std"]  = float(np.nanstd(v))
    else:
        f[f"{tag}n_base_damage_mean"] = np.nan
        f[f"{tag}n_base_damage_std"]  = np.nan

    if "has_base_damage_in_cluster" in dsb_df.columns:
        f[f"{tag}frac_has_base_damage"] = float(
            dsb_df["has_base_damage_in_cluster"].astype(float).mean()
        )
    else:
        f[f"{tag}frac_has_base_damage"] = np.nan

    # Spearman: radial distance vs. complexity score
    # Positive ρ => more complex lesions closer to track axis (expected for high LET)
    cc = ["x_um", "y_um"]
    if all(c in dsb_df.columns for c in cc) and n > 2:
        r_val = np.sqrt(dsb_df["x_um"].values ** 2 + dsb_df["y_um"].values ** 2)
        valid  = np.isfinite(scores)
        if valid.sum() > 2:
            rho, _ = stats.spearmanr(r_val[valid], scores[valid])
            f[f"{tag}complexity_r_spearman"] = float(rho)
        else:
            f[f"{tag}complexity_r_spearman"] = np.nan
    else:
        f[f"{tag}complexity_r_spearman"] = np.nan

    return f

# ══════════════════════════════════════════════════════════════════════════════
# RUN-LEVEL ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def process_run(
    run_info:     Dict,
    features_dir: Path,
) -> Optional[Dict]:
    """
    Extract m1–m6 feature vectors for one simulation run.
    Writes analysis/features/{prefix}_features.json.
    Returns the cleaned feature dict, or None on failure.
    """
    prefix  = run_info["prefix"]
    run_dir = run_info["run_dir"]
    out_path = features_dir / f"{prefix}_features.json"

    logger.info(f"  → {prefix}")

    try:
        # ── Load DSB complexity ───────────────────────────────────────────
        dsb_df = pd.read_csv(run_dir / f"{prefix}_dsb_complexity.csv")

        # ── Load optional voxel grids ─────────────────────────────────────
        energy_grid = load_voxel_grid(run_dir / f"{prefix}_EnergyDeposit.csv")
        dose_grid   = load_voxel_grid(run_dir / f"{prefix}_Dose.csv")

        # ── Build feature dict ────────────────────────────────────────────
        feats: Dict = {
            # Metadata (not used as ML features; carried through to matrix)
            "prefix":       prefix,
            "particle_key": run_info["particle_key"],
            "particle":     run_info["particle"],
            "sobp":         run_info["sobp"],
            "let":          run_info["let"],
            "dir_name":     run_info["dir_name"],
            "o2":           run_info["o2"],
            "run_id":       run_info["run_id"],
            "is_normoxic":  int(run_info["is_normoxic"]),
        }

        feats.update(extract_modality_1_spatial(dsb_df))
        feats.update(extract_modality_2_radial(dsb_df))
        feats.update(extract_modality_3_energy(energy_grid, dsb_df))
        feats.update(extract_modality_4_dose(dose_grid))
        feats.update(extract_modality_5_genomic(dsb_df))
        feats.update(extract_modality_6_complexity(dsb_df))
        # m7 Topological Summaries are computed in 02_ph_topology_analysis.py
        # and merged into the feature matrix by 04_build_feature_matrix.py.

        # ── Serialise ─────────────────────────────────────────────────────
        clean = {k: _clean(v) for k, v in feats.items()}
        with open(out_path, "w") as fh:
            json.dump(clean, fh, indent=2)

        n_dsbs  = int(feats.get("m1_n_dsbs", 0) or 0)
        n_valid = sum(1 for v in clean.values() if v is not None)
        logger.info(f"     OK  {n_dsbs:>4} DSBs | {n_valid} non-null features")
        return clean

    except Exception as exc:
        logger.error(f"     FAILED {prefix}: {exc}", exc_info=True)
        return None

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract DSB topology features (modalities 1–6).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--basedir", type=Path, default=Path("."),
        help="Project root directory containing particle subdirs (default: .).",
    )
    parser.add_argument(
        "--particle",
        choices=[c["key"] for c in PARTICLE_CONFIGS],
        metavar="PARTICLE_KEY",
        help=(
            "Restrict to one particle configuration. "
            "Choices: " + ", ".join(c["key"] for c in PARTICLE_CONFIGS)
        ),
    )
    parser.add_argument(
        "--o2", choices=O2_ORDERED, metavar="O2_LEVEL",
        help="Restrict to one O2 level (e.g. 21.0).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel worker processes (default: 1).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-extract runs even if output JSON already exists.",
    )
    args = parser.parse_args()

    base_dir     = args.basedir.resolve()
    features_dir = base_dir / "analysis" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_RIPSER:
        logger.warning(
            "ripser not installed — m1 PH features will be NaN.\n"
            "  pip install ripser"
        )

    # ── Discover runs ─────────────────────────────────────────────────────
    runs = discover_runs(base_dir, args.particle, args.o2)
    if not runs:
        logger.error(
            "No valid run directories found.\n"
            "  Check that 01_extract_dsb.py has been run and that\n"
            "  <base_dir>/<dir_name>/<dir_name>_<o2>/<dir_name>_<o2>_<run_id>/"
            "<prefix>_dsb_complexity.csv exists."
        )
        return 1

    # ── Skip already-processed runs (unless --overwrite) ─────────────────
    if not args.overwrite:
        before = len(runs)
        runs = [
            r for r in runs
            if not (features_dir / f"{r['prefix']}_features.json").exists()
        ]
        skipped = before - len(runs)
        if skipped:
            logger.info(
                f"Skipping {skipped} already-processed runs "
                f"({len(runs)} remaining). Use --overwrite to re-extract."
            )

    if not runs:
        logger.info("All runs already processed.")
        return 0

    # ── Process ───────────────────────────────────────────────────────────
    results: List[Dict] = []
    n_fail: int = 0

    if args.workers > 1 and len(runs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(process_run, r, features_dir): r
                for r in runs
            }
            for fut in as_completed(futs):
                res = fut.result()
                if res is not None:
                    results.append(res)
                else:
                    n_fail += 1
    else:
        for r in runs:
            res = process_run(r, features_dir)
            if res is not None:
                results.append(res)
            else:
                n_fail += 1

    # ── Summary ───────────────────────────────────────────────────────────
    n_ok = len(results)
    logger.info("=" * 60)
    logger.info(f"COMPLETE  {n_ok} succeeded, {n_fail} failed.")
    if n_ok > 0:
        meta_keys = {"prefix", "particle_key", "particle", "sobp",
                     "let", "dir_name", "o2", "run_id", "is_normoxic"}
        n_feats = len([k for k in results[0] if k not in meta_keys])
        logger.info(f"  m1–m6 features per run : {n_feats}")
        logger.info(f"  Output                 : {features_dir}")
        logger.info(
            "  Next step             : run 04_build_feature_matrix.py "
            "to merge m1–m6 with m7 from 02_ph_topology_analysis.py"
        )
    logger.info("=" * 60)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
