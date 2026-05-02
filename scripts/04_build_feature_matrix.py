#!/usr/bin/env python3
"""
================================================================================
04_build_feature_matrix.py
================================================================================
Merge per-run m1–m6 feature JSONs (from 03_compute_features.py) with the m7
Topological Summary features (from 02_ph_topology_analysis.py) into a single
tidy feature matrix for downstream random-forest classification.

PIPELINE POSITION
-----------------
  01  extract_dsb.py
  02  ph_topology_analysis.py    →  analysis/ph/m7_topological_features.json
  03  compute_features.py        →  analysis/features/{prefix}_features.json
  04  build_feature_matrix.py    →  analysis/feature_matrix.csv           ← THIS
                                    analysis/feature_matrix_imputed.csv
                                    analysis/feature_metadata.json
                                    analysis/feature_matrix_summary.csv
  05  random_forest.py
  06  additional_analyses.py
  07  regenerate_figures.py

NOTE: Script 00 (parse_sdd_particle_history.py) is no longer part of the
pipeline. It existed solely for the old Event Attribution modality (m7).
With that modality retired, script 00 can be deleted.

MERGE LOGIC
-----------
  1. Load all analysis/features/{prefix}_features.json  → m1–m6 per run
  2. Load analysis/ph/m7_topological_features.json      → 10 m7 features per run
  3. Join on "prefix"  (e.g. "carbon_40.9_21.0_01")
  4. Runs with no matching m7 entry get NaN for all m7 columns (warning emitted).
  5. Column order: meta | m1 | m2 | m3 | m4 | m5 | m6 | m7

STUDY DESIGN
------------
  7 particle configurations × 7 O2 levels × 50 runs = 2,450 rows (full corpus)
  97 m1–m6 features + 10 m7 features = 107 features per run
  Meta columns (not used as features): prefix, particle_key, particle, sobp,
                                        let, dir_name, o2, run_id, is_normoxic

  Prefix format: {dir_name}_{o2}_{run_id}
  Example:       carbon_40.9_21.0_01
  This matches the naming convention used by 01_extract_dsb.py,
  02_ph_topology_analysis.py, and 03_compute_features.py.

O2 LEVELS (7)
-------------
  21.0%   Atmospheric normoxia
   5.0%   Tumour normoxia
   2.1%   Mild hypoxia
   0.5%   Moderate hypoxia
   0.1%   Severe hypoxia / HIF-1α threshold
   0.021% Radiobiological anoxia
   0.005% True anoxia (maximum OER)

OUTPUTS
-------
  analysis/
    feature_matrix.csv          tidy matrix (up to 2,450 × ~108 cols)
    feature_matrix_imputed.csv  NaN-imputed version (only if --impute)
    feature_metadata.json       column names grouped by modality
    feature_matrix_summary.csv  per-condition descriptive statistics

USAGE
-----
  python 04_build_feature_matrix.py
  python 04_build_feature_matrix.py --basedir /path/to/project
  python 04_build_feature_matrix.py --impute
  python 04_build_feature_matrix.py --no-m7
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# Keep in sync with 02_ph_topology_analysis.py and 03_compute_features.py.
# ══════════════════════════════════════════════════════════════════════════════

# ── Particle key ordering (LET-ascending) ────────────────────────────────────
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

# ── O2 levels (ordered normoxic → anoxic) ────────────────────────────────────
O2_ORDERED: List[str] = [
    "21.0", "5.0", "2.1", "0.5", "0.1", "0.021", "0.005"
]

O2_LABELS: Dict[str, str] = {
    "21.0":  "Atmospheric normoxia (21.0%)",
    "5.0":   "Tumour normoxia (5.0%)",
    "2.1":   "Mild hypoxia (2.1%)",
    "0.5":   "Moderate hypoxia (0.5%)",
    "0.1":   "Severe hypoxia / HIF-1α (0.1%)",
    "0.021": "Radiobiological anoxia (0.021%)",
    "0.005": "True anoxia (0.005%)",
}

# ── Expected runs per condition ───────────────────────────────────────────────
N_RUNS_EXPECTED: int = 50

# ── Modality tags ─────────────────────────────────────────────────────────────
MODALITY_TAGS: Dict[str, str] = {
    "m1_": "Spatial Distribution",
    "m2_": "Radial Track Structure",
    "m3_": "Local Energy Heterogeneity",
    "m4_": "Dose Distribution",
    "m5_": "Genomic Location",
    "m6_": "Damage Complexity",
    "m7_": "Topological Summaries",
}

# ── Metadata columns (carried through to matrix but not used as features) ─────
META_COLS: List[str] = [
    "prefix", "particle_key", "particle", "sobp",
    "let", "dir_name", "o2", "run_id", "is_normoxic",
]

# ══════════════════════════════════════════════════════════════════════════════
# LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_m1_to_m6(features_dir: Path) -> pd.DataFrame:
    """
    Read every *_features.json under features_dir into a single DataFrame.
    Each file corresponds to one run and contains m1–m6 features plus metadata.
    """
    json_files = sorted(features_dir.glob("*_features.json"))
    if not json_files:
        logger.error(f"No feature JSON files found in: {features_dir}\n"
                    "  Run 03_compute_features.py first. Expected files named\n"
                    "  {dir_name}_{o2}_{run_id}_features.json, "
                    "e.g. carbon_40.9_21.0_01_features.json")
        sys.exit(1)

    logger.info(f"Loading m1–m6: found {len(json_files)} feature files.")
    records: List[Dict] = []
    n_failed = 0
    for jf in json_files:
        try:
            with open(jf) as fh:
                records.append(json.load(fh))
        except Exception as exc:
            logger.warning(f"  Could not read {jf.name}: {exc}")
            n_failed += 1

    if not records:
        logger.error("All feature files failed to load — aborting.")
        sys.exit(1)

    df = pd.DataFrame(records)
    if n_failed:
        logger.warning(f"  {n_failed} file(s) could not be read.")
    logger.info(f"  Loaded {len(df)} rows × {len(df.columns)} raw columns.")
    return df


def load_m7(m7_path: Path) -> Optional[Dict[str, Dict]]:
    """
    Load the m7 Topological Summary features produced by
    02_ph_topology_analysis.py.

    Returns a dict keyed by prefix (e.g. "carbon_40.9_21.0_01"),
    or None if the file is absent.
    """
    if not m7_path.exists():
        logger.warning(
            f"m7 feature file not found: {m7_path}\n"
            "  Run 02_ph_topology_analysis.py first, or pass --no-m7 to\n"
            "  build the matrix with m1–m6 only."
        )
        return None

    try:
        with open(m7_path) as fh:
            data = json.load(fh)
        logger.info(
            f"Loaded m7 features: {m7_path.name}  "
            f"({len(data)} entries)"
        )
        return data
    except Exception as exc:
        logger.error(f"Could not read {m7_path}: {exc}")
        return None


def merge_m7(df: pd.DataFrame, m7_data: Dict[str, Dict]) -> pd.DataFrame:
    """
    Join the m7 Topological Summary features into df on the "prefix" column.

    Runs present in df but absent from m7_data receive NaN for all m7 columns.
    A warning is emitted listing any missing prefixes so the user can diagnose
    whether 02 was run on a partial corpus.
    """
    if "prefix" not in df.columns:
        logger.error(
            '"prefix" column missing from feature DataFrame — '
            "cannot join m7 features."
        )
        return df

    # Discover all m7 column names from any entry in the dict
    m7_cols: List[str] = []
    for entry in m7_data.values():
        m7_cols = sorted(entry.keys())
        break

    # Build m7 rows aligned to df
    m7_rows: List[Dict] = []
    missing: List[str]  = []
    for prefix in df["prefix"]:
        if prefix in m7_data:
            m7_rows.append(m7_data[prefix])
        else:
            missing.append(prefix)
            m7_rows.append({col: np.nan for col in m7_cols})

    if missing:
        logger.warning(
            f"  {len(missing)} run(s) have no m7 entry "
            f"(NaN substituted). First 5: {missing[:5]}"
        )

    m7_df = pd.DataFrame(m7_rows, index=df.index)
    merged = pd.concat([df, m7_df], axis=1)

    n_joined = len(df) - len(missing)
    logger.info(
        f"  m7 merge: {n_joined}/{len(df)} runs matched "
        f"({len(m7_cols)} m7 columns added)."
    )
    return merged

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN ORDERING
# ══════════════════════════════════════════════════════════════════════════════

def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce column order: meta | m1 | m2 | m3 | m4 | m5 | m6 | m7 | other.
    Columns not matching any modality prefix come last.
    """
    meta   = [c for c in META_COLS if c in df.columns]
    feat   = [c for c in df.columns if c not in meta]

    ordered_feat: List[str] = []
    for tag in MODALITY_TAGS:
        ordered_feat.extend(sorted(c for c in feat if c.startswith(tag)))

    other = [c for c in feat if not any(c.startswith(t) for t in MODALITY_TAGS)]
    if other:
        logger.warning(f"  {len(other)} column(s) not assigned to any modality: {other[:5]}")
        ordered_feat.extend(sorted(other))

    return df[meta + ordered_feat]

# ══════════════════════════════════════════════════════════════════════════════
# SORTING
# ══════════════════════════════════════════════════════════════════════════════

def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort rows by particle_key (LET-ascending) → O2 (normoxic → anoxic) → run_id.
    Falls back to particle/o2 string sort if particle_key column is absent.
    """
    df = df.copy()

    if "particle_key" in df.columns:
        pk_order = {pk: i for i, pk in enumerate(PARTICLE_KEY_ORDER)}
        df["_pk_ord"] = df["particle_key"].map(pk_order).fillna(999)
    else:
        logger.warning(
            '"particle_key" column missing — sorting by "particle" string.'
        )
        df["_pk_ord"] = df.get("particle", pd.Series(dtype=str))

    if "o2" in df.columns:
        o2_order = {o: i for i, o in enumerate(O2_ORDERED)}
        df["_o2_ord"] = df["o2"].map(o2_order).fillna(999)
    else:
        df["_o2_ord"] = 0

    df = df.sort_values(["_pk_ord", "_o2_ord", "run_id"])
    df = df.drop(columns=["_pk_ord", "_o2_ord"])
    return df.reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE METADATA
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_metadata(
    df: pd.DataFrame,
    feature_cols: List[str],
    meta_cols_used: List[str],
) -> Dict:
    """
    Produce a JSON-serialisable metadata dict grouping feature columns by
    modality, with per-modality NaN fractions and the full study design summary.
    """
    modality_info: Dict = {}
    for tag, name in MODALITY_TAGS.items():
        cols     = [c for c in feature_cols if c.startswith(tag)]
        nan_frac = float(df[cols].isna().values.mean()) if cols else 0.0
        modality_info[name] = {
            "prefix":     tag,
            "n_features": len(cols),
            "nan_frac":   round(nan_frac, 4),
            "features":   cols,
        }

    unassigned = [c for c in feature_cols
                  if not any(c.startswith(t) for t in MODALITY_TAGS)]

    # Condition summary
    n_conditions = 0
    conditions_found: List[Dict] = []
    if "particle_key" in df.columns and "o2" in df.columns:
        for (pk, o2), grp in df.groupby(["particle_key", "o2"]):
            n_conditions += 1
            conditions_found.append({
                "particle_key": pk,
                "o2":           o2,
                "n_runs":       int(len(grp)),
            })

    meta: Dict = {
        "total_features":  len(feature_cols),
        "n_runs":          int(len(df)),
        "n_conditions":    n_conditions,
        "o2_levels":       O2_ORDERED,
        "particle_keys":   PARTICLE_KEY_ORDER,
        "meta_columns":    meta_cols_used,
        "modalities":      modality_info,
        "conditions":      conditions_found,
    }

    if unassigned:
        meta["unassigned_features"] = unassigned

    return meta

# ══════════════════════════════════════════════════════════════════════════════
# IMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def mean_impute_within_condition(
    df:           pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Replace NaN values with the within-condition mean
    (condition = particle_key × o2).
    Any remaining NaN (whole condition missing) is replaced with the
    overall column mean. Imputation fractions are reported per modality.
    """
    df = df.copy()
    group_cols = ["particle_key", "o2"] if "particle_key" in df.columns \
                 else ["particle", "o2"]

    for tag, name in MODALITY_TAGS.items():
        mod_cols = [c for c in feature_cols if c.startswith(tag)]
        if not mod_cols:
            continue

        n_before = int(df[mod_cols].isna().sum().sum())
        if n_before == 0:
            continue

        df[mod_cols] = (
            df.groupby(group_cols)[mod_cols]
            .transform(lambda x: x.fillna(x.mean()))
        )
        # Global fallback for conditions with all-NaN
        df[mod_cols] = df[mod_cols].fillna(df[mod_cols].mean())

        n_after    = int(df[mod_cols].isna().sum().sum())
        total_vals = df[mod_cols].size
        logger.info(
            f"  {name:34s}: {n_before:>5} NaN → {n_after:>3}  "
            f"({n_before / total_vals * 100:.1f}% → "
            f"{n_after / total_vals * 100:.1f}%)"
        )

    return df

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Per-condition (particle_key × o2) descriptive statistics for every feature.
    Returns a tidy DataFrame with one row per (condition, feature).
    """
    group_cols = ["particle_key", "o2"] if "particle_key" in df.columns \
                 else ["particle", "o2"]

    rows: List[Dict] = []
    for group_vals, grp in df.groupby(group_cols):
        if len(group_cols) == 2:
            pk, o2 = group_vals
        else:
            pk, o2 = group_vals[0], group_vals[1]

        for col in feature_cols:
            vals = grp[col].dropna().values.astype(float)
            if len(vals) == 0:
                continue
            rows.append({
                "particle_key": pk,
                "o2":           o2,
                "o2_label":     O2_LABELS.get(str(o2), str(o2)),
                "particle_label": PARTICLE_LABELS.get(str(pk), str(pk)),
                "feature":      col,
                "modality":     next(
                    (name for tag, name in MODALITY_TAGS.items()
                     if col.startswith(tag)),
                    "Unassigned",
                ),
                "mean":    float(np.mean(vals)),
                "std":     float(np.std(vals)),
                "median":  float(np.median(vals)),
                "q25":     float(np.percentile(vals, 25)),
                "q75":     float(np.percentile(vals, 75)),
                "n_valid": int(len(vals)),
            })

    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

def validate_matrix(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """
    Print a validation report: run counts per condition, NaN fractions,
    and overall completeness.
    """
    logger.info("=" * 64)
    logger.info("FEATURE MATRIX VALIDATION REPORT")
    logger.info("=" * 64)

    # ── Condition run counts ──────────────────────────────────────────────────
    logger.info(
        f"  {'Particle key':<22} {'O2 %':>7}   {'Runs':>5}   Status"
    )
    logger.info(f"  {'-'*22} {'-'*7}   {'-'*5}   {'-'*20}")

    group_cols = ["particle_key", "o2"] if "particle_key" in df.columns \
                 else ["particle", "o2"]

    # Produce counts in canonical order
    all_particle_keys = [pk for pk in PARTICLE_KEY_ORDER
                         if pk in df.get("particle_key", pd.Series()).unique()]
    if not all_particle_keys and "particle" in df.columns:
        all_particle_keys = sorted(df["particle"].unique())

    total_runs = 0
    n_short    = 0
    for pk in all_particle_keys:
        col = group_cols[0]
        for o2 in O2_ORDERED:
            mask = (df[col] == pk) & (df["o2"] == o2)
            n    = int(mask.sum())
            if n == 0:
                continue
            total_runs += n
            flag = "✓" if n == N_RUNS_EXPECTED else f"⚠  expected {N_RUNS_EXPECTED}"
            if n != N_RUNS_EXPECTED:
                n_short += 1
            logger.info(f"  {pk:<22} {o2:>7}%   {n:>5}   {flag}")

    logger.info(f"\n  Total runs loaded : {total_runs}")
    n_cond = df.groupby(group_cols).ngroups
    logger.info(f"  Conditions found  : {n_cond}  (expected {len(PARTICLE_KEY_ORDER) * len(O2_ORDERED)})")
    if n_short:
        logger.warning(
            f"  {n_short} condition(s) have fewer than {N_RUNS_EXPECTED} runs."
        )

    # ── NaN fractions per modality ────────────────────────────────────────────
    logger.info("")
    logger.info(f"  {'Modality':<34}   {'NaN %':>6}")
    logger.info(f"  {'-'*34}   {'-'*6}")
    for tag, name in MODALITY_TAGS.items():
        mod_cols = [c for c in feature_cols if c.startswith(tag)]
        if not mod_cols:
            continue
        nan_pct = df[mod_cols].isna().values.mean() * 100
        flag    = "  ← missing (run 02/03 first?)" if nan_pct == 100.0 else ""
        logger.info(f"  {name:<34}   {nan_pct:>5.1f}%{flag}")

    total_nan = df[feature_cols].isna().values.mean() * 100
    logger.info(f"\n  Overall NaN : {total_nan:.1f}%")
    logger.info("=" * 64)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge m1–m6 feature JSONs (03) with m7 Topological Summary "
            "features (02) into a single feature matrix."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--basedir", type=Path, default=Path("."),
        help="Project root directory (default: current directory).",
    )
    parser.add_argument(
        "--impute", action="store_true",
        help=(
            "Mean-impute NaN values within condition (particle_key × o2). "
            "The raw NaN matrix is always saved before imputation."
        ),
    )
    parser.add_argument(
        "--no-m7", action="store_true", dest="no_m7",
        help=(
            "Build the matrix with m1–m6 only, skipping the m7 merge. "
            "Useful if 02_ph_topology_analysis.py has not yet been run."
        ),
    )
    args = parser.parse_args()

    base_dir    = args.basedir.resolve()
    analysis_dir = base_dir / "analysis"
    features_dir = analysis_dir / "features"
    ph_dir       = analysis_dir / "ph"
    m7_path      = ph_dir / "m7_topological_features.json"

    analysis_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 64)
    logger.info("04_build_feature_matrix.py")
    logger.info(f"  Base dir     : {base_dir}")
    logger.info(f"  Features dir : {features_dir}")
    logger.info(f"  m7 path      : {m7_path}")
    logger.info(f"  Impute       : {args.impute}")
    logger.info(f"  Include m7   : {not args.no_m7}")
    logger.info(
        f"  Design       : {len(PARTICLE_KEY_ORDER)} particles × "
        f"{len(O2_ORDERED)} O2 levels × {N_RUNS_EXPECTED} runs "
        f"= {len(PARTICLE_KEY_ORDER) * len(O2_ORDERED) * N_RUNS_EXPECTED:,} max rows"
    )
    logger.info("=" * 64)

    # ── 1. Load m1–m6 ─────────────────────────────────────────────────────────
    if not features_dir.exists():
        logger.error(
            f"Features directory not found: {features_dir}\n"
            "  Run 03_compute_features.py first."
        )
        return 1

    df = load_m1_to_m6(features_dir)

    # ── 2. Load and merge m7 ──────────────────────────────────────────────────
    if not args.no_m7:
        m7_data = load_m7(m7_path)
        if m7_data is not None:
            df = merge_m7(df, m7_data)
        else:
            logger.warning(
                "Proceeding without m7 features. "
                "Pass --no-m7 to suppress this warning."
            )
    else:
        logger.info("--no-m7 flag set: skipping m7 merge.")

    # ── 3. Sort and order columns ─────────────────────────────────────────────
    df = sort_dataframe(df)
    df = order_columns(df)

    # ── 4. Separate meta from feature columns ─────────────────────────────────
    meta_cols_used = [c for c in META_COLS if c in df.columns]
    feature_cols   = [c for c in df.columns if c not in meta_cols_used]

    # Coerce all feature columns to numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 5. Validate ───────────────────────────────────────────────────────────
    validate_matrix(df, feature_cols)

    # ── 6. Save raw matrix ────────────────────────────────────────────────────
    raw_path = analysis_dir / "feature_matrix.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Raw feature matrix saved: {raw_path}")
    logger.info(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns "
                f"({len(meta_cols_used)} meta + {len(feature_cols)} features)")

    # ── 7. Feature metadata ───────────────────────────────────────────────────
    feat_meta = build_feature_metadata(df, feature_cols, meta_cols_used)

    logger.info("\nFeature counts by modality:")
    for mod_name, mod_info in feat_meta["modalities"].items():
        if mod_info["n_features"] == 0:
            continue
        logger.info(
            f"  {mod_name:<34}: {mod_info['n_features']:>3} features  "
            f"({mod_info['nan_frac']*100:.1f}% NaN)"
        )
    logger.info(f"  {'TOTAL':<34}: {feat_meta['total_features']:>3} features")

    meta_path = analysis_dir / "feature_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(feat_meta, fh, indent=2)
    logger.info(f"\nFeature metadata saved: {meta_path}")

    # ── 8. Optional imputation ────────────────────────────────────────────────
    if args.impute:
        logger.info("\nImputing NaN values within condition (particle_key × o2)…")
        df_imp = mean_impute_within_condition(df, feature_cols)
        imp_path = analysis_dir / "feature_matrix_imputed.csv"
        df_imp.to_csv(imp_path, index=False)
        logger.info(f"Imputed matrix saved: {imp_path}")

    # ── 9. Per-condition summary statistics ───────────────────────────────────
    logger.info("\nComputing per-condition summary statistics…")
    summary = build_summary(df, feature_cols)
    summary_path = analysis_dir / "feature_matrix_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary statistics saved: {summary_path}")
    logger.info(
        f"  ({len(summary):,} rows: {len(feature_cols)} features × "
        f"{feat_meta['n_conditions']} conditions)"
    )

    # ── Final status ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 64)
    logger.info("COMPLETE")
    logger.info(f"  feature_matrix.csv  →  {df.shape[0]} runs × "
                f"{len(feature_cols)} features")
    logger.info(
        "  Next step: run 05_random_forest.py"
    )
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
