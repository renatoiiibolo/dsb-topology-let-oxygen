#!/usr/bin/env python3
"""
================================================================================
BATCH GENERATE HYPOXIC DATASETS
================================================================================

This script batch-processes multiple TOPAS-nBio runs to generate hypoxic
datasets at specified oxygen levels.

UPDATED for explicit LET in directory structure:
  - Old: carbon_21.0_01 (LET implicit)
  - New: carbon_40.9_21.0_01 (LET explicit)

CRITICAL: For nested subset property (lower O2 ⊂ higher O2), the random seed
must be IDENTICAL across all O2 levels for the same run. This is automatically
handled when --seed is NOT specified (uses prefix-based seeding in the 
underlying generate_hypoxic_dataset.py script).

USAGE
-----
# Process all carbon runs (LET=40.9) at severe hypoxia
python batch_generate_hypoxic.py --particle carbon --let 40.9 --runs 1-50 --o2levels 0.021

# Process specific proton runs at multiple O2 levels (nested subsets)
python batch_generate_hypoxic.py --particle proton --let 4.6 --runs 1,5,10 --o2levels 0.021,0.21,2.1

# All electron runs with explicit seed
python batch_generate_hypoxic.py --particle electron --let 2.0 --runs 1-50 --o2levels 0.001 --seed 42

# Save batch summary
python batch_generate_hypoxic.py --particle carbon --let 70.7 --runs 1-10 --o2levels 0.021,0.21 --save-summary summary.json

================================================================================
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

from generate_hypoxic_dataset import generate_hypoxic_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_run_range(run_spec: str) -> List[int]:
    runs = []
    for part in run_spec.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            runs.extend(range(int(start), int(end) + 1))
        else:
            runs.append(int(part))
    return sorted(set(runs))

def parse_o2_levels(o2_spec: str) -> List[float]:
    return [float(x.strip()) for x in o2_spec.split(',')]

def batch_generate(
    particle: str,
    let_value: str,
    normoxic_o2: str,
    runs: List[int],
    target_o2_levels: List[float],
    base_dir: Path = Path('.'),
    neighborhood_size: int = 2,
    global_seed: int = None,
    save_summary: str = None
) -> Tuple[int, int, Dict[str, Any]]:
    
    n_total = len(runs) * len(target_o2_levels)
    n_success = 0
    failed_runs = []  # Explicitly track failed configurations
    successful_items = []
    
    logger.info("=" * 70)
    logger.info("BATCH HYPOXIC DATASET GENERATION")
    logger.info("=" * 70)
    logger.info(f"Particle: {particle}")
    logger.info(f"LET: {let_value} keV/μm")
    logger.info(f"Normoxic O2: {normoxic_o2}%")
    logger.info(f"Runs: {len(runs)} (e.g., {runs[0]}-{runs[-1]})")
    logger.info(f"Target O2 levels: {target_o2_levels}")
    logger.info(f"Total combinations: {n_total}")
    logger.info(f"Neighborhood size: {neighborhood_size}×{neighborhood_size}×{neighborhood_size}")
    
    if global_seed is None:
        logger.info("Seed: Auto-generate per run (prefix-based, ensuring nested O2 subsets)")
    else:
        logger.info(f"Seed: {global_seed} (Global seed applied to ALL runs)")
        
    logger.info("=" * 70)
    
    for idx, (run_num, o2_level) in enumerate([(r, o) for r in runs for o in target_o2_levels], 1):
        # Build 4-part prefix keeping letting let and normoxic_o2 strictly as strings
        prefix = f"{particle}_{let_value}_{normoxic_o2}_{run_num:02d}"
        
        logger.info(f"\n[{idx}/{n_total}] Processing {prefix} → {o2_level}% O2...")
        
        try:
            summary = generate_hypoxic_dataset(
                prefix=prefix,
                o2_level=o2_level,
                base_dir=base_dir,
                neighborhood_size=neighborhood_size,
                random_seed=global_seed
            )
            n_success += 1
            successful_items.append(summary)
            
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            failed_runs.append(f"{prefix} → {o2_level}% O2")

    n_failed = len(failed_runs)

    # ----------------------------------------------------------------------
    # NEW: Final Summary Statistics Block
    # ----------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total combinations attempted : {n_total}")
    logger.info(f"Successful combinations      : {n_success}/{n_total}")
    logger.info(f"Failed combinations          : {n_failed}/{n_total}")
    
    if failed_runs:
        logger.info("-" * 70)
        logger.info("FAILED RUNS LIST:")
        for fail in failed_runs:
            logger.info(f"  - {fail}")
    logger.info("=" * 70)
    # ----------------------------------------------------------------------

    batch_summary_data = {
        'timestamp': datetime.now().isoformat(),
        'particle': particle,
        'LET': let_value,
        'normoxic_o2': normoxic_o2,
        'runs_attempted': n_total,
        'runs_successful': n_success,
        'runs_failed': n_failed,
        'failed_runs_list': failed_runs,
        'successful_items': successful_items
    }
    
    if save_summary:
        with open(save_summary, 'w') as f:
            json.dump(batch_summary_data, f, indent=2)
        logger.info(f"Saved batch summary to {save_summary}")

    return n_success, n_failed, batch_summary_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch generate hypoxic datasets'
    )
    
    parser.add_argument(
        '--particle',
        type=str,
        required=True,
        choices=['electron', 'proton', 'carbon','helium'],
        help='Particle type'
    )
    
    parser.add_argument(
        '--let',
        type=str,
        required=True,
        help='LET value as string (e.g. 40.9)'
    )
    
    parser.add_argument(
        '--normoxic_o2',
        type=str,
        default='21.0',
        help='Normoxic O2 string (default: 21.0)'
    )
    
    parser.add_argument(
        '--runs',
        type=str,
        required=True,
        help='Run numbers (e.g. 1-50 or 1,5,10)'
    )
    
    parser.add_argument(
        '--o2levels',
        type=str,
        required=True,
        help='Target O2 levels in %% (e.g. 0.005,0.021,21.0)'
    )
    
    parser.add_argument(
        '--basedir',
        type=str,
        default='.',
        help='Base directory containing particle folders'
    )
    
    parser.add_argument(
        '--neighborhood',
        type=int,
        default=2,
        help='Size of voxel neighborhood for local energy computation'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Global random seed (Not recommended for nested subset generation)'
    )
    
    parser.add_argument(
        '--save-summary',
        type=str,
        default=None,
        help='Path to save JSON summary of the batch process'
    )
    
    args = parser.parse_args()
    
    # Parse O2 levels
    try:
        o2_levels = parse_o2_levels(args.o2levels)
    except Exception as e:
        logger.error(f"Failed to parse O2 levels: {e}")
        sys.exit(1)
        
    # Parse runs
    try:
        runs = parse_run_range(args.runs)
    except Exception as e:
        logger.error(f"Failed to parse run specification: {e}")
        sys.exit(1)
    
    if not runs:
        logger.error("No runs specified")
        sys.exit(1)
    
    # Warn about --seed usage
    if args.seed is not None and len(o2_levels) > 1:
        logger.warning("=" * 70)
        logger.warning("⚠️  WARNING: Using --seed with multiple O2 levels")
        logger.warning("   This will make the SAME DSBs appear in all runs!")
        logger.warning("   For nested subsets, remove --seed and use auto-seeding instead.")
        logger.warning("=" * 70)
        response = input("\nContinue anyway? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Aborted by user")
            sys.exit(1)
    
    # Run batch processing
    try:
        n_success, n_failed, batch_summary = batch_generate(
            particle=args.particle,
            let_value=args.let,
            normoxic_o2=args.normoxic_o2,
            runs=runs,
            target_o2_levels=o2_levels,
            base_dir=Path(args.basedir),
            neighborhood_size=args.neighborhood,
            global_seed=args.seed,
            save_summary=args.save_summary
        )
        
        if n_failed > 0:
            logger.warning(f"{n_failed} datasets failed to generate")
            sys.exit(1)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        sys.exit(1)