#!/usr/bin/env python3
"""
================================================================================
GENERATE HYPOXIC DATASETS FROM TOPAS-nBIO NORMOXIC SIMULATIONS
================================================================================

This script processes TOPAS-nBio normoxic DNA damage simulations and generates
hypoxic datasets using the Voxel-Aware Oxygen Model (VOxA v1.0).

UPDATED for explicit LET in directory structure:
  - Old: carbon_21.0_01 (LET implicit in parent folder)
  - New: carbon_40.9_21.0_01 (LET explicit in all paths)

USAGE
-----
python generate_hypoxic_dataset.py --prefix carbon_40.9_21.0_01 --o2level 0.021

The script:
1. Reads normoxic DSB complexity data
2. Reads energy deposit voxel grid
3. Computes local energy in 2×2×2 voxel neighborhoods
4. Applies VOxA model to compute P_DSB for each DSB
5. Stochastically samples DSBs based on retention probability
6. Outputs hypoxic dataset files

OUTPUT FILES
------------
For carbon_40.9_21.0_01 at 0.021% O2, creates:
- carbon_40.9/carbon_40.9_0.021/carbon_40.9_0.021_01/carbon_40.9_0.021_01_dsb_complexity.csv
- carbon_40.9/carbon_40.9_0.021/carbon_40.9_0.021_01/carbon_40.9_0.021_01_EnergyDeposit.csv
- carbon_40.9/carbon_40.9_0.021/carbon_40.9_0.021_01/carbon_40.9_0.021_01_Dose.csv
- carbon_40.9/carbon_40.9_0.021/carbon_40.9_0.021_01/carbon_40.9_0.021_01_complexity_summary.json

DIRECTORY STRUCTURE
------------------
Input (normoxic):
  carbon_40.9/
    carbon_40.9_21.0/
      carbon_40.9_21.0_01/
        carbon_40.9_21.0_01_dsb_complexity.csv
        carbon_40.9_21.0_01_EnergyDeposit.csv
        ...
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Import VOxA model
from unified_voxel_aware_oxygen_model_updated import UnifiedVoxelAwareOxygenModel

# Atomic number lookup for all supported particles.
# Used by compute_P_DSB_unseen() to select δf and x50 via Z-interpolation.
PARTICLE_Z: dict = {
    "electron": 0,
    "photon":   0,
    "proton":   1,
    "deuteron": 1,
    "helium":   2,
    "carbon":   6,
    "nitrogen": 7,
    "neon":    10,
    "argon":   18,
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoxelGrid:
    def __init__(self):
        self.n_bins = 60
        self.half_width = 9.28055
        self.voxel_size = (2 * self.half_width) / self.n_bins
        self.x_min, self.x_max = -self.half_width, self.half_width
        self.y_min, self.y_max = -self.half_width, self.half_width
        self.z_min, self.z_max = -self.half_width, self.half_width
    
    def position_to_voxel(self, x_um: float, y_um: float, z_um: float) -> Tuple[int, int, int]:
        ix = int(np.floor((x_um - self.x_min) / self.voxel_size))
        iy = int(np.floor((y_um - self.y_min) / self.voxel_size))
        iz = int(np.floor((z_um - self.z_min) / self.voxel_size))
        
        ix = np.clip(ix, 0, self.n_bins - 1)
        iy = np.clip(iy, 0, self.n_bins - 1)
        iz = np.clip(iz, 0, self.n_bins - 1)
        return (ix, iy, iz)
    
    def get_neighborhood_indices(self, ix: int, iy: int, iz: int, neighborhood_size: int = 2) -> List[Tuple[int, int, int]]:
        indices = []
        half = neighborhood_size // 2
        for dx in range(neighborhood_size):
            for dy in range(neighborhood_size):
                for dz in range(neighborhood_size):
                    nx = ix - half + dx
                    ny = iy - half + dy
                    nz = iz - half + dz
                    if (0 <= nx < self.n_bins and 0 <= ny < self.n_bins and 0 <= nz < self.n_bins):
                        indices.append((nx, ny, nz))
        return indices

def load_energy_grid(filepath: Path) -> np.ndarray:
    """Load energy deposit grid, handling TOPAS metadata safely."""
    logger.info(f"Loading energy grid from: {filepath}")
    grid = np.zeros((60, 60, 60), dtype=np.float32)
    
    # Read CSV, skipping TOPAS metadata comments starting with '#'
    df = pd.read_csv(
        filepath, 
        comment='#', 
        header=None, 
        names=['ix', 'iy', 'iz', 'energy_MeV']
    )
    
    # Clean up any residual non-numeric headers
    df['ix'] = pd.to_numeric(df['ix'], errors='coerce')
    df = df.dropna()
    
    for _, row in df.iterrows():
        ix, iy, iz = int(row['ix']), int(row['iy']), int(row['iz'])
        energy = float(row['energy_MeV'])
        grid[ix, iy, iz] = energy
        
    logger.info(f"Loaded energy grid: {grid.shape}, total energy = {np.sum(grid):.6f} MeV")
    return grid

def compute_local_energies(dsb_df: pd.DataFrame, energy_grid: np.ndarray, voxel_grid: VoxelGrid, neighborhood_size: int = 2) -> np.ndarray:
    local_energies = np.zeros(len(dsb_df))
    for idx, row in dsb_df.iterrows():
        ix, iy, iz = voxel_grid.position_to_voxel(row['x_um'], row['y_um'], row['z_um'])
        neighbor_indices = voxel_grid.get_neighborhood_indices(ix, iy, iz, neighborhood_size)
        local_energy = sum(energy_grid[nx, ny, nz] for (nx, ny, nz) in neighbor_indices)
        local_energies[idx] = local_energy
    return local_energies

def compute_energy_zscores(local_energies: np.ndarray) -> np.ndarray:
    mean_energy = np.mean(local_energies)
    std_energy = np.std(local_energies)
    if std_energy == 0:
        return np.zeros_like(local_energies)
    return (local_energies - mean_energy) / std_energy

def generate_hypoxic_dataset(
    prefix: str, 
    o2_level: float, 
    base_dir: Path = Path('.'), 
    neighborhood_size: int = 2, 
    random_seed: Optional[int] = None
) -> Dict:
    
    # Parse the 4-part explicit prefix
    parts = prefix.split('_')
    if len(parts) < 4:
        raise ValueError(f"Invalid prefix format: {prefix}. Expected format: particle_LET_normoxicO2_runID")
    
    particle = parts[0]
    let_str = parts[1]
    normoxic_o2_str = parts[2]
    run_id = parts[3]
    
    logger.info("=" * 70)
    logger.info("GENERATING HYPOXIC DATASET")
    logger.info("=" * 70)
    logger.info(f"Prefix: {prefix}")
    logger.info(f"Particle: {particle}")
    logger.info(f"LET: {let_str} keV/μm")
    logger.info(f"Input O2: {normoxic_o2_str}% (normoxic)")
    logger.info(f"Target O2: {o2_level}%")
    logger.info(f"Run ID: {run_id}")
    
    if random_seed is None:
        random_seed = hash(f"{prefix}") % (2**32)
        logger.info(f"Random seed: {random_seed} (prefix-based for nested subsets)")
    else:
        logger.info(f"Random seed: {random_seed} (user-provided)")
        
    np.random.seed(random_seed)
    
    particle_dir = base_dir / f"{particle}_{let_str}"
    normoxic_dir = particle_dir / f"{particle}_{let_str}_{normoxic_o2_str}"
    run_dir = normoxic_dir / prefix
    logger.info(f"Particle directory: {particle_dir}")
    
    dsb_file = run_dir / f"{prefix}_dsb_complexity.csv"
    energy_file = run_dir / f"{prefix}_EnergyDeposit.csv"
    dose_file = run_dir / f"{prefix}_Dose.csv"
    
    logger.info("\nInput files:")
    logger.info(f"  DSB complexity: {dsb_file}")
    logger.info(f"  Energy deposit: {energy_file}")
    logger.info(f"  Dose: {dose_file}\n")
    
    for filepath in [dsb_file, energy_file]:
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")
            
    logger.info("Loading DSB complexity data...")
    dsb_df = pd.read_csv(dsb_file)
    logger.info(f"Loaded {len(dsb_df)} DSBs")
    
    energy_grid = load_energy_grid(energy_file)
    voxel_grid = VoxelGrid()
    local_energies = compute_local_energies(dsb_df, energy_grid, voxel_grid, neighborhood_size)
    energy_zscores = compute_energy_zscores(local_energies)
    
    model = UnifiedVoxelAwareOxygenModel()

    # Resolve atomic number — raises KeyError for unrecognised particles,
    # which is intentional: better to fail loudly than silently mis-model.
    if particle not in PARTICLE_Z:
        raise ValueError(
            f"Particle '{particle}' not in PARTICLE_Z table. "
            f"Supported: {sorted(PARTICLE_Z.keys())}"
        )
    Z = PARTICLE_Z[particle]
    LET = float(let_str)

    # Use compute_P_DSB_unseen for all particles:
    #   • Accepts raw (un-z-scored) local energies; z-scores internally per run
    #   • Gets δf via Z-indexed log-linear interpolation (handles helium and
    #     any other non-calibrated particle without a PARTICLE_LIBRARY entry)
    #   • Computes p1/p2/p3 from LET at runtime — correct for SOBP variants
    #     and for any LET that differs from the single calibration LET
    P_DSB_array = model.compute_P_DSB_unseen(local_energies, Z, LET, o2_level)
    
    random_values = np.random.uniform(0, 1, size=len(dsb_df))
    retained_mask = random_values < P_DSB_array
    
    hypoxic_dsb_df = dsb_df[retained_mask].copy().reset_index(drop=True)
    hypoxic_dsb_df['O2_percent'] = o2_level
    hypoxic_dsb_df['P_DSB'] = P_DSB_array[retained_mask]
    hypoxic_dsb_df['E_local_MeV'] = local_energies[retained_mask]
    hypoxic_dsb_df['E_zscore'] = energy_zscores[retained_mask]
    
    target_o2_str = f"{o2_level}"
    hypoxic_parent_dir = particle_dir / f"{particle}_{let_str}_{target_o2_str}"
    hypoxic_run_dir = hypoxic_parent_dir / f"{particle}_{let_str}_{target_o2_str}_{run_id}"
    hypoxic_run_dir.mkdir(parents=True, exist_ok=True)
    
    base_out = f"{particle}_{let_str}_{target_o2_str}_{run_id}"
    out_dsb_file = hypoxic_run_dir / f"{base_out}_dsb_complexity.csv"
    out_energy_file = hypoxic_run_dir / f"{base_out}_EnergyDeposit.csv"
    out_dose_file = hypoxic_run_dir / f"{base_out}_Dose.csv"
    out_summary_file = hypoxic_run_dir / f"{base_out}_complexity_summary.json"
    
    hypoxic_dsb_df.to_csv(out_dsb_file, index=False)
    shutil.copy2(energy_file, out_energy_file)
    if dose_file.exists():
        shutil.copy2(dose_file, out_dose_file)
        
    sdd_file = run_dir / f"{prefix}_DNADamage_sdd.txt"
    out_sdd_file = hypoxic_run_dir / f"{base_out}_DNADamage_sdd.txt"
    if sdd_file.exists():
        shutil.copy2(sdd_file, out_sdd_file)
        
    dna_full_file = run_dir / f"{prefix}_DNADamage_full.csv"
    out_dna_full_file = hypoxic_run_dir / f"{base_out}_DNADamage_full.csv"
    if dna_full_file.exists():
        shutil.copy2(dna_full_file, out_dna_full_file)
        
    summary = {
        'model_version': 'VOxA v1.0.1',
        'prefix': prefix,
        'particle': particle,
        'Z': Z,
        'LET': float(let_str),
        'O2_percent': float(o2_level),
        'random_seed': int(random_seed),
        'normoxic_n_dsbs': int(len(dsb_df)),
        'hypoxic_n_dsbs': int(len(hypoxic_dsb_df)),
        'retention_fraction': float(len(hypoxic_dsb_df) / len(dsb_df)) if len(dsb_df) > 0 else 0.0,
        'output_files': {
            'dsb_complexity': str(out_dsb_file)
        }
    }
    
    with open(out_summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
        
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate hypoxic dataset from TOPAS-nBio normoxic simulation'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        required=True,
        help='Run prefix (e.g., carbon_40.9_21.0_01)'
    )
    
    parser.add_argument(
        '--o2level',
        type=float,
        required=True,
        help='Target oxygen level in %% (e.g., 0.021, 0.21, 2.1)'
    )
    
    parser.add_argument(
        '--basedir',
        type=str,
        default='.',
        help='Base directory containing particle folders (default: current directory)'
    )
    
    parser.add_argument(
        '--neighborhood',
        type=int,
        default=2,
        help='Size of voxel neighborhood for local energy computation (default: 2 for 2×2×2)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: auto-generated from prefix and O2 level)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate hypoxic dataset
    try:
        summary = generate_hypoxic_dataset(
            prefix=args.prefix,
            o2_level=args.o2level,
            base_dir=Path(args.basedir),
            neighborhood_size=args.neighborhood,
            random_seed=args.seed
        )
        
        print("\n" + "=" * 70)
        print("SUCCESS")
        print("=" * 70)
        print(f"Generated hypoxic dataset for {args.prefix} at {args.o2level}% O2")
        print(f"Retained: {summary['hypoxic_n_dsbs']} / {summary['normoxic_n_dsbs']} DSBs")
        print(f"Output: {summary['output_files']['dsb_complexity']}")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Failed to generate dataset: {e}")
        import sys
        sys.exit(1)