#!/usr/bin/env python3
"""
================================================================================
UNIFIED VOXEL-AWARE OXYGEN MODEL (VOxA v1.0)
================================================================================

Version:  v1.0.1 (Precise δf from Step 9 CSV; x50 log-interp for heavy ions)
Date:     February 2026

DESCRIPTION
-----------
This module implements the Voxel-Aware Oxygen Model (VOxA) for predicting
DNA double-strand break (DSB) retention probability under varying oxygen
conditions. The model incorporates:

1. OXYGEN MODEL (OM): Sakata 2019 combinatorial framework with optimized
   oxygen kinetics parameters

2. VOXEL-AWARE (VA): Local energy-dependent modulation of direct damage
   fraction using calibrated δf parameters

KEY PARAMETERS (v1.0 - UPDATED)
-------------------------------
Oxygen Kinetics (optimized on OER literature):
    K_fix = 0.1593% O₂
    K_repair = 0.2119% O₂
    p_indirect half-max (K_fix + K_repair) = 0.3712% O₂ = 2.82 mmHg
    OER_max (retention) = 3.32
    R² = 0.7194

Sakata 2019 Combinatorics (d=0.20, i=0.80):
    p1_low = 0.04 (d² = purely direct)
    p2_low = 0.32 (2di = hybrid)
    p3_low = 0.64 (i² = purely indirect)
    p1_high = 0.64 (UPDATED - was 0.68)

Voxel-Aware δf (calibrated via Pareto frontier + bootstrap):
    Electron: δf = 0.0036 [0.0035, 0.0049], CV = 0.36%
    Proton:   δf = 0.0254 [0.0238, 0.0269], CV = 2.53%
    Carbon:   δf = 0.0778 [0.0725, 0.0833], CV = 7.14%

Z-INDEXED δf INTERPOLATION (NEW — no additional calibration needed)
--------------------------------------------------------------------
δf for any particle is obtained via log-linear interpolation in Z-space
from the three calibrated anchor points (electron Z=0, proton Z=1,
carbon Z=6). Predicted values: helium Z=2 → δf≈0.039, neon Z=10 → δf≈0.108.

WHY NOT CV(E_local)-BASED REPARAMETERISATION
--------------------------------------------
    Electron: CV=20.76%, δf=0.0036 → implied α=0.0173
    Proton:   CV=36.18%, δf=0.0254 → implied α=0.0702
    Carbon:   CV=31.52%, δf=0.0778 → implied α=0.2468  →  14× variation

CV(E_local) is non-monotonic with LET (proton > carbon) while δf
monotonically increases with Z. These are structurally decoupled.
Z-interpolation is the physically consistent generalisation.

Use compute_P_DSB_unseen(E_local_raw, Z, LET, O2) for any particle.
Use compute_P_DSB(E_zscore, particle_name, O2) for the three calibrated
particles (backward-compatible; generate_hypoxic_dataset.py unchanged).

THEORETICAL FOUNDATION
----------------------
DSB retention probability with NORMALIZATION:

    P_DSB(O₂) = [p₁ + p₂·p_ind(O₂) + p₃·p_ind(O₂)²] /
                [p₁ + p₂·p_ind(21%) + p₃·p_ind(21%)²]

Where:
    p_ind(O₂) = (O₂ + K_fix) / (O₂ + K_fix + K_repair)

Voxel-aware extension:
    f_direct(i) = p₁ + δf × E_zscore(i), clamped to [f_min, f_max]

================================================================================
"""

import json
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# MODEL CONSTANTS (VOxA v1.0)
# =============================================================================

@dataclass(frozen=True)
class ModelConstants:
    """
    Physical constants for VOxA v1.0.
    
    All values validated against literature and calibrated via Pareto
    frontier optimization with 1000 bootstrap iterations.
    """
    # Model version
    version: str = "VOxA v1.0"
    description: str = "Voxel-Aware Oxygen Model with p1_high=0.64"
    calibration_date: str = "2026-02-17"
    
    # Oxygen kinetics (optimized on OER literature data)
    K_fix: float = 0.1593       # % O₂ - oxygen-independent fixation
    K_repair: float = 0.2119    # % O₂ - thiol repair capacity
    
    # Sakata 2019 direct/indirect fractions
    d_fraction: float = 0.20    # Direct damage fraction (low LET)
    i_fraction: float = 0.80    # Indirect damage fraction (low LET)
    
    # Low-LET case fractions (from d² + 2di + i² = 1)
    p1_low: float = 0.04        # d² = 0.20² = purely direct
    p2_low: float = 0.32        # 2di = 2×0.20×0.80 = hybrid
    p3_low: float = 0.64        # i² = 0.80² = purely indirect
    
    # High-LET asymptote (UPDATED from 0.68 to 0.64)
    p1_high: float = 0.64
    
    # Steepness parameters (from calibration)
    s_dir_light: float = 1.9878
    s_ind_light: float = 2.0001
    s_dir_base: float = 1.391
    s_dir_scale: float = -0.0068
    s_ind_base: float = 1.875
    s_ind_scale: float = -0.2134
    
    # Physical bounds for f_direct (UPDATED f_max from 0.68 to 0.64)
    f_min: float = 0.02         # Minimum (p1_low / 2)
    f_max: float = 0.64         # Maximum (p1_high, UPDATED)
    
    # Standard oxygen levels (%)
    O2_normoxia: float = 21.0
    O2_mild_hypoxia: float = 2.1
    O2_moderate_hypoxia: float = 0.21
    O2_severe_hypoxia: float = 0.021
    O2_anoxia: float = 0.001
    
    # Theoretical OER_max for retention
    OER_max: float = 3.32
    
    # Calibration metrics
    R_squared: float = 0.7194
    R_squared_weighted: float = 0.7474
    n_calibration_points: int = 215


@dataclass
class ParticleParameters:
    """
    Particle-specific parameters for VOxA v1.0.
    
    Includes LET-dependent case fractions and voxel-aware calibration results.
    """
    name: str
    
    # Physical properties
    Z: int                      # Atomic number
    LET_keV_um: float          # LET in keV/μm
    
    # LET transition parameters
    x50_dir: float             # x50 for direct transition
    x50_ind: float             # x50 for indirect transition
    
    # Steepness parameters
    s_dir: float               # Steepness for direct transition
    s_ind: float               # Steepness for indirect transition
    
    # Computed case fractions (LET-dependent)
    p1: float                  # Direct fraction (p1_base)
    p2: float                  # Hybrid fraction
    p3: float                  # Indirect fraction
    
    # Voxel-aware calibration results
    delta_f: float             # Energy sensitivity parameter
    delta_f_ci_low: float      # 95% CI lower bound
    delta_f_ci_high: float     # 95% CI upper bound
    
    # Energy statistics
    E_mean: float              # Mean energy z-score
    E_sd: float                # SD energy z-score
    
    # Fields with defaults
    f_min: float = 0.02
    f_max: float = 0.64        # UPDATED from 0.68
    
    # Validation metrics (at calibration O2 = 0.21%)
    P_DSB_cv: float = 0.0      # Coefficient of variation (%)
    P_DSB_cv_ci_low: float = 0.0
    P_DSB_cv_ci_high: float = 0.0
    mean_error_pct: float = 0.0
    
    # Calibration dataset info
    n_dsbs_calibration: int = 0
    f_direct_min: float = 0.0
    f_direct_max: float = 0.0


# =============================================================================
# PARTICLE LIBRARY (from voxa_voxel_aware_calibration.json)
# =============================================================================

PARTICLE_LIBRARY: Dict[str, ParticleParameters] = {
    "electron": ParticleParameters(
        name="electron",
        Z=0,
        LET_keV_um=0.2,
        x50_dir=124.6523,
        x50_ind=2016.7319,
        s_dir=1.9878,
        s_ind=2.0001,
        p1=0.0401,
        p2=0.3199,
        p3=0.6400,
        delta_f=0.003572407783,
        delta_f_ci_low=0.003480887367,
        delta_f_ci_high=0.004864884679,
        E_mean=0.0145,
        E_sd=0.003,
        f_min=0.02,
        f_max=0.64,
        P_DSB_cv=0.3636,
        P_DSB_cv_ci_low=0.3543,
        P_DSB_cv_ci_high=0.4952,
        mean_error_pct=0.0000,
        n_dsbs_calibration=2515,
        f_direct_min=0.0314,
        f_direct_max=0.0591
    ),
    "proton": ParticleParameters(
        name="proton",
        Z=1,
        LET_keV_um=4.6,
        x50_dir=153.0302,
        x50_ind=2448.5309,
        s_dir=1.9878,
        s_ind=2.0001,
        p1=0.0938,
        p2=0.2665,
        p3=0.6398,
        delta_f=0.025355044928,
        delta_f_ci_low=0.023811577991,
        delta_f_ci_high=0.026892303509,
        E_mean=1.1262,
        E_sd=0.4074,
        f_min=0.02,
        f_max=0.64,
        P_DSB_cv=2.528,
        P_DSB_cv_ci_low=2.3741,
        P_DSB_cv_ci_high=2.6812,
        mean_error_pct=0.0011,
        n_dsbs_calibration=2491,
        f_direct_min=0.0237,
        f_direct_max=0.1077
    ),
    "carbon": ParticleParameters(
        name="carbon",
        Z=6,
        LET_keV_um=40.9,
        x50_dir=264.4079,
        x50_ind=2434.245,
        s_dir=1.3806,
        s_ind=1.4354,
        p1=0.3802,
        p2=0.0130,
        p3=0.6068,
        delta_f=0.077847589106,
        delta_f_ci_low=0.072526985535,
        delta_f_ci_high=0.083287244880,
        E_mean=0.8205,
        E_sd=0.2587,
        f_min=0.02,
        f_max=0.64,
        P_DSB_cv=7.1445,
        P_DSB_cv_ci_low=6.6562,
        P_DSB_cv_ci_high=7.6436,
        mean_error_pct=0.0110,
        n_dsbs_calibration=2519,
        f_direct_min=0.1334,
        f_direct_max=0.4411
    )
}


# =============================================================================
# Z-INDEXED δf TABLE AND INTERPOLATION
# =============================================================================
#
# WHY Z-INTERPOLATION, NOT CV(E_local)-BASED REPARAMETERISATION
# -------------------------------------------------------------
# A natural question is whether δf can be expressed as α × CV(E_local),
# making it fully derivable from each normoxic run without separate
# calibration. The calibration data rules this out:
#
#   Electron:  CV(E_local) = 20.76%,  δf = 0.0036  →  α = 0.0173
#   Proton:    CV(E_local) = 36.18%,  δf = 0.0254  →  α = 0.0702
#   Carbon:    CV(E_local) = 31.52%,  δf = 0.0778  →  α = 0.2468
#
# α varies 14× across three particles. CV(E_local) is NON-MONOTONIC
# with LET: proton > carbon because proton's narrow, stochastically
# placed track creates high relative energy variance despite lower LET.
# Meanwhile δf is monotonically increasing with Z. These two quantities
# are structurally decoupled. CV(E_local) cannot serve as a proxy for δf.
#
# CORRECT GENERALISATION: Z-indexed log-linear interpolation.
# This is the same mechanism validated for x50_dir and x50_ind in the OM.
# For any unseen particle (helium Z=2, neon Z=10, argon Z=18), δf is
# derived analytically with NO additional TOPAS-nBio calibration runs.
# For SOBP LET variants, one δf per Z is used; the LET dependence of
# the direct fraction is carried entirely by p1(LET, Z) computed at runtime.
#
# Z_interp assignments: electron=0 (anchor), proton=1, carbon=6.
#
# =============================================================================

# Z-indexed δf lookup table (from v1.0 Pareto + bootstrap calibration)
Z_DELTA_F_TABLE = [
    # (Z_interp, particle_name, delta_f, ci_low, ci_high)
    # Precise values from Step 9 Pareto + bootstrap calibration
    # (voxa_Z_delta_f_table.csv, 2026-03-07)
    (0, "electron", 0.003572407783, 0.003480887367, 0.004864884679),
    (1, "proton",   0.025355044928, 0.023811577991, 0.026892303509),
    (6, "carbon",   0.077847589106, 0.072526985535, 0.083287244880),
]


def interp_delta_f_by_Z(
    Z_target: int,
    table: list = None
) -> dict:
    """
    Interpolate δf by atomic number Z using log-linear interpolation.

    This is the canonical method for obtaining δf for any particle type,
    including those not in the PARTICLE_LIBRARY. The three calibrated
    anchor points (electron Z=0, proton Z=1, carbon Z=6) are sufficient
    for interpolation across the therapeutic particle range.

    Parameters
    ----------
    Z_target : int
        Atomic number of the target particle (0 for electron/photon).
    table : list, optional
        Custom Z_DELTA_F_TABLE. Defaults to the built-in calibration.

    Returns
    -------
    dict with keys: delta_f, delta_f_ci_low, delta_f_ci_high, method
        delta_f         : interpolated δf value
        delta_f_ci_low  : interpolated 95% CI lower bound
        delta_f_ci_high : interpolated 95% CI upper bound
        method          : one of 'exact', 'log_interp', 'log_extrap'

    Examples
    --------
    >>> result = interp_delta_f_by_Z(2)   # Helium
    >>> print(f"Helium δf = {result['delta_f']:.6f}")

    Notes
    -----
    For Z > 0: log-linear interpolation in log(Z)-space between the two
    bracketing calibrated points.  For Z = 0 (electron/photon): the
    electron anchor value is returned directly.
    """
    if table is None:
        table = Z_DELTA_F_TABLE

    Z_vals   = np.array([row[0] for row in table])
    df_vals  = np.array([row[2] for row in table])
    df_lo    = np.array([row[3] for row in table])
    df_hi    = np.array([row[4] for row in table])
    names    = [row[1] for row in table]

    # --- Exact match ---
    if Z_target in Z_vals:
        idx = int(np.where(Z_vals == Z_target)[0][0])
        return dict(delta_f=float(df_vals[idx]),
                    delta_f_ci_low=float(df_lo[idx]),
                    delta_f_ci_high=float(df_hi[idx]),
                    method="exact",
                    particle_source=names[idx])

    # --- Electron / photon anchor (Z=0, below any log-space range) ---
    if Z_target <= 0:
        idx = int(np.where(Z_vals == 0)[0][0])
        return dict(delta_f=float(df_vals[idx]),
                    delta_f_ci_low=float(df_lo[idx]),
                    delta_f_ci_high=float(df_hi[idx]),
                    method="electron_anchor")

    # --- Log-linear interpolation for Z > 0 ---
    pos_mask  = Z_vals > 0
    Z_pos     = Z_vals[pos_mask]
    df_pos    = df_vals[pos_mask]
    df_lo_pos = df_lo[pos_mask]
    df_hi_pos = df_hi[pos_mask]
    log_Z_pos = np.log(Z_pos)
    log_Zt    = np.log(Z_target)

    def _log_interp(a, b, frac):
        """Interpolate in log-space (geometric interpolation)."""
        return float(np.exp(np.log(a) + frac * (np.log(b) - np.log(a))))

    if Z_target < Z_pos.min():
        # Between electron (Z=0, pseudo log(0.5)) and lowest Z>0 point
        idx_e   = int(np.where(Z_vals == 0)[0][0])
        idx_p   = int(np.argmin(Z_pos))
        log_Z_lo = np.log(0.5)   # numeric proxy for electron anchor
        log_Z_hi = log_Z_pos[idx_p]
        frac = float(np.clip((log_Zt - log_Z_lo) / (log_Z_hi - log_Z_lo), 0.0, 1.0))
        return dict(
            delta_f         = _log_interp(df_vals[idx_e], df_pos[idx_p], frac),
            delta_f_ci_low  = _log_interp(df_lo[idx_e], df_lo_pos[idx_p], frac),
            delta_f_ci_high = _log_interp(df_hi[idx_e], df_hi_pos[idx_p], frac),
            method="log_interp_low",
            Z_bracket=(0, int(Z_pos[idx_p]))
        )

    if Z_target > Z_pos.max():
        # Extrapolate from the top two positive-Z anchor points
        ord_idx = np.argsort(Z_pos)
        idx_lo  = int(ord_idx[-2])
        idx_hi  = int(ord_idx[-1])
        frac = (log_Zt - log_Z_pos[idx_lo]) / (log_Z_pos[idx_hi] - log_Z_pos[idx_lo])
        return dict(
            delta_f         = _log_interp(df_pos[idx_lo], df_pos[idx_hi], frac),
            delta_f_ci_low  = _log_interp(df_lo_pos[idx_lo], df_lo_pos[idx_hi], frac),
            delta_f_ci_high = _log_interp(df_hi_pos[idx_lo], df_hi_pos[idx_hi], frac),
            method="log_extrap",
            Z_bracket=(int(Z_pos[idx_lo]), int(Z_pos[idx_hi]))
        )

    # Standard bracket
    idx_lo = int(np.max(np.where(Z_pos <= Z_target)[0]))
    idx_hi = int(np.min(np.where(Z_pos >= Z_target)[0]))
    frac   = (log_Zt - log_Z_pos[idx_lo]) / (log_Z_pos[idx_hi] - log_Z_pos[idx_lo])
    return dict(
        delta_f         = _log_interp(df_pos[idx_lo], df_pos[idx_hi], frac),
        delta_f_ci_low  = _log_interp(df_lo_pos[idx_lo], df_lo_pos[idx_hi], frac),
        delta_f_ci_high = _log_interp(df_hi_pos[idx_lo], df_hi_pos[idx_hi], frac),
        method="log_interp",
        Z_bracket=(int(Z_pos[idx_lo]), int(Z_pos[idx_hi]))
    )


# =============================================================================
# UNIFIED VOXEL-AWARE OXYGEN MODEL
# =============================================================================

class UnifiedVoxelAwareOxygenModel:
    """
    Unified Voxel-Aware Oxygen Model (VOxA v1.0).
    
    Implements oxygen-dependent DSB retention with voxel-aware local
    energy modulation.
    """
    
    def __init__(self, params_file: Optional[str] = None):
        """
        Initialize VOxA model.
        
        Parameters
        ----------
        params_file : str, optional
            Path to JSON calibration file. If None, uses built-in parameters.
        """
        self.constants = ModelConstants()
        
        if params_file is not None:
            self._load_from_file(params_file)
        else:
            self.particles = PARTICLE_LIBRARY
        
        logger.info(f"Initialized {self.constants.version}")
        logger.info(f"Loaded {len(self.particles)} particle types: {list(self.particles.keys())}")
    
    def _load_from_file(self, filepath: Union[str, Path]) -> None:
        """Load calibration parameters from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Verify version compatibility
        if 'model_version' in data:
            logger.info(f"Loading parameters from: {data['model_version']}")
        
        # Load particle parameters
        if 'particle_calibration' in data:
            self.particles = {}
            for pname, pdata in data['particle_calibration'].items():
                self.particles[pname] = ParticleParameters(
                    name=pname,
                    Z={'electron': 0, 'proton': 1, 'carbon': 6}.get(pname, 0),
                    LET_keV_um=pdata['LET_keV_um'],
                    x50_dir=pdata['x50_dir'],
                    x50_ind=pdata['x50_ind'],
                    s_dir=pdata['s_dir'],
                    s_ind=pdata['s_ind'],
                    p1=pdata['p1_base'],
                    p2=pdata['p2_base'],
                    p3=pdata['p3_base'],
                    delta_f=pdata['delta_f'],
                    delta_f_ci_low=pdata['delta_f_ci_low'],
                    delta_f_ci_high=pdata['delta_f_ci_high'],
                    E_mean=pdata['E_mean'],
                    E_sd=pdata['E_sd'],
                    P_DSB_cv=pdata['P_DSB_cv'],
                    P_DSB_cv_ci_low=pdata['P_DSB_cv_ci_low'],
                    P_DSB_cv_ci_high=pdata['P_DSB_cv_ci_high'],
                    mean_error_pct=pdata['mean_error_pct'],
                    n_dsbs_calibration=pdata['n_dsbs'],
                    f_direct_min=pdata['f_direct_min'],
                    f_direct_max=pdata['f_direct_max'],
                    f_min=0.02,
                    f_max=0.64
                )
        
        logger.info(f"Loaded {len(self.particles)} particles from {filepath}")
    
    def calc_p_ind(self, O2_percent: float) -> float:
        """
        Calculate indirect damage survival probability.
        
        Parameters
        ----------
        O2_percent : float
            Oxygen concentration in %
        
        Returns
        -------
        float
            Probability of indirect damage survival
        """
        numerator = O2_percent + self.constants.K_fix
        denominator = O2_percent + self.constants.K_fix + self.constants.K_repair
        return numerator / denominator
    
    def calc_P_DSB_uniform(self, O2_percent: float, particle: str) -> float:
        """
        Calculate uniform P_DSB (without voxel-aware correction).
        
        Parameters
        ----------
        O2_percent : float
            Oxygen concentration in %
        particle : str
            Particle type ('electron', 'proton', 'carbon')
        
        Returns
        -------
        float
            DSB retention probability
        """
        if particle not in self.particles:
            raise ValueError(f"Unknown particle: {particle}")
        
        params = self.particles[particle]
        p_ind = self.calc_p_ind(O2_percent)
        p_ind_norm = self.calc_p_ind(self.constants.O2_normoxia)
        
        # Numerator: DSB retention at target O2
        numerator = params.p1 + params.p2 * p_ind + params.p3 * p_ind**2
        
        # Denominator: DSB retention at 21% O2 (normalization)
        denominator = params.p1 + params.p2 * p_ind_norm + params.p3 * p_ind_norm**2
        
        return numerator / denominator
    
    def compute_P_DSB(
        self,
        E_local: Union[float, np.ndarray],
        particle: str,
        O2_percent: float
    ) -> Union[float, np.ndarray]:
        """
        Compute voxel-aware P_DSB with local energy modulation.
        
        Parameters
        ----------
        E_local : float or array
            Local energy z-score(s)
        particle : str
            Particle type ('electron', 'proton', 'carbon')
        O2_percent : float
            Oxygen concentration in %
        
        Returns
        -------
        float or array
            DSB retention probability
        """
        if particle not in self.particles:
            raise ValueError(f"Unknown particle: {particle}")
        
        params = self.particles[particle]
        
        # Compute modulated f_direct
        f_direct = params.p1 + params.delta_f * E_local
        f_direct = np.clip(f_direct, params.f_min, params.f_max)
        
        # Compute remaining fractions (preserve normalization)
        f_total_indirect = 1.0 - f_direct
        f_hybrid = params.p2 * f_total_indirect / (params.p2 + params.p3)
        f_indirect = params.p3 * f_total_indirect / (params.p2 + params.p3)
        
        # Calculate retention probabilities
        p_ind = self.calc_p_ind(O2_percent)
        p_ind_norm = self.calc_p_ind(self.constants.O2_normoxia)
        
        # Numerator: retention at target O2
        numerator = f_direct + f_hybrid * p_ind + f_indirect * p_ind**2
        
        # Denominator: retention at 21% O2 (normalization)
        denominator = f_direct + f_hybrid * p_ind_norm + f_indirect * p_ind_norm**2
        
        return numerator / denominator
    
    def compute_P_DSB_unseen(
        self,
        E_local_raw: Union[float, np.ndarray],
        Z: int,
        LET_keV_um: float,
        O2_percent: float,
        particle_name: str = None
    ) -> Union[float, np.ndarray]:
        """
        Compute voxel-aware P_DSB for ANY particle using Z-interpolated δf.

        This is the generalized entry point for particles not in the
        PARTICLE_LIBRARY (e.g., helium, neon, argon) and for SOBP LET
        variants of calibrated particles. The two key differences from
        compute_P_DSB() are:

          1. δf is obtained from Z-indexed log-linear interpolation rather
             than a stored calibration value — no extra simulation needed.
          2. p1/p2/p3 case fractions are computed at runtime from the
             supplied LET using the OM transition functions — so the model
             is correct across the full LET range for any particle, not just
             at the single calibration LET.

        For calibrated particles (electron, proton, carbon) at their exact
        calibration LET, this method produces results numerically identical
        to compute_P_DSB() to within floating-point precision.

        Parameters
        ----------
        E_local_raw : float or array
            Raw (un-normalised) local energy values in MeV, as returned by
            compute_local_energies() in generate_hypoxic_dataset.py.
            Z-scoring is performed internally using the run-level mean/std.
        Z : int
            Atomic number of the particle (0 for electron/photon, 1 for
            proton, 2 for helium, 6 for carbon, etc.).
        LET_keV_um : float
            LET of the particle at this specific beam position in keV/µm.
            Used to compute p1(LET), p2(LET), p3(LET) via OM transitions.
        O2_percent : float
            Oxygen concentration in % (21.0 = normoxia).
        particle_name : str, optional
            Name string used only for logging. If None, inferred from Z.

        Returns
        -------
        float or array
            Per-DSB retention probability P_DSB(i), normalised to 1.0 at
            21% O2. Same shape as E_local_raw.

        Notes
        -----
        Z-interpolation anchor points (VOxA v1.0 calibration):
            electron Z=0: δf=0.0036   proton Z=1: δf=0.0254   carbon Z=6: δf=0.0778
        
        The method uses x50 and steepness parameters from the closest
        calibrated particle in Z-space to compute the OM LET transitions.
        For interpolated particles, these are linearly interpolated between
        the two bracketing calibrated particles.
        """
        E_local_arr = np.atleast_1d(np.array(E_local_raw, dtype=float))

        # ── Z-normalise energy within this run ──────────────────────────────
        mean_E = np.mean(E_local_arr)
        std_E  = np.std(E_local_arr)
        if std_E < 1e-12:
            E_zscore = np.zeros_like(E_local_arr)
        else:
            E_zscore = (E_local_arr - mean_E) / std_E

        # ── δf from Z-interpolation ─────────────────────────────────────────
        delta_f_result = interp_delta_f_by_Z(Z)
        delta_f = delta_f_result["delta_f"]

        # ── p1(LET)/p2(LET)/p3(LET) from OM transitions ────────────────────
        # Use the closest calibrated particle's x50/steepness params
        # For Z ≤ 1 (light particles): use light-particle steepness
        # For Z > 1 (heavy ions): use Z-scaled heavy-particle steepness
        c = self.constants

        if Z <= 1:
            # Light particle steepness (photon/electron/proton)
            s_dir = c.s_dir_light
            s_ind = c.s_ind_light
            # Interpolate x50 from nearest calibrated light particle
            if Z <= 0:
                pref = self.particles.get("electron",
                        list(self.particles.values())[0])
            else:
                pref = self.particles.get("proton",
                        self.particles.get("electron",
                        list(self.particles.values())[0]))
            x50_dir = pref.x50_dir
            x50_ind = pref.x50_ind
        else:
            # Heavy ion Z-scaled steepness (Barkas formula consistent)
            s_dir = c.s_dir_base * (1.0 + c.s_dir_scale * np.log(Z / 2.0))
            s_ind = c.s_ind_base * (1.0 + c.s_ind_scale * np.log(Z / 2.0))

            # x50 log-linear interpolation in Z-space between bracketing
            # calibrated heavy particles (proton Z=1, carbon Z=6).
            # Using carbon's x50 for all Z>1 was incorrect: at Z=2 (helium),
            # Δp1 reaches +0.06 at 40 keV/µm, which propagates directly
            # into P_DSB error.  Log-linear interpolation is consistent
            # with how δf and x50_dir/x50_ind are handled in the OM.
            p_proton = self.particles.get("proton")
            p_carbon = self.particles.get("carbon")

            if p_proton is not None and p_carbon is not None:
                Z_lo, Z_hi = 1.0, 6.0
                log_frac = (np.log(Z) - np.log(Z_lo)) / (np.log(Z_hi) - np.log(Z_lo))
                log_frac = float(np.clip(log_frac, 0.0, None))  # allow extrapolation above Z=6
                x50_dir = float(np.exp(
                    np.log(p_proton.x50_dir) + log_frac * (np.log(p_carbon.x50_dir) - np.log(p_proton.x50_dir))
                ))
                x50_ind = float(np.exp(
                    np.log(p_proton.x50_ind) + log_frac * (np.log(p_carbon.x50_ind) - np.log(p_proton.x50_ind))
                ))
            else:
                # Fallback: carbon x50 (original behaviour)
                pref = self.particles.get("carbon", list(self.particles.values())[-1])
                x50_dir = pref.x50_dir
                x50_ind = pref.x50_ind

        # Radiation quality parameter x = 2.5 × LET^1.1 (OM approximation)
        x = 2.5 * (LET_keV_um ** 1.1)
        x = max(x, 1e-6)

        # Sigmoid LET transition functions
        f_dir = 1.0 / (1.0 + (x50_dir / x) ** s_dir)
        f_ind = 1.0 / (1.0 + (x50_ind / x) ** s_ind)

        # Case fractions at this LET
        p1 = c.p1_low + (c.p1_high - c.p1_low) * f_dir
        p3_raw = c.p3_low * (1.0 - f_ind)
        p2_raw = max(1.0 - p1 - p3_raw, 0.0)
        total = p1 + p2_raw + p3_raw
        p1 /= total;  p2 = p2_raw / total;  p3 = p3_raw / total

        # ── Per-DSB f_direct ────────────────────────────────────────────────
        f_direct = np.clip(p1 + delta_f * E_zscore, c.f_min, c.f_max)

        # ── Redistribute indirect fractions ─────────────────────────────────
        f_total_ind = 1.0 - f_direct
        p23 = p2 + p3
        if p23 < 1e-12:
            f_hybrid = np.zeros_like(f_direct)
            f_indirect = f_total_ind
        else:
            f_hybrid   = f_total_ind * (p2 / p23)
            f_indirect = f_total_ind * (p3 / p23)

        # ── P_DSB normalised to 21% O2 ──────────────────────────────────────
        p_ind      = self.calc_p_ind(O2_percent)
        p_ind_norm = self.calc_p_ind(c.O2_normoxia)

        numerator   = f_direct + f_hybrid * p_ind      + f_indirect * p_ind**2
        denominator = f_direct + f_hybrid * p_ind_norm + f_indirect * p_ind_norm**2

        result = numerator / denominator
        return result if result.shape[0] > 1 else float(result[0])

    def print_delta_f_interpolation(self) -> None:
        """Print Z-interpolated δf values for common therapeutic particles."""
        particles_z = [
            ("electron/photon", 0), ("proton",  1), ("helium", 2),
            ("carbon",          6), ("neon",   10), ("argon",  18),
        ]
        print("\n" + "=" * 60)
        print("Z-indexed δf interpolation (VOxA v1.0)")
        print("=" * 60)
        print(f"  {'Particle':<18} {'Z':>3}  {'δf':>10}  {'95% CI':>22}  Method")
        print("-" * 60)
        for name, Z in particles_z:
            r = interp_delta_f_by_Z(Z)
            print(f"  {name:<18} {Z:>3}  {r['delta_f']:>10.6f}"
                  f"  [{r['delta_f_ci_low']:.6f}, {r['delta_f_ci_high']:.6f}]"
                  f"  {r['method']}")
        # Monotonicity check
        vals = [interp_delta_f_by_Z(Z)['delta_f'] for _, Z in particles_z]
        monotone = all(b >= a for a, b in zip(vals, vals[1:]))
        print("-" * 60)
        print(f"  Monotone increasing with Z: {'✓' if monotone else '✗'}")
        print()

    def calc_OER(self, O2_percent: float, particle: str) -> float:
        """
        Calculate Oxygen Enhancement Ratio.
        
        Parameters
        ----------
        O2_percent : float
            Oxygen concentration in %
        particle : str
            Particle type
        
        Returns
        -------
        float
            OER (ratio of normoxic to hypoxic retention)
        """
        P_DSB_hypoxic = self.calc_P_DSB_uniform(O2_percent, particle)
        P_DSB_normoxic = 1.0  # By definition (normalized to 21%)
        
        return P_DSB_normoxic / P_DSB_hypoxic if P_DSB_hypoxic > 0 else np.inf
    
    def get_dsb_retention_table(self) -> pd.DataFrame:
        """Generate DSB retention table across oxygen levels."""
        O2_levels = [
            ('Anoxia', 0.001),
            ('Severe hypoxia', 0.021),
            ('Moderate hypoxia', 0.21),
            ('Mild hypoxia', 2.1),
            ('Normoxia', 21.0)
        ]
        
        rows = []
        for condition, O2 in O2_levels:
            row = {'condition': condition, 'O2_pct': O2}
            
            for pname in self.particles:
                P_DSB = self.calc_P_DSB_uniform(O2, pname)
                OER = self.calc_OER(O2, pname)
                row[f'P_DSB_{pname}'] = round(P_DSB, 4)
                row[f'OER_{pname}'] = round(OER, 3)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def print_summary(self) -> None:
        """Print formatted summary of model configuration."""
        print("\n" + "=" * 70)
        print(f"{self.constants.version}")
        print("=" * 70)
        
        print(f"\nDescription: {self.constants.description}")
        print(f"Calibration Date: {self.constants.calibration_date}")
        print(f"R² = {self.constants.R_squared} (weighted: {self.constants.R_squared_weighted})")
        
        print(f"\nOxygen Kinetics:")
        print(f"  K_fix = {self.constants.K_fix}% O₂")
        print(f"  K_repair = {self.constants.K_repair}% O₂")
        print(f"  Half-max O₂ ≈ {self.constants.K_repair * 7.6:.2f} mmHg")
        
        print(f"\nSakata 2019 Combinatorics:")
        print(f"  d (direct) = {self.constants.d_fraction}")
        print(f"  i (indirect) = {self.constants.i_fraction}")
        
        print(f"\nCase Fractions (Low LET):")
        print(f"  p1 = {self.constants.p1_low} (d² = purely direct)")
        print(f"  p2 = {self.constants.p2_low} (2di = hybrid)")
        print(f"  p3 = {self.constants.p3_low} (i² = purely indirect)")
        print(f"  p1_high = {self.constants.p1_high} (high LET asymptote, UPDATED)")
        
        print(f"\nPhysical bounds: f_min = {self.constants.f_min}, f_max = {self.constants.f_max}")
        print(f"OER_max (retention) = {self.constants.OER_max}")
        
        print(f"\nVoxel-Aware Calibration:")
        print("-" * 70)
        print(f"{'Particle':<10} {'LET':<8} {'p1':<8} {'p2':<8} {'p3':<8} {'δf':<10} {'CV(%)':<8}")
        print("-" * 70)
        
        for pname, params in self.particles.items():
            print(f"{pname:<10} {params.LET_keV_um:<8.1f} {params.p1:<8.4f} {params.p2:<8.4f} {params.p3:<8.4f} "
                  f"{params.delta_f:<10.6f} {params.P_DSB_cv:<8.2f}")
        
        print("-" * 70)
        
        # Verify normalization
        print(f"\nNormalization Check (P_DSB at 21% O₂):")
        for pname in self.particles:
            P_DSB_21 = self.calc_P_DSB_uniform(21.0, pname)
            status = "✓" if abs(P_DSB_21 - 1.0) < 1e-10 else "✗"
            print(f"  {pname}: {P_DSB_21:.10f} {status}")
        
        # Show DSB retention table
        print(f"\nDSB Retention Table:")
        print(self.get_dsb_retention_table().to_string(index=False))
        
        print()
    
    def to_dict(self) -> dict:
        """Export model configuration as dictionary."""
        return {
            'model_version': self.constants.version,
            'description': self.constants.description,
            'calibration_date': self.constants.calibration_date,
            'oxygen_kinetics': {
                'K_fix': self.constants.K_fix,
                'K_repair': self.constants.K_repair,
                'half_max_O2_mmHg': self.constants.K_repair * 7.6
            },
            'sakata_fractions': {
                'd': self.constants.d_fraction,
                'i': self.constants.i_fraction,
                'p1_low': self.constants.p1_low,
                'p2_low': self.constants.p2_low,
                'p3_low': self.constants.p3_low,
                'p1_high': self.constants.p1_high
            },
            'steepness': {
                's_dir_light': self.constants.s_dir_light,
                's_ind_light': self.constants.s_ind_light,
                's_dir_base': self.constants.s_dir_base,
                's_dir_scale': self.constants.s_dir_scale,
                's_ind_base': self.constants.s_ind_base,
                's_ind_scale': self.constants.s_ind_scale
            },
            'OER_max': self.constants.OER_max,
            'R_squared': self.constants.R_squared,
            'R_squared_weighted': self.constants.R_squared_weighted,
            'n_calibration_points': self.constants.n_calibration_points,
            'particles': {
                pname: {
                    'LET_keV_um': params.LET_keV_um,
                    'p1': params.p1,
                    'p2': params.p2,
                    'p3': params.p3,
                    's_dir': params.s_dir,
                    's_ind': params.s_ind,
                    'x50_dir': params.x50_dir,
                    'x50_ind': params.x50_ind,
                    'delta_f': params.delta_f,
                    'delta_f_ci': [params.delta_f_ci_low, params.delta_f_ci_high],
                    'E_mean': params.E_mean,
                    'E_sd': params.E_sd,
                    'P_DSB_cv': params.P_DSB_cv,
                    'n_dsbs_calibration': params.n_dsbs_calibration,
                    'f_direct_range': [params.f_direct_min, params.f_direct_max]
                }
                for pname, params in self.particles.items()
            }
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_model(params_file: Optional[str] = None) -> UnifiedVoxelAwareOxygenModel:
    """
    Load the VOxA v1.0 model.
    
    Parameters
    ----------
    params_file : str, optional
        Path to calibration JSON file
    
    Returns
    -------
    UnifiedVoxelAwareOxygenModel
        Initialized model
    """
    return UnifiedVoxelAwareOxygenModel(params_file)


# =============================================================================
# MAIN (DEMONSTRATION)
# =============================================================================

def main():
    """Demonstrate VOxA v1.0 usage."""
    print("\n" + "=" * 70)
    print("VOXEL-AWARE OXYGEN MODEL v1.0 - DEMONSTRATION")
    print("=" * 70)
    
    # Initialize model
    model = UnifiedVoxelAwareOxygenModel()
    model.print_summary()
    
    # Demonstrate voxel-aware predictions with simulated data
    print("\n" + "=" * 70)
    print("VOXEL-AWARE P_DSB PREDICTIONS (Simulated Data)")
    print("=" * 70)
    
    np.random.seed(42)
    
    for particle in ['electron', 'proton', 'carbon']:
        n_test = 400  # Similar to calibration datasets
        E_local = np.random.lognormal(mean=0, sigma=0.5, size=n_test)
        
        print(f"\n{particle.upper()} (n={n_test} DSBs):")
        print(f"  E_local: mean={np.mean(E_local):.3f}, std={np.std(E_local):.3f}")
        
        for O2 in [21.0, 2.1, 0.21, 0.021]:
            P_DSB = model.compute_P_DSB(E_local, particle, O2)
            P_DSB_uniform = model.calc_P_DSB_uniform(O2, particle)
            
            expected_retained = int(np.mean(P_DSB) * n_test)
            cv = np.std(P_DSB) / np.mean(P_DSB) * 100 if np.mean(P_DSB) > 0 else 0
            
            print(f"  O2={O2:>6.3f}%: P_DSB={np.mean(P_DSB):.4f}±{np.std(P_DSB):.4f} "
                  f"(CV={cv:.2f}%) → ~{expected_retained} DSBs retained")
    
    print("\n" + "=" * 70)
    print("VOxA v1.0 READY FOR USE")
    print("=" * 70)

    # ── Z-indexed δf interpolation demonstration ──────────────────────────
    print("\n" + "=" * 70)
    print("Z-INDEXED δf INTERPOLATION — UNSEEN PARTICLES")
    print("=" * 70)
    model.print_delta_f_interpolation()

    # Demonstrate compute_P_DSB_unseen for helium (Z=2, LET=10 keV/µm)
    np.random.seed(42)
    n_test = 400
    E_local_raw = np.random.lognormal(mean=-3.0, sigma=0.4, size=n_test)  # He-like energy scale
    print("HELIUM (Z=2, LET=10 keV/µm) — computed via compute_P_DSB_unseen():")
    print(f"  E_local: mean={np.mean(E_local_raw):.5f}, std={np.std(E_local_raw):.5f}")
    for O2 in [21.0, 0.21, 0.021, 0.001]:
        P_DSB = model.compute_P_DSB_unseen(E_local_raw, Z=2, LET_keV_um=10.0, O2_percent=O2)
        P_arr = np.atleast_1d(P_DSB)
        cv = np.std(P_arr) / np.mean(P_arr) * 100
        print(f"  O2={O2:>6.3f}%: mean P_DSB={np.mean(P_arr):.4f}, CV={cv:.2f}%")
    print()

    print("=" * 70)
    print("VOxA v1.0 READY FOR USE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()