# dsb-topology-let-oxygen

**Topological structure of radiation-induced DNA damage encodes coupled LET–oxygen signatures**

**Renato III Fernan Bolo** and **Ramon Jose C. Bagunu**  
Department of Physical Sciences and Mathematics, University of the Philippines Manila

> Preprint: *arXiv* [ARXIV-TOPOLOGY-DOI]  
> Submitted to *Radiological Physics and Technology* (Springer)  
> This is **Project [2]** of a five-project computational radiation biophysics thesis arc.

---

## Overview

This repository contains the complete computational pipeline for Bolo & Bagunu (2026). Nuclear-scale persistent homology and Random Forest classification are applied to radiation-induced DNA double-strand break (DSB) topology across 2,450 simulated nuclei spanning seven particle configurations (0.2–70.7 keV/µm) and seven oxygen levels (0.005–21% O₂).

The central finding is a three-tier classification hierarchy: particle type and SOBP position are perfectly decodable (balanced accuracy = 1.000), oxygen-level classification degrades monotonically with LET with a charge-driven inversion at the helium-to-carbon transition, and the topological summary modality (m7, persistent entropy and landscape integrals) dominates oxygen encoding via two mechanistically separable channels confirmed by a partial-out test.

Hypoxic DSB populations are generated using the **Voxel-Aware Oxygen model (VOxA)** from Project [1] ([ARXIV-VOXA-DOI]).

> **Note:** The LaTeX manuscript source is not included here — it will be deposited on arXiv separately.

---

## Repository structure

```
dsb-topology-let-oxygen/
│
├── README.md
├── LICENSE                            # MIT License
├── .gitignore
├── requirements.txt                   # pip dependencies
├── environment.yml                    # conda environment
│
├── scripts/                           # Pipeline (run in order 01 → 08)
│   ├── 01_extract_dsb.py              # SDD parsing, Hopcroft-Karp matching,
│   │                                  #   DSB/DSB+/DSB++ classification
│   ├── 02_ph_topology_analysis.py     # Vietoris-Rips PH (Ripser), Wasserstein-2
│   │                                  #   matrices (Persim), m7 feature extraction
│   ├── 03_compute_features.py         # m1–m6 feature extraction per nucleus
│   ├── 04_build_feature_matrix.py     # Merge m1–m7 → feature_matrix.csv (107 feat.)
│   ├── 05_random_forest.py            # RF classification Tasks 1–4, permutation
│   │                                  #   importance, within-fold median imputation
│   ├── 06_additional_analyses.py      # ANOVA η², PCA, single-modality ablation,
│   │                                  #   SOBP Mann-Whitney, cross-modality correlation
│   ├── 07_regenerate_figures.py       # Manuscript figures at 600 DPI
│   ├── 08_partialout_test.py          # Partial-out dual-mechanism test (OLS + RF)
│   ├── generate_hypoxic_dataset.py    # VOxA hypoxic DSB generation (single condition)
│   └── batch_generate_hypoxic.py      # Batch wrapper for all 42 hypoxic conditions
│
├── simulation/                        # TOPAS-nBio parameter files (.txt)
│   │                                  # One file per particle-LET configuration.
│   │                                  # Raw SDD outputs (*_DNADamage_sdd.txt) are
│   │                                  # not tracked; see Data Availability below.
│   ├── electron_0.2.txt
│   ├── proton_4.6.txt
│   ├── proton_8.1.txt
│   ├── helium_10.0.txt
│   ├── helium_30.0.txt
│   ├── carbon_40.9.txt
│   └── carbon_70.7.txt
│
├── data/
│   ├── normoxic_dsb/                  # Per-run DSB CSV files at 21% O₂
│   │   └── [prefix]_dsb_complexity.csv
│   └── hypoxic_dsb/                   # VOxA-generated hypoxic DSB CSVs
│       └── [particle]_[LET]_[O2]/
│           └── [prefix]_dsb_complexity.csv
│                                      # feature_matrix.csv is NOT tracked;
│                                      # regenerate with scripts 01–04.
│
└── analysis/
    ├── partialout/                    # Partial-out test outputs (script 08)
    │   ├── eta2_partialout.json
    │   ├── rf_exclusion_results.json
    │   └── partialout_summary.json
    └── figures/
        └── manuscript/                # Manuscript and appendix figures
            ├── fig1_classification_hierarchy.png
            ├── fig2_single_modality_accuracy.png
            ├── fig3_effect_size_scatter.png
            ├── fig4_wasserstein_separability.png
            ├── fig5_partial_out_dual_mechanism.png
            ├── task1_o2_electron_mono_cm.png
            ├── task1_o2_proton_psobp_cm.png
            ├── task1_o2_proton_dsobp_cm.png
            ├── task1_o2_helium_psobp_cm.png
            ├── task1_o2_helium_dsobp_cm.png
            ├── task1_o2_carbon_psobp_cm.png
            ├── task1_o2_carbon_dsobp_cm.png
            ├── task2_particle_cm.png
            ├── task4_sobp_proton_cm.png
            ├── task4_sobp_helium_cm.png
            └── task4_sobp_carbon_cm.png
```

---

## Setup

### 1. Clone

```bash
git clone https://github.com/[GITHUB-REPO].git
cd dsb-topology-let-oxygen
```

### 2. Environment

**Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate dsb-topology
```

**pip:**
```bash
pip install -r requirements.txt
```

---

## Running the pipeline

All scripts are run from the repository root. Execute in order; each script reads outputs from its predecessors.

```bash
# Step 1 — Extract DSBs from SDD files (skip if using pre-processed data/normoxic_dsb/)
python scripts/01_extract_dsb.py

# Step 2 — Generate hypoxic DSB populations via VOxA
python scripts/batch_generate_hypoxic.py

# Step 3 — Compute persistent homology and m7 features
python scripts/02_ph_topology_analysis.py

# Step 4 — Extract m1–m6 features
python scripts/03_compute_features.py

# Step 5 — Build feature matrix
python scripts/04_build_feature_matrix.py

# Step 6 — Random Forest classification (Tasks 1–4)
python scripts/05_random_forest.py

# Step 7 — Additional analyses
python scripts/06_additional_analyses.py

# Step 8 — Partial-out dual-mechanism test
python scripts/08_partialout_test.py

# Step 9 — Regenerate manuscript figures
python scripts/07_regenerate_figures.py
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.10 | Runtime |
| numpy | ≥ 1.24 | Numerical arrays |
| pandas | ≥ 2.0 | Data frames |
| scipy | ≥ 1.11 | Statistics, clustering, OLS |
| scikit-learn | ≥ 1.3 | RF classifier, CV, imputation, permutation importance |
| ripser | ≥ 0.6 | Vietoris-Rips persistent homology |
| persim | ≥ 0.3 | Wasserstein-2 distances |
| umap-learn | ≥ 0.5 | UMAP dimensionality reduction (script 02, optional) |
| matplotlib | ≥ 3.7 | Figures |
| joblib | ≥ 1.3 | Parallel computation |

---

## Data availability

Raw TOPAS-nBio SDD output files (`*_DNADamage_sdd.txt`) are not tracked due to file size. The processed DSB CSV files in `data/normoxic_dsb/` are the primary pipeline input. The full feature matrix (`data/feature_matrix.csv`) is reproducible by running scripts 01–04 on those CSVs. The partial-out JSON results in `analysis/partialout/` are provided directly as they require the full 5-fold × 10-repeat RF runs to reproduce.

---

## Key results

| Task | Target | Classes | Balanced accuracy |
|------|--------|---------|-------------------|
| 1 | O₂ level, per particle | 7 | 0.189–0.517 |
| 2 | Particle configuration | 7 | 1.000 ± 0.000 |
| 3 | Joint particle–O₂ condition | 49 | 0.346 ± 0.017 |
| 4 | SOBP position | 2 | 1.000 ± 0.000 |

H₀ Wasserstein-2 separation ratio: **2.399**  
m7 H₀ persistent entropy η²_O₂: **0.617**  
m7 BA survival ratio after DSB count removal: **1.011** (5/7 configurations)

---

## Citation

```bibtex
@article{BoloBagunu2026topology,
  author  = {Bolo, Renato III Fernan and Bagunu, Ramon Jose C.},
  title   = {Topological structure of radiation-induced {DNA} damage
             encodes coupled {LET}--oxygen signatures},
  journal = {Radiological Physics and Technology},
  year    = {2026},
  note    = {[DOI to be added upon acceptance]}
}
```

This work builds on the VOxA model from Project [1]:

```bibtex
@article{BoloBagunu2026voxa,
  author  = {Bolo, Renato III Fernan and Bagunu, Ramon Jose C.},
  title   = {Voxel-aware oxygen kinetics resolves radiation-induced
             {DNA} damage retention across {LET}-oxygen conditions
             in particle therapy},
  journal = {arXiv},
  year    = {2026},
  note    = {[ARXIV-VOXA-DOI]}
}
```

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

Renato III Fernan Bolo — rfbolo@up.edu.ph  
Department of Physical Sciences and Mathematics, University of the Philippines Manila
