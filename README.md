# Topological structure of radiation-induced DNA damage encodes coupled LET–oxygen signatures

**Renato III Fernan Bolo** and **Ramon Jose C. Bagunu**  
Department of Physical Sciences and Mathematics, University of the Philippines Manila  

> **Project [2]** of a five-project computational radiation biophysics thesis arc.  
> This builds on the Voxel-Aware Oxygen model (VOxA, Project [1]) at https://doi.org/10.48550/arXiv.2605.12558.

---

## What this repository contains

This repository provides the **complete analysis pipeline** — all Python scripts, TOPAS-nBio simulation parameter files, key results JSON files, and manuscript figures — sufficient to understand, reproduce, and extend the analysis.

**Raw simulation and derived data files are not shared.** These include SDD output files, per-run DSB complexity CSVs, per-run feature JSON files, and the full feature matrix. If you require these for your work, please contact the corresponding author at rfbolo@up.edu.ph.

---

## Repository structure

```
dsb-topology-let-oxygen/
│
├── README.md
├── LICENSE
├── .gitignore
├── environment.yml
├── requirements.txt
│
├── scripts/
│   ├── 01_extract_dsb.py
│   ├── 02_ph_topology_analysis.py
│   ├── 03_compute_features.py
│   ├── 04_build_feature_matrix.py
│   ├── 05_random_forest.py
│   ├── 06_additional_analyses.py
│   ├── 07_regenerate_figures.py
│   ├── 08_partialout_test.py
│   ├── generate_hypoxic_dataset.py
│   └── batch_generate_hypoxic.py
│
├── simulation/
│   ├── electron_0.2.txt       # e⁻, 0.2 keV/µm (6 MV photon surrogate)
│   ├── proton_4.6.txt         # p⁺, proximal SOBP, 4.6 keV/µm
│   ├── proton_8.1.txt         # p⁺, distal SOBP, 8.1 keV/µm
│   ├── helium_10.0.txt        # He²⁺, proximal SOBP, 10.0 keV/µm
│   ├── helium_30.0.txt        # He²⁺, distal SOBP, 30.0 keV/µm
│   ├── carbon_40.9.txt        # C⁶⁺, proximal SOBP, 40.9 keV/µm
│   └── carbon_70.7.txt        # C⁶⁺, distal SOBP, 70.7 keV/µm
│
└── results/
    ├── json/
    │   ├── results_summary.json
    │   ├── ablation_results.json
    │   ├── effect_sizes.json
    │   ├── single_modality_o2_accuracy.json
    │   ├── additional_analyses_summary.json
    │   ├── ph_summary.json
    │   ├── eta2_partialout.json
    │   ├── rf_exclusion_results.json
    │   └── partialout_summary.json
    │
    └── figures/
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

## Simulation design

Each of the seven TOPAS-nBio parameter files produces the **normoxic dataset** (21% O₂) for one particle configuration. Each file is written for a single nucleus rotation; **50 independent runs** are obtained by editing only the seed line:

```
i:Run/Seed = 1   # change to 1–50 for the 50 independent runs
```

The output prefix follows the naming convention `[particle]_[LET]_21.0_[seed]`, for example `helium_30.0_21.0_47`. No other parameters change between runs.

The **six hypoxic conditions** (5.0%, 2.1%, 0.5%, 0.1%, 0.021%, 0.005% O₂) are derived computationally from the normoxic outputs using the VOxA model. No additional TOPAS-nBio simulations are needed for hypoxia:

```bash
python scripts/batch_generate_hypoxic.py
```

---

## Running the pipeline

All scripts are run from the repository root. Steps 1–2 require raw SDD data (available on request); Steps 3–9 can be reproduced from the provided JSON results if you do not have SDD access.

```bash
# Requires raw SDD data (available on request):
python scripts/01_extract_dsb.py
python scripts/batch_generate_hypoxic.py

# Computationally intensive (~8.5 h on Apple M-series, 8 workers):
python scripts/02_ph_topology_analysis.py --workers 8
python scripts/03_compute_features.py
python scripts/04_build_feature_matrix.py

# Random Forest (~7.5 h):
python scripts/05_random_forest.py

# Fast (minutes each):
python scripts/06_additional_analyses.py
python scripts/08_partialout_test.py
python scripts/07_regenerate_figures.py
```

---

## Key results

| Task | Target | Classes | Balanced accuracy |
|------|--------|---------|-------------------|
| 1 | O₂ level, per particle | 7 | 0.189–0.517 |
| 2 | Particle configuration | 7 | 1.000 ± 0.000 |
| 3 | Joint particle–O₂ condition | 49 | 0.346 ± 0.017 |
| 4 | SOBP position (per species) | 2 | 1.000 ± 0.000 |

H₀ Wasserstein-2 separation ratio: **2.399** (within 6.600 / between 15.832)  
m7 H₀ persistent entropy η²_O₂: **0.617**  
m7 η²_O₂ survival ratio after DSB count removal: **0.062**  
m7 BA survival ratio after DSB count removal: **1.011** (5/7 configurations)

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.10 | |
| numpy | ≥ 1.24 | Numerical arrays |
| pandas | ≥ 2.0 | Data frames |
| scipy | ≥ 1.11 | Statistics, OLS |
| scikit-learn | ≥ 1.3 | Random Forest, CV, imputation |
| ripser | ≥ 0.6 | Vietoris-Rips persistent homology |
| persim | ≥ 0.3 | Wasserstein-2 distances |
| matplotlib | ≥ 3.7 | Figures |
| joblib | ≥ 1.3 | Parallel computation |

```bash
conda env create -f environment.yml && conda activate project2-topology
# or
pip install -r requirements.txt
```

---

## Data availability

Raw SDD outputs, per-run DSB CSVs, per-run feature JSONs, and `feature_matrix.csv` are not publicly hosted. Requests for data access for bona fide research purposes may be directed to rfbolo@up.edu.ph.

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

Renato III Fernan Bolo — rfbolo@up.edu.ph  
Department of Physical Sciences and Mathematics, University of the Philippines Manila
