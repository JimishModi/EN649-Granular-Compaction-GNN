# High-Fidelity 3D Granular DEM Compaction and Physics-Conditioned Graph Neural Network Surrogates

[![Course](https://img.shields.io/badge/Course-EN%20649%20ML%20in%20Mechanics%20%26%20Materials-0F2C59)](EN649_Project_Report.pdf)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20PyG-EE4C2C)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%204060%20(8GB)-76B900)](.)
[![Speedup](https://img.shields.io/badge/Compaction%20Speedup-101.4%C3%97-brightgreen)](.)

> **EN 649: Machine Learning in Mechanics & Materials — Final Course Project**  
> Complete implementation of a verified 3D Discrete Element Method (DEM) compaction simulation engine (Hertzian elasticity + DMT cohesion with a genuine tensile branch + exact viscous normal damping + regularized Coulomb sliding friction), continuous constant-velocity parameter sweeps, leak-free grouped cross-validation, capacity-matched Graph Neural Network (GNN) surrogates, DeepMind Graph Network Simulator (GNS) baseline, temporal convergence analysis, and publication-ready reporting.

---

## 🔬 Project Overview

Uniaxial die compaction of cohesive powders is a core manufacturing process across pharmaceutical tableting, powder metallurgy, ceramic sintering, and battery electrode manufacturing. While the **Discrete Element Method (DEM)** provides microscale fidelity by resolving grain-scale equations of motion alongside non-linear Hertzian contact elasticity and Derjaguin-Müller-Toporov (DMT) surface energy adhesion, its reliance on small Rayleigh time integration ($\Delta t = 50\ \mu\text{s}$) makes large parameter sweeps computationally demanding.

In this work, we demonstrate through mathematical derivation and numerical verification that macroscopic bulk density $\rho_b$ and void ratio $e$ are closed-form kinematic identities of $(N, H_{\text{settled}}, \varepsilon)$ under rigid container boundaries ($R^2 = 1.0000$). Consequently, we establish that geometric properties serve solely as sanity checks. True surrogate evaluation must focus on contact-mechanics observables: **peak compaction force $F_{\text{peak}}$**, **continuous force-displacement profiles $F(\varepsilon)$**, coordination numbers, and stored elastic energy.

We design and evaluate a **physics-conditioned Graph Neural Network (ImprovedGNN)** that injects the global state vector $\mathbf{u} = [\gamma, \varepsilon, N, H_{\text{settled}}]$ directly into message passing and node update operations, benchmarked under a leak-free 4-fold `GroupKFold` cross-validation protocol grouped strictly by random seed.

### 🌟 Key Results
- **Contact Force Generalization**: ImprovedGNN achieves **$R^2 = 0.8968 \pm 0.0690$** on continuous force curves $F(\varepsilon)$ and **$R^2 = 0.5261 \pm 0.2290$** on peak compaction force $F_{\text{peak}}$, cutting peak force RMSE nearly in half compared to decoupled baseline GNNs ($8.66\text{ N}$ vs $16.47\text{ N}$).
- **Rigorous Timing Accounting**: Serial uncontented hardware benchmarks on an Intel Core i7 CPU and NVIDIA RTX 4060 GPU demonstrate a **$101.4\times$ speedup** during forward compaction inference ($1.72\text{ ms}$ vs $2.97\text{ s}$) and an honest **$2.55\times$ end-to-end acceleration** including physical settling.
- **Direct Regression vs. Rollouts**: Benchmarking against Google DeepMind's Graph Network Simulator (GNS) reveals that while single-step position error is low ($0.108\text{ mm}$, ~1% particle diameter), multi-step autoregressive rollouts compound errors ($0.616\text{ mm}$), causing boundary penetrations ($R^2 < 0$). Direct static graph prediction completely avoids temporal accumulation and respects container boundaries.
- **Empirical Learning Curve**: Training across $N_{\text{runs}} \in \{8, 16, 24\}$ demonstrates steep sample efficiency gains, with test peak force $R^2$ scaling from $-4.61$ to $+0.36$ to $\mathbf{+0.72}$ (RMSE decreasing from $23.06\text{ N}$ to $5.19\text{ N}$).

---

## 📊 4-Fold Grouped Cross-Validation Benchmarks

All models share identical capacity ($d_{\text{hidden}} = 64$) and training-only normalization across 4 disjoint test folds (8 unseen compaction test runs per fold).

| Model Architecture | Conditioning Mechanism | Peak Force $F_{\text{peak}}$ $R^2$ | Force Curve $F(\varepsilon)$ $R^2$ | Bulk Density $\rho_b$ $R^2$ | Void Ratio $e$ $R^2$ | Mechanics Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Closed-Form Formula** | Exact Kinematic Identity | — | — | **$1.0000 \pm 0.0000$** | **$1.0000 \pm 0.0000$** | Exact kinematic identity in $(N, H_{\text{settled}}, \varepsilon)$ |
| **Mean Predictor Baseline** | Training Split Mean | $-2.2313 \pm 2.2283$ | $-0.1936 \pm 0.2227$ | $-0.4299 \pm 0.4225$ | $-0.4478 \pm 0.4561$ | Uninformed mean baseline |
| **Linear Ridge Regression** | Ridge on Global $\mathbf{u}$ | $-0.6206 \pm 0.5069$ | $0.3852 \pm 0.1226$ | $0.4142 \pm 0.1181$ | $0.4167 \pm 0.1211$ | Linear fit on $[N, H_{\text{settled}}, \varepsilon, \gamma]$ |
| **BaselineGNN** (Decoupled) | Readout Head MLP | $-0.6860 \pm 1.4995$ | $0.6249 \pm 0.1361$ | $0.8502 \pm 0.0899$ | $0.8424 \pm 0.0947$ | Decoupled message passing |
| **ImprovedGNN** (Proposed) | Conditioned Message Passing | **$0.5261 \pm 0.2290$** | **$0.8968 \pm 0.0690$** | $0.9753 \pm 0.0194$ | $0.9750 \pm 0.0205$ | Physics-conditioned MP + dual pooling |

---

## ⏱️ Computational Hardware Timing Breakdown

Benchmarked in serial, uncontented execution on dedicated hardware (Intel Core i7 CPU, NVIDIA GeForce RTX 4060 GPU with CUDA synchronization and warm-up):

| Execution Stage | Numerical DEM Simulator | GNN Surrogate Pipeline | Acceleration Factor |
| :--- | :---: | :---: | :---: |
| **Stage 1: Bed Settling / Graph Construction** | $1.873\text{ s}$ | $0.47\text{ ms}$ | $3984\times$ |
| **Stage 2: Compaction / GPU Forward Pass** | $2.970\text{ s}$ | $1.72\text{ ms}$ (per step) | **$101.4\times$** |
| **Full 17-Point Compaction Response Curve** | $2.970\text{ s}$ | $29.76\text{ ms}$ | **$99.8\times$** |
| **End-to-End Pipeline (Settling + Compaction)** | $4.843\text{ s}$ | $1.903\text{ s}$ | **$2.55\times$** |

---

## 📈 Visual Analytics & Convergence Diagnostics

All figures are automatically generated by analysis scripts in [`analysis/plots/`](analysis/plots):

| Time-Step ($\Delta t$) Numerical Convergence | Empirical Sample Efficiency Learning Curve |
| :---: | :---: |
| ![Time Step Convergence](analysis/plots/dt_convergence.png) | ![Learning Curve](analysis/plots/learning_curve.png) |
| **Held-Out Force Profiles $F(\varepsilon)$** | **Held-Out Density Profiles $\rho_b(\varepsilon)$** |
| ![Held-Out Force](analysis/plots/heldout_force_displacement.png) | ![Held-Out Density](analysis/plots/heldout_density_strain.png) |

---

## 📁 Repository Structure

```
.
├── EN649_Project_Report.pdf              # Generated publication-quality 4-page PDF report
├── EN649_Final_Project_Report.md         # Comprehensive markdown final project report
├── README.md                             # Repository overview and documentation
├── docs/                                 # Consolidated technical documentation
│   ├── 01_physical_system.md             # Parameter table, contact mechanics & 32-run sanity table
│   ├── 02_assumptions.md                 # DEM vs GNN modeling assumptions & regime boundaries
│   ├── 03_gnn_architecture.md            # Graph representations, network topologies & CV tables
│   └── 04_temporal_analysis.md            # Time step convergence, held-out curves & learning curves
├── analysis/                             # Evaluation and plotting scripts
│   ├── plots/                            # High-resolution figures (PNG, 300 DPI)
│   ├── generate_pdf_report.py            # PDF report generator using ReportLab
│   ├── temporal_analysis.py              # dt convergence and held-out trajectory evaluator
│   ├── evaluate_gns.py                   # GNS baseline rollout evaluator
│   ├── temporal_convergence_results.json # Verified dt convergence numerical results
│   └── learning_curve_results.json       # Measured sample efficiency metrics
├── dem_sim/                              # Discrete Element Method simulation engine
│   ├── python_dem.py                     # High-fidelity Hertz-DMT-Coulomb DEM solver
│   ├── sweep_driver.py                   # Continuous 32-run parameter sweep orchestrator
│   ├── test_dem_physics.py               # Analytical unit test for Hertz & DMT contact
│   └── output/                           # 32 completed simulation runs (metadata + npz)
├── models/                               # GNN surrogate architectures & training
│   ├── baseline_gnn.py                   # Decoupled graph neural network (64-dim)
│   ├── improved_gnn.py                   # Physics-conditioned message passing network (64-dim)
│   ├── dataset.py                        # Spatial contact graph dataset builder
│   ├── train.py                          # 4-Fold GroupKFold training engine
│   ├── checkpoints/                      # Saved PyTorch model weights
│   ├── cv_benchmark_results.json         # Exact cross-validation benchmark metrics
│   ├── timing_benchmarks.json            # Verified serial latency accounting
│   └── gns_results.json                  # Retrained GNS rollout metrics
├── dataset/                              # Formatted graph tensors and metadata
│   ├── compaction_data.npz               # Compiled snapshot dataset (544 graphs)
│   ├── metadata.json                     # Split definitions and chamber metadata
│   ├── extract_dataset.py                # Dataset extraction script from dem_sim/output
│   ├── convert_to_gns_format.py          # Exporter to geoelements/gns format
│   └── gns/                              # GNS-formatted dataset splits
├── gns_baseline/                         # DeepMind Graph Network Simulator baseline
├── legacy_v1/                            # Preserved initial trial checkpoints and outputs
├── dataset_v2a/                          # Fallback dataset backup
├── dem_output_v2a/                       # Fallback DEM output backup
├── scratch/                              # Temporary inspection and execution scripts
│   ├── generate_sanity_table.py          # Sanity check table generator
│   ├── measure_honest_timing.py          # Hardware latency benchmarking script
│   └── run_learning_curve.py             # Sample efficiency training script
└── archive/                              # Reversible archive of superseded files
    ├── analysis/                         # Superseded Phase A scripts & notes
    ├── dem_sim/                          # Early single-run prototype & validation suite
    └── reports/                          # Early draft skeleton and report copy
```

---

## 🚀 How to Reproduce

All numbers, figures, and metrics in this report are fully reproducible from repository scripts. To execute the end-to-end pipeline from scratch:

### 1. Environment Setup
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric reportlab scikit-learn matplotlib pypdf
```

### 2. Verify DEM Contact Mechanics
Execute the analytical unit tests for Hertzian contact and DMT pull-off force:
```powershell
python dem_sim/test_dem_physics.py
```

### 3. Run the 32-Run DEM Parameter Sweep
Execute the continuous constant-velocity compaction sweep across 4 cohesion levels $\times$ 8 packing seeds:
```powershell
python dem_sim/sweep_driver.py
```

### 4. Extract Dataset & Build Spatial Contact Graphs
Compile raw DEM simulation outputs into graph format with shared cutoff $r_{\text{cutoff}} = 0.0105\text{ m}$:
```powershell
python dataset/extract_dataset.py
python dataset/convert_to_gns_format.py
```

### 5. Run 4-Fold Grouped Cross-Validation Benchmarks
Train Closed-Form, Mean Predictor, Linear Ridge Regression, BaselineGNN, and ImprovedGNN across 4 seed-grouped folds:
```powershell
python models/train.py
```

### 6. Run Temporal Dynamics & Time Step Convergence Analysis
Benchmark multi-scale time-step convergence ($\Delta t \in \{50, 25, 12.5\}\ \mu\text{s}$) and generate held-out trajectory plots:
```powershell
python analysis/temporal_analysis.py
```

### 7. Run Sample Efficiency Learning Curve
Train ImprovedGNN on data subsets ($N_{\text{runs}} \in \{8, 16, 24\}$) to generate the learning curve:
```powershell
python scratch/run_learning_curve.py
```

### 8. Benchmark Serial Hardware Latency
Measure uncontented CPU and GPU execution times with CUDA synchronization and warm-up:
```powershell
python scratch/measure_honest_timing.py
```

### 9. Compile Publication PDF Report
Generate the final 4-page publication-quality PDF report embedding all verified tables and figures:
```powershell
python analysis/generate_pdf_report.py
```

---

## 📄 License & Attribution

Developed by **Jimish Modi** for **EN 649: Machine Learning in Mechanics & Materials** at **Johns Hopkins University**, November 2026.
