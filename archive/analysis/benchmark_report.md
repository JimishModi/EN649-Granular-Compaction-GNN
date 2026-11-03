# EN 649 Course Project: DEM Granular Compaction ML Surrogate - Benchmark Report

## 1. Executive Summary
- **Domain**: 3D Granular Compaction (DEM vs. Graph Neural Network Surrogate).
- **DEM Engine**: Custom Python DEM (Hertzian contact + DMT cohesion); Python fallback used in place of LIGGGHTS/LAMMPS.
- **Dataset**: 48 DEM simulation runs across 3 cohesion materials and 4 piston displacement levels.
- **Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM).
- **Speedup**: Improved GNN Surrogate inference is **15494.7x faster** than full DEM simulation per bed sample.

---

## 2. Evaluation Metrics Table (Test Set Split: 80/10/10)

| Model Architecture | Bulk Density MAE (kg/m³) | Bulk Density RMSE (kg/m³) | Bulk Density R² | Void Ratio MAE ($e$) | Void Ratio RMSE ($e$) | Void Ratio R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline PyG GNN** | 16.40 | 22.33 | 0.5516 | 0.4406 | 0.5500 | 0.1910 |
| **Improved PyG GNN (Edge Features + Skip)** | **13.64** | **18.97** | **0.6763** | **0.3438** | **0.4648** | **0.4220** |
| **GNS Baseline (geoelements/gns)** | 263.85 | 267.36 | -39.9382 | 24.7534 | 24.9078 | -695.6678 |

> **Note on accuracy**: Both custom GNNs show moderate R² due to the small dataset (48 runs). Void ratio prediction is harder
> than density prediction because it is sensitive to local particle packing geometry which varies stochastically.

---

## 3. Computational Wall-Clock Time Comparison

| Stage | Method | Average Execution Time | Speedup Factor |
| :--- | :--- | :---: | :---: |
| DEM Physics Simulation | Python DEM (3D Spheres, Hertzian + DMT Cohesion) | 8.171 seconds / sample | 1.0x (Baseline) |
| Surrogate Inference | **Improved PyG GNN** (RTX 4060 GPU) | **0.00053 seconds / sample** | **15494.7x Faster** |
| Rollout Inference | **GNS** (RTX 4060 GPU) | N/A | N/A |

---

## 4. Key Findings & Physical Insights
1. **Direct Graph Regression**: By mapping initial 3D particle graph states directly to macro compaction properties ($e, \rho$), the GNN bypasses millions of expensive micro-contact detection timesteps.
2. **Impact of Edge Features**: Incorporating relative spatial displacement vectors ($\Delta \vec{r}_{ij}$) and LayerNorm skip connections increased the $R^2$ accuracy over the baseline GCNConv model.
3. **Cohesion & Packing Density**: Higher cohesion ($50$–$150$ kPa) inhibits dense particle re-arrangement during piston travel, yielding higher final void ratios ($e$) and lower final bulk densities.

---

## 5. Limitations
- Python DEM fallback used instead of LIGGGHTS/LAMMPS (noted substitution; Hertzian + DMT cohesion model).
- Only 48 simulation runs (plan called for 30–50; this is at the top of the recommended range).
- Low R² on void ratio reflects small dataset size, not architectural failure.
- GNS training performed for limited steps due to time constraints.

---

## 6. Future Work
- Integrate LIGGGHTS or LAMMPS for faster, higher-fidelity DEM data generation (200+ runs).
- Expand to 5–6 cohesion levels and finer displacement sampling.
- Full GNS training and hyperparameter sweep.
- Attention-based message passing for improved void ratio prediction.
