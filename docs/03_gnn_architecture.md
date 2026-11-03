# 03. Graph Neural Network Architectures & Surrogate Formulation

This document details the machine learning formulation, graph representation, capacity-matched neural network architectures, and cross-validation benchmarking protocol for the granular compaction surrogate models.

---

## 1. Graph Construction & Feature Representation

Each compaction snapshot is formulated as an attributed spatial graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X}, \mathbf{E}, \mathbf{u})$.

### 1.1 Node Features ($\mathbf{x}_i \in \mathbb{R}^5$)
For each grain $i \in \{1, \dots, N\}$, node features encode local coordinates and wall proximity:
$$\mathbf{x}_i = \left[ x_i, \, y_i, \, z_i, \, d_{\text{bot},i}, \, d_{\text{lat},i} \right]$$
where:
- $(x_i, y_i, z_i)$ are particle coordinates in meters
- $d_{\text{bot},i} = z_i - R$ is the distance to the bottom container boundary
- $d_{\text{lat},i} = \min\left(x_i - R, W - R - x_i, y_i - R, W - R - y_i\right)$ is the distance to the nearest lateral die wall

### 1.2 Edge Construction & Edge Features ($\mathbf{e}_{ij} \in \mathbb{R}^5$)
To ensure consistent geometric representation across PyG models and GNS, edges are established using a unified physical contact cutoff:
$$r_{\text{cutoff}} = 1.05 \times 2R = 0.0105\text{ m}$$
An edge $(i, j) \in \mathcal{E}$ exists whenever $\|\mathbf{r}_i - \mathbf{r}_j\| \le r_{\text{cutoff}}$.

The edge feature vector $\mathbf{e}_{ij}$ encodes relative geometric displacement and overlap mechanics:
$$\mathbf{e}_{ij} = \left[ \Delta x_{ij}, \, \Delta y_{ij}, \, \Delta z_{ij}, \, r_{ij}, \, \frac{\delta_{n,ij}}{R} \right]$$
where:
- $\Delta \mathbf{r}_{ij} = \mathbf{r}_i - \mathbf{r}_j$
- $r_{ij} = \|\Delta \mathbf{r}_{ij}\|$
- $\delta_{n,ij} = \max(0, 2R - r_{ij})$ is the geometric normal overlap
- $\delta_{n,ij} / R$ is the normalized overlap fraction (zero for non-touching proximity neighbors)

### 1.3 Global Conditioning Vector ($\mathbf{u} \in \mathbb{R}^4$)
The global state vector encodes thermodynamic material properties and kinematic boundary constraints:
$$\mathbf{u} = \left[ \gamma, \, \varepsilon, \, N, \, H_{\text{settled}} \right]$$
where:
- $\gamma$ is the thermodynamic work of adhesion ($\text{J/m}^2$), representing cohesion
- $\varepsilon = (H_{\text{settled}} - z_{\text{piston}}) / H_{\text{settled}}$ is the axial engineering strain
- $N$ is the deterministic particle count of the bed ($N \in [380, 450]$)
- $H_{\text{settled}}$ is the settled bed height in meters

---

## 2. Capacity-Matched Neural Network Architectures

Both models are strictly configured with identical latent dimensional capacity ($d_{\text{hidden}} = 64$) and optimized under matching hyperparameters (Adam optimizer, initial learning rate $1 \times 10^{-3}$, weight decay $1 \times 10^{-4}$, Cosine Annealing learning rate schedule over 100 epochs).

### 2.1 BaselineGNN (Standard Decoupled GNN)
The BaselineGNN implements standard graph convolutions where global conditions $\mathbf{u}$ are excluded from message passing and concatenated solely at the final prediction readout head:

1. **Input Encoding**:
   $$\mathbf{h}_i^{(0)} = \text{MLP}_{\text{node}}(\mathbf{x}_i) \in \mathbb{R}^{64}, \quad \mathbf{e}_{ij}^{(0)} = \text{MLP}_{\text{edge}}(\mathbf{e}_{ij}) \in \mathbb{R}^{64}$$
2. **Message Passing (3 Layers)**:
   For $l = 0, 1, 2$:
   $$\mathbf{m}_{ij}^{(l)} = \text{MLP}_{\text{msg}}^{(l)}\left([\mathbf{h}_i^{(l)}, \, \mathbf{h}_j^{(l)}, \, \mathbf{e}_{ij}^{(0)}]\right)$$
   $$\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \text{MLP}_{\text{update}}^{(l)}\left([\mathbf{h}_i^{(l)}, \, \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(l)}]\right)$$
3. **Graph Readout & Prediction**:
   $$\mathbf{h}_{\text{graph}} = \frac{1}{N} \sum_{i=1}^N \mathbf{h}_i^{(3)}$$
   $$\hat{\mathbf{y}} = \text{MLP}_{\text{head}}\left([\mathbf{h}_{\text{graph}}, \, \mathbf{u}]\right)$$

### 2.2 ImprovedGNN (Physics-Conditioned Message Passing with Dual Pooling)
The ImprovedGNN explicitly injects the global state $\mathbf{u}$ into every message generation and node update step, allowing the neural message channels to modulate force transmission based on cohesion $\gamma$ and compaction strain $\varepsilon$:

1. **Input Encoding**:
   $$\mathbf{h}_i^{(0)} = \text{MLP}_{\text{node}}(\mathbf{x}_i) \in \mathbb{R}^{64}, \quad \mathbf{e}_{ij}^{(0)} = \text{MLP}_{\text{edge}}(\mathbf{e}_{ij}) \in \mathbb{R}^{64}, \quad \mathbf{h}_u = \text{MLP}_{\text{global}}(\mathbf{u}) \in \mathbb{R}^{64}$$
2. **Conditioned Message Passing (3 Layers)**:
   For $l = 0, 1, 2$:
   $$\mathbf{m}_{ij}^{(l)} = \text{MLP}_{\text{cond\_msg}}^{(l)}\left([\mathbf{h}_i^{(l)}, \, \mathbf{h}_j^{(l)}, \, \mathbf{e}_{ij}^{(0)}, \, \mathbf{h}_u]\right)$$
   $$\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \text{MLP}_{\text{cond\_node}}^{(l)}\left([\mathbf{h}_i^{(l)}, \, \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(l)}, \, \mathbf{h}_u]\right)$$
3. **Dual-Scale Pooling & Specialized Heads**:
   Combines mean pooling (for intensive observables) and sum pooling (for extensive mechanical quantities):
   $$\mathbf{h}_{\text{mean}} = \frac{1}{N} \sum_{i=1}^N \mathbf{h}_i^{(3)}, \quad \mathbf{h}_{\text{sum}} = \sum_{i=1}^N \mathbf{h}_i^{(3)}$$
   $$\mathbf{h}_{\text{combined}} = [\mathbf{h}_{\text{mean}}, \, \frac{1}{N_{\text{ref}}} \mathbf{h}_{\text{sum}}, \, \mathbf{h}_u]$$
   Specialized sub-heads predict mechanical targets, stress curves, and sanity checks.

---

## 3. Evaluation Targets & Baseline Models

### 3.1 Target Decomposition

| Target Variable | Symbol | Type | Physical Significance |
| :--- | :--- | :--- | :--- |
| **Peak Wall Compaction Force** | $F_{\text{peak}}$ | Contact Mechanics | Peak resistance force (N) transmitted to piston |
| **Peak Axial Stress** | $\sigma_{zz,\text{peak}}$ | Contact Mechanics | $F_{\text{peak}} / W^2$ (kPa) |
| **Coordination Number** | $Z$ | Microstructural | Mean contacting neighbors per grain ($2 |\mathcal{E}_c| / N$) |
| **Tensile Contact Fraction** | $f_{\text{tensile}}$ | Cohesion Mechanics | Fraction of contacts under negative (tensile) normal load |
| **Stored Elastic Energy** | $U_{\text{elastic}}$ | Energetic | Cumulative elastic strain energy stored in Hertzian contacts |
| **Final Bulk Density** | $\rho_b$ | Geometric / Sanity | Bed mass per current volume (closed form known) |
| **Final Void Ratio** | $e$ | Geometric / Sanity | Ratio of void volume to solid volume (closed form known) |

### 3.2 Reference Baselines
1. **Mean Baseline**: Always predicts the training split mean for each target.
2. **Linear Regression Baseline**: Multi-target Ridge regression fitted on global features $[N, H_{\text{settled}}, \varepsilon, \gamma]$.
3. **Closed-Form Kinematic Formula**:
   $$\rho_b^{\text{exact}} = \frac{N m}{W^2 H_{\text{settled}} (1 - \varepsilon)}, \quad e^{\text{exact}} = \frac{W^2 H_{\text{settled}} (1 - \varepsilon) - N v_p}{N v_p}$$
   Evaluated to establish that geometric targets are kinematically exact ($R^2 = 1.0$), demonstrating why true surrogate evaluation must focus on non-trivial contact mechanics.

---

## 4. Grouped Cross-Validation Protocol

To guarantee zero graph leakage between training and testing splits:
- All 32 runs are grouped strictly by random seed: `[11, 22, 33, 44, 55, 66, 77, 88]`.
- A 4-fold `GroupKFold` cross-validation split partitions seeds into 4 disjoint test folds (2 seeds per fold, 8 compaction runs per fold).
- All normalization statistics (node feature mean/std, target mean/std) are computed **strictly from the training split of each fold**.
- All reported metrics ($R^2$, RMSE, MAE) are calculated as the mean and standard deviation across the 4 validation folds.

---

## 5. 4-Fold Grouped Cross-Validation Benchmark Results

The table below presents the verified cross-validation performance across all 4 folds for each baseline and surrogate model. All metrics report the ensemble mean $\pm$ one standard deviation across the 4 validation folds (8 unseen compaction test runs per fold).

| Model Architecture | Compaction Target | Test $R^2$ ($\text{mean} \pm \text{std}$) | Test RMSE ($\text{mean} \pm \text{std}$) | Test MAE ($\text{mean} \pm \text{std}$) | Notes / Mechanics Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Closed-Form Formula** | Peak Force $F_{\text{peak}}$ | — | — | — | Exact closed form does not predict contact force |
| | Force Curve $F(\varepsilon)$ | — | — | — | Exact closed form does not predict contact force |
| | Bulk Density $\rho_b$ | **$1.0000 \pm 0.0000$** | $(4.66 \pm 0.34) \times 10^{-5}\text{ kg/m}^3$ | $(3.83 \pm 0.25) \times 10^{-5}\text{ kg/m}^3$ | Kinematic identity in $(N, H_{\text{settled}}, \varepsilon)$ |
| | Void Ratio $e$ | **$1.0000 \pm 0.0000$** | $(4.86 \pm 0.97) \times 10^{-8}$ | $(3.99 \pm 0.92) \times 10^{-8}$ | Kinematic identity in $(N, H_{\text{settled}}, \varepsilon)$ |
| **Mean Predictor Baseline** | Peak Force $F_{\text{peak}}$ | $-2.2313 \pm 2.2283$ | $21.32 \pm 5.88\text{ N}$ | $18.66 \pm 5.93\text{ N}$ | Trivial mean baseline |
| | Force Curve $F(\varepsilon)$ | $-0.1936 \pm 0.2227$ | $18.00 \pm 4.53\text{ N}$ | $14.60 \pm 2.47\text{ N}$ | Trivial mean baseline |
| | Bulk Density $\rho_b$ | $-0.4299 \pm 0.4225$ | $77.84 \pm 11.22\text{ kg/m}^3$ | $64.41 \pm 8.35\text{ kg/m}^3$ | Trivial mean baseline |
| | Void Ratio $e$ | $-0.4478 \pm 0.4561$ | $0.1125 \pm 0.0163$ | $0.0929 \pm 0.0121$ | Trivial mean baseline |
| **Linear Ridge Regression** | Peak Force $F_{\text{peak}}$ | $-0.6206 \pm 0.5069$ | $17.15 \pm 4.29\text{ N}$ | $14.30 \pm 3.19\text{ N}$ | Linear fit on $[N, H_{\text{settled}}, \varepsilon, \gamma]$ |
| | Force Curve $F(\varepsilon)$ | $0.3852 \pm 0.1226$ | $13.41 \pm 4.77\text{ N}$ | $10.14 \pm 3.34\text{ N}$ | Linear fit on $[N, H_{\text{settled}}, \varepsilon, \gamma]$ |
| | Bulk Density $\rho_b$ | $0.4142 \pm 0.1181$ | $50.32 \pm 6.79\text{ kg/m}^3$ | $41.34 \pm 5.08\text{ kg/m}^3$ | Linear fit on $[N, H_{\text{settled}}, \varepsilon, \gamma]$ |
| | Void Ratio $e$ | $0.4167 \pm 0.1211$ | $0.0725 \pm 0.0115$ | $0.0596 \pm 0.0091$ | Linear fit on $[N, H_{\text{settled}}, \varepsilon, \gamma]$ |
| **BaselineGNN** (Decoupled) | Peak Force $F_{\text{peak}}$ | $-0.6860 \pm 1.4995$ | $16.47 \pm 7.44\text{ N}$ | $13.91 \pm 6.95\text{ N}$ | Decoupled graph convolution ($64$-dim) |
| | Force Curve $F(\varepsilon)$ | $0.6249 \pm 0.1361$ | $9.74 \pm 1.69\text{ N}$ | $7.39 \pm 0.69\text{ N}$ | Decoupled graph convolution ($64$-dim) |
| | Bulk Density $\rho_b$ | $0.8502 \pm 0.0899$ | $24.30 \pm 9.58\text{ kg/m}^3$ | $21.69 \pm 8.78\text{ kg/m}^3$ | Decoupled graph convolution ($64$-dim) |
| | Void Ratio $e$ | $0.8424 \pm 0.0947$ | $0.0362 \pm 0.0152$ | $0.0319 \pm 0.0135$ | Decoupled graph convolution ($64$-dim) |
| **ImprovedGNN** (Conditioned) | Peak Force $F_{\text{peak}}$ | **$0.5261 \pm 0.2290$** | **$8.66 \pm 1.36\text{ N}$** | **$6.84 \pm 1.13\text{ N}$** | Physics-conditioned message passing + dual pooling |
| | Force Curve $F(\varepsilon)$ | **$0.8968 \pm 0.0690$** | **$4.82 \pm 0.80\text{ N}$** | **$3.24 \pm 0.60\text{ N}$** | Physics-conditioned message passing + dual pooling |
| | Bulk Density $\rho_b$ | $0.9753 \pm 0.0194$ | $9.49 \pm 4.35\text{ kg/m}^3$ | $7.40 \pm 4.06\text{ kg/m}^3$ | Physics-conditioned message passing + dual pooling |
| | Void Ratio $e$ | $0.9750 \pm 0.0205$ | $0.0138 \pm 0.0071$ | $0.0108 \pm 0.0068$ | Physics-conditioned message passing + dual pooling |

### 5.1 Mechanistic Insights
1. **Kinematic Sanity Targets**: Bulk density $\rho_b$ and void ratio $e$ are solved exactly by closed-form geometric baselines ($R^2 = 1.0000$). While neural models achieve high fits ($\rho_b$ $R^2 = 0.9753$, $e$ $R^2 = 0.9750$), these targets reflect kinematic chamber volume changes rather than complex contact mechanics.
2. **Contact Force Generalization**: Peak compaction force $F_{\text{peak}}$ is the primary hard contact mechanics target. The decoupled BaselineGNN fails completely on unseen configurations ($R^2 = -0.6860 \pm 1.50$), performing worse than the linear regression baseline ($R^2 = -0.6206$). In contrast, ImprovedGNN achieves positive, generalizable predictive power ($R^2 = 0.5261 \pm 0.2290$), cutting RMSE nearly in half ($8.66\text{ N}$ vs $16.47\text{ N}$).
3. **Conditioned Message Passing Impact**: By injecting $\mathbf{u} = [\gamma, \varepsilon, N, H_{\text{settled}}]$ directly into edge message generation and node updates, ImprovedGNN accurately tracks force chain intensification across the compaction trajectory, yielding $R^2 = 0.8968 \pm 0.0690$ on instantaneous axial force $F(\varepsilon)$.


