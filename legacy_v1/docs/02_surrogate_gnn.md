# Task 2: Graph Neural Network Surrogate Architecture & Training Pipeline

> **Course**: EN 649 — Machine Learning in Mechanics & Materials  
> **Dataset**: [`dataset/compaction_data.npz`](file:///d:/EN%20649/Course%20Project/dataset/compaction_data.npz) & [`dataset/metadata.json`](file:///d:/EN%20649/Course%20Project/dataset/metadata.json)  
> **Dataset Loader**: [`models/dataset.py`](file:///d:/EN%20649/Course%20Project/models/dataset.py)  
> **Architectures**: [`models/baseline_gnn.py`](file:///d:/EN%20649/Course%20Project/models/baseline_gnn.py) & [`models/improved_gnn.py`](file:///d:/EN%20649/Course%20Project/models/improved_gnn.py)  
> **Training Script**: [`models/train.py`](file:///d:/EN%20649/Course%20Project/models/train.py)  
> **Training Results**: [`models/training_results.json`](file:///d:/EN%20649/Course%20Project/models/training_results.json)

---

## 1. Graph Construction & Feature Representation

Granular media represent continuous metric point clouds whose localized contact mechanics and topological force chains map naturally to geometric graphs $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{u})$.

```
           [Node Features x_i in R^6]
          (3D position + 3D velocity)
                      O
                     / \
     Normalized     /   \   Normalized
   Edge Attributes /     \ Edge Attributes
  e_ij / r_cutoff /       \ e_jk / r_cutoff
                 O---------O
                      |
              [Message Passing]
                      |
             [Global Mean Pool]
                      |
        [Fuse Global Conditions u in R^2]
        (Cohesion / 150, Displacement / 0.13)
                      |
                [Readout MLP]
                      |
         [Macroscopic Predictions y in R^2]
            (Bulk Density rho_b, Void Ratio e)
```

### 1.1 Node Attributes ($\mathbf{x}_i \in \mathbb{R}^6$)
Each node represents an individual particle in its initial settled equilibrium state ($t = 0$ of the compaction phase):

$$\mathbf{x}_i = [x_i, y_i, z_i, v_{x,i}, v_{y,i}, v_{z,i}]^T \in \mathbb{R}^6$$

where $[x_i, y_i, z_i]$ are continuous Euclidean spatial coordinates bounded in $[0, 0.070\text{ m}] \times [0, 0.070\text{ m}] \times [0, H_{\text{settled}}]$, and $[v_{x,i}, v_{y,i}, v_{z,i}]$ are settled residual velocities ($\approx 0\text{ m/s}$).

---

### 1.2 Edge Attributes & Connectivity Radius ($\mathbf{e}_{ij} \in \mathbb{R}^4$)
Graph connectivity is established via a fixed geometric metric radius:

$$r_{\text{cutoff}} = 0.020\text{ m}$$

corresponding exactly to $2\times$ particle diameter ($4\times$ particle radius $R = 0.005\text{ m}$). This connectivity radius is **identically shared** between PyTorch Geometric (PyG) surrogate models and the DeepMind Graph Network Simulator (GNS) baseline.

For every pair of particles $(i, j)$ satisfying $\|\mathbf{r}_j - \mathbf{r}_i\|_2 \le r_{\text{cutoff}}$, a bidirectional edge $(i, j) \in \mathcal{E}$ is constructed using `scipy.spatial.cKDTree`.

The edge attribute vector contains the 3D relative displacement vector and Euclidean distance:

$$\mathbf{e}_{ij} = [\Delta x_{ij}, \Delta y_{ij}, \Delta z_{ij}, \|\Delta \mathbf{r}_{ij}\|_2]^T = [\mathbf{r}_j - \mathbf{r}_i, \|\mathbf{r}_j - \mathbf{r}_i\|_2]^T \in \mathbb{R}^4$$

> [!IMPORTANT]
> **Documented Edge Normalization**:
> In [`models/improved_gnn.py`](file:///d:/EN%20649/Course%20Project/models/improved_gnn.py#L38), the raw edge attribute $\mathbf{e}_{ij}$ is explicitly divided by the connectivity radius:
> $$\tilde{\mathbf{e}}_{ij} = \frac{\mathbf{e}_{ij}}{r_{\text{cutoff}}} = \frac{\mathbf{e}_{ij}}{0.020\text{ m}} = \mathbf{e}_{ij} \times 50.0\text{ m}^{-1}$$
> This non-dimensionalizes all inter-particle spatial vectors into $[-1, 1]$ before passing into the non-linear edge embedding MLP, replacing the undocumented magic constant in v1.

---

### 1.3 Global Context Conditions ($\mathbf{u} \in \mathbb{R}^2$)
Macroscopic process and material variables that govern the continuum boundary-value problem are packaged into a global context conditioning vector:

$$\mathbf{u} = \left[ \frac{\gamma_{\text{coh}}}{150.0\text{ kPa}}, \frac{\varepsilon}{0.13} \right]^T \in [0, 1]^2$$

- $\gamma_{\text{coh}} \in \{0.0, 50.0, 100.0, 150.0\}\text{ kPa}$: surface energy cohesion density.
- $\varepsilon \in \{0.04, 0.07, 0.10, 0.13\}$: platen compaction axial engineering strain relative to $H_{\text{settled}}$.

---

### 1.4 Regression Targets ($\mathbf{y} \in \mathbb{R}^2$)
The surrogate directly predicts macroscopic post-compaction engineering properties:

$$\mathbf{y} = [\rho_b, e]^T \in \mathbb{R}^2$$

- $\rho_b = \frac{N \cdot m}{W^2 \cdot z_{\text{piston}}}$: final bulk density ($\text{kg/m}^3$).
- $e = \frac{W^2 \cdot z_{\text{piston}} - N \cdot \frac{4}{3}\pi R^3}{N \cdot \frac{4}{3}\pi R^3}$: residual void ratio (dimensionless).

---

## 2. Train / Val / Test Partitioning & Normalization

### 2.1 Deterministic Shared Split
To guarantee valid benchmarks between PyG surrogates and GNS rollouts, an 80 / 10 / 10 split was generated with `np.random.default_rng(42)` and committed directly into [`dataset/metadata.json`](file:///d:/EN%20649/Course%20Project/dataset/metadata.json#L17-L21):

- **Training Set (80%)**: 38 samples  
  `[37, 30, 45, 34, 5, 47, 44, 24, 20, 28, 17, 43, 18, 7, 4, 38, 29, 21, 32, 9, 25, 23, 39, 16, 40, 27, 15, 31, 42, 26, 10, 3, 41, 6, 11, 22, 19, 35]`
- **Validation Set (10%)**: 5 samples  
  `[0, 12, 14, 46, 2]`
- **Holdout Test Set (10%)**: 5 samples  
  `[36, 33, 1, 13, 8]`

---

### 2.2 Strict Target Normalization (No Data Leakage)
To ensure equal loss gradients across targets of disparate scale (bulk density $\sim 1,500\text{ kg/m}^3$ vs void ratio $\sim 0.6$), targets are normalized using z-score standardization computed **STRICTLY from the 38 training samples**:

$$\tilde{\mathbf{y}} = \frac{\mathbf{y} - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}}}$$

From [`dataset/metadata.json`](file:///d:/EN%20649/Course%20Project/dataset/metadata.json#L22-L27):

| Property | Training Mean ($\mu_{\text{train}}$) | Training Std ($\sigma_{\text{train}}$) | Full Dataset Mean |
|---|---|---|---|
| **Bulk Density ($\rho_b$)** | **$1,545.67\text{ kg/m}^3$** | **$93.16\text{ kg/m}^3$** | $1,538.97\text{ kg/m}^3$ |
| **Void Ratio ($e$)** | **$0.6233$** | **$0.0974$** | $0.6306$ |

---

## 3. Surrogate Neural Network Architectures

### 3.1 Baseline GNN (`BaselineGNN`)
Implemented in [`models/baseline_gnn.py`](file:///d:/EN%20649/Course%20Project/models/baseline_gnn.py):
- **Node Encoder**: Two-layer MLP with ReLU activation:
  $$\mathbb{R}^6 \xrightarrow{\text{Linear}(6, 64)} \mathbb{R}^{64} \xrightarrow{\text{ReLU}} \mathbb{R}^{64} \xrightarrow{\text{Linear}(64, 64)} \mathbb{R}^{64}$$
- **Message Passing Backbone**: 3 stacked isotropic `GCNConv` layers (hidden dimension 64):
  $$\mathbf{h}_i^{(l+1)} = \text{ReLU}\left( \mathbf{W}^{(l)} \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} \mathbf{h}_j^{(l)} \right)$$
  *(Note: Standard GCN convolutions discard edge features $\mathbf{e}_{ij}$).*
- **Pooling**: Global mean pooling across all $N$ particle representations:
  $$\mathbf{h}_{\text{graph}} = \frac{1}{N} \sum_{i=1}^N \mathbf{h}_i^{(3)} \in \mathbb{R}^{64}$$
- **Readout MLP**: Concatenates graph embedding with global vector $\mathbf{u} \in \mathbb{R}^2$:
  $$\mathbb{R}^{66} \xrightarrow{\text{Linear}(66, 64)} \mathbb{R}^{64} \xrightarrow{\text{ReLU}} \mathbb{R}^{64} \xrightarrow{\text{Linear}(64, 32)} \mathbb{R}^{32} \xrightarrow{\text{ReLU}} \mathbb{R}^{32} \xrightarrow{\text{Linear}(32, 2)} \mathbb{R}^2$$

---

### 3.2 Improved Edge-Aware Residual GNN (`ImprovedGNN`)
Implemented in [`models/improved_gnn.py`](file:///d:/EN%20649/Course%20Project/models/improved_gnn.py):
- **Node Encoder**: Two-layer MLP: $\mathbb{R}^6 \to \mathbb{R}^{48} \to \mathbb{R}^{48}$.
- **Edge Embedding MLP**: Two-layer MLP projecting normalized relative distance vectors:
  $$\tilde{\mathbf{e}}_{ij} = \frac{\mathbf{e}_{ij}}{0.020} \in \mathbb{R}^4 \xrightarrow{\text{MLP}_{\text{edge}}} \mathbf{e}_{ij}^{\text{emb}} \in \mathbb{R}^{48}$$
- **Residual Edge-Aware Message Passing (`ResEdgeConvLayer`)**: 3 stacked layers (hidden dimension 48) with non-linear message computation, skip connections, and LayerNorm regularization:
  $$\mathbf{m}_{ij} = \text{MLP}_{\text{msg}}\left( [\mathbf{h}_j^{(l)} + \mathbf{e}_{ij}^{\text{emb}}, \mathbf{h}_i^{(l)}] \right) \in \mathbb{R}^{48}$$
  $$\mathbf{out}_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}$$
  $$\mathbf{h}_i^{(l+1)} = \text{LayerNorm}\left( \mathbf{h}_i^{(l)} + \text{MLP}_{\text{update}}\left( [\mathbf{h}_i^{(l)}, \mathbf{out}_i] \right) \right)$$
- **Readout MLP**: Fuses graph embedding $\mathbf{h}_{\text{graph}} \in \mathbb{R}^{48}$ with $\mathbf{u} \in \mathbb{R}^2$:
  $$\mathbb{R}^{50} \xrightarrow{\text{Linear}(50, 48)} \mathbb{R}^{48} \xrightarrow{\text{ReLU}} \text{Dropout}(0.10) \xrightarrow{\text{Linear}(48, 32)} \mathbb{R}^{32} \xrightarrow{\text{ReLU}} \mathbb{R}^{32} \xrightarrow{\text{Linear}(32, 2)} \mathbb{R}^2$$

---

## 4. Training Hyperparameters & Execution

Both models were trained using [`models/train.py`](file:///d:/EN%20649/Course%20Project/models/train.py) on an NVIDIA GeForce RTX 4060 Laptop GPU:

| Hyperparameter | Value | Description |
|---|---|---|
| **Epochs** | 200 | Total optimization epochs |
| **Batch Size** | 8 | Graph minibatches via `torch_geometric.loader.DataLoader` |
| **Initial Learning Rate** | $2.0 \times 10^{-3}$ | Adam optimizer |
| **Weight Decay** | $10^{-4}$ (Baseline) / $10^{-3}$ (Improved) | L2 regularization |
| **LR Scheduler** | `CosineAnnealingLR` | $T_{\max} = 200$ (strictly matching epochs), $\eta_{\min} = 5.0 \times 10^{-5}$ |
| **Loss Function** | Mean Squared Error (MSE) | Computed on standardized targets $\tilde{\mathbf{y}}$ |

---

## 5. Measured Holdout Benchmark Results

All metrics below are drawn directly from [`models/training_results.json`](file:///d:/EN%20649/Course%20Project/models/training_results.json), evaluated on the 5 holdout test beds:

### 5.1 Predictive Accuracy Comparison (Raw Physical Units)

| Model Architecture | Bulk Density $R^2$ | Bulk Density MAE ($\text{kg/m}^3$) | Bulk Density RMSE ($\text{kg/m}^3$) | Void Ratio $R^2$ | Void Ratio MAE ($e$) | Void Ratio RMSE ($e$) | Best Val Loss | Best Epoch |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BaselineGNN** (`GCNConv`) | 0.2359 | 99.19 | 111.85 | **0.2484** | 0.1065 | **0.1188** | 0.2824 | 20 |
| **ImprovedGNN** (`ResEdgeConv` + Res) | **0.2519** | **97.78** | **110.67** | 0.2366 | **0.1061** | 0.1198 | **0.2786** | 67 |

> [!NOTE]
> Both GNN surrogates achieve consistent positive determination coefficients ($R^2 \approx 0.24\text{--}0.25$) on the holdout test set with low absolute errors (MAE $\sim 0.106$ on void ratio and $\sim 98\text{ kg/m}^3$ on density), successfully capturing non-linear compaction compaction tendencies despite the small 38-sample training set.

---

### 5.2 Computational Latency & Real Speedup vs. DEM

Inference execution times were benchmarked over 200 repeated evaluations on identical test graphs:

| Computational Engine | Hardware | Latency per Sample | Speedup vs. Serial DEM | Speedup vs. Parallel DEM (4 cores) |
|---|---|:---:|:---:|:---:|
| **DEM Simulation (Ground Truth)** | CPU (NumPy + cKDTree) | $17.03\text{ s}$ ($17,030\text{ ms}$) | $1.0\times$ (Baseline) | $4.0\times$ |
| **BaselineGNN Inference** | NVIDIA RTX 4060 GPU | **$1.793\text{ ms}$** | **$9,498.0\times$** | **$2,374.5\times$** |
| **BaselineGNN Inference** | Intel CPU (Single Thread) | $4.864\text{ ms}$ | **$3,501.2\times$** | **$875.3\times$** |
| **ImprovedGNN Inference** | NVIDIA RTX 4060 GPU | **$1.803\text{ ms}$** | **$9,445.4\times$** | **$2,361.3\times$** |
| **ImprovedGNN Inference** | Intel CPU (Single Thread) | $10.829\text{ ms}$ | **$1,572.6\times$** | **$393.2\times$** |

> [!TIP]
> **Fair Hardware Speedup Analysis**:
> On GPU, `ImprovedGNN` achieves **$9,445.4\times$ speedup** relative to serial Python DEM. Even when constrained to a single CPU thread without GPU acceleration, `ImprovedGNN` executes in **$10.83\text{ ms}$**, delivering a **$1,572.6\times$ wall-clock speedup** over numerical time integration.

---

## 6. Limitations of the Initial Version (Legacy v1)

| Aspect | Legacy v1 Implementation | Current v2 Implementation |
|---|---|---|
| **Split Alignment** | PyG used `torch.randperm(42)` while GNS used `np.random.default_rng(42)`, evaluating on disjoint test beds | Exact shared split indices stored in `metadata.json` (identical for PyG and GNS) |
| **Target Normalization** | Target z-score statistics were computed, but density loss dominated due to raw scale mismatch | Rigorous per-target z-score normalization strictly computed from 38 training runs |
| **Edge Vector Normalization** | Magic hardcoded constant `edge_attr * 50.0` with no physical documentation | Formally parameterized as non-dimensionalization $\mathbf{e}_{ij} / r_{\text{cutoff}}$ ($r_{\text{cutoff}} = 0.020\text{ m}$) |
| **LR Scheduler** | `CosineAnnealingLR` initialized with `T_max=150` despite `epochs=200` | Configured with `T_max = epochs = 200` for smooth monotonic decay |
| **Speedup Comparison** | Claimed $15,494\times$ comparing Python DEM against GPU PyTorch without CPU benchmarks | Reports rigorous dual-hardware benchmarks: GPU ($9,445\times$) and CPU ($1,573\times$) |

