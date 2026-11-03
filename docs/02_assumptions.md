# 02. Modeling Assumptions: Discrete Element Method vs. Graph Neural Network

This document presents a structured, rigorous comparison of the physical and numerical assumptions underlying the Discrete Element Method (DEM) physics engine and the Graph Neural Network (GNN) surrogate architecture.

---

## 1. Discrete Element Method (DEM) Assumptions

The forward physics engine simulates 3D granular mechanics by integrating Newton's second law for discrete spherical particles interacting through non-linear contact forces.

| # | Domain / Component | Physics Engine Implementation | Explicit Assumptions & Physical Limitations |
| :--- | :--- | :--- | :--- |
| **1** | **Particle Geometry & Rheology** | Monodisperse rigid spheres ($R = 5.0\text{ mm}$, $m = 1.309\text{ g}$). | Grains are perfectly spherical and unbreakable. Polydispersity, angularity, and intra-grain fracture or plastic yield are not modeled. |
| **2** | **Rotational Degrees of Freedom** | Translational 3D equations of motion resolved ($3N$ DoF). Torques and angular momentum are omitted. | Rotational degrees of freedom ($\boldsymbol{\omega}_i, \mathbf{I}_i$) are neglected. Grain rolling and twisting resistance are omitted. While this reduces the computational burden per time step, it may underestimate shear dilation in highly packed states. |
| **3** | **Normal Elastic Contact** | Hertzian non-linear elasticity: $F_{\text{Hertz}} = \frac{4}{3} E^* \sqrt{R^*} \delta_n^{1.5}$. | Assumes frictionless, non-conforming, elastic half-space contact under small deformations ($\delta_n \ll R$). Max overlap is maintained $\le 0.05 R$. |
| **4** | **Surface Adhesion Model** | Derjaguin-Müller-Toporov (DMT) contact mechanics: $F_{\text{pull-off}} = 2 \pi R^* \gamma$. | DMT theory assumes the contact profile remains Hertzian and adhesive forces act outside the contact circle as a constant attractive force. This is strictly valid for stiff particles with low adhesion energy (Maugis-Pollock elasticity parameter $\lambda < 0.1$). JKR neck formation and contact hysteresis are not modeled. |
| **5** | **Tangential Contact & Friction** | Regularized Coulomb sliding friction: $\mathbf{F}_t = -\mu F_N \tanh(\|\mathbf{v}_t\| / v_{\text{slip}}) \mathbf{t}$, where $F_N = \max(0, F_n + F_{\text{pull-off}})$. | Sliding friction is regularized to avoid zero-velocity discontinuities. No tangential elastic spring (Mindlin-Deresiewicz compliance) is tracked; hence, microslip and static tangential memory are approximated as sliding friction. |
| **6** | **Viscous Dissipation** | Contact-level normal damping: $F_{\text{damp}} = -\eta_n v_n$ with $\zeta = 0.3579$ ($e_n = 0.30$). | Damping is restricted to the normal contact direction. Drag from interstitial air is neglected (evacuated die chamber assumption). |
| **7** | **Numerical Integration** | Semi-implicit Euler integration with $\Delta t = 5.0 \times 10^{-5}\text{ s}$. | Stable below the Rayleigh critical step ($\Delta t < 0.28 \Delta t_{\text{crit,dense}}$). Energy drift is mitigated by contact damping, though symplectic energy conservation is not strictly guaranteed in dissipative systems. |
| **8** | **Boundary Walls** | Rigid planar walls with Hertzian contact parameters ($E^*_{\text{wall}} = 2 E^*, R^*_{\text{wall}} = R$). | Wall surfaces are perfectly planar and non-deformable. Corner edge singular effects are represented via coordinate bounding checks. |

---

## 2. Graph Neural Network (GNN) Surrogate Assumptions

The GNN surrogate approximates the mapping from the settled particle configuration and loading conditions to macro-mechanical observables and stress responses.

| # | Domain / Component | Surrogate Model Formulation | Inductive Biases & Computational Assumptions |
| :--- | :--- | :--- | :--- |
| **1** | **Graph Abstraction** | Node set $\mathcal{V} = \{1, \dots, N\}$, Edge set $\mathcal{E} = \{(i, j) : \|\mathbf{r}_i - \mathbf{r}_j\| \le r_{\text{cutoff}}\}$. | Spatial interactions are strictly localized within $r_{\text{cutoff}} = 1.05 \times 2R = 0.0105\text{ m}$. Interactions beyond this distance are assumed negligible (consistent with contact mechanics). |
| **2** | **Symmetry & Invariance** | Permutation invariance over particle indices; translation equivariance via relative displacement edge vectors $\Delta \mathbf{r}_{ij} = \mathbf{r}_i - \mathbf{r}_j$. | The surrogate respects identical grain re-indexing without retraining. Rotational symmetries in the horizontal plane are approximated through Euclidean distances and relative vector representations. |
| **3** | **Global Parameter Conditioning** | Macroscopic cohesion $\gamma$, compaction strain $\varepsilon$, grain count $N$, and settled height $H_{\text{settled}}$ are injected into message passing. | Physical response depends strongly on global boundary constraints. Concatenating or modulating edge/node embeddings with global vector $\mathbf{u} = [\gamma, \varepsilon, N, H_{\text{settled}}]$ allows messages to be state-aware throughout graph convolutions. |
| **4** | **Extensivity & Aggregation** | Extensive mechanical properties ($U_{\text{elastic}}$, total contact count) vs. intensive properties ($\rho_b, e, \sigma_{zz}$). | Intensive properties (stresses, densities) are well-modeled by mean-pooling; extensive properties scale with $N$ and volume, requiring sum-pooling or explicit inclusion of $N$ and $H_{\text{settled}}$ to prevent size-dependent prediction bias. |
| **5** | **Generalization Scope** | Evaluation across unseen random packing seeds under 4-fold GroupKFold cross-validation. | The surrogate is evaluated on its ability to generalize to independent geometric realisations of the amorphous packing at identical and interpolated cohesion levels, without spatial graph leakage across folds. |

---

## 3. Methodological Comparison Matrix

| Property | Full-Physics DEM Simulator | GNN Surrogate Model |
| :--- | :--- | :--- |
| **Governing Equations** | Newton-Euler ODEs with non-linear contact force laws | Deep parameterized graph message-passing layers |
| **Temporal Discretization** | Thousands of microsecond time steps ($\Delta t = 50\ \mu\text{s}$) | Single-step forward inference for target state |
| **Computational Scaling** | $\mathcal{O}(N_{\text{steps}} \cdot N_{\text{contacts}})$ where $N_{\text{steps}} \sim 10^4$ | $\mathcal{O}(N_{\text{layers}} \cdot |\mathcal{E}|)$ with $N_{\text{layers}} = 3$ |
| **Resolution** | Full transient particle trajectories and contact forces | Target mechanical observables, stress curves, and contact statistics |
| **Inference Latency** | $\sim 5 - 12\text{ seconds}$ per compaction simulation | $\sim 1 - 5\text{ milliseconds}$ per forward pass on GPU |
| **Conservation Laws** | Conservation of linear momentum guaranteed by pairwise $\mathbf{F}_{ij} = -\mathbf{F}_{ji}$ | Soft constraints learned from empirical training data |

