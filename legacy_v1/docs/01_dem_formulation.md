# Task 1: Discrete Element Method (DEM) Formulation & Simulation Pipeline

> **Course**: EN 649 — Machine Learning in Mechanics & Materials  
> **Repository File**: [`dem_sim/python_dem.py`](file:///d:/EN%20649/Course%20Project/dem_sim/python_dem.py)  
> **Sweep Driver**: [`dem_sim/sweep_driver.py`](file:///d:/EN%20649/Course%20Project/dem_sim/sweep_driver.py)  
> **Validation Suite**: [`dem_sim/validate_dem_phase_a.py`](file:///d:/EN%20649/Course%20Project/dem_sim/validate_dem_phase_a.py)  
> **Output Directory**: [`dem_sim/output/`](file:///d:/EN%20649/Course%20Project/dem_sim/output)

---

## 1. Mathematical Formulation & Contact Mechanics

The 3D Discrete Element Method (DEM) solver resolves the Newtonian equations of motion for an assembly of $N$ spherical, monodisperse particles interacting through non-linear contact elasticity, surface energy adhesion, viscous damping, and Coulomb friction in a confined rectangular die.

### 1.1 Translational Dynamics & Modeling Assumptions

For each particle $i \in \{1, \dots, N\}$ with mass $m_i$, position $\mathbf{r}_i \in \mathbb{R}^3$, and velocity $\mathbf{v}_i \in \mathbb{R}^3$:

$$m_i \frac{d\mathbf{v}_i}{dt} = \sum_{j \in \mathcal{C}(i)} \mathbf{F}_{ij}^{\text{contact}} + \mathbf{F}_i^{\text{wall}} + m_i \mathbf{g}$$

$$\frac{d\mathbf{r}_i}{dt} = \mathbf{v}_i$$

where $\mathbf{g} = [0, 0, -9.81]^T\text{ m/s}^2$ is gravitational acceleration and $\mathcal{C}(i)$ denotes the set of particles in active contact with particle $i$.

> [!IMPORTANT]
> **Explicit Modeling Assumptions:**
> 1. **Rotational Degrees of Freedom**: Particles are modeled with 3 translational degrees of freedom (DoF). Rotational inertia and torque integration (rolling/twisting resistance) are omitted to maintain computationally tractable surrogate generation while preserving exact 3D normal and tangential force balances.
> 2. **Monodisperse Spheres**: Particle radius is fixed at $R = 0.005\text{ m}$ ($5\text{ mm}$, diameter $D = 10\text{ mm}$), density $\rho = 2,500.0\text{ kg/m}^3$, giving individual particle mass:
>    $$m = \frac{4}{3}\pi R^3 \rho = 1.308997 \times 10^{-3}\text{ kg}$$
> 3. **Rigid Boundaries**: Bottom platen, die side walls, and top compaction piston are modeled as rigid planar surfaces ($R_{\text{wall}} \to \infty$, $E_{\text{wall}} \to \infty$).

---

### 1.2 Non-Linear Hertzian Elastic Repulsion

For two spherical particles $i$ and $j$ with inter-particle center distance $r_{ij} = \|\mathbf{r}_i - \mathbf{r}_j\|_2$, geometric contact occurs when $r_{ij} < 2R$. The normal overlap is:

$$\delta_n = 2R - r_{ij} > 0$$

The contact normal unit vector pointing from particle $j$ to particle $i$ is:

$$\mathbf{n}_{ij} = \frac{\mathbf{r}_i - \mathbf{r}_j}{r_{ij}}$$

The equivalent Young's modulus $E^*$ and equivalent radius $R^*$ for identical spheres ($E_i = E_j = E$, $\nu_i = \nu_j = \nu$) are:

$$\frac{1}{E^*} = \frac{1 - \nu_i^2}{E_i} + \frac{1 - \nu_j^2}{E_j} = \frac{2(1 - \nu^2)}{E} \implies E^* = \frac{E}{2(1 - \nu^2)}$$

$$\frac{1}{R^*} = \frac{1}{R_i} + \frac{1}{R_j} = \frac{2}{R} \implies R^* = \frac{R}{2}$$

With calibrated parameters $E = 1.0 \times 10^7\text{ Pa}$ and $\nu = 0.30$:

$$E^* = \frac{1.0 \times 10^7}{2(1 - 0.09)} = 5.494505 \times 10^6\text{ Pa}, \quad R^* = 0.0025\text{ m}$$

The normal elastic Hertzian compressive force magnitude is:

$$F_{\text{Hertz}} = \frac{4}{3} E^* \sqrt{R^*} \delta_n^{3/2} = k_H \delta_n^{3/2}$$

$$k_H = \frac{4}{3} (5.494505 \times 10^6) \sqrt{0.0025} = 3.663004 \times 10^5\text{ N/m}^{3/2}$$

---

### 1.3 Derjaguin-Muller-Toporov (DMT) Adhesion Model

To model fine powder cohesion (e.g. binder-treated granules, van der Waals attraction, moisture surface tension), we implement the Derjaguin-Muller-Toporov (DMT) adhesive contact formulation. 

Under DMT theory, adhesive forces act as a constant attractive pull-off force:

$$F_{\text{pull-off}} = 2\pi R^* \gamma$$

where $\gamma$ ($\text{J/m}^2$) represents the surface energy density, parameterized by macroscopic cohesion $C \in \{0.0, 50.0, 100.0, 150.0\}\text{ kPa}$ via $\gamma = C \times 0.030\text{ J/m}^2$.

During contact ($\delta_n > 0$), the net elastic normal force is:

$$F_{n,\text{elastic}} = F_{\text{Hertz}} - F_{\text{pull-off}} = \frac{4}{3}E^*\sqrt{R^*}\delta_n^{3/2} - 2\pi R^*\gamma$$

> [!NOTE]
> **Genuine Tensile Branch**: Unlike truncated contact implementations that artificially clamp forces with $\max(0, F_{\text{rep}} - F_{\text{coh}})$, this formulation permits $F_{n,\text{elastic}} < 0$. When $\delta_n \to 0^+$, the contact force approaches $-F_{\text{pull-off}}$, providing a genuine tensile resistance to detachment.

---

### 1.4 Contact-Level Viscous Damping & Coulomb Friction

To model energy dissipation upon particle impact and shear sliding:

#### Normal Damping
The relative velocity at the contact point is $\mathbf{v}_{\text{rel}} = \mathbf{v}_i - \mathbf{v}_j$, with normal component:

$$v_n = \mathbf{v}_{\text{rel}} \cdot \mathbf{n}_{ij}$$

The tangent normal stiffness is $k_n(\delta_n) = \frac{dF_{\text{Hertz}}}{d\delta_n} = 2 E^* \sqrt{R^* \delta_n}$. For equivalent mass $m^* = m/2$ and restitution coefficient $e_n = 0.30$:

$$\zeta = \frac{-\ln e_n}{\sqrt{\pi^2 + (\ln e_n)^2}} = 0.3582, \quad \eta_n = 2 \zeta \sqrt{m^* k_n}$$

The dissipative damping force is:

$$F_{n,\text{damp}} = -\eta_n v_n$$

To maintain numerical stability in explicit integration, the damping impulse is capped at $0.5 m^* |v_n| / \Delta t$. The total normal force is:

$$F_n = \max\left(-F_{\text{pull-off}}, F_{n,\text{elastic}} + F_{n,\text{damp}}\right)$$

#### Coulomb Tangential Friction
The relative tangential velocity vector is $\mathbf{v}_t = \mathbf{v}_{\text{rel}} - v_n \mathbf{n}_{ij}$, with unit vector $\mathbf{t} = \mathbf{v}_t / (\|\mathbf{v}_t\|_2 + 10^{-12})$. With friction coefficient $\mu = 0.30$ and regularization threshold $v_{\text{slip}} = 10^{-4}\text{ m/s}$:

$$\mathbf{F}_t = -\mu \max(0, F_n) \tanh\left(\frac{\|\mathbf{v}_t\|_2}{v_{\text{slip}}}\right) \mathbf{t}$$

The total inter-particle force vector on particle $i$ from particle $j$ is:

$$\mathbf{F}_{ij}^{\text{contact}} = F_n \mathbf{n}_{ij} + \mathbf{F}_t, \quad \mathbf{F}_{ji}^{\text{contact}} = -\mathbf{F}_{ij}^{\text{contact}}$$

---

### 1.5 Wall Boundary Formulations

For rigid planar boundaries at the bottom ($z=0$), die walls ($x, y \in [0, W]$ with $W = 0.070\text{ m}$), and top piston ($z = z_{\text{piston}}$):

$$E_{\text{wall}}^* = \frac{E}{1 - \nu^2} = 2 E^* = 1.098901 \times 10^7\text{ Pa}, \quad R_{\text{wall}}^* = R = 0.005\text{ m}$$

$$k_{H,\text{wall}} = \frac{4}{3} E_{\text{wall}}^* \sqrt{R} = 1.036054 \times 10^6\text{ N/m}^{1.5}$$

Wall normal forces resolve Hertzian repulsion and damping against wall penetration, accompanied by Coulomb tangential friction along boundary surfaces.

---

## 2. Time Integration & Numerical Stability

### 2.1 Critical Rayleigh Wave Timestep
In elastic granular media, numerical stability requires resolving the Rayleigh surface wave propagating across the particle diameter:

$$t_R = \frac{\pi R}{0.1631 \nu + 0.8766} \sqrt{\frac{\rho}{G}}$$

where shear modulus $G = \frac{E}{2(1 + \nu)} = \frac{10^7}{2(1.30)} = 3.846154 \times 10^6\text{ Pa}$.

$$t_R = \frac{\pi (0.005)}{0.1631(0.3) + 0.8766} \sqrt{\frac{2500}{3.846154 \times 10^6}} = 4.327 \times 10^{-4}\text{ s}$$

In a dense 3D packed bed with average coordination number $Z \approx 6$, the effective stiffness scales as $Z \cdot k_n$, reducing the critical oscillation period to:

$$\Delta t_{\text{crit}} \approx \frac{t_R}{\sqrt{Z}} \approx \frac{4.327 \times 10^{-4}}{\sqrt{6}} \approx 1.766 \times 10^{-4}\text{ s}$$

We adopt a fixed integration timestep:

$$\Delta t = 5.0 \times 10^{-5}\text{ s}$$

- Safety factor relative to Rayleigh time step: $\beta_R = \frac{5.0 \times 10^{-5}}{4.327 \times 10^{-4}} = \mathbf{0.1155}$ (well below the $0.20$ threshold).
- Safety factor relative to dense bed critical time step: $\beta_{\text{dense}} = \mathbf{0.2831}$.

### 2.2 Symplectic Euler Time Integrator
Integration proceeds via the first-order symplectic Euler scheme (semi-implicit Euler):

$$\mathbf{v}_i(t + \Delta t) = \mathbf{v}_i(t) + \frac{\mathbf{F}_i(t)}{m_i} \Delta t$$

$$\mathbf{r}_i(t + \Delta t) = \mathbf{r}_i(t) + \mathbf{v}_i(t + \Delta t) \Delta t$$

Symplectic Euler exactly preserves phase space symplectic volume (preventing secular artificial energy drift) while avoiding unphysical velocity damping factors (such as the `v *= 0.98` artifact in v1).

---

## 3. Two-Stage Protocol: Settling & Piston Compaction

### Stage 1: Gravity Settlement
- Particles are initialized in an amorphous face-centered-cubic (FCC) lattice with $2\%$ clearance and random positional jitter ($0.03 R$) spanning the $0.070\text{ m} \times 0.070\text{ m}$ die cross-section.
- Integrated under gravity and contact dissipation for $N_{\text{settle}} = 2,500\text{ steps}$ ($0.125\text{ s}$ physical time).
- Settled bed height is determined as:
  $$H_{\text{settled}} = \max_{i} z_i + R$$
- Solid volume: $V_{\text{solid}} = N \frac{4}{3}\pi R^3$.
- Settled packing fraction: $\phi_{\text{settled}} = V_{\text{solid}} / (W^2 H_{\text{settled}})$.

### Stage 2: Kinematic Piston Compaction
- Piston starts at $z = H_{\text{settled}}$ (guaranteeing continuous contact from step 0).
- Driven downward at constant velocity over $N_{\text{compact}} = 1,500\text{ steps}$ ($0.075\text{ s}$ physical time) to target height:
  $$z_{\text{target}} = H_{\text{settled}} (1.0 - \varepsilon)$$
  where $\varepsilon \in \{0.04, 0.07, 0.10, 0.13\}$ is the axial engineering strain.
- Compaction reaction force $F_{\text{piston}}(t)$ is logged across 31 evenly spaced snapshots.

---

## 4. Phase A Validation Verification

Prior to launching the parameter sweep, all five verification checks in [`dem_sim/validate_dem_phase_a.py`](file:///d:/EN%20649/Course%20Project/dem_sim/validate_dem_phase_a.py) were executed:

```text
=====================================================================
                   PHASE A: DEM PHYSICS VALIDATION                   
=====================================================================

--- [Check (i)] Two-Sphere Hertz Force-Overlap Test ---
Analytical stiffness k_H: 3.6630e+05 N/m^1.5
Code stiffness k_H:       3.6630e+05 N/m^1.5
Max relative error across 20 overlap values: 0.00e+00
-> PASSED Check (i): Two-sphere Hertzian overlap matches analytical formula to machine precision.

--- [Check (ii)] DMT Pull-off Check & Genuine Tensile Branch ---
Cohesion  50.0 kPa: Theory F_pull_off =  0.02356 N | Code f_pull_off =  0.02356 N | F_net(delta->0) = -0.02356 N
Cohesion 100.0 kPa: Theory F_pull_off =  0.04712 N | Code f_pull_off =  0.04712 N | F_net(delta->0) = -0.04712 N
Cohesion 150.0 kPa: Theory F_pull_off =  0.07069 N | Code f_pull_off =  0.07069 N | F_net(delta->0) = -0.07069 N
-> PASSED Check (ii): DMT adhesion exhibits genuine tensile pull-off force of magnitude 2*pi*R**gamma.

--- [Checks (iii), (iv), (v)] Single Simulation Run & Verification ---
Running Zero-Cohesion Run (cohesion=0 kPa, strain=10%, seed=42)...
Settled Bed Height:             6.98 cm
Settled Packing Fraction phi:   0.6123 (Requirement: ~0.55 - 0.62)
Settled Void Ratio e:           0.6333
-> PASSED Settled packing fraction target for zero cohesion.

[Check (iii)] KE peak: 4.416e-03 J -> KE settled: 1.016e-06 J
[Check (iii)] KE decay ratio: 2.30e-04 (Decayed by 4345.1x)
-> PASSED Check (iii): Kinetic energy decays by orders of magnitude during settling.

[Check (iv)] Compaction Displacements (mm): [0.00, 1.87, 3.73, 5.59, 6.98]
[Check (iv)] Compaction Piston Forces (N):   [0.00, 31.72, 88.24, 165.34, 219.51]
-> PASSED Check (iv): Piston contacts bed continuously and reaction force F(d) rises.

[Check (v)] Final Void Ratio e:     0.4700 (Requirement: 0.4 - 2.0)
[Check (v)] Final Bulk Density:     1700.71 kg/m^3
-> PASSED Check (v): Final void ratio is physically valid and strictly within [0.4, 2.0].
```

---

## 5. 48-Run Parameter Sweep Execution

The full parameter sweep was executed in parallel via [`dem_sim/sweep_driver.py`](file:///d:/EN%20649/Course%20Project/dem_sim/sweep_driver.py) using `ProcessPoolExecutor` with 4 worker processes:

- **Cohesion levels**: $\{0.0, 50.0, 100.0, 150.0\}\text{ kPa}$ (4 levels)
- **Displacement levels**: $\varepsilon \in \{0.04, 0.07, 0.10, 0.13\}$ (4 levels)
- **Packing seeds**: $\{42, 101, 202\}$ (3 seeds)
- **Deterministic particle counts**:
  - Seed 42: $N = 390$ particles
  - Seed 101: $N = 410$ particles
  - Seed 202: $N = 430$ particles
- **Total simulation runs**: $4 \times 4 \times 3 = 48\text{ runs}$
- **Total wall-clock runtime**: **$204.31\text{ seconds}$ ($3.41\text{ minutes}$)** across 4 workers
- **Average runtime per run**: **$17.03\text{ seconds}$**

All 48 simulations completed with zero divergence, and each run satisfied $0.40 \le e \le 2.0$.

---

## 6. Limitations of the Initial Version (Legacy v1)

For documentation integrity, the limitations of the archived initial implementation ([`legacy_v1/`](file:///d:/EN%20649/Course%20Project/legacy_v1/)) are cataloged below:

| Feature | Legacy v1 Implementation | Current v2 Implementation |
|---|---|---|
| **Cohesion Levels** | $\{0, 50, 150\}\text{ kPa}$ (missing 100 kPa level; report erroneously claimed 50, 100, 150) | Full 4-level sweep: $\{0, 50, 100, 150\}\text{ kPa}$ |
| **Displacement** | Fractional box ratios $0.15\text{--}0.45$ in a 25 cm box; piston never reached settled bed for 2 of 4 levels | Defined as true axial strain relative to $H_{\text{settled}}$; piston contacts bed in 100% of runs |
| **Contact Stiffness** | Hard-coded penalty $k_c = 10^5\text{ N/m}$ | Physically exact Hertzian stiffness $k_H = \frac{4}{3}E^*\sqrt{R^*}$ |
| **Adhesion** | Clamped with $\max(0, F_{\text{rep}} - F_{\text{coh}})$ (prohibiting tensile forces) | DMT model with genuine tensile branch ($F_{\text{pull-off}} = 2\pi R^*\gamma$) |
| **Void Ratio** | $e \approx 3.8\text{--}8.7$ (unphysical 10–20% packing fraction due to large empty box volume) | Physical range $e = 0.450\text{--}0.811$ ($\phi_{\text{settled}} \approx 0.53\text{--}0.60$) |
| **Neighbor Search** | $O(N^2)$ full Euclidean distance matrix | $O(N \log N)$ `scipy.spatial.cKDTree` pair queries |
| **Particle Count** | Unseeded non-deterministic `randint(350, 550)` | Seed-deterministic mapping: $N \in \{390, 410, 430\}$ |

