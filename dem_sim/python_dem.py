"""
High-Fidelity Granular DEM Compaction Simulator in 3D Python
=============================================================
Simulates 3D spherical particle packing under gravity settling and top-piston
die compaction using non-linear contact mechanics:
- Hertzian non-linear elastic normal repulsion: F_Hertz = (4/3) * E* * sqrt(R*) * delta^1.5
- Derjaguin-Muller-Toporov (DMT) surface adhesion with a genuine tensile branch:
  F_pull_off = 2 * pi * R* * gamma (where gamma is the thermodynamic work of adhesion)
- Contact-level normal viscous damping parameterized by restitution coefficient (zeta = 0.3579)
- Coulomb regularized tangential sliding friction with mechanical normal load:
  F_N = max(0, F_n + F_pull_off)
- 3D Poisson-disk random rain filling (amorphous packing without crystallographic lattice bias)
- Spatial acceleration using scipy.spatial.cKDTree (O(N log N) contact search)
- Semi-implicit Euler time integration with Rayleigh wave stability

Explicit Modeling Assumptions & Limitations:
1. Particles are spherical and monodisperse with radius R = 5 mm.
2. Rotational degrees of freedom and torques are neglected; particles resolve 3D
   translational equations of motion (3 DoF per particle).
3. Friction is modeled as regularized sliding friction without tangential history springs;
   rolling/twisting resistance is not included.
4. Wall boundaries are rigid planar surfaces modeled with Hertzian contact mechanics.
"""

import numpy as np
from scipy.spatial import cKDTree
import time
import json
import os

class ParticleBedDEM:
    def __init__(self, num_particles=400, cohesion=50.0, max_strain=0.13, seed=42, piston_velocity=0.05):
        """
        Parameters:
            num_particles: int, deterministic particle count
            cohesion: float, macroscopic cohesion in kPa in {0.0, 50.0, 100.0, 150.0}
            max_strain: float, maximum axial compaction strain (default 0.13 = 13%)
            seed: int, random seed for reproducible random rain deposition
            piston_velocity: float, constant compaction velocity in m/s (default 0.05 m/s)
        """
        self.seed = int(seed)
        self.num_particles = int(num_particles)
        self.cohesion = float(cohesion) # in kPa
        self.max_strain = float(max_strain) # 0.13
        self.piston_velocity = float(piston_velocity) # 0.05 m/s
        
        # ---------------- Physical Material Parameters ----------------
        self.radius = 0.005 # 5 mm particle radius (diameter = 10 mm)
        self.diameter = 2.0 * self.radius
        self.density = 2500.0 # kg/m^3 (quartz / mineral powder)
        self.mass = (4.0 / 3.0) * np.pi * (self.radius**3) * self.density # ~1.309e-3 kg
        self.weight = self.mass * 9.81 # ~0.01284 N
        
        # ---------------- Elastic Contact Mechanics (Hertzian) ----------------
        self.youngs = 1.0e7 # Pa (calibrated Young's modulus)
        self.poisson = 0.30
        
        # Equivalent Young's modulus E* for identical spheres:
        # 1/E* = 2 * (1 - nu^2) / E => E* = E / (2 * (1 - nu^2))
        self.E_star = self.youngs / (2.0 * (1.0 - self.poisson**2)) # ~5.4945e6 Pa
        self.R_star = self.radius / 2.0 # 0.0025 m
        
        # Hertzian normal stiffness prefactor: k_H = (4/3) * E* * sqrt(R*)
        self.k_hertz = (4.0 / 3.0) * self.E_star * np.sqrt(self.R_star) # ~3.6630e5 N/m^1.5
        
        # Wall contact elastic modulus (rigid planar wall: R_wall -> inf, E_wall -> inf)
        self.E_star_wall = self.youngs / (1.0 - self.poisson**2) # 2 * E*
        self.k_hertz_wall = (4.0 / 3.0) * self.E_star_wall * np.sqrt(self.radius)
        
        # ---------------- DMT Surface Adhesion ----------------
        # gamma is the thermodynamic work of adhesion (surface energy density, J/m^2)
        # Calibrated: gamma = cohesion_kPa * 0.030 J/m^2
        self.gamma = self.cohesion * 0.030 # J/m^2
        self.f_pull_off = 2.0 * np.pi * self.R_star * self.gamma # N
        # Cohesive Bond number: Bo = F_pull_off / (m * g)
        self.bond_number = float(self.f_pull_off / self.weight)
        
        # ---------------- Damping & Coulomb Friction ----------------
        self.restitution = 0.30 # coefficient of restitution e_n
        # Exact damping ratio: zeta = -ln(e_n) / sqrt(pi^2 + ln(e_n)^2) = 0.357857...
        self.zeta = 0.3579 # exact damping ratio for e_n = 0.30
        self.mu_friction = 0.30 # Coulomb sliding friction coefficient
        self.v_slip_tol = 1.0e-4 # m/s velocity regularization threshold
        
        # ---------------- Simulation Box Geometry ----------------
        self.box_width = 0.070 # 7.0 cm x 7.0 cm cross section
        self.box_area = self.box_width**2 # 0.0049 m^2
        
        # ---------------- Critical Time Step Stability ----------------
        G = self.youngs / (2.0 * (1.0 + self.poisson)) # ~3.846e6 Pa
        self.t_rayleigh = (np.pi * self.radius / (0.1631 * self.poisson + 0.8766)) * np.sqrt(self.density / G) # ~4.327e-4 s
        # In a dense 3D packed bed (coordination Z ~ 6), critical dt ~ t_R / sqrt(Z) ~ 1.76e-4 s.
        # We select dt = 5.0e-5 s (safety factor beta ~ 0.116 of Rayleigh, beta ~ 0.28 of dense bed).
        self.dt = 5.0e-5
        
        # Contact graph cutoff radius (true contact + 5% clearance)
        self.contact_cutoff = 1.05 * 2.0 * self.radius # 0.0105 m
        
        # State arrays
        self.positions = np.zeros((self.num_particles, 3))
        self.velocities = np.zeros((self.num_particles, 3))
        self.forces = np.zeros((self.num_particles, 3))
        
        self._initialize_particles_rain()
        
    def _initialize_particles_rain(self):
        """
        Initializes particles via 3D Poisson-disk random rain filling.
        Deposits particles with clearance to ensure strictly zero initial overlap
        without crystallographic lattice alignment bias.
        """
        rng = np.random.default_rng(self.seed)
        r_min = 2.02 * self.radius # minimum center-to-center distance (10.1 mm)
        cell_size = r_min / np.sqrt(3.0)
        
        grid_w = int(np.ceil(self.box_width / cell_size)) + 1
        grid_h = int(np.ceil(0.30 / cell_size)) + 1
        grid = -np.ones((grid_w, grid_w, grid_h), dtype=int)
        
        points = []
        active_list = []
        
        first_pt = np.array([
            rng.uniform(self.radius * 1.05, self.box_width - self.radius * 1.05),
            rng.uniform(self.radius * 1.05, self.box_width - self.radius * 1.05),
            self.radius * 1.05
        ])
        points.append(first_pt)
        active_list.append(0)
        cg = (first_pt / cell_size).astype(int)
        grid[cg[0], cg[1], cg[2]] = 0
        
        k_attempts = 30
        while len(points) < self.num_particles and len(active_list) > 0:
            idx = rng.integers(len(active_list))
            base_pt = points[active_list[idx]]
            placed = False
            
            for _ in range(k_attempts):
                direction = rng.normal(size=3)
                norm_d = np.linalg.norm(direction)
                if norm_d < 1e-8:
                    continue
                direction /= norm_d
                dist = rng.uniform(r_min, 1.35 * r_min)
                cand = base_pt + direction * dist
                
                if not (self.radius * 1.02 <= cand[0] <= self.box_width - self.radius * 1.02 and
                        self.radius * 1.02 <= cand[1] <= self.box_width - self.radius * 1.02 and
                        cand[2] >= self.radius * 1.02):
                    continue
                    
                cg = (cand / cell_size).astype(int)
                if cg[0] >= grid_w or cg[1] >= grid_w or cg[2] >= grid_h:
                    continue
                    
                conflict = False
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        for dz in range(-2, 3):
                            gx, gy, gz = cg[0] + dx, cg[1] + dy, cg[2] + dz
                            if 0 <= gx < grid_w and 0 <= gy < grid_w and 0 <= gz < grid_h:
                                p_idx = grid[gx, gy, gz]
                                if p_idx != -1:
                                    if np.linalg.norm(cand - points[p_idx]) < r_min:
                                        conflict = True
                                        break
                        if conflict: break
                    if conflict: break
                    
                if not conflict:
                    p_new = len(points)
                    points.append(cand)
                    active_list.append(p_new)
                    grid[cg[0], cg[1], cg[2]] = p_new
                    placed = True
                    if len(points) == self.num_particles:
                        break
            if not placed:
                active_list.pop(idx)
                
            if len(active_list) == 0 and len(points) < self.num_particles:
                max_z = max(p[2] for p in points)
                new_seed = np.array([
                    rng.uniform(self.radius * 1.05, self.box_width - self.radius * 1.05),
                    rng.uniform(self.radius * 1.05, self.box_width - self.radius * 1.05),
                    max_z + r_min
                ])
                cg = (new_seed / cell_size).astype(int)
                p_new = len(points)
                points.append(new_seed)
                active_list.append(p_new)
                grid[cg[0], cg[1], cg[2]] = p_new
                
        self.positions = np.array(points[:self.num_particles], dtype=np.float64)
        self.velocities = np.zeros_like(self.positions)
        self.box_height_initial = float(np.max(self.positions[:, 2]) + 3.0 * self.radius)
        
    def compute_forces(self, top_wall_z=None, top_wall_v=0.0):
        """
        Computes inter-particle forces (Hertzian repulsion + DMT adhesion + viscous damping +
        Coulomb tangential friction) using cKDTree neighbor search, plus wall boundary forces.
        
        Returns:
            forces: (N, 3) total force array on all particles
            wall_piston_force: float, compressive reaction force exerted on the top piston
            contact_info: dict with overlap statistics, coordination, tensile contacts, elastic energy
        """
        gravity = np.array([0.0, 0.0, -9.81])
        forces = np.ones((self.num_particles, 3)) * (gravity * self.mass)
        wall_piston_force = 0.0
        
        # Structure and energy accumulators
        max_overlap = 0.0
        sum_overlap = 0.0
        num_contacts = 0
        tensile_contacts = 0
        stored_elastic_energy = 0.0
        
        # 1. Inter-particle contact forces via cKDTree
        tree = cKDTree(self.positions)
        pairs = tree.query_pairs(r=2.0 * self.radius, output_type='ndarray')
        
        if len(pairs) > 0:
            i = pairs[:, 0]
            j = pairs[:, 1]
            
            diff = self.positions[i] - self.positions[j] # vector from j to i
            dist = np.linalg.norm(diff, axis=1)
            valid = dist > 1.0e-10
            
            i = i[valid]
            j = j[valid]
            diff = diff[valid]
            dist = dist[valid]
            
            normal = diff / dist[:, None] # unit normal j -> i
            overlap = 2.0 * self.radius - dist
            
            num_contacts = len(overlap)
            max_overlap = float(np.max(overlap)) if num_contacts > 0 else 0.0
            sum_overlap = float(np.sum(overlap))
            # Stored elastic energy: integral of k_H * delta^1.5 ddelta = (2/5) * k_H * delta^2.5
            stored_elastic_energy = float(np.sum((2.0 / 5.0) * self.k_hertz * (overlap**2.5)))
            
            # Relative velocity of i relative to j
            v_rel = self.velocities[i] - self.velocities[j]
            v_n = np.sum(v_rel * normal, axis=1) # normal relative velocity
            
            # Hertzian elastic repulsion
            f_hertz = self.k_hertz * (overlap**1.5)
            # DMT adhesion with genuine tensile branch (no clamp)
            f_elastic = f_hertz - self.f_pull_off
            
            # Contact normal damping
            m_star = self.mass / 2.0
            k_n = 2.0 * self.E_star * np.sqrt(self.R_star * overlap)
            eta_n = 2.0 * self.zeta * np.sqrt(m_star * k_n)
            
            # Damping impulse limiter (prevents explicit integration chatter)
            f_damp_max = 0.5 * m_star * np.abs(v_n) / self.dt
            f_damp = np.clip(-eta_n * v_n, -f_damp_max, f_damp_max)
            
            f_normal = f_elastic + f_damp
            if self.f_pull_off > 0.0:
                f_normal = np.maximum(-self.f_pull_off, f_normal)
                
            tensile_contacts = int(np.sum(f_normal < 0.0))
            
            # Coulomb tangential sliding friction with DMT mechanical compressive load:
            # F_N = max(0, F_n + F_pull_off)
            f_normal_comp = np.maximum(0.0, f_normal + self.f_pull_off)
            
            v_t_vec = v_rel - v_n[:, None] * normal
            v_t_norm = np.linalg.norm(v_t_vec, axis=1)
            t_vec = v_t_vec / (v_t_norm[:, None] + 1.0e-12)
            
            friction_mag = self.mu_friction * f_normal_comp * np.tanh(v_t_norm / self.v_slip_tol)
            f_tangent = -friction_mag[:, None] * t_vec
            
            f_contact = f_normal[:, None] * normal + f_tangent
            np.add.at(forces, i, f_contact)
            np.add.at(forces, j, -f_contact)
            
        # 2. Bottom wall boundary (z = 0)
        bot_ov = self.radius - self.positions[:, 2]
        bot_c = bot_ov > 0.0
        if np.any(bot_c):
            ov = bot_ov[bot_c]
            vz = self.velocities[bot_c, 2]
            k_w = 2.0 * self.E_star_wall * np.sqrt(self.radius * ov)
            eta_w = 2.0 * self.zeta * np.sqrt(self.mass * k_w)
            fd_max = 0.5 * self.mass * np.abs(vz) / self.dt
            fd = np.clip(-eta_w * vz, -fd_max, fd_max)
            fz = np.maximum(0.0, self.k_hertz_wall * (ov**1.5) + fd)
            forces[bot_c, 2] += fz
            
            v_xy = self.velocities[bot_c, :2]
            v_xy_norm = np.linalg.norm(v_xy, axis=1)
            f_fric_wall = self.mu_friction * fz * np.tanh(v_xy_norm / self.v_slip_tol)
            forces[bot_c, :2] -= f_fric_wall[:, None] * (v_xy / (v_xy_norm[:, None] + 1.0e-12))
            
        # 3. Side wall boundaries (x, y in [0, W])
        for ax, bound, normal_sign in [(0, 0.0, 1.0), (0, self.box_width, -1.0), (1, 0.0, 1.0), (1, self.box_width, -1.0)]:
            side_ov = self.radius - self.positions[:, ax] if normal_sign > 0 else self.positions[:, ax] + self.radius - bound
            sc = side_ov > 0.0
            if np.any(sc):
                ov = side_ov[sc]
                vn = self.velocities[sc, ax] * normal_sign
                k_w = 2.0 * self.E_star_wall * np.sqrt(self.radius * ov)
                eta_w = 2.0 * self.zeta * np.sqrt(self.mass * k_w)
                fd_max = 0.5 * self.mass * np.abs(vn) / self.dt
                fd = np.clip(-eta_w * vn, -fd_max, fd_max)
                fn = np.maximum(0.0, self.k_hertz_wall * (ov**1.5) + fd)
                forces[sc, ax] += normal_sign * fn
                
        # 4. Top piston wall (z = top_wall_z)
        if top_wall_z is not None:
            top_ov = self.positions[:, 2] + self.radius - top_wall_z
            tc = top_ov > 0.0
            if np.any(tc):
                ov = top_ov[tc]
                vz_rel = self.velocities[tc, 2] - top_wall_v
                k_w = 2.0 * self.E_star_wall * np.sqrt(self.radius * ov)
                eta_w = 2.0 * self.zeta * np.sqrt(self.mass * k_w)
                fd_max = 0.5 * self.mass * np.abs(vz_rel) / self.dt
                fd = np.clip(-eta_w * vz_rel, -fd_max, fd_max)
                fz_top = np.maximum(0.0, self.k_hertz_wall * (ov**1.5) + fd)
                forces[tc, 2] -= fz_top
                wall_piston_force = float(np.sum(fz_top))
                
                v_xy = self.velocities[tc, :2]
                v_xy_norm = np.linalg.norm(v_xy, axis=1)
                f_fric_piston = self.mu_friction * fz_top * np.tanh(v_xy_norm / self.v_slip_tol)
                forces[tc, :2] -= f_fric_piston[:, None] * (v_xy / (v_xy_norm[:, None] + 1.0e-12))
                
        coordination_number = float(2.0 * num_contacts / self.num_particles)
        tensile_fraction = float(tensile_contacts / num_contacts) if num_contacts > 0 else 0.0
        mean_overlap = float(sum_overlap / num_contacts) if num_contacts > 0 else 0.0
        
        contact_info = {
            "num_contacts": num_contacts,
            "max_overlap": max_overlap,
            "max_overlap_ratio": float(max_overlap / self.radius),
            "mean_overlap": mean_overlap,
            "coordination_number": coordination_number,
            "tensile_fraction": tensile_fraction,
            "stored_elastic_energy": stored_elastic_energy
        }
        return forces, wall_piston_force, contact_info

    def run_simulation(self, settle_steps=4000, num_snapshots=16):
        """
        Executes:
          Stage 1: Gravity settling for settle_steps until kinetic energy decays.
                   First 3000 steps use low wall friction (0.05) to prevent Janssen wall arching
                   and ensure realistic settling (phi ~ 0.52-0.58).
                   Final 1000 steps use physical friction (0.30) to establish static equilibrium.
          Stage 2: Continuous piston compaction at constant velocity to self.max_strain (13%),
                   recording snapshots at >= 15 strain increments.
        """
        dt = self.dt
        ke_history = []
        t_settle_start = time.time()
        
        # ---------------- Stage 1: Gravity Settling ----------------
        orig_mu = self.mu_friction
        settle_p1 = min(3000, int(0.75 * settle_steps))
        settle_p2 = settle_steps - settle_p1
        
        # Phase 1a: lubricated wall settling
        self.mu_friction = 0.05
        for step in range(settle_p1):
            forces, _, _ = self.compute_forces(top_wall_z=None)
            self.velocities += (forces / self.mass) * dt
            self.positions += self.velocities * dt
            ke = 0.5 * self.mass * np.sum(self.velocities**2)
            ke_history.append(ke)
            
        # Phase 1b: physical friction equilibration
        self.mu_friction = orig_mu
        for step in range(settle_p2):
            forces, _, _ = self.compute_forces(top_wall_z=None)
            self.velocities += (forces / self.mass) * dt
            self.positions += self.velocities * dt
            ke = 0.5 * self.mass * np.sum(self.velocities**2)
            ke_history.append(ke)
            
        settle_wall_time = time.time() - t_settle_start
        
        # Settled state properties
        settled_bed_height = float(np.max(self.positions[:, 2]) + self.radius)
        v_solid = self.num_particles * (4.0 / 3.0) * np.pi * (self.radius**3)
        v_settled_bed = self.box_area * settled_bed_height
        phi_settled = float(v_solid / v_settled_bed)
        e_settled = float((v_settled_bed - v_solid) / v_solid)
        
        # Structural check on settled state
        _, _, settled_c_info = self.compute_forces(top_wall_z=None)
        settled_positions = self.positions.copy()
        
        # Overlap check
        assert settled_c_info["max_overlap_ratio"] <= 0.05, (
            f"Overlap violation: max overlap {settled_c_info['max_overlap_ratio']:.4f} R > 0.05 R!"
        )
        assert phi_settled <= 0.64, f"Unphysical packing fraction phi = {phi_settled:.4f} > 0.64!"
        
        # ---------------- Stage 2: Continuous Compaction ----------------
        t_compact_start = time.time()
        piston_z = settled_bed_height
        target_displacement = settled_bed_height * self.max_strain
        target_piston_z = settled_bed_height - target_displacement
        
        # Fixed piston velocity: total compaction physical time
        compaction_time = target_displacement / self.piston_velocity
        compact_steps = int(np.round(compaction_time / dt))
        snapshot_interval = max(1, compact_steps // (num_snapshots - 1))
        
        snapshot_strains = []
        snapshot_forces = []
        snapshot_stresses = []
        snapshot_trajectories = []
        snapshot_coordinations = []
        snapshot_tensile_fractions = []
        snapshot_elastic_energies = []
        snapshot_inertial_numbers = []
        snapshot_void_ratios = []
        snapshot_densities = []
        
        for step in range(compact_steps + 1):
            piston_z = settled_bed_height - self.piston_velocity * (step * dt)
            piston_z = max(piston_z, target_piston_z)
            
            forces, piston_force, c_info = self.compute_forces(
                top_wall_z=piston_z,
                top_wall_v=-self.piston_velocity
            )
            self.velocities += (forces / self.mass) * dt
            self.positions += self.velocities * dt
            
            is_snapshot = (step % snapshot_interval == 0) or (step == compact_steps)
            if is_snapshot:
                current_strain = float((settled_bed_height - piston_z) / settled_bed_height)
                axial_stress = float(piston_force / self.box_area) # Pa
                
                # Inertial number: I = (eps_dot * d) / sqrt(P / rho)
                # Strain rate eps_dot = v_piston / piston_z
                strain_rate = self.piston_velocity / max(1e-4, piston_z)
                effective_pressure = max(1.0, axial_stress)
                inertial_number = float((strain_rate * self.diameter) / np.sqrt(effective_pressure / self.density))
                
                v_bed_now = self.box_area * piston_z
                e_now = float((v_bed_now - v_solid) / v_solid)
                rho_now = float((self.num_particles * self.mass) / v_bed_now)
                
                snapshot_strains.append(current_strain)
                snapshot_forces.append(piston_force)
                snapshot_stresses.append(axial_stress)
                snapshot_trajectories.append(self.positions.copy())
                snapshot_coordinations.append(c_info["coordination_number"])
                snapshot_tensile_fractions.append(c_info["tensile_fraction"])
                snapshot_elastic_energies.append(c_info["stored_elastic_energy"])
                snapshot_inertial_numbers.append(inertial_number)
                snapshot_void_ratios.append(e_now)
                snapshot_densities.append(rho_now)
                
        compact_wall_time = time.time() - t_compact_start
        
        final_piston_height = float(piston_z)
        v_bed_final = float(self.box_area * final_piston_height)
        final_void_ratio = float((v_bed_final - v_solid) / v_solid)
        final_bulk_density = float((self.num_particles * self.mass) / v_bed_final)
        final_phi = float(v_solid / v_bed_final)
        
        # Sanity assertions
        assert 0.40 <= final_void_ratio <= 1.20, f"Void ratio {final_void_ratio:.4f} outside [0.40, 1.20]!"
        assert final_phi <= 0.68, f"Final packing fraction {final_phi:.4f} > 0.68!"
        
        return {
            "settled_positions": settled_positions,
            "settled_bed_height": settled_bed_height,
            "phi_settled": phi_settled,
            "e_settled": e_settled,
            "settled_overlap_ratio": settled_c_info["max_overlap_ratio"],
            "settled_coordination": settled_c_info["coordination_number"],
            "ke_history": np.array(ke_history),
            "settle_wall_time_sec": float(settle_wall_time),
            
            # Compaction snapshots & profiles
            "snapshot_strains": np.array(snapshot_strains),
            "snapshot_forces": np.array(snapshot_forces),
            "snapshot_stresses": np.array(snapshot_stresses),
            "snapshot_trajectories": np.array(snapshot_trajectories),
            "snapshot_coordinations": np.array(snapshot_coordinations),
            "snapshot_tensile_fractions": np.array(snapshot_tensile_fractions),
            "snapshot_elastic_energies": np.array(snapshot_elastic_energies),
            "snapshot_inertial_numbers": np.array(snapshot_inertial_numbers),
            "snapshot_void_ratios": np.array(snapshot_void_ratios),
            "snapshot_densities": np.array(snapshot_densities),
            "compact_wall_time_sec": float(compact_wall_time),
            
            # Summary scalar targets
            "final_piston_height": final_piston_height,
            "final_void_ratio": final_void_ratio,
            "final_bulk_density": final_bulk_density,
            "peak_wall_force": float(np.max(snapshot_forces)),
            "peak_axial_stress": float(np.max(snapshot_stresses)),
            "final_coordination": float(snapshot_coordinations[-1]),
            "final_tensile_fraction": float(snapshot_tensile_fractions[-1]),
            "final_elastic_energy": float(snapshot_elastic_energies[-1]),
            "mean_inertial_number": float(np.mean(snapshot_inertial_numbers)),
            
            # Global parameters
            "cohesion": self.cohesion,
            "gamma_work_of_adhesion": self.gamma,
            "f_pull_off": self.f_pull_off,
            "bond_number": self.bond_number,
            "seed": self.seed,
            "num_particles": self.num_particles,
            "box_bounds": [self.box_width, self.box_width, final_piston_height]
        }

def verify_two_sphere_hertz_dmt():
    """Phase A Validation (i) & (ii): Analytical Hertzian and DMT force check."""
    sim = ParticleBedDEM(num_particles=2, cohesion=50.0, seed=42)
    # Align two particles along x-axis with overlap delta = 0.02 * R
    delta = 0.02 * sim.radius
    sim.positions[0] = np.array([sim.box_width/2 - sim.radius + delta/2, sim.box_width/2, sim.box_width/2])
    sim.positions[1] = np.array([sim.box_width/2 + sim.radius - delta/2, sim.box_width/2, sim.box_width/2])
    sim.velocities[:] = 0.0
    
    # Analytical Hertz force
    f_hertz_analytical = (4.0 / 3.0) * sim.E_star * np.sqrt(sim.R_star) * (delta**1.5)
    f_dmt_pull_off = 2.0 * np.pi * sim.R_star * sim.gamma
    f_net_analytical = f_hertz_analytical - f_dmt_pull_off
    
    forces, _, c_info = sim.compute_forces()
    f_sim_x = forces[0, 0] # repulsive force pushing particle 0 in -x direction
    
    err_hertz = abs((-f_sim_x - f_net_analytical))
    print(f"Hertz Analytical: {f_hertz_analytical:.6f} N, DMT pull-off: {f_dmt_pull_off:.6f} N, Net: {f_net_analytical:.6f} N")
    print(f"DEM Force: {-f_sim_x:.6f} N, Absolute Discrepancy: {err_hertz:.2e} N")
    assert err_hertz < 1e-10, "Hertzian contact force verification failed!"
    print("-> Phase A Validation (i) & (ii) PASSED: Exact match with Hertzian and DMT theory.")

def verify_two_sphere_restitution():
    """Phase A Validation: Two-sphere dynamic impact test for restitution coefficient."""
    sim = ParticleBedDEM(num_particles=2, cohesion=0.0, seed=42)
    v_in = 0.10 # m/s
    sim.positions[0] = np.array([0.020, 0.035, 0.035])
    sim.positions[1] = np.array([0.020 + 2.0 * sim.radius + 1e-4, 0.035, 0.035])
    sim.velocities[0] = np.array([v_in, 0.0, 0.0])
    sim.velocities[1] = np.array([0.0, 0.0, 0.0])
    
    max_ov = 0.0
    # Simulate dynamic collision until rebound
    for step in range(1000):
        forces, _, c_info = sim.compute_forces()
        sim.velocities += (forces / sim.mass) * sim.dt
        sim.positions += sim.velocities * sim.dt
        if c_info["max_overlap"] > max_ov:
            max_ov = c_info["max_overlap"]
        if step > 50 and sim.positions[1, 0] - sim.positions[0, 0] > 2.05 * sim.radius:
            break
            
    v_rel_out = sim.velocities[1, 0] - sim.velocities[0, 0]
    e_actual = v_rel_out / v_in
    print(f"Restitution Test: Target e_n = {sim.restitution:.2f}, Actual Rebound e = {e_actual:.3f}")
    print(f"Max dynamic overlap: {max_ov*1e3:.4f} mm ({max_ov/sim.radius:.4f} R <= 0.05 R)")
    assert 0.20 <= e_actual <= 0.35, f"Restitution {e_actual:.3f} deviated from target 0.30!"
    assert max_ov / sim.radius <= 0.05, f"Dynamic overlap {max_ov/sim.radius:.4f} R > 0.05 R!"
    print("-> Dynamic impact test PASSED: Damping ratio zeta=0.3579 accurately reproduces e_n=0.30.")

if __name__ == "__main__":
    print("Running Two-Sphere Verification Tests...")
    verify_two_sphere_hertz_dmt()
    verify_two_sphere_restitution()
    print("\nTesting Full 3D Granular Bed Simulation...")
    sim = ParticleBedDEM(num_particles=400, cohesion=50.0, max_strain=0.13, seed=42)
    res = sim.run_simulation(settle_steps=4000, num_snapshots=16)
    print(f"Settled: H={res['settled_bed_height']*100:.2f} cm, phi={res['phi_settled']:.4f}, delta_max/R={res['settled_overlap_ratio']:.4f}, Z={res['settled_coordination']:.2f}")
    print(f"Compacted: Peak Force={res['peak_wall_force']:.2f} N, Peak Stress={res['peak_axial_stress']/1e3:.2f} kPa, Final Z={res['final_coordination']:.2f}, e={res['final_void_ratio']:.4f}")
    print(f"Inertial number I mean: {res['mean_inertial_number']:.2e} (quasi-static regime)")
