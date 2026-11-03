"""
Prototype implementation and test of ParticleBedDEM
"""

import numpy as np
from scipy.spatial import cKDTree
import time

class ParticleBedDEM:
    def __init__(self, num_particles=400, cohesion=50.0, displacement=0.10, seed=42):
        self.seed = seed
        np.random.seed(seed)
        
        # Physical parameters
        self.radius = 0.005 # 5mm radius
        self.density = 2500.0 # kg/m^3 (quartz sand)
        self.mass = (4.0 / 3.0) * np.pi * (self.radius**3) * self.density
        
        self.num_particles = num_particles
        self.cohesion = cohesion # in kPa (0, 50, 100, 150)
        self.displacement_ratio = displacement # Compaction strain relative to settled bed height
        
        # Material elastic properties
        self.youngs = 1e7 # Pa
        self.poisson = 0.3
        self.E_star = self.youngs / (2.0 * (1.0 - self.poisson**2)) # ~5.49e6 Pa
        self.R_star = self.radius / 2.0 # 0.0025 m
        self.k_hertz = (4.0 / 3.0) * self.E_star * np.sqrt(self.R_star) # ~3.66e5 N/m^1.5
        
        # Wall contact properties (rigid walls: E_wall -> inf => E*_wall = E / (1 - nu^2) = 2 * E*)
        self.E_star_wall = self.youngs / (1.0 - self.poisson**2)
        self.k_hertz_wall = (4.0 / 3.0) * self.E_star_wall * np.sqrt(self.radius)
        
        # DMT Adhesion: constant pull-off force = 2 * pi * R* * gamma
        # gamma is surface energy (J/m^2) proportional to cohesion (kPa)
        # Using gamma = cohesion * 0.05 J/m^2 gives realistic pull-off force (0 to 0.12 N, Bond number ~ 3-10)
        self.gamma = self.cohesion * 0.05 # J/m^2
        self.f_pull_off = 2.0 * np.pi * self.R_star * self.gamma # N
        
        # Damping & Friction
        self.restitution = 0.5
        self.zeta = -np.log(self.restitution) / np.sqrt(np.pi**2 + (np.log(self.restitution))**2)
        self.mu_friction = 0.35 # Coulomb friction coefficient
        self.v_slip_tol = 1e-4 # m/s regularization
        
        # Box dimensions
        self.box_width = 0.08 # 8 cm x 8 cm box
        self.box_area = self.box_width**2
        
        # Rayleigh critical timestep calculation
        G = self.youngs / (2.0 * (1.0 + self.poisson))
        self.t_rayleigh = (np.pi * self.radius / (0.1631 * self.poisson + 0.8766)) * np.sqrt(self.density / G)
        self.safety_factor = 0.20
        self.dt = self.safety_factor * self.t_rayleigh # ~8.65e-5 s (we round to 1e-4 s with safety factor ~0.23)
        self.dt = 1e-4
        
        # State arrays
        self.positions = np.zeros((num_particles, 3))
        self.velocities = np.zeros((num_particles, 3))
        self.forces = np.zeros((num_particles, 3))
        
        self._initialize_particle_bed()
        
    def _initialize_particle_bed(self):
        """
        Initializes particles in a dense, jittered grid spanning the box width
        so that particles fill the cross-section and settle with minimal void space.
        """
        nx = 7
        ny = 7
        nz = int(np.ceil(self.num_particles / (nx * ny)))
        
        xs = np.linspace(1.2 * self.radius, self.box_width - 1.2 * self.radius, nx)
        ys = np.linspace(1.2 * self.radius, self.box_width - 1.2 * self.radius, ny)
        
        grid = []
        z = 1.2 * self.radius
        while len(grid) < self.num_particles:
            for x in xs:
                for y in ys:
                    if len(grid) < self.num_particles:
                        jitter = (np.random.rand(3) - 0.5) * self.radius * 0.2
                        grid.append(np.array([x, y, z]) + jitter)
            z += 2.05 * self.radius
            
        self.positions = np.array(grid[:self.num_particles])
        self.velocities = np.zeros_like(self.positions)
        self.box_height_initial = z + 2.0 * self.radius
        
    def compute_forces(self, top_wall_z, top_wall_v=0.0):
        """
        Computes inter-particle forces (Hertzian + DMT adhesion + damping + Coulomb friction)
        and wall boundary forces using cKDTree.
        """
        gravity = np.array([0.0, 0.0, -9.81])
        forces = np.ones((self.num_particles, 3)) * (gravity * self.mass)
        wall_piston_force = 0.0
        
        # 1. Inter-particle contact forces via cKDTree
        tree = cKDTree(self.positions)
        pairs = tree.query_pairs(r=2.0 * self.radius, output_type='ndarray')
        
        if len(pairs) > 0:
            i = pairs[:, 0]
            j = pairs[:, 1]
            
            # Vector pointing from j to i
            diff = self.positions[i] - self.positions[j]
            dist = np.linalg.norm(diff, axis=1)
            valid = dist > 1e-10
            
            i = i[valid]
            j = j[valid]
            diff = diff[valid]
            dist = dist[valid]
            
            normal = diff / dist[:, None] # points j -> i
            overlap = 2.0 * self.radius - dist
            
            # Relative velocity of i relative to j
            v_rel = self.velocities[i] - self.velocities[j]
            v_n = np.sum(v_rel * normal, axis=1) # normal component (positive when separating)
            
            # Elastic Hertz force + DMT pull-off
            f_hertz = self.k_hertz * (overlap**1.5)
            f_elastic = f_hertz - self.f_pull_off # genuine tensile branch
            
            # Contact normal damping
            m_star = self.mass / 2.0
            k_n = 2.0 * self.E_star * np.sqrt(self.R_star * overlap)
            eta_n = 2.0 * self.zeta * np.sqrt(m_star * k_n)
            f_damp = -eta_n * v_n
            
            # Total normal force (limited to -f_pull_off in tension)
            f_normal_mag = f_elastic + f_damp
            if self.f_pull_off > 0:
                f_normal_mag = np.maximum(-self.f_pull_off, f_normal_mag)
            
            # Coulomb tangential friction
            v_t_vec = v_rel - v_n[:, None] * normal
            v_t_norm = np.linalg.norm(v_t_vec, axis=1)
            t_vec = v_t_vec / (v_t_norm[:, None] + 1e-12)
            
            f_normal_comp = np.maximum(0.0, f_normal_mag)
            friction_mag = self.mu_friction * f_normal_comp * np.tanh(v_t_norm / self.v_slip_tol)
            f_tangent_vec = -friction_mag[:, None] * t_vec
            
            # Total contact force vector on particle i
            f_contact = f_normal_mag[:, None] * normal + f_tangent_vec
            
            # Accumulate onto i and j
            np.add.at(forces, i, f_contact)
            np.add.at(forces, j, -f_contact)
            
        # 2. Wall contact forces
        # Bottom wall (z = 0)
        bot_overlap = self.radius - self.positions[:, 2]
        bot_contact = bot_overlap > 0
        if np.any(bot_contact):
            ov = bot_overlap[bot_contact]
            vz = self.velocities[bot_contact, 2]
            k_w = 2.0 * self.E_star_wall * np.sqrt(self.radius * ov)
            eta_w = 2.0 * self.zeta * np.sqrt(self.mass * k_w)
            f_z = self.k_hertz_wall * (ov**1.5) - eta_w * vz
            f_z = np.maximum(0.0, f_z)
            forces[bot_contact, 2] += f_z
            
            # Bottom wall friction
            v_xy = self.velocities[bot_contact, :2]
            v_xy_norm = np.linalg.norm(v_xy, axis=1)
            f_fric_mag = self.mu_friction * f_z * np.tanh(v_xy_norm / self.v_slip_tol)
            forces[bot_contact, :2] -= f_fric_mag[:, None] * (v_xy / (v_xy_norm[:, None] + 1e-12))
            
        # Top wall / Piston (z = top_wall_z)
        top_overlap = self.positions[:, 2] + self.radius - top_wall_z
        top_contact = top_overlap > 0
        if np.any(top_contact):
            ov = top_overlap[top_contact]
            vz_rel = self.velocities[top_contact, 2] - top_wall_v
            k_w = 2.0 * self.E_star_wall * np.sqrt(self.radius * ov)
            eta_w = 2.0 * self.zeta * np.sqrt(self.mass * k_w)
            f_z = self.k_hertz_wall * (ov**1.5) + eta_w * (-vz_rel)
            f_z = np.maximum(0.0, f_z)
            forces[top_contact, 2] -= f_z
            wall_piston_force = float(np.sum(f_z))
            
            # Piston wall friction
            v_xy = self.velocities[top_contact, :2]
            v_xy_norm = np.linalg.norm(v_xy, axis=1)
            f_fric_mag = self.mu_friction * f_z * np.tanh(v_xy_norm / self.v_slip_tol)
            forces[top_contact, :2] -= f_fric_mag[:, None] * (v_xy / (v_xy_norm[:, None] + 1e-12))
            
        # Side walls: x in [0, box_width], y in [0, box_width]
        for ax, bound, normal_sign in [(0, 0.0, 1.0), (0, self.box_width, -1.0), (1, 0.0, 1.0), (1, self.box_width, -1.0)]:
            if normal_sign > 0:
                side_ov = self.radius - self.positions[:, ax]
            else:
                side_ov = self.positions[:, ax] + self.radius - bound
            side_contact = side_ov > 0
            if np.any(side_contact):
                ov = side_ov[side_contact]
                v_norm = self.velocities[side_contact, ax] * normal_sign
                k_w = 2.0 * self.E_star_wall * np.sqrt(self.radius * ov)
                eta_w = 2.0 * self.zeta * np.sqrt(self.mass * k_w)
                f_n = self.k_hertz_wall * (ov**1.5) - eta_w * v_norm
                f_n = np.maximum(0.0, f_n)
                forces[side_contact, ax] += normal_sign * f_n
                
        return forces, wall_piston_force

    def run_simulation(self, settle_steps=400, compact_steps=400, dump_interval=20):
        """
        Runs two-phase simulation:
        1. Gravity settling until kinetic energy decays
        2. Compaction by top piston starting from settled bed height
        """
        dt = self.dt
        
        # ---------------- Phase 1: Gravity Settling ----------------
        top_wall_z = self.box_height_initial
        ke_history = []
        
        for step in range(settle_steps):
            forces, _ = self.compute_forces(top_wall_z=top_wall_z, top_wall_v=0.0)
            
            # Symplectic Euler integration:
            # v(t+dt) = v(t) + a(t) * dt
            # x(t+dt) = x(t) + v(t+dt) * dt
            accel = forces / self.mass
            self.velocities += accel * dt
            self.positions += self.velocities * dt
            
            # Keep bounded
            self.positions[:, 0] = np.clip(self.positions[:, 0], self.radius, self.box_width - self.radius)
            self.positions[:, 1] = np.clip(self.positions[:, 1], self.radius, self.box_width - self.radius)
            self.positions[:, 2] = np.clip(self.positions[:, 2], self.radius, top_wall_z - self.radius)
            
            ke = 0.5 * self.mass * np.sum(self.velocities**2)
            ke_history.append(ke)
            
        # Determine settled bed height
        settled_bed_height = float(np.max(self.positions[:, 2]) + self.radius)
        v_solid = self.num_particles * (4.0 / 3.0) * np.pi * (self.radius**3)
        v_settled_bed = self.box_area * settled_bed_height
        phi_settled = v_solid / v_settled_bed
        e_settled = (v_settled_bed - v_solid) / v_solid
        
        # ---------------- Phase 2: Compaction by Piston ----------------
        # Piston starts at settled bed height
        piston_z = settled_bed_height
        target_piston_z = settled_bed_height * (1.0 - self.displacement_ratio)
        piston_speed = (settled_bed_height - target_piston_z) / (compact_steps * dt)
        
        trajectory = []
        velocities_hist = []
        wall_forces = []
        piston_heights = []
        
        for step in range(compact_steps):
            piston_z -= piston_speed * dt
            piston_z = max(piston_z, target_piston_z)
            
            forces, piston_force = self.compute_forces(top_wall_z=piston_z, top_wall_v=-piston_speed)
            
            # Symplectic Euler
            accel = forces / self.mass
            self.velocities += accel * dt
            self.positions += self.velocities * dt
            
            self.positions[:, 0] = np.clip(self.positions[:, 0], self.radius, self.box_width - self.radius)
            self.positions[:, 1] = np.clip(self.positions[:, 1], self.radius, self.box_width - self.radius)
            self.positions[:, 2] = np.clip(self.positions[:, 2], self.radius, piston_z - self.radius)
            
            if step % dump_interval == 0 or step == compact_steps - 1:
                trajectory.append(self.positions.copy())
                velocities_hist.append(self.velocities.copy())
                wall_forces.append(piston_force)
                piston_heights.append(piston_z)
                
        # Final macroscopic metrics
        final_piston_height = piston_z
        v_bed_final = self.box_area * final_piston_height
        final_void_ratio = float((v_bed_final - v_solid) / v_solid)
        final_bulk_density = float((self.num_particles * self.mass) / v_bed_final)
        
        # Sanity check: void ratio must be in [0.4, 2.0]
        assert 0.40 <= final_void_ratio <= 2.0, (
            f"SANITY CHECK FAILED: Final void ratio e = {final_void_ratio:.4f} "
            f"is outside physical bounds [0.4, 2.0]!"
        )
        
        return {
            "trajectory": np.array(trajectory),
            "velocities": np.array(velocities_hist),
            "wall_forces": np.array(wall_forces),
            "piston_heights": np.array(piston_heights),
            "ke_history": np.array(ke_history),
            "settled_bed_height": settled_bed_height,
            "phi_settled": phi_settled,
            "e_settled": e_settled,
            "final_piston_height": final_piston_height,
            "final_void_ratio": final_void_ratio,
            "final_bulk_density": final_bulk_density,
            "cohesion": self.cohesion,
            "displacement": self.displacement_ratio,
            "seed": self.seed,
            "num_particles": self.num_particles,
            "box_bounds": [self.box_width, self.box_width, final_piston_height]
        }

if __name__ == "__main__":
    print("Testing single simulation run...")
    t0 = time.time()
    sim = ParticleBedDEM(num_particles=400, cohesion=50.0, displacement=0.10, seed=42)
    res = sim.run_simulation(settle_steps=300, compact_steps=300, dump_interval=20)
    t1 = time.time()
    
    print(f"Elapsed time: {t1 - t0:.2f} s")
    print(f"Settled bed height: {res['settled_bed_height']*100:.2f} cm")
    print(f"Settled packing fraction phi: {res['phi_settled']:.4f}, void ratio: {res['e_settled']:.4f}")
    print(f"Final piston height: {res['final_piston_height']*100:.2f} cm")
    print(f"Final void ratio: {res['final_void_ratio']:.4f}")
    print(f"Final bulk density: {res['final_bulk_density']:.2f} kg/m^3")
    print(f"Piston wall forces (last 5 dumps): {res['wall_forces'][-5:]}")
    print(f"KE initial: {res['ke_history'][0]:.3e} J -> KE settled: {res['ke_history'][-1]:.3e} J (decay: {res['ke_history'][-1]/res['ke_history'][0]:.2e})")

