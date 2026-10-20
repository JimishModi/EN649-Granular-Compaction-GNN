"""
Granular DEM Compaction Simulator in 3D Python
Simulates 3D sphere packing under gravity settling and piston compression
with Hertzian contact dynamics and DMT/JKR surface cohesion.
"""

import numpy as np
import os
import json
import time

class ParticleBedDEM:
    def __init__(self, num_particles=500, cohesion=50.0, displacement=0.25, seed=42):
        np.random.seed(seed)
        self.num_particles = num_particles
        self.cohesion = cohesion  # Surface energy / cohesion factor (kPa equivalent)
        self.displacement_ratio = displacement # Fractional height compression
        self.seed = seed
        
        # Physical parameters
        self.radius = 0.005 # 5mm radius particles
        self.density = 2500.0 # kg/m^3 (quartz/sand)
        self.mass = (4/3) * np.pi * (self.radius**3) * self.density
        self.youngs = 1e7 # Pa (stiff sphere contact stiffness)
        self.poisson = 0.3
        self.gamma_c = cohesion * 1e3 # Cohesion scale
        
        # Box dimensions
        self.box_width = 0.1 # 10 cm x 10 cm
        self.box_height_initial = 0.25 # 25 cm height
        
        # Initial positions - generate a loose random particle bed
        self.positions = np.zeros((num_particles, 3))
        self.velocities = np.zeros((num_particles, 3))
        self.forces = np.zeros((num_particles, 3))
        
        self._initialize_particles()
        self.particle_types = np.ones(num_particles, dtype=int)
        
    def _initialize_particles(self):
        # Grid-like random jittered initial placement to prevent severe overlap
        nx = int(np.ceil((self.num_particles)**(1/3)))
        ny = nx
        nz = int(np.ceil(self.num_particles / (nx * ny)))
        
        xs = np.linspace(2*self.radius, self.box_width - 2*self.radius, nx)
        ys = np.linspace(2*self.radius, self.box_width - 2*self.radius, ny)
        zs = np.linspace(2*self.radius, self.box_height_initial * 0.7, nz)
        
        idx = 0
        for z in zs:
            for y in ys:
                for x in xs:
                    if idx < self.num_particles:
                        jitter = (np.random.rand(3) - 0.5) * self.radius * 0.2
                        self.positions[idx] = np.array([x, y, z]) + jitter
                        idx += 1
                        
    def run_simulation(self, total_steps=200, dt=1e-4, dump_interval=5):
        """
        Runs settlement under gravity and top wall compression.
        Returns:
            trajectory: (timesteps, num_particles, 3)
            velocities_hist: (timesteps, num_particles, 3)
            wall_forces: (timesteps,)
            metrics: dict with final_density, final_void_ratio, etc.
        """
        gravity = np.array([0.0, 0.0, -9.81])
        damp = 0.98 # Velocity damping for numerical stability
        
        top_wall_z = self.box_height_initial
        target_top_wall_z = self.box_height_initial * (1.0 - self.displacement_ratio)
        piston_speed = (self.box_height_initial - target_top_wall_z) / (total_steps * dt)
        
        trajectory = []
        velocities_hist = []
        wall_forces = []
        
        for step in range(total_steps):
            # Move piston downward
            if top_wall_z > target_top_wall_z:
                top_wall_z -= piston_speed * dt
            top_wall_z = max(top_wall_z, target_top_wall_z)
            
            # Reset forces & add gravity
            self.forces = np.ones((self.num_particles, 3)) * gravity * self.mass
            
            # 1. Contact forces between particles (Hertzian + DMT cohesion)
            # Distance matrix
            diff = self.positions[:, None, :] - self.positions[None, :, :] # (N, N, 3)
            dist = np.linalg.norm(diff, axis=-1) # (N, N)
            np.fill_diagonal(dist, 1e9) # ignore self-interaction
            
            overlap = 2 * self.radius - dist
            contact_mask = overlap > 0
            
            wall_force_step = 0.0
            
            if np.any(contact_mask):
                # Contact normal vectors
                normal = np.zeros_like(diff)
                valid_mask = dist > 1e-8
                normal[valid_mask] = diff[valid_mask] / dist[valid_mask][..., None]
                
                # Hertzian normal force magnitude: F_n = k * overlap^1.5 - F_cohesion
                k_contact = 1e5
                f_repulsion = k_contact * np.maximum(0, overlap)**1.5
                f_cohesion = self.gamma_c * (self.radius * np.maximum(0, overlap))**0.5
                f_net = np.maximum(0.0, f_repulsion - f_cohesion)
                
                force_vectors = normal * f_net[..., None]
                self.forces += np.sum(force_vectors, axis=1)
                
            # 2. Wall boundary constraints
            # Bottom wall (z = 0)
            bottom_contact = self.positions[:, 2] < self.radius
            self.forces[bottom_contact, 2] += 1e5 * (self.radius - self.positions[bottom_contact, 2])
            
            # Top wall (z = top_wall_z)
            top_contact = self.positions[:, 2] > (top_wall_z - self.radius)
            if np.any(top_contact):
                top_force = 1e5 * (self.positions[top_contact, 2] - (top_wall_z - self.radius))
                self.forces[top_contact, 2] -= top_force
                wall_force_step = np.sum(top_force)
                
            # Side walls (x, y bounds [0, box_width])
            x_min_c = self.positions[:, 0] < self.radius
            x_max_c = self.positions[:, 0] > (self.box_width - self.radius)
            y_min_c = self.positions[:, 1] < self.radius
            y_max_c = self.positions[:, 1] > (self.box_width - self.radius)
            
            self.forces[x_min_c, 0] += 1e5 * (self.radius - self.positions[x_min_c, 0])
            self.forces[x_max_c, 0] -= 1e5 * (self.positions[x_max_c, 0] - (self.box_width - self.radius))
            self.forces[y_min_c, 1] += 1e5 * (self.radius - self.positions[y_min_c, 1])
            self.forces[y_max_c, 1] -= 1e5 * (self.positions[y_max_c, 1] - (self.box_width - self.radius))
            
            # Update position and velocity (Verlet integration with damping)
            accel = self.forces / self.mass
            self.velocities = (self.velocities + accel * dt) * damp
            self.positions += self.velocities * dt
            
            # Keep within box
            self.positions[:, 0] = np.clip(self.positions[:, 0], self.radius, self.box_width - self.radius)
            self.positions[:, 1] = np.clip(self.positions[:, 1], self.radius, self.box_width - self.radius)
            self.positions[:, 2] = np.clip(self.positions[:, 2], self.radius, top_wall_z - self.radius)
            
            if step % dump_interval == 0:
                trajectory.append(self.positions.copy())
                velocities_hist.append(self.velocities.copy())
                wall_forces.append(wall_force_step)
                
        # Final state calculations
        final_bed_height = np.max(self.positions[:, 2]) + self.radius
        v_total = (self.box_width**2) * final_bed_height
        v_spheres = self.num_particles * (4/3) * np.pi * (self.radius**3)
        v_voids = max(1e-6, v_total - v_spheres)
        
        final_void_ratio = float(v_voids / v_spheres)
        final_bulk_density = float((self.num_particles * self.mass) / v_total)
        
        results = {
            "trajectory": np.array(trajectory), # (T, N, 3)
            "velocities": np.array(velocities_hist), # (T, N, 3)
            "wall_forces": np.array(wall_forces),
            "final_void_ratio": final_void_ratio,
            "final_bulk_density": final_bulk_density,
            "cohesion": self.cohesion,
            "displacement": self.displacement_ratio,
            "seed": self.seed,
            "num_particles": self.num_particles,
            "box_bounds": [self.box_width, self.box_width, final_bed_height]
        }
        return results

if __name__ == "__main__":
    sim = ParticleBedDEM(num_particles=400, cohesion=50.0, displacement=0.30, seed=42)
    res = sim.run_simulation(total_steps=100)
    print(f"Test run completed: {res['trajectory'].shape} frames, Final Void Ratio = {res['final_void_ratio']:.4f}, Bulk Density = {res['final_bulk_density']:.2f} kg/m^3")
