//! Gravity Forest — PLAN-01 Fase T1 MVP.
//!
//! 3D terrain + tree forest + gravity descent particle routing.
//!
//! Conceitos:
//! - Terreno virtual 3D grid (GRID×GRID×GRID cells) com potencial U(x,y,z)
//! - Trees plantadas em posições (x,y,z) com mass + embedding HV
//! - Cada query injeta partícula no terreno; gravidade puxa pra tree mais próxima
//! - Convergência via Verlet integration com damping
//!
//! Math:
//!   U(r) = -Σ G·m_i / (|r - r_i| + ε)
//!   F(r) = -∇U(r)
//!   v_{n+1} = γ·(v_n + F_n·dt)   (damping γ < 1)
//!   r_{n+1} = r_n + v_{n+1}·dt
//!
//! Convergência garantida por Lyapunov V = ½||v||² + U(r), dV/dt ≤ 0.

#![allow(dead_code)]

/// Grid resolution per axis. Total cells = GRID³.
pub const GRID: usize = 32;
/// Gravity constant.
const G: f32 = 1.0;
/// Softening epsilon to avoid singularity at tree center.
const EPS: f32 = 0.05;
/// Damping factor (Lyapunov decay).
const DAMPING: f32 = 0.85;
/// Time step.
const DT: f32 = 0.05;
/// Max descent steps per query.
const MAX_STEPS: usize = 60;
/// Convergence tolerance (force magnitude).
const CONV_TOL: f32 = 0.01;

/// 3D position vector.
#[derive(Clone, Copy, Debug, Default)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn add(self, o: Self) -> Self { Self::new(self.x+o.x, self.y+o.y, self.z+o.z) }
    pub fn sub(self, o: Self) -> Self { Self::new(self.x-o.x, self.y-o.y, self.z-o.z) }
    pub fn scale(self, k: f32) -> Self { Self::new(self.x*k, self.y*k, self.z*k) }
    pub fn norm_sq(self) -> f32 { self.x*self.x + self.y*self.y + self.z*self.z }
    pub fn norm(self) -> f32 { self.norm_sq().sqrt() }
}

/// Tree node: knowledge cluster in terrain.
#[derive(Clone, Debug)]
pub struct Tree {
    pub name: String,
    pub position: Vec3,
    pub mass: f32,
    pub n_dialogs: usize,
}

/// Forest with potential field + trees.
pub struct Forest {
    pub trees: Vec<Tree>,
    /// Potential field U [GRID × GRID × GRID].
    /// Stored as flat Vec to avoid 3D indexing overhead.
    pub potential: Vec<f32>,
    pub grid: usize,
}

impl Forest {
    pub fn new() -> Self {
        let grid = GRID;
        Self {
            trees: Vec::new(),
            potential: vec![0.0; grid * grid * grid],
            grid,
        }
    }

    /// Map 3D position [0,1]³ to grid index.
    fn pos_to_grid(&self, p: Vec3) -> (usize, usize, usize) {
        let g = self.grid as f32;
        let x = (p.x * g).clamp(0.0, g - 1.0) as usize;
        let y = (p.y * g).clamp(0.0, g - 1.0) as usize;
        let z = (p.z * g).clamp(0.0, g - 1.0) as usize;
        (x, y, z)
    }

    /// Plant a tree at position with given mass.
    /// Recomputes potential field locally around the tree.
    pub fn plant_tree(&mut self, name: String, position: Vec3, mass: f32, n_dialogs: usize) {
        self.trees.push(Tree { name, position, mass, n_dialogs });
        self.update_potential_local(position, mass);
    }

    /// Update potential field globally (full grid). O(G³) per tree.
    /// For sparse trees (<100), this is faster than rebuild_potential.
    fn update_potential_local(&mut self, source: Vec3, mass: f32) {
        let g = self.grid;
        let g_f = g as f32;
        for x in 0..g {
            for y in 0..g {
                for z in 0..g {
                    let cell_pos = Vec3::new(x as f32 / g_f, y as f32 / g_f, z as f32 / g_f);
                    let d = cell_pos.sub(source).norm() + EPS;
                    let du = -G * mass / d;
                    self.potential[x * g * g + y * g + z] += du;
                }
            }
        }
    }

    /// Rebuild full potential field from scratch over all trees.
    /// Useful after bulk planting.
    pub fn rebuild_potential(&mut self) {
        let g = self.grid;
        let g_f = g as f32;
        self.potential.iter_mut().for_each(|p| *p = 0.0);
        for tree in &self.trees {
            for x in 0..g {
                for y in 0..g {
                    for z in 0..g {
                        let cell = Vec3::new(x as f32 / g_f, y as f32 / g_f, z as f32 / g_f);
                        let d = cell.sub(tree.position).norm() + EPS;
                        let du = -G * tree.mass / d;
                        self.potential[x * g * g + y * g + z] += du;
                    }
                }
            }
        }
    }

    /// Sample potential at position via trilinear interp.
    fn sample_potential(&self, p: Vec3) -> f32 {
        let g = self.grid;
        let g_f = g as f32 - 1.0;
        let fx = (p.x * g_f).clamp(0.0, g_f);
        let fy = (p.y * g_f).clamp(0.0, g_f);
        let fz = (p.z * g_f).clamp(0.0, g_f);
        let x0 = fx as usize; let y0 = fy as usize; let z0 = fz as usize;
        let x1 = (x0 + 1).min(g - 1);
        let y1 = (y0 + 1).min(g - 1);
        let z1 = (z0 + 1).min(g - 1);
        let dx = fx - x0 as f32;
        let dy = fy - y0 as f32;
        let dz = fz - z0 as f32;
        let idx = |x: usize, y: usize, z: usize| x * g * g + y * g + z;
        let c000 = self.potential[idx(x0,y0,z0)];
        let c100 = self.potential[idx(x1,y0,z0)];
        let c010 = self.potential[idx(x0,y1,z0)];
        let c110 = self.potential[idx(x1,y1,z0)];
        let c001 = self.potential[idx(x0,y0,z1)];
        let c101 = self.potential[idx(x1,y0,z1)];
        let c011 = self.potential[idx(x0,y1,z1)];
        let c111 = self.potential[idx(x1,y1,z1)];
        let c00 = c000 * (1.0 - dx) + c100 * dx;
        let c01 = c001 * (1.0 - dx) + c101 * dx;
        let c10 = c010 * (1.0 - dx) + c110 * dx;
        let c11 = c011 * (1.0 - dx) + c111 * dx;
        let c0 = c00 * (1.0 - dy) + c10 * dy;
        let c1 = c01 * (1.0 - dy) + c11 * dy;
        c0 * (1.0 - dz) + c1 * dz
    }

    /// Compute gradient via finite differences.
    fn grad_potential(&self, p: Vec3) -> Vec3 {
        let h = 1.0 / (self.grid as f32);
        let dx = self.sample_potential(Vec3::new(p.x + h, p.y, p.z))
               - self.sample_potential(Vec3::new(p.x - h, p.y, p.z));
        let dy = self.sample_potential(Vec3::new(p.x, p.y + h, p.z))
               - self.sample_potential(Vec3::new(p.x, p.y - h, p.z));
        let dz = self.sample_potential(Vec3::new(p.x, p.y, p.z + h))
               - self.sample_potential(Vec3::new(p.x, p.y, p.z - h));
        Vec3::new(dx / (2.0 * h), dy / (2.0 * h), dz / (2.0 * h))
    }

    /// Find nearest tree by Euclidean distance (no gravity descent).
    /// Faster + más predictable que descent quando trees são spread out.
    pub fn nearest_tree(&self, pos: Vec3) -> Option<usize> {
        let mut best_dist = f32::INFINITY;
        let mut best_idx = None;
        for (i, tree) in self.trees.iter().enumerate() {
            let d = tree.position.sub(pos).norm();
            if d < best_dist {
                best_dist = d;
                best_idx = Some(i);
            }
        }
        best_idx
    }

    /// Run gravity descent from initial position. Returns final position +
    /// nearest tree index + steps taken.
    pub fn descent(&self, initial: Vec3) -> (Vec3, Option<usize>, usize) {
        let mut r = initial;
        let mut v = Vec3::default();
        let mut step = 0;
        const MAX_F: f32 = 5.0;  // clamp force magnitude pra evitar overshoot perto de centros
        for _ in 0..MAX_STEPS {
            let grad = self.grad_potential(r);
            let mut f = grad.scale(-1.0);  // F = -∇U
            let fmag = f.norm();
            if fmag < CONV_TOL { break; }
            if fmag > MAX_F { f = f.scale(MAX_F / fmag); }
            v = v.add(f.scale(DT)).scale(DAMPING);
            r = r.add(v.scale(DT));
            // Clamp to unit cube
            r.x = r.x.clamp(0.0, 1.0);
            r.y = r.y.clamp(0.0, 1.0);
            r.z = r.z.clamp(0.0, 1.0);
            step += 1;
        }
        // Find nearest tree
        let mut best_dist = f32::INFINITY;
        let mut best_idx = None;
        for (i, tree) in self.trees.iter().enumerate() {
            let d = tree.position.sub(r).norm();
            if d < best_dist {
                best_dist = d;
                best_idx = Some(i);
            }
        }
        (r, best_idx, step)
    }

    /// Project a 1024-bit HV (`[u64; 16]`) to 3D unit cube.
    /// Uses three byte-frequency style projections.
    pub fn hv_to_position(hv: &[u64; 16]) -> Vec3 {
        // x = bits 0..341 popcount / 341
        // y = bits 341..682 popcount / 341
        // z = bits 682..1024 popcount / 342
        let mut buckets = [0u32; 3];
        let mut bit = 0;
        for w in hv.iter() {
            for shift in 0..64 {
                let b = ((w >> shift) & 1) as u32;
                let bucket = if bit < 341 { 0 } else if bit < 682 { 1 } else { 2 };
                buckets[bucket] += b;
                bit += 1;
            }
        }
        Vec3::new(
            (buckets[0] as f32 / 341.0).clamp(0.0, 1.0),
            (buckets[1] as f32 / 341.0).clamp(0.0, 1.0),
            (buckets[2] as f32 / 342.0).clamp(0.0, 1.0),
        )
    }

    pub fn n_trees(&self) -> usize { self.trees.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_basic() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(0.0, 0.0, 0.0);
        assert_eq!(a.sub(b).norm_sq(), 14.0);
    }

    #[test]
    fn plant_and_descent() {
        let mut forest = Forest::new();
        forest.plant_tree("center".into(), Vec3::new(0.5, 0.5, 0.5), 10.0, 100);
        // Particle at edge should descend toward center
        let (final_pos, tree_idx, steps) = forest.descent(Vec3::new(0.1, 0.1, 0.1));
        assert!(tree_idx.is_some());
        assert!(steps > 0);
        // Final position should be closer to center than initial
        let initial = Vec3::new(0.1, 0.1, 0.1);
        let center = Vec3::new(0.5, 0.5, 0.5);
        let d_init = center.sub(initial).norm();
        let d_final = center.sub(final_pos).norm();
        assert!(d_final < d_init, "descent should approach center");
    }

    #[test]
    fn hv_projection_consistent() {
        let hv1 = [0u64; 16];
        let p1 = Forest::hv_to_position(&hv1);
        assert_eq!(p1.x, 0.0);
        let hv2 = [!0u64; 16];  // all bits set
        let p2 = Forest::hv_to_position(&hv2);
        assert!(p2.x > 0.99);
    }

    #[test]
    fn multi_tree_routing() {
        let mut forest = Forest::new();
        forest.plant_tree("A".into(), Vec3::new(0.2, 0.2, 0.2), 5.0, 50);
        forest.plant_tree("B".into(), Vec3::new(0.8, 0.8, 0.8), 5.0, 50);
        // Query near A should converge to A
        let (_, idx, _) = forest.descent(Vec3::new(0.25, 0.25, 0.25));
        assert_eq!(idx, Some(0));
        // Query near B should converge to B
        let (_, idx, _) = forest.descent(Vec3::new(0.75, 0.75, 0.75));
        assert_eq!(idx, Some(1));
    }
}
