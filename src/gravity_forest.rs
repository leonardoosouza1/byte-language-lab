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

/// B-spline cubic branch: 4 control points + sub-topic embedding.
#[derive(Clone, Debug)]
pub struct Branch {
    pub label: String,
    /// Control points em torno da tree (relative offsets)
    pub control_points: [Vec3; 4],
    /// Sub-topic signature (subset of dialogs)
    pub sub_indices: Vec<usize>,
    /// Bitwise-majority HV centroid of this branch's dialogs (1024-bit).
    /// Lets retrieval pick the right branch before scanning sub_indices —
    /// a 2-level index (tree → branch → dialog) instead of a flat scan.
    pub centroid: [u64; 16],
}

impl Branch {
    /// Evaluate cubic B-spline at t∈[0,1].
    /// B(t) = (1-t)³/6·P0 + (3t³-6t²+4)/6·P1 + (-3t³+3t²+3t+1)/6·P2 + t³/6·P3
    pub fn eval(&self, t: f32) -> Vec3 {
        let t2 = t * t;
        let t3 = t2 * t;
        let omt = 1.0 - t;
        let omt3 = omt * omt * omt;
        let b0 = omt3 / 6.0;
        let b1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0;
        let b2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0;
        let b3 = t3 / 6.0;
        let p0 = self.control_points[0].scale(b0);
        let p1 = self.control_points[1].scale(b1);
        let p2 = self.control_points[2].scale(b2);
        let p3 = self.control_points[3].scale(b3);
        p0.add(p1).add(p2).add(p3)
    }
}

/// Tree node: knowledge cluster in terrain.
#[derive(Clone, Debug)]
pub struct Tree {
    pub name: String,
    pub position: Vec3,
    pub mass: f32,
    pub n_dialogs: usize,
    /// B-spline branches radiating from tree (Fase T2)
    pub branches: Vec<Branch>,
}

/// Mycelium edge: weighted connection between two trees.
/// Edges grown when topics co-occur in dialogs (Fase T3).
#[derive(Clone, Debug)]
pub struct MyceliumEdge {
    pub tree_a: usize,
    pub tree_b: usize,
    pub weight: f32,
}

/// Forest with potential field + trees + mycelium edges.
pub struct Forest {
    pub trees: Vec<Tree>,
    /// Potential field U [GRID × GRID × GRID].
    /// Stored as flat Vec to avoid 3D indexing overhead.
    pub potential: Vec<f32>,
    pub grid: usize,
    /// Mycelium graph: weighted edges between trees (Fase T3)
    pub mycelium: Vec<MyceliumEdge>,
}

impl Forest {
    pub fn new() -> Self {
        let grid = GRID;
        Self {
            trees: Vec::new(),
            potential: vec![0.0; grid * grid * grid],
            grid,
            mycelium: Vec::new(),
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
        self.trees.push(Tree {
            name, position, mass, n_dialogs,
            branches: Vec::new(),
        });
        self.update_potential_local(position, mass);
    }

    /// Grow a B-spline branch on tree `tree_idx`. Control points are randomly
    /// distributed around the tree position (deterministic via seed).
    /// `centroid` is the HV centroid of the branch's dialogs (computed by the
    /// caller, which owns the dialog HVs) — used for 2-level retrieval.
    pub fn grow_branch(&mut self, tree_idx: usize, label: String, sub_indices: Vec<usize>,
                       seed: u64, centroid: [u64; 16]) {
        if tree_idx >= self.trees.len() { return; }
        let pos = self.trees[tree_idx].position;
        let mut state = seed;
        let mut rand = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as u32) as f32 / u32::MAX as f32) * 0.2 - 0.1  // [-0.1, 0.1]
        };
        let mut cps = [Vec3::default(); 4];
        for i in 0..4 {
            let dx = rand();
            let dy = rand();
            let dz = rand();
            let factor = (i as f32 + 1.0) / 4.0;
            cps[i] = Vec3::new(
                (pos.x + dx * factor).clamp(0.0, 1.0),
                (pos.y + dy * factor).clamp(0.0, 1.0),
                (pos.z + dz * factor).clamp(0.0, 1.0),
            );
        }
        self.trees[tree_idx].branches.push(Branch {
            label, control_points: cps, sub_indices, centroid,
        });
    }

    /// Add mycelium edge between two trees (Fase T3).
    pub fn add_mycelium(&mut self, tree_a: usize, tree_b: usize, weight: f32) {
        if tree_a == tree_b || tree_a >= self.trees.len() || tree_b >= self.trees.len() { return; }
        // Skip duplicates
        for e in &self.mycelium {
            if (e.tree_a == tree_a && e.tree_b == tree_b)
            || (e.tree_a == tree_b && e.tree_b == tree_a) {
                return;
            }
        }
        self.mycelium.push(MyceliumEdge { tree_a, tree_b, weight });
    }

    /// Find related trees via mycelium 1-hop. Returns (tree_idx, edge_weight) sorted desc.
    pub fn mycelium_neighbors(&self, tree_idx: usize) -> Vec<(usize, f32)> {
        let mut nbrs: Vec<(usize, f32)> = self.mycelium.iter()
            .filter_map(|e| {
                if e.tree_a == tree_idx { Some((e.tree_b, e.weight)) }
                else if e.tree_b == tree_idx { Some((e.tree_a, e.weight)) }
                else { None }
            })
            .collect();
        nbrs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        nbrs
    }

    /// Build mycelium graph from tree positions: edges between trees with proximity > threshold.
    /// Math: weight = 1 - euclidean_distance, threshold for edge: weight > 0.4
    pub fn auto_build_mycelium(&mut self, weight_threshold: f32) {
        self.mycelium.clear();
        let n = self.trees.len();
        for i in 0..n {
            for j in i+1..n {
                let d = self.trees[i].position.sub(self.trees[j].position).norm();
                let weight = (1.0 - d).max(0.0);
                if weight >= weight_threshold {
                    self.mycelium.push(MyceliumEdge { tree_a: i, tree_b: j, weight });
                }
            }
        }
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
    /// Uses random projection (Johnson-Lindenstrauss) via 3 deterministic
    /// anchor HVs. Hamming distance to each anchor normalizes to one axis.
    /// This preserves semantic distances much better than naïve bit-bucket popcount.
    pub fn hv_to_position(hv: &[u64; 16]) -> Vec3 {
        // 3 deterministic anchor HVs derived from fixed seeds (xorshift)
        // Compile-time computed via const fn equivalent.
        let anchors = anchor_hvs();
        let d0 = hamming_distance(hv, &anchors[0]) as f32;
        let d1 = hamming_distance(hv, &anchors[1]) as f32;
        let d2 = hamming_distance(hv, &anchors[2]) as f32;
        Vec3::new(
            (d0 / 1024.0).clamp(0.05, 0.95),
            (d1 / 1024.0).clamp(0.05, 0.95),
            (d2 / 1024.0).clamp(0.05, 0.95),
        )
    }

    pub fn n_trees(&self) -> usize { self.trees.len() }
}

/// 3 deterministic anchor HVs for HV→3D random projection.
fn anchor_hvs() -> [[u64; 16]; 3] {
    let seeds = [0xA53F_9B17_C4DE_2901u64, 0x6F1A_3C8B_5D2E_7400, 0x9E37_79B9_7F4A_7C15];
    let mut out = [[0u64; 16]; 3];
    for (axis, &seed) in seeds.iter().enumerate() {
        let mut state = seed;
        for i in 0..16 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out[axis][i] = state;
        }
    }
    out
}

#[inline]
fn hamming_distance(a: &[u64; 16], b: &[u64; 16]) -> u32 {
    let mut d = 0u32;
    for i in 0..16 { d += (a[i] ^ b[i]).count_ones(); }
    d
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
        // With random projection (hamming-to-anchors), all-zero HV gives popcount/1024 of anchor.
        // All-bits-set HV gives (1024 - popcount)/1024 of anchor.
        // The two must sum to ~1.0 per axis.
        let hv0 = [0u64; 16];
        let hv1 = [!0u64; 16];
        let p0 = Forest::hv_to_position(&hv0);
        let p1 = Forest::hv_to_position(&hv1);
        let sum_x = p0.x + p1.x;
        let sum_y = p0.y + p1.y;
        let sum_z = p0.z + p1.z;
        // Should sum near 1.0 (within clamp range 0.05..0.95)
        assert!((sum_x - 1.0).abs() < 0.2);
        assert!((sum_y - 1.0).abs() < 0.2);
        assert!((sum_z - 1.0).abs() < 0.2);
    }

    #[test]
    fn bspline_eval_endpoints() {
        let b = Branch {
            label: "test".into(),
            control_points: [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            sub_indices: vec![],
            centroid: [0u64; 16],
        };
        // B-spline cubic eval at t=0.5 should be inside convex hull
        let mid = b.eval(0.5);
        assert!(mid.x > 0.0 && mid.x < 1.0);
        assert!(mid.y > 0.0 && mid.y < 1.0);
    }

    #[test]
    fn mycelium_neighbors_works() {
        let mut forest = Forest::new();
        forest.plant_tree("A".into(), Vec3::new(0.1, 0.1, 0.1), 1.0, 10);
        forest.plant_tree("B".into(), Vec3::new(0.2, 0.2, 0.2), 1.0, 10);
        forest.plant_tree("C".into(), Vec3::new(0.9, 0.9, 0.9), 1.0, 10);
        forest.add_mycelium(0, 1, 0.8);
        forest.add_mycelium(0, 2, 0.3);
        let nbrs = forest.mycelium_neighbors(0);
        assert_eq!(nbrs.len(), 2);
        assert_eq!(nbrs[0].0, 1); // higher weight first
        assert!((nbrs[0].1 - 0.8).abs() < 0.01);
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
