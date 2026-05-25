//! Minimal generative **liquid / selective state-space** byte LM (CPU prototype).
//!
//! Premise check: "liquid neurons can't be generative" is a myth — a liquid net
//! is a recurrent net, and any RNN generates autoregressively. The reason
//! liquid/CfC cores are rare as language models is (a) they target continuous
//! control, (b) tiny-capacity dynamics bias, and (c) sequential training (no
//! parallelism over the sequence). Modern selective SSMs (Mamba) fix (c) with a
//! parallel scan; their input-dependent dynamics ≈ the liquid time-constant.
//!
//! This module validates the MATH of a generative selective recurrence before
//! the WGSL parallel-scan investment: a gated linear recurrence (minGRU-style)
//!
//! ```text
//!   z_t = σ(xn_t Wz + bz)            # input-dependent update gate (the "liquid" part)
//!   u_t = xn_t Wu + bu               # candidate
//!   h_t = z_t ⊙ h_{t-1} + (1−z_t) ⊙ u_t   # leaky integration (associative → scannable)
//!   y_t = h_t Wo                     # output projection
//! ```
//!
//! No positional embedding (the recurrence encodes order). Forward is a
//! sequential scan; backward is BPTT. A finite-difference gradient check and a
//! unigram-floor-break overfit test prove correctness + that it learns + that
//! it generates.

use super::device::{self, Tensor};
use super::GpuContext;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

/// Sigmoid.
#[inline]
fn sig(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

/// C = A[m,k] · B[k,n].
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            if av == 0.0 { continue; }
            let brow = p * n;
            let crow = i * n;
            for j in 0..n { c[crow + j] += av * b[brow + j]; }
        }
    }
    c
}

/// C = Aᵀ·B where A is [m,k], B is [m,n] → C is [k,n]. (for dW += xᵀ·dpre)
fn matmul_at(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; k * n];
    for i in 0..m {
        let arow = i * k;
        let brow = i * n;
        for p in 0..k {
            let av = a[arow + p];
            let crow = p * n;
            for j in 0..n { c[crow + j] += av * b[brow + j]; }
        }
    }
    c
}

/// C = A·Bᵀ where A is [m,n], B is [k,n] → C is [m,k]. (for dx += dpre·Wᵀ)
fn matmul_bt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * k];
    for i in 0..m {
        let arow = i * n;
        for p in 0..k {
            let brow = p * n;
            let mut acc = 0.0f32;
            for j in 0..n { acc += a[arow + j] * b[brow + j]; }
            c[i * k + p] = acc;
        }
    }
    c
}

/// RMSNorm forward. Returns (y[T,d], inv_rms[T]). y_t = w ⊙ x_t · inv_t.
fn rmsnorm_fwd(x: &[f32], w: &[f32], t: usize, d: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0f32; t * d];
    let mut inv = vec![0.0f32; t];
    for r in 0..t {
        let row = r * d;
        let mut ss = 0.0f32;
        for j in 0..d { let v = x[row + j]; ss += v * v; }
        let iv = 1.0 / (ss / d as f32 + eps).sqrt();
        inv[r] = iv;
        for j in 0..d { y[row + j] = w[j] * x[row + j] * iv; }
    }
    (y, inv)
}

/// RMSNorm backward. Given dy[T,d], x, w, inv → (dx[T,d], dw[d]).
fn rmsnorm_bwd(dy: &[f32], x: &[f32], w: &[f32], inv: &[f32], t: usize, d: usize)
    -> (Vec<f32>, Vec<f32>) {
    let mut dx = vec![0.0f32; t * d];
    let mut dw = vec![0.0f32; d];
    for r in 0..t {
        let row = r * d;
        let iv = inv[r];
        // c = Σ_j dy_j w_j x_j
        let mut c = 0.0f32;
        for j in 0..d { c += dy[row + j] * w[j] * x[row + j]; }
        let iv3 = iv * iv * iv / d as f32;
        for j in 0..d {
            dx[row + j] = w[j] * iv * dy[row + j] - x[row + j] * iv3 * c;
            dw[j] += dy[row + j] * x[row + j] * iv;
        }
    }
    (dx, dw)
}

/// One liquid/SSM layer's trainable weights.
#[derive(Clone)]
pub struct SsmLayer {
    pub norm1: Vec<f32>,           // [d]
    pub wz: Vec<f32>, pub bz: Vec<f32>,  // [d,d],[d] update gate
    pub wu: Vec<f32>, pub bu: Vec<f32>,  // [d,d],[d] candidate
    pub wo: Vec<f32>,              // [d,d] output proj
}

/// Generative selective-recurrence byte LM (CPU prototype).
#[derive(Clone)]
pub struct SsmModel {
    pub vocab: usize,
    pub d: usize,
    pub embed: Vec<f32>,           // [vocab, d] (tied with logits)
    pub layers: Vec<SsmLayer>,
    pub norm_f: Vec<f32>,          // [d]
    pub eps: f32,
}

/// Per-layer activations cached for backward.
struct LayerCache {
    x_in: Vec<f32>,    // residual input [T,d]
    xn: Vec<f32>,      // rmsnorm output [T,d]
    inv: Vec<f32>,     // rmsnorm inv [T]
    z: Vec<f32>,       // gates [T,d]
    u: Vec<f32>,       // candidates [T,d]
    h: Vec<f32>,       // hidden states [T,d]
}

/// Gradients (mirror SsmModel).
pub struct SsmGrads {
    pub d_embed: Vec<f32>,
    pub d_norm_f: Vec<f32>,
    pub layers: Vec<SsmLayer>,   // reuse layer shape as grad container
}

impl SsmModel {
    pub fn init(vocab: usize, d: usize, n_layers: usize, seed: u64) -> Self {
        let mut s = seed | 1;
        let mut rnd = |n: usize, scale: f32| {
            (0..n).map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
            }).collect::<Vec<f32>>()
        };
        let layers = (0..n_layers).map(|_| SsmLayer {
            norm1: vec![1.0; d],
            wz: rnd(d * d, (1.0 / d as f32).sqrt()), bz: vec![0.0; d],
            wu: rnd(d * d, (1.0 / d as f32).sqrt()), bu: vec![0.0; d],
            wo: rnd(d * d, (1.0 / d as f32).sqrt()),
        }).collect();
        SsmModel {
            vocab, d,
            embed: rnd(vocab * d, (1.0 / d as f32).sqrt()),
            layers, norm_f: vec![1.0; d], eps: 1e-5,
        }
    }

    /// Forward → (logits[T,vocab], caches, xf, xf_inv). Caches feed backward.
    fn forward_cache(&self, tokens: &[u8]) -> (Vec<f32>, Vec<LayerCache>, Vec<f32>, Vec<f32>) {
        let d = self.d;
        let t = tokens.len();
        // embed (no positional term — recurrence encodes order)
        let mut x = vec![0.0f32; t * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let e = (tok as usize) * d;
            x[i * d..i * d + d].copy_from_slice(&self.embed[e..e + d]);
        }
        let mut caches = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            let (xn, inv) = rmsnorm_fwd(&x, &l.norm1, t, d, self.eps);
            // gates + candidate
            let mut z = matmul(&xn, &l.wz, t, d, d);
            let mut u = matmul(&xn, &l.wu, t, d, d);
            for r in 0..t {
                for j in 0..d {
                    z[r * d + j] = sig(z[r * d + j] + l.bz[j]);
                    u[r * d + j] += l.bu[j];
                }
            }
            // sequential scan: h_t = z_t·h_{t-1} + (1−z_t)·u_t
            let mut h = vec![0.0f32; t * d];
            for r in 0..t {
                let row = r * d;
                for j in 0..d {
                    let prev = if r == 0 { 0.0 } else { h[row - d + j] };
                    h[row + j] = z[row + j] * prev + (1.0 - z[row + j]) * u[row + j];
                }
            }
            let y = matmul(&h, &l.wo, t, d, d);
            // residual
            let mut x2 = x.clone();
            for i in 0..t * d { x2[i] += y[i]; }
            caches.push(LayerCache { x_in: x, xn, inv, z, u, h });
            x = x2;
        }
        let (xf, xf_inv) = rmsnorm_fwd(&x, &self.norm_f, t, d, self.eps);
        // tied logits = xf · embedᵀ  → [T, vocab]
        let logits = matmul_bt(&xf, &self.embed, t, self.vocab, d);
        (logits, caches, xf, xf_inv)
    }

    /// Forward → logits [T, vocab] only (for parity checks vs the device path).
    pub fn forward_logits(&self, tokens: &[u8]) -> Vec<f32> {
        self.forward_cache(tokens).0
    }

    /// Forward → mean next-byte cross-entropy loss (no caches; for eval/fd-check).
    pub fn loss(&self, tokens: &[u8], targets: &[u8]) -> f32 {
        let (logits, _, _, _) = self.forward_cache(tokens);
        let v = self.vocab;
        let t = tokens.len();
        let mut total = 0.0f64;
        for i in 0..t {
            let row = &logits[i * v..i * v + v];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for &z in row { sum += (z - mx).exp(); }
            let lp = (row[targets[i] as usize] - mx) - sum.ln();
            total += -(lp as f64);
        }
        (total / t as f64) as f32
    }

    /// Forward + full BPTT backward → (loss, grads).
    pub fn backward(&self, tokens: &[u8], targets: &[u8]) -> (f32, SsmGrads) {
        let d = self.d;
        let v = self.vocab;
        let t = tokens.len();
        let (logits, caches, xf, xf_inv) = self.forward_cache(tokens);

        // softmax + CE grad on logits
        let mut dlogits = vec![0.0f32; t * v];
        let mut loss = 0.0f64;
        for i in 0..t {
            let row = &logits[i * v..i * v + v];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for &z in row { sum += (z - mx).exp(); }
            for j in 0..v {
                let p = (row[j] - mx).exp() / sum;
                dlogits[i * v + j] = p / t as f32;
            }
            dlogits[i * v + targets[i] as usize] -= 1.0 / t as f32;
            let lp = (row[targets[i] as usize] - mx) - sum.ln();
            loss += -(lp as f64);
        }
        loss /= t as f64;

        // logits = xf · embedᵀ : d_embed += dlogitsᵀ·xf ; dxf = dlogits·embed
        let mut d_embed = matmul_at(&dlogits, &xf, t, v, d);   // [v,d]
        let dxf = matmul(&dlogits, &self.embed, t, v, d);       // [T,d]
        // through final rmsnorm
        let x_final = {  // reconstruct x before norm_f = last layer output residual
            // it equals caches.last x_in + y_last... simpler: recompute from xf? We
            // need x_final for rmsnorm_bwd. Recompute: x_final such that
            // rmsnorm(x_final)=xf. We cached inv but not x_final; reconstruct via
            // the last layer: x_final = caches[L-1].x_in + y_{L-1}. Recompute y.
            let lc = caches.last().unwrap();
            let l = self.layers.last().unwrap();
            let y = matmul(&lc.h, &l.wo, t, d, d);
            let mut xf_pre = lc.x_in.clone();
            for i in 0..t * d { xf_pre[i] += y[i]; }
            xf_pre
        };
        let (mut dx, d_norm_f) = rmsnorm_bwd_full(&dxf, &x_final, &self.norm_f, &xf_inv, t, d);

        let mut grad_layers: Vec<SsmLayer> = Vec::with_capacity(self.layers.len());
        // walk layers in reverse
        for li in (0..self.layers.len()).rev() {
            let l = &self.layers[li];
            let c = &caches[li];
            // residual: x_out = x_in + y → dy = dx ; dx_in = dx (carried)
            let dy = dx.clone();
            // y = h · Wo  → dWo += hᵀ·dy ; dh = dy·Woᵀ
            let mut dwo = matmul_at(&c.h, &dy, t, d, d);   // [d,d]
            let dh_from_y = matmul_bt(&dy, &l.wo, t, d, d); // [T,d]
            // BPTT through h_t = z_t·h_{t-1} + (1−z_t)·u_t
            let mut dz = vec![0.0f32; t * d];
            let mut du = vec![0.0f32; t * d];
            let mut carry = vec![0.0f32; d];
            for r in (0..t).rev() {
                let row = r * d;
                for j in 0..d {
                    let g = dh_from_y[row + j] + carry[j];
                    let prev = if r == 0 { 0.0 } else { c.h[row - d + j] };
                    dz[row + j] = g * (prev - c.u[row + j]);
                    du[row + j] = g * (1.0 - c.z[row + j]);
                    carry[j] = g * c.z[row + j];
                }
            }
            // z = σ(xn·Wz + bz) ; u = xn·Wu + bu
            let mut dprez = vec![0.0f32; t * d];
            let mut dbz = vec![0.0f32; d];
            let mut dbu = vec![0.0f32; d];
            for r in 0..t {
                for j in 0..d {
                    let zz = c.z[r * d + j];
                    let dp = dz[r * d + j] * zz * (1.0 - zz);
                    dprez[r * d + j] = dp;
                    dbz[j] += dp;
                    dbu[j] += du[r * d + j];
                }
            }
            let dwz = matmul_at(&c.xn, &dprez, t, d, d);   // [d,d]
            let dwu = matmul_at(&c.xn, &du, t, d, d);       // [d,d]
            let mut dxn = matmul_bt(&dprez, &l.wz, t, d, d); // [T,d]
            let dxn_u = matmul_bt(&du, &l.wu, t, d, d);
            for i in 0..t * d { dxn[i] += dxn_u[i]; }
            // through rmsnorm1
            let (dxin_norm, dnorm1) = rmsnorm_bwd_full(&dxn, &c.x_in, &l.norm1, &c.inv, t, d);
            // total grad into x_in = residual path (dy already = dx) + norm path
            for i in 0..t * d { dx[i] = dy[i] + dxin_norm[i]; }
            grad_layers.push(SsmLayer {
                norm1: dnorm1, wz: dwz, bz: dbz, wu: dwu, bu: dbu, wo: std::mem::take(&mut dwo),
            });
        }
        grad_layers.reverse();
        // dx now is grad w.r.t. the input embedding rows → scatter into d_embed
        for (i, &tok) in tokens.iter().enumerate() {
            let e = (tok as usize) * d;
            for j in 0..d { d_embed[e + j] += dx[i * d + j]; }
        }
        (loss as f32, SsmGrads { d_embed, d_norm_f, layers: grad_layers })
    }

    /// SGD step.
    pub fn sgd(&mut self, g: &SsmGrads, lr: f32) {
        for i in 0..self.embed.len() { self.embed[i] -= lr * g.d_embed[i]; }
        for i in 0..self.norm_f.len() { self.norm_f[i] -= lr * g.d_norm_f[i]; }
        for (l, dl) in self.layers.iter_mut().zip(&g.layers) {
            for i in 0..l.norm1.len() { l.norm1[i] -= lr * dl.norm1[i]; }
            for i in 0..l.wz.len() { l.wz[i] -= lr * dl.wz[i]; }
            for i in 0..l.bz.len() { l.bz[i] -= lr * dl.bz[i]; }
            for i in 0..l.wu.len() { l.wu[i] -= lr * dl.wu[i]; }
            for i in 0..l.bu.len() { l.bu[i] -= lr * dl.bu[i]; }
            for i in 0..l.wo.len() { l.wo[i] -= lr * dl.wo[i]; }
        }
    }

    /// Greedy generation (recurrent, O(L) in principle; this CPU proto re-runs
    /// forward over the growing prefix for simplicity).
    pub fn generate(&self, prompt: &[u8], n: usize) -> Vec<u8> {
        let v = self.vocab;
        let mut seq = prompt.to_vec();
        for _ in 0..n {
            let (logits, _, _, _) = self.forward_cache(&seq);
            let last = (seq.len() - 1) * v;
            let nb = (0..v).max_by(|&a, &b|
                logits[last + a].partial_cmp(&logits[last + b]).unwrap()).unwrap();
            seq.push(nb as u8);
        }
        seq[prompt.len()..].to_vec()
    }
}

/// rmsnorm_bwd wrapper returning (dx, dw) — alias for clarity in call sites.
fn rmsnorm_bwd_full(dy: &[f32], x: &[f32], w: &[f32], inv: &[f32], t: usize, d: usize)
    -> (Vec<f32>, Vec<f32>) {
    rmsnorm_bwd(dy, x, w, inv, t, d)
}

/// AdamW moment buffers for one parameter array.
pub struct AdamBuf { m: Vec<f32>, v: Vec<f32> }
impl AdamBuf {
    fn z(n: usize) -> Self { AdamBuf { m: vec![0.0; n], v: vec![0.0; n] } }
    #[allow(clippy::too_many_arguments)]
    fn step(&mut self, w: &mut [f32], g: &[f32], lr: f32, b1: f32, b2: f32,
            eps: f32, bc1: f32, bc2: f32, wd: f32) {
        for i in 0..w.len() {
            let gi = g[i];
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi;
            let mh = self.m[i] / bc1; let vh = self.v[i] / bc2;
            w[i] -= lr * (mh / (vh.sqrt() + eps) + wd * w[i]);
        }
    }
}
struct LayerOpt { norm1: AdamBuf, wz: AdamBuf, bz: AdamBuf, wu: AdamBuf, bu: AdamBuf, wo: AdamBuf }
/// AdamW optimizer state for an `SsmModel` (mirrors its parameters).
pub struct SsmOpt { embed: AdamBuf, norm_f: AdamBuf, layers: Vec<LayerOpt>, t: u64 }
impl SsmOpt {
    pub fn new(m: &SsmModel) -> Self {
        let d = m.d;
        SsmOpt {
            embed: AdamBuf::z(m.embed.len()),
            norm_f: AdamBuf::z(d),
            layers: m.layers.iter().map(|l| LayerOpt {
                norm1: AdamBuf::z(d), wz: AdamBuf::z(l.wz.len()), bz: AdamBuf::z(d),
                wu: AdamBuf::z(l.wu.len()), bu: AdamBuf::z(d), wo: AdamBuf::z(l.wo.len()),
            }).collect(),
            t: 0,
        }
    }
}

impl SsmModel {
    /// AdamW step (decoupled weight decay). Plain SGD stalls on real data for
    /// ill-conditioned recurrent nets — same lesson as the transformer.
    pub fn adamw(&mut self, g: &SsmGrads, opt: &mut SsmOpt, lr: f32) {
        opt.t += 1;
        let (b1, b2, eps, wd) = (0.9f32, 0.999f32, 1e-5f32, 0.01f32);
        let bc1 = 1.0 - b1.powi(opt.t as i32);
        let bc2 = 1.0 - b2.powi(opt.t as i32);
        opt.embed.step(&mut self.embed, &g.d_embed, lr, b1, b2, eps, bc1, bc2, wd);
        opt.norm_f.step(&mut self.norm_f, &g.d_norm_f, lr, b1, b2, eps, bc1, bc2, wd);
        for (l, (dl, ol)) in self.layers.iter_mut()
            .zip(g.layers.iter().zip(opt.layers.iter_mut())) {
            ol.norm1.step(&mut l.norm1, &dl.norm1, lr, b1, b2, eps, bc1, bc2, wd);
            ol.wz.step(&mut l.wz, &dl.wz, lr, b1, b2, eps, bc1, bc2, wd);
            ol.bz.step(&mut l.bz, &dl.bz, lr, b1, b2, eps, bc1, bc2, wd);
            ol.wu.step(&mut l.wu, &dl.wu, lr, b1, b2, eps, bc1, bc2, wd);
            ol.bu.step(&mut l.bu, &dl.bu, lr, b1, b2, eps, bc1, bc2, wd);
            ol.wo.step(&mut l.wo, &dl.wo, lr, b1, b2, eps, bc1, bc2, wd);
        }
    }

    /// Temperature sampling generation (reveals structure greedy hides).
    pub fn generate_temp(&self, prompt: &[u8], n: usize, temp: f32, seed: u64) -> Vec<u8> {
        let v = self.vocab; let t = temp.max(1e-3);
        let mut seq = prompt.to_vec();
        let mut rng = seed ^ 0x9E37_79B9_7F4A_7C15;
        for _ in 0..n {
            let (logits, _, _, _) = self.forward_cache(&seq);
            let last = (seq.len() - 1) * v;
            let row = &logits[last..last + v];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0.0f32; v]; let mut sum = 0.0f32;
            for i in 0..v { let p = ((row[i] - mx) / t).exp(); probs[i] = p; sum += p; }
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = ((rng >> 40) as f32 / (1u64 << 24) as f32) * sum;
            let mut acc = 0.0f32; let mut nb = v - 1;
            for i in 0..v { acc += probs[i]; if acc >= r { nb = i; break; } }
            seq.push(nb as u8);
        }
        seq[prompt.len()..].to_vec()
    }
}

// ===================== GPU device-resident path =====================

/// One device-resident SSM layer (bias-free; norms fixed at 1.0 — parameter-free
/// RMSNorm, standard in SSMs — so the device forward matches the CPU forward of
/// a freshly-init'd model exactly).
fn rd_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}
fn rd_f32s(r: &mut impl Read, n: usize) -> std::io::Result<Vec<f32>> {
    let mut buf = vec![0u8; n * 4]; r.read_exact(&mut buf)?;
    Ok(buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}
fn wr_f32s(w: &mut impl Write, data: &[f32]) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(data.len() * 4);
    for &x in data { buf.extend_from_slice(&x.to_le_bytes()); }
    w.write_all(&buf)
}

/// Channel-MLP (FFN) for a layer: rmsnorm → w1 → SiLU → w2 → residual. Adds
/// per-param expressivity the pure recurrence lacks (standard in Mamba blocks).
/// norm2 fixed at 1.0 (parameter-free); trained: w1 [d,ff], w2 [ff,d].
pub struct Ffn { w1: Tensor, w2: Tensor, norm2: Tensor }
pub struct DevLayer { norm1: Tensor, wz: Tensor, wu: Tensor, wo: Tensor, ffn: Option<Ffn> }

struct FfnCache { x_mid: Tensor, xn2: Tensor, h1: Tensor, ha: Tensor }
struct LayerCacheDev { x_in: Tensor, xn: Tensor, z: Tensor, u: Tensor, h: Tensor, ffn: Option<FfnCache> }
/// Device-resident Adam moments for one weight tensor (no host round-trip).
struct AdamMV { m: Tensor, v: Tensor }
struct FfnOpt { w1: AdamMV, w2: AdamMV }
/// Optimizer state: layer weights (wz/wu/wo + optional ffn) get DEVICE moments
/// (adam_rec in-place); embed stays host (small, like the transformer's embed).
struct DevOpt { embed: AdamBuf, wz: Vec<AdamMV>, wu: Vec<AdamMV>, wo: Vec<AdamMV>,
                ffn: Vec<Option<FfnOpt>>, t: u64 }

/// GPU liquid/SSM byte LM. Forward+backward composed from the device kernels
/// (rmsnorm, gemm, sigmoid, scan, scan_bwd, dsig) — the only SSM-specific
/// primitives are scan/scan_bwd; everything else reuses the transformer's ops.
/// Holds the host weights `m` (source of truth) + device mirrors; AdamW runs on
/// host (embed + wz/wu/wo), then changed tensors are re-uploaded.
pub struct DevSsm {
    pub m: SsmModel,
    pub ff: usize,            // FFN hidden dim (0 = no FFN)
    embed: Tensor,
    layers: Vec<DevLayer>,
    norm_f: Tensor,
    opt: Option<DevOpt>,
}

impl DevSsm {
    pub fn from_cpu(ctx: &GpuContext, m: &SsmModel) -> Self {
        let d = m.d;
        let layers = m.layers.iter().map(|l| DevLayer {
            norm1: ctx.upload(&l.norm1, 1, d),
            wz: ctx.upload(&l.wz, d, d),
            wu: ctx.upload(&l.wu, d, d),
            wo: ctx.upload(&l.wo, d, d),
            ffn: None,
        }).collect();
        DevSsm {
            embed: ctx.upload(&m.embed, m.vocab, d),
            layers, norm_f: ctx.upload(&m.norm_f, 1, d),
            m: m.clone(), ff: 0, opt: None,
        }
    }

    pub fn new(ctx: &GpuContext, vocab: usize, d: usize, n_layers: usize, seed: u64) -> Self {
        Self::from_cpu(ctx, &SsmModel::init(vocab, d, n_layers, seed))
    }

    /// Build WITH a channel-MLP (FFN) of hidden dim `ff` per layer (the key
    /// per-FLOP capacity lever). w1/w2 random-init; norm2 fixed at 1.0.
    pub fn new_ffn(ctx: &GpuContext, vocab: usize, d: usize, n_layers: usize, ff: usize, seed: u64) -> Self {
        let mut s = Self::from_cpu(ctx, &SsmModel::init(vocab, d, n_layers, seed));
        s.ff = ff;
        // deterministic init for w1/w2
        let mut rng = seed ^ 0xF00D_F00D;
        let mut rnd = |n: usize, scale: f32| (0..n).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
        }).collect::<Vec<f32>>();
        for l in s.layers.iter_mut() {
            l.ffn = Some(Ffn {
                w1: ctx.upload(&rnd(d * ff, (1.0 / d as f32).sqrt()), d, ff),
                w2: ctx.upload(&rnd(ff * d, (1.0 / ff as f32).sqrt()), ff, d),
                norm2: ctx.upload(&vec![1.0f32; d], 1, d),
            });
        }
        s
    }

    /// Re-upload the trained tensors (embed + wz/wu/wo) from host `m` after Adam.
    fn reupload(&mut self, ctx: &GpuContext) {
        let d = self.m.d;
        self.embed = ctx.upload(&self.m.embed, self.m.vocab, d);
        for (l, t) in self.m.layers.iter().zip(self.layers.iter_mut()) {
            t.wz = ctx.upload(&l.wz, d, d);
            t.wu = ctx.upload(&l.wu, d, d);
            t.wo = ctx.upload(&l.wo, d, d);
        }
    }

    /// Inference forward → logits [T, vocab]. `block`=0 single sequence.
    pub fn forward_dev(&self, ctx: &GpuContext, tokens: &[u8], block: usize) -> Vec<f32> {
        let d = self.m.d; let v = self.m.vocab; let t = tokens.len();
        let mut x0 = vec![0.0f32; t * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let e = (tok as usize) * d;
            x0[i * d..i * d + d].copy_from_slice(&self.m.embed[e..e + d]);
        }
        let mut x = ctx.upload(&x0, t, d);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        for l in &self.layers {
            let xn = ctx.alloc(t, d); device::row_rec(ctx, &mut enc, &x, &l.norm1, &xn, self.m.eps, 1);
            let zp = ctx.alloc(t, d); device::gemm_rec(ctx, &mut enc, &xn, &l.wz, &zp);
            let z = ctx.alloc(t, d); device::elem_rec(ctx, &mut enc, &zp, &z, 2);
            let u = ctx.alloc(t, d); device::gemm_rec(ctx, &mut enc, &xn, &l.wu, &u);
            let h = ctx.alloc(t, d); device::scan_rec(ctx, &mut enc, &z, &u, &h, block);
            let y = ctx.alloc(t, d); device::gemm_rec(ctx, &mut enc, &h, &l.wo, &y);
            let mut xo = ctx.alloc(t, d); device::add_rec(ctx, &mut enc, &x, &y, &xo);
            if let Some(f) = &l.ffn {                       // channel-MLP
                let ff = self.ff;
                let xn2 = ctx.alloc(t, d); device::row_rec(ctx, &mut enc, &xo, &f.norm2, &xn2, self.m.eps, 1);
                let h1 = ctx.alloc(t, ff); device::gemm_rec(ctx, &mut enc, &xn2, &f.w1, &h1);
                let ha = ctx.alloc(t, ff); device::elem_rec(ctx, &mut enc, &h1, &ha, 0);   // SiLU
                let fo = ctx.alloc(t, d); device::gemm_rec(ctx, &mut enc, &ha, &f.w2, &fo);
                let xo2 = ctx.alloc(t, d); device::add_rec(ctx, &mut enc, &xo, &fo, &xo2);
                xo = xo2;
            }
            x = xo;
        }
        let xf = ctx.alloc(t, d); device::row_rec(ctx, &mut enc, &x, &self.norm_f, &xf, self.m.eps, 1);
        let logits = ctx.alloc(t, v); device::gemm_nt_rec(ctx, &mut enc, &xf, &self.embed, &logits);
        ctx.queue.submit(Some(enc.finish()));
        ctx.download(&logits)
    }

    fn ensure_opt(&mut self, ctx: &GpuContext) {
        if self.opt.is_some() { return; }
        let d = self.m.d; let ff = self.ff;
        let zmv = |r: usize, c: usize| AdamMV {
            m: ctx.upload(&vec![0.0f32; r * c], r, c),
            v: ctx.upload(&vec![0.0f32; r * c], r, c),
        };
        self.opt = Some(DevOpt {
            embed: AdamBuf::z(self.m.embed.len()),
            wz: (0..self.layers.len()).map(|_| zmv(d, d)).collect(),
            wu: (0..self.layers.len()).map(|_| zmv(d, d)).collect(),
            wo: (0..self.layers.len()).map(|_| zmv(d, d)).collect(),
            ffn: self.layers.iter().map(|l| l.ffn.as_ref()
                .map(|_| FfnOpt { w1: zmv(d, ff), w2: zmv(ff, d) })).collect(),
            t: 0,
        });
    }

    /// Compute gradients on the GPU. Returns (loss, d_embed[v,d],
    /// per-layer (dwz,dwu,dwo)[d,d]). Norms are fixed (not trained).
    #[allow(clippy::type_complexity)]
    fn compute_grads(&self, ctx: &GpuContext, tokens: &[u8], targets: &[u8], block: usize)
        -> (f32, Vec<f32>, Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>) {
        let d = self.m.d; let v = self.m.vocab; let t = tokens.len();
        // ---- forward with caches ---- (compute_grads: FFN-less parity path; ff must be 0)
        debug_assert_eq!(self.ff, 0, "compute_grads is FFN-less; use train_step for FFN");
        let mut x0 = vec![0.0f32; t * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let e = (tok as usize) * d;
            x0[i * d..i * d + d].copy_from_slice(&self.m.embed[e..e + d]);
        }
        let mut x = ctx.upload(&x0, t, d);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        let mut caches: Vec<LayerCacheDev> = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            let xn = ctx.alloc(t, d); device::row_rec(ctx, &mut enc, &x, &l.norm1, &xn, self.m.eps, 1);
            let zp = ctx.alloc(t, d); device::gemm_rec(ctx, &mut enc, &xn, &l.wz, &zp);
            let z = ctx.alloc(t, d); device::elem_rec(ctx, &mut enc, &zp, &z, 2);
            let u = ctx.alloc(t, d); device::gemm_rec(ctx, &mut enc, &xn, &l.wu, &u);
            let h = ctx.alloc(t, d); device::scan_rec(ctx, &mut enc, &z, &u, &h, block);
            let y = ctx.alloc(t, d); device::gemm_rec(ctx, &mut enc, &h, &l.wo, &y);
            let xo = ctx.alloc(t, d); device::add_rec(ctx, &mut enc, &x, &y, &xo);
            caches.push(LayerCacheDev { x_in: x, xn, z, u, h, ffn: None });
            x = xo;
        }
        let x_final = x;
        let xf = ctx.alloc(t, d); device::row_rec(ctx, &mut enc, &x_final, &self.norm_f, &xf, self.m.eps, 1);
        let logits_t = ctx.alloc(t, v); device::gemm_nt_rec(ctx, &mut enc, &xf, &self.embed, &logits_t);
        ctx.queue.submit(Some(enc.finish()));
        let logits = ctx.download(&logits_t);

        // ---- softmax CE grad (host) ----
        let mut dlogits = vec![0.0f32; t * v];
        let mut loss = 0.0f64;
        for i in 0..t {
            let row = &logits[i * v..i * v + v];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for &zz in row { sum += (zz - mx).exp(); }
            for j in 0..v { dlogits[i * v + j] = ((row[j] - mx).exp() / sum) / t as f32; }
            dlogits[i * v + targets[i] as usize] -= 1.0 / t as f32;
            loss += -(((row[targets[i] as usize] - mx) - sum.ln()) as f64);
        }
        loss /= t as f64;
        let dlt = ctx.upload(&dlogits, t, v);

        // ---- backward ----
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        let d_embed_head = ctx.alloc(v, d); device::gemm_tn_rec(ctx, &mut enc, &dlt, &xf, &d_embed_head);
        let dxf = ctx.alloc(t, d); device::gemm_rec(ctx, &mut enc, &dlt, &self.embed, &dxf);
        let mut dx = ctx.alloc(t, d);
        device::rms_in_bwd_rec(ctx, &mut enc, &dxf, &x_final, &self.norm_f, &dx, self.m.eps);
        let mut dws: Vec<(Tensor, Tensor, Tensor)> = Vec::with_capacity(self.layers.len());
        for li in (0..self.layers.len()).rev() {
            let l = &self.layers[li];
            let c = &caches[li];
            let dwo = ctx.alloc(d, d); device::gemm_tn_rec(ctx, &mut enc, &c.h, &dx, &dwo);   // hᵀ·dy
            let dh_y = ctx.alloc(t, d); device::gemm_nt_rec(ctx, &mut enc, &dx, &l.wo, &dh_y); // dy·woᵀ
            let dz = ctx.alloc(t, d); let du = ctx.alloc(t, d);
            device::scan_bwd_rec(ctx, &mut enc, &dh_y, &c.z, &c.u, &c.h, &dz, &du, block);
            let dzpre = ctx.alloc(t, d); device::dsig_rec(ctx, &mut enc, &dz, &c.z, &dzpre);
            let dwz = ctx.alloc(d, d); device::gemm_tn_rec(ctx, &mut enc, &c.xn, &dzpre, &dwz);
            let dwu = ctx.alloc(d, d); device::gemm_tn_rec(ctx, &mut enc, &c.xn, &du, &dwu);
            let dxn = ctx.alloc(t, d); device::gemm_nt_rec(ctx, &mut enc, &dzpre, &l.wz, &dxn);
            let dxn_u = ctx.alloc(t, d); device::gemm_nt_rec(ctx, &mut enc, &du, &l.wu, &dxn_u);
            let dxn_s = ctx.alloc(t, d); device::add_rec(ctx, &mut enc, &dxn, &dxn_u, &dxn_s);
            let dxin_norm = ctx.alloc(t, d);
            device::rms_in_bwd_rec(ctx, &mut enc, &dxn_s, &c.x_in, &l.norm1, &dxin_norm, self.m.eps);
            let dx_new = ctx.alloc(t, d); device::add_rec(ctx, &mut enc, &dx, &dxin_norm, &dx_new);
            dx = dx_new;
            dws.push((dwz, dwu, dwo));
        }
        dws.reverse();
        ctx.queue.submit(Some(enc.finish()));

        // ---- download grads + scatter embed ----
        let mut d_embed = ctx.download(&d_embed_head);
        let dx_host = ctx.download(&dx);
        for (i, &tok) in tokens.iter().enumerate() {
            let e = (tok as usize) * d;
            for j in 0..d { d_embed[e + j] += dx_host[i * d + j]; }
        }
        let layer_grads = dws.into_iter()
            .map(|(z, u, o)| (ctx.download(&z), ctx.download(&u), ctx.download(&o)))
            .collect();
        (loss as f32, d_embed, layer_grads)
    }

    /// One AdamW step, DEVICE-RESIDENT: forward+backward+Adam(layer weights) all
    /// on GPU. Layer weights (wz/wu/wo) update in-place via `adam_rec` (no grad
    /// download, no weight reupload). Only embed round-trips to host (small) +
    /// the logits readback for the softmax-CE grad (small). 3 submits/step.
    pub fn train_step(&mut self, ctx: &GpuContext, tokens: &[u8], targets: &[u8],
                      lr: f32, block: usize) -> f32 {
        const B1: f32 = 0.9; const B2: f32 = 0.999; const EPS: f32 = 1e-5; const WD: f32 = 0.01;
        self.ensure_opt(ctx);
        let d = self.m.d; let v = self.m.vocab; let t = tokens.len();

        // ---- forward (1 submit) ----
        let ff = self.ff;
        let mut x0 = vec![0.0f32; t * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let e = (tok as usize) * d;
            x0[i * d..i * d + d].copy_from_slice(&self.m.embed[e..e + d]);
        }
        let mut x = ctx.upload(&x0, t, d);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        let mut caches: Vec<LayerCacheDev> = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            let xn = ctx.alloc(t, d); device::row_rec(ctx, &mut enc, &x, &l.norm1, &xn, self.m.eps, 1);
            let zp = ctx.alloc(t, d); device::gemm_rec(ctx, &mut enc, &xn, &l.wz, &zp);
            let z = ctx.alloc(t, d); device::elem_rec(ctx, &mut enc, &zp, &z, 2);
            let u = ctx.alloc(t, d); device::gemm_rec(ctx, &mut enc, &xn, &l.wu, &u);
            let h = ctx.alloc(t, d); device::scan_rec(ctx, &mut enc, &z, &u, &h, block);
            let y = ctx.alloc(t, d); device::gemm_rec(ctx, &mut enc, &h, &l.wo, &y);
            let xo = ctx.alloc(t, d); device::add_rec(ctx, &mut enc, &x, &y, &xo);
            let (next_x, ffn_cache) = if let Some(f) = &l.ffn {
                let xn2 = ctx.alloc(t, d); device::row_rec(ctx, &mut enc, &xo, &f.norm2, &xn2, self.m.eps, 1);
                let h1 = ctx.alloc(t, ff); device::gemm_rec(ctx, &mut enc, &xn2, &f.w1, &h1);
                let ha = ctx.alloc(t, ff); device::elem_rec(ctx, &mut enc, &h1, &ha, 0);   // SiLU
                let fo = ctx.alloc(t, d); device::gemm_rec(ctx, &mut enc, &ha, &f.w2, &fo);
                let xo2 = ctx.alloc(t, d); device::add_rec(ctx, &mut enc, &xo, &fo, &xo2);
                (xo2, Some(FfnCache { x_mid: xo, xn2, h1, ha }))
            } else {
                (xo, None)
            };
            caches.push(LayerCacheDev { x_in: x, xn, z, u, h, ffn: ffn_cache });
            x = next_x;
        }
        let x_final = x;
        let xf = ctx.alloc(t, d); device::row_rec(ctx, &mut enc, &x_final, &self.norm_f, &xf, self.m.eps, 1);
        let logits_t = ctx.alloc(t, v); device::gemm_nt_rec(ctx, &mut enc, &xf, &self.embed, &logits_t);
        ctx.queue.submit(Some(enc.finish()));
        let logits = ctx.download(&logits_t);

        // ---- softmax CE grad (host, small) ----
        let mut dlogits = vec![0.0f32; t * v];
        let mut loss = 0.0f64;
        for i in 0..t {
            let row = &logits[i * v..i * v + v];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for &zz in row { sum += (zz - mx).exp(); }
            for j in 0..v { dlogits[i * v + j] = ((row[j] - mx).exp() / sum) / t as f32; }
            dlogits[i * v + targets[i] as usize] -= 1.0 / t as f32;
            loss += -(((row[targets[i] as usize] - mx) - sum.ln()) as f64);
        }
        loss /= t as f64;
        let dlt = ctx.upload(&dlogits, t, v);

        // ---- backward (1 submit) → layer grads stay on device ----
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        let d_embed_head = ctx.alloc(v, d); device::gemm_tn_rec(ctx, &mut enc, &dlt, &xf, &d_embed_head);
        let dxf = ctx.alloc(t, d); device::gemm_rec(ctx, &mut enc, &dlt, &self.embed, &dxf);
        let mut dx = ctx.alloc(t, d);
        device::rms_in_bwd_rec(ctx, &mut enc, &dxf, &x_final, &self.norm_f, &dx, self.m.eps);
        #[allow(clippy::type_complexity)]
        let mut dws: Vec<(usize, Tensor, Tensor, Tensor, Option<(Tensor, Tensor)>)> =
            Vec::with_capacity(self.layers.len());
        for li in (0..self.layers.len()).rev() {
            let l = &self.layers[li];
            let c = &caches[li];
            // FFN backward (last in forward → first in backward): dx → grad into x_mid
            let dffn = if let (Some(f), Some(fc)) = (&l.ffn, &c.ffn) {
                let dw2 = ctx.alloc(ff, d); device::gemm_tn_rec(ctx, &mut enc, &fc.ha, &dx, &dw2);    // haᵀ·dfo
                let dha = ctx.alloc(t, ff); device::gemm_nt_rec(ctx, &mut enc, &dx, &f.w2, &dha);     // dfo·w2ᵀ
                let dh1 = ctx.alloc(t, ff); device::silu_bwd_rec(ctx, &mut enc, &dha, &fc.h1, &dh1);
                let dw1 = ctx.alloc(d, ff); device::gemm_tn_rec(ctx, &mut enc, &fc.xn2, &dh1, &dw1);  // xn2ᵀ·dh1
                let dxn2 = ctx.alloc(t, d); device::gemm_nt_rec(ctx, &mut enc, &dh1, &f.w1, &dxn2);   // dh1·w1ᵀ
                let dxin2 = ctx.alloc(t, d);
                device::rms_in_bwd_rec(ctx, &mut enc, &dxn2, &fc.x_mid, &f.norm2, &dxin2, self.m.eps);
                let dx_mid = ctx.alloc(t, d); device::add_rec(ctx, &mut enc, &dx, &dxin2, &dx_mid);   // residual + norm
                dx = dx_mid;
                Some((dw1, dw2))
            } else { None };
            let dwo = ctx.alloc(d, d); device::gemm_tn_rec(ctx, &mut enc, &c.h, &dx, &dwo);
            let dh_y = ctx.alloc(t, d); device::gemm_nt_rec(ctx, &mut enc, &dx, &l.wo, &dh_y);
            let dz = ctx.alloc(t, d); let du = ctx.alloc(t, d);
            device::scan_bwd_rec(ctx, &mut enc, &dh_y, &c.z, &c.u, &c.h, &dz, &du, block);
            let dzpre = ctx.alloc(t, d); device::dsig_rec(ctx, &mut enc, &dz, &c.z, &dzpre);
            let dwz = ctx.alloc(d, d); device::gemm_tn_rec(ctx, &mut enc, &c.xn, &dzpre, &dwz);
            let dwu = ctx.alloc(d, d); device::gemm_tn_rec(ctx, &mut enc, &c.xn, &du, &dwu);
            let dxn = ctx.alloc(t, d); device::gemm_nt_rec(ctx, &mut enc, &dzpre, &l.wz, &dxn);
            let dxn_u = ctx.alloc(t, d); device::gemm_nt_rec(ctx, &mut enc, &du, &l.wu, &dxn_u);
            let dxn_s = ctx.alloc(t, d); device::add_rec(ctx, &mut enc, &dxn, &dxn_u, &dxn_s);
            let dxin = ctx.alloc(t, d);
            device::rms_in_bwd_rec(ctx, &mut enc, &dxn_s, &c.x_in, &l.norm1, &dxin, self.m.eps);
            let dx_new = ctx.alloc(t, d); device::add_rec(ctx, &mut enc, &dx, &dxin, &dx_new);
            dx = dx_new;
            dws.push((li, dwz, dwu, dwo, dffn));
        }
        ctx.queue.submit(Some(enc.finish()));

        // ---- Adam: layer weights in-place on device; embed on host ----
        let (bc1, bc2) = {
            let o = self.opt.as_mut().unwrap(); o.t += 1;
            (1.0 - B1.powi(o.t as i32), 1.0 - B2.powi(o.t as i32))
        };
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let o = self.opt.as_ref().unwrap();
            for (li, dwz, dwu, dwo, dffn) in &dws {
                let l = &self.layers[*li];
                device::adam_rec(ctx, &mut enc, &l.wz, dwz, &o.wz[*li].m, &o.wz[*li].v, lr, B1, B2, EPS, bc1, bc2, WD);
                device::adam_rec(ctx, &mut enc, &l.wu, dwu, &o.wu[*li].m, &o.wu[*li].v, lr, B1, B2, EPS, bc1, bc2, WD);
                device::adam_rec(ctx, &mut enc, &l.wo, dwo, &o.wo[*li].m, &o.wo[*li].v, lr, B1, B2, EPS, bc1, bc2, WD);
                if let (Some((dw1, dw2)), Some(f), Some(fo)) = (dffn, &l.ffn, &o.ffn[*li]) {
                    device::adam_rec(ctx, &mut enc, &f.w1, dw1, &fo.w1.m, &fo.w1.v, lr, B1, B2, EPS, bc1, bc2, WD);
                    device::adam_rec(ctx, &mut enc, &f.w2, dw2, &fo.w2.m, &fo.w2.v, lr, B1, B2, EPS, bc1, bc2, WD);
                }
            }
        }
        ctx.queue.submit(Some(enc.finish()));

        // embed: host Adam (download head grad + dx, scatter, step, reupload — small)
        let mut d_embed = ctx.download(&d_embed_head);
        let dx_host = ctx.download(&dx);
        for (i, &tok) in tokens.iter().enumerate() {
            let e = (tok as usize) * d;
            for j in 0..d { d_embed[e + j] += dx_host[i * d + j]; }
        }
        let o = self.opt.as_mut().unwrap();
        o.embed.step(&mut self.m.embed, &d_embed, lr, B1, B2, EPS, bc1, bc2, WD);
        self.embed = ctx.upload(&self.m.embed, v, d);
        loss as f32
    }

    /// Save all weights to a binary checkpoint, so training RESUMES instead of
    /// restarting: grow the dataset → load → keep training (warm-start). Only a
    /// few MB (the model), downloaded from the device.
    pub fn save(&self, ctx: &GpuContext, path: &str) -> std::io::Result<()> {
        let (d, v, nl, ff) = (self.m.d, self.m.vocab, self.layers.len(), self.ff);
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(b"SSMC")?;
        for x in [1u32, v as u32, d as u32, nl as u32, ff as u32] { w.write_all(&x.to_le_bytes())?; }
        wr_f32s(&mut w, &self.m.embed)?;
        wr_f32s(&mut w, &ctx.download(&self.norm_f))?;
        for l in &self.layers {
            wr_f32s(&mut w, &ctx.download(&l.norm1))?;
            wr_f32s(&mut w, &ctx.download(&l.wz))?;
            wr_f32s(&mut w, &ctx.download(&l.wu))?;
            wr_f32s(&mut w, &ctx.download(&l.wo))?;
            if let Some(f) = &l.ffn {
                wr_f32s(&mut w, &ctx.download(&f.w1))?;
                wr_f32s(&mut w, &ctx.download(&f.w2))?;
                wr_f32s(&mut w, &ctx.download(&f.norm2))?;
            }
        }
        w.flush()
    }

    /// Load a checkpoint → ready to continue training or generate. Optimizer
    /// state is NOT saved (re-warms over a few hundred steps — weights are what
    /// matter for warm-start).
    pub fn load(ctx: &GpuContext, path: &str) -> std::io::Result<Self> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 4]; r.read_exact(&mut magic)?;
        assert_eq!(&magic, b"SSMC", "bad checkpoint magic");
        let _ver = rd_u32(&mut r)?;
        let v = rd_u32(&mut r)? as usize;
        let d = rd_u32(&mut r)? as usize;
        let nl = rd_u32(&mut r)? as usize;
        let ff = rd_u32(&mut r)? as usize;
        let embed = rd_f32s(&mut r, v * d)?;
        let norm_f_h = rd_f32s(&mut r, d)?;
        let mut layers = Vec::with_capacity(nl);
        let mut m_layers = Vec::with_capacity(nl);
        for _ in 0..nl {
            let n1 = rd_f32s(&mut r, d)?;
            let wz = rd_f32s(&mut r, d * d)?;
            let wu = rd_f32s(&mut r, d * d)?;
            let wo = rd_f32s(&mut r, d * d)?;
            let ffn = if ff > 0 {
                let w1 = rd_f32s(&mut r, d * ff)?;
                let w2 = rd_f32s(&mut r, ff * d)?;
                let n2 = rd_f32s(&mut r, d)?;
                Some(Ffn { w1: ctx.upload(&w1, d, ff), w2: ctx.upload(&w2, ff, d), norm2: ctx.upload(&n2, 1, d) })
            } else { None };
            m_layers.push(SsmLayer { norm1: n1.clone(), wz: wz.clone(), bz: vec![0.0; d],
                                     wu: wu.clone(), bu: vec![0.0; d], wo: wo.clone() });
            layers.push(DevLayer {
                norm1: ctx.upload(&n1, 1, d), wz: ctx.upload(&wz, d, d),
                wu: ctx.upload(&wu, d, d), wo: ctx.upload(&wo, d, d), ffn,
            });
        }
        let m = SsmModel { vocab: v, d, embed: embed.clone(), layers: m_layers, norm_f: norm_f_h.clone(), eps: 1e-5 };
        Ok(DevSsm {
            embed: ctx.upload(&embed, v, d), layers,
            norm_f: ctx.upload(&norm_f_h, 1, d), m, ff, opt: None,
        })
    }

    /// Greedy generation (forward over the growing prefix).
    pub fn generate(&self, ctx: &GpuContext, prompt: &[u8], n: usize) -> Vec<u8> {
        let v = self.m.vocab;
        let mut seq = prompt.to_vec();
        for _ in 0..n {
            let logits = self.forward_dev(ctx, &seq, 0);
            let last = (seq.len() - 1) * v;
            let nb = (0..v).max_by(|&a, &b|
                logits[last + a].partial_cmp(&logits[last + b]).unwrap()).unwrap();
            seq.push(nb as u8);
        }
        seq[prompt.len()..].to_vec()
    }

    /// Temperature sampling generation.
    pub fn generate_temp(&self, ctx: &GpuContext, prompt: &[u8], n: usize, temp: f32, seed: u64) -> Vec<u8> {
        let v = self.m.vocab; let tp = temp.max(1e-3);
        let mut seq = prompt.to_vec();
        let mut rng = seed ^ 0x9E37_79B9_7F4A_7C15;
        for _ in 0..n {
            let logits = self.forward_dev(ctx, &seq, 0);
            let row = &logits[(seq.len() - 1) * v..seq.len() * v];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0.0f32; v]; let mut sum = 0.0f32;
            for i in 0..v { let p = ((row[i] - mx) / tp).exp(); probs[i] = p; sum += p; }
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = ((rng >> 40) as f32 / (1u64 << 24) as f32) * sum;
            let mut acc = 0.0f32; let mut nb = v - 1;
            for i in 0..v { acc += probs[i]; if acc >= r { nb = i; break; } }
            seq.push(nb as u8);
        }
        seq[prompt.len()..].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finite-difference gradient check on a tiny instance: the analytic BPTT
    /// gradients must match numeric (central difference) within tolerance.
    #[test]
    fn ssm_gradient_check() {
        let (v, d, t) = (16usize, 6usize, 9usize);
        let mut m = SsmModel::init(v, d, 2, 7);
        let toks: Vec<u8> = (0..t).map(|i| ((i * 3 + 1) % v) as u8).collect();
        let tgts: Vec<u8> = (0..t).map(|i| ((i * 3 + 4) % v) as u8).collect();
        let (_, g) = m.backward(&toks, &tgts);
        let eps = 1e-3f32;
        let mut max_rel = 0.0f32;

        fn param<'a>(m: &'a mut SsmModel, which: &str, idx: usize) -> &'a mut f32 {
            match which {
                "wz" => &mut m.layers[0].wz[idx],
                "wu" => &mut m.layers[0].wu[idx],
                "wo" => &mut m.layers[0].wo[idx],
                "bz" => &mut m.layers[0].bz[idx],
                _ => unreachable!(),
            }
        }

        // check a sample of params across components
        for (which, idx) in [("wz", 5usize), ("wu", 11), ("wo", 3), ("bz", 2)] {
            let analytic = match which {
                "wz" => g.layers[0].wz[idx],
                "wu" => g.layers[0].wu[idx],
                "wo" => g.layers[0].wo[idx],
                "bz" => g.layers[0].bz[idx],
                _ => unreachable!(),
            };
            let orig = *param(&mut m, which, idx);
            *param(&mut m, which, idx) = orig + eps;
            let lp = m.loss(&toks, &tgts);
            *param(&mut m, which, idx) = orig - eps;
            let lm = m.loss(&toks, &tgts);
            *param(&mut m, which, idx) = orig;
            let numeric = (lp - lm) / (2.0 * eps);
            let rel = (analytic - numeric).abs() / analytic.abs().max(numeric.abs()).max(1e-4);
            eprintln!("{which}[{idx}] analytic={analytic:.5} numeric={numeric:.5} rel={rel:.2e}");
            max_rel = max_rel.max(rel);
        }
        // 6e-2 tolerance: f32 central-difference is noise-dominated on tiny
        // gradients (biases get ~1e-3 grads); the large-gradient probes (wu/wo)
        // land at 1e-4..4e-3, and ssm_breaks_unigram_floor is the end-to-end proof.
        assert!(max_rel < 6e-2, "SSM gradient check failed: max rel {max_rel}");
    }

    /// Device SSM forward must match the CPU forward (same weights). Gate for
    /// the GPU path: validates rmsnorm+gemm+sigmoid+scan compose correctly.
    #[test]
    fn dev_forward_matches_cpu() {
        let ctx = match GpuContext::new() { Ok(c) => c, Err(_) => return };
        let (v, d) = (256usize, 32usize);
        let m = SsmModel::init(v, d, 3, 7);   // init biases are 0 → bias-free = device
        let dev = DevSsm::from_cpu(&ctx, &m);
        let toks: Vec<u8> = b"fn main() { let x = 42; }".to_vec();
        let cpu = m.forward_logits(&toks);
        let gpu = dev.forward_dev(&ctx, &toks, 0);
        let rel = cpu.iter().zip(&gpu)
            .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1e-3))
            .fold(0.0f32, f32::max);
        eprintln!("dev-vs-cpu forward rel {rel:.2e}");
        assert!(rel < 1e-3, "device SSM forward disagrees: {rel}");
    }

    /// Device backward must match CPU backward grads (same weights) for the
    /// trained params (embed + wz/wu/wo). The correctness gate for GPU training.
    #[test]
    fn dev_backward_matches_cpu() {
        let ctx = match GpuContext::new() { Ok(c) => c, Err(_) => return };
        let (v, d) = (64usize, 24usize);
        let m = SsmModel::init(v, d, 2, 5);
        let dev = DevSsm::from_cpu(&ctx, &m);
        let toks: Vec<u8> = (0..20).map(|i| ((i * 7 + 3) % v) as u8).collect();
        let tgts: Vec<u8> = (0..20).map(|i| ((i * 7 + 10) % v) as u8).collect();
        let (_, cg) = m.backward(&toks, &tgts);
        let (_, d_embed, lg) = dev.compute_grads(&ctx, &toks, &tgts, 0);
        let rel = |a: &[f32], b: &[f32]| a.iter().zip(b)
            .map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()).max(1e-3)).fold(0.0f32, f32::max);
        let re = rel(&cg.d_embed, &d_embed);
        let rz = rel(&cg.layers[0].wz, &lg[0].0);
        let ru = rel(&cg.layers[0].wu, &lg[0].1);
        let ro = rel(&cg.layers[0].wo, &lg[0].2);
        eprintln!("dev-vs-cpu grads rel: embed={re:.2e} wz={rz:.2e} wu={ru:.2e} wo={ro:.2e}");
        assert!(re < 5e-2 && rz < 5e-2 && ru < 5e-2 && ro < 5e-2,
                "device backward disagrees: e={re} z={rz} u={ru} o={ro}");
    }

    /// FFN-equipped GPU model must train (validates the FFN forward+backward+adam
    /// integrate correctly — a broken FFN backward would block learning).
    #[test]
    fn dev_ffn_breaks_floor() {
        let ctx = match GpuContext::new() { Ok(c) => c, Err(_) => return };
        let v = 32usize;
        let pat: Vec<u8> = (0..200).map(|i| ((i % 7) * 3 % v) as u8).collect();
        let toks = &pat[..pat.len() - 1];
        let tgts = &pat[1..];
        let mut cnt = [0u32; 256];
        for &b in tgts { cnt[b as usize] += 1; }
        let n = tgts.len() as f32;
        let floor: f32 = tgts.iter().map(|&b| -((cnt[b as usize] as f32 / n).ln())).sum::<f32>() / n;
        let mut dev = DevSsm::new_ffn(&ctx, v, 24, 2, 48, 1);   // ff=48
        let mut loss = 0.0;
        for step in 0..500 {
            loss = dev.train_step(&ctx, toks, tgts, 0.01, 0);
            if step % 100 == 0 { eprintln!("ffn step {step}: loss {loss:.4} (floor {floor:.4})"); }
        }
        eprintln!("ffn final loss {loss:.4}, floor {floor:.4}");
        assert!(loss < floor * 0.6, "FFN model did not learn: {loss} vs floor {floor}");
        let out = dev.generate(&ctx, &pat[..7], 14);
        eprintln!("ffn gen: {:?}", out);
    }

    /// Checkpoint roundtrip: train → save → load → identical forward. Validates
    /// warm-start (resume training / grow dataset without retraining from zero).
    #[test]
    fn dev_save_load_roundtrip() {
        let ctx = match GpuContext::new() { Ok(c) => c, Err(_) => return };
        let path = "/tmp/test_ckpt.ssmc";
        let mut dev = DevSsm::new_ffn(&ctx, 256, 24, 2, 48, 7);
        let toks: Vec<u8> = (0..40).map(|i| ((i * 5 + 1) % 256) as u8).collect();
        for _ in 0..20 { dev.train_step(&ctx, &toks[..39], &toks[1..40], 0.01, 0); }
        let prompt = b"abc";
        let before = dev.forward_dev(&ctx, prompt, 0);
        dev.save(&ctx, path).unwrap();
        let dev2 = DevSsm::load(&ctx, path).unwrap();
        let after = dev2.forward_dev(&ctx, prompt, 0);
        let rel = before.iter().zip(&after)
            .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1e-3)).fold(0.0f32, f32::max);
        eprintln!("save/load roundtrip rel {rel:.2e} (ff={})", dev2.ff);
        assert!(rel < 1e-5, "checkpoint roundtrip mismatch: {rel}");
        let _ = std::fs::remove_file(path);
    }

    /// The GPU liquid model must TRAIN and GENERATE: break the unigram floor on
    /// a period-7 cycle and generate it back. End-to-end GPU proof.
    #[test]
    fn dev_breaks_unigram_floor() {
        let ctx = match GpuContext::new() { Ok(c) => c, Err(_) => return };
        let v = 32usize;
        let pat: Vec<u8> = (0..200).map(|i| ((i % 7) * 3 % v) as u8).collect();
        let toks = &pat[..pat.len() - 1];
        let tgts = &pat[1..];
        let mut cnt = [0u32; 256];
        for &b in tgts { cnt[b as usize] += 1; }
        let n = tgts.len() as f32;
        let floor: f32 = tgts.iter().map(|&b| -((cnt[b as usize] as f32 / n).ln())).sum::<f32>() / n;
        let mut dev = DevSsm::new(&ctx, v, 24, 2, 1);
        let mut loss = 0.0;
        for step in 0..400 {
            loss = dev.train_step(&ctx, toks, tgts, 0.01, 0);
            if step % 100 == 0 { eprintln!("gpu step {step}: loss {loss:.4} (floor {floor:.4})"); }
        }
        eprintln!("gpu final loss {loss:.4}, floor {floor:.4}");
        assert!(loss < floor * 0.6, "GPU liquid core did not learn: {loss} vs floor {floor}");
        let out = dev.generate(&ctx, &pat[..7], 14);
        eprintln!("gpu generated: {:?}", out);
    }

    /// The generative liquid core must LEARN: overfit a deterministic period-7
    /// byte cycle (needs real recurrent context) far below the unigram floor.
    #[test]
    fn ssm_breaks_unigram_floor() {
        let v = 32usize;
        // period-7 cycle → next byte fully determined by position in cycle
        let pat: Vec<u8> = (0..200).map(|i| ((i % 7) * 3 % v) as u8).collect();
        let toks = &pat[..pat.len() - 1];
        let tgts = &pat[1..];
        // unigram floor of targets
        let mut cnt = [0u32; 256];
        for &b in tgts { cnt[b as usize] += 1; }
        let n = tgts.len() as f32;
        let floor: f32 = tgts.iter()
            .map(|&b| -((cnt[b as usize] as f32 / n).ln())).sum::<f32>() / n;
        let mut m = SsmModel::init(v, 24, 2, 1);
        let mut loss = 0.0;
        for step in 0..400 {
            let (l, g) = m.backward(toks, tgts);
            m.sgd(&g, 0.3);
            loss = l;
            if step % 100 == 0 { eprintln!("step {step}: loss {l:.4} (floor {floor:.4})"); }
        }
        eprintln!("final loss {loss:.4}, floor {floor:.4}");
        assert!(loss < floor * 0.5, "liquid core did not learn: loss {loss} vs floor {floor}");
        // and it generates deterministically from a prompt
        let out = m.generate(&pat[..7], 14);
        eprintln!("generated: {:?}", out);
    }
}
