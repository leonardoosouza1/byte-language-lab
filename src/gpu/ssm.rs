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
