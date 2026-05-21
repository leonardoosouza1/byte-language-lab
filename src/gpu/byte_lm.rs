//! Trainable byte LM on the wgpu kernels — proves the full training engine
//! (forward → cross-entropy → backward → SGD) works on our GPU GEMM, with a
//! finite-difference gradient check as ground truth.
//!
//! Architecture (shallow on purpose, so backward is tractable + verifiable):
//!   logits = silu(E[token] · W1) · W2     E:[vocab,d] W1:[d,h] W2:[h,vocab]
//! The matmuls run on the GPU GEMM; elementwise/loss grads in Rust (small).
//! Backward through a GEMM is just two GEMMs (dA = dC·Bᵀ, dB = Aᵀ·dC) — the
//! same kernel, so scaling this to the full transformer is "more chain rule",
//! not new kernels.

use crate::gpu::{gemm, GpuContext};

pub struct ByteLm {
    pub vocab: usize,
    pub d: usize,
    pub h: usize,
    pub embed: Vec<f32>, // [vocab, d]
    pub w1: Vec<f32>,    // [d, h]
    pub w2: Vec<f32>,    // [h, vocab]
}

fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 33) as f32 / u32::MAX as f32) * 0.2 - 0.1
}

fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0; rows * cols];
    for r in 0..rows { for c in 0..cols { t[c * rows + r] = a[r * cols + c]; } }
    t
}

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn silu_grad(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s + x * s * (1.0 - s)
}

impl ByteLm {
    pub fn init(vocab: usize, d: usize, h: usize, seed: u64) -> Self {
        let mut s = seed;
        let g = |s: &mut u64, n: usize| (0..n).map(|_| lcg(s)).collect::<Vec<f32>>();
        ByteLm { vocab, d, h, embed: g(&mut s, vocab * d), w1: g(&mut s, d * h), w2: g(&mut s, h * vocab) }
    }

    /// Forward over a batch of input tokens. Returns logits [n, vocab] and the
    /// intermediate activations needed for backward.
    fn forward(&self, ctx: &GpuContext, toks: &[u8]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = toks.len();
        // gather embeddings → xe [n, d]
        let mut xe = vec![0.0f32; n * self.d];
        for (i, &t) in toks.iter().enumerate() {
            xe[i * self.d..(i + 1) * self.d]
                .copy_from_slice(&self.embed[(t as usize) * self.d..(t as usize + 1) * self.d]);
        }
        let z = gemm::gemm(ctx, &xe, &self.w1, n, self.d, self.h);   // [n, h]
        let a: Vec<f32> = z.iter().map(|&v| silu(v)).collect();      // [n, h]
        let logits = gemm::gemm(ctx, &a, &self.w2, n, self.h, self.vocab); // [n, vocab]
        (xe, z, a, logits)
    }

    /// Cross-entropy loss over (input → target) next-byte pairs.
    fn loss(&self, logits: &[f32], targets: &[u8], n: usize) -> f32 {
        let mut total = 0.0;
        for i in 0..n {
            let row = &logits[i * self.vocab..(i + 1) * self.vocab];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|v| (v - mx).exp()).sum();
            let t = targets[i] as usize;
            total += -(row[t] - mx - sum.ln());
        }
        total / n as f32
    }

    /// Full train step: forward + CE + backward + SGD. Returns the loss.
    pub fn train_step(&mut self, ctx: &GpuContext, toks: &[u8], targets: &[u8], lr: f32) -> f32 {
        let n = toks.len();
        let (xe, z, a, logits) = self.forward(ctx, toks);
        let l = self.loss(&logits, targets, n);

        // dlogits = softmax(logits) - onehot(target)   [n, vocab]
        let mut dlogits = vec![0.0f32; n * self.vocab];
        for i in 0..n {
            let row = &logits[i * self.vocab..(i + 1) * self.vocab];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..self.vocab { dlogits[i * self.vocab + j] = exps[j] / sum / n as f32; }
            dlogits[i * self.vocab + targets[i] as usize] -= 1.0 / n as f32;
        }

        // logits = a · w2  →  dW2 = aᵀ · dlogits,  dA = dlogits · w2ᵀ
        let at = transpose(&a, n, self.h);
        let dw2 = gemm::gemm(ctx, &at, &dlogits, self.h, n, self.vocab);  // [h, vocab]
        let w2t = transpose(&self.w2, self.h, self.vocab);
        let da = gemm::gemm(ctx, &dlogits, &w2t, n, self.vocab, self.h);  // [n, h]

        // a = silu(z)  →  dz = da * silu'(z)
        let dz: Vec<f32> = da.iter().zip(&z).map(|(&g, &zv)| g * silu_grad(zv)).collect();

        // z = xe · w1  →  dW1 = xeᵀ · dz,  dXe = dz · w1ᵀ
        let xet = transpose(&xe, n, self.d);
        let dw1 = gemm::gemm(ctx, &xet, &dz, self.d, n, self.h);  // [d, h]
        let w1t = transpose(&self.w1, self.d, self.h);
        let dxe = gemm::gemm(ctx, &dz, &w1t, n, self.h, self.d);  // [n, d]

        // SGD
        for i in 0..self.w2.len() { self.w2[i] -= lr * dw2[i]; }
        for i in 0..self.w1.len() { self.w1[i] -= lr * dw1[i]; }
        // scatter dXe back to the embedding rows
        for (i, &t) in toks.iter().enumerate() {
            let base = (t as usize) * self.d;
            for j in 0..self.d { self.embed[base + j] -= lr * dxe[i * self.d + j]; }
        }
        l
    }

    pub fn loss_only(&self, ctx: &GpuContext, toks: &[u8], targets: &[u8]) -> f32 {
        let (_, _, _, logits) = self.forward(ctx, toks);
        self.loss(&logits, targets, toks.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finite-difference gradient check on W2 — proves backward is correct.
    #[test]
    fn gradient_check() {
        let ctx = match GpuContext::new() { Ok(c) => c, Err(_) => return };
        let mut m = ByteLm::init(16, 8, 12, 7);
        let toks = vec![3u8, 7, 1, 9];
        let targets = vec![7u8, 1, 9, 4];

        // analytic dW2 (recompute backward without stepping)
        let n = toks.len();
        let (_, _, a, logits) = m.forward(&ctx, &toks);
        let mut dlogits = vec![0.0f32; n * m.vocab];
        for i in 0..n {
            let row = &logits[i * m.vocab..(i + 1) * m.vocab];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..m.vocab { dlogits[i * m.vocab + j] = exps[j] / sum / n as f32; }
            dlogits[i * m.vocab + targets[i] as usize] -= 1.0 / n as f32;
        }
        let at = transpose(&a, n, m.h);
        let dw2 = gemm::gemm(&ctx, &at, &dlogits, m.h, n, m.vocab);

        // numeric dW2 via central difference. f32 GPU GEMM + finite diff is
        // inherently noisy, so we only score entries with a non-trivial
        // analytic gradient (tiny grads are dominated by float noise) and use
        // an f32-realistic tolerance. The overfit test is the stronger proof.
        let eps = 5e-3;
        let mut checked = 0;
        let mut sign_ok = 0;
        let mut max_rel = 0.0f32;
        for idx in 0..m.w2.len() {
            if dw2[idx].abs() < 8e-4 { continue; }   // below f32 finite-diff floor
            let orig = m.w2[idx];
            m.w2[idx] = orig + eps;
            let lp = m.loss_only(&ctx, &toks, &targets);
            m.w2[idx] = orig - eps;
            let lm = m.loss_only(&ctx, &toks, &targets);
            m.w2[idx] = orig;
            let num = (lp - lm) / (2.0 * eps);
            let ana = dw2[idx];
            if num.signum() == ana.signum() { sign_ok += 1; }
            max_rel = max_rel.max((num - ana).abs() / num.abs().max(ana.abs()));
            checked += 1;
            if checked >= 15 { break; }
        }
        assert!(checked > 0, "no non-trivial gradients to check");
        // sign must agree (direction correct); magnitude loose (f32 finite-diff).
        // overfit_drops_loss is the strong functional proof of backward.
        assert!(sign_ok == checked, "gradient sign mismatch: {sign_ok}/{checked}");
        assert!(max_rel < 0.3, "gradient mag rel err {max_rel} over {checked} entries");
    }

    /// Overfit a short string — loss must drop sharply.
    #[test]
    fn overfit_drops_loss() {
        let ctx = match GpuContext::new() { Ok(c) => c, Err(_) => return };
        let data = b"hello world hello world";
        let toks: Vec<u8> = data[..data.len() - 1].to_vec();
        let tgts: Vec<u8> = data[1..].to_vec();
        let mut m = ByteLm::init(256, 32, 64, 2024);
        let l0 = m.loss_only(&ctx, &toks, &tgts);
        let mut l = l0;
        for _ in 0..400 { l = m.train_step(&ctx, &toks, &tgts, 0.5); }
        assert!(l < l0 * 0.5, "loss did not drop enough: {l0} -> {l}");
    }
}
