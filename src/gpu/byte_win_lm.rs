//! Windowed byte LM — a context-aware neural byte predictor trained on real
//! code via the GPU GEMM kernels. Unlike byte_lm (single-token context) this
//! conditions on a window of W previous bytes, so it learns real code n-grams
//! (`{` after `)`, `;` ending statements, indentation, identifiers).
//!
//!   logits = silu( concat(E[b_{t-W..t}]) · W1 ) · W2
//!   E:[vocab,d]  W1:[W*d, h]  W2:[h, vocab]
//!
//! Same gradient-checked backward as byte_lm (GEMM-backward = two GEMMs),
//! just a wider input. A real trained code model on our own kernels — the
//! bridge from "kernels work" to "a model that learned something".

use crate::gpu::gemm;
use crate::gpu::GpuContext;

pub struct ByteWinLm {
    pub vocab: usize,
    pub d: usize,
    pub win: usize,
    pub h: usize,
    pub embed: Vec<f32>, // [vocab, d]
    pub w1: Vec<f32>,    // [win*d, h]
    pub w2: Vec<f32>,    // [h, vocab]
}

fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 33) as f32 / u32::MAX as f32) * 0.1 - 0.05
}
fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0; rows * cols];
    for r in 0..rows { for c in 0..cols { t[c * rows + r] = a[r * cols + c]; } }
    t
}
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn silu_grad(x: f32) -> f32 { let s = 1.0 / (1.0 + (-x).exp()); s + x * s * (1.0 - s) }

impl ByteWinLm {
    pub fn init(vocab: usize, d: usize, win: usize, h: usize, seed: u64) -> Self {
        let mut s = seed;
        let g = |s: &mut u64, n: usize| (0..n).map(|_| lcg(s)).collect::<Vec<f32>>();
        ByteWinLm { vocab, d, win, h,
            embed: g(&mut s, vocab * d), w1: g(&mut s, win * d * h), w2: g(&mut s, h * vocab) }
    }

    /// Build concatenated-embedding input [n, win*d] from windows of bytes.
    fn gather(&self, windows: &[&[u8]]) -> Vec<f32> {
        let wd = self.win * self.d;
        let mut x = vec![0.0f32; windows.len() * wd];
        for (i, w) in windows.iter().enumerate() {
            for (p, &b) in w.iter().enumerate() {
                let src = (b as usize) * self.d;
                x[i * wd + p * self.d..i * wd + p * self.d + self.d]
                    .copy_from_slice(&self.embed[src..src + self.d]);
            }
        }
        x
    }

    fn forward(&self, ctx: &GpuContext, x: &[f32], n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let z = gemm::gemm(ctx, x, &self.w1, n, self.win * self.d, self.h); // [n,h]
        let a: Vec<f32> = z.iter().map(|&v| silu(v)).collect();
        let logits = gemm::gemm(ctx, &a, &self.w2, n, self.h, self.vocab);  // [n,vocab]
        (z, a, logits)
    }

    fn ce(&self, logits: &[f32], targets: &[u8], n: usize) -> f32 {
        let mut tot = 0.0;
        for i in 0..n {
            let row = &logits[i * self.vocab..(i + 1) * self.vocab];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|v| (v - mx).exp()).sum();
            tot += -(row[targets[i] as usize] - mx - sum.ln());
        }
        tot / n as f32
    }

    pub fn train_step(&mut self, ctx: &GpuContext, windows: &[&[u8]], targets: &[u8], lr: f32) -> f32 {
        let n = windows.len();
        let wd = self.win * self.d;
        let x = self.gather(windows);
        let (z, a, logits) = self.forward(ctx, &x, n);
        let loss = self.ce(&logits, targets, n);

        let mut dlogits = vec![0.0f32; n * self.vocab];
        for i in 0..n {
            let row = &logits[i * self.vocab..(i + 1) * self.vocab];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..self.vocab { dlogits[i * self.vocab + j] = exps[j] / sum / n as f32; }
            dlogits[i * self.vocab + targets[i] as usize] -= 1.0 / n as f32;
        }
        let at = transpose(&a, n, self.h);
        let dw2 = gemm::gemm(ctx, &at, &dlogits, self.h, n, self.vocab);
        let w2t = transpose(&self.w2, self.h, self.vocab);
        let da = gemm::gemm(ctx, &dlogits, &w2t, n, self.vocab, self.h);
        let dz: Vec<f32> = da.iter().zip(&z).map(|(&g, &zv)| g * silu_grad(zv)).collect();
        let xt = transpose(&x, n, wd);
        let dw1 = gemm::gemm(ctx, &xt, &dz, wd, n, self.h);          // [wd, h]
        let w1t = transpose(&self.w1, wd, self.h);
        let dx = gemm::gemm(ctx, &dz, &w1t, n, self.h, wd);          // [n, wd]

        for i in 0..self.w2.len() { self.w2[i] -= lr * dw2[i]; }
        for i in 0..self.w1.len() { self.w1[i] -= lr * dw1[i]; }
        for (i, w) in windows.iter().enumerate() {
            for (p, &b) in w.iter().enumerate() {
                let base = (b as usize) * self.d;
                let off = i * wd + p * self.d;
                for j in 0..self.d { self.embed[base + j] -= lr * dx[off + j]; }
            }
        }
        loss
    }

    pub fn eval_loss(&self, ctx: &GpuContext, windows: &[&[u8]], targets: &[u8]) -> f32 {
        let x = self.gather(windows);
        let (_, _, logits) = self.forward(ctx, &x, windows.len());
        self.ce(&logits, targets, windows.len())
    }

    /// Greedy/temperature byte generation from a seed window.
    pub fn generate(&self, ctx: &GpuContext, seed: &[u8], n: usize, temp: f32, rng: &mut u64) -> Vec<u8> {
        let mut ctxw: Vec<u8> = seed.to_vec();
        while ctxw.len() < self.win { ctxw.insert(0, b' '); }
        let mut out = Vec::new();
        for _ in 0..n {
            let w = &ctxw[ctxw.len() - self.win..];
            let x = self.gather(&[w]);
            let (_, _, logits) = self.forward(ctx, &x, 1);
            let row = &logits[..self.vocab];
            let b = if temp <= 0.0 {
                (0..self.vocab).max_by(|&a, &c| row[a].partial_cmp(&row[c]).unwrap()).unwrap() as u8
            } else {
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = row.iter().map(|v| ((v - mx) / temp).exp()).collect();
                let sum: f32 = exps.iter().sum();
                *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let r = ((*rng >> 33) as f32 / u32::MAX as f32) * sum;
                let mut acc = 0.0; let mut pick = 0;
                for (j, &e) in exps.iter().enumerate() { acc += e; if r <= acc { pick = j; break; } }
                pick as u8
            };
            out.push(b);
            ctxw.push(b);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn win_lm_overfits_code() {
        let ctx = match GpuContext::new() { Ok(c) => c, Err(_) => return };
        let code = b"function add(a, b) { return a + b; }";
        let win = 8;
        let mut wins: Vec<&[u8]> = Vec::new();
        let mut tgts: Vec<u8> = Vec::new();
        for i in win..code.len() {
            wins.push(&code[i - win..i]);
            tgts.push(code[i]);
        }
        let mut m = ByteWinLm::init(256, 24, win, 96, 11);
        let l0 = m.eval_loss(&ctx, &wins, &tgts);
        let mut l = l0;
        for _ in 0..300 { l = m.train_step(&ctx, &wins, &tgts, 0.5); }
        assert!(l < l0 * 0.4, "win-lm loss not dropping: {l0} -> {l}");
    }
}
