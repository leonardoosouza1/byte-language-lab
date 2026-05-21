//! Byte-level transformer FORWARD pass, composed from the wgpu kernels
//! (GEMM + softmax + RMSNorm + SiLU). This is the proof that the
//! size-parametric kernels assemble into a real transformer — the student
//! architecture from CODE-MODEL-STRATEGY.md (state in VRAM, no LDS ceiling).
//!
//! Decoder-only, pre-norm, single-head per layer (multi-head = head slicing,
//! deferred), learned positional embedding, SwiGLU FFN, RMSNorm, weight-tied
//! lm_head. Forward only — backward + training is the next phase.
//!
//! Each op round-trips host↔device (correctness demo, not speed). The point:
//! the GPU forward matches a pure-CPU reference within float tolerance, so
//! the kernel set is proven to build a transformer.

use crate::gpu::{gemm, nn_ops, GpuContext};

/// Tiny config — deliberately small for a fast parity test.
#[derive(Clone, Copy)]
pub struct Config {
    pub vocab: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq: usize,
}

impl Config {
    pub fn tiny() -> Self {
        Config { vocab: 256, d_model: 64, n_layers: 2, d_ff: 128, max_seq: 32 }
    }
}

/// Per-layer weights (row-major). Single-head attention: d_head = d_model.
pub struct Layer {
    pub wq: Vec<f32>,   // [d, d]
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub norm1: Vec<f32>, // [d]
    pub norm2: Vec<f32>,
    pub w1: Vec<f32>,   // [d, d_ff]
    pub w2: Vec<f32>,   // [d_ff, d]
}

pub struct Model {
    pub cfg: Config,
    pub embed: Vec<f32>,    // [vocab, d]
    pub pos: Vec<f32>,      // [max_seq, d]
    pub layers: Vec<Layer>,
    pub norm_f: Vec<f32>,   // [d]
    pub eps: f32,
}

fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 33) as f32 / u32::MAX as f32) * 0.1 - 0.05  // small init
}

impl Model {
    /// Deterministic random init — same weights drive CPU and GPU forward.
    pub fn init(cfg: Config, seed: u64) -> Self {
        let mut s = seed;
        let d = cfg.d_model;
        let rnd = |s: &mut u64, n: usize| (0..n).map(|_| lcg(s)).collect::<Vec<f32>>();
        let mut layers = Vec::new();
        for _ in 0..cfg.n_layers {
            layers.push(Layer {
                wq: rnd(&mut s, d * d), wk: rnd(&mut s, d * d),
                wv: rnd(&mut s, d * d), wo: rnd(&mut s, d * d),
                norm1: (0..d).map(|_| 1.0 + lcg(&mut s)).collect(),
                norm2: (0..d).map(|_| 1.0 + lcg(&mut s)).collect(),
                w1: rnd(&mut s, d * cfg.d_ff), w2: rnd(&mut s, cfg.d_ff * d),
            });
        }
        Model {
            cfg,
            embed: rnd(&mut s, cfg.vocab * d),
            pos: rnd(&mut s, cfg.max_seq * d),
            layers,
            norm_f: (0..d).map(|_| 1.0 + lcg(&mut s)).collect(),
            eps: 1e-6,
        }
    }

    fn embed_tokens(&self, tokens: &[u8]) -> Vec<f32> {
        let d = self.cfg.d_model;
        let mut x = vec![0.0f32; tokens.len() * d];
        for (i, &t) in tokens.iter().enumerate() {
            for j in 0..d {
                x[i * d + j] = self.embed[(t as usize) * d + j] + self.pos[i * d + j];
            }
        }
        x
    }

    // ---- GPU forward: composes the kernels ----
    pub fn forward_gpu(&self, ctx: &GpuContext, tokens: &[u8]) -> Vec<f32> {
        let d = self.cfg.d_model;
        let seq = tokens.len();
        let scale = 1.0 / (d as f32).sqrt();
        let mut x = self.embed_tokens(tokens);

        for l in &self.layers {
            let xn = nn_ops::rmsnorm(ctx, &x, &l.norm1, seq, d, self.eps);
            let mut q = gemm::gemm(ctx, &xn, &l.wq, seq, d, d);
            let k = gemm::gemm(ctx, &xn, &l.wk, seq, d, d);
            let v = gemm::gemm(ctx, &xn, &l.wv, seq, d, d);
            for q_i in q.iter_mut() { *q_i *= scale; }
            let kt = transpose(&k, seq, d);                      // [d, seq]
            let scores = gemm::gemm(ctx, &q, &kt, seq, d, seq);  // [seq, seq]
            let probs = nn_ops::causal_softmax(ctx, &scores, seq, seq);
            let attn = gemm::gemm(ctx, &probs, &v, seq, seq, d);  // [seq, d]
            let ao = gemm::gemm(ctx, &attn, &l.wo, seq, d, d);
            for i in 0..x.len() { x[i] += ao[i]; }               // residual

            let xn2 = nn_ops::rmsnorm(ctx, &x, &l.norm2, seq, d, self.eps);
            let h = gemm::gemm(ctx, &xn2, &l.w1, seq, d, self.cfg.d_ff);
            let h = nn_ops::silu(ctx, &h);
            let ffn = gemm::gemm(ctx, &h, &l.w2, seq, self.cfg.d_ff, d);
            for i in 0..x.len() { x[i] += ffn[i]; }              // residual
        }

        let xf = nn_ops::rmsnorm(ctx, &x, &self.norm_f, seq, d, self.eps);
        // weight-tied lm_head: logits = xf @ embed^T  →  [seq, vocab]
        let et = transpose(&self.embed, self.cfg.vocab, d);      // [d, vocab]
        gemm::gemm(ctx, &xf, &et, seq, d, self.cfg.vocab)
    }

    // ---- CPU reference forward (same math) ----
    pub fn forward_cpu(&self, tokens: &[u8]) -> Vec<f32> {
        let d = self.cfg.d_model;
        let seq = tokens.len();
        let scale = 1.0 / (d as f32).sqrt();
        let mut x = self.embed_tokens(tokens);

        for l in &self.layers {
            let xn = rmsnorm_cpu(&x, &l.norm1, seq, d, self.eps);
            let mut q = gemm::gemm_cpu(&xn, &l.wq, seq, d, d);
            let k = gemm::gemm_cpu(&xn, &l.wk, seq, d, d);
            let v = gemm::gemm_cpu(&xn, &l.wv, seq, d, d);
            for q_i in q.iter_mut() { *q_i *= scale; }
            let kt = transpose(&k, seq, d);
            let scores = gemm::gemm_cpu(&q, &kt, seq, d, seq);
            let probs = causal_softmax_cpu(&scores, seq, seq);
            let attn = gemm::gemm_cpu(&probs, &v, seq, seq, d);
            let ao = gemm::gemm_cpu(&attn, &l.wo, seq, d, d);
            for i in 0..x.len() { x[i] += ao[i]; }

            let xn2 = rmsnorm_cpu(&x, &l.norm2, seq, d, self.eps);
            let mut h = gemm::gemm_cpu(&xn2, &l.w1, seq, d, self.cfg.d_ff);
            for hv in h.iter_mut() { *hv = *hv / (1.0 + (-*hv).exp()); }
            let ffn = gemm::gemm_cpu(&h, &l.w2, seq, self.cfg.d_ff, d);
            for i in 0..x.len() { x[i] += ffn[i]; }
        }
        let xf = rmsnorm_cpu(&x, &self.norm_f, seq, d, self.eps);
        let et = transpose(&self.embed, self.cfg.vocab, d);
        gemm::gemm_cpu(&xf, &et, seq, d, self.cfg.vocab)
    }
}

fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = a[r * cols + c];
        }
    }
    t
}

fn rmsnorm_cpu(x: &[f32], w: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let mut o = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for c in 0..cols { o[r * cols + c] = row[c] * inv * w[c]; }
    }
    o
}

fn causal_softmax_cpu(s: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let lim = r + 1;
        let row = &s[r * cols..r * cols + lim];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for c in 0..lim { o[r * cols + c] = exps[c] / sum; }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformer_forward_parity() {
        let ctx = match GpuContext::new() { Ok(c) => c, Err(_) => return };
        let cfg = Config::tiny();
        let model = Model::init(cfg, 2024);
        let tokens: Vec<u8> = b"fn main() { let x = 42; }".to_vec();
        let tokens = &tokens[..tokens.len().min(cfg.max_seq)];

        let gpu = model.forward_gpu(&ctx, tokens);
        let cpu = model.forward_cpu(tokens);
        assert_eq!(gpu.len(), tokens.len() * cfg.vocab);

        let max_err = gpu.iter().zip(&cpu).map(|(g, c)| (g - c).abs()).fold(0.0f32, f32::max);
        // many chained GEMMs accumulate float drift — generous but tight bound
        assert!(max_err < 1e-2, "forward max_err {max_err}");

        // argmax of last position must agree (the prediction itself)
        let last = (tokens.len() - 1) * cfg.vocab;
        let amax = |v: &[f32]| (0..cfg.vocab).max_by(|&a, &b|
            v[last + a].partial_cmp(&v[last + b]).unwrap()).unwrap();
        assert_eq!(amax(&gpu), amax(&cpu), "argmax disagree");
    }
}
