//! iara_liquid.rs — Liquid Time-Constant byte model
//! (Bit-Aware LTC, ~10k params, treina em CPU em segundos)
//!
//! Arquitetura aproveita a tese:
//!   - Não usa softmax 256-way completo
//!   - Predição FATORADA por papel do bit:
//!       bit_7,6 = fixed/ASCII boundary (1 head simples)
//!       bit_5   = case (KEY bit, validado KL=0.003) (head dedicado)
//!       bits 0-4 = ident (5 bits = 32 classes, softmax pequeno)
//!   - LTC dynamics em vez de RNN/transformer:
//!       x_{t+1} = exp(-1/tau) * x_t + (1-exp(-1/tau)) * tanh(W x + V u + b)
//!     com tau aprendido por neurônio (memória contínua por dim)
//!
//! Não é GRU, não é transformer, não é Mamba — é LTC (Hasani et al MIT 2020).
//! ~10k params, treina em ~1min CPU.

use std::fs;
use std::io::Write;
use std::time::Instant;

const D: usize = 96;       // hidden state (3x capacity)
const E: usize = 16;       // input embed
const WIN: usize = 96;     // bptt window (longer context)
const BATCH: usize = 32;
const LR: f32 = 0.004;
const EPOCHS: usize = 150;
const STEPS_PER_EP: usize = 200;
const SEED: u64 = 7;

// ── Bit decomposition ────────────────────────────────────────────────────
#[inline] fn bit7(b: u8) -> u8 { (b >> 7) & 1 }
#[inline] fn bit6(b: u8) -> u8 { (b >> 6) & 1 }
#[inline] fn bit5(b: u8) -> u8 { (b >> 5) & 1 }
#[inline] fn ident5(b: u8) -> u8 { b & 0x1F }      // bits 0-4
#[inline] fn reassemble(b7: u8, b6: u8, b5: u8, id: u8) -> u8 {
    (b7 << 7) | (b6 << 6) | (b5 << 5) | (id & 0x1F)
}

// ── RNG ──────────────────────────────────────────────────────────────────
struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Self(s.max(1)) }
    fn step(&mut self) -> u32 {
        self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn unit(&mut self) -> f32 { self.step() as f32 / u32::MAX as f32 }
    fn signed(&mut self) -> f32 { 2.0 * self.unit() - 1.0 }
    fn idx(&mut self, n: usize) -> usize { (self.step() as usize) % n.max(1) }
}

// ── Matriz row-major ─────────────────────────────────────────────────────
#[inline]
fn matvec(out: &mut [f32], w: &[f32], x: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &w[r*cols..(r+1)*cols];
        let mut acc = 0.0f32;
        for c in 0..cols { acc += row[c] * x[c]; }
        out[r] = acc;
    }
}
#[inline] fn sig(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// ── Modelo: Bit-Aware LTC ─────────────────────────────────────────────────
struct Model {
    // Input embeddings per bit semântica
    // bit7 = 2 → E_b7=4 ; bit6 = 2 → E_b6=4 ; bit5 = 2 → E_b5=4 ; ident = 32 → E_id=4
    emb_b7: Vec<f32>,    // [2, 4]
    emb_b6: Vec<f32>,    // [2, 4]
    emb_b5: Vec<f32>,    // [2, 4]
    emb_id: Vec<f32>,    // [32, E_id=4]
    // Concat input = 4+4+4+4 = 16 = E

    // LTC recurrent + input + bias
    w_h: Vec<f32>,    // [D, D]
    v_in: Vec<f32>,   // [D, E]
    b_h: Vec<f32>,    // [D]
    tau_log: Vec<f32>, // [D] — tau = 1 + exp(tau_log), alpha = exp(-1/tau)

    // Output heads (factored prediction)
    head_b7: Vec<f32>, head_b7_bias: f32,    // [D] → 1 sigmoid
    head_b6: Vec<f32>, head_b6_bias: f32,
    head_b5: Vec<f32>, head_b5_bias: f32,
    head_id: Vec<f32>, head_id_bias: Vec<f32>, // [32, D] → 32 softmax
}

const EID: usize = 4;
const E_B: usize = 4;
const E_TOTAL: usize = E_B * 3 + EID;  // 16

impl Model {
    fn new(rng: &mut Rng) -> Self {
        let scale = (1.0 / (D as f32).sqrt()) * 1.2;
        let scale_e = 1.0;
        let mut emb_b7 = vec![0.0; 2*E_B];   for v in &mut emb_b7 { *v = rng.signed() * scale_e; }
        let mut emb_b6 = vec![0.0; 2*E_B];   for v in &mut emb_b6 { *v = rng.signed() * scale_e; }
        let mut emb_b5 = vec![0.0; 2*E_B];   for v in &mut emb_b5 { *v = rng.signed() * scale_e; }
        let mut emb_id = vec![0.0; 32*EID];  for v in &mut emb_id { *v = rng.signed() * scale_e; }

        let mut w_h = vec![0.0; D*D];        for v in &mut w_h { *v = rng.signed() * scale; }
        let mut v_in = vec![0.0; D*E_TOTAL]; for v in &mut v_in { *v = rng.signed() * scale; }
        let b_h = vec![0.0; D];
        // tau_log init: tau = 1+e^(-1) ≈ 1.37 → alpha = exp(-1/1.37) ≈ 0.48
        // Mix de neurons leaky e sticky para diversidade temporal
        let mut tau_log = vec![0.0f32; D];
        for d in 0..D {
            // half leaky (tau~1.37 α=0.48), half medium (tau~3.7 α=0.76)
            tau_log[d] = if d < D / 2 { -1.0 } else { 1.0 };
        }

        let mut head_b7 = vec![0.0; D]; for v in &mut head_b7 { *v = rng.signed() * scale; }
        let mut head_b6 = vec![0.0; D]; for v in &mut head_b6 { *v = rng.signed() * scale; }
        let mut head_b5 = vec![0.0; D]; for v in &mut head_b5 { *v = rng.signed() * scale; }
        let mut head_id = vec![0.0; 32*D]; for v in &mut head_id { *v = rng.signed() * scale; }
        let head_id_bias = vec![0.0; 32];

        Model {
            emb_b7, emb_b6, emb_b5, emb_id,
            w_h, v_in, b_h, tau_log,
            head_b7, head_b7_bias: 0.0,
            head_b6, head_b6_bias: 0.0,
            head_b5, head_b5_bias: 0.0,
            head_id, head_id_bias,
        }
    }

    fn n_params(&self) -> usize {
        self.emb_b7.len() + self.emb_b6.len() + self.emb_b5.len() + self.emb_id.len()
            + self.w_h.len() + self.v_in.len() + self.b_h.len() + self.tau_log.len()
            + self.head_b7.len() + 1 + self.head_b6.len() + 1 + self.head_b5.len() + 1
            + self.head_id.len() + self.head_id_bias.len()
    }

    /// Build input vector u (size E_TOTAL) from byte
    fn embed(&self, b: u8) -> Vec<f32> {
        let mut u = vec![0.0f32; E_TOTAL];
        let b7 = bit7(b) as usize;
        let b6 = bit6(b) as usize;
        let b5 = bit5(b) as usize;
        let id = ident5(b) as usize;
        for i in 0..E_B {
            u[i]           = self.emb_b7[b7*E_B + i];
            u[E_B + i]     = self.emb_b6[b6*E_B + i];
            u[E_B*2 + i]   = self.emb_b5[b5*E_B + i];
        }
        for i in 0..EID {
            u[E_B*3 + i] = self.emb_id[id*EID + i];
        }
        u
    }

    fn alpha(&self, d: usize) -> f32 {
        let tau = 1.0 + self.tau_log[d].exp();
        (-1.0 / tau).exp()
    }

    /// Forward de toda janela; retorna estados + outputs por step
    fn forward(&self, win: &[u8]) -> (Vec<Vec<f32>>, Vec<(f32, f32, f32, Vec<f32>)>) {
        let t = win.len();
        let mut states = vec![vec![0.0f32; D]; t + 1];
        let mut outs   = Vec::with_capacity(t);

        for ti in 0..t {
            let u = self.embed(win[ti]);
            let mut pre_v = vec![0.0f32; D];
            let mut pre_w = vec![0.0f32; D];
            matvec(&mut pre_v, &self.v_in, &u, D, E_TOTAL);
            matvec(&mut pre_w, &self.w_h,   &states[ti], D, D);
            let mut s_next = vec![0.0f32; D];
            for d in 0..D {
                let pre = (pre_v[d] + pre_w[d] + self.b_h[d]).tanh();
                let a = self.alpha(d);
                s_next[d] = a * states[ti][d] + (1.0 - a) * pre;
            }
            states[ti+1] = s_next.clone();

            // Outputs: 3 sigmoids + 32-way softmax
            let mut logit_b7 = self.head_b7_bias;
            let mut logit_b6 = self.head_b6_bias;
            let mut logit_b5 = self.head_b5_bias;
            for d in 0..D {
                logit_b7 += self.head_b7[d] * s_next[d];
                logit_b6 += self.head_b6[d] * s_next[d];
                logit_b5 += self.head_b5[d] * s_next[d];
            }
            let p_b7 = sig(logit_b7);
            let p_b6 = sig(logit_b6);
            let p_b5 = sig(logit_b5);

            let mut id_logits = vec![0.0f32; 32];
            for k in 0..32 {
                let mut acc = self.head_id_bias[k];
                let row = &self.head_id[k*D..(k+1)*D];
                for d in 0..D { acc += row[d] * s_next[d]; }
                id_logits[k] = acc;
            }
            outs.push((p_b7, p_b6, p_b5, id_logits));
        }
        (states, outs)
    }

    /// BPTT train step on batch.
    /// Each sample: window of WIN bytes; targets = win[1..] (predict next byte).
    /// Loss = BCE(b7) + BCE(b6) + BCE(b5) + CE(id, 32-way)
    fn train_step(&mut self, windows: &[Vec<u8>], lr: f32) -> f32 {
        let batch = windows.len();
        let mut total_loss = 0.0;

        // Accumulated gradients
        let mut g_emb_b7 = vec![0.0f32; self.emb_b7.len()];
        let mut g_emb_b6 = vec![0.0f32; self.emb_b6.len()];
        let mut g_emb_b5 = vec![0.0f32; self.emb_b5.len()];
        let mut g_emb_id = vec![0.0f32; self.emb_id.len()];
        let mut g_w_h    = vec![0.0f32; self.w_h.len()];
        let mut g_v_in   = vec![0.0f32; self.v_in.len()];
        let mut g_b_h    = vec![0.0f32; self.b_h.len()];
        let mut g_tau_log= vec![0.0f32; self.tau_log.len()];
        let mut g_head_b7= vec![0.0f32; self.head_b7.len()];
        let mut g_head_b7_b = 0.0f32;
        let mut g_head_b6= vec![0.0f32; self.head_b6.len()];
        let mut g_head_b6_b = 0.0f32;
        let mut g_head_b5= vec![0.0f32; self.head_b5.len()];
        let mut g_head_b5_b = 0.0f32;
        let mut g_head_id = vec![0.0f32; self.head_id.len()];
        let mut g_head_id_b = vec![0.0f32; 32];

        for win in windows {
            let t = win.len();
            if t < 2 { continue; }
            let (states, outs) = self.forward(&win[..t-1]);
            // Targets: bytes win[1..]
            // backward: d_state for each step accumulated reverse
            let mut d_state_next = vec![0.0f32; D];

            for step in (0..t-1).rev() {
                let target = win[step+1];
                let tgt_b7 = bit7(target) as f32;
                let tgt_b6 = bit6(target) as f32;
                let tgt_b5 = bit5(target) as f32;
                let tgt_id = ident5(target) as usize;

                let (p_b7, p_b6, p_b5, id_logits) = &outs[step];
                let s_next = &states[step+1];
                let s_prev = &states[step];

                // BCE loss + grads
                let eps = 1e-7f32;
                let pb7 = p_b7.clamp(eps, 1.0-eps);
                let pb6 = p_b6.clamp(eps, 1.0-eps);
                let pb5 = p_b5.clamp(eps, 1.0-eps);
                total_loss += -(tgt_b7 * pb7.ln() + (1.0-tgt_b7) * (1.0-pb7).ln());
                total_loss += -(tgt_b6 * pb6.ln() + (1.0-tgt_b6) * (1.0-pb6).ln());
                total_loss += -(tgt_b5 * pb5.ln() + (1.0-tgt_b5) * (1.0-pb5).ln());

                let d_b7 = p_b7 - tgt_b7;
                let d_b6 = p_b6 - tgt_b6;
                let d_b5 = p_b5 - tgt_b5;

                // Softmax over id_logits + CE
                let mut mx = id_logits[0];
                for &v in id_logits.iter() { if v > mx { mx = v; } }
                let exps: Vec<f32> = id_logits.iter().map(|v| (v - mx).exp()).collect();
                let sum: f32 = exps.iter().sum();
                let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
                total_loss += -probs[tgt_id].max(eps).ln();

                let mut d_id_logit = probs.clone();
                d_id_logit[tgt_id] -= 1.0;

                // Heads grads
                g_head_b7_b += d_b7;
                g_head_b6_b += d_b6;
                g_head_b5_b += d_b5;
                for d in 0..D {
                    g_head_b7[d] += d_b7 * s_next[d];
                    g_head_b6[d] += d_b6 * s_next[d];
                    g_head_b5[d] += d_b5 * s_next[d];
                }
                for k in 0..32 {
                    g_head_id_b[k] += d_id_logit[k];
                    let g_row = &mut g_head_id[k*D..(k+1)*D];
                    for d in 0..D { g_row[d] += d_id_logit[k] * s_next[d]; }
                }

                // d_state from heads (gradient back to s_next)
                let mut d_s = d_state_next.clone();
                for d in 0..D {
                    d_s[d] += d_b7 * self.head_b7[d];
                    d_s[d] += d_b6 * self.head_b6[d];
                    d_s[d] += d_b5 * self.head_b5[d];
                    let mut acc = 0.0f32;
                    for k in 0..32 { acc += d_id_logit[k] * self.head_id[k*D + d]; }
                    d_s[d] += acc;
                }

                // Through LTC update
                // s_next = a * s_prev + (1-a) * pre
                // d_pre = d_s * (1-a) * (1 - pre^2)   (using tanh)
                // d_s_prev (via state path) += d_s * a
                // d_alpha = d_s * (s_prev - pre)
                let mut d_pre = vec![0.0f32; D];
                let mut d_s_prev_via_a = vec![0.0f32; D];
                for d in 0..D {
                    let a = self.alpha(d);
                    let pre = if a < 1.0 { (s_next[d] - a * s_prev[d]) / (1.0 - a) } else { 0.0 };
                    // d L / d pre  = d_s * (1 - a) * (1 - pre^2)
                    d_pre[d] = d_s[d] * (1.0 - a) * (1.0 - pre*pre);
                    d_s_prev_via_a[d] = d_s[d] * a;

                    // d_alpha = d_s * (s_prev - pre)
                    let d_alpha = d_s[d] * (s_prev[d] - pre);
                    // alpha = exp(-1/tau), tau = 1 + exp(tau_log)
                    // d_alpha / d_tau_log:
                    //   d_alpha / d_tau = alpha / tau^2
                    //   d_tau / d_tau_log = exp(tau_log) = tau - 1
                    let tau = 1.0 + self.tau_log[d].exp();
                    let d_alpha_d_tau = a / (tau * tau);
                    let d_tau_d_log = tau - 1.0;
                    g_tau_log[d] += d_alpha * d_alpha_d_tau * d_tau_d_log;
                }

                // d_pre → d_b_h, d_w_h, d_v_in, d_state[step]
                let u = self.embed(win[step]);
                for d in 0..D {
                    g_b_h[d] += d_pre[d];
                    let g_row_w = &mut g_w_h[d*D..(d+1)*D];
                    for c in 0..D { g_row_w[c] += d_pre[d] * s_prev[c]; }
                    let g_row_v = &mut g_v_in[d*E_TOTAL..(d+1)*E_TOTAL];
                    for c in 0..E_TOTAL { g_row_v[c] += d_pre[d] * u[c]; }
                }

                // d_state[step] via w_h^T @ d_pre + via_a
                let mut d_s_prev = d_s_prev_via_a;
                for d in 0..D {
                    let dp = d_pre[d];
                    let w_row = &self.w_h[d*D..(d+1)*D];
                    for c in 0..D { d_s_prev[c] += dp * w_row[c]; }
                }

                // d_u via v_in^T @ d_pre — propagar pra embeddings
                let mut d_u = vec![0.0f32; E_TOTAL];
                for d in 0..D {
                    let dp = d_pre[d];
                    let v_row = &self.v_in[d*E_TOTAL..(d+1)*E_TOTAL];
                    for c in 0..E_TOTAL { d_u[c] += dp * v_row[c]; }
                }
                // Split d_u entre os 4 segmentos do embedding
                let inp_byte = win[step];
                let b7i = bit7(inp_byte) as usize;
                let b6i = bit6(inp_byte) as usize;
                let b5i = bit5(inp_byte) as usize;
                let idi = ident5(inp_byte) as usize;
                for i in 0..E_B {
                    g_emb_b7[b7i*E_B + i] += d_u[i];
                    g_emb_b6[b6i*E_B + i] += d_u[E_B + i];
                    g_emb_b5[b5i*E_B + i] += d_u[E_B*2 + i];
                }
                for i in 0..EID {
                    g_emb_id[idi*EID + i] += d_u[E_B*3 + i];
                }

                d_state_next = d_s_prev;
            }
        }

        // Adam-light: scale + clip
        let scale = 1.0 / batch as f32;
        let clip = 1.0f32;
        let apply = |w: &mut [f32], g: &mut [f32]| {
            let mut norm = 0.0f32;
            for &v in g.iter() { norm += v*v; }
            norm = (norm * scale * scale).sqrt();
            let factor = if norm > clip { clip / norm } else { 1.0 };
            for (p, gi) in w.iter_mut().zip(g.iter_mut()) {
                *p -= lr * (*gi * scale * factor);
            }
        };
        let apply_s = |p: &mut f32, g: f32| {
            let gg = g * scale;
            let gc = gg.clamp(-clip, clip);
            *p -= lr * gc;
        };

        apply(&mut self.emb_b7, &mut g_emb_b7);
        apply(&mut self.emb_b6, &mut g_emb_b6);
        apply(&mut self.emb_b5, &mut g_emb_b5);
        apply(&mut self.emb_id, &mut g_emb_id);
        apply(&mut self.w_h, &mut g_w_h);
        apply(&mut self.v_in, &mut g_v_in);
        apply(&mut self.b_h, &mut g_b_h);
        apply(&mut self.tau_log, &mut g_tau_log);
        apply(&mut self.head_b7, &mut g_head_b7);
        apply_s(&mut self.head_b7_bias, g_head_b7_b);
        apply(&mut self.head_b6, &mut g_head_b6);
        apply_s(&mut self.head_b6_bias, g_head_b6_b);
        apply(&mut self.head_b5, &mut g_head_b5);
        apply_s(&mut self.head_b5_bias, g_head_b5_b);
        apply(&mut self.head_id, &mut g_head_id);
        apply(&mut self.head_id_bias, &mut g_head_id_b);

        total_loss / batch as f32
    }

    /// Generate N bytes after prompt
    fn generate(&self, prompt: &[u8], n: usize, temp: f32, rng: &mut Rng) -> Vec<u8> {
        // Warm-up: 2 passadas pelo prompt para memorizar contexto
        let mut s = vec![0.0f32; D];
        for _ in 0..2 {
            for &b in prompt {
                let u = self.embed(b);
                let mut pre_v = vec![0.0f32; D];
                let mut pre_w = vec![0.0f32; D];
                matvec(&mut pre_v, &self.v_in, &u, D, E_TOTAL);
                matvec(&mut pre_w, &self.w_h, &s, D, D);
                let mut s_new = vec![0.0f32; D];
                for d in 0..D {
                    let pre = (pre_v[d] + pre_w[d] + self.b_h[d]).tanh();
                    let a = self.alpha(d);
                    s_new[d] = a * s[d] + (1.0 - a) * pre;
                }
                s = s_new;
            }
        }

        let mut out = Vec::with_capacity(n);
        let temp = temp.max(0.01);
        for _ in 0..n {
            // Predict next byte
            let mut logit_b7 = self.head_b7_bias;
            let mut logit_b6 = self.head_b6_bias;
            let mut logit_b5 = self.head_b5_bias;
            for d in 0..D {
                logit_b7 += self.head_b7[d] * s[d];
                logit_b6 += self.head_b6[d] * s[d];
                logit_b5 += self.head_b5[d] * s[d];
            }
            let p_b7 = sig(logit_b7 / temp);
            let p_b6 = sig(logit_b6 / temp);
            let p_b5 = sig(logit_b5 / temp);

            let mut id_logits = vec![0.0f32; 32];
            for k in 0..32 {
                let mut acc = self.head_id_bias[k];
                let row = &self.head_id[k*D..(k+1)*D];
                for d in 0..D { acc += row[d] * s[d]; }
                id_logits[k] = acc / temp;
            }
            let mx = id_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = id_logits.iter().map(|v| (v - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

            let b7s: u8 = if rng.unit() < p_b7 { 1 } else { 0 };
            let b6s: u8 = if rng.unit() < p_b6 { 1 } else { 0 };
            let b5s: u8 = if rng.unit() < p_b5 { 1 } else { 0 };
            // sample ident
            let r = rng.unit();
            let mut cum = 0.0f32;
            let mut id_pick: u8 = 0;
            for (k, &p) in probs.iter().enumerate() {
                cum += p;
                if r < cum { id_pick = k as u8; break; }
            }
            let byte = reassemble(b7s, b6s, b5s, id_pick);
            out.push(byte);
            // Feed back
            let u = self.embed(byte);
            let mut pre_v = vec![0.0f32; D];
            let mut pre_w = vec![0.0f32; D];
            matvec(&mut pre_v, &self.v_in, &u, D, E_TOTAL);
            matvec(&mut pre_w, &self.w_h, &s, D, D);
            let mut s_new = vec![0.0f32; D];
            for d in 0..D {
                let pre = (pre_v[d] + pre_w[d] + self.b_h[d]).tanh();
                let a = self.alpha(d);
                s_new[d] = a * s[d] + (1.0 - a) * pre;
            }
            s = s_new;
        }
        out
    }
}

fn main() -> std::io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA Liquid — Bit-Aware LTC byte model                     ║");
    println!("║   Closed-form Continuous-time, factored output, pure Rust    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load corpus
    let mut text = String::new();
    for p in &["data/dialogues_augmented.txt", "data/qwen_sustained.txt", "data/online_learning.txt"] {
        if std::path::Path::new(p).exists() {
            text.push_str(&fs::read_to_string(p)?);
            text.push('\n');
        }
    }
    let data = text.as_bytes();
    println!("  Corpus: {} bytes", data.len());

    let mut rng = Rng::new(SEED);
    let mut model = Model::new(&mut rng);
    println!("  Modelo: {} params", model.n_params());
    println!("  D={} E_total={} WIN={} LR={} EPOCHS={} BATCH={}\n",
             D, E_TOTAL, WIN, LR, EPOCHS, BATCH);

    let n_train = data.len() - WIN - 1;
    println!("  Pairs avaliáveis: {}\n", n_train);

    println!("  epoch  train_loss   tempo");
    println!("  -----  ----------   -----");
    let t0 = Instant::now();
    for epoch in 0..EPOCHS {
        let t_ep = Instant::now();
        let warm = (EPOCHS / 10).max(1);
        let lr = if epoch < warm {
            LR * (epoch + 1) as f32 / warm as f32
        } else {
            let p = (epoch - warm) as f32 / (EPOCHS - warm).max(1) as f32;
            LR * 0.5 * (1.0 + (std::f32::consts::PI * p).cos()).max(0.1)
        };

        let mut sum_loss = 0.0;
        for _ in 0..STEPS_PER_EP {
            let mut wins = Vec::with_capacity(BATCH);
            for _ in 0..BATCH {
                let start = rng.idx(n_train);
                let win: Vec<u8> = data[start..start+WIN].to_vec();
                wins.push(win);
            }
            sum_loss += model.train_step(&wins, lr);
        }
        let avg = sum_loss / STEPS_PER_EP as f32;
        println!("  {:>5}    {:>8.4}    {:>5.1}s", epoch, avg, t_ep.elapsed().as_secs_f32());
        std::io::stdout().flush().ok();
    }
    println!("\n  Wall: {:.0}s", t0.elapsed().as_secs_f32());

    // Generation demo
    let prompts: &[&str] = &[
        "USER: oi\nMODEL: ",
        "USER: o que e o Brasil\nMODEL: ",
        "USER: quem foi Pele\nMODEL: ",
        "USER: feijoada\nMODEL: ",
    ];
    println!("\n▸ Geração demo:\n");
    for &p in prompts {
        let prompt_b = p.as_bytes();
        // Hash prompt para seed única
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in prompt_b { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
        for &temp in &[0.5f32, 0.8] {
            let mut g_rng = Rng::new(h ^ ((temp * 1000.0) as u64) ^ 0xbeef);
            let gen_out = model.generate(prompt_b, 80, temp, &mut g_rng);
            let s = String::from_utf8_lossy(&gen_out);
            println!("  [t={:.1}] {:?}", temp, p);
            println!("            → {:?}", s);
        }
    }

    Ok(())
}
