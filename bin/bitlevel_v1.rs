//! bitlevel_v1.rs — Modelo bit-level autocontido em CPU Rust puro
//!
//! Arquitetura V1: bits independentes + sum-pool intra-byte
//!   - Input: byte → 8 bits paralelos
//!   - Embedding: (bit_position, bit_value) → R^D  [8 × 2 × D]
//!   - intra_byte_t = sum_i embed[i, bit_i]
//!   - RNN: state_t = tanh(W_h @ state_{t-1} + W_x @ intra_byte_t + b_h)
//!   - 8 heads binárias: P(bit_i = 1 | state_t) = sigmoid(head_i @ state_t + b_i)
//!   - Loss = avg BCE over 8 bits
//!
//! Treina em data/dialogues_pt.txt e reporta val_loss + amostra de geração.

use std::fs;
use std::io::Write;
use std::time::Instant;

// ── Hiperparâmetros ────────────────────────────────────────────────────────
const W:        usize = 128;   // janela em bytes
const D:        usize = 128;   // dim do estado oculto
const LR:       f32   = 0.01;
const EPOCHS:   usize = 30;
const STEPS:    usize = 200;
const BATCH:    usize = 16;    // CPU-friendly
const SEED:     u64   = 7;
const CLIP:     f32   = 5.0;   // gradient clip
const DATA:     &str  = "/home/leonardo/projects/LLM/byte-language-lab/data/dialogues_pt.txt";

// ── RNG simples (LCG xorshift) ─────────────────────────────────────────────
struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Self(s) }
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }
}

// ── Matriz row-major ───────────────────────────────────────────────────────
#[inline]
fn matvec(out: &mut [f32], w: &[f32], x: &[f32], rows: usize, cols: usize) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    for r in 0..rows {
        let mut acc = 0.0f32;
        let row = &w[r * cols..(r + 1) * cols];
        for c in 0..cols { acc += row[c] * x[c]; }
        out[r] = acc;
    }
}

#[inline] fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
#[inline] fn sigmoid_grad_from_y(y: f32) -> f32 { y * (1.0 - y) }
#[inline] fn tanh_grad_from_y(y: f32) -> f32 { 1.0 - y * y }

// ── Modelo ─────────────────────────────────────────────────────────────────
struct Model {
    embed: Vec<f32>,        // [8, 2, D]    flat
    w_h:   Vec<f32>,        // [D, D]
    w_x:   Vec<f32>,        // [D, D]
    b_h:   Vec<f32>,        // [D]
    head:  Vec<f32>,        // [8, D]       8 heads
    head_b:Vec<f32>,        // [8]
}

impl Model {
    fn new(rng: &mut Rng) -> Self {
        let init_emb = (1.0 / (D as f32).sqrt()) * 0.5;
        let init_w   = (1.0 / (D as f32).sqrt()) * 0.5;
        let mut m = Model {
            embed:  vec![0.0; 8 * 2 * D],
            w_h:    vec![0.0; D * D],
            w_x:    vec![0.0; D * D],
            b_h:    vec![0.0; D],
            head:   vec![0.0; 8 * D],
            head_b: vec![0.0; 8],
        };
        for v in &mut m.embed { *v = rng.next_f32() * init_emb; }
        for v in &mut m.w_h   { *v = rng.next_f32() * init_w; }
        for v in &mut m.w_x   { *v = rng.next_f32() * init_w; }
        for v in &mut m.head  { *v = rng.next_f32() * init_w; }
        m
    }

    /// Forward + BPTT. Atualiza pesos in-place. Retorna loss média do batch.
    fn step_train(&mut self, windows: &[Vec<u8>], targets: &[u8], lr: f32) -> f32 {
        let batch = windows.len();
        let mut total_loss = 0.0f32;

        // Gradientes acumulados
        let mut g_embed = vec![0.0f32; self.embed.len()];
        let mut g_w_h   = vec![0.0f32; self.w_h.len()];
        let mut g_w_x   = vec![0.0f32; self.w_x.len()];
        let mut g_b_h   = vec![0.0f32; self.b_h.len()];
        let mut g_head  = vec![0.0f32; self.head.len()];
        let mut g_head_b= vec![0.0f32; self.head_b.len()];

        for bi in 0..batch {
            let window = &windows[bi];
            let target = targets[bi];

            let t = window.len();
            // Estados salvos por timestep
            let mut states = vec![vec![0.0f32; D]; t + 1];     // states[0..=t]
            let mut intra  = vec![vec![0.0f32; D]; t];         // input agregado por byte

            // ── Forward ─────────────────────────────────────────
            for ti in 0..t {
                let byte = window[ti];
                // intra-byte = sum_i embed[i, bit_i]
                let intra_t = &mut intra[ti];
                for d in 0..D { intra_t[d] = 0.0; }
                for i in 0..8 {
                    let bit = ((byte >> (7 - i)) & 1) as usize;
                    let off = (i * 2 + bit) * D;
                    for d in 0..D { intra_t[d] += self.embed[off + d]; }
                }

                // pre = W_h @ s_prev + W_x @ intra + b_h
                let mut pre_h = vec![0.0f32; D];
                let mut pre_x = vec![0.0f32; D];
                matvec(&mut pre_h, &self.w_h, &states[ti],   D, D);
                matvec(&mut pre_x, &self.w_x, intra_t,        D, D);

                let s_next = &mut states[ti + 1];
                for d in 0..D { s_next[d] = (pre_h[d] + pre_x[d] + self.b_h[d]).tanh(); }
            }

            let final_state = states[t].clone();

            // ── Heads (8 bits independentes do target) ────────────
            let mut probs = [0.0f32; 8];
            let mut d_pre_heads = [0.0f32; 8];
            for i in 0..8 {
                let mut logit = self.head_b[i];
                let head_row = &self.head[i * D..(i + 1) * D];
                for d in 0..D { logit += head_row[d] * final_state[d]; }
                probs[i] = sigmoid(logit);
                let tgt = ((target >> (7 - i)) & 1) as f32;
                // BCE: -[t*log(p) + (1-t)*log(1-p)]
                let eps = 1e-7f32;
                let p = probs[i].clamp(eps, 1.0 - eps);
                total_loss += -(tgt * p.ln() + (1.0 - tgt) * (1.0 - p).ln()) / 8.0;
                // d/d_logit BCE = p - t
                d_pre_heads[i] = probs[i] - tgt;
                // Acumula grad head bias e head row
                g_head_b[i] += d_pre_heads[i];
                let g_head_row = &mut g_head[i * D..(i + 1) * D];
                for d in 0..D { g_head_row[d] += d_pre_heads[i] * final_state[d]; }
            }

            // d_state_t = sum_i d_logit_i * head_i
            let mut d_state = vec![0.0f32; D];
            for i in 0..8 {
                let head_row = &self.head[i * D..(i + 1) * D];
                for d in 0..D { d_state[d] += d_pre_heads[i] * head_row[d]; }
            }

            // ── Backward through time ─────────────────────────────
            // states[ti+1] = tanh(W_h @ states[ti] + W_x @ intra[ti] + b_h)
            for ti in (0..t).rev() {
                let s_next = &states[ti + 1];
                // d_pre = d_state * tanh_grad(s_next)
                let mut d_pre = vec![0.0f32; D];
                for d in 0..D { d_pre[d] = d_state[d] * tanh_grad_from_y(s_next[d]); }

                // grad b_h
                for d in 0..D { g_b_h[d] += d_pre[d]; }

                // grad w_h += d_pre ⊗ states[ti]
                let s_prev = &states[ti];
                for r in 0..D {
                    let dp = d_pre[r];
                    let g_row = &mut g_w_h[r * D..(r + 1) * D];
                    for c in 0..D { g_row[c] += dp * s_prev[c]; }
                }
                // grad w_x += d_pre ⊗ intra[ti]
                let in_t = &intra[ti];
                for r in 0..D {
                    let dp = d_pre[r];
                    let g_row = &mut g_w_x[r * D..(r + 1) * D];
                    for c in 0..D { g_row[c] += dp * in_t[c]; }
                }

                // grad intra[ti] = W_x^T @ d_pre   (vai pra embeddings)
                let mut d_intra = vec![0.0f32; D];
                for r in 0..D {
                    let dp = d_pre[r];
                    let w_row = &self.w_x[r * D..(r + 1) * D];
                    for c in 0..D { d_intra[c] += dp * w_row[c]; }
                }
                // dispatch d_intra de volta pros embeddings dos 8 bits desse byte
                let byte = window[ti];
                for i in 0..8 {
                    let bit = ((byte >> (7 - i)) & 1) as usize;
                    let off = (i * 2 + bit) * D;
                    for d in 0..D { g_embed[off + d] += d_intra[d]; }
                }

                // grad states[ti] = W_h^T @ d_pre
                let mut d_prev = vec![0.0f32; D];
                for r in 0..D {
                    let dp = d_pre[r];
                    let w_row = &self.w_h[r * D..(r + 1) * D];
                    for c in 0..D { d_prev[c] += dp * w_row[c]; }
                }
                d_state = d_prev;
            }
        }

        // ── Clip + SGD ──────────────────────────────────────────
        let inv_b = 1.0 / batch as f32;
        let scale = inv_b;
        let clip_one = |g: &mut [f32]| {
            let norm = g.iter().map(|v| v * v).sum::<f32>().sqrt() * scale;
            let s = if norm > CLIP { CLIP / norm } else { 1.0 };
            for v in g.iter_mut() { *v *= scale * s; }
        };
        clip_one(&mut g_embed);
        clip_one(&mut g_w_h);
        clip_one(&mut g_w_x);
        clip_one(&mut g_b_h);
        clip_one(&mut g_head);
        clip_one(&mut g_head_b);

        for (p, g) in self.embed.iter_mut().zip(&g_embed)   { *p -= lr * g; }
        for (p, g) in self.w_h.iter_mut().zip(&g_w_h)       { *p -= lr * g; }
        for (p, g) in self.w_x.iter_mut().zip(&g_w_x)       { *p -= lr * g; }
        for (p, g) in self.b_h.iter_mut().zip(&g_b_h)       { *p -= lr * g; }
        for (p, g) in self.head.iter_mut().zip(&g_head)     { *p -= lr * g; }
        for (p, g) in self.head_b.iter_mut().zip(&g_head_b) { *p -= lr * g; }

        total_loss / batch as f32
    }

    /// Forward só (sem grad) → loss + probs
    fn eval(&self, window: &[u8], target: u8) -> f32 {
        let mut s = vec![0.0f32; D];
        for &byte in window {
            let mut intra = vec![0.0f32; D];
            for i in 0..8 {
                let bit = ((byte >> (7 - i)) & 1) as usize;
                let off = (i * 2 + bit) * D;
                for d in 0..D { intra[d] += self.embed[off + d]; }
            }
            let mut new_s = vec![0.0f32; D];
            for r in 0..D {
                let mut acc = self.b_h[r];
                let wh_row = &self.w_h[r * D..(r + 1) * D];
                let wx_row = &self.w_x[r * D..(r + 1) * D];
                for c in 0..D { acc += wh_row[c] * s[c] + wx_row[c] * intra[c]; }
                new_s[r] = acc.tanh();
            }
            s = new_s;
        }
        let mut loss = 0.0;
        for i in 0..8 {
            let mut logit = self.head_b[i];
            let head_row = &self.head[i * D..(i + 1) * D];
            for d in 0..D { logit += head_row[d] * s[d]; }
            let p = sigmoid(logit).clamp(1e-7, 1.0 - 1e-7);
            let tgt = ((target >> (7 - i)) & 1) as f32;
            loss += -(tgt * p.ln() + (1.0 - tgt) * (1.0 - p).ln()) / 8.0;
        }
        loss
    }

    /// Gera próximos N bytes começando do contexto
    fn generate(&self, prompt: &[u8], n: usize) -> Vec<u8> {
        let mut buf = prompt.to_vec();
        let mut s = vec![0.0f32; D];

        // Aquecer estado com prompt
        for &byte in prompt {
            let mut intra = vec![0.0f32; D];
            for i in 0..8 {
                let bit = ((byte >> (7 - i)) & 1) as usize;
                let off = (i * 2 + bit) * D;
                for d in 0..D { intra[d] += self.embed[off + d]; }
            }
            let mut new_s = vec![0.0f32; D];
            for r in 0..D {
                let mut acc = self.b_h[r];
                let wh_row = &self.w_h[r * D..(r + 1) * D];
                let wx_row = &self.w_x[r * D..(r + 1) * D];
                for c in 0..D { acc += wh_row[c] * s[c] + wx_row[c] * intra[c]; }
                new_s[r] = acc.tanh();
            }
            s = new_s;
        }

        for _ in 0..n {
            let mut byte_out: u8 = 0;
            for i in 0..8 {
                let mut logit = self.head_b[i];
                let head_row = &self.head[i * D..(i + 1) * D];
                for d in 0..D { logit += head_row[d] * s[d]; }
                let p = sigmoid(logit);
                if p > 0.5 { byte_out |= 1 << (7 - i); }
            }
            buf.push(byte_out);
            // Avança estado com o byte gerado
            let mut intra = vec![0.0f32; D];
            for i in 0..8 {
                let bit = ((byte_out >> (7 - i)) & 1) as usize;
                let off = (i * 2 + bit) * D;
                for d in 0..D { intra[d] += self.embed[off + d]; }
            }
            let mut new_s = vec![0.0f32; D];
            for r in 0..D {
                let mut acc = self.b_h[r];
                let wh_row = &self.w_h[r * D..(r + 1) * D];
                let wx_row = &self.w_x[r * D..(r + 1) * D];
                for c in 0..D { acc += wh_row[c] * s[c] + wx_row[c] * intra[c]; }
                new_s[r] = acc.tanh();
            }
            s = new_s;
        }
        buf[prompt.len()..].to_vec()
    }
}

// ── Main ───────────────────────────────────────────────────────────────────
fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   bitlevel_v1 — RNN bit-level CPU Rust (V1: bits indep)    ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("  W={} D={} LR={} EPOCHS={} STEPS={} BATCH={}", W, D, LR, EPOCHS, STEPS, BATCH);

    let data = fs::read(DATA).expect("corpus");
    println!("  Corpus: {} bytes", data.len());

    let mut rng = Rng::new(SEED);
    let mut model = Model::new(&mut rng);

    // Janelas e targets
    let n_pairs = if data.len() > W { data.len() - W } else { 0 };
    let val_size = (n_pairs / 8).max(200);
    let train_size = n_pairs - val_size;
    println!("  Pairs: train={} val={}", train_size, val_size);

    let make_pair = |start: usize, src: &[u8]| -> (Vec<u8>, u8) {
        (src[start..start + W].to_vec(), src[start + W])
    };

    let t0 = Instant::now();
    println!("\n  epoch  train_loss  val_loss   tempo");
    println!("  ──────────────────────────────────────");

    let mut best_val = f32::INFINITY;

    for epoch in 0..EPOCHS {
        let ep_start = Instant::now();

        // LR schedule: warmup + cosine
        let warm = EPOCHS / 10;
        let lr = if epoch < warm {
            LR * (epoch + 1) as f32 / warm as f32
        } else {
            let p = (epoch - warm) as f32 / (EPOCHS - warm) as f32;
            LR * 0.5 * (1.0 + (std::f32::consts::PI * p).cos()).max(0.1)
        };

        // Train
        let mut train_loss = 0.0;
        for _ in 0..STEPS {
            let mut wins = Vec::with_capacity(BATCH);
            let mut tgts = Vec::with_capacity(BATCH);
            for _ in 0..BATCH {
                let start = rng.next_usize(train_size);
                let (w, t) = make_pair(start, &data);
                wins.push(w);
                tgts.push(t);
            }
            train_loss += model.step_train(&wins, &tgts, lr);
        }
        train_loss /= STEPS as f32;

        // Val
        let mut val_loss = 0.0;
        let val_n = 100; // amostragem pra ser rápido
        for _ in 0..val_n {
            let start = train_size + rng.next_usize(val_size);
            let (w, t) = make_pair(start, &data);
            val_loss += model.eval(&w, t);
        }
        val_loss /= val_n as f32;

        if val_loss < best_val { best_val = val_loss; }

        println!("  {:>5}  {:>9.4}  {:>8.4}  {:>5.1}s", epoch, train_loss, val_loss, ep_start.elapsed().as_secs_f32());
        std::io::stdout().flush().ok();
    }

    let elapsed = t0.elapsed().as_secs_f32();
    println!("\n  ━━━ Wall: {:.1}s  best_val={:.4} ━━━", elapsed, best_val);

    // Geração demo
    let prompts: Vec<&str> = vec!["USER: oi\nMODEL:", "USER: bom dia\nMODEL:", "USER: tudo bem?\nMODEL:"];
    for p in &prompts {
        let prompt_bytes = p.as_bytes();
        let out = model.generate(prompt_bytes, 60);
        let s = String::from_utf8_lossy(&out);
        println!("\n  ▸ {:?}", p);
        println!("    → {:?}", s);
        let hex: String = out.iter().take(16).map(|b| format!("{:02X}", b)).collect::<Vec<_>>().join(" ");
        println!("    bytes(16): [{}]", hex);
    }
}
