//! bitlevel_v2.rs — RNN bit-level CPU Rust (V2: AUTOREGRESSIVE bits)
//!
//! Diferença chave de V1: cada head sequencial recebe os bits já preditos
//! do MESMO byte como conditioning.
//!   head_0: state              → b7
//!   head_1: state, b7          → b6
//!   head_2: state, b7, b6      → b5
//!   ...
//!   head_7: state, b7..b1      → b0
//!
//! Captura correlações entre bits (ex: bits 7,6 sempre `01` em ASCII).
//! Loss = avg BCE over 8 bits, teach-forcing durante treino.

use std::fs;
use std::io::Write;
use std::time::Instant;

const W:        usize = 128;
const D:        usize = 256;   // 128 → 256: 4× compute, mais capacidade
const LR:       f32   = 0.005;
const EPOCHS:   usize = 60;    // 30 → 60: mais treino
const STEPS:    usize = 300;   // 200 → 300
const BATCH:    usize = 16;
const SEED:     u64   = 7;
const CLIP:     f32   = 5.0;
const TEMP:     f32   = 0.8;   // sampling temperature (≠ greedy)
const DATA:     &str  = "/home/leonardo/projects/LLM/byte-language-lab/data/dialogues_pt.txt";

// Tamanho de entrada da head i: D + i bits anteriores
// Total params head: sum_{i=0..8} (D + i) = 8*D + 28 = 1052
// Implementação: armazenamos head[i] com D+i floats
const HEAD_OFFSET: [usize; 9] = [0, D, 2*D+1, 3*D+3, 4*D+6, 5*D+10, 6*D+15, 7*D+21, 8*D+28];
const HEAD_TOTAL: usize = 8*D + 28; // = HEAD_OFFSET[8]

// ── RNG ───────────────────────────────────────────────────────────────────
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

#[inline] fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
#[inline] fn tanh_grad_from_y(y: f32) -> f32 { 1.0 - y * y }

struct Model {
    embed: Vec<f32>,        // [8, 2, D]
    w_h:   Vec<f32>,        // [D, D]
    w_x:   Vec<f32>,        // [D, D]
    b_h:   Vec<f32>,        // [D]
    head:  Vec<f32>,        // flat: 8 heads, sizes [D, D+1, ..., D+7], total HEAD_TOTAL
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
            head:   vec![0.0; HEAD_TOTAL],
            head_b: vec![0.0; 8],
        };
        for v in &mut m.embed { *v = rng.next_f32() * init_emb; }
        for v in &mut m.w_h   { *v = rng.next_f32() * init_w; }
        for v in &mut m.w_x   { *v = rng.next_f32() * init_w; }
        for v in &mut m.head  { *v = rng.next_f32() * init_w; }
        m
    }

    /// Para conveniência: head_i tem (D + i) pesos a partir de HEAD_OFFSET[i].
    #[inline]
    fn head_slice(&self, i: usize) -> &[f32] {
        &self.head[HEAD_OFFSET[i]..HEAD_OFFSET[i + 1]]
    }

    /// Forward + BPTT autoregressive bits. Retorna loss média do batch.
    fn step_train(&mut self, windows: &[Vec<u8>], targets: &[u8], lr: f32) -> f32 {
        let batch = windows.len();
        let mut total_loss = 0.0f32;

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
            let mut states = vec![vec![0.0f32; D]; t + 1];
            let mut intra  = vec![vec![0.0f32; D]; t];

            // ── Forward RNN ─────────────────────────────────────
            for ti in 0..t {
                let byte = window[ti];
                let intra_t = &mut intra[ti];
                for d in 0..D { intra_t[d] = 0.0; }
                for i in 0..8 {
                    let bit = ((byte >> (7 - i)) & 1) as usize;
                    let off = (i * 2 + bit) * D;
                    for d in 0..D { intra_t[d] += self.embed[off + d]; }
                }
                let mut pre_h = vec![0.0f32; D];
                let mut pre_x = vec![0.0f32; D];
                for r in 0..D {
                    let mut a = 0.0;
                    let row = &self.w_h[r * D..(r + 1) * D];
                    for c in 0..D { a += row[c] * states[ti][c]; }
                    pre_h[r] = a;
                }
                for r in 0..D {
                    let mut a = 0.0;
                    let row = &self.w_x[r * D..(r + 1) * D];
                    for c in 0..D { a += row[c] * intra_t[c]; }
                    pre_x[r] = a;
                }
                let s_next = &mut states[ti + 1];
                for d in 0..D { s_next[d] = (pre_h[d] + pre_x[d] + self.b_h[d]).tanh(); }
            }

            let final_state = states[t].clone();

            // ── Heads autoregressivas (teach-forcing) ──────────────
            let mut prev_bits = [0.0f32; 8]; // ±1 encoding pra entradas
            let mut d_state = vec![0.0f32; D];

            // Forward pass das 8 heads (com bits true)
            let mut probs = [0.0f32; 8];
            for i in 0..8 {
                let head_w = self.head_slice(i);
                let mut logit = self.head_b[i];
                // primeiros D pesos: state
                for d in 0..D { logit += head_w[d] * final_state[d]; }
                // próximos i pesos: prev_bits[0..i]
                for k in 0..i { logit += head_w[D + k] * prev_bits[k]; }
                probs[i] = sigmoid(logit);
                let tgt = ((target >> (7 - i)) & 1) as f32;
                let eps = 1e-7f32;
                let p = probs[i].clamp(eps, 1.0 - eps);
                total_loss += -(tgt * p.ln() + (1.0 - tgt) * (1.0 - p).ln()) / 8.0;
                // Teach-force: usa true bit como condicionamento da próxima head
                prev_bits[i] = 2.0 * tgt - 1.0;  // ±1
            }

            // ── Backward pelas heads ───────────────────────────────
            // Loop reverso: i = 7..0
            for i in (0..8).rev() {
                let tgt = ((target >> (7 - i)) & 1) as f32;
                let d_logit = probs[i] - tgt;
                g_head_b[i] += d_logit;

                let head_w_offset = HEAD_OFFSET[i];
                // grad head_w[d] (state weights)
                for d in 0..D {
                    g_head[head_w_offset + d] += d_logit * final_state[d];
                    d_state[d] += d_logit * self.head[head_w_offset + d];
                }
                // grad head_w[D+k] (prev bits weights) — NÃO propaga em prev_bits
                // (teach-force, são true labels, sem gradient through)
                for k in 0..i {
                    g_head[head_w_offset + D + k] += d_logit * prev_bits[k];
                }
            }

            // ── BPTT (igual V1) ────────────────────────────────────
            for ti in (0..t).rev() {
                let s_next = &states[ti + 1];
                let mut d_pre = vec![0.0f32; D];
                for d in 0..D { d_pre[d] = d_state[d] * tanh_grad_from_y(s_next[d]); }
                for d in 0..D { g_b_h[d] += d_pre[d]; }

                let s_prev = &states[ti];
                for r in 0..D {
                    let dp = d_pre[r];
                    let g_row = &mut g_w_h[r * D..(r + 1) * D];
                    for c in 0..D { g_row[c] += dp * s_prev[c]; }
                }
                let in_t = &intra[ti];
                for r in 0..D {
                    let dp = d_pre[r];
                    let g_row = &mut g_w_x[r * D..(r + 1) * D];
                    for c in 0..D { g_row[c] += dp * in_t[c]; }
                }
                let mut d_intra = vec![0.0f32; D];
                for r in 0..D {
                    let dp = d_pre[r];
                    let w_row = &self.w_x[r * D..(r + 1) * D];
                    for c in 0..D { d_intra[c] += dp * w_row[c]; }
                }
                let byte = window[ti];
                for i in 0..8 {
                    let bit = ((byte >> (7 - i)) & 1) as usize;
                    let off = (i * 2 + bit) * D;
                    for d in 0..D { g_embed[off + d] += d_intra[d]; }
                }
                let mut d_prev = vec![0.0f32; D];
                for r in 0..D {
                    let dp = d_pre[r];
                    let w_row = &self.w_h[r * D..(r + 1) * D];
                    for c in 0..D { d_prev[c] += dp * w_row[c]; }
                }
                d_state = d_prev;
            }
        }

        let inv_b = 1.0 / batch as f32;
        let clip_one = |g: &mut [f32]| {
            let norm = g.iter().map(|v| v * v).sum::<f32>().sqrt() * inv_b;
            let s = if norm > CLIP { CLIP / norm } else { 1.0 };
            for v in g.iter_mut() { *v *= inv_b * s; }
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
                let wh = &self.w_h[r * D..(r + 1) * D];
                let wx = &self.w_x[r * D..(r + 1) * D];
                for c in 0..D { acc += wh[c] * s[c] + wx[c] * intra[c]; }
                new_s[r] = acc.tanh();
            }
            s = new_s;
        }
        let mut prev_bits = [0.0f32; 8];
        let mut loss = 0.0;
        for i in 0..8 {
            let head_w = self.head_slice(i);
            let mut logit = self.head_b[i];
            for d in 0..D { logit += head_w[d] * s[d]; }
            for k in 0..i { logit += head_w[D + k] * prev_bits[k]; }
            let p = sigmoid(logit).clamp(1e-7, 1.0 - 1e-7);
            let tgt = ((target >> (7 - i)) & 1) as f32;
            loss += -(tgt * p.ln() + (1.0 - tgt) * (1.0 - p).ln()) / 8.0;
            prev_bits[i] = 2.0 * tgt - 1.0;
        }
        loss
    }

    fn generate(&self, prompt: &[u8], n: usize, rng: &mut Rng, temp: f32) -> Vec<u8> {
        let mut buf = prompt.to_vec();
        let mut s = vec![0.0f32; D];
        let advance = |s: &mut Vec<f32>, byte: u8, m: &Model| {
            let mut intra = vec![0.0f32; D];
            for i in 0..8 {
                let bit = ((byte >> (7 - i)) & 1) as usize;
                let off = (i * 2 + bit) * D;
                for d in 0..D { intra[d] += m.embed[off + d]; }
            }
            let mut new_s = vec![0.0f32; D];
            for r in 0..D {
                let mut acc = m.b_h[r];
                let wh = &m.w_h[r * D..(r + 1) * D];
                let wx = &m.w_x[r * D..(r + 1) * D];
                for c in 0..D { acc += wh[c] * s[c] + wx[c] * intra[c]; }
                new_s[r] = acc.tanh();
            }
            *s = new_s;
        };
        for &byte in prompt { advance(&mut s, byte, self); }

        for _ in 0..n {
            let mut byte_out: u8 = 0;
            let mut prev_bits = [0.0f32; 8];
            for i in 0..8 {
                let head_w = self.head_slice(i);
                let mut logit = self.head_b[i];
                for d in 0..D { logit += head_w[d] * s[d]; }
                for k in 0..i { logit += head_w[D + k] * prev_bits[k]; }
                // Temperature sampling: scale logit, sigmoid → bernoulli
                let p = sigmoid(logit / temp.max(0.01));
                // u in [0, 1)
                let u = (rng.next_u32() as f32) / (u32::MAX as f32);
                let bit = if u < p { 1u8 } else { 0u8 };
                if bit == 1 { byte_out |= 1 << (7 - i); }
                prev_bits[i] = if bit == 1 { 1.0 } else { -1.0 };
            }
            buf.push(byte_out);
            advance(&mut s, byte_out, self);
        }
        buf[prompt.len()..].to_vec()
    }
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   bitlevel_v2 — RNN bit-level CPU (V2: AUTOREGRESSIVE)     ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("  W={} D={} LR={} EPOCHS={} STEPS={} BATCH={}", W, D, LR, EPOCHS, STEPS, BATCH);
    println!("  Head sizes: D, D+1, ..., D+7  (total params head = {})", HEAD_TOTAL);

    let data = fs::read(DATA).expect("corpus");
    println!("  Corpus: {} bytes", data.len());

    let mut rng = Rng::new(SEED);
    let mut model = Model::new(&mut rng);

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
        let warm = EPOCHS / 10;
        let lr = if epoch < warm {
            LR * (epoch + 1) as f32 / warm as f32
        } else {
            let p = (epoch - warm) as f32 / (EPOCHS - warm) as f32;
            LR * 0.5 * (1.0 + (std::f32::consts::PI * p).cos()).max(0.1)
        };

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

        let mut val_loss = 0.0;
        let val_n = 100;
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
    println!("\n  ━━━ Wall: {:.1}s  best_val={:.4}  bits/byte={:.2} ━━━",
             elapsed, best_val, best_val * 8.0 / std::f32::consts::LN_2);

    let prompts: Vec<&str> = vec!["USER: oi\nMODEL:", "USER: bom dia\nMODEL:", "USER: tudo bem?\nMODEL:"];
    for p in &prompts {
        let prompt_bytes = p.as_bytes();
        println!("\n  ▸ {:?}", p);
        // Greedy
        let out_g = model.generate(prompt_bytes, 60, &mut Rng::new(SEED ^ 0xdead), 0.001);
        println!("    [greedy] → {:?}", String::from_utf8_lossy(&out_g));
        // 3 samples com temperatura
        for run in 0..3 {
            let mut g = Rng::new(SEED ^ (run as u64 * 0xbeef + 1));
            let out_t = model.generate(prompt_bytes, 60, &mut g, TEMP);
            println!("    [t={:.1} run{}] → {:?}", TEMP, run+1, String::from_utf8_lossy(&out_t));
        }
    }
}
