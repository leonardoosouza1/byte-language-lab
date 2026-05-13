//! iara_brain.rs — IARA-Byte Brain v4 (MVP step 1)
//!
//! Hybrid byte-level generator inspired by Universe dual-brain + saci swarm + iara-brain GWT.
//!
//! MVP STEP 1: deterministic fast pathway only
//!   - Bigram LUT 256×256 (corpus freq)
//!   - Trigram LUT sparse (top-k cont per pair)
//!   - Phonetic mask (vowel/cons constraint via bits 0-4)
//!   - Char walker with backoff: trigram → bigram → unigram
//!
//! Next steps: graph memory, CfC backbone, specialists, fusion.
//!
//! Reference: META/lib/BYTE-LANGUAGE-SWARM.md

use std::collections::HashMap;
use std::fs;
use std::time::Instant;

// ── Bit decomposition ────────────────────────────────────────────────────
#[inline] fn bit5(b: u8) -> u8 { (b >> 5) & 1 }
#[inline] fn ident5(b: u8) -> u8 { b & 0x1F }

// ── RNG ──────────────────────────────────────────────────────────────────
struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Self(s.max(1)) }
    fn step(&mut self) -> u32 {
        self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn unit(&mut self) -> f32 { self.step() as f32 / u32::MAX as f32 }
}

// ── Phonetic class for ASCII letters ─────────────────────────────────────
// Returns class: 0=other, 1=vowel, 2=consonant, 3=digit, 4=space, 5=punct
fn phono_class(b: u8) -> u8 {
    match b {
        b'a'|b'e'|b'i'|b'o'|b'u'|b'A'|b'E'|b'I'|b'O'|b'U' => 1,
        b'A'..=b'Z' | b'a'..=b'z' => 2,
        0xC0..=0xFF => 2, // assume UTF-8 PT-BR letter (á, é, ç, ã, etc)
        b'0'..=b'9' => 3,
        b' ' | b'\n' | b'\t' => 4,
        b'.'|b','|b'!'|b'?'|b';'|b':'|b'-'|b'\''|b'"' => 5,
        _ => 0,
    }
}

// ── CfC cell (saci-style input-dependent gating) ─────────────────────────
// Forward: f_gate = σ(W_f h), i_gate = tanh(W_xi x + W_hi h),
//          decay = exp(-dt * f_gate), h' = h*decay + i_gate*(1-decay)
const CFC_D: usize = 16;     // hidden state dim
const CFC_E: usize = 8;      // class-input embed dim
const N_CLASSES: usize = 6;  // phono class count

struct CfC {
    emb_class: Vec<f32>,   // [N_CLASSES, CFC_E]
    w_xi: Vec<f32>,        // [CFC_D, CFC_E]
    w_hi: Vec<f32>,        // [CFC_D, CFC_D]
    w_f:  Vec<f32>,        // [CFC_D, CFC_D]
    b_f:  Vec<f32>,        // [CFC_D]
    b_i:  Vec<f32>,        // [CFC_D]
    head_temp: Vec<f32>,   // [CFC_D]
    bias_temp: f32,
    head_word: Vec<f32>,   // [CFC_D]
    bias_word: f32,
    head_cls: Vec<f32>,    // [N_CLASSES, CFC_D] — predict next class
    bias_cls: Vec<f32>,    // [N_CLASSES]
    // Bit specialists (sigmoid)
    head_b7: Vec<f32>,     // [CFC_D]
    bias_b7: f32,
    head_b6: Vec<f32>,
    bias_b6: f32,
    head_b5: Vec<f32>,
    bias_b5: f32,
}

#[inline] fn sig(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn matvec(out: &mut [f32], w: &[f32], x: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &w[r*cols..(r+1)*cols];
        let mut acc = 0.0f32;
        for c in 0..cols { acc += row[c] * x[c]; }
        out[r] = acc;
    }
}

impl CfC {
    fn new(rng: &mut Rng) -> Self {
        let scale_e = 0.4;
        let scale_h = (1.0 / (CFC_D as f32).sqrt()) * 0.6;
        let scale_x = (1.0 / (CFC_E as f32).sqrt()) * 0.6;
        let mut mk = |n: usize, s: f32| {
            (0..n).map(|_| (rng.unit()*2.0 - 1.0) * s).collect::<Vec<f32>>()
        };
        CfC {
            emb_class: mk(N_CLASSES * CFC_E, scale_e),
            w_xi: mk(CFC_D * CFC_E, scale_x),
            w_hi: mk(CFC_D * CFC_D, scale_h),
            w_f:  mk(CFC_D * CFC_D, scale_h),
            b_f:  vec![0.5f32; CFC_D],
            b_i:  vec![0.0f32; CFC_D],
            head_temp: mk(CFC_D, scale_h),
            bias_temp: 0.0,
            head_word: mk(CFC_D, scale_h),
            bias_word: 0.0,
            head_cls: mk(N_CLASSES * CFC_D, scale_h),
            bias_cls: vec![0.0f32; N_CLASSES],
            head_b7: mk(CFC_D, scale_h), bias_b7: 0.0,
            head_b6: mk(CFC_D, scale_h), bias_b6: 0.0,
            head_b5: mk(CFC_D, scale_h), bias_b5: 0.0,
        }
    }

    fn n_params(&self) -> usize {
        self.emb_class.len() + self.w_xi.len() + self.w_hi.len() + self.w_f.len()
        + self.b_f.len() + self.b_i.len()
        + self.head_temp.len() + 1 + self.head_word.len() + 1
        + self.head_cls.len() + self.bias_cls.len()
        + self.head_b7.len() + 1 + self.head_b6.len() + 1 + self.head_b5.len() + 1
    }

    /// Forward returning bit probabilities (consensus signal for byte distribution).
    fn step_full(&self, h: &[f32], cls_in: u8) -> (Vec<f32>, f32, f32, [f32; 3]) {
        let (h_new, t_mod, w_mod, _cls) = self.step_fwd(h, cls_in);
        let mut l7 = self.bias_b7;
        let mut l6 = self.bias_b6;
        let mut l5 = self.bias_b5;
        for d in 0..CFC_D {
            l7 += self.head_b7[d] * h_new[d];
            l6 += self.head_b6[d] * h_new[d];
            l5 += self.head_b5[d] * h_new[d];
        }
        (h_new, t_mod, w_mod, [sig(l7), sig(l6), sig(l5)])
    }

    fn embed(&self, cls: u8) -> Vec<f32> {
        let c = cls as usize % N_CLASSES;
        self.emb_class[c*CFC_E..(c+1)*CFC_E].to_vec()
    }

    /// Step forward — returns new hidden state + (temp_mod, word_mod, cls_logits).
    fn step_fwd(&self, h: &[f32], cls_in: u8) -> (Vec<f32>, f32, f32, Vec<f32>) {
        let x = self.embed(cls_in);
        let mut pre_xi = vec![0.0f32; CFC_D];
        let mut pre_hi = vec![0.0f32; CFC_D];
        let mut pre_f  = vec![0.0f32; CFC_D];
        matvec(&mut pre_xi, &self.w_xi, &x, CFC_D, CFC_E);
        matvec(&mut pre_hi, &self.w_hi, h, CFC_D, CFC_D);
        matvec(&mut pre_f,  &self.w_f,  h, CFC_D, CFC_D);
        let mut h_new = vec![0.0f32; CFC_D];
        for d in 0..CFC_D {
            let i_g = (pre_xi[d] + pre_hi[d] + self.b_i[d]).tanh();
            let f_g = sig(pre_f[d] + self.b_f[d]);
            let decay = (-1.0 * f_g).exp();
            h_new[d] = h[d] * decay + i_g * (1.0 - decay);
        }
        // Heads
        let mut t_mod = self.bias_temp;
        let mut w_mod = self.bias_word;
        for d in 0..CFC_D {
            t_mod += self.head_temp[d] * h_new[d];
            w_mod += self.head_word[d] * h_new[d];
        }
        let mut cls_logits = vec![0.0f32; N_CLASSES];
        for k in 0..N_CLASSES {
            let mut acc = self.bias_cls[k];
            let row = &self.head_cls[k*CFC_D..(k+1)*CFC_D];
            for d in 0..CFC_D { acc += row[d] * h_new[d]; }
            cls_logits[k] = acc;
        }
        (h_new, t_mod, w_mod, cls_logits)
    }

    /// Train one BPTT batch. Loss = CE(class) + BCE(bit7) + BCE(bit6) + BCE(bit5).
    /// Inputs: `classes` for CfC input, `bytes` for bit targets (same len).
    fn train_step(&mut self, classes: &[u8], bytes: &[u8], lr: f32) -> f32 {
        let t = classes.len().min(bytes.len());
        if t < 2 { return 0.0; }
        // Forward, store all states + logits
        let mut hs = vec![vec![0.0f32; CFC_D]; t + 1];
        let mut all_xi = vec![vec![0.0f32; CFC_D]; t];
        let mut all_hi = vec![vec![0.0f32; CFC_D]; t];
        let mut all_f  = vec![vec![0.0f32; CFC_D]; t];
        let mut all_ig = vec![vec![0.0f32; CFC_D]; t];
        let mut all_fg = vec![vec![0.0f32; CFC_D]; t];
        let mut all_decay = vec![vec![0.0f32; CFC_D]; t];
        let mut all_x = vec![vec![0.0f32; CFC_E]; t];
        let mut all_logits = vec![vec![0.0f32; N_CLASSES]; t];
        for ti in 0..t {
            let x = self.embed(classes[ti]);
            all_x[ti] = x.clone();
            matvec(&mut all_xi[ti], &self.w_xi, &x, CFC_D, CFC_E);
            matvec(&mut all_hi[ti], &self.w_hi, &hs[ti], CFC_D, CFC_D);
            matvec(&mut all_f[ti],  &self.w_f,  &hs[ti], CFC_D, CFC_D);
            for d in 0..CFC_D {
                let i_g = (all_xi[ti][d] + all_hi[ti][d] + self.b_i[d]).tanh();
                let f_g = sig(all_f[ti][d] + self.b_f[d]);
                let decay = (-f_g).exp();
                all_ig[ti][d] = i_g;
                all_fg[ti][d] = f_g;
                all_decay[ti][d] = decay;
                hs[ti+1][d] = hs[ti][d] * decay + i_g * (1.0 - decay);
            }
            // Compute class logits for cross-entropy
            for k in 0..N_CLASSES {
                let mut acc = self.bias_cls[k];
                let row = &self.head_cls[k*CFC_D..(k+1)*CFC_D];
                for d in 0..CFC_D { acc += row[d] * hs[ti+1][d]; }
                all_logits[ti][k] = acc;
            }
        }

        // Backward — only train head_cls + bias_cls + propagate through state.
        let mut g_emb = vec![0.0f32; self.emb_class.len()];
        let mut g_w_xi = vec![0.0f32; self.w_xi.len()];
        let mut g_w_hi = vec![0.0f32; self.w_hi.len()];
        let mut g_w_f  = vec![0.0f32; self.w_f.len()];
        let mut g_b_f  = vec![0.0f32; CFC_D];
        let mut g_b_i  = vec![0.0f32; CFC_D];
        let mut g_head_cls = vec![0.0f32; self.head_cls.len()];
        let mut g_bias_cls = vec![0.0f32; N_CLASSES];
        let mut g_head_b7 = vec![0.0f32; CFC_D];
        let mut g_head_b6 = vec![0.0f32; CFC_D];
        let mut g_head_b5 = vec![0.0f32; CFC_D];
        let mut g_bias_b7 = 0.0f32;
        let mut g_bias_b6 = 0.0f32;
        let mut g_bias_b5 = 0.0f32;
        let mut dh_next = vec![0.0f32; CFC_D];
        let mut total_loss = 0.0f32;

        for ti in (0..t-1).rev() {
            // softmax + CE on target classes[ti+1]
            let target = classes[ti+1] as usize % N_CLASSES;
            let logits = &all_logits[ti];
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|v| (v-mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e/sum).collect();
            total_loss += -(probs[target].max(1e-9)).ln();

            // grad on logits
            let mut dlogits = probs.clone();
            dlogits[target] -= 1.0;

            // d_head_cls + d_bias_cls + grad to h_{ti+1}
            let mut dh = dh_next.clone();
            for k in 0..N_CLASSES {
                g_bias_cls[k] += dlogits[k];
                let row_off = k*CFC_D;
                for d in 0..CFC_D {
                    g_head_cls[row_off + d] += dlogits[k] * hs[ti+1][d];
                    dh[d] += dlogits[k] * self.head_cls[row_off + d];
                }
            }

            // Bit specialists — BCE on next byte's bits
            let next_byte = bytes[ti+1];
            let tgt7 = ((next_byte >> 7) & 1) as f32;
            let tgt6 = ((next_byte >> 6) & 1) as f32;
            let tgt5 = ((next_byte >> 5) & 1) as f32;
            let mut l7 = self.bias_b7;
            let mut l6 = self.bias_b6;
            let mut l5 = self.bias_b5;
            for d in 0..CFC_D {
                l7 += self.head_b7[d] * hs[ti+1][d];
                l6 += self.head_b6[d] * hs[ti+1][d];
                l5 += self.head_b5[d] * hs[ti+1][d];
            }
            let p7 = sig(l7); let p6 = sig(l6); let p5 = sig(l5);
            let eps = 1e-7;
            total_loss += -(tgt7*p7.max(eps).ln() + (1.0-tgt7)*(1.0-p7).max(eps).ln());
            total_loss += -(tgt6*p6.max(eps).ln() + (1.0-tgt6)*(1.0-p6).max(eps).ln());
            total_loss += -(tgt5*p5.max(eps).ln() + (1.0-tgt5)*(1.0-p5).max(eps).ln());
            let d7 = p7 - tgt7;
            let d6 = p6 - tgt6;
            let d5 = p5 - tgt5;
            g_bias_b7 += d7;
            g_bias_b6 += d6;
            g_bias_b5 += d5;
            for d in 0..CFC_D {
                g_head_b7[d] += d7 * hs[ti+1][d];
                g_head_b6[d] += d6 * hs[ti+1][d];
                g_head_b5[d] += d5 * hs[ti+1][d];
                dh[d] += d7 * self.head_b7[d];
                dh[d] += d6 * self.head_b6[d];
                dh[d] += d5 * self.head_b5[d];
            }

            // backprop into state update: h_{ti+1}[d] = h_{ti}[d]*decay[d] + i_g[d]*(1-decay[d])
            // decay[d] = exp(-f_g[d]); f_g[d] = sig(pre_f[d]+b_f[d]); i_g[d]=tanh(pre_xi+pre_hi+b_i)
            let mut dh_prev = vec![0.0f32; CFC_D];
            let mut dpre_xi = vec![0.0f32; CFC_D];
            let mut dpre_hi = vec![0.0f32; CFC_D];
            let mut dpre_f  = vec![0.0f32; CFC_D];
            for d in 0..CFC_D {
                let decay = all_decay[ti][d];
                let i_g = all_ig[ti][d];
                let f_g = all_fg[ti][d];
                // d/d h_prev[d]
                dh_prev[d] = dh[d] * decay;
                // d/d decay[d] = h_prev - i_g
                let d_decay = dh[d] * (hs[ti][d] - i_g);
                // decay = exp(-f_g) → d decay / d f_g = -decay
                let d_fg = d_decay * (-decay);
                // f_g = sig(pre_f); d f_g / d pre_f = f_g*(1-f_g)
                dpre_f[d] = d_fg * f_g * (1.0 - f_g);
                g_b_f[d] += dpre_f[d];
                // d/d i_g[d] = (1 - decay)
                let d_ig = dh[d] * (1.0 - decay);
                // i_g = tanh(pre); d i_g / d pre = 1 - i_g^2
                let d_pre_ig = d_ig * (1.0 - i_g*i_g);
                dpre_xi[d] = d_pre_ig;
                dpre_hi[d] = d_pre_ig;
                g_b_i[d] += d_pre_ig;
            }
            // Grad to w_f, w_hi (from h_prev) — accumulate into dh_prev[d_prev]
            for d in 0..CFC_D {
                for d2 in 0..CFC_D {
                    let pre_f_grad = dpre_f[d];
                    let pre_hi_grad = dpre_hi[d];
                    g_w_f [d*CFC_D + d2] += pre_f_grad * hs[ti][d2];
                    g_w_hi[d*CFC_D + d2] += pre_hi_grad * hs[ti][d2];
                    dh_prev[d2] += pre_f_grad  * self.w_f [d*CFC_D + d2];
                    dh_prev[d2] += pre_hi_grad * self.w_hi[d*CFC_D + d2];
                }
            }
            // Grad to w_xi + emb_class
            let cls = classes[ti] as usize % N_CLASSES;
            for d in 0..CFC_D {
                for e in 0..CFC_E {
                    g_w_xi[d*CFC_E + e] += dpre_xi[d] * all_x[ti][e];
                    g_emb[cls*CFC_E + e] += dpre_xi[d] * self.w_xi[d*CFC_E + e];
                }
            }
            dh_next = dh_prev;
        }

        // Apply SGD with grad clip
        let clip = 1.0f32;
        let mut apply = |w: &mut [f32], g: &[f32]| {
            let norm: f32 = g.iter().map(|x| x*x).sum::<f32>().sqrt();
            let s = if norm > clip { clip / norm } else { 1.0 };
            for i in 0..w.len() { w[i] -= lr * g[i] * s; }
        };
        apply(&mut self.emb_class, &g_emb);
        apply(&mut self.w_xi, &g_w_xi);
        apply(&mut self.w_hi, &g_w_hi);
        apply(&mut self.w_f,  &g_w_f);
        apply(&mut self.b_f,  &g_b_f);
        apply(&mut self.b_i,  &g_b_i);
        apply(&mut self.head_cls, &g_head_cls);
        apply(&mut self.bias_cls, &g_bias_cls);
        apply(&mut self.head_b7, &g_head_b7);
        apply(&mut self.head_b6, &g_head_b6);
        apply(&mut self.head_b5, &g_head_b5);
        let mut scalar_apply = |w: &mut f32, g: f32| {
            let s = if g.abs() > clip { clip / g.abs() } else { 1.0 };
            *w -= lr * g * s;
        };
        scalar_apply(&mut self.bias_b7, g_bias_b7);
        scalar_apply(&mut self.bias_b6, g_bias_b6);
        scalar_apply(&mut self.bias_b5, g_bias_b5);

        total_loss / (t-1) as f32
    }
}

// ── Word boundary detection ──────────────────────────────────────────────
#[inline] fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphabetic() || b >= 0x80 || b == b'\''
}
#[inline] fn is_word_boundary(b: u8) -> bool { !is_word_byte(b) }

// ── Brain ────────────────────────────────────────────────────────────────
struct Brain {
    // Fast pathway: deterministic LUTs
    unigram: [u32; 256],
    bigram:  Vec<u32>,                          // [256*256]
    trigram: HashMap<(u8, u8), HashMap<u8, u32>>, // sparse
    // Phonetic stats
    class_trans: [[u32; 6]; 6],

    // Graph memory (word level)
    word_count: HashMap<Vec<u8>, u32>,
    word_bigram: HashMap<Vec<u8>, HashMap<Vec<u8>, u32>>, // prev_word → next_word
    word_after_punct: HashMap<u8, HashMap<Vec<u8>, u32>>, // boundary_byte → starting_word
    // Bridges char→word: which words start with byte b
    word_by_first: HashMap<u8, Vec<Vec<u8>>>,

    // Slow pathway: CfC modulator
    cfc: CfC,
}

impl Brain {
    fn new(rng: &mut Rng) -> Self {
        Brain {
            unigram: [0; 256],
            bigram: vec![0; 256 * 256],
            trigram: HashMap::new(),
            class_trans: [[0u32; 6]; 6],
            word_count: HashMap::new(),
            word_bigram: HashMap::new(),
            word_after_punct: HashMap::new(),
            word_by_first: HashMap::new(),
            cfc: CfC::new(rng),
        }
    }

    fn split_words(data: &[u8]) -> Vec<(Vec<u8>, u8)> {
        // Returns list of (word_bytes, byte_after_word) pairs
        let mut out = Vec::new();
        let mut cur = Vec::new();
        for i in 0..data.len() {
            let b = data[i];
            if is_word_byte(b) {
                cur.push(b);
            } else if !cur.is_empty() {
                out.push((std::mem::take(&mut cur), b));
            }
        }
        if !cur.is_empty() { out.push((cur, b'\n')); }
        out
    }

    fn ingest_words(&mut self, data: &[u8]) {
        let words = Self::split_words(data);
        let mut prev_word: Option<Vec<u8>> = None;
        let mut prev_punct: Option<u8> = None;
        for (w, after) in &words {
            // Filter only somewhat decent words
            if w.len() < 1 || w.len() > 30 { continue; }
            *self.word_count.entry(w.clone()).or_insert(0) += 1;
            self.word_by_first.entry(w[0]).or_insert_with(Vec::new).push(w.clone());
            if let Some(prev) = &prev_word {
                let inner = self.word_bigram.entry(prev.clone()).or_insert_with(HashMap::new);
                *inner.entry(w.clone()).or_insert(0) += 1;
            }
            if let Some(p) = prev_punct {
                let inner = self.word_after_punct.entry(p).or_insert_with(HashMap::new);
                *inner.entry(w.clone()).or_insert(0) += 1;
            }
            prev_punct = Some(*after);
            prev_word = Some(w.clone());
        }
        // Dedup word_by_first
        for words in self.word_by_first.values_mut() {
            words.sort();
            words.dedup();
        }
    }

    fn ingest_full(&mut self, data: &[u8]) {
        self.ingest_chars(data);
        self.ingest_words(data);
    }

    fn ingest_chars(&mut self, data: &[u8]) {
        for i in 0..data.len() {
            self.unigram[data[i] as usize] += 1;
            if i > 0 {
                let a = data[i-1];
                let b = data[i];
                self.bigram[(a as usize)*256 + (b as usize)] += 1;
                self.class_trans[phono_class(a) as usize][phono_class(b) as usize] += 1;
            }
            if i > 1 {
                let a = data[i-2];
                let b = data[i-1];
                let c = data[i];
                let inner = self.trigram.entry((a, b)).or_insert_with(HashMap::new);
                *inner.entry(c).or_insert(0) += 1;
            }
        }
    }

    /// Sample next byte. Uses backoff trigram→bigram + phono prior + bit consensus.
    /// `bit_probs` = [p_b7, p_b6, p_b5] from CfC specialists (None=skip).
    fn sample_next_with_bits(&self, ctx: &[u8], rng: &mut Rng, temp: f32,
                              bit_probs: Option<[f32; 3]>) -> u8 {
        // Build candidate distribution
        let mut probs = [0.0f64; 256];
        let mut has_data = false;

        // 1. Try trigram (most specific)
        if ctx.len() >= 2 {
            let key = (ctx[ctx.len()-2], ctx[ctx.len()-1]);
            if let Some(inner) = self.trigram.get(&key) {
                let total: u32 = inner.values().sum();
                if total >= 2 {
                    for (&b, &c) in inner.iter() {
                        probs[b as usize] = c as f64 / total as f64;
                    }
                    has_data = true;
                }
            }
        }

        // 2. Bigram backoff with weight 0.3 mix
        if let Some(&last) = ctx.last() {
            let row = &self.bigram[(last as usize)*256..(last as usize+1)*256];
            let total: u32 = row.iter().sum();
            if total > 0 {
                let mix = if has_data { 0.3 } else { 1.0 };
                for (i, &c) in row.iter().enumerate() {
                    probs[i] += mix * (c as f64 / total as f64);
                }
                has_data = true;
            }
        }

        // 3. Unigram fallback
        if !has_data {
            let total: u32 = self.unigram.iter().sum();
            for (i, &c) in self.unigram.iter().enumerate() {
                probs[i] = c as f64 / total as f64;
            }
        }

        // 4. Phonetic class prior — multiplicative
        if let Some(&last) = ctx.last() {
            let pc = phono_class(last) as usize;
            let row = &self.class_trans[pc];
            let total: u32 = row.iter().sum();
            if total > 0 {
                for b in 0..256u32 {
                    let nc = phono_class(b as u8) as usize;
                    let class_p = row[nc] as f64 / total as f64;
                    probs[b as usize] *= class_p.max(1e-6);
                }
            }
        }

        // 5. Bit consensus — soft mix (avoid suppressing UTF-8 bytes via underconfident heads)
        // mix = (1 - α) + α * agree^0.5 ; α=0.4 keeps most of original distribution
        if let Some([p7, p6, p5]) = bit_probs {
            let alpha = 0.4f64;
            let p7 = p7 as f64;
            let p6 = p6 as f64;
            let p5 = p5 as f64;
            for b in 0..256u32 {
                let bb = b as u8;
                let b7 = ((bb >> 7) & 1) as f64;
                let b6 = ((bb >> 6) & 1) as f64;
                let b5 = ((bb >> 5) & 1) as f64;
                let agree = (if b7 == 1.0 { p7 } else { 1.0 - p7 }).max(1e-3)
                          * (if b6 == 1.0 { p6 } else { 1.0 - p6 }).max(1e-3)
                          * (if b5 == 1.0 { p5 } else { 1.0 - p5 }).max(1e-3);
                let mix = (1.0 - alpha) + alpha * agree.sqrt();
                probs[b as usize] *= mix;
            }
        }

        // 6. Temperature + sample
        let t = temp.max(0.05) as f64;
        let mut sum = 0.0f64;
        for p in probs.iter_mut() {
            *p = p.powf(1.0 / t);
            sum += *p;
        }
        if sum < 1e-9 {
            // uniform fallback
            return rng.step() as u8;
        }
        for p in probs.iter_mut() { *p /= sum; }

        let r = rng.unit() as f64;
        let mut acc = 0.0f64;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if r < acc { return i as u8; }
        }
        255
    }

    #[allow(dead_code)]
    fn sample_next(&self, ctx: &[u8], rng: &mut Rng, temp: f32) -> u8 {
        self.sample_next_with_bits(ctx, rng, temp, None)
    }

    /// Last full word in prompt buffer (for word-bigram lookup).
    fn last_word(ctx: &[u8]) -> Option<Vec<u8>> {
        let mut end = ctx.len();
        while end > 0 && is_word_byte(ctx[end-1]) { end -= 1; }
        let mut start = end;
        let mut e = end;
        if e == 0 { return None; }
        while e > 0 && !is_word_byte(ctx[e-1]) { e -= 1; }
        if e == 0 { return None; }
        let word_end = e;
        let mut s = word_end;
        while s > 0 && is_word_byte(ctx[s-1]) { s -= 1; }
        start = s;
        if start < word_end { Some(ctx[start..word_end].to_vec()) } else { None }
    }

    fn sample_word(&self, prev_word: Option<&[u8]>, after_punct: Option<u8>,
                    rng: &mut Rng, word_temp: f32) -> Option<Vec<u8>> {
        // Try word→word first
        if let Some(pw) = prev_word {
            if let Some(inner) = self.word_bigram.get(pw) {
                let total: u32 = inner.values().sum();
                if total >= 2 {
                    return Some(self.weighted_sample_word(inner, rng, word_temp));
                }
            }
        }
        // Punct-conditioned
        if let Some(p) = after_punct {
            if let Some(inner) = self.word_after_punct.get(&p) {
                let total: u32 = inner.values().sum();
                if total >= 2 {
                    return Some(self.weighted_sample_word(inner, rng, word_temp));
                }
            }
        }
        None
    }

    fn weighted_sample_word(&self, m: &HashMap<Vec<u8>, u32>, rng: &mut Rng, temp: f32) -> Vec<u8> {
        let t = temp.max(0.05) as f64;
        let mut entries: Vec<(&Vec<u8>, f64)> = m.iter()
            .map(|(w, &c)| (w, (c as f64).powf(1.0 / t)))
            .collect();
        let sum: f64 = entries.iter().map(|(_, p)| p).sum();
        let r = rng.unit() as f64 * sum;
        let mut acc = 0.0f64;
        for (w, p) in &entries {
            acc += p;
            if r < acc { return (*w).clone(); }
        }
        entries.last().map(|(w, _)| (*w).clone()).unwrap_or_default()
    }

    /// Train CfC on corpus class transitions
    fn train_cfc(&mut self, data: &[u8], epochs: usize, win: usize, batch: usize,
                 lr: f32, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let mut losses = Vec::with_capacity(epochs);
        let classes: Vec<u8> = data.iter().map(|&b| phono_class(b)).collect();
        let n = classes.len();
        if n < win + 2 { return losses; }
        let steps_per_ep = 80;
        for _ep in 0..epochs {
            let mut sum_loss = 0.0;
            for _ in 0..steps_per_ep {
                let mut batch_loss = 0.0;
                for _ in 0..batch {
                    let start = (rng.step() as usize) % (n - win - 1);
                    let seq_cls = &classes[start..start+win];
                    let seq_b   = &data[start..start+win];
                    batch_loss += self.cfc.train_step(seq_cls, seq_b, lr / batch as f32);
                }
                sum_loss += batch_loss / batch as f32;
            }
            losses.push(sum_loss / steps_per_ep as f32);
        }
        losses
    }

    /// Run CfC over prompt bytes, return final state.
    fn cfc_warm(&self, prompt: &[u8]) -> Vec<f32> {
        let mut h = vec![0.0f32; CFC_D];
        for &b in prompt {
            let (h_new, _, _, _) = self.cfc.step_fwd(&h, phono_class(b));
            h = h_new;
        }
        h
    }

    fn generate(&self, prompt: &[u8], n: usize, temp: f32, seed: u64) -> Vec<u8> {
        let mut rng = Rng::new(seed);
        let mut ctx: Vec<u8> = prompt.to_vec();
        let mut out = Vec::with_capacity(n);
        let stop_seq = b"\nUSER:";
        let mut cfc_h = self.cfc_warm(prompt);
        // Base word_prob — modulated below
        let base_word_prob: f32 = 0.65;

        while out.len() < n {
            // Check stop
            if out.len() >= stop_seq.len() {
                let tail = &out[out.len()-stop_seq.len()..];
                if tail == stop_seq { break; }
            }

            // Step CfC with latest byte — get modulators + bit specialists
            let last = *ctx.last().unwrap_or(&b' ');
            let (h_new, t_mod, w_mod, bits) = self.cfc.step_full(&cfc_h, phono_class(last));
            cfc_h = h_new;

            // Apply CfC modulations (scaled to small range)
            let effective_temp = (temp + 0.3 * t_mod.tanh()).clamp(0.1, 2.5);
            let effective_word_prob = (base_word_prob + 0.25 * w_mod.tanh()).clamp(0.05, 0.95);

            let at_boundary = !is_word_byte(last);
            let try_word = at_boundary && rng.unit() < effective_word_prob;

            if try_word {
                let prev_w = Self::last_word(&ctx);
                let punct = if at_boundary { Some(last) } else { None };
                if let Some(word) = self.sample_word(prev_w.as_deref(), punct, &mut rng, effective_temp) {
                    // emit word bytes
                    for &wb in &word {
                        if out.len() >= n { break; }
                        out.push(wb);
                        ctx.push(wb);
                    }
                    // emit a space or punct — no bit consensus (boundary char)
                    let b = self.sample_next_with_bits(&ctx, &mut rng, effective_temp, None);
                    if out.len() < n {
                        let space_b = if is_word_byte(b) { b' ' } else { b };
                        out.push(space_b);
                        ctx.push(space_b);
                    }
                    continue;
                }
            }

            // Char-mode fallback — use bit consensus from specialists
            let b = self.sample_next_with_bits(&ctx, &mut rng, effective_temp, Some(bits));
            out.push(b);
            ctx.push(b);
        }
        out
    }

    /// Top K outgoing word edges (most frequent transitions) - global view.
    fn top_word_edges(&self, k: usize) -> Vec<(String, String, u32)> {
        let mut all: Vec<(String, String, u32)> = Vec::new();
        for (prev, nexts) in &self.word_bigram {
            for (next, &c) in nexts {
                if c >= 5 {
                    all.push((
                        String::from_utf8_lossy(prev).to_string(),
                        String::from_utf8_lossy(next).to_string(),
                        c,
                    ));
                }
            }
        }
        all.sort_by(|a, b| b.2.cmp(&a.2));
        all.truncate(k);
        all
    }

    /// For a given byte: top K predecessors + top K successors.
    fn byte_relations(&self, b: u8, k: usize) -> (Vec<(u8, u32)>, Vec<(u8, u32)>) {
        let mut succs: Vec<(u8, u32)> = (0..256u32)
            .map(|i| (i as u8, self.bigram[(b as usize)*256 + i as usize]))
            .filter(|&(_, c)| c > 0).collect();
        succs.sort_by(|a, b| b.1.cmp(&a.1));
        succs.truncate(k);
        let mut preds: Vec<(u8, u32)> = (0..256u32)
            .map(|i| (i as u8, self.bigram[(i as usize)*256 + b as usize]))
            .filter(|&(_, c)| c > 0).collect();
        preds.sort_by(|a, b| b.1.cmp(&a.1));
        preds.truncate(k);
        (preds, succs)
    }

    /// PT-essence: fraction of generated tokens that exist in word vocab.
    fn pt_essence(&self, text: &[u8]) -> f32 {
        let words = Self::split_words(text);
        if words.is_empty() { return 0.0; }
        let hit = words.iter().filter(|(w, _)| self.word_count.contains_key(w)).count();
        hit as f32 / words.len() as f32
    }

    fn analyze_text(&self, label: &str, text: &[u8]) {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║  Comprehension Analysis: {:35}  ║", label);
        println!("╚══════════════════════════════════════════════════════════════╝");

        let words = Self::split_words(text);
        let unique_words: std::collections::HashSet<&Vec<u8>> = words.iter().map(|(w, _)| w).collect();
        println!("  Texto: {} bytes, {} word tokens, {} unique words",
                 text.len(), words.len(), unique_words.len());
        // Coverage: quantas words estão no vocab do brain?
        let in_vocab = unique_words.iter().filter(|w| self.word_count.contains_key(**w)).count();
        let cov = in_vocab as f32 / unique_words.len().max(1) as f32;
        println!("  Cobertura vocab: {}/{} ({:.1}%) palavras do texto existem no grafo",
                 in_vocab, unique_words.len(), cov * 100.0);

        // Sample 5 word→word transitions from text and show top 3 in graph
        println!("\n  ▸ Relações word→word (texto → top 3 sucessores no grafo):");
        let mut shown = 0;
        for (w, _) in words.iter().take(80) {
            if shown >= 8 { break; }
            if let Some(inner) = self.word_bigram.get(w) {
                let total: u32 = inner.values().sum();
                if total < 5 { continue; }
                let mut top: Vec<(&Vec<u8>, &u32)> = inner.iter().collect();
                top.sort_by(|a, b| b.1.cmp(a.1));
                let s: String = top.iter().take(3)
                    .map(|(n, c)| format!("{}({})", String::from_utf8_lossy(n), c))
                    .collect::<Vec<_>>().join(" ");
                println!("    {:<14} → {}", String::from_utf8_lossy(w), s);
                shown += 1;
            }
        }

        // Letter relationships for common PT bytes
        println!("\n  ▸ Letras: vizinhança bigram (top 4 antes / top 4 depois):");
        for &b in b"oaeqic".iter() {
            let (preds, succs) = self.byte_relations(b, 4);
            let p: String = preds.iter()
                .map(|(c, n)| format!("{}({})", char::from(*c).escape_default(), n))
                .collect::<Vec<_>>().join(" ");
            let s: String = succs.iter()
                .map(|(c, n)| format!("{}({})", char::from(*c).escape_default(), n))
                .collect::<Vec<_>>().join(" ");
            println!("    '{}': pred[{}]  →  succ[{}]", char::from(b), p, s);
        }
    }

    fn report(&self) {
        let trigrams: usize = self.trigram.values().map(|m| m.len()).sum();
        let trig_keys = self.trigram.len();
        let bigram_active = self.bigram.iter().filter(|&&c| c > 0).count();
        let uni_total: u32 = self.unigram.iter().sum();
        println!("  Corpus tokens (bytes): {}", uni_total);
        println!("  Unique bytes:          {}", self.unigram.iter().filter(|&&c| c > 0).count());
        println!("  Bigram entries (>0):   {}", bigram_active);
        println!("  Trigram keys (pairs):  {}", trig_keys);
        println!("  Trigram continuations: {}", trigrams);

        let n_words = self.word_count.len();
        let total_word_tokens: u32 = self.word_count.values().sum();
        let word_pair_keys = self.word_bigram.len();
        let word_pair_total: usize = self.word_bigram.values().map(|m| m.len()).sum();
        println!("  Word vocab:            {}", n_words);
        println!("  Word tokens:           {}", total_word_tokens);
        println!("  Word→Word pairs:       {} keys, {} edges", word_pair_keys, word_pair_total);
        let after_punct: usize = self.word_after_punct.values().map(|m| m.len()).sum();
        println!("  Punct→Word edges:      {}", after_punct);
        // Top 10 words
        let mut top: Vec<(&Vec<u8>, &u32)> = self.word_count.iter().collect();
        top.sort_by(|a,b| b.1.cmp(a.1));
        print!("  Top 10 words: ");
        for (w, c) in top.iter().take(10) {
            print!("{}({}) ", String::from_utf8_lossy(w), c);
        }
        println!();
        // class transition matrix
        println!("\n  Phonetic class transitions (rows=prev, cols=next):");
        println!("        other vowel cons  digit space punct");
        let names = ["other","vowel","cons ","digit","space","punct"];
        for r in 0..6 {
            print!("  {}  ", names[r]);
            let row_total: u32 = self.class_trans[r].iter().sum();
            for c in 0..6 {
                let p = if row_total > 0 { self.class_trans[r][c] as f32 / row_total as f32 } else { 0.0 };
                print!("{:>5.2} ", p);
            }
            println!();
        }
    }
}

fn main() -> std::io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA-Byte Brain v4 — Step 2 (Word Graph)                   ║");
    println!("║   Bigram+Trigram char + Word↔Word graph + Phono mask         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut init_rng = Rng::new(7);
    let mut brain = Brain::new(&mut init_rng);
    let corpus_files = &[
        "data/dialogues_augmented.txt",
        "data/qwen_sustained.txt",
        "data/online_learning.txt",
    ];

    let mut all_data = Vec::new();
    let t0 = Instant::now();
    for p in corpus_files {
        if std::path::Path::new(p).exists() {
            let text = fs::read(p)?;
            brain.ingest_full(&text);
            all_data.extend_from_slice(&text);
            println!("  ✓ ingested {} ({} bytes)", p, text.len());
        }
    }
    println!("  Ingestion: {:.2}s", t0.elapsed().as_secs_f32());
    println!("  CfC params: {}\n", brain.cfc.n_params());

    brain.report();

    // Train CfC modulator
    println!("\n▸ Training CfC modulator (class transition CE):");
    let t_train = Instant::now();
    let losses = brain.train_cfc(&all_data, 25, 64, 16, 0.02, 13);
    println!("  Epochs: {}, Wall: {:.1}s", losses.len(), t_train.elapsed().as_secs_f32());
    for (i, l) in losses.iter().enumerate() {
        println!("    ep {:>2}  CE={:.4}", i, l);
    }

    // Generation demo
    let prompts: &[&str] = &[
        "USER: Oi\nMODEL: ",
        "USER: o que e o Brasil\nMODEL: ",
        "USER: quem foi Pele\nMODEL: ",
        "USER: feijoada\nMODEL: ",
        "USER: tudo bem\nMODEL: ",
    ];

    // Comprehension test on book-like passage
    let book_path = "data/teste_livro_pt.txt";
    if std::path::Path::new(book_path).exists() {
        let book = fs::read(book_path)?;
        brain.analyze_text("data/teste_livro_pt.txt", &book);

        // PT-essence: brain generates 500 bytes, measure fraction of real words
        let neutral_prompt = b"USER: conta uma historia\nMODEL: ";
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in neutral_prompt { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
        let g = brain.generate(neutral_prompt, 500, 0.7, h);
        let ess = brain.pt_essence(&g);
        println!("\n  ▸ PT-essence test (500 bytes gerados from neutral prompt):");
        println!("    Palavras válidas no vocab: {:.1}%", ess * 100.0);
        println!("    Amostra (200 bytes): {:?}", String::from_utf8_lossy(&g[..g.len().min(200)]));

        // Also measure essence of the book itself (sanity check)
        let book_ess = brain.pt_essence(&book);
        println!("    [sanity] Essência do livro (cobertura no grafo): {:.1}%", book_ess * 100.0);
    }

    println!("\n▸ Geração demo:\n");
    for &p in prompts {
        let prompt_b = p.as_bytes();
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in prompt_b { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
        for &temp in &[0.5f32, 0.8] {
            let seed = h ^ ((temp * 1000.0) as u64);
            let out = brain.generate(prompt_b, 80, temp, seed);
            let s = String::from_utf8_lossy(&out);
            println!("  [t={:.1}] {:?}", temp, p);
            println!("            → {:?}\n", s);
        }
    }

    Ok(())
}
