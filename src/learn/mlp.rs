//! # Tiny MLP from scratch
//!
//! 1-hidden-layer MLP com forward + backprop manual, sem dependências.
//! Arquitetura:
//!
//! ```text
//! input (K*V one-hot) → Linear(K*V → H) → ReLU → Linear(H → V) → softmax
//! ```
//!
//! Treinado via SGD com mini-batch, cross-entropy loss.
//! Usado para comparar quão rapidamente um modelo pequeno aprende a prever
//! próximo token sob diferentes encodings (ASCII, Qwen, byte-positional, Path C).

use crate::Rng;

/// Configuração do MLP.
#[derive(Debug, Clone)]
pub struct MlpConfig {
    pub vocab_size: usize,
    pub window: usize,
    pub hidden: usize,
    pub learning_rate: f32,
    pub seed: u64,
}

/// MLP de 1 camada oculta. Pesos armazenados row-major.
pub struct Mlp {
    cfg: MlpConfig,
    /// W1: shape (input_dim, hidden), input_dim = window * vocab_size
    w1: Vec<f32>,
    b1: Vec<f32>, // (hidden,)
    /// W2: shape (hidden, vocab_size)
    w2: Vec<f32>,
    b2: Vec<f32>, // (vocab_size,)
}

impl Mlp {
    pub fn new(cfg: MlpConfig) -> Self {
        let mut rng = Rng::new(cfg.seed);
        let input_dim = cfg.window * cfg.vocab_size;
        // Xavier init: σ = sqrt(2 / fan_in)
        let s1 = (2.0 / input_dim as f32).sqrt();
        let s2 = (2.0 / cfg.hidden as f32).sqrt();

        let w1: Vec<f32> = (0..input_dim * cfg.hidden)
            .map(|_| (rng.next_f64() as f32 - 0.5) * 2.0 * s1)
            .collect();
        let w2: Vec<f32> = (0..cfg.hidden * cfg.vocab_size)
            .map(|_| (rng.next_f64() as f32 - 0.5) * 2.0 * s2)
            .collect();
        let b1 = vec![0.0_f32; cfg.hidden];
        let b2 = vec![0.0_f32; cfg.vocab_size];
        Self { cfg, w1, b1, w2, b2 }
    }

    /// Forward pass: input window de K tokens → distribution over next token.
    /// Retorna (logits, hidden_pre_relu, hidden_post_relu, input_onehot) para backprop.
    pub fn forward(&self, window: &[u32]) -> ForwardCache {
        let v = self.cfg.vocab_size;
        let k = self.cfg.window;
        let h = self.cfg.hidden;
        debug_assert_eq!(window.len(), k);

        // Input one-hot (k * v dim)
        let mut input = vec![0.0_f32; k * v];
        for (i, &tok) in window.iter().enumerate() {
            let idx = (tok as usize).min(v - 1);
            input[i * v + idx] = 1.0;
        }

        // Hidden pre-activation: h_pre = input @ W1 + b1
        let mut h_pre = self.b1.clone();
        for i in 0..(k * v) {
            if input[i] == 0.0 {
                continue;
            }
            for j in 0..h {
                h_pre[j] += input[i] * self.w1[i * h + j];
            }
        }

        // ReLU
        let h_post: Vec<f32> = h_pre.iter().map(|&x| x.max(0.0)).collect();

        // Output logits: logits = h_post @ W2 + b2
        let mut logits = self.b2.clone();
        for i in 0..h {
            if h_post[i] == 0.0 {
                continue;
            }
            for j in 0..v {
                logits[j] += h_post[i] * self.w2[i * v + j];
            }
        }

        ForwardCache {
            input,
            h_pre,
            h_post,
            logits,
        }
    }

    /// Backward pass + SGD update.
    /// `target` = ID do próximo token correto.
    /// Retorna loss (cross-entropy).
    pub fn step(&mut self, window: &[u32], target: u32) -> f32 {
        let v = self.cfg.vocab_size;
        let k = self.cfg.window;
        let h = self.cfg.hidden;
        let lr = self.cfg.learning_rate;
        let target = (target as usize).min(v - 1);

        let cache = self.forward(window);

        // Softmax + cross-entropy
        let max_logit = cache.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = cache.logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_probs: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum_probs;
        }
        let loss = -probs[target].max(1e-20).ln();

        // Backprop
        // dL/d_logits = probs - one_hot(target)
        let mut d_logits = probs.clone();
        d_logits[target] -= 1.0;

        // dL/d_W2[i,j] = h_post[i] * d_logits[j]
        // dL/d_b2[j] = d_logits[j]
        // dL/d_h_post[i] = sum_j W2[i,j] * d_logits[j]
        let mut d_h_post = vec![0.0_f32; h];
        for i in 0..h {
            for j in 0..v {
                self.w2[i * v + j] -= lr * cache.h_post[i] * d_logits[j];
                d_h_post[i] += self.w2[i * v + j] * d_logits[j];
            }
        }
        for j in 0..v {
            self.b2[j] -= lr * d_logits[j];
        }

        // dL/d_h_pre[i] = d_h_post[i] * (h_pre[i] > 0 ? 1 : 0)
        let mut d_h_pre = vec![0.0_f32; h];
        for i in 0..h {
            d_h_pre[i] = if cache.h_pre[i] > 0.0 {
                d_h_post[i]
            } else {
                0.0
            };
        }

        // dL/d_W1[i,j] = input[i] * d_h_pre[j]
        // dL/d_b1[j] = d_h_pre[j]
        for i in 0..(k * v) {
            if cache.input[i] == 0.0 {
                continue;
            }
            for j in 0..h {
                self.w1[i * h + j] -= lr * cache.input[i] * d_h_pre[j];
            }
        }
        for j in 0..h {
            self.b1[j] -= lr * d_h_pre[j];
        }

        loss
    }

    /// Predição: argmax do output.
    pub fn predict(&self, window: &[u32]) -> u32 {
        let cache = self.forward(window);
        let mut best = 0;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &x) in cache.logits.iter().enumerate() {
            if x > best_v {
                best_v = x;
                best = i;
            }
        }
        best as u32
    }

    /// Cross-entropy médio em batch.
    pub fn eval_loss(&self, samples: &[(Vec<u32>, u32)]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let v = self.cfg.vocab_size;
        let mut total = 0.0_f32;
        for (window, target) in samples {
            let cache = self.forward(window);
            let max_logit = cache.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = cache.logits.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_exps: f32 = exps.iter().sum();
            let target = (*target as usize).min(v - 1);
            let p = exps[target] / sum_exps;
            total -= p.max(1e-20).ln();
        }
        total / samples.len() as f32
    }

    /// Accuracy em batch.
    pub fn eval_accuracy(&self, samples: &[(Vec<u32>, u32)]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let v = self.cfg.vocab_size;
        let mut correct = 0;
        for (window, target) in samples {
            let pred = self.predict(window);
            let target = (*target as u32).min(v as u32 - 1);
            if pred == target {
                correct += 1;
            }
        }
        correct as f32 / samples.len() as f32
    }

    /// Aloca um buffer de gradientes zerado, dimensionado para esta config.
    pub fn alloc_grads(&self) -> Grads {
        let v = self.cfg.vocab_size;
        let h = self.cfg.hidden;
        let k = self.cfg.window;
        Grads {
            g_w1: vec![0.0f32; k * v * h],
            g_b1: vec![0.0f32; h],
            g_w2: vec![0.0f32; h * v],
            g_b2: vec![0.0f32; v],
            sum_loss: 0.0,
        }
    }

    /// Acumula gradientes do batch DENTRO de um buffer pré-alocado, **zerando-o**
    /// no início. Read-only sobre `self`. Permite reuso de buffer entre batches.
    pub fn compute_grads_into(&self, batch: &[(Vec<u32>, u32)], grads: &mut Grads) {
        let v = self.cfg.vocab_size;
        let h = self.cfg.hidden;

        // zera buffer reusado
        grads.g_w1.fill(0.0);
        grads.g_b1.fill(0.0);
        grads.g_w2.fill(0.0);
        grads.g_b2.fill(0.0);
        grads.sum_loss = 0.0;

        for (window, target) in batch {
            let target = (*target as usize).min(v - 1);
            let cache = self.forward(window);

            // softmax + cross-entropy
            let max_logit = cache.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = cache.logits.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_p: f32 = probs.iter().sum();
            for p in &mut probs { *p /= sum_p; }
            grads.sum_loss -= probs[target].max(1e-20).ln();

            // dL/d_logits = probs - one_hot(target)
            let mut d_logits = probs;
            d_logits[target] -= 1.0;

            // d_h_post = W2 @ d_logits  (calcular ANTES de modificar g_w2)
            let mut d_h_pre = vec![0.0f32; h];
            for i in 0..h {
                if cache.h_pre[i] <= 0.0 { continue; } // ReLU: apenas i com h_pre>0 contribuem
                let mut acc = 0.0f32;
                for j in 0..v {
                    acc += self.w2[i * v + j] * d_logits[j];
                }
                d_h_pre[i] = acc;
            }

            // grads para W2, b2
            for i in 0..h {
                let hp = cache.h_post[i];
                if hp == 0.0 { continue; }
                for j in 0..v {
                    grads.g_w2[i * v + j] += hp * d_logits[j];
                }
            }
            for j in 0..v {
                grads.g_b2[j] += d_logits[j];
            }

            // grads para W1, b1 (input é one-hot esparso → só K posições firingam)
            for (pos, &tok) in window.iter().enumerate() {
                let idx = (tok as usize).min(v - 1);
                let row_off = (pos * v + idx) * h;
                for j in 0..h {
                    grads.g_w1[row_off + j] += d_h_pre[j];
                }
            }
            for j in 0..h {
                grads.g_b1[j] += d_h_pre[j];
            }
        }
    }

    /// Versão owning para conveniência. Aloca novo buffer cada chamada (não use no hot loop).
    #[must_use]
    pub fn compute_grads(&self, batch: &[(Vec<u32>, u32)]) -> Grads {
        let mut g = self.alloc_grads();
        self.compute_grads_into(batch, &mut g);
        g
    }

    /// Aplica gradientes acumulados ao MLP. `scale` permite normalizar pelo batch
    /// (ex.: `lr / batch_size` para mini-batch médio).
    pub fn apply_grads(&mut self, grads: &Grads, scale: f32) {
        for i in 0..self.w1.len() { self.w1[i] -= scale * grads.g_w1[i]; }
        for i in 0..self.b1.len() { self.b1[i] -= scale * grads.g_b1[i]; }
        for i in 0..self.w2.len() { self.w2[i] -= scale * grads.g_w2[i]; }
        for i in 0..self.b2.len() { self.b2[i] -= scale * grads.g_b2[i]; }
    }

    /// Treina 1 época em mini-batch SGD paralelo usando `n_threads`.
    ///
    /// Estratégia:
    /// - itera em batches de `batch_size`
    /// - cada batch é dividido em `n_threads` chunks paralelos
    /// - cada thread computa gradientes locais sobre seu chunk
    /// - main soma e aplica 1 update (LR escalado por 1/batch_size)
    ///
    /// Mantém dinâmica de SGD real (muitos updates por época) com paralelismo
    /// dentro de cada update. `pairs` é assumido já shuffled pelo caller.
    /// Retorna loss médio do epoch.
    pub fn train_epoch_parallel(
        &mut self,
        pairs: &[(Vec<u32>, u32)],
        n_threads: usize,
        batch_size: usize,
    ) -> f32 {
        if pairs.is_empty() { return 0.0; }
        let n_threads = n_threads.max(1);
        let batch_size = batch_size.max(1).min(pairs.len());

        // **Pré-aloca** buffers de gradiente — 1 por thread + 1 acumulador.
        // Reusa entre batches; evita malloc/free no hot loop.
        let mut bufs: Vec<Grads> = (0..n_threads).map(|_| self.alloc_grads()).collect();

        let mut total_loss = 0.0f32;
        let total_count = pairs.len();

        for batch_start in (0..total_count).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_count);
            let batch = &pairs[batch_start..batch_end];
            let actual_threads = n_threads.min(batch.len());
            let chunk_size = batch.len().div_ceil(actual_threads);
            let chunks: Vec<&[(Vec<u32>, u32)]> = batch.chunks(chunk_size).collect();
            let actual_n = chunks.len();
            let mlp_ref = &*self;

            // Cada thread recebe (chunk, &mut buf único). Split exclusivo via iter_mut.
            std::thread::scope(|s| {
                let handles: Vec<_> = chunks
                    .iter()
                    .zip(bufs.iter_mut().take(actual_n))
                    .map(|(chunk, buf)| {
                        s.spawn(move || mlp_ref.compute_grads_into(chunk, buf))
                    })
                    .collect();
                for h in handles { h.join().expect("thread panicked"); }
            });

            // Soma deltas no primeiro buffer (em vez de alocar novo) usando split_first_mut
            // para borrow checker aceitar &mut[0] + & dos resto.
            let scale = self.cfg.learning_rate / batch.len() as f32;
            let (head, tail) = bufs.split_first_mut().expect("bufs empty");
            for r in tail.iter().take(actual_n.saturating_sub(1)) {
                for i in 0..head.g_w1.len() { head.g_w1[i] += r.g_w1[i]; }
                for i in 0..head.g_b1.len() { head.g_b1[i] += r.g_b1[i]; }
                for i in 0..head.g_w2.len() { head.g_w2[i] += r.g_w2[i]; }
                for i in 0..head.g_b2.len() { head.g_b2[i] += r.g_b2[i]; }
                head.sum_loss += r.sum_loss;
            }
            total_loss += head.sum_loss;
            self.apply_grads(head, scale);
            // (buffers serão zerados na próxima compute_grads_into; sum_loss reset implícito)
        }

        total_loss / total_count as f32
    }
}

/// Gradientes acumulados (resultado de `compute_grads`).
#[derive(Debug, Clone)]
pub struct Grads {
    pub g_w1: Vec<f32>,
    pub g_b1: Vec<f32>,
    pub g_w2: Vec<f32>,
    pub g_b2: Vec<f32>,
    pub sum_loss: f32,
}

#[derive(Debug)]
pub struct ForwardCache {
    pub input: Vec<f32>,
    pub h_pre: Vec<f32>,
    pub h_post: Vec<f32>,
    pub logits: Vec<f32>,
}

/// Gera (window, target) de um stream de tokens.
pub fn build_training_pairs(streams: &[Vec<u32>], window: usize) -> Vec<(Vec<u32>, u32)> {
    let mut pairs = Vec::new();
    for seq in streams {
        if seq.len() <= window {
            continue;
        }
        for i in 0..seq.len() - window {
            let w = seq[i..i + window].to_vec();
            let target = seq[i + window];
            pairs.push((w, target));
        }
    }
    pairs
}

/// Mapeia stream de tokens arbitrários para IDs compactos 0..N-1.
/// Retorna (mapped_streams, vocab_size).
pub fn compact_token_ids(streams: &[Vec<u32>]) -> (Vec<Vec<u32>>, usize) {
    use std::collections::HashMap;
    let mut id_map: HashMap<u32, u32> = HashMap::new();
    for seq in streams {
        for &tok in seq {
            let next_id = id_map.len() as u32;
            id_map.entry(tok).or_insert(next_id);
        }
    }
    let mapped: Vec<Vec<u32>> = streams
        .iter()
        .map(|s| s.iter().map(|t| id_map[t]).collect())
        .collect();
    (mapped, id_map.len())
}

/// Versão para byte streams (u8 → u32 IDs compactos).
pub fn compact_from_bytes(streams: &[Vec<u8>]) -> (Vec<Vec<u32>>, usize) {
    let as_u32: Vec<Vec<u32>> = streams
        .iter()
        .map(|s| s.iter().map(|&b| u32::from(b)).collect())
        .collect();
    compact_token_ids(&as_u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_forward_shapes() {
        let cfg = MlpConfig {
            vocab_size: 8,
            window: 3,
            hidden: 4,
            learning_rate: 0.01,
            seed: 1,
        };
        let mlp = Mlp::new(cfg);
        let cache = mlp.forward(&[0, 1, 2]);
        assert_eq!(cache.logits.len(), 8);
        assert_eq!(cache.h_pre.len(), 4);
    }

    #[test]
    fn mlp_can_overfit_single_pair() {
        // MLP deve aprender a memorizar 1 par de treino
        let cfg = MlpConfig {
            vocab_size: 4,
            window: 2,
            hidden: 8,
            learning_rate: 0.1,
            seed: 42,
        };
        let mut mlp = Mlp::new(cfg);
        let window = vec![0, 1];
        let target = 2;

        let initial_pred = mlp.predict(&window);
        for _ in 0..200 {
            mlp.step(&window, target);
        }
        let final_pred = mlp.predict(&window);

        // Com 200 steps em LR=0.1, deve ter aprendido (mesmo se inicial estava certo)
        let loss_after = mlp.step(&window, target);
        assert!(loss_after < 0.1, "loss after = {} (initial pred {} → final pred {})",
            loss_after, initial_pred, final_pred);
    }

    #[test]
    fn compact_token_ids_works() {
        let streams = vec![vec![100_u32, 200, 100], vec![300, 200]];
        let (mapped, n) = compact_token_ids(&streams);
        assert_eq!(n, 3); // 100, 200, 300 → IDs 0, 1, 2
        assert_eq!(mapped[0].len(), 3);
        // First occurrence of 100 → id 0, then 200 → 1, then 100 → 0 again
        assert_eq!(mapped[0][0], mapped[0][2]);
    }
}
