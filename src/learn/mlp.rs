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
