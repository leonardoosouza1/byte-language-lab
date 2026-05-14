//! Curator — fast router brain (dual-brain architecture).
//!
//! Pequeno classifier que decide rota da query antes de pipeline pesado:
//!   query HV (1024 bit) → [expert_id, confidence, need_qwen]
//!
//! Math fundamento:
//!   - Information Bottleneck (Tishby 2000): T = f(X) compressão útil
//!   - Linear classifier: y = W·x + b, treinado online
//!   - Confidence via sigmoid; expert via softmax
//!
//! Init strategy:
//!   - Pesos iniciais = correlation entre HV bits e expert centroid HV
//!   - Sem necessidade de treinamento offline imediato
//!   - Online learning via observação routing decisions reais
//!
//! Performance:
//!   - 1024 inputs × 12 outputs = 12K FLOPs/query
//!   - <0.01ms latency típica
//!   - Memory: ~50KB (negligible)

#![allow(dead_code)]

pub const HV_BITS: usize = 1024;

/// Routing output do curator.
#[derive(Debug, Clone, Default)]
pub struct CuratorOutput {
    /// Logits per expert (10 logits, antes de softmax).
    pub expert_logits: Vec<f32>,
    /// Softmax sobre expert_logits (sum = 1.0).
    pub expert_probs: Vec<f32>,
    /// Top expert index.
    pub top_expert: usize,
    /// Confidence score [0, 1].
    pub confidence: f32,
    /// Escalation flag [0, 1]: how likely query needs Qwen.
    pub need_qwen: f32,
}

/// Curator router — linear classifier sobre byte HV.
pub struct Curator {
    pub n_experts: usize,
    /// Weights shape: [n_experts + 2][HV_BITS]
    /// Last 2 outputs = (confidence, need_qwen)
    pub weights: Vec<f32>,
    /// Biases shape: [n_experts + 2]
    pub bias: Vec<f32>,
    /// Hits counter pra observar uso.
    pub n_routes: u64,
}

impl Curator {
    /// Cria curator vazio (zero weights). Use `init_from_centroids` pra
    /// inicializar com sentido semântico.
    pub fn new(n_experts: usize) -> Self {
        let n_out = n_experts + 2;  // +confidence +need_qwen
        Self {
            n_experts,
            weights: vec![0.0; n_out * HV_BITS],
            bias: vec![0.0; n_out],
            n_routes: 0,
        }
    }

    /// Inicializa weights a partir dos centroid HVs de cada expert.
    /// Cada bit i do centroid expert_e vira peso W[e][i] = bit_value * 2 - 1.
    /// Bits comuns a múltiplos experts terão weights baixos (média = 0).
    pub fn init_from_centroids(&mut self, centroids: &[[u64; 16]]) {
        assert_eq!(centroids.len(), self.n_experts, "centroids count mismatch");
        let n_out = self.n_experts + 2;
        self.weights = vec![0.0; n_out * HV_BITS];
        for (e_idx, centroid) in centroids.iter().enumerate() {
            for bit_i in 0..HV_BITS {
                let word_i = bit_i / 64;
                let bit_pos = bit_i % 64;
                let bit_val = ((centroid[word_i] >> bit_pos) & 1) as f32;
                // bit 1 → +1, bit 0 → -1
                self.weights[e_idx * HV_BITS + bit_i] = bit_val * 2.0 - 1.0;
            }
            self.bias[e_idx] = 0.0;
        }
        // Confidence + need_qwen heads remain zero initially — learned online
        self.bias[self.n_experts] = 0.5;     // confidence default 0.5
        self.bias[self.n_experts + 1] = 0.5; // need_qwen default 0.5
    }

    /// Forward: HV → routing output.
    pub fn route(&mut self, query_hv: &[u64; 16]) -> CuratorOutput {
        self.n_routes += 1;
        let n_out = self.n_experts + 2;
        let mut logits = vec![0.0f32; n_out];

        // Convert HV bits → ±1 f32, then dot product per output head
        for bit_i in 0..HV_BITS {
            let word_i = bit_i / 64;
            let bit_pos = bit_i % 64;
            let bit_val = ((query_hv[word_i] >> bit_pos) & 1) as f32;
            let x = bit_val * 2.0 - 1.0; // {-1, +1}
            for o in 0..n_out {
                logits[o] += x * self.weights[o * HV_BITS + bit_i];
            }
        }
        for o in 0..n_out {
            logits[o] += self.bias[o];
        }

        // Scale logits pra evitar one-hot collapse (temperature softmax)
        let scale = 1.0 / (HV_BITS as f32).sqrt();
        let expert_logits: Vec<f32> = logits[..self.n_experts].iter().map(|x| x * scale).collect();
        let expert_probs = softmax(&expert_logits);
        let top_expert = argmax(&expert_logits);

        // Normalize confidence + need_qwen via sigmoid
        // (scaled by 1/sqrt(HV_BITS) to avoid saturation)
        let scale = 1.0 / (HV_BITS as f32).sqrt();
        let confidence = sigmoid(logits[self.n_experts] * scale);
        let need_qwen = sigmoid(logits[self.n_experts + 1] * scale);

        CuratorOutput {
            expert_logits,
            expert_probs,
            top_expert,
            confidence,
            need_qwen,
        }
    }

    /// Online update via observation of (query, actual_outcome).
    /// outcome.expert_used: which expert delivered the answer
    /// outcome.qwen_called: whether brain had to escalate
    /// Uses simple perceptron-like update.
    pub fn observe(&mut self, query_hv: &[u64; 16], outcome: &Outcome, lr: f32) {
        let out = {
            let mut clone = Curator {
                n_experts: self.n_experts,
                weights: self.weights.clone(),
                bias: self.bias.clone(),
                n_routes: 0,
            };
            clone.route(query_hv)
        };

        // Update expert head: increase weight of correct expert, decrease others
        for e in 0..self.n_experts {
            let target = if e == outcome.expert_used { 1.0 } else { 0.0 };
            let pred = out.expert_probs[e];
            let err = target - pred;
            for bit_i in 0..HV_BITS {
                let word_i = bit_i / 64;
                let bit_pos = bit_i % 64;
                let x = ((query_hv[word_i] >> bit_pos) & 1) as f32 * 2.0 - 1.0;
                self.weights[e * HV_BITS + bit_i] += lr * err * x;
            }
            self.bias[e] += lr * err;
        }

        // Update confidence head: target = 1 - qwen_called (high when brain answered)
        let conf_target = if outcome.qwen_called { 0.1 } else { 0.9 };
        let conf_err = conf_target - out.confidence;
        for bit_i in 0..HV_BITS {
            let word_i = bit_i / 64;
            let bit_pos = bit_i % 64;
            let x = ((query_hv[word_i] >> bit_pos) & 1) as f32 * 2.0 - 1.0;
            self.weights[self.n_experts * HV_BITS + bit_i] += lr * conf_err * x;
        }
        self.bias[self.n_experts] += lr * conf_err;

        // Update need_qwen head: target = qwen_called
        let qwen_target = if outcome.qwen_called { 0.9 } else { 0.1 };
        let qwen_err = qwen_target - out.need_qwen;
        for bit_i in 0..HV_BITS {
            let word_i = bit_i / 64;
            let bit_pos = bit_i % 64;
            let x = ((query_hv[word_i] >> bit_pos) & 1) as f32 * 2.0 - 1.0;
            self.weights[(self.n_experts + 1) * HV_BITS + bit_i] += lr * qwen_err * x;
        }
        self.bias[self.n_experts + 1] += lr * qwen_err;
    }
}

#[derive(Debug, Clone)]
pub struct Outcome {
    pub expert_used: usize,
    pub qwen_called: bool,
}

impl Curator {
    /// Save weights binary to file.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(&(self.n_experts as u32).to_le_bytes())?;
        f.write_all(&(self.weights.len() as u32).to_le_bytes())?;
        for w in &self.weights {
            f.write_all(&w.to_le_bytes())?;
        }
        f.write_all(&(self.bias.len() as u32).to_le_bytes())?;
        for b in &self.bias {
            f.write_all(&b.to_le_bytes())?;
        }
        Ok(())
    }

    /// Load weights binary from file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let n_experts = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let n_weights = u32::from_le_bytes(buf4) as usize;
        let mut weights = Vec::with_capacity(n_weights);
        for _ in 0..n_weights {
            f.read_exact(&mut buf4)?;
            weights.push(f32::from_le_bytes(buf4));
        }
        f.read_exact(&mut buf4)?;
        let n_bias = u32::from_le_bytes(buf4) as usize;
        let mut bias = Vec::with_capacity(n_bias);
        for _ in 0..n_bias {
            f.read_exact(&mut buf4)?;
            bias.push(f32::from_le_bytes(buf4));
        }
        Ok(Self { n_experts, weights, bias, n_routes: 0 })
    }

    /// Batch train: run N epochs over labeled (hv, expert, qwen_called) tuples.
    pub fn train_batch(&mut self, samples: &[(&[u64; 16], Outcome)], n_epochs: usize, lr: f32) -> f32 {
        let mut final_loss = 0.0f32;
        for _ in 0..n_epochs {
            let mut epoch_loss = 0.0f32;
            for (hv, outcome) in samples {
                let out = self.route(hv);
                let expert_loss = -out.expert_probs[outcome.expert_used].max(1e-9).ln();
                epoch_loss += expert_loss;
                self.observe(hv, outcome, lr);
            }
            final_loss = epoch_loss / samples.len().max(1) as f32;
        }
        final_loss
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|&v| v / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_hv(bits_set: &[usize]) -> [u64; 16] {
        let mut hv = [0u64; 16];
        for &bit in bits_set {
            if bit < 1024 {
                hv[bit / 64] |= 1u64 << (bit % 64);
            }
        }
        hv
    }

    #[test]
    fn new_curator_zero_routes() {
        let c = Curator::new(10);
        assert_eq!(c.n_experts, 10);
        assert_eq!(c.n_routes, 0);
    }

    #[test]
    fn init_from_centroids_sets_weights() {
        let mut c = Curator::new(2);
        let centroids = vec![
            make_test_hv(&[0, 1, 2]),
            make_test_hv(&[100, 200, 300]),
        ];
        c.init_from_centroids(&centroids);
        // Weight for expert 0, bit 0 should be +1 (bit set in centroid 0)
        assert!((c.weights[0] - 1.0).abs() < 0.01);
        // Weight for expert 0, bit 100 should be -1 (bit NOT set in centroid 0)
        assert!((c.weights[100] - (-1.0)).abs() < 0.01);
        // Weight for expert 1, bit 100 should be +1
        assert!((c.weights[1024 + 100] - 1.0).abs() < 0.01);
    }

    #[test]
    fn route_picks_closest_expert() {
        let mut c = Curator::new(2);
        let centroids = vec![
            make_test_hv(&(0..512).collect::<Vec<_>>()),     // first half
            make_test_hv(&(512..1024).collect::<Vec<_>>()),  // second half
        ];
        c.init_from_centroids(&centroids);
        // Query similar to centroid 0
        let q1 = make_test_hv(&(0..400).collect::<Vec<_>>());
        let o1 = c.route(&q1);
        assert_eq!(o1.top_expert, 0);
        // Query similar to centroid 1
        let q2 = make_test_hv(&(600..1024).collect::<Vec<_>>());
        let o2 = c.route(&q2);
        assert_eq!(o2.top_expert, 1);
    }

    #[test]
    fn online_learning_improves() {
        let mut c = Curator::new(2);
        let q = make_test_hv(&(0..400).collect::<Vec<_>>());
        let initial = c.route(&q);
        // Observe many times that this query should go to expert 0
        for _ in 0..50 {
            c.observe(&q, &Outcome { expert_used: 0, qwen_called: false }, 0.01);
        }
        let learned = c.route(&q);
        // After learning, expert 0 prob should be higher
        assert!(learned.expert_probs[0] > initial.expert_probs[0],
            "learning should increase prob of expert 0: {} → {}",
            initial.expert_probs[0], learned.expert_probs[0]);
    }
}
