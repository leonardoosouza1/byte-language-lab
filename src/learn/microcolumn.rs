//! # Soft Microcolumn Grid (SMG) — implementação para exp17.
//!
//! Arquitetura comitada em `docs/exp17-design.md`.
//!
//! - **FSM**: estado K-dim contínuo, atualizado por
//!   `softmax(state @ T_f + I_f[byte] + b_f + pos_bias)`.
//! - **Cell (microcoluna)**: 8 FSMs paralelas, uma por família.
//! - **Grid**: 4×4 = 16 cells, todas veem mesmo input por tick, distinguidas
//!   apenas por `pos_bias` learnable.
//! - **Decoder**: flatten do estado final → linear → logits sobre vocab.
//!
//! Treino: BPTT clássico através dos ticks. Discretização final fica para exp18+.

use crate::Rng;

/// Dimensão do estado interno por FSM.
pub const K: usize = 16;
/// Número de famílias FSM (1 por slot do microcell).
pub const N_FAMILIES: usize = 8;
/// FSMs por cell = 1 por família.
pub const FSMS_PER_CELL: usize = N_FAMILIES;
/// Cells na grade (4×4).
pub const N_CELLS: usize = 16;
/// Dimensão do vetor achatado para o decoder.
pub const FLAT_DIM: usize = N_CELLS * FSMS_PER_CELL * K;

/// Configuração da grade.
#[derive(Debug, Clone)]
pub struct GridConfig {
    /// Tamanho do vocab compacto (igual ao do MLP em exp16).
    pub vocab_size: usize,
    /// Janela de bytes por pair de treino.
    pub window: usize,
    /// Learning rate.
    pub learning_rate: f32,
    /// Seed determinístico.
    pub seed: u64,
}

/// Grade SMG com 16 microcolunas × 8 FSMs.
pub struct SmgGrid {
    /// Configuração.
    pub cfg: GridConfig,
    /// Recorrência por família. Shape `[N_FAMILIES, K, K]` row-major em (família, j, k).
    t_f: Vec<f32>,
    /// Embedding de input por família. Shape `[N_FAMILIES, vocab, K]`.
    i_f: Vec<f32>,
    /// Viés por família. Shape `[N_FAMILIES, K]`.
    b_f: Vec<f32>,
    /// Viés por cell (quebra simetria). Shape `[N_CELLS, K]`.
    pos_bias: Vec<f32>,
    /// Decoder linear. Shape `[FLAT_DIM, vocab]` (out-major: `w_out[i * V + j]`).
    w_out: Vec<f32>,
    /// Viés do decoder. Shape `[vocab]`.
    b_out: Vec<f32>,
}

/// Cache completo da forward para BPTT.
#[derive(Debug, Clone)]
pub struct ForwardCache {
    /// Estados em todos os ticks. Shape `[(T+1), N_CELLS, FSMS_PER_CELL, K]`.
    /// `states[0]` é o estado inicial (uniforme `1/K`).
    /// `states[T]` é o estado final usado pelo decoder.
    pub states: Vec<f32>,
    /// Logits do decoder. Shape `[vocab]`.
    pub logits: Vec<f32>,
    /// Janela vista (cópia para conveniência do backward).
    pub window: Vec<u32>,
}

/// Gradientes acumulados sobre um batch.
#[derive(Debug, Clone)]
pub struct Grads {
    /// dL/dT_f. Shape igual a `t_f`.
    pub g_t_f: Vec<f32>,
    /// dL/dI_f.
    pub g_i_f: Vec<f32>,
    /// dL/db_f.
    pub g_b_f: Vec<f32>,
    /// dL/d pos_bias.
    pub g_pos_bias: Vec<f32>,
    /// dL/dW_out.
    pub g_w_out: Vec<f32>,
    /// dL/db_out.
    pub g_b_out: Vec<f32>,
    /// Soma das losses (cross-entropy) sobre o batch.
    pub sum_loss: f32,
}

impl SmgGrid {
    /// Inicializa pesos com Xavier-uniforme + viés zero.
    #[must_use]
    pub fn new(cfg: GridConfig) -> Self {
        let mut rng = Rng::new(cfg.seed);
        let v = cfg.vocab_size;
        let s_t = (1.0 / K as f32).sqrt();
        let s_i = (1.0 / v as f32).sqrt();
        let s_o = (1.0 / FLAT_DIM as f32).sqrt();

        let mut sample = |scale: f32| (rng.next_f64() as f32 - 0.5) * 2.0 * scale;

        let t_f: Vec<f32> = (0..N_FAMILIES * K * K).map(|_| sample(s_t)).collect();
        let i_f: Vec<f32> = (0..N_FAMILIES * v * K).map(|_| sample(s_i)).collect();
        let b_f = vec![0.0_f32; N_FAMILIES * K];
        // pos_bias pequeno mas diferente por cell (essencial para quebrar simetria)
        let pos_bias: Vec<f32> = (0..N_CELLS * K).map(|_| sample(0.1)).collect();
        let w_out: Vec<f32> = (0..FLAT_DIM * v).map(|_| sample(s_o)).collect();
        let b_out = vec![0.0_f32; v];

        Self {
            cfg,
            t_f,
            i_f,
            b_f,
            pos_bias,
            w_out,
            b_out,
        }
    }

    /// Total de parâmetros learnable (debug/report).
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.t_f.len()
            + self.i_f.len()
            + self.b_f.len()
            + self.pos_bias.len()
            + self.w_out.len()
            + self.b_out.len()
    }

    /// Forward com cache completo para BPTT. `window` em compact ids.
    pub fn forward(&self, window: &[u32]) -> ForwardCache {
        let v = self.cfg.vocab_size;
        let t_steps = window.len();
        let stride_step = N_CELLS * FSMS_PER_CELL * K;
        let init_val = 1.0 / K as f32;

        let mut states = vec![init_val; (t_steps + 1) * stride_step];

        // pré-aloca buffer para pre_act/softmax tmp
        let mut pre = [0.0_f32; K];

        for step in 0..t_steps {
            let input_byte = (window[step] as usize).min(v.saturating_sub(1));
            let s_in = step * stride_step;
            let s_out = (step + 1) * stride_step;

            for cell in 0..N_CELLS {
                let pos_off = cell * K;
                for fsm in 0..FSMS_PER_CELL {
                    let f = fsm; // família = índice do FSM
                    let in_off = s_in + (cell * FSMS_PER_CELL + fsm) * K;
                    let out_off = s_out + (cell * FSMS_PER_CELL + fsm) * K;

                    let t_off = f * K * K;
                    let i_off = (f * v + input_byte) * K;
                    let b_off = f * K;

                    // pre[k] = I[byte, k] + b[k] + pos[k] + sum_j state_in[j] * T[j, k]
                    for k in 0..K {
                        pre[k] = self.i_f[i_off + k]
                            + self.b_f[b_off + k]
                            + self.pos_bias[pos_off + k];
                    }
                    for j in 0..K {
                        let s = states[in_off + j];
                        if s == 0.0 {
                            continue;
                        }
                        for k in 0..K {
                            pre[k] += s * self.t_f[t_off + j * K + k];
                        }
                    }
                    // softmax → states[out_off..]
                    let mx = pre.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0_f32;
                    for k in 0..K {
                        let e = (pre[k] - mx).exp();
                        states[out_off + k] = e;
                        sum += e;
                    }
                    let inv = 1.0 / sum;
                    for k in 0..K {
                        states[out_off + k] *= inv;
                    }
                }
            }
        }

        // Decoder: logits[j] = b_out[j] + sum_i flat[i] * w_out[i * V + j]
        let final_off = t_steps * stride_step;
        let mut logits = self.b_out.clone();
        for i in 0..FLAT_DIM {
            let s = states[final_off + i];
            if s == 0.0 {
                continue;
            }
            let row_off = i * v;
            for j in 0..v {
                logits[j] += s * self.w_out[row_off + j];
            }
        }

        ForwardCache {
            states,
            logits,
            window: window.to_vec(),
        }
    }

    /// Predição argmax (avaliação).
    #[must_use]
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
            let mx = cache.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = cache.logits.iter().map(|&x| (x - mx).exp()).collect();
            let s: f32 = exps.iter().sum();
            let t = (*target as usize).min(v - 1);
            let p = exps[t] / s;
            total -= p.max(1e-20).ln();
        }
        total / samples.len() as f32
    }

    /// Accuracy em batch (next-byte).
    pub fn eval_accuracy(&self, samples: &[(Vec<u32>, u32)]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let v = self.cfg.vocab_size;
        let mut correct = 0;
        for (window, target) in samples {
            let pred = self.predict(window);
            let t = (*target as u32).min(v as u32 - 1);
            if pred == t {
                correct += 1;
            }
        }
        correct as f32 / samples.len() as f32
    }

    /// Aloca buffer de gradientes zerado.
    #[must_use]
    pub fn alloc_grads(&self) -> Grads {
        Grads {
            g_t_f: vec![0.0; self.t_f.len()],
            g_i_f: vec![0.0; self.i_f.len()],
            g_b_f: vec![0.0; self.b_f.len()],
            g_pos_bias: vec![0.0; self.pos_bias.len()],
            g_w_out: vec![0.0; self.w_out.len()],
            g_b_out: vec![0.0; self.b_out.len()],
            sum_loss: 0.0,
        }
    }

    /// Computa gradientes do batch acumulados em `grads` (zera no início).
    /// Read-only sobre `self`. Permite reuso de buffer.
    pub fn compute_grads_into(&self, batch: &[(Vec<u32>, u32)], grads: &mut Grads) {
        let v = self.cfg.vocab_size;
        let stride_step = N_CELLS * FSMS_PER_CELL * K;

        // Zera buffers reusados
        grads.g_t_f.fill(0.0);
        grads.g_i_f.fill(0.0);
        grads.g_b_f.fill(0.0);
        grads.g_pos_bias.fill(0.0);
        grads.g_w_out.fill(0.0);
        grads.g_b_out.fill(0.0);
        grads.sum_loss = 0.0;

        for (window, target) in batch {
            let target = (*target as usize).min(v - 1);
            let cache = self.forward(window);
            let t_steps = window.len();

            // softmax + cross-entropy
            let mx = cache.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = cache.logits.iter().map(|&x| (x - mx).exp()).collect();
            let s_p: f32 = probs.iter().sum();
            for p in &mut probs {
                *p /= s_p;
            }
            grads.sum_loss -= probs[target].max(1e-20).ln();

            // dL/d_logits = probs - one_hot(target)
            let mut d_logits = probs;
            d_logits[target] -= 1.0;

            // ── Decoder backward ────────────────────────────────────────────
            // dL/d_w_out[i, j] = flat[i] * d_logits[j]
            // dL/d_b_out[j]    = d_logits[j]
            // dL/d_flat[i]     = sum_j w_out[i, j] * d_logits[j]
            let final_off = t_steps * stride_step;
            let flat = &cache.states[final_off..final_off + FLAT_DIM];
            let mut d_flat = vec![0.0_f32; FLAT_DIM];
            for i in 0..FLAT_DIM {
                let s = flat[i];
                let row_off = i * v;
                for j in 0..v {
                    grads.g_w_out[row_off + j] += s * d_logits[j];
                    d_flat[i] += self.w_out[row_off + j] * d_logits[j];
                }
            }
            for j in 0..v {
                grads.g_b_out[j] += d_logits[j];
            }

            // ── BPTT: passa por todos os ticks de trás para frente ───────────
            // d_state_next[N_CELLS, FSMS_PER_CELL, K] = derivada do estado
            // pós-update no tick step (= pré-update no tick step+1).
            let mut d_state_next: Vec<f32> = d_flat;

            for step in (0..t_steps).rev() {
                let input_byte = (window[step] as usize).min(v - 1);
                let s_in_off = step * stride_step;
                let s_out_off = (step + 1) * stride_step;

                let mut d_state_prev = vec![0.0_f32; stride_step];

                for cell in 0..N_CELLS {
                    let pos_off = cell * K;
                    for fsm in 0..FSMS_PER_CELL {
                        let f = fsm;
                        let in_off = s_in_off + (cell * FSMS_PER_CELL + fsm) * K;
                        let out_off = s_out_off + (cell * FSMS_PER_CELL + fsm) * K;
                        let d_off = (cell * FSMS_PER_CELL + fsm) * K;

                        let t_off = f * K * K;
                        let i_off = (f * v + input_byte) * K;
                        let b_off = f * K;

                        // softmax backward
                        // s = state[out], ds_in = d_state_next[d_off..]
                        // d_pre[k] = s[k] * (ds_in[k] - <s, ds_in>)
                        let s: &[f32] = &cache.states[out_off..out_off + K];
                        let dsn: &[f32] = &d_state_next[d_off..d_off + K];
                        let mut sd = 0.0_f32;
                        for k in 0..K {
                            sd += s[k] * dsn[k];
                        }
                        let mut d_pre = [0.0_f32; K];
                        for k in 0..K {
                            d_pre[k] = s[k] * (dsn[k] - sd);
                        }

                        // grads para I_f, b_f, pos_bias
                        for k in 0..K {
                            grads.g_i_f[i_off + k] += d_pre[k];
                            grads.g_b_f[b_off + k] += d_pre[k];
                            grads.g_pos_bias[pos_off + k] += d_pre[k];
                        }

                        // grads para T_f e d_state_prev
                        // pre[k] = ... + sum_j state_in[j] * T[j, k]
                        // dT[j, k] += state_in[j] * d_pre[k]
                        // d_state_prev[j] += sum_k T[j, k] * d_pre[k]
                        let state_in: &[f32] = &cache.states[in_off..in_off + K];
                        for j in 0..K {
                            let sj = state_in[j];
                            let row_off = t_off + j * K;
                            let mut acc = 0.0_f32;
                            for k in 0..K {
                                grads.g_t_f[row_off + k] += sj * d_pre[k];
                                acc += self.t_f[row_off + k] * d_pre[k];
                            }
                            d_state_prev[d_off + j] += acc;
                        }
                    }
                }

                d_state_next = d_state_prev;
            }
            // d_state_next agora referente a state[0] (estado inicial uniforme).
            // Como state[0] não é learnable, descartamos.
        }
    }

    /// Aplica gradientes acumulados. `scale` permite normalizar pelo batch.
    pub fn apply_grads(&mut self, grads: &Grads, scale: f32) {
        for i in 0..self.t_f.len() {
            self.t_f[i] -= scale * grads.g_t_f[i];
        }
        for i in 0..self.i_f.len() {
            self.i_f[i] -= scale * grads.g_i_f[i];
        }
        for i in 0..self.b_f.len() {
            self.b_f[i] -= scale * grads.g_b_f[i];
        }
        for i in 0..self.pos_bias.len() {
            self.pos_bias[i] -= scale * grads.g_pos_bias[i];
        }
        for i in 0..self.w_out.len() {
            self.w_out[i] -= scale * grads.g_w_out[i];
        }
        for i in 0..self.b_out.len() {
            self.b_out[i] -= scale * grads.g_b_out[i];
        }
    }

    /// Treina 1 época em mini-batch SGD paralelo (mesmo padrão do MLP).
    /// `pairs` assumido shuffled pelo caller.
    pub fn train_epoch_parallel(
        &mut self,
        pairs: &[(Vec<u32>, u32)],
        n_threads: usize,
        batch_size: usize,
    ) -> f32 {
        if pairs.is_empty() {
            return 0.0;
        }
        let n_threads = n_threads.max(1);
        let batch_size = batch_size.max(1).min(pairs.len());

        let mut bufs: Vec<Grads> = (0..n_threads).map(|_| self.alloc_grads()).collect();

        let mut total_loss = 0.0_f32;
        let total_count = pairs.len();

        for batch_start in (0..total_count).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_count);
            let batch = &pairs[batch_start..batch_end];
            let actual_threads = n_threads.min(batch.len());
            let chunk_size = batch.len().div_ceil(actual_threads);
            let chunks: Vec<&[(Vec<u32>, u32)]> = batch.chunks(chunk_size).collect();
            let actual_n = chunks.len();
            let grid_ref = &*self;

            std::thread::scope(|s| {
                let handles: Vec<_> = chunks
                    .iter()
                    .zip(bufs.iter_mut().take(actual_n))
                    .map(|(chunk, buf)| {
                        s.spawn(move || grid_ref.compute_grads_into(chunk, buf))
                    })
                    .collect();
                for h in handles {
                    h.join().expect("thread panicked");
                }
            });

            let scale = self.cfg.learning_rate / batch.len() as f32;
            let (head, tail) = bufs.split_first_mut().expect("bufs empty");
            for r in tail.iter().take(actual_n.saturating_sub(1)) {
                for i in 0..head.g_t_f.len() {
                    head.g_t_f[i] += r.g_t_f[i];
                }
                for i in 0..head.g_i_f.len() {
                    head.g_i_f[i] += r.g_i_f[i];
                }
                for i in 0..head.g_b_f.len() {
                    head.g_b_f[i] += r.g_b_f[i];
                }
                for i in 0..head.g_pos_bias.len() {
                    head.g_pos_bias[i] += r.g_pos_bias[i];
                }
                for i in 0..head.g_w_out.len() {
                    head.g_w_out[i] += r.g_w_out[i];
                }
                for i in 0..head.g_b_out.len() {
                    head.g_b_out[i] += r.g_b_out[i];
                }
                head.sum_loss += r.sum_loss;
            }
            total_loss += head.sum_loss;
            self.apply_grads(head, scale);
        }

        total_loss / total_count as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GridConfig {
        GridConfig {
            vocab_size: 8,
            window: 4,
            learning_rate: 0.05,
            seed: 42,
        }
    }

    #[test]
    fn forward_shapes_and_simplex() {
        let g = SmgGrid::new(cfg());
        let cache = g.forward(&[0, 1, 2, 3]);
        assert_eq!(cache.logits.len(), 8);
        // states tem (T+1) blocos de stride
        let stride = N_CELLS * FSMS_PER_CELL * K;
        assert_eq!(cache.states.len(), 5 * stride);
        // Cada FSM final deve estar no simplex (soma ≈ 1)
        let final_off = 4 * stride;
        for cell in 0..N_CELLS {
            for fsm in 0..FSMS_PER_CELL {
                let off = final_off + (cell * FSMS_PER_CELL + fsm) * K;
                let s: f32 = cache.states[off..off + K].iter().sum();
                assert!((s - 1.0).abs() < 1e-4, "FSM state não está no simplex: soma={s}");
            }
        }
    }

    #[test]
    fn can_overfit_single_pair() {
        let mut cfg = cfg();
        cfg.window = 4;
        let mut grid = SmgGrid::new(cfg);
        let window = vec![0_u32, 1, 2, 3];
        let target = 5_u32;

        let initial_loss = grid.eval_loss(&[(window.clone(), target)]);
        let mut buf = grid.alloc_grads();
        for _ in 0..200 {
            grid.compute_grads_into(&[(window.clone(), target)], &mut buf);
            grid.apply_grads(&buf, 0.1);
        }
        let final_loss = grid.eval_loss(&[(window.clone(), target)]);
        assert!(
            final_loss < initial_loss * 0.3,
            "overfit falhou: initial={initial_loss}, final={final_loss}"
        );
    }

    #[test]
    fn n_params_match_design_doc() {
        let g = SmgGrid::new(GridConfig {
            vocab_size: 41,
            window: 48,
            learning_rate: 0.05,
            seed: 1,
        });
        // Conforme docs/exp17-design.md §2.5
        // T: 8 × 16 × 16 = 2048
        // I: 8 × 41 × 16 = 5248
        // b: 8 × 16 = 128
        // pos: 16 × 16 = 256
        // W_out: 2048 × 41 = 83968
        // b_out: 41
        // total = 91 689
        assert_eq!(g.n_params(), 91_689);
    }
}
