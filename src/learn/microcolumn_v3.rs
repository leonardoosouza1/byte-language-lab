//! # Soft Microcolumn Grid v3 — v2 + bus lateral toroidal.
//!
//! v2 (`microcolumn_v2.rs`) achou train byte-agreement 49.5%, test 33.9% vs MLP
//! 38.5%. **Estrutura por si só foi neutra**. Hipótese para v3: cada cell estava
//! isolada, sem partilha de padrões — adicionar **bus lateral** entre cells
//! vizinhas no grid 4×4 deve permitir que padrões se propaguem espacialmente
//! e (a) acelere memorização (b) melhore generalização por reuso.
//!
//! Mudanças vs v2:
//! - **Lateral bus 4-connected toroidal**: cada cell `c` recebe contribuição das
//!   4 vizinhas (cima/baixo/esq/dir) na pre-ativação. Por família, há matriz
//!   `W_lat[f]: [K, K]` que projeta estado vizinho para pre-act.
//!   `pre[c,f,k] += sum_{n∈neigh(c)} sum_j W_lat[f,j,k] * state_in[n,f,j]`
//! - Tudo o mais idêntico a v2 (leaky integrator α=0.1, K=32, 8 famílias).
//!
//! Custo de params: `8 * 32 * 32 = 8192` extras. Total esperado ~195K.

use crate::Rng;

/// Dimensão do estado interno por FSM.
pub const K: usize = 32;
/// Famílias FSM (1 por slot).
pub const N_FAMILIES: usize = 8;
/// FSMs por cell.
pub const FSMS_PER_CELL: usize = N_FAMILIES;
/// Cells na grade.
pub const N_CELLS: usize = 16; // 4×4
/// Dimensão do vetor achatado para o decoder.
pub const FLAT_DIM: usize = N_CELLS * FSMS_PER_CELL * K;
/// Coeficiente do leaky integrator (`state' = (1-α)*state + α*tanh(pre)`).
/// Valores menores → mais memória de longo prazo, gradiente menor por tick.
pub const ALPHA: f32 = 0.1;
/// Lado do grid 2D (`GRID_SIDE * GRID_SIDE = N_CELLS`).
pub const GRID_SIDE: usize = 4;

const _: () = assert!(GRID_SIDE * GRID_SIDE == N_CELLS);

/// Retorna os 4 vizinhos toroidais (cima, baixo, esq, dir) de uma cell.
#[inline]
pub const fn neighbors(c: usize) -> [usize; 4] {
    let row = c / GRID_SIDE;
    let col = c % GRID_SIDE;
    let up = ((row + GRID_SIDE - 1) % GRID_SIDE) * GRID_SIDE + col;
    let down = ((row + 1) % GRID_SIDE) * GRID_SIDE + col;
    let left = row * GRID_SIDE + (col + GRID_SIDE - 1) % GRID_SIDE;
    let right = row * GRID_SIDE + (col + 1) % GRID_SIDE;
    [up, down, left, right]
}

/// Configuração da grade v2.
#[derive(Debug, Clone)]
pub struct GridConfig {
    /// Tamanho do vocab compacto.
    pub vocab_size: usize,
    /// Janela de bytes por pair de treino.
    pub window: usize,
    /// Learning rate.
    pub learning_rate: f32,
    /// Seed determinístico.
    pub seed: u64,
}

/// Grade SMG v3 com 16 microcolunas × 8 FSMs + bus lateral.
pub struct SmgGrid {
    /// Configuração.
    pub cfg: GridConfig,
    /// Recorrência por família. Shape `[N_FAMILIES, K, K]`.
    t_f: Vec<f32>,
    /// Embedding de input por família. Shape `[N_FAMILIES, vocab, K]`.
    i_f: Vec<f32>,
    /// Viés por família. Shape `[N_FAMILIES, K]`.
    b_f: Vec<f32>,
    /// Viés por cell (quebra simetria). Shape `[N_CELLS, K]`.
    pos_bias: Vec<f32>,
    /// **Bus lateral** por família. Shape `[N_FAMILIES, K, K]`.
    /// `lat_w[f, j, k]` projeta estado vizinho da família `f` para pre-act.
    /// Compartilhado entre todas as 4 direções (sem direção learnable).
    lat_w: Vec<f32>,
    /// Decoder linear. Shape `[FLAT_DIM, vocab]` row-major em (i, j).
    w_out: Vec<f32>,
    /// Viés do decoder. Shape `[vocab]`.
    b_out: Vec<f32>,
}

/// Cache da forward para BPTT.
#[derive(Debug, Clone)]
pub struct ForwardCache {
    /// Estados em todos os ticks. Shape `[(T+1), N_CELLS, FSMS_PER_CELL, K]`.
    /// `states[0]` = estado inicial (zeros).
    pub states: Vec<f32>,
    /// Logits do decoder. Shape `[vocab]`.
    pub logits: Vec<f32>,
    /// Janela vista.
    pub window: Vec<u32>,
}

/// Gradientes acumulados.
#[derive(Debug, Clone)]
pub struct Grads {
    /// dL/dT_f.
    pub g_t_f: Vec<f32>,
    /// dL/dI_f.
    pub g_i_f: Vec<f32>,
    /// dL/db_f.
    pub g_b_f: Vec<f32>,
    /// dL/d pos_bias.
    pub g_pos_bias: Vec<f32>,
    /// dL/d lat_w.
    pub g_lat_w: Vec<f32>,
    /// dL/dW_out.
    pub g_w_out: Vec<f32>,
    /// dL/db_out.
    pub g_b_out: Vec<f32>,
    /// Soma das losses.
    pub sum_loss: f32,
}

impl SmgGrid {
    /// Inicializa pesos.
    #[must_use]
    pub fn new(cfg: GridConfig) -> Self {
        let mut rng = Rng::new(cfg.seed);
        let v = cfg.vocab_size;
        // Xavier-ish: scale para tanh (tan' max = 1)
        let s_t = (1.0 / K as f32).sqrt();
        let s_i = (1.0 / v as f32).sqrt();
        let s_o = (1.0 / FLAT_DIM as f32).sqrt();
        // Lateral: 4 vizinhos somam → escala / sqrt(4*K) para ficar O(1)
        let s_lat = (1.0 / (4.0 * K as f32)).sqrt();

        let mut sample = |scale: f32| (rng.next_f64() as f32 - 0.5) * 2.0 * scale;

        let t_f: Vec<f32> = (0..N_FAMILIES * K * K).map(|_| sample(s_t)).collect();
        let i_f: Vec<f32> = (0..N_FAMILIES * v * K).map(|_| sample(s_i)).collect();
        let b_f = vec![0.0_f32; N_FAMILIES * K];
        let pos_bias: Vec<f32> = (0..N_CELLS * K).map(|_| sample(0.1)).collect();
        let lat_w: Vec<f32> = (0..N_FAMILIES * K * K).map(|_| sample(s_lat)).collect();
        let w_out: Vec<f32> = (0..FLAT_DIM * v).map(|_| sample(s_o)).collect();
        let b_out = vec![0.0_f32; v];

        Self { cfg, t_f, i_f, b_f, pos_bias, lat_w, w_out, b_out }
    }

    /// Total de parâmetros learnable.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.t_f.len() + self.i_f.len() + self.b_f.len()
            + self.pos_bias.len() + self.lat_w.len()
            + self.w_out.len() + self.b_out.len()
    }

    // --- Accessors read-only para os pesos (usados pelo backend GPU). ---
    /// `T_f`: recurrence intra-FSM. Shape `[N_FAMILIES, K, K]`.
    #[must_use]
    pub fn weights_t_f(&self) -> &[f32] { &self.t_f }
    /// `I_f`: input embedding. Shape `[N_FAMILIES, vocab, K]`.
    #[must_use]
    pub fn weights_i_f(&self) -> &[f32] { &self.i_f }
    /// `b_f`: viés. Shape `[N_FAMILIES, K]`.
    #[must_use]
    pub fn weights_b_f(&self) -> &[f32] { &self.b_f }
    /// Position bias por cell. Shape `[N_CELLS, K]`.
    #[must_use]
    pub fn weights_pos_bias(&self) -> &[f32] { &self.pos_bias }
    /// Bus lateral. Shape `[N_FAMILIES, K, K]`.
    #[must_use]
    pub fn weights_lat_w(&self) -> &[f32] { &self.lat_w }
    /// Decoder linear. Shape `[FLAT_DIM, vocab]`.
    #[must_use]
    pub fn weights_w_out(&self) -> &[f32] { &self.w_out }
    /// Decoder bias. Shape `[vocab]`.
    #[must_use]
    pub fn weights_b_out(&self) -> &[f32] { &self.b_out }

    /// Forward com cache para BPTT. Estado: `state' = (1-α)*state + α*tanh(pre)`.
    /// **v3**: pre inclui contribuição lateral das 4 cells vizinhas (toroidal),
    /// usando `state_in` (mesmo tick) — sem dependência serial entre cells.
    pub fn forward(&self, window: &[u32]) -> ForwardCache {
        let v = self.cfg.vocab_size;
        let t_steps = window.len();
        let stride_step = N_CELLS * FSMS_PER_CELL * K;

        // Estado inicial = zero (não simplex como em v1)
        let mut states = vec![0.0_f32; (t_steps + 1) * stride_step];
        let mut pre = [0.0_f32; K];

        // Buffer reutilizado: lat_proj[cell, fsm, k] = (state_in[cell,fsm,:] @ lat_w[fsm])[k]
        // Computado UMA VEZ por tick por (cell, fsm) e somado nos 4 vizinhos depois.
        // Reduz custo do bus lateral 4× (de 4 mat-vecs por consumidor para 1 por produtor).
        let mut lat_proj = vec![0.0_f32; N_CELLS * FSMS_PER_CELL * K];

        for step in 0..t_steps {
            let input_byte = (window[step] as usize).min(v.saturating_sub(1));
            let s_in = step * stride_step;
            let s_out = (step + 1) * stride_step;

            // Fase 1: pré-projeção lateral por cell+fsm.
            //   lat_proj[c, f, k] = sum_j state_in[c, f, j] * lat_w[f, j, k]
            for cell in 0..N_CELLS {
                for fsm in 0..FSMS_PER_CELL {
                    let in_off = s_in + (cell * FSMS_PER_CELL + fsm) * K;
                    let proj_off = (cell * FSMS_PER_CELL + fsm) * K;
                    let lat_off = fsm * K * K;
                    for k in 0..K {
                        lat_proj[proj_off + k] = 0.0;
                    }
                    for j in 0..K {
                        let s = states[in_off + j];
                        if s == 0.0 {
                            continue;
                        }
                        let row_off = lat_off + j * K;
                        for k in 0..K {
                            lat_proj[proj_off + k] += s * self.lat_w[row_off + k];
                        }
                    }
                }
            }

            // Fase 2: forward propriamente dito por cell+fsm, somando lat_proj dos vizinhos.
            for cell in 0..N_CELLS {
                let pos_off = cell * K;
                let neigh = neighbors(cell);
                for fsm in 0..FSMS_PER_CELL {
                    let f = fsm;
                    let in_off = s_in + (cell * FSMS_PER_CELL + fsm) * K;
                    let out_off = s_out + (cell * FSMS_PER_CELL + fsm) * K;
                    let t_off = f * K * K;
                    let i_off = (f * v + input_byte) * K;
                    let b_off = f * K;

                    // pre[k] = I + b + pos + recorrência + lateral
                    for k in 0..K {
                        pre[k] = self.i_f[i_off + k]
                            + self.b_f[b_off + k]
                            + self.pos_bias[pos_off + k];
                    }
                    // recorrência intra-FSM
                    for j in 0..K {
                        let s = states[in_off + j];
                        if s == 0.0 {
                            continue;
                        }
                        for k in 0..K {
                            pre[k] += s * self.t_f[t_off + j * K + k];
                        }
                    }
                    // lateral: soma lat_proj dos 4 vizinhos
                    for &n in &neigh {
                        let n_proj_off = (n * FSMS_PER_CELL + fsm) * K;
                        for k in 0..K {
                            pre[k] += lat_proj[n_proj_off + k];
                        }
                    }
                    for k in 0..K {
                        states[out_off + k] =
                            (1.0 - ALPHA) * states[in_off + k] + ALPHA * pre[k].tanh();
                    }
                }
            }
        }

        // Decoder
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

        ForwardCache { states, logits, window: window.to_vec() }
    }

    /// Predição argmax.
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

    /// Accuracy em batch.
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

    /// Aloca buffer de gradientes.
    #[must_use]
    pub fn alloc_grads(&self) -> Grads {
        Grads {
            g_t_f: vec![0.0; self.t_f.len()],
            g_i_f: vec![0.0; self.i_f.len()],
            g_b_f: vec![0.0; self.b_f.len()],
            g_pos_bias: vec![0.0; self.pos_bias.len()],
            g_lat_w: vec![0.0; self.lat_w.len()],
            g_w_out: vec![0.0; self.w_out.len()],
            g_b_out: vec![0.0; self.b_out.len()],
            sum_loss: 0.0,
        }
    }

    /// Computa gradientes acumulados em `grads`.
    pub fn compute_grads_into(&self, batch: &[(Vec<u32>, u32)], grads: &mut Grads) {
        let v = self.cfg.vocab_size;
        let stride_step = N_CELLS * FSMS_PER_CELL * K;

        grads.g_t_f.fill(0.0);
        grads.g_i_f.fill(0.0);
        grads.g_b_f.fill(0.0);
        grads.g_pos_bias.fill(0.0);
        grads.g_lat_w.fill(0.0);
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

            // Decoder backward
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

            // BPTT — agora com tanh + residual
            let mut d_state_next: Vec<f32> = d_flat;

            for step in (0..t_steps).rev() {
                let input_byte = (window[step] as usize).min(v - 1);
                let s_in_off = step * stride_step;
                let s_out_off = (step + 1) * stride_step;

                let mut d_state_prev = vec![0.0_f32; stride_step];

                // Buffer compartilhado: d_pre[c, f, k]. Computado na Fase 1 e
                // reutilizado nas fases laterais (mesmo trick do forward).
                let mut d_pre_all = vec![0.0_f32; N_CELLS * FSMS_PER_CELL * K];

                // === Fase 1: por (cell, fsm) — d_pre, params I/b/pos, T_f, recorrência ===
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

                        let s_out: &[f32] = &cache.states[out_off..out_off + K];
                        let s_in: &[f32] = &cache.states[in_off..in_off + K];
                        let dsn: &[f32] = &d_state_next[d_off..d_off + K];

                        // d_pre + residual
                        for k in 0..K {
                            d_state_prev[d_off + k] += (1.0 - ALPHA) * dsn[k];
                            let tp = (s_out[k] - (1.0 - ALPHA) * s_in[k]) / ALPHA;
                            d_pre_all[d_off + k] = ALPHA * (1.0 - tp * tp) * dsn[k];
                        }

                        let d_pre = &d_pre_all[d_off..d_off + K];

                        // params I, b, pos
                        for k in 0..K {
                            grads.g_i_f[i_off + k] += d_pre[k];
                            grads.g_b_f[b_off + k] += d_pre[k];
                            grads.g_pos_bias[pos_off + k] += d_pre[k];
                        }

                        // T_f e recorrência intra-FSM
                        for j in 0..K {
                            let sj = s_in[j];
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

                // === Fase 2: pool de gradientes laterais ===
                //   d_pool[n, f, k] = sum_{c | n in neighbors(c)} d_pre[c, f, k]
                //   Por simetria 4-connected toroidal: n ∈ neighbors(c) ⇔ c ∈ neighbors(n).
                let mut d_pool = vec![0.0_f32; N_CELLS * FSMS_PER_CELL * K];
                for cell in 0..N_CELLS {
                    let neigh = neighbors(cell);
                    for fsm in 0..FSMS_PER_CELL {
                        let d_off = (cell * FSMS_PER_CELL + fsm) * K;
                        // Para cada vizinho n de cell: d_pool[n,f] += d_pre[cell,f]
                        // (cell contribui para o pool de cada um dos seus vizinhos)
                        for &n in &neigh {
                            let n_pool_off = (n * FSMS_PER_CELL + fsm) * K;
                            for k in 0..K {
                                d_pool[n_pool_off + k] += d_pre_all[d_off + k];
                            }
                        }
                    }
                }

                // === Fase 3: por (n, fsm) — g_lat_w (outer-product) e d_state_prev (mat-vec) ===
                for n in 0..N_CELLS {
                    for fsm in 0..FSMS_PER_CELL {
                        let f = fsm;
                        let lat_off = f * K * K;
                        let n_in_off = s_in_off + (n * FSMS_PER_CELL + fsm) * K;
                        let n_d_off = (n * FSMS_PER_CELL + fsm) * K;
                        let pool = &d_pool[n_d_off..n_d_off + K];

                        for j in 0..K {
                            let sj = cache.states[n_in_off + j];
                            let row_off = lat_off + j * K;
                            let mut acc = 0.0_f32;
                            for k in 0..K {
                                grads.g_lat_w[row_off + k] += sj * pool[k];
                                acc += self.lat_w[row_off + k] * pool[k];
                            }
                            d_state_prev[n_d_off + j] += acc;
                        }
                    }
                }

                d_state_next = d_state_prev;
            }
        }
    }

    /// L2 norm global de todos os gradientes (sqrt da soma dos quadrados).
    fn grads_global_norm(grads: &Grads) -> f32 {
        let mut acc = 0.0_f64;
        for v in [&grads.g_t_f, &grads.g_i_f, &grads.g_b_f, &grads.g_pos_bias,
                  &grads.g_lat_w, &grads.g_w_out, &grads.g_b_out] {
            for &x in v {
                acc += (x as f64) * (x as f64);
            }
        }
        acc.sqrt() as f32
    }

    /// Aplica gradientes acumulados com **gradient clipping** global.
    ///
    /// Se a norma L2 total dos grads exceder `max_norm`, escala TODOS os grads
    /// uniformemente para `max_norm`. Imprescindível para v3 com bus lateral
    /// (loop recorrente dentro de cada tick → instabilidade fácil sem clip).
    pub fn apply_grads_clipped(&mut self, grads: &Grads, scale: f32, max_norm: f32) {
        let norm = Self::grads_global_norm(grads);
        let clip_scale = if norm > max_norm { max_norm / norm } else { 1.0 };
        let s = scale * clip_scale;
        self.apply_grads_raw(grads, s);
    }

    /// Aplica gradientes sem clipping (uso interno e backward compatibility).
    pub fn apply_grads(&mut self, grads: &Grads, scale: f32) {
        self.apply_grads_raw(grads, scale);
    }

    fn apply_grads_raw(&mut self, grads: &Grads, scale: f32) {
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
        for i in 0..self.lat_w.len() {
            self.lat_w[i] -= scale * grads.g_lat_w[i];
        }
        for i in 0..self.w_out.len() {
            self.w_out[i] -= scale * grads.g_w_out[i];
        }
        for i in 0..self.b_out.len() {
            self.b_out[i] -= scale * grads.g_b_out[i];
        }
    }

    /// Treina 1 época em mini-batch SGD paralelo.
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
                    .map(|(chunk, buf)| s.spawn(move || grid_ref.compute_grads_into(chunk, buf)))
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
                for i in 0..head.g_lat_w.len() {
                    head.g_lat_w[i] += r.g_lat_w[i];
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
            // gradient clipping global @ max_norm=20.0 — atua só como failsafe.
            // Em treino normal, norm fica abaixo desse limiar (não interfere);
            // captura apenas runaway do bus lateral antes de divergir.
            self.apply_grads_clipped(head, scale, 20.0);
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
    fn forward_state_bounded_in_neg1_pos1() {
        let g = SmgGrid::new(cfg());
        let cache = g.forward(&[0, 1, 2, 3]);
        for &s in &cache.states {
            assert!(s >= -1.0 && s <= 1.0, "state fora de [-1,1]: {s}");
        }
    }

    #[test]
    fn neighbors_toroidal_4connected() {
        // cell 0 (linha 0, col 0): up=12 (3,0), down=4 (1,0), left=3 (0,3), right=1 (0,1)
        assert_eq!(neighbors(0), [12, 4, 3, 1]);
        // cell 5 (linha 1, col 1): up=1, down=9, left=4, right=6
        assert_eq!(neighbors(5), [1, 9, 4, 6]);
        // cell 15 (linha 3, col 3): up=11, down=3, left=14, right=12
        assert_eq!(neighbors(15), [11, 3, 14, 12]);
    }

    #[test]
    fn can_overfit_single_pair() {
        let mut grid = SmgGrid::new(cfg());
        let window = vec![0_u32, 1, 2, 3];
        let target = 5_u32;
        let initial = grid.eval_loss(&[(window.clone(), target)]);
        let mut buf = grid.alloc_grads();
        for _ in 0..200 {
            grid.compute_grads_into(&[(window.clone(), target)], &mut buf);
            grid.apply_grads(&buf, 0.1);
        }
        let final_loss = grid.eval_loss(&[(window.clone(), target)]);
        assert!(
            final_loss < initial * 0.1,
            "v3 deve convergir em pair único: initial={initial}, final={final_loss}"
        );
    }

    #[test]
    fn lateral_bus_actually_changes_output() {
        // Sanity: zerar lat_w deve dar resultado diferente de lat_w random.
        let mut g_with = SmgGrid::new(cfg());
        let mut g_zero = SmgGrid::new(cfg());
        for w in &mut g_zero.lat_w {
            *w = 0.0;
        }
        let win = vec![0_u32, 1, 2, 3, 4, 5];
        let l_with = g_with.forward(&win).logits;
        let l_zero = g_zero.forward(&win).logits;
        let diff: f32 = l_with.iter().zip(l_zero.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-3, "lateral bus não está afetando saída (diff={diff})");
        // Mexa também em g_with pra evitar dead_code (não estritamente necessário)
        let _ = &mut g_with;
    }
}
