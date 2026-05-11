//! `SmgGridHsa` — espelho GPU (HSA AQL) de `microcolumn_v3::SmgGrid`.
//!
//! Interface equivalente a `SmgGridGpu` mas:
//!   * Sem wgpu; despacha kernels HIP nativos via AQL ring buffer.
//!   * Buffers de per-batch pre-alocados → ZERO hipMalloc/hipFree no hot path.
//!   * `dispatch_async` + `drain()`: latência esperada <<1ms por epoch.
//!
//! # Buffers persistentes
//!
//! Para eliminar `hipMalloc`/`hipFree` do hot path, pre-alocamos buffers
//! dimensionados para `max_batch_size`. No `train_step` usamos `hip_h2d`
//! (hipMemcpy H→D) para atualizar win/tgt sem realocar.
//!
//! ```text
//!   pst_win   [max_batch × T]                  ← win_flat via hip_h2d
//!   pst_tgt   [max_batch]                       ← targets  via hip_h2d
//!   pst_log   [max_batch × V]                   ← escrito por forward_main
//!   pst_st    [max_batch × (T+1) × FLAT_DIM]   ← escrito por forward_main (49 MB a max_batch=64)
//!   pst_dpre  [max_batch × T × FLAT_DIM]        ← escrito por backward_bptt (48 MB a max_batch=64)
//!   pst_dlog  [max_batch × V]                   ← escrito por backward_bptt (64 KB a max_batch=64)
//! ```
//!
//! # Fluxo de train_step (2-phase backward sem atomics para g_t_f/g_lat_w/g_w_out)
//! ```text
//! hip_h2d(pst_win ← win_flat)               ~µs
//! hip_h2d(pst_tgt ← targets)                ~µs
//! dispatch_forward → pst_log, pst_st
//! drain()                                    sync CPU (logits readback)
//! hip_d2h(logits_host ← pst_log)
//! -- CPU softmax → loss --
//! dispatch_backward_bptt → pst_dpre, pst_dlog, g_i_f, g_b_out
//! barrier()                                  GPU-side barrier (sem stall CPU)
//! dispatch_grad_acc_tf_latw → g_t_f, g_lat_w  (128 WGs, atomic_add 16-way)
//! dispatch_grad_acc_wout    → g_w_out          (8192 WGs, escrita directa)
//! dispatch_grad_acc_b_f     → g_b_f            (128 WGs, atomic_add 16-way)
//! dispatch_grad_acc_pos     → g_pos            (128 WGs, atomic_add 8-way)
//! drain()                                    sync CPU (todos os grads prontos)
//! dispatch_sgd × 7
//! drain()
//! ```
//! Medido: ~10ms/step vs ~58ms/step wgpu = 5.7×.
//! Após fix 128-WG grad_acc: alvo ~6–8ms/step.

#![allow(unsafe_code)]

use hip_ffi::safe::{DeviceBuffer, hip_h2d, hip_d2h};

use crate::learn::microcolumn_v3::{GridConfig, SmgGrid, FLAT_DIM, K, N_FAMILIES, N_CELLS};

use super::dispatch::{SmgRuntime, ParamsFwd, ParamsBwd, ParamsGradAcc};

/// Tipo de erro de construção (HipError ou HsaError).
pub type GridInitError = Box<dyn std::error::Error>;

/// Pesos descarregados da GPU para o host (para validação / parity tests).
#[derive(Debug, Clone)]
pub struct WeightsHost {
    pub t_f:      Vec<f32>,
    pub i_f:      Vec<f32>,
    pub nibble_h: Vec<f32>,   // [N_FAMILIES, 16, K] — high nibble (zona ASCII)
    pub nibble_l: Vec<f32>,   // [N_FAMILIES, 16, K] — low  nibble (posição na zona)
    pub b_f:      Vec<f32>,
    pub pos_bias: Vec<f32>,
    pub lat_w:    Vec<f32>,
    pub w_out:    Vec<f32>,
    pub b_out:    Vec<f32>,
}

// ---------------------------------------------------------------------------
// Estrutura principal
// ---------------------------------------------------------------------------

/// SMG v3 em GPU via HSA AQL. Cria com `SmgGridHsa::from_cpu(&cpu, max_batch)`.
pub struct SmgGridHsa {
    cfg:        GridConfig,
    rt:         SmgRuntime,
    max_batch:  usize,

    // Pesos persistentes (f32)
    buf_t_f:   DeviceBuffer<f32>,
    buf_i_f:   DeviceBuffer<f32>,
    buf_b_f:   DeviceBuffer<f32>,
    buf_pos:   DeviceBuffer<f32>,
    buf_lat:   DeviceBuffer<f32>,
    buf_w_out: DeviceBuffer<f32>,
    buf_b_out: DeviceBuffer<f32>,

    // Nibble geometric embedding buffers (aditivos a i_f)
    buf_nibble_h:   DeviceBuffer<f32>,   // [N_FAMILIES × 16 × K] — high nibble (zona)
    buf_nibble_l:   DeviceBuffer<f32>,   // [N_FAMILIES × 16 × K] — low  nibble (posição)

    // Grad buffers persistentes (u32 = f32 bit-pattern via CAS-loop)
    buf_g_t_f:      DeviceBuffer<u32>,
    buf_g_i_f:      DeviceBuffer<u32>,
    buf_g_nibble_h: DeviceBuffer<u32>,   // grad nibble_h (atomic CAS)
    buf_g_nibble_l: DeviceBuffer<u32>,   // grad nibble_l (atomic CAS)
    buf_g_b_f:      DeviceBuffer<u32>,
    buf_g_pos:      DeviceBuffer<u32>,
    buf_g_lat_w:    DeviceBuffer<u32>,
    buf_g_w_out:    DeviceBuffer<u32>,
    buf_g_b_out:    DeviceBuffer<u32>,

    // Buffers de per-batch pre-alocados (eliminam hipMalloc do hot path).
    pst_win:  DeviceBuffer<u32>,   // [max_batch × T]
    pst_tgt:  DeviceBuffer<u32>,   // [max_batch]
    pst_log:  DeviceBuffer<f32>,   // [max_batch × V]
    pst_st:   DeviceBuffer<f32>,   // [max_batch × (T+1) × FLAT_DIM]  (~49 MB @ max_batch=64)
    pst_dpre: DeviceBuffer<f32>,   // [max_batch × T × FLAT_DIM]       (~48 MB @ max_batch=64)
    pst_dlog: DeviceBuffer<f32>,   // [max_batch × V]                   (~64 KB @ max_batch=64)

    // Tamanhos dos tensores de gradiente (para dispatch_sgd)
    n_t_f:    u32,
    n_i_f:    u32,
    n_nibble: u32,   // N_FAMILIES × 16 × K = 4096 (igual para nibble_h e nibble_l)
    n_b_f:    u32,
    n_pos:    u32,
    n_lat_w:  u32,
    n_w_out:  u32,
    n_b_out:  u32,
}

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

impl SmgGridHsa {
    /// Cria instância a partir de `SmgGrid` CPU.
    ///
    /// `max_batch_size` dimensiona os buffers de per-batch (pst_win, pst_tgt,
    /// pst_log, pst_st). Para treino com batch fixo, passe o batch_size do loop.
    /// Para batch variável, passe o maior batch esperado.
    pub fn from_cpu(cpu: &SmgGrid, max_batch_size: usize) -> Result<Self, GridInitError> {
        assert!(max_batch_size > 0, "max_batch_size deve ser > 0");
        let rt = SmgRuntime::new()?;
        let v  = cpu.cfg.vocab_size;
        let t  = cpu.cfg.window;

        // Upload pesos.
        let buf_t_f   = DeviceBuffer::<f32>::from_slice(cpu.weights_t_f())?;
        let buf_i_f   = DeviceBuffer::<f32>::from_slice(cpu.weights_i_f())?;
        let buf_b_f   = DeviceBuffer::<f32>::from_slice(cpu.weights_b_f())?;
        let buf_pos   = DeviceBuffer::<f32>::from_slice(cpu.weights_pos_bias())?;
        let buf_lat   = DeviceBuffer::<f32>::from_slice(cpu.weights_lat_w())?;
        let buf_w_out = DeviceBuffer::<f32>::from_slice(cpu.weights_w_out())?;
        let buf_b_out = DeviceBuffer::<f32>::from_slice(cpu.weights_b_out())?;

        // Tamanhos dos tensores de gradiente.
        let n_t_f    = (N_FAMILIES * K * K)  as u32;
        let n_i_f    = (N_FAMILIES * v * K)  as u32;
        let n_nibble = (N_FAMILIES * 16 * K) as u32;  // 8 × 16 × 32 = 4096
        let n_b_f    = (N_FAMILIES * K)      as u32;
        let n_pos    = (N_CELLS   * K)       as u32;
        let n_lat_w  = (N_FAMILIES * K * K)  as u32;
        let n_w_out  = (FLAT_DIM  * v)       as u32;
        let n_b_out  = v                     as u32;

        // Nibble geometric embedding — inicializados a zero.
        // O modelo começa com apenas i_f activo; nibble_h e nibble_l aprendem
        // adicionalmente a partir de zero → inductive bias emergente do treino.
        let buf_nibble_h = DeviceBuffer::<f32>::from_slice(&vec![0.0f32; n_nibble as usize])?;
        let buf_nibble_l = DeviceBuffer::<f32>::from_slice(&vec![0.0f32; n_nibble as usize])?;

        // Grad buffers zerados (0u32 == 0.0f32 via bit-pattern).
        let buf_g_t_f      = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_t_f    as usize])?;
        let buf_g_i_f      = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_i_f    as usize])?;
        let buf_g_nibble_h = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_nibble as usize])?;
        let buf_g_nibble_l = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_nibble as usize])?;
        let buf_g_b_f      = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_b_f    as usize])?;
        let buf_g_pos      = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_pos    as usize])?;
        let buf_g_lat_w    = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_lat_w  as usize])?;
        let buf_g_w_out    = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_w_out  as usize])?;
        let buf_g_b_out    = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_b_out  as usize])?;

        // Buffers de per-batch pre-alocados (uma única hipMalloc na construção).
        //
        // Maiores buffers (@ max_batch=64, T=48, FLAT_DIM=4096):
        //   pst_st   = 64 × 49 × 4096 × 4 ≈  49 MB
        //   pst_dpre = 64 × 48 × 4096 × 4 ≈  48 MB
        //   Total intermediário ≈ 97 MB (< 1% do VRAM de 12 GB).
        let pst_win  = DeviceBuffer::<u32>::alloc(max_batch_size * t)?;
        let pst_tgt  = DeviceBuffer::<u32>::alloc(max_batch_size)?;
        let pst_log  = DeviceBuffer::<f32>::alloc(max_batch_size * v)?;
        let pst_st   = DeviceBuffer::<f32>::alloc(max_batch_size * (t + 1) * FLAT_DIM)?;
        let pst_dpre = DeviceBuffer::<f32>::alloc(max_batch_size * t * FLAT_DIM)?;
        let pst_dlog = DeviceBuffer::<f32>::alloc(max_batch_size * v)?;

        Ok(Self {
            cfg: cpu.cfg.clone(),
            rt,
            max_batch: max_batch_size,
            buf_t_f, buf_i_f,
            buf_nibble_h, buf_nibble_l,
            buf_b_f, buf_pos, buf_lat, buf_w_out, buf_b_out,
            buf_g_t_f, buf_g_i_f,
            buf_g_nibble_h, buf_g_nibble_l,
            buf_g_b_f, buf_g_pos,
            buf_g_lat_w, buf_g_w_out, buf_g_b_out,
            pst_win, pst_tgt, pst_log, pst_st, pst_dpre, pst_dlog,
            n_t_f, n_i_f, n_nibble, n_b_f, n_pos, n_lat_w, n_w_out, n_b_out,
        })
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    pub fn n_params(&self) -> usize {
        (self.n_t_f + self.n_i_f + 2 * self.n_nibble
            + self.n_b_f + self.n_pos
            + self.n_lat_w + self.n_w_out + self.n_b_out) as usize
    }

    /// Actualiza a learning rate em runtime (para LR annealing entre épocas).
    ///
    /// O `scale` em `train_step` é `cfg.learning_rate / batch_size`, pelo que
    /// basta mudar este campo antes de cada época para implementar qualquer schedule.
    pub fn set_lr(&mut self, lr: f32) {
        self.cfg.learning_rate = lr;
    }

    /// Retorna a learning rate actual.
    pub fn lr(&self) -> f32 {
        self.cfg.learning_rate
    }

    /// Baixa os pesos actuais da GPU para o host.
    /// Usado em testes de parity (comparação com CPU após train_step).
    pub fn weights_to_host(&self) -> WeightsHost {
        WeightsHost {
            t_f:      self.buf_t_f.to_host().expect("to_host t_f"),
            i_f:      self.buf_i_f.to_host().expect("to_host i_f"),
            nibble_h: self.buf_nibble_h.to_host().expect("to_host nibble_h"),
            nibble_l: self.buf_nibble_l.to_host().expect("to_host nibble_l"),
            b_f:      self.buf_b_f.to_host().expect("to_host b_f"),
            pos_bias: self.buf_pos.to_host().expect("to_host pos"),
            lat_w:    self.buf_lat.to_host().expect("to_host lat_w"),
            w_out:    self.buf_w_out.to_host().expect("to_host w_out"),
            b_out:    self.buf_b_out.to_host().expect("to_host b_out"),
        }
    }

    // -----------------------------------------------------------------------
    // Inferência — alloc por chamada (fora do hot path de treino)
    // -----------------------------------------------------------------------

    /// Forward em batch. Para inferência (sem cache de estados).
    pub fn forward_batch(&self, windows: &[Vec<u32>]) -> Vec<Vec<f32>> {
        assert!(!windows.is_empty(), "forward_batch: batch vazio");
        let batch_size = windows.len();
        let t_steps    = windows[0].len();
        let v          = self.cfg.vocab_size;

        let win_flat: Vec<u32> = windows.iter()
            .flat_map(|w| w.iter().copied())
            .collect();

        // Alloc por chamada (inferência — fora do hot path).
        let buf_win     = DeviceBuffer::<u32>::from_slice(&win_flat).expect("DevBuf windows");
        let mut buf_log = DeviceBuffer::<f32>::alloc(batch_size * v).expect("DevBuf logits");
        let mut buf_st  = DeviceBuffer::<f32>::alloc(1).expect("DevBuf states_dummy");

        let params = ParamsFwd {
            vocab_size:   v          as u32,
            t_steps:      t_steps    as u32,
            batch_size:   batch_size as u32,
            cache_states: 0,
        };
        // SAFETY: ponteiros são buffers GPU válidos.
        unsafe {
            self.rt.dispatch_forward(
                params,
                self.buf_t_f.as_ptr()       as u64,
                self.buf_i_f.as_ptr()       as u64,
                self.buf_nibble_h.as_ptr()  as u64,   // [N_FAMILIES, 16, K]
                self.buf_nibble_l.as_ptr()  as u64,   // [N_FAMILIES, 16, K]
                self.buf_b_f.as_ptr()       as u64,
                self.buf_pos.as_ptr()       as u64,
                self.buf_lat.as_ptr()       as u64,
                self.buf_w_out.as_ptr()     as u64,
                self.buf_b_out.as_ptr()     as u64,
                buf_win.as_ptr()            as u64,
                buf_log.as_mut_ptr()        as u64,
                buf_st.as_mut_ptr()         as u64,
            ).expect("dispatch_forward");
            self.rt.drain().expect("drain forward_batch");
        }

        let flat = buf_log.to_host().expect("to_host logits");
        flat.chunks(v).map(|c| c.to_vec()).collect()
    }

    /// Argmax para uma única janela (equivale a `SmgGrid::predict`).
    pub fn predict(&self, window: &[u32]) -> u32 {
        let logits = &self.forward_batch(&[window.to_vec()])[0];
        logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    // -----------------------------------------------------------------------
    // Treino — hot path com buffers persistentes
    // -----------------------------------------------------------------------

    /// Um passo de treino (forward → backward → SGD × 7).
    ///
    /// **Zero alocações GPU no hot path**: usa `hip_h2d` para atualizar
    /// `pst_win`/`pst_tgt` em vez de hipMalloc. Isso é o gargalo principal
    /// vs wgpu (49 MB hipMalloc/hipFree por step no caminho ingênuo).
    ///
    /// # Panics
    /// Pânica se `batch.len() > max_batch_size` (passado em `from_cpu`).
    pub fn train_step(&mut self, batch: &[(Vec<u32>, u32)]) -> f32 {
        assert!(!batch.is_empty(), "train_step: batch vazio");
        let batch_size = batch.len();
        assert!(
            batch_size <= self.max_batch,
            "batch_size={} > max_batch={} — recrie SmgGridHsa com max_batch maior",
            batch_size, self.max_batch
        );

        let t_steps = batch[0].0.len();
        let v       = self.cfg.vocab_size;

        // Flatten host-side (O(batch×T) Vec alloc — barato, apenas RAM).
        let win_flat: Vec<u32> = batch.iter()
            .flat_map(|(w, _)| w.iter().copied())
            .collect();
        let targets: Vec<u32>  = batch.iter().map(|(_, t)| *t).collect();

        // ==== Upload inputs via hipMemcpy (sem hipMalloc) ====
        // SAFETY: pst_win/pst_tgt foram alocados em from_cpu com capacidade
        // >= max_batch×T e >= max_batch respectivamente.
        unsafe {
            hip_h2d(self.pst_win.as_mut_ptr(), &win_flat)
                .expect("hip_h2d windows");
            hip_h2d(self.pst_tgt.as_mut_ptr(), &targets)
                .expect("hip_h2d targets");
        }

        let params_fwd = ParamsFwd {
            vocab_size:   v          as u32,
            t_steps:      t_steps    as u32,
            batch_size:   batch_size as u32,
            cache_states: 1,
        };

        // ==== Fase 1: Forward ====
        // SAFETY: todos os ponteiros são buffers GPU válidos.
        unsafe {
            self.rt.dispatch_forward(
                params_fwd,
                self.buf_t_f.as_ptr()        as u64,
                self.buf_i_f.as_ptr()        as u64,
                self.buf_nibble_h.as_ptr()   as u64,   // [N_FAMILIES, 16, K]
                self.buf_nibble_l.as_ptr()   as u64,   // [N_FAMILIES, 16, K]
                self.buf_b_f.as_ptr()        as u64,
                self.buf_pos.as_ptr()        as u64,
                self.buf_lat.as_ptr()        as u64,
                self.buf_w_out.as_ptr()      as u64,
                self.buf_b_out.as_ptr()      as u64,
                self.pst_win.as_ptr()        as u64,
                self.pst_log.as_mut_ptr()    as u64,
                self.pst_st.as_mut_ptr()     as u64,
            ).expect("dispatch_forward");
            self.rt.drain().expect("drain forward");
            // drain() necessário: logits/states prontos antes de readback + backward.
        }

        // ==== Fase 2: Loss (CPU softmax sobre logits readback) ====
        // Copia apenas batch_size×V floats do início do pst_log.
        let mut logits_host = vec![0.0f32; batch_size * v];
        // SAFETY: pst_log contém >= batch_size×V elementos escritos pelo forward.
        unsafe {
            hip_d2h(&mut logits_host, self.pst_log.as_ptr())
                .expect("hip_d2h logits");
        }

        let mut total_loss = 0.0_f32;
        for (b, (_, target)) in batch.iter().enumerate() {
            let logits = &logits_host[b * v..(b + 1) * v];
            let t  = (*target as usize).min(v - 1);
            let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&x| (x - mx).exp()).collect();
            let s: f32 = exps.iter().sum();
            total_loss -= (exps[t] / s).max(1e-20).ln();
        }

        let params_bwd = ParamsBwd {
            vocab_size: v          as u32,
            t_steps:    t_steps    as u32,
            batch_size: batch_size as u32,
            _pad:       0,
        };
        let params_grad = ParamsGradAcc {
            vocab_size: v          as u32,
            t_steps:    t_steps    as u32,
            batch_size: batch_size as u32,
            _pad:       0,
        };
        let scale = self.cfg.learning_rate / batch_size as f32;

        // ==== Fase 3a: Backward BPTT (sem atomics para g_t_f/g_lat_w/g_w_out) ====
        // Escreve d_pre_out[B,T,FLAT_DIM] e d_logits_out[B,V] para grad_acc consumir.
        // Mantém atomics apenas para g_i_f, g_nibble_h, g_nibble_l, g_b_out.
        // SAFETY: todos os ponteiros são buffers GPU válidos.
        unsafe {
            self.rt.dispatch_backward_bptt(
                params_bwd,
                self.buf_t_f.as_ptr()              as u64,   // read T_f
                self.buf_lat.as_ptr()               as u64,   // read lat_w
                self.buf_w_out.as_ptr()             as u64,   // read W_out
                self.pst_st.as_ptr()                as u64,   // read states_all [B, T+1, FLAT_DIM]
                self.pst_log.as_ptr()               as u64,   // read logits_in  [B, V]
                self.pst_win.as_ptr()               as u64,   // read windows    [B, T]
                self.pst_tgt.as_ptr()               as u64,   // read targets    [B]
                self.buf_g_i_f.as_mut_ptr()         as u64,   // g_i_f (atomic, baixo conflito)
                self.buf_g_nibble_h.as_mut_ptr()    as u64,   // g_nibble_h (atomic)
                self.buf_g_nibble_l.as_mut_ptr()    as u64,   // g_nibble_l (atomic)
                self.buf_g_b_out.as_mut_ptr()       as u64,   // g_b_out (atomic, muito baixo)
                self.pst_dpre.as_mut_ptr()          as u64,   // write d_pre_out
                self.pst_dlog.as_mut_ptr()          as u64,   // write d_logits_out
            ).expect("dispatch_backward_bptt");

            // Barrier GPU-side: todos os grad_acc só começam após backward_bptt completar.
            self.rt.barrier().expect("barrier bptt→grad_acc");

            // ==== Fase 3b: Grad accumulation (sem atomics) ====
            // Todos os 4 kernels são independentes entre si — GPU pipelina-os.
            self.rt.dispatch_grad_acc_tf_latw(
                params_grad,
                self.pst_dpre.as_ptr()        as u64,   // read d_pre_out
                self.pst_st.as_ptr()           as u64,   // read states_all
                self.buf_g_t_f.as_mut_ptr()   as u64,   // write g_t_f
                self.buf_g_lat_w.as_mut_ptr() as u64,   // write g_lat_w
            ).expect("dispatch_grad_acc_tf_latw");

            self.rt.dispatch_grad_acc_wout(
                params_grad,
                self.pst_st.as_ptr()          as u64,   // read states_all (estado final)
                self.pst_dlog.as_ptr()         as u64,   // read d_logits_out
                self.buf_g_w_out.as_mut_ptr() as u64,   // write g_w_out
            ).expect("dispatch_grad_acc_wout");

            self.rt.dispatch_grad_acc_b_f(
                params_grad,
                self.pst_dpre.as_ptr()        as u64,   // read d_pre_out
                self.buf_g_b_f.as_mut_ptr()   as u64,   // write g_b_f (sem atomic)
            ).expect("dispatch_grad_acc_b_f");

            self.rt.dispatch_grad_acc_pos(
                params_grad,
                self.pst_dpre.as_ptr()        as u64,   // read d_pre_out
                self.buf_g_pos.as_mut_ptr()   as u64,   // write g_pos (sem atomic)
            ).expect("dispatch_grad_acc_pos");

            self.rt.drain().expect("drain grad_acc");
            // drain() garante que TODOS os grads estão prontos antes do SGD.
        }

        // ==== Fase 4: SGD × 7 tensors ====
        // sgd_main: w[i] -= scale * g[i]; g[i] = 0 (zero inline para próximo step).
        // SAFETY: ponteiros de peso e grad são buffers GPU válidos.
        unsafe {
            macro_rules! sgd {
                ($n:expr, $w:expr, $g:expr) => {
                    self.rt.dispatch_sgd(
                        $n, scale,
                        $w.as_mut_ptr() as u64,
                        $g.as_mut_ptr() as u64,
                    ).expect(concat!("dispatch_sgd"));
                };
            }
            sgd!(self.n_t_f,    self.buf_t_f,      self.buf_g_t_f);
            sgd!(self.n_i_f,    self.buf_i_f,      self.buf_g_i_f);
            sgd!(self.n_nibble, self.buf_nibble_h, self.buf_g_nibble_h);   // high nibble
            sgd!(self.n_nibble, self.buf_nibble_l, self.buf_g_nibble_l);   // low  nibble
            sgd!(self.n_b_f,    self.buf_b_f,      self.buf_g_b_f);
            sgd!(self.n_pos,    self.buf_pos,      self.buf_g_pos);
            sgd!(self.n_lat_w,  self.buf_lat,      self.buf_g_lat_w);
            sgd!(self.n_w_out,  self.buf_w_out,    self.buf_g_w_out);
            sgd!(self.n_b_out,  self.buf_b_out,    self.buf_g_b_out);
            self.rt.drain().expect("drain sgd");
        }

        total_loss / batch_size as f32
    }

    /// Treina uma época completa.
    pub fn train_epoch(&mut self, pairs: &[(Vec<u32>, u32)], batch_size: usize) -> f32 {
        assert!(batch_size > 0, "batch_size deve ser > 0");
        assert!(
            batch_size <= self.max_batch,
            "batch_size={} > max_batch={}", batch_size, self.max_batch
        );
        let mut total = 0.0_f32;
        let mut n     = 0_usize;
        for chunk in pairs.chunks(batch_size) {
            if chunk.is_empty() { continue; }
            total += self.train_step(chunk);
            n     += 1;
        }
        if n > 0 { total / n as f32 } else { 0.0 }
    }
}
