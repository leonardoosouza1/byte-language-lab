//! Dispatch layer para os kernels SMG v3 via HSA AQL.
//!
//! # Kernarg layouts (extraídos via extract_layouts.py do .hsaco compilado)
//!
//! `forward_main`       kernarg_size=112 (16B struct + 12×8B ptrs; SEM hidden args)
//!   Adicionados nibble_h e nibble_l (2×8B) para geometric byte embedding.
//! `backward_main`      kernarg_size=128 (16B struct + 14×8B ptrs; SEM hidden args)
//! `sumsq_main`         kernarg_size=288 (32B expl. + hidden args em 32..98)
//! `zero_acc`           kernarg_size=8   (1×8B ptr; SEM hidden args)
//! `sgd_main`           kernarg_size=288 (32B expl. + hidden args em 32..98)
//! `zero_grads`         kernarg_size=288 (32B expl. + hidden args em 32..98)
//!
//! Kernels da Fase 2 BPTT (sem atomics pesados para g_t_f/g_lat_w/g_w_out):
//! `backward_bptt`      kernarg_size=120 (16B struct + 13×8B ptrs; SEM hidden args)
//!   g_nibble_h/g_nibble_l adicionados (+2×8B) para gradiente do nibble embedding.
//! `grad_acc_tf_latw`   kernarg_size=48  (16B struct + 4×8B ptrs; SEM hidden args)
//!   Grid: (128×128, 1, 1) — 128 WGs (um por par f×c), 16-way atomic ao final.
//! `grad_acc_wout`      kernarg_size=40  (16B struct + 3×8B ptrs; SEM hidden args)
//!   Grid: (ceil(FLAT_DIM×V/128)×128, 1, 1) — sem alteração.
//! `grad_acc_b_f`       kernarg_size=32  (16B struct + 2×8B ptrs; SEM hidden args)
//!   Grid: (128×32, 1, 1) — 128 WGs (um por par f×c), 16-way atomic ao final.
//! `grad_acc_pos`       kernarg_size=32  (16B struct + 2×8B ptrs; SEM hidden args)
//!   Grid: (128×32, 1, 1) — 128 WGs (um por par c×f), 8-way atomic ao final.
//!
//! Hidden args (para kernels que usam `blockDim.*` / `gridDim.*`):
//!   offset 32: block_count_x (u32) — n_blocks
//!   offset 36: block_count_y (u32) — 1
//!   offset 40: block_count_z (u32) — 1
//!   offset 44: group_size_x  (u16) — 256
//!   offset 46: group_size_y  (u16) — 1
//!   offset 48: group_size_z  (u16) — 1
//!   offset 50: remainder_x   (u16) — 0
//!   offset 52: remainder_y   (u16) — 0
//!   offset 54: remainder_z   (u16) — 0
//!   offset 96: grid_dims     (u16) — 3

use hip_ffi::hsa_dispatch::{HsaDispatcher, HsaError, KernelInfo};

// ---------------------------------------------------------------------------
// Bytes dos .hsaco compilados em build-time por `build.rs`.
// ---------------------------------------------------------------------------
static FORWARD_HSACO:   &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smg_forward.hsaco"));
static BACKWARD_HSACO:  &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smg_backward.hsaco"));
static BPTT_HSACO:      &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smg_backward_bptt.hsaco"));
static BPTT_BIT_HSACO:  &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smg_backward_bptt_bit.hsaco"));
static BPTT_KL_HSACO:   &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smg_backward_bptt_kl.hsaco"));
static GRAD_ACC_HSACO:  &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smg_grad_acc.hsaco"));
static NORM_HSACO:      &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smg_norm.hsaco"));
static SGD_HSACO:       &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smg_sgd.hsaco"));

// ---------------------------------------------------------------------------
// Kernarg sizes reais (validados contra o hsaco pelo check_karg em new()).
// Fase 1/2 (bptt + grad_acc): estimativas baseadas no layout explícito;
// confirmar com `python3 scripts/tools/extract_layouts.py` após compilação.
// ---------------------------------------------------------------------------
const KARG_FORWARD:       u32 = 112;   // 16 + 12×8 — nibble_h + nibble_l adicionados
const KARG_BACKWARD:      u32 = 128;
const KARG_BPTT:          u32 = 120;   // 16 + 13×8 — g_nibble_h + g_nibble_l adicionados
const KARG_GRAD_TF_LATW:  u32 = 48;    // 16 + 4×8  — SEM hidden args
const KARG_GRAD_WOUT:     u32 = 40;    // 16 + 3×8  — SEM hidden args
const KARG_GRAD_BF:       u32 = 32;    // 16 + 2×8  — SEM hidden args
const KARG_GRAD_POS:      u32 = 32;    // 16 + 2×8  — SEM hidden args
const KARG_SUMSQ:         u32 = 288;
const KARG_ZERO_ACC:      u32 = 8;
const KARG_SGD:           u32 = 288;
const KARG_ZERO_G:        u32 = 288;

// ---------------------------------------------------------------------------
// Repr(C) mirrors dos structs nos .hip files.
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ParamsFwd {
    pub vocab_size:   u32,
    pub t_steps:      u32,
    pub batch_size:   u32,
    pub cache_states: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ParamsBwd {
    pub vocab_size: u32,
    pub t_steps:    u32,
    pub batch_size: u32,
    pub _pad:       u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ParamsNorm {
    pub length: u32,
    pub _pad0:  u32,
    pub _pad1:  u32,
    pub _pad2:  u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ParamsSgd {
    pub length:     u32,
    pub scale_bits: u32,
    pub _pad0:      u32,
    pub _pad1:      u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ParamsGradAcc {
    pub vocab_size:  u32,
    pub t_steps:     u32,
    pub batch_size:  u32,
    pub _pad:        u32,
}

// ---------------------------------------------------------------------------
// SmgRuntime
// ---------------------------------------------------------------------------
pub struct SmgRuntime {
    pub disp:        HsaDispatcher,
    forward:         KernelInfo,
    backward:        KernelInfo,
    backward_bptt:   KernelInfo,   // BPTT softmax+CE
    backward_bptt_bit: KernelInfo, // BPTT bit-level BCE (marginal)
    backward_bptt_kl:  KernelInfo, // BPTT KL soft-target (distill)
    grad_tf_latw:    KernelInfo,
    grad_wout:       KernelInfo,
    grad_b_f:        KernelInfo,
    grad_pos:        KernelInfo,
    sumsq:           KernelInfo,
    zero_acc:        KernelInfo,
    sgd:             KernelInfo,
    zero_g:          KernelInfo,
}

/// Maior kernarg_size = 288 (sumsq/sgd). Slot de 512 para margem.
const KARG_SLOT_SIZE: usize = 512;
/// 128 slots = 64 KB de ring; suficiente para 1 train_step com T=48 + 7 grads.
const KARG_SLOTS: usize = 128;

impl SmgRuntime {
    pub fn new() -> Result<Self, HsaError> {
        let modules: &[&[u8]] = &[
            FORWARD_HSACO,
            BACKWARD_HSACO,
            BPTT_HSACO,
            BPTT_BIT_HSACO,
            BPTT_KL_HSACO,
            GRAD_ACC_HSACO,
            NORM_HSACO,
            SGD_HSACO,
        ];
        let mut disp = HsaDispatcher::new(modules, KARG_SLOT_SIZE, KARG_SLOTS)?;

        let forward            = disp.register("forward_main")?;
        let backward           = disp.register("backward_main")?;
        let backward_bptt      = disp.register("backward_bptt")?;
        let backward_bptt_bit  = disp.register("backward_bptt_bit")?;
        let backward_bptt_kl   = disp.register("backward_bptt_kl")?;
        let grad_tf_latw       = disp.register("grad_acc_tf_latw")?;
        let grad_wout          = disp.register("grad_acc_wout")?;
        let grad_b_f           = disp.register("grad_acc_b_f")?;
        let grad_pos           = disp.register("grad_acc_pos")?;
        let sumsq              = disp.register("sumsq_main")?;
        let zero_acc           = disp.register("zero_acc")?;
        let sgd                = disp.register("sgd_main")?;
        let zero_g             = disp.register("zero_grads")?;

        // Validação: aborta cedo se hipcc mudou o layout.
        // Para os novos kernels (bptt/grad_acc), os tamanhos esperados são estimativas
        // — se falhar, rodar `extract_layouts.py` e actualizar as constantes KARG_*.
        for (name, actual, expected) in [
            ("forward_main",      forward.kernarg_size,           KARG_FORWARD),
            ("backward_main",     backward.kernarg_size,          KARG_BACKWARD),
            ("backward_bptt",     backward_bptt.kernarg_size,     KARG_BPTT),
            ("backward_bptt_bit", backward_bptt_bit.kernarg_size, KARG_BPTT),
            ("backward_bptt_kl",  backward_bptt_kl.kernarg_size,  KARG_BPTT),
            ("grad_acc_tf_latw",  grad_tf_latw.kernarg_size,      KARG_GRAD_TF_LATW),
            ("grad_acc_wout",     grad_wout.kernarg_size,         KARG_GRAD_WOUT),
            ("grad_acc_b_f",      grad_b_f.kernarg_size,          KARG_GRAD_BF),
            ("grad_acc_pos",      grad_pos.kernarg_size,          KARG_GRAD_POS),
            ("sumsq_main",        sumsq.kernarg_size,             KARG_SUMSQ),
            ("zero_acc",          zero_acc.kernarg_size,          KARG_ZERO_ACC),
            ("sgd_main",          sgd.kernarg_size,               KARG_SGD),
            ("zero_grads",        zero_g.kernarg_size,            KARG_ZERO_G),
        ] {
            assert_eq!(
                actual, expected,
                "Kernel '{}': kernarg_size {}≠{}. Recompile e atualize KARG_* em dispatch.rs.",
                name, actual, expected
            );
        }

        Ok(Self { disp, forward, backward, backward_bptt, backward_bptt_bit, backward_bptt_kl,
                  grad_tf_latw, grad_wout, grad_b_f, grad_pos,
                  sumsq, zero_acc, sgd, zero_g })
    }

    // -----------------------------------------------------------------------
    // Helpers internos
    // -----------------------------------------------------------------------

    /// Escreve `val` no slot kernarg na posição `off`.
    ///
    /// # Safety
    /// `karg` deve apontar para um slot kernarg válido alocado por `next_kernarg`;
    /// `off + size_of::<T>()` deve caber dentro do tamanho do slot.
    #[inline(always)]
    unsafe fn write_at<T: Copy>(karg: *mut u8, off: usize, val: T) {
        // SAFETY: contrato do caller (ver doc).
        unsafe { std::ptr::write_unaligned(karg.add(off) as *mut T, val); }
    }

    /// Preenche hidden args para kernels que usam `blockDim.*` / `gridDim.*`
    /// (sumsq_main, sgd_main, zero_grads). Offsets extraídos via extract_layouts.py.
    ///
    /// # Safety
    /// Mesmo contrato de `write_at`; offsets 32..54 e 96 devem estar dentro do slot.
    unsafe fn fill_hidden_1d(karg: *mut u8, n_blocks: u32, group_size: u16) {
        // SAFETY: todos os offsets (32..54, 96) estão dentro de KARG_SLOT_SIZE=512.
        unsafe {
            // BlockCountX/Y/Z (u32 cada)
            Self::write_at::<u32>(karg, 32, n_blocks);
            Self::write_at::<u32>(karg, 36, 1u32);
            Self::write_at::<u32>(karg, 40, 1u32);
            // GroupSizeX/Y/Z (u16 cada)
            Self::write_at::<u16>(karg, 44, group_size);
            Self::write_at::<u16>(karg, 46, 1u16);
            Self::write_at::<u16>(karg, 48, 1u16);
            // RemainderX/Y/Z = 0 (u16 cada — já zero pelo next_kernarg zero-fill)
            // GridDims = 3 (u16 at offset 96)
            Self::write_at::<u16>(karg, 96, 3u16);
        }
    }

    // -----------------------------------------------------------------------
    // Dispatches públicos (async — chamar drain() no fim do step)
    // -----------------------------------------------------------------------

    /// forward_main — 1 workgroup por sample, 128 threads, 32 KB LDS.
    ///
    /// embed(byte) = i_f[byte] + nibble_h[byte>>4] + nibble_l[byte&0xF]
    /// → inductive bias geométrico: bytes em mesma zona ASCII partilham nibble_h;
    ///   bytes na mesma posição dentro da zona partilham nibble_l.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dispatch_forward(
        &self,
        params:     ParamsFwd,
        t_f:        u64, i_f:       u64,
        nibble_h:   u64, nibble_l:  u64,   // [N_FAMILIES, 16, K] cada
        b_f:        u64, pos_bias:  u64,
        lat_w:      u64, w_out:     u64,
        b_out:      u64, windows:   u64,
        logits_out: u64, states_out: u64,
    ) -> Result<(), HsaError> {
        // SAFETY: caller garante que todos os ponteiros (u64) são válidos em GPU memory.
        unsafe {
            let karg = self.disp.next_kernarg(KARG_FORWARD as usize);
            Self::write_at::<ParamsFwd>(karg,   0, params);
            Self::write_at::<u64>(karg,  16, t_f);
            Self::write_at::<u64>(karg,  24, i_f);
            Self::write_at::<u64>(karg,  32, nibble_h);   // NEW: high nibble embedding
            Self::write_at::<u64>(karg,  40, nibble_l);   // NEW: low  nibble embedding
            Self::write_at::<u64>(karg,  48, b_f);
            Self::write_at::<u64>(karg,  56, pos_bias);
            Self::write_at::<u64>(karg,  64, lat_w);
            Self::write_at::<u64>(karg,  72, w_out);
            Self::write_at::<u64>(karg,  80, b_out);
            Self::write_at::<u64>(karg,  88, windows);
            Self::write_at::<u64>(karg,  96, logits_out);
            Self::write_at::<u64>(karg, 104, states_out);
            // Sem hidden args para forward_main.
            // AQL: grid_size_x = TOTAL work-items (não blocos).
            // 1 workgroup = 1 sample; 128 threads/workgroup → grid = batch * 128.
            let grid = (params.batch_size * 128, 1, 1);
            self.disp.ctx.dispatch_async(&self.forward, karg, grid, (128, 1, 1), 0)
        }
    }

    /// backward_main — 1 workgroup por sample, 128 threads, 36 KB LDS.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dispatch_backward(
        &self,
        params:     ParamsBwd,
        t_f: u64, lat_w: u64, w_out: u64, states_all: u64, logits_in: u64,
        windows: u64, targets: u64,
        g_t_f: u64, g_i_f: u64, g_b_f: u64, g_pos: u64,
        g_lat_w: u64, g_w_out: u64, g_b_out: u64,
    ) -> Result<(), HsaError> {
        // SAFETY: caller garante que todos os ponteiros são válidos em GPU memory.
        unsafe {
            let karg = self.disp.next_kernarg(KARG_BACKWARD as usize);
            Self::write_at::<ParamsBwd>(karg, 0,   params);
            Self::write_at::<u64>(karg, 16,  t_f);
            Self::write_at::<u64>(karg, 24,  lat_w);
            Self::write_at::<u64>(karg, 32,  w_out);
            Self::write_at::<u64>(karg, 40,  states_all);
            Self::write_at::<u64>(karg, 48,  logits_in);
            Self::write_at::<u64>(karg, 56,  windows);
            Self::write_at::<u64>(karg, 64,  targets);
            Self::write_at::<u64>(karg, 72,  g_t_f);
            Self::write_at::<u64>(karg, 80,  g_i_f);
            Self::write_at::<u64>(karg, 88,  g_b_f);
            Self::write_at::<u64>(karg, 96,  g_pos);
            Self::write_at::<u64>(karg, 104, g_lat_w);
            Self::write_at::<u64>(karg, 112, g_w_out);
            Self::write_at::<u64>(karg, 120, g_b_out);
            // Sem hidden args para backward_main.
            // AQL: grid_size_x = TOTAL work-items. 1 wg/sample, 128 threads/wg.
            let grid = (params.batch_size * 128, 1, 1);
            self.disp.ctx.dispatch_async(&self.backward, karg, grid, (128, 1, 1), 0)
        }
    }

    // -----------------------------------------------------------------------
    // Fase 2 BPTT: dispatch dos kernels sem atomics para g_t_f/g_lat_w/g_w_out.
    // -----------------------------------------------------------------------

    /// backward_bptt — Fase 1 do backward refactored.
    /// Propaga BPTT e escreve d_pre_out[B,T,FLAT_DIM] + d_logits_out[B,V].
    /// Mantém atomics para tensores pequenos: g_i_f, g_nibble_h, g_nibble_l, g_b_out.
    ///
    /// Gradiente do nibble embedding:
    ///   d_nibble_h[f][byte>>4][k] += d_pre[k]   (mesmo grad que g_i_f)
    ///   d_nibble_l[f][byte&0xF][k] += d_pre[k]
    ///   Contention alta (16 bytes/slot) mas correcto — optimizar em Step 2 se necessário.
    ///
    /// Grid: (batch_size × 128, 1, 1) — 1 WG/sample, 128 threads/WG.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dispatch_backward_bptt(
        &self,
        params:        ParamsBwd,
        t_f:           u64, lat_w:        u64, w_out:         u64,
        states_all:    u64, logits_in:    u64,
        windows:       u64, targets:      u64,
        g_i_f:         u64,
        g_nibble_h:    u64, g_nibble_l:   u64,   // NEW: nibble grads
        g_b_out:       u64,
        d_pre_out:     u64, d_logits_out: u64,
    ) -> Result<(), HsaError> {
        // SAFETY: caller garante que todos os ponteiros são válidos em GPU memory.
        unsafe {
            let karg = self.disp.next_kernarg(KARG_BPTT as usize);
            Self::write_at::<ParamsBwd>(karg,   0, params);
            Self::write_at::<u64>(karg,  16, t_f);
            Self::write_at::<u64>(karg,  24, lat_w);
            Self::write_at::<u64>(karg,  32, w_out);
            Self::write_at::<u64>(karg,  40, states_all);
            Self::write_at::<u64>(karg,  48, logits_in);
            Self::write_at::<u64>(karg,  56, windows);
            Self::write_at::<u64>(karg,  64, targets);
            Self::write_at::<u64>(karg,  72, g_i_f);
            Self::write_at::<u64>(karg,  80, g_nibble_h);  // NEW
            Self::write_at::<u64>(karg,  88, g_nibble_l);  // NEW
            Self::write_at::<u64>(karg,  96, g_b_out);
            Self::write_at::<u64>(karg, 104, d_pre_out);
            Self::write_at::<u64>(karg, 112, d_logits_out);
            // Sem hidden args (usa apenas blockIdx.x e threadIdx.x).
            let grid = (params.batch_size * 128, 1, 1);
            self.disp.ctx.dispatch_async(&self.backward_bptt, karg, grid, (128, 1, 1), 0)
        }
    }

    /// dispatch_backward_bptt_bit — variante com BCE bit-level (marginal sobre softmax).
    /// Mesma assinatura/layout que `dispatch_backward_bptt`. Só muda o kernel chamado.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dispatch_backward_bptt_bit(
        &self,
        params:        ParamsBwd,
        t_f:           u64, lat_w:        u64, w_out:         u64,
        states_all:    u64, logits_in:    u64,
        windows:       u64, targets:      u64,
        g_i_f:         u64,
        g_nibble_h:    u64, g_nibble_l:   u64,
        g_b_out:       u64,
        d_pre_out:     u64, d_logits_out: u64,
    ) -> Result<(), HsaError> {
        unsafe {
            let karg = self.disp.next_kernarg(KARG_BPTT as usize);
            Self::write_at::<ParamsBwd>(karg,   0, params);
            Self::write_at::<u64>(karg,  16, t_f);
            Self::write_at::<u64>(karg,  24, lat_w);
            Self::write_at::<u64>(karg,  32, w_out);
            Self::write_at::<u64>(karg,  40, states_all);
            Self::write_at::<u64>(karg,  48, logits_in);
            Self::write_at::<u64>(karg,  56, windows);
            Self::write_at::<u64>(karg,  64, targets);
            Self::write_at::<u64>(karg,  72, g_i_f);
            Self::write_at::<u64>(karg,  80, g_nibble_h);
            Self::write_at::<u64>(karg,  88, g_nibble_l);
            Self::write_at::<u64>(karg,  96, g_b_out);
            Self::write_at::<u64>(karg, 104, d_pre_out);
            Self::write_at::<u64>(karg, 112, d_logits_out);
            let grid = (params.batch_size * 128, 1, 1);
            self.disp.ctx.dispatch_async(&self.backward_bptt_bit, karg, grid, (128, 1, 1), 0)
        }
    }

    /// dispatch_backward_bptt_kl — variante KL soft-target (distillation).
    /// `target_dist`: ponteiro pra [B, V] f32 (soft target distribution).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dispatch_backward_bptt_kl(
        &self,
        params:        ParamsBwd,
        t_f:           u64, lat_w:        u64, w_out:         u64,
        states_all:    u64, logits_in:    u64,
        windows:       u64, target_dist:  u64,   // f32* [B, V]
        g_i_f:         u64,
        g_nibble_h:    u64, g_nibble_l:   u64,
        g_b_out:       u64,
        d_pre_out:     u64, d_logits_out: u64,
    ) -> Result<(), HsaError> {
        unsafe {
            let karg = self.disp.next_kernarg(KARG_BPTT as usize);
            Self::write_at::<ParamsBwd>(karg,   0, params);
            Self::write_at::<u64>(karg,  16, t_f);
            Self::write_at::<u64>(karg,  24, lat_w);
            Self::write_at::<u64>(karg,  32, w_out);
            Self::write_at::<u64>(karg,  40, states_all);
            Self::write_at::<u64>(karg,  48, logits_in);
            Self::write_at::<u64>(karg,  56, windows);
            Self::write_at::<u64>(karg,  64, target_dist);
            Self::write_at::<u64>(karg,  72, g_i_f);
            Self::write_at::<u64>(karg,  80, g_nibble_h);
            Self::write_at::<u64>(karg,  88, g_nibble_l);
            Self::write_at::<u64>(karg,  96, g_b_out);
            Self::write_at::<u64>(karg, 104, d_pre_out);
            Self::write_at::<u64>(karg, 112, d_logits_out);
            let grid = (params.batch_size * 128, 1, 1);
            self.disp.ctx.dispatch_async(&self.backward_bptt_kl, karg, grid, (128, 1, 1), 0)
        }
    }

    /// grad_acc_tf_latw — Fase 2: acumula g_t_f e g_lat_w via GEMM-like reduce.
    ///
    /// Grid: (N_FAMILIES × N_CELLS = 128, 1, 1) — 128 WGs (um por par f×c).
    /// Escrita via atomic_add (16-way contention em 8.192 elementos — aceitável).
    /// Utilização: 128 WGs × 4 waves = 512 waves → 40 CUs totalmente activos.
    pub unsafe fn dispatch_grad_acc_tf_latw(
        &self,
        params:       ParamsGradAcc,
        d_pre_out:    u64,
        states_all:   u64,
        g_t_f:        u64,
        g_lat_w:      u64,
    ) -> Result<(), HsaError> {
        // SAFETY: caller garante que todos os ponteiros são válidos em GPU memory.
        unsafe {
            let karg = self.disp.next_kernarg(KARG_GRAD_TF_LATW as usize);
            Self::write_at::<ParamsGradAcc>(karg, 0,  params);
            Self::write_at::<u64>(karg, 16, d_pre_out);
            Self::write_at::<u64>(karg, 24, states_all);
            Self::write_at::<u64>(karg, 32, g_t_f);
            Self::write_at::<u64>(karg, 40, g_lat_w);
            // 128 WGs × 128 threads — sem hidden args.
            // blockIdx.x = f × N_CELLS + c (f=0..7, c=0..15).
            self.disp.ctx.dispatch_async(&self.grad_tf_latw, karg, (128 * 128, 1, 1), (128, 1, 1), 0)
        }
    }

    /// grad_acc_wout — Fase 2: acumula g_w_out via GEMM-like reduce.
    ///
    /// Sem atomics. Grid: (ceil(FLAT_DIM × V / 128) × 128, 1, 1).
    pub unsafe fn dispatch_grad_acc_wout(
        &self,
        params:       ParamsGradAcc,
        states_all:   u64,
        d_logits_out: u64,
        g_w_out:      u64,
    ) -> Result<(), HsaError> {
        // SAFETY: caller garante que todos os ponteiros são válidos em GPU memory.
        unsafe {
            const FLAT_DIM: u32 = 4096;
            let v          = params.vocab_size;
            let n_elems    = FLAT_DIM * v;
            let n_wg       = (n_elems + 127) / 128;
            let karg = self.disp.next_kernarg(KARG_GRAD_WOUT as usize);
            Self::write_at::<ParamsGradAcc>(karg, 0,  params);
            Self::write_at::<u64>(karg, 16, states_all);
            Self::write_at::<u64>(karg, 24, d_logits_out);
            Self::write_at::<u64>(karg, 32, g_w_out);
            // Sem hidden args (usa constante 128u em vez de blockDim.x).
            self.disp.ctx.dispatch_async(&self.grad_wout, karg, (n_wg * 128, 1, 1), (128, 1, 1), 0)
        }
    }

    /// grad_acc_b_f — Fase 2: acumula g_b_f[N_FAMILIES, K] via redução por (f,c).
    ///
    /// Grid: (N_FAMILIES × N_CELLS = 128, 1, 1) × Block (K=32, 1, 1) — 1 wave/WG.
    /// Escrita via atomic_add (16-way contention em 256 elementos — negligível).
    pub unsafe fn dispatch_grad_acc_b_f(
        &self,
        params:    ParamsGradAcc,
        d_pre_out: u64,
        g_b_f:     u64,
    ) -> Result<(), HsaError> {
        // SAFETY: caller garante que todos os ponteiros são válidos em GPU memory.
        unsafe {
            let karg = self.disp.next_kernarg(KARG_GRAD_BF as usize);
            Self::write_at::<ParamsGradAcc>(karg, 0,  params);
            Self::write_at::<u64>(karg, 16, d_pre_out);
            Self::write_at::<u64>(karg, 24, g_b_f);
            // 128 WGs × 32 threads — sem hidden args.
            // blockIdx.x = f × N_CELLS + c (f=0..7, c=0..15).
            self.disp.ctx.dispatch_async(&self.grad_b_f, karg, (128 * 32, 1, 1), (32, 1, 1), 0)
        }
    }

    /// grad_acc_pos — Fase 2: acumula g_pos[N_CELLS, K] via redução por (c,f).
    ///
    /// Grid: (N_CELLS × N_FAMILIES = 128, 1, 1) × Block (K=32, 1, 1) — 1 wave/WG.
    /// Escrita via atomic_add (8-way contention em 512 elementos — negligível).
    pub unsafe fn dispatch_grad_acc_pos(
        &self,
        params:    ParamsGradAcc,
        d_pre_out: u64,
        g_pos:     u64,
    ) -> Result<(), HsaError> {
        // SAFETY: caller garante que todos os ponteiros são válidos em GPU memory.
        unsafe {
            let karg = self.disp.next_kernarg(KARG_GRAD_POS as usize);
            Self::write_at::<ParamsGradAcc>(karg, 0,  params);
            Self::write_at::<u64>(karg, 16, d_pre_out);
            Self::write_at::<u64>(karg, 24, g_pos);
            // 128 WGs × 32 threads — sem hidden args.
            // blockIdx.x = c × N_FAMILIES + f (c=0..15, f=0..7).
            self.disp.ctx.dispatch_async(&self.grad_pos, karg, (128 * 32, 1, 1), (32, 1, 1), 0)
        }
    }

    /// sumsq_main — blocks de 256 threads, 1 acumula por workgroup.
    pub unsafe fn dispatch_sumsq(
        &self,
        length:      u32,
        g:           u64,
        norm_sq_acc: u64,
    ) -> Result<(), HsaError> {
        // SAFETY: ponteiros válidos em GPU memory.
        unsafe {
            let params = ParamsNorm { length, _pad0: 0, _pad1: 0, _pad2: 0 };
            let karg   = self.disp.next_kernarg(KARG_SUMSQ as usize);
            Self::write_at::<ParamsNorm>(karg, 0,  params);
            Self::write_at::<u64>(karg, 16, g);
            Self::write_at::<u64>(karg, 24, norm_sq_acc);
            let n_blocks = (length + 255) / 256;
            Self::fill_hidden_1d(karg, n_blocks, 256);
            let grid_wi = (n_blocks * 256, 1, 1);
            self.disp.ctx.dispatch_async(&self.sumsq, karg, grid_wi, (256, 1, 1), 0)
        }
    }

    /// zero_acc — 1 thread, zera norm_sq_acc.
    pub unsafe fn dispatch_zero_acc(&self, norm_sq_acc: u64) -> Result<(), HsaError> {
        // SAFETY: ponteiro válido em GPU memory.
        unsafe {
            let karg = self.disp.next_kernarg(KARG_ZERO_ACC as usize);
            Self::write_at::<u64>(karg, 0, norm_sq_acc);
            self.disp.ctx.dispatch_async(&self.zero_acc, karg, (1, 1, 1), (1, 1, 1), 0)
        }
    }

    /// sgd_main — step + zero_grads inline, blocks de 256.
    pub unsafe fn dispatch_sgd(
        &self,
        length: u32,
        scale:  f32,
        w:      u64,
        g:      u64,
    ) -> Result<(), HsaError> {
        // SAFETY: ponteiros válidos em GPU memory.
        unsafe {
            let params = ParamsSgd { length, scale_bits: scale.to_bits(), _pad0: 0, _pad1: 0 };
            let karg   = self.disp.next_kernarg(KARG_SGD as usize);
            Self::write_at::<ParamsSgd>(karg, 0,  params);
            Self::write_at::<u64>(karg, 16, w);
            Self::write_at::<u64>(karg, 24, g);
            let n_blocks = (length + 255) / 256;
            Self::fill_hidden_1d(karg, n_blocks, 256);
            let grid_wi = (n_blocks * 256, 1, 1);
            self.disp.ctx.dispatch_async(&self.sgd, karg, grid_wi, (256, 1, 1), 0)
        }
    }

    /// zero_grads — zera g[] sem step (ex.: início do treino).
    pub unsafe fn dispatch_zero_grads(
        &self,
        length: u32,
        w:      u64,
        g:      u64,
    ) -> Result<(), HsaError> {
        // SAFETY: ponteiros válidos em GPU memory.
        unsafe {
            let params = ParamsSgd { length, scale_bits: 0, _pad0: 0, _pad1: 0 };
            let karg   = self.disp.next_kernarg(KARG_ZERO_G as usize);
            Self::write_at::<ParamsSgd>(karg, 0,  params);
            Self::write_at::<u64>(karg, 16, w);
            Self::write_at::<u64>(karg, 24, g);
            let n_blocks = (length + 255) / 256;
            Self::fill_hidden_1d(karg, n_blocks, 256);
            let grid_wi = (n_blocks * 256, 1, 1);
            self.disp.ctx.dispatch_async(&self.zero_g, karg, grid_wi, (256, 1, 1), 0)
        }
    }

    /// Insere um barrier_and packet na queue (sem stall de CPU).
    ///
    /// ⚠️  **NÃO USAR em sequência com drain()** — ambos partilham o mesmo
    /// `completion_signal`. Se `barrier()` for chamado ANTES de `drain()`,
    /// o barrier_and de `barrier()` decrementa o signal quando bptt termina;
    /// `drain()` armazena 1 no signal e submete outro barrier_and, mas o wait
    /// vê 0 (decrementado pelo barrier anterior) e retorna ANTES dos kernels
    /// grad_acc terminarem → drain prematuro → use-after-free no buf_log da
    /// validação → page fault @ 0x100000000.
    ///
    /// **Alternativa correcta**: Os dispatch packets já têm o bit BARRIER setado,
    /// garantindo ordering automático numa fila QUEUE_TYPE_SINGLE. Não é necessário
    /// nenhum barrier explícito — basta chamar `drain()` no fim de cada grupo.
    #[allow(dead_code)]
    pub fn barrier(&self) -> Result<(), HsaError> {
        self.disp.submit_barrier()
    }

    /// Drena todos os dispatches pendentes (barrier_and + wait).
    /// Chamar 1× no fim de cada train_step.
    pub fn drain(&self) -> Result<(), HsaError> {
        self.disp.submit_barrier_and_wait()
    }
}
