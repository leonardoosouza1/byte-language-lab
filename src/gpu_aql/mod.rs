//! # GPU runtime — HSA AQL backend (RDNA2 gfx1031).
//!
//! Bypass wgpu/Vulkan. Dispatch direto via packets AQL escritos no ring buffer
//! da hardware queue (com doorbell signal MMIO). Stack: `hip-ffi` →
//! `HsaDispatcher` → `dispatch_async` + `submit_barrier_and_wait`.
//!
//! Por que? Pesquisa em `docs/research/hsa-aql-spec.md` mostra:
//!   - wgpu/Vulkan/RADV: ~50 ms por dispatch (driver overhead).
//!   - HIP sync: 16.33 µs.
//!   - HSA sync: 11.96 µs.
//!   - HSA async batched: 2.61 µs ← target.
//!
//! Este módulo é gated em `feature = "gpu-aql"`. Build sem ROCm continua
//! funcionando via wgpu (`src/gpu/`).

#![allow(missing_docs)] // será documentado quando estabilizar

pub mod dispatch;
pub use dispatch::{SmgRuntime, ParamsFwd, ParamsBwd, ParamsNorm, ParamsSgd};

pub mod smg_grid_hsa;
pub use smg_grid_hsa::{SmgGridHsa, WeightsHost, GridInitError};

use std::sync::Arc;

use hip_ffi::hsa_dispatch::{HsaDispatcher, HsaError};

/// Handle para o runtime HSA. Construído uma vez por processo.
///
/// Wrap em `Arc` ao compartilhar entre threads — `HsaDispatcher` já é `Send + Sync`.
pub struct AqlContext {
    /// Dispatcher HSA com módulos pré-carregados e kernarg ring buffer.
    pub dispatcher: HsaDispatcher,
    /// Nome do agente GPU (ex.: "gfx1031").
    pub agent_name: String,
}

impl AqlContext {
    /// Inicializa runtime HSA carregando os módulos `.hsaco` compilados pelo
    /// `build.rs`. Aloca kernarg ring com `karg_slots` slots de `karg_slot_size`
    /// bytes cada — dimensionado para o maior kernarg buffer dos kernels SMG v3.
    ///
    /// # Errors
    ///
    /// Retorna [`HsaError`] se HSA falhar em init, falhar em carregar algum
    /// módulo, ou se nenhum agente GPU for encontrado.
    pub fn new() -> Result<Self, HsaError> {
        // Em fase 1 ainda não temos kernels compilados — passa lista vazia
        // só pra validar init/teardown do HSA + smoke-test do agente.
        let modules: Vec<&[u8]> = Vec::new();

        // Slot size 4 KiB e 64 slots = 256 KiB ring. Suficiente para kernels
        // SMG v3 que tem ~25 ponteiros + scalars (≤ 256 B kernarg cada).
        // Será refinado em fase 2 com base no maior layout extraído.
        let dispatcher = HsaDispatcher::new(&modules, 4096, 64)?;

        // Agent name é só para diagnóstico; HsaDispatcher já validou que existe.
        // (Detalhe exato exposto em fase posterior quando precisarmos.)
        let agent_name = "gfx1031".to_string();

        Ok(Self {
            dispatcher,
            agent_name,
        })
    }

    /// Wrap em `Arc` para compartilhamento entre `SmgGridGpu` e workers.
    #[must_use]
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }
}

/// Hardware specs do RDNA2 gfx1031 (RX 6750 XT). Constantes de compile-time
/// usadas para sizing de workgroups, LDS layout, occupancy planning.
pub mod hw {
    /// 40 CUs, 20 WGPs.
    pub const N_CUS: u32 = 40;
    /// Wavefront nativo em RDNA2 (configurado via `-mwavefrontsize32`).
    pub const WAVE_SIZE: u32 = 32;
    /// LDS por workgroup em WGP mode (2× CU mode).
    pub const LDS_BYTES_PER_WG: u32 = 65_536;
    /// LDS efetivo do shader SMG v3 (state_curr 16 KiB + lat_proj 16 KiB).
    pub const SMG_LDS_BYTES: u32 = 32_768;
    /// Wavefronts por workgroup nos kernels SMG v3 (128 threads / wave32).
    pub const SMG_WAVES_PER_WG: u32 = 4;
}
