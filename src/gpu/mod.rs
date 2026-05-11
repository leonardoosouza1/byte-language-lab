//! # GPU runtime — wgpu/WGSL para SMG v3 em RDNA2.
//!
//! Por que wgpu? Porta-bilidade (Vulkan/Metal/DX12), e RDNA2 do dev usa
//! Vulkan stack (mesma stack do projeto LSH, onde validamos matmul WGSL
//! 470× speedup vs CPU naïve).
//!
//! Setup:
//! - `GpuContext` é construído sincronamente via `pollster::block_on` —
//!   nada de async espalhado pelo código.
//! - `required_limits.max_compute_workgroup_storage_size = 32_768` (LDS por
//!   workgroup): nosso shader usa 32KB (state_curr 16KB + lat_proj 16KB).
//!   Default wgpu = 16384, RDNA2 suporta 65536.
//! - `required_features` vazio — não precisamos de extensões.

use std::sync::Arc;

/// Handle para device/queue da GPU. Construído uma vez por processo.
pub struct GpuContext {
    /// Device wgpu (Vulkan no Linux+RDNA2).
    pub device: Arc<wgpu::Device>,
    /// Fila de comandos.
    pub queue: Arc<wgpu::Queue>,
    /// Info do adapter (nome/backend) — útil para logs.
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Inicializa GPU com limites custom para suportar 32KB LDS por workgroup.
    ///
    /// # Errors
    ///
    /// Retorna erro se adapter ou device não puderem ser inicializados.
    pub fn new() -> Result<Self, GpuInitError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuInitError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuInitError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();

        // Pedimos 36KB de workgroup storage (forward 32KB + backward d_logits 4KB).
        // Backward usa 15 storage bindings → bump max_storage_buffers_per_shader_stage.
        let mut required_limits = wgpu::Limits::downlevel_defaults();
        required_limits.max_compute_workgroup_storage_size =
            36_864.max(required_limits.max_compute_workgroup_storage_size);
        required_limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;
        required_limits.max_buffer_size = adapter_limits.max_buffer_size;
        required_limits.max_compute_invocations_per_workgroup =
            adapter_limits.max_compute_invocations_per_workgroup;
        required_limits.max_compute_workgroup_size_x =
            adapter_limits.max_compute_workgroup_size_x;
        required_limits.max_storage_buffers_per_shader_stage =
            adapter_limits.max_storage_buffers_per_shader_stage;
        required_limits.max_bind_groups = adapter_limits.max_bind_groups;
        required_limits.max_bindings_per_bind_group =
            adapter_limits.max_bindings_per_bind_group;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("blab-smg-v3"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None, // trace path
            )
            .await
            .map_err(|e| GpuInitError::DeviceRequest(format!("{e:?}")))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }

    /// Nome amigável do dispositivo (ex.: "AMD Radeon RX 6700 XT (RADV NAVI22)").
    #[must_use]
    pub fn device_name(&self) -> String {
        format!(
            "{} via {:?}",
            self.adapter_info.name, self.adapter_info.backend
        )
    }
}

/// Erros de inicialização do `GpuContext`.
#[derive(Debug)]
pub enum GpuInitError {
    /// Nenhum adapter compatível.
    NoAdapter,
    /// Falha em `request_device`.
    DeviceRequest(String),
}

impl std::fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no compatible GPU adapter found"),
            Self::DeviceRequest(s) => write!(f, "GPU device request failed: {s}"),
        }
    }
}

impl std::error::Error for GpuInitError {}
