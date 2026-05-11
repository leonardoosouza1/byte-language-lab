//! # SmgGridGpu — espelho GPU de `microcolumn_v3::SmgGrid`, treino 100% GPU.
//!
//! Pipelines persistentes:
//!   * `forward` — calcula logits e (em modo treino) cacheia `states_all` em VRAM.
//!   * `backward` — recomputa softmax, acumula d_logits→grads via atomic-CAS f32.
//!   * `sgd` — aplica `w -= scale * g` por tensor.
//!   * `zero_grads` — zera buffers atomic<u32>.
//!
//! Loop de treino fica inteiramente na GPU (1 readback por epoch para o loss).
//!
//! Layout dos buffers de gradientes: `array<atomic<u32>>` com mesma cardinalidade
//! do tensor de peso correspondente; backward acumula via atomic CAS sobre f32.

use crate::gpu::GpuContext;
use crate::learn::microcolumn_v3::{
    FLAT_DIM, FSMS_PER_CELL, GridConfig, K, N_CELLS, N_FAMILIES, SmgGrid,
};
use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsFwd {
    vocab_size: u32,
    t_steps: u32,
    batch_size: u32,
    cache_states: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsBwd {
    vocab_size: u32,
    t_steps: u32,
    batch_size: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsSgd {
    length: u32,
    scale_bits: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsNorm {
    length: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Identifica um par (peso, grad). Fields são strings para labeling/debug.
struct ParamSlot {
    label: &'static str,
    w_buf: wgpu::Buffer,
    g_buf: wgpu::Buffer,
    n_elements: u32,
}

/// SmgGrid em GPU com forward + backward + SGD + gradient clipping.
pub struct SmgGridGpu {
    cfg: GridConfig,
    ctx: Arc<GpuContext>,

    /// Limite de norma L2 global do gradiente (clipping). 0.0 = sem clipping.
    pub max_grad_norm: f32,

    // Pipelines persistentes.
    pipe_fwd: wgpu::ComputePipeline,
    pipe_bwd: wgpu::ComputePipeline,
    pipe_sgd: wgpu::ComputePipeline,
    pipe_zero: wgpu::ComputePipeline,
    pipe_sumsq: wgpu::ComputePipeline,
    pipe_zero_acc: wgpu::ComputePipeline,

    bgl_fwd: wgpu::BindGroupLayout,
    bgl_bwd: wgpu::BindGroupLayout,
    bgl_sgd: wgpu::BindGroupLayout,
    bgl_norm: wgpu::BindGroupLayout,

    // Buffer 1-element f32 (storage atomic<u32>) para accumulator de norma.
    buf_norm_acc: wgpu::Buffer,
    // Staging para readback da norma global.
    staging_norm: wgpu::Buffer,

    // Pesos e seus grads emparelhados.
    p_t_f: ParamSlot,
    p_i_f: ParamSlot,
    p_b_f: ParamSlot,
    p_pos_bias: ParamSlot,
    p_lat_w: ParamSlot,
    p_w_out: ParamSlot,
    p_b_out: ParamSlot,
}

impl SmgGridGpu {
    /// Constrói a partir de um `SmgGrid` CPU (copia pesos para a GPU).
    #[must_use]
    pub fn from_cpu(cpu: &SmgGrid, ctx: Arc<GpuContext>) -> Self {
        let device = &ctx.device;
        let v = cpu.cfg.vocab_size;

        // ---- Shaders ----
        let mod_fwd = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smg_v3_fwd"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../gpu/shaders/microcolumn_v3_forward.wgsl"
            ))),
        });
        let mod_bwd = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smg_v3_bwd"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../gpu/shaders/microcolumn_v3_backward.wgsl"
            ))),
        });
        let mod_sgd = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smg_v3_sgd"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../gpu/shaders/microcolumn_v3_sgd.wgsl"
            ))),
        });
        let mod_norm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smg_v3_norm"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../gpu/shaders/microcolumn_v3_norm.wgsl"
            ))),
        });

        let bgl_fwd = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smg_v3_fwd_bgl"),
            entries: &Self::fwd_bgl_entries(),
        });
        let bgl_bwd = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smg_v3_bwd_bgl"),
            entries: &Self::bwd_bgl_entries(),
        });
        let bgl_sgd = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smg_v3_sgd_bgl"),
            entries: &Self::sgd_bgl_entries(),
        });
        let bgl_norm = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smg_v3_norm_bgl"),
            entries: &Self::norm_bgl_entries(),
        });

        let pl_fwd = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("smg_v3_fwd_pl"),
            bind_group_layouts: &[&bgl_fwd],
            push_constant_ranges: &[],
        });
        let pl_bwd = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("smg_v3_bwd_pl"),
            bind_group_layouts: &[&bgl_bwd],
            push_constant_ranges: &[],
        });
        let pl_sgd = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("smg_v3_sgd_pl"),
            bind_group_layouts: &[&bgl_sgd],
            push_constant_ranges: &[],
        });
        let pl_norm = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("smg_v3_norm_pl"),
            bind_group_layouts: &[&bgl_norm],
            push_constant_ranges: &[],
        });

        let pipe_fwd = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("smg_v3_fwd_pipe"),
            layout: Some(&pl_fwd),
            module: &mod_fwd,
            entry_point: Some("forward_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipe_bwd = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("smg_v3_bwd_pipe"),
            layout: Some(&pl_bwd),
            module: &mod_bwd,
            entry_point: Some("backward_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipe_sgd = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("smg_v3_sgd_pipe"),
            layout: Some(&pl_sgd),
            module: &mod_sgd,
            entry_point: Some("sgd_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipe_zero = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("smg_v3_zero_pipe"),
            layout: Some(&pl_sgd),
            module: &mod_sgd,
            entry_point: Some("zero_grads"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipe_sumsq = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("smg_v3_sumsq_pipe"),
            layout: Some(&pl_norm),
            module: &mod_norm,
            entry_point: Some("sumsq_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipe_zero_acc = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("smg_v3_zero_acc_pipe"),
            layout: Some(&pl_norm),
            module: &mod_norm,
            entry_point: Some("zero_acc"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Buffer accumulator de norma + staging para readback (4 bytes).
        let buf_norm_acc = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("norm_sq_acc"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_norm = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_norm"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ---- Buffers de peso + grad pareados ----
        let mk_param = |label: &'static str, data: &[f32]| -> ParamSlot {
            let w_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
            let g_zero = vec![0_u32; data.len()];
            let g_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("g"),
                contents: bytemuck::cast_slice(&g_zero),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            ParamSlot {
                label,
                w_buf,
                g_buf,
                n_elements: data.len() as u32,
            }
        };

        let p_t_f = mk_param("t_f", cpu.weights_t_f());
        let p_i_f = mk_param("i_f", cpu.weights_i_f());
        let p_b_f = mk_param("b_f", cpu.weights_b_f());
        let p_pos_bias = mk_param("pos_bias", cpu.weights_pos_bias());
        let p_lat_w = mk_param("lat_w", cpu.weights_lat_w());
        let p_w_out = mk_param("w_out", cpu.weights_w_out());
        let p_b_out = mk_param("b_out", cpu.weights_b_out());

        debug_assert_eq!(cpu.weights_t_f().len(), N_FAMILIES * K * K);
        debug_assert_eq!(cpu.weights_i_f().len(), N_FAMILIES * v * K);
        debug_assert_eq!(cpu.weights_b_f().len(), N_FAMILIES * K);
        debug_assert_eq!(cpu.weights_pos_bias().len(), N_CELLS * K);
        debug_assert_eq!(cpu.weights_lat_w().len(), N_FAMILIES * K * K);
        debug_assert_eq!(cpu.weights_w_out().len(), FLAT_DIM * v);
        debug_assert_eq!(cpu.weights_b_out().len(), v);
        let _ = FSMS_PER_CELL;

        Self {
            cfg: cpu.cfg.clone(),
            ctx,
            max_grad_norm: 20.0,
            pipe_fwd,
            pipe_bwd,
            pipe_sgd,
            pipe_zero,
            pipe_sumsq,
            pipe_zero_acc,
            bgl_fwd,
            bgl_bwd,
            bgl_sgd,
            bgl_norm,
            buf_norm_acc,
            staging_norm,
            p_t_f,
            p_i_f,
            p_b_f,
            p_pos_bias,
            p_lat_w,
            p_w_out,
            p_b_out,
        }
    }

    // ---------- Bind group layouts ----------
    fn fwd_bgl_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
        let st_ro = storage_ro();
        let st_rw = storage_rw();
        let uni = uniform();
        vec![
            entry(0, uni),
            entry(1, st_ro), entry(2, st_ro), entry(3, st_ro), entry(4, st_ro),
            entry(5, st_ro), entry(6, st_ro), entry(7, st_ro), entry(8, st_ro),
            entry(9, st_rw),     // logits_out
            entry(10, st_rw),    // states_out (cache)
        ]
    }
    fn bwd_bgl_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
        let st_ro = storage_ro();
        let st_rw = storage_rw();
        let uni = uniform();
        vec![
            entry(0, uni),
            entry(1, st_ro), entry(2, st_ro), entry(3, st_ro), // t_f, lat_w, w_out
            entry(4, st_ro), entry(5, st_ro), entry(6, st_ro), entry(7, st_ro),  // states/logits/win/tgt
            entry(8, st_rw), entry(9, st_rw), entry(10, st_rw), entry(11, st_rw),
            entry(12, st_rw), entry(13, st_rw), entry(14, st_rw),  // 7 grads
        ]
    }
    fn sgd_bgl_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
        let st_rw = storage_rw();
        let uni = uniform();
        vec![entry(0, uni), entry(1, st_rw), entry(2, st_rw)]
    }
    fn norm_bgl_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
        let st_ro = storage_ro();
        let st_rw = storage_rw();
        let uni = uniform();
        vec![entry(0, uni), entry(1, st_ro), entry(2, st_rw)]
    }

    /// Configuração subjacente.
    #[must_use]
    pub const fn cfg(&self) -> &GridConfig {
        &self.cfg
    }

    /// Contexto GPU.
    #[must_use]
    pub fn ctx(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    // ============================================================
    // FORWARD (público)
    // ============================================================

    /// Forward de janela única → logits. Sem cache de estados (modo inferência).
    #[must_use]
    pub fn forward(&self, window: &[u32]) -> Vec<f32> {
        let mut out = self.forward_batch(std::slice::from_ref(&window.to_vec()));
        out.pop().expect("forward_batch retornou vazio")
    }

    /// Argmax do logits para uma janela única (espelha `SmgGrid::predict`).
    #[must_use]
    pub fn predict(&self, window: &[u32]) -> u32 {
        let logits = self.forward(window);
        let mut best = 0u32;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &x) in logits.iter().enumerate() {
            if x > best_v {
                best_v = x;
                best = i as u32;
            }
        }
        best
    }

    /// Total de elementos somados sobre todos os tensores treináveis.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.iter_param_slots()
            .iter()
            .map(|s| s.n_elements as usize)
            .sum()
    }

    /// Forward em batch sem cache de estados (modo inferência).
    #[must_use]
    pub fn forward_batch(&self, windows: &[Vec<u32>]) -> Vec<Vec<f32>> {
        self.forward_internal(windows, false).0
    }

    /// Forward em batch que cacheia `states_all` em VRAM (preparação do backward).
    /// Retorna (logits_per_sample, states_buffer, win_buffer).
    fn forward_with_cache(
        &self,
        windows: &[Vec<u32>],
    ) -> (Vec<Vec<f32>>, wgpu::Buffer, wgpu::Buffer) {
        let (logits, ext) = self.forward_internal(windows, true);
        let (states, win) = ext.expect("forward com cache deve retornar buffers");
        (logits, states, win)
    }

    /// Implementação compartilhada. Quando `cache=true`, devolve `(states_buf, windows_buf)`
    /// para uso pelo backward.
    fn forward_internal(
        &self,
        windows: &[Vec<u32>],
        cache: bool,
    ) -> (Vec<Vec<f32>>, Option<(wgpu::Buffer, wgpu::Buffer)>) {
        assert!(!windows.is_empty(), "batch vazio");
        let t_steps = windows[0].len();
        assert!(
            windows.iter().all(|w| w.len() == t_steps),
            "todas as janelas precisam ter o mesmo comprimento"
        );
        let batch_size = windows.len();
        let v = self.cfg.vocab_size;
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let mut win_flat = Vec::with_capacity(batch_size * t_steps);
        for w in windows {
            win_flat.extend_from_slice(w);
        }
        let buf_windows = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("windows"),
            contents: bytemuck::cast_slice(&win_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let logits_bytes = (batch_size * v * 4) as u64;
        let buf_logits = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("logits_out"),
            size: logits_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Cache de estados — sempre alocado (binding obrigatório), mas só preenchido se cache=true.
        let states_bytes = if cache {
            (batch_size * (t_steps + 1) * FLAT_DIM * 4) as u64
        } else {
            // Mínimo válido: 1 elemento (binding precisa existir).
            (4) as u64
        };
        let buf_states = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("states_all"),
            size: states_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = ParamsFwd {
            vocab_size: v as u32,
            t_steps: t_steps as u32,
            batch_size: batch_size as u32,
            cache_states: u32::from(cache),
        };
        let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_fwd"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smg_v3_fwd_bg"),
            layout: &self.bgl_fwd,
            entries: &[
                bg_entry(0, &buf_params),
                bg_entry(1, &self.p_t_f.w_buf),
                bg_entry(2, &self.p_i_f.w_buf),
                bg_entry(3, &self.p_b_f.w_buf),
                bg_entry(4, &self.p_pos_bias.w_buf),
                bg_entry(5, &self.p_lat_w.w_buf),
                bg_entry(6, &self.p_w_out.w_buf),
                bg_entry(7, &self.p_b_out.w_buf),
                bg_entry(8, &buf_windows),
                bg_entry(9, &buf_logits),
                bg_entry(10, &buf_states),
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("smg_v3_fwd_enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("smg_v3_fwd_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_fwd);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(batch_size as u32, 1, 1);
        }

        // Readback dos logits.
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_logits"),
            size: logits_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&buf_logits, 0, &staging, 0, logits_bytes);
        queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map_async canal").expect("map_async erro");
        let data = slice.get_mapped_range();
        let flat: &[f32] = bytemuck::cast_slice(&data);
        let mut out = Vec::with_capacity(batch_size);
        for s in 0..batch_size {
            out.push(flat[s * v..(s + 1) * v].to_vec());
        }
        drop(data);
        staging.unmap();

        let extra = if cache {
            Some((buf_states, buf_windows))
        } else {
            None
        };
        (out, extra)
    }

    // ============================================================
    // TRAIN STEP
    // ============================================================

    /// Um step de treino (forward+cache, backward, SGD) sobre um mini-batch.
    /// Retorna `sum_loss` do batch (somatório, não média; divida por len).
    ///
    /// Internamente:
    /// - dispatcha forward (cache=1)
    /// - faz readback dos logits para computar loss host-side (cross-entropy)
    /// - dispatcha backward (acumula grads atomically)
    /// - dispatcha SGD por tensor com `scale = lr / batch_size`
    /// - dispatcha zero_grads em todos os 7 tensores
    pub fn train_step(&self, batch: &[(Vec<u32>, u32)]) -> f32 {
        assert!(!batch.is_empty(), "batch vazio");
        let t_steps = batch[0].0.len();
        let batch_size = batch.len();
        let v = self.cfg.vocab_size;
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        // Forward com cache.
        let windows: Vec<Vec<u32>> = batch.iter().map(|(w, _)| w.clone()).collect();
        let (logits_per_sample, buf_states, buf_windows) = self.forward_with_cache(&windows);

        // Loss host-side (precisamos para reportagem; barato — só V valores por sample).
        let mut sum_loss = 0.0_f32;
        for (i, (_, target)) in batch.iter().enumerate() {
            let logits = &logits_per_sample[i];
            let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&x| (x - mx).exp()).collect();
            let s: f32 = exps.iter().sum();
            let t = (*target as usize).min(v - 1);
            sum_loss -= (exps[t] / s).max(1e-20).ln();
        }

        // Targets buffer (u32 batched).
        let targets_u32: Vec<u32> = batch.iter().map(|(_, t)| *t).collect();
        let buf_targets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("targets"),
            contents: bytemuck::cast_slice(&targets_u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Logits-buffer para backward: re-upload (barato — V*B floats).
        let mut logits_flat: Vec<f32> = Vec::with_capacity(batch_size * v);
        for l in &logits_per_sample {
            logits_flat.extend_from_slice(l);
        }
        let buf_logits = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("logits_in_bwd"),
            contents: bytemuck::cast_slice(&logits_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let pb = ParamsBwd {
            vocab_size: v as u32,
            t_steps: t_steps as u32,
            batch_size: batch_size as u32,
            _pad: 0,
        };
        let buf_pb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_bwd"),
            contents: bytemuck::bytes_of(&pb),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_bwd = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smg_v3_bwd_bg"),
            layout: &self.bgl_bwd,
            entries: &[
                bg_entry(0, &buf_pb),
                bg_entry(1, &self.p_t_f.w_buf),
                bg_entry(2, &self.p_lat_w.w_buf),
                bg_entry(3, &self.p_w_out.w_buf),
                bg_entry(4, &buf_states),
                bg_entry(5, &buf_logits),
                bg_entry(6, &buf_windows),
                bg_entry(7, &buf_targets),
                bg_entry(8, &self.p_t_f.g_buf),
                bg_entry(9, &self.p_i_f.g_buf),
                bg_entry(10, &self.p_b_f.g_buf),
                bg_entry(11, &self.p_pos_bias.g_buf),
                bg_entry(12, &self.p_lat_w.g_buf),
                bg_entry(13, &self.p_w_out.g_buf),
                bg_entry(14, &self.p_b_out.g_buf),
            ],
        });

        // ===== Encoder 1: backward + (se clip ativo) zero_acc + sumsq por tensor =====
        let mut enc1 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("smg_v3_bwd_norm_enc"),
        });
        {
            let mut pass = enc1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("smg_v3_bwd_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_bwd);
            pass.set_bind_group(0, &bind_bwd, &[]);
            pass.dispatch_workgroups(batch_size as u32, 1, 1);
        }

        // Calcular norma via reduce-sumsq se clipping ativo.
        let mut norm_keepalive: Vec<(wgpu::Buffer, wgpu::BindGroup)> = Vec::with_capacity(8);
        let clip_active = self.max_grad_norm > 0.0;
        if clip_active {
            // zero_acc (1 thread).
            let pu_zero = ParamsNorm { length: 1, _pad0: 0, _pad1: 0, _pad2: 0 };
            let buf_pu_zero = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params_norm_zero"),
                contents: bytemuck::bytes_of(&pu_zero),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bg_zero = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_norm_zero"),
                layout: &self.bgl_norm,
                entries: &[
                    bg_entry(0, &buf_pu_zero),
                    bg_entry(1, &self.p_t_f.g_buf),  // qualquer storage ro válido
                    bg_entry(2, &self.buf_norm_acc),
                ],
            });
            {
                let mut pass = enc1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("zero_acc_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipe_zero_acc);
                pass.set_bind_group(0, &bg_zero, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            norm_keepalive.push((buf_pu_zero, bg_zero));

            for slot in self.iter_param_slots() {
                let pu = ParamsNorm { length: slot.n_elements, _pad0: 0, _pad1: 0, _pad2: 0 };
                let buf_pu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("params_norm"),
                    contents: bytemuck::bytes_of(&pu),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bg_sumsq"),
                    layout: &self.bgl_norm,
                    entries: &[
                        bg_entry(0, &buf_pu),
                        bg_entry(1, &slot.g_buf),
                        bg_entry(2, &self.buf_norm_acc),
                    ],
                });
                let mut pass = enc1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sumsq_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipe_sumsq);
                pass.set_bind_group(0, &bg, &[]);
                let groups = slot.n_elements.div_ceil(256);
                pass.dispatch_workgroups(groups, 1, 1);
                drop(pass);
                norm_keepalive.push((buf_pu, bg));
            }

            // Copy accumulator → staging para readback.
            enc1.copy_buffer_to_buffer(&self.buf_norm_acc, 0, &self.staging_norm, 0, 4);
        }
        queue.submit(Some(enc1.finish()));
        drop(norm_keepalive);

        // Readback da norma (somente se clip ativo). Fixa scale com clipping.
        let scale = if clip_active {
            let slice = self.staging_norm.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            let _ = device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let bits: &[u32] = bytemuck::cast_slice(&data);
            let norm_sq = f32::from_bits(bits[0]).max(0.0);
            drop(data);
            self.staging_norm.unmap();
            let norm = norm_sq.sqrt();
            let clip = if norm > self.max_grad_norm && norm > 0.0 {
                self.max_grad_norm / norm
            } else {
                1.0
            };
            self.cfg.learning_rate / batch_size as f32 * clip
        } else {
            self.cfg.learning_rate / batch_size as f32
        };
        let scale_bits = scale.to_bits();

        // ===== Encoder 2: SGD por tensor + zero_grads por tensor =====
        let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("smg_v3_sgd_enc"),
        });
        let mut sgd_keepalive: Vec<(wgpu::Buffer, wgpu::BindGroup)> = Vec::with_capacity(7);
        for slot in self.iter_param_slots() {
            let pu = ParamsSgd {
                length: slot.n_elements,
                scale_bits,
                _pad0: 0,
                _pad1: 0,
            };
            let buf_pu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params_sgd"),
                contents: bytemuck::bytes_of(&pu),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bg_sgd = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("smg_v3_sgd_bg"),
                layout: &self.bgl_sgd,
                entries: &[
                    bg_entry(0, &buf_pu),
                    bg_entry(1, &slot.w_buf),
                    bg_entry(2, &slot.g_buf),
                ],
            });
            {
                let mut pass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("smg_v3_sgd_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipe_sgd);
                pass.set_bind_group(0, &bg_sgd, &[]);
                let groups = slot.n_elements.div_ceil(256);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            {
                let mut pass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("smg_v3_zero_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipe_zero);
                pass.set_bind_group(0, &bg_sgd, &[]);
                let groups = slot.n_elements.div_ceil(256);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            sgd_keepalive.push((buf_pu, bg_sgd));
        }

        queue.submit(Some(enc2.finish()));
        let _ = device.poll(wgpu::Maintain::Wait);
        drop(sgd_keepalive);
        let _ = device.poll(wgpu::Maintain::Wait);

        sum_loss
    }

    /// Treina 1 época em mini-batches.
    pub fn train_epoch(&self, pairs: &[(Vec<u32>, u32)], batch_size: usize) -> f32 {
        if pairs.is_empty() {
            return 0.0;
        }
        let bs = batch_size.max(1).min(pairs.len());
        let mut total_loss = 0.0_f32;
        for batch_start in (0..pairs.len()).step_by(bs) {
            let batch_end = (batch_start + bs).min(pairs.len());
            let batch = &pairs[batch_start..batch_end];
            total_loss += self.train_step(batch);
        }
        total_loss / pairs.len() as f32
    }

    fn iter_param_slots(&self) -> [&ParamSlot; 7] {
        [
            &self.p_t_f,
            &self.p_i_f,
            &self.p_b_f,
            &self.p_pos_bias,
            &self.p_lat_w,
            &self.p_w_out,
            &self.p_b_out,
        ]
    }

    // ============================================================
    // ACCURACY / EVAL helpers
    // ============================================================

    /// Argmax de cada window.
    #[must_use]
    pub fn predict_batch(&self, windows: &[Vec<u32>]) -> Vec<u32> {
        let logits = self.forward_batch(windows);
        logits
            .iter()
            .map(|l| {
                let mut best = 0u32;
                let mut best_v = f32::NEG_INFINITY;
                for (i, &x) in l.iter().enumerate() {
                    if x > best_v {
                        best_v = x;
                        best = i as u32;
                    }
                }
                best
            })
            .collect()
    }

    /// Cross-entropy médio sobre samples (forward GPU + softmax host).
    #[must_use]
    pub fn eval_loss(&self, samples: &[(Vec<u32>, u32)]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let v = self.cfg.vocab_size;
        let windows: Vec<Vec<u32>> = samples.iter().map(|(w, _)| w.clone()).collect();
        let logits = self.forward_batch(&windows);
        let mut total = 0.0_f32;
        for (l, (_, target)) in logits.iter().zip(samples.iter()) {
            let mx = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = l.iter().map(|&x| (x - mx).exp()).collect();
            let s: f32 = exps.iter().sum();
            let t = (*target as usize).min(v - 1);
            total -= (exps[t] / s).max(1e-20).ln();
        }
        total / samples.len() as f32
    }

    /// Accuracy @ argmax.
    #[must_use]
    pub fn eval_accuracy(&self, samples: &[(Vec<u32>, u32)]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let v = self.cfg.vocab_size as u32;
        let windows: Vec<Vec<u32>> = samples.iter().map(|(w, _)| w.clone()).collect();
        let preds = self.predict_batch(&windows);
        let mut correct = 0;
        for (p, (_, t)) in preds.iter().zip(samples.iter()) {
            let tt = (*t).min(v - 1);
            if *p == tt {
                correct += 1;
            }
        }
        correct as f32 / samples.len() as f32
    }

    /// Baixa um peso específico para CPU (debug/snapshots).
    #[must_use]
    pub fn download_weight(&self, name: &str) -> Vec<f32> {
        let slot = match name {
            "t_f" => &self.p_t_f,
            "i_f" => &self.p_i_f,
            "b_f" => &self.p_b_f,
            "pos_bias" => &self.p_pos_bias,
            "lat_w" => &self.p_lat_w,
            "w_out" => &self.p_w_out,
            "b_out" => &self.p_b_out,
            _ => panic!("nome de peso desconhecido: {name}"),
        };
        let bytes = (slot.n_elements as u64) * 4;
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_w"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dl_w"),
        });
        enc.copy_buffer_to_buffer(&slot.w_buf, 0, &staging, 0, bytes);
        queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        let _ = slot.label;
        v
    }

    /// Baixa o gradiente acumulado de um peso (debug — esperado zero entre steps).
    #[must_use]
    pub fn download_grad(&self, name: &str) -> Vec<f32> {
        let slot = match name {
            "t_f" => &self.p_t_f,
            "i_f" => &self.p_i_f,
            "b_f" => &self.p_b_f,
            "pos_bias" => &self.p_pos_bias,
            "lat_w" => &self.p_lat_w,
            "w_out" => &self.p_w_out,
            "b_out" => &self.p_b_out,
            _ => panic!("nome de peso desconhecido: {name}"),
        };
        let bytes = (slot.n_elements as u64) * 4;
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_g"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dl_g"),
        });
        enc.copy_buffer_to_buffer(&slot.g_buf, 0, &staging, 0, bytes);
        queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        // Bits são interpretados como f32 (atomic<u32> → bitcast).
        let bits: &[u32] = bytemuck::cast_slice(&data);
        let v: Vec<f32> = bits.iter().map(|b| f32::from_bits(*b)).collect();
        drop(data);
        staging.unmap();
        v
    }
}

// ---------- helpers de bind layout ----------
fn storage_ro() -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}
fn storage_rw() -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}
fn uniform() -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}
fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}
fn bg_entry<'a>(binding: u32, buf: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::microcolumn_v3::{GridConfig, SmgGrid};

    fn cfg() -> GridConfig {
        GridConfig {
            vocab_size: 16,
            window: 6,
            learning_rate: 0.05,
            seed: 42,
        }
    }

    fn try_ctx() -> Option<Arc<GpuContext>> {
        match GpuContext::new() {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                eprintln!("[skip] sem GPU: {e}");
                None
            }
        }
    }

    #[test]
    fn gpu_forward_matches_cpu() {
        let Some(ctx) = try_ctx() else { return };
        let cpu = SmgGrid::new(cfg());
        let gpu = SmgGridGpu::from_cpu(&cpu, ctx);
        let window: Vec<u32> = vec![0, 3, 1, 7, 2, 5];
        let cpu_logits = cpu.forward(&window).logits;
        let gpu_logits = gpu.forward(&window);
        let max_diff = max_abs_diff(&cpu_logits, &gpu_logits);
        eprintln!("forward max abs diff = {max_diff:.6e}");
        assert!(max_diff < 1e-3, "forward diverge: {max_diff:.6e}");
    }

    #[test]
    fn gpu_train_step_reduces_loss_on_overfit() {
        let Some(ctx) = try_ctx() else { return };
        let cpu = SmgGrid::new(cfg());
        let gpu = SmgGridGpu::from_cpu(&cpu, ctx);
        let pair = (vec![0_u32, 1, 2, 3, 4, 5], 7_u32);
        let initial = gpu.eval_loss(&[pair.clone()]);
        for _ in 0..50 {
            gpu.train_step(&[pair.clone()]);
        }
        let final_loss = gpu.eval_loss(&[pair.clone()]);
        eprintln!("overfit: initial={initial:.4} final={final_loss:.4}");
        assert!(
            final_loss < initial * 0.5,
            "GPU train não reduziu loss: initial={initial:.4} final={final_loss:.4}"
        );
    }

    /// Treino em batch multi-sample reduz loss (testa atomics sob contenção).
    #[test]
    fn gpu_train_step_batch_reduces_loss() {
        let Some(ctx) = try_ctx() else { return };
        let cpu = SmgGrid::new(cfg());
        let gpu = SmgGridGpu::from_cpu(&cpu, ctx);
        // 8 pares determinísticos de mesma janela com targets distintos.
        let batch: Vec<(Vec<u32>, u32)> = (0..8)
            .map(|i| (vec![0_u32, 1, 2, 3, 4, 5], (i % 8) as u32))
            .collect();
        let initial = gpu.eval_loss(&batch);
        for _ in 0..30 {
            gpu.train_step(&batch);
        }
        let final_loss = gpu.eval_loss(&batch);
        eprintln!("batch overfit: initial={initial:.4} final={final_loss:.4}");
        assert!(
            final_loss < initial * 0.95,
            "GPU train (batch) não reduziu loss: initial={initial:.4} final={final_loss:.4}"
        );
    }

    /// Smoke test: forward não diverge depois de várias steps de treino
    /// (verifica que atomic grads não corromperam pesos com NaN/Inf).
    #[test]
    fn gpu_weights_stay_finite_after_training() {
        let Some(ctx) = try_ctx() else { return };
        let cpu = SmgGrid::new(cfg());
        let gpu = SmgGridGpu::from_cpu(&cpu, ctx);
        let batch: Vec<(Vec<u32>, u32)> = (0..4)
            .map(|i| (vec![i as u32, (i + 1) % 8, 2, 3, 4, 5], (i * 3 % 8) as u32))
            .collect();
        for _ in 0..20 {
            gpu.train_step(&batch);
        }
        for name in ["t_f", "i_f", "b_f", "pos_bias", "lat_w", "w_out", "b_out"] {
            let w = gpu.download_weight(name);
            for &x in &w {
                assert!(x.is_finite(), "{name} contém não-finito após treino: {x}");
            }
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }
}
