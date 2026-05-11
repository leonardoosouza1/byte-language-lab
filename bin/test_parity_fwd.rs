//! Parity test: forward pass HSA AQL vs wgpu/WGSL.
//!
//! Carrega um SmgGrid fixo (seed=42), roda forward em batch de 4 janelas com
//! t_steps=4. Compara logits do path HSA AQL (via SmgRuntime) com os do path
//! wgpu (via SmgGridGpu). Tolerância: max |a-b| ≤ 1e-3 (ffast-math pode dar
//! erros de arredondamento vs wgpu que não usa ffast-math).
//!
//! ```bash
//! ROCM_PATH=/opt/rocm-7.2.1 cargo run --features gpu-aql --bin test_parity_fwd
//! ```

#![cfg(feature = "gpu-aql")]
#![allow(unsafe_code)]

use std::sync::Arc;

use blab::{
    gpu::GpuContext,
    gpu_aql::{dispatch::ParamsFwd, SmgRuntime},
    learn::microcolumn_v3::{GridConfig, SmgGrid, FLAT_DIM, K, N_CELLS, N_FAMILIES},
    learn::microcolumn_gpu::SmgGridGpu,
};

use hip_ffi::safe::DeviceBuffer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const BATCH: usize = 4;
    const T:     usize = 4;   // window length
    const VOCAB: usize = 64;  // small vocab for speed

    println!("=== test_parity_fwd ===");
    println!("batch={BATCH}, t_steps={T}, vocab={VOCAB}");

    // ---- 1. Criar SmgGrid CPU com seed fixo ----
    let cfg = GridConfig {
        vocab_size:    VOCAB,
        window:        T,
        learning_rate: 1e-3,
        seed:          42,
    };
    let cpu = SmgGrid::new(cfg.clone());

    // ---- 2. Entrada determinística ----
    let mut windows: Vec<Vec<u32>> = Vec::new();
    for b in 0..BATCH {
        let win: Vec<u32> = (0..T).map(|t| ((b * 7 + t * 3 + 1) % VOCAB) as u32).collect();
        windows.push(win);
    }
    let win_flat: Vec<u32> = windows.iter().flat_map(|w| w.iter().copied()).collect();

    // ---- 3. Referência wgpu ----
    println!("\n[wgpu] Forward...");
    let gpu_ctx = Arc::new(GpuContext::new()?);
    let smg_gpu = SmgGridGpu::from_cpu(&cpu, gpu_ctx.clone());
    let ref_logits_nested = smg_gpu.forward_batch(&windows);
    let ref_logits: Vec<f32> = ref_logits_nested.iter().flat_map(|v| v.iter().copied()).collect();
    println!("  logits[0..4] = {:?}", &ref_logits[..4.min(ref_logits.len())]);

    // ---- 4. HSA AQL forward ----
    println!("\n[HSA] SmgRuntime init...");
    let rt = SmgRuntime::new()?;

    // Aloca buffers HIP e faz upload dos pesos.
    let v = VOCAB;
    let buf_t_f     = DeviceBuffer::<f32>::from_slice(cpu.weights_t_f())?;
    let buf_i_f     = DeviceBuffer::<f32>::from_slice(cpu.weights_i_f())?;
    let buf_b_f     = DeviceBuffer::<f32>::from_slice(cpu.weights_b_f())?;
    let buf_pos     = DeviceBuffer::<f32>::from_slice(cpu.weights_pos_bias())?;
    let buf_lat     = DeviceBuffer::<f32>::from_slice(cpu.weights_lat_w())?;
    let buf_w_out   = DeviceBuffer::<f32>::from_slice(cpu.weights_w_out())?;
    let buf_b_out   = DeviceBuffer::<f32>::from_slice(cpu.weights_b_out())?;
    let buf_windows = DeviceBuffer::<u32>::from_slice(&win_flat)?;

    // Buffers de output.
    let mut buf_logits = DeviceBuffer::<f32>::alloc(BATCH * v)?;
    // states_out: BATCH × (T+1) × FLAT_DIM  (cache_states=1)
    let mut buf_states = DeviceBuffer::<f32>::alloc(BATCH * (T + 1) * FLAT_DIM)?;

    println!("[HSA] Dispatching forward_main...");
    let params = ParamsFwd {
        vocab_size:   v as u32,
        t_steps:      T as u32,
        batch_size:   BATCH as u32,
        cache_states: 1,
    };
    unsafe {
        rt.dispatch_forward(
            params,
            buf_t_f.as_ptr()    as u64,
            buf_i_f.as_ptr()    as u64,
            buf_b_f.as_ptr()    as u64,
            buf_pos.as_ptr()    as u64,
            buf_lat.as_ptr()    as u64,
            buf_w_out.as_ptr()  as u64,
            buf_b_out.as_ptr()  as u64,
            buf_windows.as_ptr() as u64,
            buf_logits.as_mut_ptr() as u64,
            buf_states.as_mut_ptr() as u64,
        )?;
        rt.drain()?;
    }

    let hsa_logits = buf_logits.to_host()?;
    println!("  logits[0..4] = {:?}", &hsa_logits[..4.min(hsa_logits.len())]);

    // ---- 5. Comparação ----
    println!("\n[COMPARE] Tolerância 1e-3 (ffast-math rounding)...");
    let len = ref_logits.len().min(hsa_logits.len());
    let mut max_err = 0.0f32;
    let mut n_fail  = 0usize;
    for i in 0..len {
        let err = (ref_logits[i] - hsa_logits[i]).abs();
        if err > max_err { max_err = err; }
        if err > 1e-3 { n_fail += 1; }
    }
    println!("  max |err| = {:.6}", max_err);
    println!("  n > 1e-3  = {}/{}", n_fail, len);

    if n_fail > 0 {
        // Mostra primeiros 8 discrepantes para diagnóstico.
        let mut shown = 0;
        for i in 0..len {
            let err = (ref_logits[i] - hsa_logits[i]).abs();
            if err > 1e-3 && shown < 8 {
                println!("  idx={} wgpu={:.6} hsa={:.6} err={:.6}",
                         i, ref_logits[i], hsa_logits[i], err);
                shown += 1;
            }
        }
        panic!("PARITY FAIL: {}/{} elementos acima de tolerância 1e-3", n_fail, len);
    }

    println!("\n✅ PARITY OK: max_err={:.6}", max_err);
    Ok(())
}
