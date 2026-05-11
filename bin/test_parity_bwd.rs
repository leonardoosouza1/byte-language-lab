//! Parity test: backward pass HSA AQL vs CPU reference.
//!
//! Usa mesmo setup que test_parity_fwd (seed=42, batch=4, T=4, vocab=64).
//! Referência: SmgGrid::compute_grads_into (CPU).
//! GPU: dispatch_forward → (drain) → dispatch_backward → (drain) → compare.
//!
//! Tolerância: 2e-3 por tensor (CAS-loop re-ordena atomics; ffast-math).
//!
//! ```bash
//! ROCM_PATH=/opt/rocm-7.2.1 cargo run --features gpu-aql --bin test_parity_bwd
//! ```

#![cfg(feature = "gpu-aql")]
#![allow(unsafe_code)]

use blab::{
    gpu_aql::{
        dispatch::{ParamsFwd, ParamsBwd},
        SmgRuntime,
    },
    learn::microcolumn_v3::{GridConfig, SmgGrid, FLAT_DIM, K, N_FAMILIES, N_CELLS},
};
use hip_ffi::safe::DeviceBuffer;

// FSMS_PER_CELL = N_FAMILIES = 8 (não exportado como pub const no módulo)
const FSMS_PER_CELL: usize = 8;

fn check_tensor(name: &str, cpu: &[f32], gpu_raw: &[u32], tol: f32) -> bool {
    assert_eq!(
        cpu.len(), gpu_raw.len(),
        "tensor '{}' size mismatch: cpu={} gpu={}",
        name, cpu.len(), gpu_raw.len()
    );
    let mut max_err = 0.0_f32;
    let mut n_fail  = 0usize;
    for (i, (&c, &g)) in cpu.iter().zip(gpu_raw.iter()).enumerate() {
        let gf = f32::from_bits(g);
        let err = (c - gf).abs();
        if err > max_err { max_err = err; }
        if err > tol { n_fail += 1; }
        let _ = i; // usado abaixo
    }
    let ok = n_fail == 0;
    if ok {
        println!("  {:12} max|err|={:.2e}  ✅", name, max_err);
    } else {
        println!("  {:12} max|err|={:.2e}  ❌  n_fail={}/{}", name, max_err, n_fail, cpu.len());
        // Mostra primeiros 4 discrepantes para diagnóstico.
        let mut shown = 0;
        for (i, (&c, &g)) in cpu.iter().zip(gpu_raw.iter()).enumerate() {
            let gf = f32::from_bits(g);
            let err = (c - gf).abs();
            if err > tol && shown < 4 {
                println!("    idx={} cpu={:.7} gpu={:.7} err={:.2e}", i, c, gf, err);
                shown += 1;
            }
        }
    }
    ok
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const BATCH: usize = 4;
    const T:     usize = 4;
    const VOCAB: usize = 64;

    // Tamanhos dos tensores de gradiente.
    let n_t_f   = N_FAMILIES * K * K;
    let n_i_f   = N_FAMILIES * VOCAB * K;
    let n_b_f   = N_FAMILIES * K;
    let n_pos   = N_CELLS * K;
    let n_lat_w = N_FAMILIES * K * K;
    let n_w_out = FLAT_DIM * VOCAB;
    let n_b_out = VOCAB;
    // Sanity: N_CELLS * FSMS_PER_CELL * K deve ser FLAT_DIM.
    debug_assert_eq!(N_CELLS * FSMS_PER_CELL * K, FLAT_DIM);

    println!("=== test_parity_bwd ===");
    println!("batch={BATCH}, t_steps={T}, vocab={VOCAB}");
    println!("n_params_grad: t_f={n_t_f} i_f={n_i_f} b_f={n_b_f} pos={n_pos} lat_w={n_lat_w} w_out={n_w_out} b_out={n_b_out}");

    // ---- 1. SmgGrid CPU com seed fixo ----
    let cfg = GridConfig {
        vocab_size:    VOCAB,
        window:        T,
        learning_rate: 1e-3,
        seed:          42,
    };
    let cpu = SmgGrid::new(cfg.clone());

    // ---- 2. Inputs determinísticos (mesmos do forward test) ----
    let mut batch_cpu: Vec<(Vec<u32>, u32)> = Vec::with_capacity(BATCH);
    let mut win_flat:  Vec<u32>             = Vec::with_capacity(BATCH * T);
    let mut targets:   Vec<u32>             = Vec::with_capacity(BATCH);
    for b in 0..BATCH {
        let win: Vec<u32> = (0..T).map(|t| ((b * 7 + t * 3 + 1) % VOCAB) as u32).collect();
        let tgt = ((b + 1) * 11 % VOCAB) as u32;
        win_flat.extend_from_slice(&win);
        targets.push(tgt);
        batch_cpu.push((win, tgt));
    }

    // ---- 3. Referência CPU ----
    println!("\n[CPU] compute_grads_into...");
    let mut cpu_grads = cpu.alloc_grads();
    cpu.compute_grads_into(&batch_cpu, &mut cpu_grads);
    println!("  sum_loss = {:.4}", cpu_grads.sum_loss);

    // ---- 4. HSA runtime ----
    println!("\n[HSA] SmgRuntime init...");
    let rt = SmgRuntime::new()?;

    // Upload pesos.
    let buf_t_f   = DeviceBuffer::<f32>::from_slice(cpu.weights_t_f())?;
    let buf_i_f   = DeviceBuffer::<f32>::from_slice(cpu.weights_i_f())?;
    let buf_b_f   = DeviceBuffer::<f32>::from_slice(cpu.weights_b_f())?;
    let buf_pos   = DeviceBuffer::<f32>::from_slice(cpu.weights_pos_bias())?;
    let buf_lat   = DeviceBuffer::<f32>::from_slice(cpu.weights_lat_w())?;
    let buf_w_out = DeviceBuffer::<f32>::from_slice(cpu.weights_w_out())?;
    let buf_b_out = DeviceBuffer::<f32>::from_slice(cpu.weights_b_out())?;

    // Inputs/targets.
    let buf_windows = DeviceBuffer::<u32>::from_slice(&win_flat)?;
    let buf_targets = DeviceBuffer::<u32>::from_slice(&targets)?;

    // Buffers de saída do forward.
    let mut buf_logits = DeviceBuffer::<f32>::alloc(BATCH * VOCAB)?;
    let mut buf_states = DeviceBuffer::<f32>::alloc(BATCH * (T + 1) * FLAT_DIM)?;

    // ---- 5. Forward dispatch (para obter states + logits em GPU) ----
    println!("[HSA] dispatch_forward...");
    let params_fwd = ParamsFwd {
        vocab_size:   VOCAB as u32,
        t_steps:      T as u32,
        batch_size:   BATCH as u32,
        cache_states: 1,
    };
    unsafe {
        rt.dispatch_forward(
            params_fwd,
            buf_t_f.as_ptr()     as u64,
            buf_i_f.as_ptr()     as u64,
            buf_b_f.as_ptr()     as u64,
            buf_pos.as_ptr()     as u64,
            buf_lat.as_ptr()     as u64,
            buf_w_out.as_ptr()   as u64,
            buf_b_out.as_ptr()   as u64,
            buf_windows.as_ptr() as u64,
            buf_logits.as_mut_ptr() as u64,
            buf_states.as_mut_ptr() as u64,
        )?;
        rt.drain()?;
    }
    println!("  forward OK");

    // ---- 6. Grad buffers zerados (u32 = 0 → f32 = 0.0) ----
    // O kernel usa atomic_add_f32(uint32_t*) via CAS-loop; aloca como u32.
    let mut buf_g_t_f   = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_t_f])?;
    let mut buf_g_i_f   = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_i_f])?;
    let mut buf_g_b_f   = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_b_f])?;
    let mut buf_g_pos   = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_pos])?;
    let mut buf_g_lat_w = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_lat_w])?;
    let mut buf_g_w_out = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_w_out])?;
    let mut buf_g_b_out = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_b_out])?;

    // ---- 7. Backward dispatch ----
    println!("[HSA] dispatch_backward...");
    let params_bwd = ParamsBwd {
        vocab_size: VOCAB as u32,
        t_steps:    T as u32,
        batch_size: BATCH as u32,
        _pad:       0,
    };
    unsafe {
        rt.dispatch_backward(
            params_bwd,
            buf_t_f.as_ptr()    as u64,   // read: T_f
            buf_lat.as_ptr()    as u64,   // read: lat_w
            buf_w_out.as_ptr()  as u64,   // read: W_out
            buf_states.as_ptr() as u64,   // read: states_all [B, T+1, FLAT_DIM]
            buf_logits.as_ptr() as u64,   // read: logits_in  [B, V]
            buf_windows.as_ptr() as u64,  // read: windows    [B, T]
            buf_targets.as_ptr() as u64,  // read: targets    [B]
            buf_g_t_f.as_mut_ptr()   as u64,  // write: g_t_f
            buf_g_i_f.as_mut_ptr()   as u64,  // write: g_i_f
            buf_g_b_f.as_mut_ptr()   as u64,  // write: g_b_f
            buf_g_pos.as_mut_ptr()   as u64,  // write: g_pos
            buf_g_lat_w.as_mut_ptr() as u64,  // write: g_lat_w
            buf_g_w_out.as_mut_ptr() as u64,  // write: g_w_out
            buf_g_b_out.as_mut_ptr() as u64,  // write: g_b_out
        )?;
        rt.drain()?;
    }
    println!("  backward OK");

    // ---- 8. Download e comparação ----
    println!("\n[COMPARE] Tolerância 2e-3 (CAS-loop + ffast-math)...");
    let tol = 2e-3_f32;
    let g_t_f   = buf_g_t_f.to_host()?;
    let g_i_f   = buf_g_i_f.to_host()?;
    let g_b_f   = buf_g_b_f.to_host()?;
    let g_pos   = buf_g_pos.to_host()?;
    let g_lat_w = buf_g_lat_w.to_host()?;
    let g_w_out = buf_g_w_out.to_host()?;
    let g_b_out = buf_g_b_out.to_host()?;

    let mut all_ok = true;
    all_ok &= check_tensor("g_t_f",   &cpu_grads.g_t_f,      &g_t_f,   tol);
    all_ok &= check_tensor("g_i_f",   &cpu_grads.g_i_f,      &g_i_f,   tol);
    all_ok &= check_tensor("g_b_f",   &cpu_grads.g_b_f,      &g_b_f,   tol);
    all_ok &= check_tensor("g_pos",   &cpu_grads.g_pos_bias, &g_pos,   tol);
    all_ok &= check_tensor("g_lat_w", &cpu_grads.g_lat_w,    &g_lat_w, tol);
    all_ok &= check_tensor("g_w_out", &cpu_grads.g_w_out,    &g_w_out, tol);
    all_ok &= check_tensor("g_b_out", &cpu_grads.g_b_out,    &g_b_out, tol);

    if !all_ok {
        panic!("PARITY FAIL: um ou mais tensores de gradiente fora de tolerância 2e-3");
    }

    println!("\n✅ BACKWARD PARITY OK");
    Ok(())
}
