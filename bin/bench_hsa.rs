//! Benchmark HSA AQL vs wgpu/WGSL para o loop de treino SMG v3.
//!
//! Mede throughput de `train_epoch` com dados sintéticos comparáveis ao exp18.
//! Referência: exp18_smg_lateral (~570ms/época, wgpu baseline).
//! Target: ≤ 10ms/época com HSA AQL (batch async dispatch + barrier_and).
//!
//! ```bash
//! # Benchmark HSA (exige ROCm 7.x)
//! ROCM_PATH=/opt/rocm-7.2.1 cargo run --release --features gpu-aql --bin bench_hsa
//!
//! # Referência wgpu (sem feature gpu-aql)
//! cargo run --release --bin bench_wgpu_ref
//! ```

#![cfg(feature = "gpu-aql")]
#![allow(unsafe_code)]

use std::time::Instant;
use std::sync::Arc;

use blab::{
    gpu::GpuContext,
    gpu_aql::SmgGridHsa,
    learn::microcolumn_gpu::SmgGridGpu,
    learn::microcolumn_v3::{GridConfig, SmgGrid},
};

// Parâmetros iguais ao exp18_smg_lateral para comparação justa.
const SEED:       u64   = 9001;
const WINDOW:     usize = 48;
const VOCAB:      usize = 256;  // exp18 usa vocab dinâmico; usamos 256 para benchmark
const LR:         f32   = 0.1;
const BATCH_SIZE: usize = 64;
const N_EPOCHS:   usize = 10;
const N_SAMPLES:  usize = 640; // ~10 batches por época

fn make_dataset(n: usize, window: usize, vocab: usize) -> Vec<(Vec<u32>, u32)> {
    // Dataset sintético determinístico: i-th sample = bytes (i*7+t*3+1) % vocab.
    (0..n).map(|i| {
        let win: Vec<u32> = (0..window).map(|t| ((i * 7 + t * 3 + 1) % vocab) as u32).collect();
        let tgt = ((i + 1) * 11 % vocab) as u32;
        (win, tgt)
    }).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== bench_hsa: SMG v3 HSA AQL vs wgpu ===");
    println!("vocab={VOCAB}, window={WINDOW}, batch={BATCH_SIZE}, epochs={N_EPOCHS}, samples={N_SAMPLES}\n");

    let cfg = GridConfig {
        vocab_size:    VOCAB,
        window:        WINDOW,
        learning_rate: LR,
        seed:          SEED,
    };
    let cpu = SmgGrid::new(cfg.clone());

    let dataset = make_dataset(N_SAMPLES, WINDOW, VOCAB);
    println!("dataset: {} samples, {} batches/epoch\n", dataset.len(), dataset.len() / BATCH_SIZE);

    // ---- Warmup (para evitar JIT / ROCm init overhead na primeira medição) ----
    println!("[HSA] Warmup (1 epoch)...");
    let mut hsa = SmgGridHsa::from_cpu(&cpu, BATCH_SIZE)?;
    let _ = hsa.train_epoch(&dataset, BATCH_SIZE);
    println!("  warmup done, n_params = {}", hsa.n_params());

    // ---- Benchmark HSA ----
    println!("\n[HSA] Benchmark ({N_EPOCHS} epochs)...");
    let t0 = Instant::now();
    let mut hsa_losses = Vec::with_capacity(N_EPOCHS);
    for ep in 0..N_EPOCHS {
        let loss = hsa.train_epoch(&dataset, BATCH_SIZE);
        hsa_losses.push(loss);
        if ep == 0 || ep == N_EPOCHS - 1 {
            println!("  epoch {} loss={:.4}", ep, loss);
        }
    }
    let hsa_total = t0.elapsed();
    let hsa_per_ep = hsa_total / N_EPOCHS as u32;
    println!("  TOTAL: {:?}  |  per epoch: {:?}", hsa_total, hsa_per_ep);
    println!("  final loss: {:.4}", hsa_losses.last().copied().unwrap_or(0.0));

    // ---- Benchmark wgpu (referência) ----
    println!("\n[wgpu] Warmup (1 epoch)...");
    let gpu_ctx = Arc::new(GpuContext::new()?);
    let wgpu_grid = SmgGridGpu::from_cpu(&cpu, gpu_ctx.clone());
    let _ = wgpu_grid.train_epoch(&dataset, BATCH_SIZE);
    println!("  warmup done");

    println!("\n[wgpu] Benchmark ({N_EPOCHS} epochs)...");
    let t1 = Instant::now();
    let mut wgpu_losses = Vec::with_capacity(N_EPOCHS);
    for ep in 0..N_EPOCHS {
        let loss = wgpu_grid.train_epoch(&dataset, BATCH_SIZE);
        wgpu_losses.push(loss);
        if ep == 0 || ep == N_EPOCHS - 1 {
            println!("  epoch {} loss={:.4}", ep, loss);
        }
    }
    let wgpu_total = t1.elapsed();
    let wgpu_per_ep = wgpu_total / N_EPOCHS as u32;
    println!("  TOTAL: {:?}  |  per epoch: {:?}", wgpu_total, wgpu_per_ep);
    println!("  final loss: {:.4}", wgpu_losses.last().copied().unwrap_or(0.0));

    // ---- Resultado ----
    let speedup = wgpu_total.as_secs_f64() / hsa_total.as_secs_f64();
    println!("\n=== RESULTADO ===");
    println!("  HSA:  {:>10?} / epoch", hsa_per_ep);
    println!("  wgpu: {:>10?} / epoch", wgpu_per_ep);
    println!("  Speedup HSA/wgpu: {:.1}×", speedup);

    Ok(())
}
