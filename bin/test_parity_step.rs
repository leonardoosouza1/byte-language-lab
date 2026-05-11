//! Parity test: train_step completo HSA vs CPU reference.
//!
//! Valida que após 1 train_step (forward+backward+SGD) os pesos da GPU
//! batem com os pesos computados manualmente no CPU.
//!
//! Fluxo do teste:
//!   1. Criar SmgGrid CPU com seed=42 (mesmo de test_parity_fwd/bwd).
//!   2. CPU reference:
//!      a. compute_grads_into(batch) → grads
//!      b. w_new[i] = w[i] - (lr / batch_size) * g[i]  (SGD manual)
//!   3. HSA: SmgGridHsa::from_cpu + train_step(batch) + weights_to_host
//!   4. Compare pesos finais tensor a tensor (tolerância 2e-3).
//!
//! Nota: ffast-math + CAS-loop atomics podem introduzir diferenças de
//! arredondamento na acumulação dos gradientes. A tolerância 2e-3 cobre isso.
//!
//! ```bash
//! ROCM_PATH=/opt/rocm-7.2.1 cargo run --features gpu-aql --bin test_parity_step
//! ```

#![cfg(feature = "gpu-aql")]
#![allow(unsafe_code)]

use blab::{
    gpu_aql::SmgGridHsa,
    learn::microcolumn_v3::{GridConfig, SmgGrid},
};

fn check_weights(name: &str, cpu: &[f32], gpu: &[f32], tol: f32) -> bool {
    assert_eq!(cpu.len(), gpu.len(), "tensor '{}' size mismatch", name);
    let mut max_err = 0.0_f32;
    let mut n_fail  = 0_usize;
    for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let err = (c - g).abs();
        if err > max_err { max_err = err; }
        if err > tol { n_fail += 1; }
        let _ = i;
    }
    let ok = n_fail == 0;
    if ok {
        println!("  {:12} max|err|={:.2e}  ✅", name, max_err);
    } else {
        println!("  {:12} max|err|={:.2e}  ❌  n_fail={}/{}", name, max_err, n_fail, cpu.len());
        let mut shown = 0;
        for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let err = (c - g).abs();
            if err > tol && shown < 4 {
                println!("    idx={} cpu={:.7} gpu={:.7} err={:.2e}", i, c, g, err);
                shown += 1;
            }
        }
    }
    ok
}

/// Aplica SGD manualmente: `w_new[i] = w[i] - scale * g[i]`.
fn apply_sgd(w: &[f32], g: &[f32], scale: f32) -> Vec<f32> {
    w.iter().zip(g.iter()).map(|(&wi, &gi)| wi - scale * gi).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const BATCH: usize = 4;
    const T:     usize = 4;
    const VOCAB: usize = 64;
    const LR:    f32   = 1e-3;

    println!("=== test_parity_step ===");
    println!("batch={BATCH}, t_steps={T}, vocab={VOCAB}, lr={LR}");

    // ---- 1. SmgGrid CPU com seed fixo ----
    let cfg = GridConfig {
        vocab_size:    VOCAB,
        window:        T,
        learning_rate: LR,
        seed:          42,
    };
    let cpu = SmgGrid::new(cfg.clone());

    // ---- 2. Inputs determinísticos ----
    let mut batch: Vec<(Vec<u32>, u32)> = Vec::with_capacity(BATCH);
    for b in 0..BATCH {
        let win: Vec<u32> = (0..T).map(|t| ((b * 7 + t * 3 + 1) % VOCAB) as u32).collect();
        let tgt = ((b + 1) * 11 % VOCAB) as u32;
        batch.push((win, tgt));
    }

    // ---- 3. CPU reference: grads + SGD manual ----
    println!("\n[CPU] compute_grads_into + SGD manual...");
    let mut grads = cpu.alloc_grads();
    cpu.compute_grads_into(&batch, &mut grads);
    println!("  sum_loss = {:.4}", grads.sum_loss);

    let scale = LR / BATCH as f32;
    let cpu_t_f      = apply_sgd(cpu.weights_t_f(),      &grads.g_t_f,      scale);
    let cpu_i_f      = apply_sgd(cpu.weights_i_f(),      &grads.g_i_f,      scale);
    let cpu_b_f      = apply_sgd(cpu.weights_b_f(),      &grads.g_b_f,      scale);
    let cpu_pos_bias = apply_sgd(cpu.weights_pos_bias(),  &grads.g_pos_bias, scale);
    let cpu_lat_w    = apply_sgd(cpu.weights_lat_w(),     &grads.g_lat_w,    scale);
    let cpu_w_out    = apply_sgd(cpu.weights_w_out(),     &grads.g_w_out,    scale);
    let cpu_b_out    = apply_sgd(cpu.weights_b_out(),     &grads.g_b_out,    scale);
    println!("  CPU weights after SGD computed.");

    // ---- 4. HSA: 1 train_step ----
    println!("\n[HSA] SmgGridHsa::from_cpu + train_step...");
    let mut hsa = SmgGridHsa::from_cpu(&cpu, BATCH)?;
    let hsa_loss = hsa.train_step(&batch);
    println!("  loss = {:.4}", hsa_loss);

    // ---- 5. Download pesos GPU ----
    let hsa_w = hsa.weights_to_host();
    println!("  weights downloaded.");

    // ---- 6. Comparação ----
    println!("\n[COMPARE] Tolerância 2e-3 (CAS-loop + ffast-math)...");
    let tol = 2e-3_f32;
    let mut all_ok = true;
    all_ok &= check_weights("t_f",      &cpu_t_f,      &hsa_w.t_f,      tol);
    all_ok &= check_weights("i_f",      &cpu_i_f,      &hsa_w.i_f,      tol);
    all_ok &= check_weights("b_f",      &cpu_b_f,      &hsa_w.b_f,      tol);
    all_ok &= check_weights("pos_bias", &cpu_pos_bias, &hsa_w.pos_bias,  tol);
    all_ok &= check_weights("lat_w",    &cpu_lat_w,    &hsa_w.lat_w,    tol);
    all_ok &= check_weights("w_out",    &cpu_w_out,    &hsa_w.w_out,    tol);
    all_ok &= check_weights("b_out",    &cpu_b_out,    &hsa_w.b_out,    tol);

    if !all_ok {
        panic!("STEP PARITY FAIL: um ou mais tensores de peso divergem após train_step");
    }

    println!("\n✅ STEP PARITY OK — pesos GPU == CPU após 1 train_step");
    Ok(())
}
