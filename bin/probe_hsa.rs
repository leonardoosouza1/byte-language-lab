//! Smoke-test do backend HSA AQL.
//!
//! Valida que `hip-ffi` é construível como path-dep e que o runtime HSA
//! consegue inicializar + alocar kernarg ring + tearing down sem panic.
//!
//! Não dispatcha kernel ainda — isso é Fase 2 da migração.
//! Ver `docs/migration-hsa-aql.md`.
//!
//! Build/run:
//! ```bash
//! cargo run --features gpu-aql --bin probe_hsa
//! ```

#![cfg(feature = "gpu-aql")]

use blab::gpu_aql::{hw, AqlContext, SmgRuntime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== probe_hsa: HSA AQL backend smoke-test ===");
    println!("Hardware target: RDNA2 gfx1031 (RX 6750 XT)");
    println!("  CUs           : {}", hw::N_CUS);
    println!("  Wavefront     : wave{}", hw::WAVE_SIZE);
    println!("  LDS/wg max    : {} KiB", hw::LDS_BYTES_PER_WG / 1024);
    println!("  SMG LDS use   : {} KiB", hw::SMG_LDS_BYTES / 1024);

    println!("\nFase 1 — AqlContext init...");
    let ctx = AqlContext::new()?;
    println!("  agent: {}", ctx.agent_name);
    drop(ctx);
    println!("  OK\n");

    println!("Fase 2 — SmgRuntime (carrega 4 módulos .hsaco + registra kernels)...");
    let rt = SmgRuntime::new()?;
    println!("  forward_main  kernarg_size = {}",
             rt.disp.get("forward_main").map(|k| k.kernarg_size).unwrap_or(0));
    println!("  backward_main kernarg_size = {}",
             rt.disp.get("backward_main").map(|k| k.kernarg_size).unwrap_or(0));
    println!("  sumsq_main    kernarg_size = {}",
             rt.disp.get("sumsq_main").map(|k| k.kernarg_size).unwrap_or(0));
    println!("  zero_acc      kernarg_size = {}",
             rt.disp.get("zero_acc").map(|k| k.kernarg_size).unwrap_or(0));
    println!("  sgd_main      kernarg_size = {}",
             rt.disp.get("sgd_main").map(|k| k.kernarg_size).unwrap_or(0));
    drop(rt);
    println!("  OK\n");

    println!("Probes passed. Próximo passo: Fase 3 (parity tests vs wgpu).");
    Ok(())
}
