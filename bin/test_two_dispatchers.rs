//! HMQW Phase 1 smoke test — can two HsaDispatcher instances run forward
//! passes concurrently on the same GPU?
//!
//! Each thread owns one SmgGridHsa (own HsaContext → own HSA queue → own
//! kernarg ring → own device buffers). Threads run N forward batches in
//! parallel. We compare wall-clock to a single-thread baseline running 2N
//! batches sequentially. Parallel < ~1.8× single-thread → real concurrency.
//!
//! Run:  cargo run --release --features gpu-aql --bin test_two_dispatchers

use blab::learn::microcolumn_v3::{GridConfig, SmgGrid};
use blab::gpu_aql::smg_grid_hsa::SmgGridHsa;
use std::time::Instant;

const WINDOW: usize = 96;
const BATCH:  usize = 256;
const ITERS:  usize = 50;

fn make_cpu() -> SmgGrid {
    SmgGrid::new(GridConfig {
        vocab_size: 256,
        window: WINDOW,
        learning_rate: 0.003,
        seed: 7,
    })
}

fn fake_windows(seed: u64) -> Vec<Vec<u32>> {
    let mut s = seed;
    (0..BATCH).map(|_| {
        (0..WINDOW).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as u32) & 0xff
        }).collect()
    }).collect()
}

fn run_block(hsa: &SmgGridHsa, ws: &[Vec<u32>], iters: usize) -> usize {
    let mut total = 0usize;
    for _ in 0..iters {
        let logits = hsa.forward_batch(ws);
        total += logits.len();
    }
    total
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("HMQW Phase 1 smoke — 2 dispatchers concurrent vs sequential");
    println!("  window={WINDOW}  batch={BATCH}  iters/thread={ITERS}");

    let cpu = make_cpu();
    let ws_a = fake_windows(11);
    let ws_b = fake_windows(22);

    // ---- baseline: 1 dispatcher doing 2× the work, sequential ----
    let hsa_solo = SmgGridHsa::from_cpu(&cpu, BATCH)?;
    let t0 = Instant::now();
    let n1 = run_block(&hsa_solo, &ws_a, ITERS);
    let n2 = run_block(&hsa_solo, &ws_b, ITERS);
    let dt_solo = t0.elapsed();
    println!("\nSEQUENTIAL  1 dispatcher × {} iter × 2 batches", ITERS);
    println!("  {} forward calls, total {} logits, {:.2}s",
             ITERS * 2, n1 + n2, dt_solo.as_secs_f64());
    drop(hsa_solo);

    // ---- parallel: 2 dispatchers, each doing 1× the work, concurrent ----
    let hsa_x = SmgGridHsa::from_cpu(&cpu, BATCH)?;
    let hsa_y = SmgGridHsa::from_cpu(&cpu, BATCH)?;
    println!("\nPARALLEL    2 dispatchers × {} iter, concurrent threads", ITERS);
    let t0 = Instant::now();
    std::thread::scope(|s| {
        let ha = s.spawn(|| run_block(&hsa_x, &ws_a, ITERS));
        let hb = s.spawn(|| run_block(&hsa_y, &ws_b, ITERS));
        let _ = ha.join();
        let _ = hb.join();
    });
    let dt_par = t0.elapsed();
    println!("  {:.2}s", dt_par.as_secs_f64());

    // ---- verdict ----
    let speedup = dt_solo.as_secs_f64() / dt_par.as_secs_f64();
    println!("\nspeedup parallel/sequential = {:.2}×", speedup);
    if speedup >= 1.6 {
        println!("✓ MULTI-STREAM REAL — 2 queues run concurrently on the GPU");
    } else if speedup >= 1.1 {
        println!("~ partial overlap — driver serializing some, dispatch ok");
    } else {
        println!("✗ NO PARALLELISM — GPU serialized dispatches");
    }
    println!("\nMonitor rocm-smi --showpower --showuse during run to confirm.");
    Ok(())
}
