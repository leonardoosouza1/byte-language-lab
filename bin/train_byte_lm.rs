//! train_byte_lm — train the windowed byte LM on REAL local code (wgpu GEMM),
//! report the loss curve, and generate a sample. The first model trained
//! end-to-end on our own kernels against real data.
//!
//! Usage:  cargo run --release --bin train_byte_lm -- [corpus] [subset_bytes]

use blab::gpu::GpuContext;
use blab::gpu::byte_win_lm::ByteWinLm;
use std::time::Instant;

const WIN: usize = 12;
const D: usize = 32;
const H: usize = 256;
const BATCH: usize = 512;
const STEPS_PER_EPOCH: usize = 200;
const EPOCHS: usize = 30;

fn main() {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "data/corpus_local.txt".into());
    let cap: usize = std::env::args().nth(2)
        .and_then(|s| s.parse().ok()).unwrap_or(2_000_000);

    let ctx = match GpuContext::new() {
        Ok(c) => c, Err(e) => { eprintln!("no GPU: {e:?}"); return; }
    };
    println!("GPU: {}", ctx.device_name());

    let data: Vec<u8> = std::fs::read(&path).expect("corpus").into_iter().take(cap).collect();
    println!("corpus: {} bytes from {}", data.len(), path);

    // (window, next-byte) pairs
    let mut wins: Vec<&[u8]> = Vec::new();
    let mut tgts: Vec<u8> = Vec::new();
    for i in WIN..data.len() { wins.push(&data[i - WIN..i]); tgts.push(data[i]); }
    let split = wins.len() * 9 / 10;
    println!("pairs: {} train / {} val   WIN={WIN} D={D} H={H}", split, wins.len() - split);

    let mut m = ByteWinLm::init(256, D, WIN, H, 2024);
    let n_params = 256 * D + WIN * D * H + H * 256;
    println!("params: {} ({:.2}M)\n", n_params, n_params as f64 / 1e6);
    println!("{:>5}  {:>10}  {:>10}  {:>8}", "epoch", "train", "val", "ms");

    let mut rng = 99u64;
    for ep in 0..EPOCHS {
        let t0 = Instant::now();
        let lr = 0.5 * (1.0 - ep as f32 / EPOCHS as f32).max(0.1);
        let mut tl = 0.0;
        for _ in 0..STEPS_PER_EPOCH {
            // random batch from train split
            let mut bw: Vec<&[u8]> = Vec::with_capacity(BATCH);
            let mut bt: Vec<u8> = Vec::with_capacity(BATCH);
            for _ in 0..BATCH {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let idx = (rng >> 33) as usize % split;
                bw.push(wins[idx]); bt.push(tgts[idx]);
            }
            tl += m.train_step(&ctx, &bw, &bt, lr);
        }
        let train = tl / STEPS_PER_EPOCH as f32;
        // val on a fixed slice
        let vw: Vec<&[u8]> = wins[split..(split + BATCH).min(wins.len())].to_vec();
        let vt: Vec<u8> = tgts[split..(split + BATCH).min(tgts.len())].to_vec();
        let val = m.eval_loss(&ctx, &vw, &vt);
        println!("{:>5}  {:>10.4}  {:>10.4}  {:>8}", ep, train, val, t0.elapsed().as_millis());
    }

    println!("\n=== generation (temp 0.6, 200 bytes) ===");
    for seed in &["function ", "const x =", "import "] {
        let g = m.generate(&ctx, seed.as_bytes(), 120, 0.6, &mut rng);
        println!("seed {:?}\n  {}\n", seed,
            String::from_utf8_lossy(&g).replace('\n', "\\n"));
    }
}
