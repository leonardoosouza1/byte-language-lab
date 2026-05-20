//! train_cpu_k64 — minimal CPU trainer for SMG v3 to validate whether
//! scaling K (32→64) helps before committing to a HSACO kernel rewrite.
//!
//! Loads a corpus subset, builds (window, next-byte) pairs, runs the
//! parallel CPU training loop a few epochs, prints loss curve + a tiny
//! generated sample. No GPU.
//!
//! Usage:  cargo run --release --bin train_cpu_k64 -- <corpus_file>

use blab::learn::microcolumn_v3::{GridConfig, SmgGrid, K};
use std::time::Instant;

const WINDOW: usize = 48;
const VOCAB: usize = 256;
const SUBSET_BYTES: usize = 1_500_000;   // 1.5 MB for a fast CPU test
const EPOCHS: usize = 10;
const BATCH: usize = 64;

fn build_pairs(data: &[u8]) -> Vec<(Vec<u32>, u32)> {
    let mut pairs = Vec::with_capacity(data.len().saturating_sub(WINDOW));
    for i in 0..data.len().saturating_sub(WINDOW) {
        let win: Vec<u32> = data[i..i + WINDOW].iter().map(|&b| b as u32).collect();
        let tgt = data[i + WINDOW] as u32;
        pairs.push((win, tgt));
    }
    pairs
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = exp.iter().sum();
    exp.into_iter().map(|x| x / s).collect()
}

fn sample(probs: &[f32], temp: f32, rng_state: &mut u64) -> u8 {
    // scale by temperature then sample multinomial
    let scaled: Vec<f32> = if temp <= 0.0 {
        probs.to_vec()
    } else {
        probs.iter().map(|p| p.powf(1.0 / temp)).collect()
    };
    let s: f32 = scaled.iter().sum();
    *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let r = (((*rng_state >> 33) as u32) as f32 / u32::MAX as f32) * s;
    let mut acc = 0.0;
    for (i, &p) in scaled.iter().enumerate() {
        acc += p;
        if r <= acc { return i as u8; }
    }
    255
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: train_cpu_k64 <corpus>")?;
    let data = std::fs::read(&path)?;
    let data: Vec<u8> = data.into_iter().take(SUBSET_BYTES).collect();
    println!("\nCorpus subset: {} bytes ({:.2} MB) from {}", data.len(),
             data.len() as f64 / 1_048_576.0, path);

    let pairs = build_pairs(&data);
    println!("Pares: {}   WINDOW={}   K={}", pairs.len(), WINDOW, K);

    let cfg = GridConfig {
        vocab_size: VOCAB,
        window: WINDOW,
        learning_rate: 0.003,
        seed: 9001,
    };
    let mut grid = SmgGrid::new(cfg);

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get()).unwrap_or(8);
    println!("Threads: {}   Batch: {}   Epochs: {}\n", n_threads, BATCH, EPOCHS);
    println!("época   loss      ms");

    for ep in 1..=EPOCHS {
        let t0 = Instant::now();
        let loss = grid.train_epoch_parallel(&pairs, n_threads, BATCH);
        let dt = t0.elapsed().as_millis();
        println!("  {:>3}   {:.4}   {:>6}", ep, loss / pairs.len() as f32, dt);
    }

    // small generation sample
    println!("\n=== Geração (temp=0.8, 96 bytes) ===");
    let seed_idx = 1000usize.min(data.len() - WINDOW - 1);
    let seed: Vec<u32> = data[seed_idx..seed_idx + WINDOW].iter()
        .map(|&b| b as u32).collect();
    print!("seed:   {:?}\ngerado: ",
        String::from_utf8_lossy(&data[seed_idx..seed_idx + WINDOW]));
    let mut win = seed.clone();
    let mut rng: u64 = 42;
    let mut out = String::new();
    for _ in 0..96 {
        let cache = grid.forward(&win);
        let probs = softmax(&cache.logits);
        let b = sample(&probs, 0.8, &mut rng);
        out.push(b as char);
        win.remove(0);
        win.push(b as u32);
    }
    println!("{}", out.replace('\n', "\\n").replace('\t', "\\t"));
    Ok(())
}
