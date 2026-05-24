//! GPU trainer for the generative liquid / selective-SSM byte LM.
//!
//! Trains `blab::gpu::ssm::DevSsm` (forward+backward on GPU, AdamW), reports
//! held-out bits/byte, and GENERATES. The liquid recurrence gives O(L) inference
//! (no attention, no KV-cache) and is parameter-efficient. A/B target: the
//! transformer baseline on the same corpus (~3.71 bits/byte on corpus_diverse).
//!
//! Usage: train_ssm_gpu [corpus] [budget_secs] [d_model] [n_layers] [seq] [lr]

use blab::gpu::ssm::DevSsm;
use blab::gpu::GpuContext;
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self, bound: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % bound.max(1)
    }
}
const LN2: f64 = std::f64::consts::LN_2;

fn heldout_bpb(ctx: &GpuContext, dev: &DevSsm, held: &[u8], seq: usize,
               rng: &mut Lcg, nwin: usize) -> f32 {
    if held.len() <= seq + 1 { return f32::NAN; }
    let v = 256usize;
    let mut total = 0.0f64; let mut cnt = 0usize;
    for _ in 0..nwin {
        let s = rng.next(held.len() - seq - 1);
        let toks = &held[s..s + seq];
        let tgts = &held[s + 1..s + seq + 1];
        let logits = dev.forward_dev(ctx, toks, 0);
        for i in 0..seq {
            let row = &logits[i * v..i * v + v];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sm: f32 = row.iter().map(|z| (z - mx).exp()).sum();
            let lp = (row[tgts[i] as usize] - mx) - sm.ln();
            total += -(lp as f64) / LN2;
            cnt += 1;
        }
    }
    (total / cnt as f64) as f32
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).cloned().unwrap_or_else(|| "data/corpus_diverse.txt".into());
    let budget: u64 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);
    let d: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(128);
    let n_layers: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);
    let seq: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(64);
    let lr: f32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(3e-3);

    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    println!("corpus {path}: {} bytes", bytes.len());
    let held_len = (bytes.len() / 20).min(32768);
    let split = bytes.len() - held_len;
    let (train, held) = bytes.split_at(split);
    let mut cnt = [0u64; 256];
    for &b in held { cnt[b as usize] += 1; }
    let n = held.len() as f64;
    let floor: f64 = (0..256).filter(|&i| cnt[i] > 0)
        .map(|i| { let p = cnt[i] as f64 / n; -p * p.log2() }).sum();
    println!("train {} | held {} bytes | unigram floor {floor:.3} bits/byte", train.len(), held.len());

    let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU: {e}"); return; } };
    println!("GPU: {}", ctx.device_name());
    let n_params = 256 * d + n_layers * 3 * d * d + d;
    println!("LIQUID/SSM (GPU): d={d} layers={n_layers} seq={seq} lr={lr} (~{:.2}M params)",
             n_params as f64 / 1e6);

    let mut dev = DevSsm::new(&ctx, 256, d, n_layers, 1234);
    let mut rng = Lcg(0xDEAD_BEEF_99);
    let t0 = Instant::now();
    let warmup = 200u64;
    let (mut step, mut toks_seen) = (0u64, 0u64);
    let mut loss_ema = 0.0f32;
    let mut last = Instant::now();

    while t0.elapsed().as_secs() < budget {
        let s = rng.next(train.len() - seq - 1);
        let wlr = lr * ((step + 1) as f32 / warmup as f32).min(1.0);
        let l = dev.train_step(&ctx, &train[s..s + seq], &train[s + 1..s + seq + 1], wlr, 0);
        loss_ema = if step == 0 { l } else { 0.99 * loss_ema + 0.01 * l };
        step += 1; toks_seen += seq as u64;
        if last.elapsed().as_secs_f64() >= 5.0 {
            let secs = t0.elapsed().as_secs_f64();
            let bpb = heldout_bpb(&ctx, &dev, held, seq, &mut rng, 16);
            println!("[{secs:5.1}s] step {step:6} | {:6.0} tok/s | train {:.3} bits | held-out {bpb:.3} (floor {floor:.3})",
                     toks_seen as f64 / secs, loss_ema as f64 / LN2);
            last = Instant::now();
        }
    }

    let secs = t0.elapsed().as_secs_f64();
    let bpb = heldout_bpb(&ctx, &dev, held, seq, &mut rng, 64);
    println!("\n=== done: {step} steps in {secs:.1}s | {:.0} tok/s ===", toks_seen as f64 / secs);
    println!("held-out: {bpb:.3} bits/byte (floor {floor:.3})");
    println!("epochs: {:.2}", toks_seen as f64 / train.len() as f64);

    let prompt = b"fn ";
    let greedy = dev.generate(&ctx, prompt, 80);
    println!("\n[greedy]  {:?} -> {}", String::from_utf8_lossy(prompt), String::from_utf8_lossy(&greedy));
    for &temp in &[0.7f32, 1.0] {
        let g = dev.generate_temp(&ctx, prompt, 80, temp, 0xC0FFEE ^ temp.to_bits() as u64);
        println!("[temp {temp}] {:?} -> {}", String::from_utf8_lossy(prompt), String::from_utf8_lossy(&g));
    }
}
