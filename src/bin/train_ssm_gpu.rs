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
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
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
    let batch: usize = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(8).max(1);
    let ff: usize = a.get(8).and_then(|s| s.parse().ok()).unwrap_or(0);   // FFN hidden dim (0 = off)

    // Windowed (mmap-style) loading: the corpus is NOT read whole into RAM.
    // We keep the file open and seek+read each training window on demand — app
    // RAM stays ~constant regardless of corpus size (scales to GB; the OS page
    // cache keeps hot pages fast). Only the small held-out tail lives in RAM.
    let total = std::fs::metadata(&path).unwrap_or_else(|e| panic!("stat {path}: {e}")).len() as usize;
    let held_len = (total / 20).min(32768);
    let train_len = total - held_len;
    let mut file = File::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut held = vec![0u8; held_len];
    file.seek(SeekFrom::Start(train_len as u64)).unwrap();
    file.read_exact(&mut held).unwrap();
    println!("corpus {path}: {total} bytes (windowed read, RAM ~constant)");
    let mut cnt = [0u64; 256];
    for &b in &held { cnt[b as usize] += 1; }
    let n = held.len() as f64;
    let floor: f64 = (0..256).filter(|&i| cnt[i] > 0)
        .map(|i| { let p = cnt[i] as f64 / n; -p * p.log2() }).sum();
    println!("train {train_len} | held {held_len} bytes | unigram floor {floor:.3} bits/byte");

    let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU: {e}"); return; } };
    println!("GPU: {}", ctx.device_name());
    let n_params = 256 * d + n_layers * (3 * d * d + if ff > 0 { 2 * d * ff } else { 0 }) + d;
    println!("LIQUID/SSM (GPU): d={d} layers={n_layers} seq={seq} batch={batch} ff={ff} lr={lr} (~{:.2}M params)",
             n_params as f64 / 1e6);

    let mut dev = if ff > 0 {
        DevSsm::new_ffn(&ctx, 256, d, n_layers, ff, 1234)
    } else {
        DevSsm::new(&ctx, 256, d, n_layers, 1234)
    };
    let mut rng = Lcg(0xDEAD_BEEF_99);
    let t0 = Instant::now();
    let warmup = 200u64;
    let (mut step, mut toks_seen) = (0u64, 0u64);
    let mut loss_ema = 0.0f32;
    let mut last = Instant::now();

    let mut bt = vec![0u8; batch * seq];   // block-diagonal batch scratch
    let mut bg = vec![0u8; batch * seq];
    let mut win = vec![0u8; seq + 1];      // reused per-window read buffer
    while t0.elapsed().as_secs() < budget {
        let wlr = lr * ((step + 1) as f32 / warmup as f32).min(1.0);
        for b in 0..batch {
            let s = rng.next(train_len - seq - 1);
            file.seek(SeekFrom::Start(s as u64)).unwrap();
            file.read_exact(&mut win).unwrap();           // seek+read 1 window (page-cached)
            bt[b * seq..(b + 1) * seq].copy_from_slice(&win[..seq]);
            bg[b * seq..(b + 1) * seq].copy_from_slice(&win[1..seq + 1]);
        }
        let l = dev.train_step(&ctx, &bt, &bg, wlr, seq);   // block=seq → block-diagonal
        loss_ema = if step == 0 { l } else { 0.99 * loss_ema + 0.01 * l };
        step += 1; toks_seen += (batch * seq) as u64;
        if last.elapsed().as_secs_f64() >= 5.0 {
            let secs = t0.elapsed().as_secs_f64();
            let bpb = heldout_bpb(&ctx, &dev, &held, seq, &mut rng, 16);
            println!("[{secs:5.1}s] step {step:6} | {:6.0} tok/s | train {:.3} bits | held-out {bpb:.3} (floor {floor:.3})",
                     toks_seen as f64 / secs, loss_ema as f64 / LN2);
            last = Instant::now();
        }
    }

    let secs = t0.elapsed().as_secs_f64();
    let bpb = heldout_bpb(&ctx, &dev, &held, seq, &mut rng, 64);
    println!("\n=== done: {step} steps in {secs:.1}s | {:.0} tok/s ===", toks_seen as f64 / secs);
    println!("held-out: {bpb:.3} bits/byte (floor {floor:.3})");
    println!("epochs: {:.2}", toks_seen as f64 / train_len as f64);

    let prompt = b"fn ";
    let greedy = dev.generate(&ctx, prompt, 80);
    println!("\n[greedy]  {:?} -> {}", String::from_utf8_lossy(prompt), String::from_utf8_lossy(&greedy));
    for &temp in &[0.7f32, 1.0] {
        let g = dev.generate_temp(&ctx, prompt, 80, temp, 0xC0FFEE ^ temp.to_bits() as u64);
        println!("[temp {temp}] {:?} -> {}", String::from_utf8_lossy(prompt), String::from_utf8_lossy(&g));
    }
}
