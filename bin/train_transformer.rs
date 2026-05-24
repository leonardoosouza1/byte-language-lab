//! Real-corpus trainer for the device-resident byte transformer.
//!
//! Streams random windows from a byte corpus, trains `ModelDev` fully on GPU
//! (forward+backward+SGD, on-device grad-clip), and reports tok/s + held-out
//! bits/byte. This is the scale measurement behind the "what could a 1GB corpus
//! give us" question: it shows the real throughput and where a held-out byte
//! model lands vs the unigram floor.
//!
//! Usage:
//!   train_transformer [corpus_path] [budget_secs] [d_model] [n_layers] [max_seq]
//! Defaults: data/corpus_local.txt 60 128 4 128

use blab::gpu::transformer::{Config, Model, ModelDev};
use blab::gpu::GpuContext;
use std::time::Instant;

/// Deterministic LCG (Numerical Recipes) — no rand dep, reproducible windows.
struct Lcg(u64);
impl Lcg {
    fn next_usize(&mut self, bound: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % bound.max(1)
    }
}

/// Mean held-out bits/byte over `n_windows` random windows (lower = better;
/// log2(256)=8 is uniform, the unigram floor is the entropy of byte freqs).
fn heldout_bits_per_byte(ctx: &GpuContext, dev: &ModelDev, held: &[u8], seq: usize,
                         rng: &mut Lcg, n_windows: usize) -> f32 {
    if held.len() <= seq + 1 { return f32::NAN; }
    let v = 256usize;
    let mut total_bits = 0.0f64;
    let mut count = 0usize;
    for _ in 0..n_windows {
        let start = rng.next_usize(held.len() - seq - 1);
        let toks = &held[start..start + seq];
        let tgts = &held[start + 1..start + seq + 1];
        let logits = dev.forward(ctx, toks);            // [seq, v]
        for i in 0..seq {
            let row = &logits[i * v..(i + 1) * v];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|z| (z - mx).exp()).sum();
            let logp = (row[tgts[i] as usize] - mx) - sum.ln();   // natural log
            total_bits += -(logp as f64) / std::f64::consts::LN_2;
            count += 1;
        }
    }
    (total_bits / count as f64) as f32
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "data/corpus_local.txt".into());
    let budget: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60);
    let d_model: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(128);
    let n_layers: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);
    let max_seq: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(128);
    let lr: f32 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.1);
    let clip: f32 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let opt = args.get(8).cloned().unwrap_or_else(|| "adam".into());
    let batch: usize = args.get(9).and_then(|s| s.parse().ok()).unwrap_or(1).max(1);
    let qk_lr_mult: f32 = args.get(10).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let qk_norm: bool = args.get(11).map(|s| s == "1" || s == "true").unwrap_or(false);
    let alibi_slope: f32 = args.get(12).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let save_path = args.get(13).cloned().filter(|s| !s.is_empty());
    let load_path = args.get(14).cloned().filter(|s| !s.is_empty());

    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    println!("corpus {path}: {} bytes", bytes.len());
    // held-out = last 64KB (or 5%, whichever smaller); rest is train.
    let held_len = (bytes.len() / 20).min(65536);
    let split = bytes.len() - held_len;
    let (train, held) = bytes.split_at(split);
    println!("train {} bytes | held-out {} bytes", train.len(), held.len());

    // unigram floor on held-out (the bar a context-free model would hit).
    let mut cnt = [0u64; 256];
    for &b in held { cnt[b as usize] += 1; }
    let n = held.len() as f64;
    let floor: f64 = (0..256).filter(|&i| cnt[i] > 0)
        .map(|i| { let p = cnt[i] as f64 / n; -p * p.log2() })
        .sum();
    println!("held-out unigram floor: {floor:.3} bits/byte");

    let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU: {e}"); return; } };
    println!("GPU: {}", ctx.device_name());

    let cfg = Config { vocab: 256, d_model, n_layers, d_ff: d_model * 4, max_seq, n_predict: 1, qk_norm, alibi_slope };
    let n_params = 256 * d_model + max_seq * d_model
        + n_layers * (4 * d_model * d_model + 2 * d_model * (d_model * 4)) + d_model;
    println!("config: d_model={d_model} n_layers={n_layers} d_ff={} max_seq={max_seq} (~{:.2}M params)",
             d_model * 4, n_params as f64 / 1e6);

    let model = match &load_path {
        Some(p) => { println!("resuming from checkpoint {p}"); Model::load(p).expect("load checkpoint") }
        None => Model::init(cfg, 1234),
    };
    let mut dev = ModelDev::new(&ctx, &model);
    dev.set_qk_lr_mult(qk_lr_mult);
    if let Some(p) = &save_path { println!("checkpointing to {p} (periodic + final)"); }
    let mut rng = Lcg(0x1234_5678_9abc_def0);

    println!("optimizer={opt} lr={lr} clip={clip} batch={batch} qk_lr_mult={qk_lr_mult} qk_norm={qk_norm}");
    let t0 = Instant::now();
    let mut step = 0u64;
    let mut tokens_seen = 0u64;
    let mut loss_ema = 0.0f32;
    let mut last_report = Instant::now();
    let report_every = std::time::Duration::from_secs(5);
    let mut last_ckpt = Instant::now();
    let ckpt_every = std::time::Duration::from_secs(120);

    let warmup = 300u64;   // linear lr warmup — without it Adam blows up early (bc2≈1e-3)
    let mut bt = vec![0u8; batch * max_seq];   // batched tokens/targets scratch
    let mut bg = vec![0u8; batch * max_seq];
    while t0.elapsed().as_secs() < budget {
        let wlr = if opt == "sgd" { lr }
                  else { lr * ((step + 1) as f32 / warmup as f32).min(1.0) };
        let l = if batch == 1 {
            let start = rng.next_usize(train.len() - max_seq - 1);
            let toks = &train[start..start + max_seq];
            let tgts = &train[start + 1..start + max_seq + 1];
            if opt == "sgd" { dev.train_step_clip(&ctx, toks, tgts, wlr, clip) }
            else { dev.train_step_adam(&ctx, toks, tgts, wlr, clip) }
        } else {
            // gather B random windows into one block-diagonal batch
            for b in 0..batch {
                let start = rng.next_usize(train.len() - max_seq - 1);
                bt[b * max_seq..(b + 1) * max_seq].copy_from_slice(&train[start..start + max_seq]);
                bg[b * max_seq..(b + 1) * max_seq].copy_from_slice(&train[start + 1..start + max_seq + 1]);
            }
            dev.train_step_adam_batch(&ctx, &bt, &bg, wlr, clip, max_seq)
        };
        loss_ema = if step == 0 { l } else { 0.99 * loss_ema + 0.01 * l };
        step += 1;
        tokens_seen += (batch * max_seq) as u64;

        if last_report.elapsed() >= report_every {
            let secs = t0.elapsed().as_secs_f64();
            let tps = tokens_seen as f64 / secs;
            let bpb = heldout_bits_per_byte(&ctx, &dev, held, max_seq, &mut rng, 32);
            println!("[{secs:5.1}s] step {step:6} | {tps:8.0} tok/s | train loss {loss_ema:.3} \
                      | held-out {bpb:.3} bits/byte (floor {floor:.3})");
            last_report = Instant::now();
        }
        if save_path.is_some() && last_ckpt.elapsed() >= ckpt_every {
            if let Some(p) = &save_path {
                if let Err(e) = dev.to_model(&ctx).save(p) { eprintln!("checkpoint save failed: {e}"); }
                else { println!("  [checkpoint @ step {step}]"); }
            }
            last_ckpt = Instant::now();
        }
    }
    if let Some(p) = &save_path {
        dev.to_model(&ctx).save(p).expect("final checkpoint save");
        println!("final checkpoint saved to {p}");
    }

    let secs = t0.elapsed().as_secs_f64();
    let bpb = heldout_bits_per_byte(&ctx, &dev, held, max_seq, &mut rng, 128);
    println!("\n=== done: {step} steps in {secs:.1}s ===");
    println!("throughput: {:.0} tok/s | {:.2}M tokens total", tokens_seen as f64 / secs, tokens_seen as f64 / 1e6);
    println!("held-out: {bpb:.3} bits/byte (unigram floor {floor:.3}, uniform 8.0)");
    let epochs = tokens_seen as f64 / train.len() as f64;
    println!("corpus passes: {epochs:.2} epochs of {} bytes", train.len());
    let gb_hours = 1.0e9 / (tokens_seen as f64 / secs) / 3600.0;
    println!("extrapolation: 1 pass over 1GB ≈ {gb_hours:.1} h at this throughput");

    // DIAGNÓSTICO: forward latency vs seq → dispatch-bound (flat) vs compute-bound (cresce)
    println!("\n[forward latency vs seq] (50 iters)");
    for &s in &[1usize, 8, 32, 64, 128] {
        if train.len() < s + 1 { continue; }
        let w = &train[0..s];
        let _ = dev.forward(&ctx, w); // warmup
        let t = Instant::now();
        let iters = 50;
        for _ in 0..iters { let _ = dev.forward(&ctx, w); }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("  seq={s:3}  {ms:.3} ms/forward", );
    }

    // sample generation from a code-ish prompt: greedy (mode) + sampled (structure)
    let prompt = b"fn ";
    let g0 = Instant::now();
    let greedy = dev.generate(&ctx, prompt, 256);
    let gsecs = g0.elapsed().as_secs_f64();
    println!("\n[gen speed] 256 bytes em {:.3}s = {:.0} bytes/s ({:.2} ms/byte, batch=1 autoregressive)",
             gsecs, 256.0 / gsecs, gsecs * 1000.0 / 256.0);
    println!("[greedy]  {:?} -> {}", String::from_utf8_lossy(prompt),
             String::from_utf8_lossy(&greedy));

    // batched generation throughput (overhead-bound single stream → B× aggregate)
    println!("\n[gen batched] vs single ({:.0} bytes/s single):", 256.0 / gsecs);
    for &bsz in &[4usize, 8, 16, 32] {
        let b0 = Instant::now();
        let outs = dev.generate_batch(&ctx, prompt, bsz, 128, 1.0, 0xBEEF);
        let s = b0.elapsed().as_secs_f64();
        let total = (bsz * 128) as f64;
        println!("  batch={bsz:2}: {:.0} bytes/s agregado ({:.0}/stream) — {bsz}×128 bytes em {s:.3}s",
                 total / s, total / s / bsz as f64);
        let _ = outs;
    }
    for &t in &[0.7f32, 1.0] {
        let s = dev.generate_temp(&ctx, prompt, 96, t, 0xC0FFEE ^ (t.to_bits() as u64));
        println!("[temp {t}] {:?} -> {}", String::from_utf8_lossy(prompt),
                 String::from_utf8_lossy(&s));
    }
}
