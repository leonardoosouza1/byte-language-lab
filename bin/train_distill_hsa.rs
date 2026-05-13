//! train_distill_hsa.rs — treina SMG via DISTILLATION de Qwen.
//!
//! Lê data/qwen_distill.bin (saída de collect_qwen_distill.py).
//! Cada amostra: 96 bytes contexto + distribuição f32[256] do Qwen.
//! Usa SmgGridHsa::train_step_distill (KL gradient).
//!
//! Saída: dialogue_weights.blab (compatível com chat_hsa).

use std::fs::File;
use std::io::{Read, Write as _};
use std::path::Path;

use blab::gpu_aql::SmgGridHsa;
use blab::learn::microcolumn_v3::{GridConfig, SmgGrid};

// ── Hyperparams ───────────────────────────────────────────────────────────
const WINDOW:          usize = 96;
const VOCAB:           usize = 256;
const LR:              f32   = 0.003;
const BATCH:           usize = 32;
const N_EPOCHS:        usize = 80;
const SEED:            u64   = 7;
const DATA_PATH:       &str  = "data/qwen_distill.bin";
const OUT_WEIGHTS:     &str  = "dialogue_weights.blab";

// ── Carrega .bin ───────────────────────────────────────────────────────────
fn load_distill(path: &str) -> std::io::Result<(usize, Vec<(Vec<u8>, Vec<f32>)>)> {
    let mut f = File::open(path)?;
    let mut hdr = [0u8; 8];
    f.read_exact(&mut hdr)?;
    let n   = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as usize;
    let win = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
    println!("  qwen_distill.bin: {} samples, window={}", n, win);

    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let mut ctx = vec![0u8; win];
        f.read_exact(&mut ctx)?;
        let mut dist = vec![0f32; 256];
        for d in dist.iter_mut() {
            let mut bb = [0u8; 4];
            f.read_exact(&mut bb)?;
            *d = f32::from_le_bytes(bb);
        }
        samples.push((ctx, dist));
    }
    Ok((win, samples))
}

// ── .blab serializer (compat com chat_hsa) ────────────────────────────────
const BLAB_MAGIC:   u32 = 0x424C4142;
const BLAB_VERSION: u32 = 1;

fn save_weights(path: &str, hsa: &SmgGridHsa) -> std::io::Result<()> {
    let wh = hsa.weights_to_host();
    let mut f = File::create(path)?;
    f.write_all(&BLAB_MAGIC.to_le_bytes())?;
    f.write_all(&BLAB_VERSION.to_le_bytes())?;
    let tensors = [
        &wh.t_f, &wh.i_f, &wh.nibble_h, &wh.nibble_l,
        &wh.b_f, &wh.pos_bias, &wh.lat_w, &wh.w_out, &wh.b_out,
    ];
    for t in tensors {
        f.write_all(&(t.len() as u32).to_le_bytes())?;
        for &x in t {
            f.write_all(&x.to_le_bytes())?;
        }
    }
    Ok(())
}

// ── RNG ─────────────────────────────────────────────────────────────────────
fn xorshift(s: &mut u64) -> u32 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s >> 16) as u32
}

fn shuffle<T>(v: &mut [T], seed: u64) {
    let mut s = seed.max(1);
    for i in (1..v.len()).rev() {
        let r = xorshift(&mut s) as usize;
        v.swap(i, r % (i + 1));
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   train_distill_hsa — SMG via distill Qwen (GPU/HSA)       ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    if !Path::new(DATA_PATH).exists() {
        eprintln!("ERRO: {} não existe. Rode data/collect_qwen_distill.py antes.", DATA_PATH);
        std::process::exit(1);
    }

    // ── Init HSA early ────────────────────────────────────────────────
    // (SmgGridHsa::from_cpu já cria contexto internamente)

    let (win, mut samples) = load_distill(DATA_PATH)?;
    if win != WINDOW {
        eprintln!("ERRO: window mismatch (.bin={}, code={})", win, WINDOW);
        std::process::exit(1);
    }

    let cfg = GridConfig {
        vocab_size:    VOCAB,
        window:        WINDOW,
        learning_rate: LR,
        seed:          SEED,
    };
    let cpu = SmgGrid::new(cfg.clone());
    let mut hsa = SmgGridHsa::from_cpu(&cpu, BATCH)?;
    println!("Modelo: {} parâmetros", hsa.n_params());
    println!("Samples: {}\n", samples.len());

    println!("{:>5}  {:>10}  {:>6}", "época", "loss_kl", "tempo");
    println!("{}", "─".repeat(30));

    let t0 = std::time::Instant::now();
    for epoch in 0..N_EPOCHS {
        let ep0 = std::time::Instant::now();

        // LR schedule warmup+cosine
        let warm = (N_EPOCHS / 10).max(1);
        let lr = if epoch < warm {
            LR * (epoch + 1) as f32 / warm as f32
        } else {
            let p = (epoch - warm) as f32 / (N_EPOCHS - warm) as f32;
            LR * 0.5 * (1.0 + (std::f32::consts::PI * p).cos()).max(0.1)
        };
        hsa.set_lr(lr);

        shuffle(&mut samples, SEED ^ epoch as u64);
        let mut loss_sum = 0.0;
        let mut n_batch = 0usize;
        for chunk in samples.chunks(BATCH) {
            if chunk.len() < BATCH { continue; }
            let windows: Vec<Vec<u32>> = chunk.iter().map(|(c, _)| {
                c.iter().map(|&b| b as u32).collect()
            }).collect();
            let dists: Vec<Vec<f32>> = chunk.iter().map(|(_, d)| d.clone()).collect();
            loss_sum += hsa.train_step_distill(&windows, &dists);
            n_batch += 1;
        }
        let l = if n_batch > 0 { loss_sum / n_batch as f32 } else { f32::NAN };
        println!("  {:>3}    {:>10.4}  {:>5.1}s", epoch, l, ep0.elapsed().as_secs_f32());
        std::io::stdout().flush().ok();
    }

    println!("\n━━━ Wall: {:.0}s ━━━", t0.elapsed().as_secs_f32());

    println!("\n▸ Salvando {}", OUT_WEIGHTS);
    save_weights(OUT_WEIGHTS, &hsa)?;
    let sz = std::fs::metadata(OUT_WEIGHTS)?.len();
    println!("  {} bytes ({:.2} MB)", sz, sz as f32 / 1e6);

    println!("\n▸ Teste com chat_hsa:\n  cargo run --release --features gpu-aql --bin chat_hsa");
    Ok(())
}
