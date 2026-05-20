//! train_smg_hsa — treino byte-level real em AMD RDNA2 (HSA AQL).
//!
//! Lê bytes de um ficheiro (ou do código-fonte do projecto como default),
//! treina SMG v3 via HSA e reporta loss + bits/byte por época.
//!
//! ```bash
//! # Treino no código-fonte do projecto (default)
//! cargo run --release --features gpu-aql --bin train_smg_hsa
//!
//! # Treino num ficheiro específico
//! cargo run --release --features gpu-aql --bin train_smg_hsa -- /caminho/para/texto.txt
//! ```
//!
//! Baseline aleatório: ln(256) ≈ 5.545 nats = 8.000 bits/byte.
//! Bom modelo em código Rust: ~3.5–4.5 nats = 5.0–6.5 bits/byte.
//!
//! # LR Schedule (cosine com warmup)
//! Épocas 0..LR_WARMUP: rampa linear LR_MIN→LR_MAX.
//! Épocas LR_WARMUP..N_EPOCHS: cosine decay LR_MAX→LR_MIN.
//! Evita divergência inicial e permite fine-tuning na fase final.

#![cfg(feature = "gpu-aql")]
#![allow(unsafe_code)]

use std::io::Read;
use std::time::Instant;

use blab::{
    gpu_aql::SmgGridHsa,
    learn::microcolumn_v3::{GridConfig, SmgGrid},
};

// ---- Hiperparâmetros ----
const WINDOW:          usize = 96;
const VOCAB:           usize = 256;
const LR_MAX:          f32   = 0.006;   // 2× original (linear scaling pra batch 4×)
const LR_MIN:          f32   = 0.0002;
const LR_WARMUP:       usize = 4;
const BATCH:           usize = 256;     // sweet spot: compute-saturated, async pipeline win
const N_EPOCHS:        usize = 50;
const STEPS_PER_EPOCH: usize = 500;     // 500 × 256 ≈ 12 MB texto/epoch
const VAL_STEPS:       usize = 50;      // steps de validação por avaliação
const SEED:            u64   = 42;
const TEMPERATURE:     f32   = 0.8;     // temperatura de sampling (< 1 = mais focado)
const TOP_K:           usize = 20;      // top-k truncation antes de amostrar

// ---- Utilidades ----

/// Cria pares (janela de 48 bytes, próximo byte) de um corpus de bytes.
fn make_pairs(data: &[u8]) -> Vec<(Vec<u32>, u32)> {
    if data.len() <= WINDOW {
        panic!("corpus demasiado pequeno (precisa > {} bytes)", WINDOW);
    }
    data.windows(WINDOW + 1)
        .map(|w| {
            let win: Vec<u32> = w[..WINDOW].iter().map(|&b| b as u32).collect();
            let tgt = w[WINDOW] as u32;
            (win, tgt)
        })
        .collect()
}

/// Cross-entropy de um batch de logits vs targets (computado em CPU).
fn batch_loss(logits_flat: &[Vec<f32>], targets: &[u32]) -> f32 {
    let mut total = 0.0f32;
    for (logits, &tgt) in logits_flat.iter().zip(targets) {
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - mx).exp()).collect();
        let s: f32 = exps.iter().sum();
        let p = (exps[tgt as usize] / s).max(1e-20);
        total -= p.ln();
    }
    total / logits_flat.len() as f32
}

/// Embaralha um slice usando LCG determinístico.
fn shuffle<T>(v: &mut Vec<T>, seed: u64) {
    let mut rng = seed;
    for i in (1..v.len()).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng >> 33) as usize % (i + 1);
        v.swap(i, j);
    }
}

/// Cosine LR schedule com warmup linear.
///
/// - Épocas 0..LR_WARMUP: rampa linear LR_MIN → LR_MAX.
/// - Épocas LR_WARMUP..n_epochs: cosine decay LR_MAX → LR_MIN.
fn lr_schedule(ep: usize, n_epochs: usize) -> f32 {
    if ep < LR_WARMUP {
        LR_MIN + (LR_MAX - LR_MIN) * (ep + 1) as f32 / LR_WARMUP as f32
    } else {
        let t     = (ep - LR_WARMUP) as f32;
        let t_max = (n_epochs - LR_WARMUP).max(1) as f32;
        let cos   = (std::f32::consts::PI * t / t_max).cos();
        LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + cos)
    }
}

/// Sampling com temperatura e top-k a partir de logits brutos.
///
/// `temperature < 1.0` → distribuição mais concentrada (menos aleatório).
/// `top_k` → zera todos os tokens fora do top-k antes de amostrar.
fn sample_topk(logits: &[f32], temperature: f32, top_k: usize, rng: &mut u64) -> u32 {
    let inv_t = 1.0_f32 / temperature.max(1e-6);
    let mx    = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Probabilidades não normalizadas com temperatura aplicada
    let mut probs: Vec<(usize, f32)> = logits.iter()
        .enumerate()
        .map(|(i, &x)| (i, ((x - mx) * inv_t).exp()))
        .collect();

    // Top-k: mantém apenas os k tokens mais prováveis
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1)
        .unwrap_or(std::cmp::Ordering::Equal));
    probs.truncate(top_k.min(probs.len()));

    let s: f32 = probs.iter().map(|(_, p)| p).sum();
    if s == 0.0 { return probs[0].0 as u32; }

    // Amostragem com LCG determinístico
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let r = (*rng >> 11) as f32 / ((1u64 << 53) as f32);

    let mut cum = 0.0_f32;
    for &(i, p) in &probs {
        cum += p / s;
        if r <= cum { return i as u32; }
    }
    probs[0].0 as u32
}

/// Corpus default: todos os ficheiros .rs e .hip do projecto.
fn load_default_corpus() -> Vec<u8> {
    let mut buf = Vec::new();
    let dirs = ["src", "bin", "kernels"];
    let exts = [".rs", ".hip"];
    for dir in &dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            let mut paths: Vec<_> = entries
                .flatten()
                .filter(|e| exts.iter().any(|ext| e.file_name().to_string_lossy().ends_with(ext)))
                .map(|e| e.path())
                .collect();
            paths.sort();  // ordem determinística
            for path in paths {
                if let Ok(mut f) = std::fs::File::open(&path) {
                    let before = buf.len();
                    f.read_to_end(&mut buf).ok();
                    let added = buf.len() - before;
                    if added > 0 {
                        eprintln!("  + {} ({} bytes)", path.display(), added);
                        buf.push(b'\n');
                    }
                }
            }
        }
    }
    buf
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== train_smg_hsa: byte-level learning em AMD RDNA2 ===");
    println!("  window={WINDOW}  vocab={VOCAB}  batch={BATCH}");
    println!("  lr: {LR_MIN:.4}→{LR_MAX:.4} (warmup={LR_WARMUP})→{LR_MIN:.4} (cosine)");
    println!("  {STEPS_PER_EPOCH} steps/epoch × {N_EPOCHS} epochs\n");

    // ---- Carrega corpus ----
    let data: Vec<u8> = match std::env::args().nth(1) {
        Some(path) => {
            println!("Corpus: {}", path);
            let mut f = std::fs::File::open(&path)?;
            let mut buf = Vec::new();
            f.read_to_end(&mut buf)?;
            buf
        }
        None => {
            println!("Corpus: código-fonte do projecto");
            load_default_corpus()
        }
    };

    println!(
        "\nTotal corpus: {} bytes ({:.1} KB)\n",
        data.len(),
        data.len() as f64 / 1024.0
    );
    if data.len() < WINDOW + 1 {
        eprintln!("ERRO: corpus demasiado pequeno.");
        std::process::exit(1);
    }

    // ---- Cria pares e split train/val ----
    let mut all_pairs = make_pairs(&data);
    // Separação por posição: 90% treino, 10% validação (sem sobreposição de janelas)
    let n_val   = (all_pairs.len() / 10).max(VAL_STEPS * BATCH);
    let n_train = all_pairs.len() - n_val;
    let val_pairs: Vec<_> = all_pairs.split_off(n_train);
    let mut train_pairs   = all_pairs;

    println!(
        "Pares: {} treino  |  {} validação",
        train_pairs.len(),
        val_pairs.len()
    );
    println!(
        "Baseline aleatório: {:.4} nats = {:.3} bits/byte\n",
        (VOCAB as f32).ln(),
        (VOCAB as f32).log2()
    );

    // ---- Constrói modelo HSA ----
    let cfg = GridConfig {
        vocab_size:    VOCAB,
        window:        WINDOW,
        learning_rate: LR_MAX,
        seed:          SEED,
    };
    let cpu = SmgGrid::new(cfg);
    let mut hsa = SmgGridHsa::from_cpu(&cpu, BATCH)?;
    println!("Modelo SMG v3: {} parâmetros\n", hsa.n_params());

    // ---- Cabeçalho da tabela ----
    println!(
        "{:>5}  {:>7}  {:>10}  {:>10}  {:>10}  {:>8}",
        "época", "lr", "loss_treino", "loss_val", "bits/byte", "tempo"
    );
    println!("{}", "─".repeat(62));

    // ---- ASYNC PIPELINE: CPU prep worker overlaps GPU compute ----
    // While the GPU trains epoch N, a CPU worker shuffles + clones the pairs
    // for epoch N+1. Bounded channel keeps memory in check (max 2 epochs in
    // flight). User idea: "1 worker GPU, 1 worker CPU, ambos contribuem".
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<(Vec<u32>, u32)>>(2);
    let n_epochs = N_EPOCHS;
    let producer = std::thread::spawn(move || {
        let mut rng_seed = SEED;
        for ep in 0..n_epochs {
            rng_seed = rng_seed.wrapping_add(ep as u64 * 0x9e3779b97f4a7c15);
            shuffle(&mut train_pairs, rng_seed);
            let epoch_pairs: Vec<_> = train_pairs
                .iter()
                .take(STEPS_PER_EPOCH * BATCH)
                .cloned()
                .collect();
            if tx.send(epoch_pairs).is_err() { break; }
        }
    });

    // ---- Loop de treino ----
    let total_t0 = Instant::now();

    for ep in 0..N_EPOCHS {
        // Cosine LR schedule — aplicado antes de cada época
        let lr = lr_schedule(ep, N_EPOCHS);
        hsa.set_lr(lr);

        let t0 = Instant::now();
        let epoch_pairs = match rx.recv() {
            Ok(p) => p,
            Err(_) => break,
        };

        // Treina (forward + backward + SGD por batch)
        let train_loss = hsa.train_epoch(&epoch_pairs, BATCH);
        let elapsed = t0.elapsed();

        // Validação a cada 3 épocas (ou na primeira e última)
        let val_loss = if ep % 3 == 0 || ep == 0 || ep == N_EPOCHS - 1 {
            // Usa primeiros VAL_STEPS × BATCH pares de validação
            let val_slice: Vec<_> = val_pairs
                .iter()
                .take(VAL_STEPS * BATCH)
                .cloned()
                .collect();

            // Forward sem backward
            let mut total_loss = 0.0f32;
            let mut n_batches  = 0usize;
            for chunk in val_slice.chunks(BATCH) {
                if chunk.is_empty() { continue; }
                let windows: Vec<Vec<u32>> = chunk.iter().map(|(w, _)| w.clone()).collect();
                let targets: Vec<u32>       = chunk.iter().map(|(_, t)| *t).collect();
                let logits = hsa.forward_batch(&windows);
                total_loss += batch_loss(&logits, &targets);
                n_batches  += 1;
            }
            if n_batches > 0 { total_loss / n_batches as f32 } else { f32::NAN }
        } else {
            f32::NAN
        };

        let bits_str = if val_loss.is_nan() {
            "          —".to_string()
        } else {
            format!("{:>10.3}", val_loss / std::f32::consts::LN_2)
        };
        let val_str = if val_loss.is_nan() {
            "          —".to_string()
        } else {
            format!("{:>10.4}", val_loss)
        };

        println!(
            "{:>5}  {:>7.5}  {:>10.4}  {}  {}  {:>8.2?}",
            ep + 1, lr, train_loss, val_str, bits_str, elapsed
        );
    }

    println!(
        "\nTreino completo: {:.1?} total\n",
        total_t0.elapsed()
    );

    // ---- Geração com temperatura + top-k (3 amostras de posições distintas) ----
    println!("=== Geração (temp={TEMPERATURE}, top-{TOP_K}, 128 bytes × 3 amostras) ===");

    let mut gen_rng = SEED ^ 0xc0ffee_u64;

    // Três sementes em posições distintas do corpus
    let offsets = [
        0usize,
        (data.len() / 3).saturating_sub(WINDOW),
        (data.len() * 2 / 3).saturating_sub(WINDOW),
    ];

    for (n, &off) in offsets.iter().enumerate() {
        let off = off.min(data.len().saturating_sub(WINDOW + 1));

        let seed_bytes: Vec<u32> = data[off..off + WINDOW]
            .iter().map(|&b| b as u32).collect();
        let seed_str: String = data[off..off + WINDOW]
            .iter()
            .map(|&b| if b.is_ascii_graphic() || b == b' ' || b == b'\n' || b == b'\t' {
                b as char
            } else {
                '·'
            })
            .collect();

        println!("\nAmostra {} (offset={}):", n + 1, off);
        println!("  seed:   {:?}", seed_str);

        let mut window = seed_bytes;
        let mut generated = Vec::with_capacity(128);
        for _ in 0..128 {
            let logits = &hsa.forward_batch(&[window.clone()])[0];
            let next = sample_topk(logits, TEMPERATURE, TOP_K, &mut gen_rng);
            generated.push(next as u8);
            window.remove(0);
            window.push(next);
        }

        let gen_str: String = generated.iter()
            .map(|&b| if b.is_ascii_graphic() || b == b' ' || b == b'\n' || b == b'\t' {
                b as char
            } else {
                '·'
            })
            .collect();
        println!("  gerado: {:?}", gen_str);
    }

    Ok(())
}
