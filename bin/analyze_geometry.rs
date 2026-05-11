//! analyze_geometry — Valida se o modelo SMG v3 aprende geometria de bytes.
//!
//! Treina SmgGridHsa por 20 épocas e depois analisa se bytes próximos no espaço
//! de Hamming (ex: 'A'/'a', distância 1) têm embeddings próximos no espaço
//! cosine. Uma correlação alta entre hamming_distance e (1 - cosine_sim) indica
//! que o modelo aprendeu estrutura geométrica do espaço de bytes sem inductive
//! bias explícito.
//!
//! ```bash
//! cargo run --release --features gpu-aql --bin analyze_geometry
//! cargo run --release --features gpu-aql --bin analyze_geometry -- /path/corpus.txt
//! ```

#![cfg(feature = "gpu-aql")]
#![allow(unsafe_code)]

use std::io::Read;

use blab::{
    gpu_aql::SmgGridHsa,
    learn::microcolumn_v3::{GridConfig, SmgGrid},
};

// ---- Hiperparâmetros (iguais a train_smg_hsa, excepto N_EPOCHS=20) ----
const WINDOW:          usize = 48;
const VOCAB:           usize = 256;
const LR_MAX:          f32   = 0.003;
const LR_MIN:          f32   = 0.0002;
const LR_WARMUP:       usize = 4;
const BATCH:           usize = 64;
const N_EPOCHS:        usize = 20;
const STEPS_PER_EPOCH: usize = 200;
const SEED:            u64   = 42;
const N_FAMILIES:      usize = 8;
const K:               usize = 32;

// ---- Funções de utilidade (copiadas de train_smg_hsa) ----

/// Cria pares (janela de 48 bytes, próximo byte) de um corpus.
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
            paths.sort();
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

// ---- Funções de análise geométrica ----

/// Distância de Hamming entre dois bytes (número de bits diferentes).
fn hamming_distance(a: u8, b: u8) -> u32 {
    (a ^ b).count_ones()
}

/// Cosine similarity entre dois vectores f32.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32  = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32  = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Extrai o embedding do byte `b` na família `f` a partir do tensor i_f achatado.
///
/// Shape de i_f: [N_FAMILIES, VOCAB, K] row-major.
fn byte_embedding<'a>(i_f: &'a [f32], family: usize, byte: u8) -> &'a [f32] {
    let base = family * VOCAB * K + (byte as usize) * K;
    &i_f[base..base + K]
}

/// Calcula cosine_sim média entre dois bytes ao longo de todas as famílias.
fn avg_family_cosine(i_f: &[f32], a: u8, b: u8) -> f32 {
    let mut total = 0.0f32;
    for f in 0..N_FAMILIES {
        let ea = byte_embedding(i_f, f, a);
        let eb = byte_embedding(i_f, f, b);
        total += cosine_similarity(ea, eb);
    }
    total / N_FAMILIES as f32
}

/// Coeficiente de correlação de Pearson entre dois slices de f32.
fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
    assert_eq!(xs.len(), ys.len(), "pearson: slices de tamanho diferente");
    let n = xs.len() as f32;
    if n < 2.0 {
        return 0.0;
    }
    let mx: f32 = xs.iter().sum::<f32>() / n;
    let my: f32 = ys.iter().sum::<f32>() / n;
    let cov: f32  = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum::<f32>();
    let sx: f32   = xs.iter().map(|x| (x - mx).powi(2)).sum::<f32>().sqrt();
    let sy: f32   = ys.iter().map(|y| (y - my).powi(2)).sum::<f32>().sqrt();
    if sx < 1e-12 || sy < 1e-12 {
        return 0.0;
    }
    (cov / (sx * sy)).clamp(-1.0, 1.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Análise de Geometria de Bytes (SMG v3) ===");
    println!("  window={WINDOW}  vocab={VOCAB}  batch={BATCH}  K={K}  N_FAMILIES={N_FAMILIES}");
    println!();

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
        "Total corpus: {} bytes ({:.1} KB)\n",
        data.len(),
        data.len() as f64 / 1024.0
    );
    if data.len() < WINDOW + 1 {
        eprintln!("ERRO: corpus demasiado pequeno.");
        std::process::exit(1);
    }

    // ---- Constrói pares de treino ----
    let mut all_pairs = make_pairs(&data);
    let n_val   = (all_pairs.len() / 10).max(50 * BATCH);
    let n_train = all_pairs.len() - n_val;
    let _val_pairs: Vec<_> = all_pairs.split_off(n_train);
    let mut train_pairs = all_pairs;

    println!(
        "Pares: {} treino  |  {} validação\n",
        train_pairs.len(),
        _val_pairs.len()
    );

    // ---- Constrói modelo ----
    let cfg = GridConfig {
        vocab_size:    VOCAB,
        window:        WINDOW,
        learning_rate: LR_MAX,
        seed:          SEED,
    };
    let cpu = SmgGrid::new(cfg);
    let mut hsa = SmgGridHsa::from_cpu(&cpu, BATCH)?;
    println!("Modelo SMG v3: {} parâmetros\n", hsa.n_params());

    // ---- Treino por N_EPOCHS épocas ----
    println!("Treino em {} épocas...", N_EPOCHS);
    println!(
        "{:>5}  {:>8}  {:>12}",
        "época", "lr", "loss_treino"
    );
    println!("{}", "─".repeat(32));

    let mut rng_seed = SEED;
    for ep in 0..N_EPOCHS {
        let lr = lr_schedule(ep, N_EPOCHS);
        hsa.set_lr(lr);

        rng_seed = rng_seed.wrapping_add(ep as u64 * 0x9e3779b97f4a7c15);
        shuffle(&mut train_pairs, rng_seed);

        let epoch_pairs: Vec<_> = train_pairs
            .iter()
            .take(STEPS_PER_EPOCH * BATCH)
            .cloned()
            .collect();

        let train_loss = hsa.train_epoch(&epoch_pairs, BATCH);
        println!("{:>5}  {:>8.5}  {:>12.4}", ep + 1, lr, train_loss);
    }
    println!();

    // ---- Extrai pesos para análise ----
    let weights = hsa.weights_to_host();
    let i_f = &weights.i_f;

    // Valida shape: deve ter N_FAMILIES * VOCAB * K elementos
    assert_eq!(
        i_f.len(),
        N_FAMILIES * VOCAB * K,
        "i_f com tamanho inesperado: {} (esperado {})",
        i_f.len(),
        N_FAMILIES * VOCAB * K
    );

    // ====================================================================
    // ANÁLISE 1: Pares Uppercase / Lowercase (Hamming = 1, bit 5)
    // ====================================================================
    println!("--- Pares Uppercase/Lowercase (Hamming=1, bit 5) ---");

    let mut case_sims: Vec<f32> = Vec::with_capacity(26);
    for letter in 0u8..26 {
        let upper = b'A' + letter;
        let lower = b'a' + letter;
        let h = hamming_distance(upper, lower);
        debug_assert_eq!(h, 1, "distância de Hamming para case deve ser 1");

        let sim = avg_family_cosine(i_f, upper, lower);
        case_sims.push(sim);
        println!(
            "  ('{}' = 0x{:02X}, '{}' = 0x{:02X}): cosine_sim média ({} famílias) = {:.4}",
            upper as char, upper,
            lower as char, lower,
            N_FAMILIES, sim
        );
    }
    let case_mean: f32 = case_sims.iter().sum::<f32>() / case_sims.len() as f32;
    println!("  Média H=1 (case): {:.4}\n", case_mean);

    // ====================================================================
    // ANÁLISE 2: Amostra de pares por distância de Hamming
    // ====================================================================
    println!("--- Amostra de pares por distância ---");

    // Gera todos os pares (a, b) com a < b e agrupa por distância
    // Para distâncias altas, usa amostragem determinística (máx. 200 pares)
    let target_distances: &[u32] = &[1, 2, 4, 8];
    let max_samples_per_dist = 200usize;

    let mut pearson_xs: Vec<f32> = Vec::new();  // hamming distance
    let mut pearson_ys: Vec<f32> = Vec::new();  // 1 - cosine_sim

    for &target_h in target_distances {
        let mut sims = Vec::new();
        let mut rng  = SEED ^ (target_h as u64 * 0xdeadbeef);

        // Colecta todos os pares com a distância alvo, amostrado se necessário
        for a in 0u8..=255 {
            for b in (a + 1)..=255 {
                if hamming_distance(a, b) == target_h {
                    // Guarda para análise
                    sims.push((a, b));
                }
            }
        }

        // Amostragem aleatória se demasiados pares
        if sims.len() > max_samples_per_dist {
            // Fisher-Yates parcial com LCG
            for i in 0..max_samples_per_dist {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = i + (rng >> 33) as usize % (sims.len() - i);
                sims.swap(i, j);
            }
            sims.truncate(max_samples_per_dist);
        }

        let n = sims.len();
        let mut total_sim = 0.0f32;
        for &(a, b) in &sims {
            let s = avg_family_cosine(i_f, a, b);
            total_sim += s;
            pearson_xs.push(target_h as f32);
            pearson_ys.push(1.0 - s);
        }
        let mean_sim = if n > 0 { total_sim / n as f32 } else { f32::NAN };

        println!(
            "  Hamming={}: cosine_sim média = {:.4}  (N={} pares)",
            target_h, mean_sim, n
        );
    }
    println!();

    // ====================================================================
    // ANÁLISE 3: Correlação de Pearson entre Hamming e (1 - cosine_sim)
    // ====================================================================
    println!("--- Correlação ---");

    let r = pearson(&pearson_xs, &pearson_ys);
    println!("  Pearson(hamming, 1 - cosine_sim) = {:.4}", r);

    let interpretacao = if r > 0.6 {
        "alta → modelo aprendeu geometria de bytes"
    } else if r > 0.3 {
        "moderada → geometria parcialmente aprendida"
    } else {
        "baixa → geometria precisa de inductive bias explícito"
    };
    println!("  Interpretação: {}", interpretacao);
    println!();

    // ====================================================================
    // ANÁLISE 4: Resumo final
    // ====================================================================
    println!("--- Resumo ---");
    println!(
        "  Bytes com Hamming=1 (case) têm cosine_sim média de {:.4}",
        case_mean
    );
    println!(
        "  Esperado alto (> 0.5) se o modelo aprendeu geometria naturalmente."
    );

    Ok(())
}
