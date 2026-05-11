//! generate_hsa — Geração autoregressiva byte-a-byte com SMG v3 em RDNA2.
//!
//! Fluxo:
//!   1. Carrega corpus (código-fonte ou ficheiro passado como argumento).
//!   2. Treina SmgGridHsa por N_EPOCHS épocas.
//!   3. Gera ~GEN_LEN bytes a partir de vários prompts de teste.
//!   4. Reporta: loss de validação (nats/bits), diversidade e entropia.
//!
//! Amostragem suportada:
//!   - Greedy (temperature=0.0) → argmax puro
//!   - Temperature (temperature>0.0) → softmax + amostragem multinomial
//!   - Top-k filtering (k>0) → restringe ao top-k logits antes do softmax
//!
//! ```bash
//! cargo run --release --features gpu-aql --bin generate_hsa
//! cargo run --release --features gpu-aql --bin generate_hsa -- /caminho/para/corpus.txt
//! ```

#![cfg(feature = "gpu-aql")]
#![allow(unsafe_code)]

use std::io::Read as _;
use std::time::Instant;

use blab::{
    gpu_aql::SmgGridHsa,
    learn::microcolumn_v3::{GridConfig, SmgGrid},
};

// ---- Hiperparâmetros de treino ----
const WINDOW:          usize = 48;
const VOCAB:           usize = 256;
const LR:              f32   = 0.002;
const BATCH:           usize = 64;
const N_EPOCHS:        usize = 40;
const STEPS_PER_EPOCH: usize = 200;
const VAL_STEPS:       usize = 50;
const SEED:            u64   = 42;

// ---- Hiperparâmetros de geração ----
const GEN_LEN:         usize = 512;   // bytes a gerar por prompt
const TEMPERATURE:     f32   = 0.8;   // 0.0 = greedy; >0 = amostrado
const TOP_K:           usize = 40;    // 0 = desactivado

// ---- Prompts de teste (serão truncados/padded a WINDOW bytes) ----
const PROMPTS: &[&str] = &[
    "fn main() -> Result<(), Box<dyn std::error",
    "// SMG v3 — byte-level language model on",
    "impl SmgGridHsa {\n    pub fn forward_batch",
    "use hip_ffi::safe::{DeviceBuffer, hip_h2d",
    "let mut rng = seed;\nfor i in (1..v.len())",
];

// ---------------------------------------------------------------------------
// Utilitários
// ---------------------------------------------------------------------------

fn softmax_prob(logits: &[f32], idx: usize) -> f32 {
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - mx).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps[idx] / s
}

fn batch_cross_entropy(logits_batch: &[Vec<f32>], targets: &[u32]) -> f32 {
    let mut total = 0.0f32;
    for (logits, &tgt) in logits_batch.iter().zip(targets) {
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - mx).exp()).collect();
        let s: f32 = exps.iter().sum();
        let p = (exps[tgt as usize] / s).max(1e-20);
        total -= p.ln();
    }
    total / logits_batch.len() as f32
}

fn shuffle<T>(v: &mut Vec<T>, seed: u64) {
    let mut rng = seed;
    for i in (1..v.len()).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng >> 33) as usize % (i + 1);
        v.swap(i, j);
    }
}

/// LCG simples para amostragem — sem deps rand.
fn lcg_next(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    // retorna f32 em [0, 1)
    (*state >> 11) as f32 / (1u64 << 53) as f32
}

/// Amostragem com temperatura e top-k a partir de logits brutos.
///
/// - `temperature = 0.0` → greedy (argmax)
/// - `top_k = 0`        → usa todos os logits
fn sample(logits: &[f32], temperature: f32, top_k: usize, rng: &mut u64) -> usize {
    if temperature == 0.0 {
        return logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    // Aplica temperatura
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    // Top-k filtering
    let mut indices: Vec<usize> = (0..scaled.len()).collect();
    if top_k > 0 && top_k < scaled.len() {
        indices.sort_unstable_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal));
        indices.truncate(top_k);
    }

    // Softmax sobre os índices seleccionados
    let mx = indices.iter().map(|&i| scaled[i]).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = indices.iter().map(|&i| (scaled[i] - mx).exp()).collect();
    let s: f32 = exps.iter().sum();

    // CDF sampling
    let u = lcg_next(rng);
    let mut cdf = 0.0f32;
    for (&idx, &e) in indices.iter().zip(&exps) {
        cdf += e / s;
        if u <= cdf {
            return idx;
        }
    }
    *indices.last().unwrap()
}

fn make_pairs(data: &[u8]) -> Vec<(Vec<u32>, u32)> {
    if data.len() <= WINDOW {
        panic!("corpus demasiado pequeno (> {} bytes necessários)", WINDOW);
    }
    data.windows(WINDOW + 1)
        .map(|w| {
            let win: Vec<u32> = w[..WINDOW].iter().map(|&b| b as u32).collect();
            let tgt = w[WINDOW] as u32;
            (win, tgt)
        })
        .collect()
}

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
                    f.read_to_end(&mut buf).ok();
                    buf.push(b'\n');
                }
            }
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// Validação
// ---------------------------------------------------------------------------

fn val_loss(hsa: &SmgGridHsa, val_pairs: &[(Vec<u32>, u32)]) -> f32 {
    let n = (VAL_STEPS * BATCH).min(val_pairs.len());
    if n == 0 { return f32::NAN; }

    let mut total = 0.0f32;
    let mut nb    = 0usize;
    for chunk in val_pairs[..n].chunks(BATCH) {
        if chunk.is_empty() { continue; }
        let windows: Vec<Vec<u32>> = chunk.iter().map(|(w, _)| w.clone()).collect();
        let targets: Vec<u32>      = chunk.iter().map(|(_, t)| *t).collect();
        total += batch_cross_entropy(&hsa.forward_batch(&windows), &targets);
        nb    += 1;
    }
    if nb > 0 { total / nb as f32 } else { f32::NAN }
}

// ---------------------------------------------------------------------------
// Geração autoregressiva
// ---------------------------------------------------------------------------

struct GenResult {
    text:       Vec<u8>,      // bytes gerados (inclui prompt)
    perplexity: f32,          // exp(mean cross-entropy por byte gerado)
    diversity:  usize,        // n.º de bytes únicos gerados
    entropy:    f32,          // entropia empírica da distribuição gerada (bits)
}

fn generate(
    hsa:         &SmgGridHsa,
    prompt_str:  &str,
    gen_len:     usize,
    temperature: f32,
    top_k:       usize,
    rng:         &mut u64,
) -> GenResult {
    // Constrói janela inicial a partir do prompt (preenchida com espaços se curto)
    let prompt_bytes = prompt_str.as_bytes();
    let mut window: Vec<u32> = if prompt_bytes.len() >= WINDOW {
        prompt_bytes[prompt_bytes.len() - WINDOW..]
            .iter().map(|&b| b as u32).collect()
    } else {
        let pad = WINDOW - prompt_bytes.len();
        std::iter::repeat(b' ' as u32)
            .take(pad)
            .chain(prompt_bytes.iter().map(|&b| b as u32))
            .collect()
    };

    let mut generated: Vec<u8> = Vec::with_capacity(gen_len);
    let mut log_sum = 0.0f32;  // acumula -ln P(next | ctx) para perplexidade

    for _ in 0..gen_len {
        let logits = &hsa.forward_batch(&[window.clone()])[0];
        let next   = sample(logits, temperature, top_k, rng) as u32;

        // Acumula probabilidade do token gerado (para perplexidade interna)
        log_sum -= softmax_prob(logits, next as usize).max(1e-20).ln();

        generated.push(next as u8);

        // Desliza janela
        window.remove(0);
        window.push(next);
    }

    // Métricas
    let perplexity = (log_sum / gen_len as f32).exp();

    let mut freq = [0u32; 256];
    for &b in &generated { freq[b as usize] += 1; }
    let diversity = freq.iter().filter(|&&c| c > 0).count();

    // Entropia empírica H = -Σ p·log2(p)
    let n = gen_len as f32;
    let entropy = freq.iter()
        .filter(|&&c| c > 0)
        .map(|&c| { let p = c as f32 / n; -p * p.log2() })
        .sum::<f32>();

    GenResult { text: generated, perplexity, diversity, entropy }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== generate_hsa: geração autoregressiva SMG v3 em RDNA2 ===");
    println!("  window={WINDOW}  vocab={VOCAB}  batch={BATCH}  lr={LR:.4}");
    println!("  {N_EPOCHS} epochs × {STEPS_PER_EPOCH} steps/epoch");
    println!("  gen_len={GEN_LEN}  temperature={TEMPERATURE}  top_k={TOP_K}\n");

    // ---- Corpus ----
    let data: Vec<u8> = match std::env::args().nth(1) {
        Some(path) => {
            println!("Corpus: {}", path);
            let mut buf = Vec::new();
            std::fs::File::open(&path)?.read_to_end(&mut buf)?;
            buf
        }
        None => {
            println!("Corpus: código-fonte do projecto (src/ + bin/ + kernels/)");
            load_default_corpus()
        }
    };
    println!("  {} bytes ({:.1} KB)\n", data.len(), data.len() as f64 / 1024.0);

    let mut all_pairs = make_pairs(&data);
    let n_val   = (all_pairs.len() / 10).max(VAL_STEPS * BATCH);
    let n_train = all_pairs.len() - n_val;
    let val_pairs: Vec<_>  = all_pairs.split_off(n_train);
    let mut train_pairs    = all_pairs;
    println!(
        "Pares: {} treino | {} validação\n",
        train_pairs.len(), val_pairs.len()
    );

    // ---- Modelo ----
    let cfg = GridConfig {
        vocab_size:    VOCAB,
        window:        WINDOW,
        learning_rate: LR,
        seed:          SEED,
    };
    let cpu_init = SmgGrid::new(cfg.clone());
    let mut hsa  = SmgGridHsa::from_cpu(&cpu_init, BATCH)?;
    println!("Modelo: {} parâmetros\n", hsa.n_params());

    // ---- Treino ----
    println!("{:>5}  {:>10}  {:>10}  {:>8}", "época", "train_loss", "val_loss", "tempo");
    println!("{}", "─".repeat(42));

    let t_train = Instant::now();
    let mut rng_seed = SEED;

    for ep in 0..N_EPOCHS {
        // Learning rate schedule: warm-up linear 0→LR nas primeiras 5 épocas,
        // depois constante até 30, cosine decay 30→40.
        let lr = if ep < 5 {
            LR * (ep + 1) as f32 / 5.0
        } else if ep < 30 {
            LR
        } else {
            let t = (ep - 30) as f32 / (N_EPOCHS - 30) as f32;
            LR * 0.5 * (1.0 + (std::f32::consts::PI * t).cos())
        };
        hsa.set_lr(lr);

        rng_seed = rng_seed.wrapping_add(ep as u64 * 0x9e3779b97f4a7c15);
        shuffle(&mut train_pairs, rng_seed);

        let epoch_pairs: Vec<_> = train_pairs
            .iter()
            .take(STEPS_PER_EPOCH * BATCH)
            .cloned()
            .collect();

        let t_ep = Instant::now();
        let tl   = hsa.train_epoch(&epoch_pairs, BATCH);
        let vl   = val_loss(&hsa, &val_pairs);
        let dur  = t_ep.elapsed();

        println!(
            "{:>5}  {:>10.4}  {:>10.4}  {:>7.2?}",
            ep + 1, tl, vl, dur
        );
    }

    let train_elapsed = t_train.elapsed();
    println!("\nTreino total: {:.1?}\n", train_elapsed);

    // ---- Métricas finais de validação ----
    let final_vl = val_loss(&hsa, &val_pairs);
    let bits_per_byte = final_vl / std::f32::consts::LN_2;
    println!("Val final:  {:.4} nats  ({:.3} bits/byte)\n", final_vl, bits_per_byte);

    // ---- Geração autoregressiva ----
    println!("━━━ Geração Autoregressiva ━━━");
    println!("  temperature={TEMPERATURE}  top_k={TOP_K}  gen_len={GEN_LEN}\n");

    let mut gen_rng = SEED ^ 0xdeadbeef_cafebabe;

    for (idx, &prompt) in PROMPTS.iter().enumerate() {
        let result = generate(&hsa, prompt, GEN_LEN, TEMPERATURE, TOP_K, &mut gen_rng);

        println!("── Prompt {} ──", idx + 1);
        println!("  Prompt   : {:?}", &prompt[..prompt.len().min(48)]);
        println!("  Métricas : perplexidade={:.2}  diversidade={}/256  entropia={:.3} bits",
            result.perplexity, result.diversity, result.entropy);
        println!("  Gerado   :");
        println!("  ┌─────────────────────────────────────────────────────────────┐");

        // Imprime em linhas de ~65 chars, sanitizando bytes não-ASCII como \xNN
        let mut col = 0;
        print!("  │ ");
        for &b in &result.text {
            let ch: String = if b.is_ascii_graphic() || b == b' ' || b == b'\n' || b == b'\t' {
                if b == b'\n' {
                    format!("↵")
                } else if b == b'\t' {
                    format!("→")
                } else {
                    (b as char).to_string()
                }
            } else {
                format!("·")
            };
            let ch_len = ch.chars().count();
            if col + ch_len >= 63 {
                println!(" │");
                print!("  │ ");
                col = 0;
            }
            print!("{}", ch);
            col += ch_len;
        }
        if col > 0 {
            println!(" │");
        }
        println!("  └─────────────────────────────────────────────────────────────┘");
        println!();
    }

    // ---- Análise de qualidade ----
    println!("━━━ Análise de Qualidade da Geração ━━━");

    // Gera samples extras para calcular estatísticas agregadas
    let n_samples = 5;
    let mut all_perp  = Vec::with_capacity(n_samples * PROMPTS.len());
    let mut all_div   = Vec::with_capacity(n_samples * PROMPTS.len());
    let mut all_entr  = Vec::with_capacity(n_samples * PROMPTS.len());

    for _ in 0..n_samples {
        for &prompt in PROMPTS {
            let r = generate(&hsa, prompt, GEN_LEN, TEMPERATURE, TOP_K, &mut gen_rng);
            all_perp.push(r.perplexity);
            all_div.push(r.diversity as f32);
            all_entr.push(r.entropy);
        }
    }

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let std  = |v: &[f32]| {
        let m = mean(v);
        (v.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
    };

    println!(
        "  Perplexidade interna : {:.2} ± {:.2}",
        mean(&all_perp), std(&all_perp)
    );
    println!(
        "  Diversidade (bytes/256): {:.1} ± {:.1}",
        mean(&all_div), std(&all_div)
    );
    println!(
        "  Entropia empírica    : {:.3} ± {:.3} bits",
        mean(&all_entr), std(&all_entr)
    );
    println!("  Validação loss       : {:.4} nats = {:.3} bits/byte", final_vl, bits_per_byte);

    // Nota de interpretação
    let max_entropy = (VOCAB as f32).log2();
    let norm_entropy = mean(&all_entr) / max_entropy * 100.0;
    println!("\n  Entropia normalizada : {:.1}% do máximo ({:.3} bits = uniforme)",
        norm_entropy, max_entropy);
    if norm_entropy < 30.0 {
        println!("  → Geração muito concentrada; considere temperatura mais alta.");
    } else if norm_entropy > 80.0 {
        println!("  → Geração muito aleatória; considere temperatura mais baixa ou mais treino.");
    } else {
        println!("  → Diversidade razoável para modelo byte-level com {} parâmetros.", hsa.n_params());
    }

    println!("\n=== Concluído ===");
    Ok(())
}
