//! train_nibble_hsa — Ablação: baseline (só i_f) vs nibble (i_f + nibble_h + nibble_l).
//!
//! Valida quantitativamente se a geometria de nibble ajuda na convergência byte-level.
//!
//! Os dois modelos são treinados SEQUENCIALMENTE (não em simultâneo) para evitar
//! ter dois HsaContext/queues HSA ativos ao mesmo tempo — ROCm não suporta bem
//! múltiplos contextos simultâneos no mesmo processo em gfx1031.
//!
//! Fase 1: Treino BASELINE (nibble_h/l = 0, frozen) → recolhe loss_val por época.
//! Fase 2: Treino NIBBLE   (nibble_h/l activos)     → recolhe loss_val por época.
//! Saída: tabela combinada + teste de case generalization.
//!
//! Mesmos pesos iniciais, mesma sequência de shuffles (mesma semente) → fair ablation.
//!
//! Expectativa teórica:
//!   - nibble converge mais rápido (inductive bias ASCII grátis)
//!   - generalização case-invariante emergente (A ↔ a partilham nibble_l[1])
//!
//! ```bash
//! cargo run --release --features gpu-aql --bin train_nibble_hsa
//! cargo run --release --features gpu-aql --bin train_nibble_hsa -- /path/to/corpus.txt
//! ```

#![cfg(feature = "gpu-aql")]
#![allow(unsafe_code)]

use std::io::Read;
use std::time::Instant;

use blab::{
    gpu_aql::SmgGridHsa,
    learn::microcolumn_v3::{GridConfig, SmgGrid},
};

// ---- Hiperparâmetros (iguais para os dois modelos) ----
const WINDOW:          usize = 48;
const VOCAB:           usize = 256;
const LR:              f32   = 0.002;
const LR_NIBBLE_BOOST: f32   = LR * 1.5;  // boost inicial para nibble aprender mais rápido
const LR_BOOST_EPOCHS: usize = 3;         // épocas com LR boosted
const BATCH:           usize = 64;
const N_EPOCHS:        usize = 30;
const STEPS_PER_EPOCH: usize = 150;
const VAL_STEPS:       usize = 40;
const SEED:            u64   = 42;

// ---- Probabilidade softmax para um índice dado logits ----
fn softmax_prob(logits: &[f32], idx: usize) -> f32 {
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - mx).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps[idx] / s
}

// ---- Cross-entropy de um batch ----
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

fn shuffle<T>(v: &mut Vec<T>, seed: u64) {
    let mut rng = seed;
    for i in (1..v.len()).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng >> 33) as usize % (i + 1);
        v.swap(i, j);
    }
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

// ---- Validação: loss médio sobre val_steps×batch pares ----
fn val_loss(hsa: &blab::gpu_aql::SmgGridHsa, val_pairs: &[(Vec<u32>, u32)]) -> f32 {
    let val_slice: Vec<_> = val_pairs.iter().take(VAL_STEPS * BATCH).cloned().collect();
    let mut total = 0.0f32;
    let mut nb    = 0usize;
    for chunk in val_slice.chunks(BATCH) {
        if chunk.is_empty() { continue; }
        let windows: Vec<Vec<u32>> = chunk.iter().map(|(w, _)| w.clone()).collect();
        let targets: Vec<u32>      = chunk.iter().map(|(_, t)| *t).collect();
        total += batch_loss(&hsa.forward_batch(&windows), &targets);
        nb    += 1;
    }
    if nb > 0 { total / nb as f32 } else { f32::NAN }
}

// ---- Tipo de resultado por run ----
struct RunResult {
    val_losses:  Vec<f32>,              // por época (len = N_EPOCHS)
    train_losses: Vec<f32>,             // por época
    /// (upper_p_R, upper_p_r, lower_p_r, lower_p_R) — case generalization
    case_gen: (f32, f32, f32, f32),
}

// ---- Função de treino genérica (cria e dropa SmgGridHsa internamente) ----
//
// Garante que só existe UM HsaContext durante o treino.
// A string `label` é usada para progresso no stderr.
fn train_run(
    init_weights: &SmgGrid,
    train_pairs:  &mut Vec<(Vec<u32>, u32)>,
    val_pairs:    &[(Vec<u32>, u32)],
    lr_schedule:  &dyn Fn(usize) -> f32,   // epoch → LR
    label:        &str,
) -> Result<RunResult, Box<dyn std::error::Error>> {
    // Sequências para case generalization (fixas, fora do loop de treino).
    let test_upper: Vec<u32> = b"RUST LANGUAGE MODELLING BYTE LEVEL SYSTEM DESIGN APPROACH GEOMETRIC"
        [..WINDOW].iter().map(|&b| b as u32).collect();
    let test_lower: Vec<u32> = b"rust language modelling byte level system design approach geometric"
        [..WINDOW].iter().map(|&b| b as u32).collect();

    let mut val_losses   = Vec::with_capacity(N_EPOCHS);
    let mut train_losses = Vec::with_capacity(N_EPOCHS);

    // Escopo restrito: hsa é criado aqui e dropped no fim deste bloco.
    // Enquanto este bloco executa NÃO existe nenhum outro HsaContext.
    {
        let mut hsa = SmgGridHsa::from_cpu(init_weights, BATCH)?;
        hsa.set_lr(lr_schedule(0));
        println!("  {} params", hsa.n_params());

        let mut rng_seed = SEED;
        let t0_run = Instant::now();

        for ep in 0..N_EPOCHS {
            hsa.set_lr(lr_schedule(ep));

            rng_seed = rng_seed.wrapping_add(ep as u64 * 0x9e3779b97f4a7c15);
            shuffle(train_pairs, rng_seed);

            let epoch_pairs: Vec<_> = train_pairs
                .iter()
                .take(STEPS_PER_EPOCH * BATCH)
                .cloned()
                .collect();

            let t_ep = Instant::now();
            let tl   = hsa.train_epoch(&epoch_pairs, BATCH);
            let vl   = val_loss(&hsa, val_pairs);
            let dur  = t_ep.elapsed();

            train_losses.push(tl);
            val_losses.push(vl);

            // Progresso em tempo real para stderr.
            eprintln!(
                "  [{label}] ep {:2}/{N_EPOCHS}  lr={:.4}  train={:.4}  val={:.4}  {:.2?}",
                ep + 1, lr_schedule(ep), tl, vl, dur
            );
        }

        println!("  [{label}] total {:.1?}", t0_run.elapsed());

        // Case generalization (forward-only, enquanto o modelo ainda está vivo).
        let log_u = &hsa.forward_batch(&[test_upper])[0];
        let log_l = &hsa.forward_batch(&[test_lower])[0];
        let case_gen = (
            softmax_prob(log_u, b'R' as usize),
            softmax_prob(log_u, b'r' as usize),
            softmax_prob(log_l, b'r' as usize),
            softmax_prob(log_l, b'R' as usize),
        );

        Ok(RunResult { val_losses, train_losses, case_gen })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== train_nibble_hsa: ablação nibble embedding em RDNA2 ===");
    println!("  window={WINDOW}  vocab={VOCAB}  batch={BATCH}  lr_base={LR:.4}");
    println!("  {STEPS_PER_EPOCH} steps/epoch × {N_EPOCHS} epochs");
    println!("  Treino SEQUENCIAL: baseline → nibble (evita dois HsaContext simultâneos)\n");

    // ---- Corpus ----
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
    println!("  {} bytes ({:.1} KB)", data.len(), data.len() as f64 / 1024.0);

    let mut all_pairs = make_pairs(&data);
    let n_val   = (all_pairs.len() / 10).max(VAL_STEPS * BATCH);
    let n_train = all_pairs.len() - n_val;
    let val_pairs: Vec<_> = all_pairs.split_off(n_train);
    let mut train_pairs   = all_pairs;
    println!(
        "  Pares: {} treino | {} validação\n",
        train_pairs.len(), val_pairs.len()
    );

    // ---- Pesos iniciais (CPU) — partilhados pelos dois runs ----
    let cfg = GridConfig {
        vocab_size:    VOCAB,
        window:        WINDOW,
        learning_rate: LR,
        seed:          SEED,
    };
    let init_weights = SmgGrid::new(cfg);

    // ---- Fase 1: BASELINE ----
    // nibble_h/l começam a zero; SGD actualiza-os, mas o sinal de gradient é fraco.
    // (Idêntico ao nibble mas com LR normal desde o início.)
    println!("━━━ Fase 1/2: BASELINE ━━━");
    let t_total = Instant::now();

    let mut train_pairs_base = train_pairs.clone();   // clone para reproducibilidade
    let base_result = train_run(
        &init_weights,
        &mut train_pairs_base,
        &val_pairs,
        &|_ep| LR,   // LR constante
        "BASE",
    )?;

    println!();

    // ---- Fase 2: NIBBLE ----
    // nibble_h/l com LR boosted nos primeiros {LR_BOOST_EPOCHS} épocas.
    println!("━━━ Fase 2/2: NIBBLE ━━━");
    let nibble_result = train_run(
        &init_weights,
        &mut train_pairs,
        &val_pairs,
        &|ep| if ep < LR_BOOST_EPOCHS { LR_NIBBLE_BOOST } else { LR },
        "NIBBLE",
    )?;

    let total_elapsed = t_total.elapsed();
    println!("\nTotal combinado: {:.1?}\n", total_elapsed);

    // ---- Tabela combinada ----
    println!(
        "{:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "época", "vl_base", "tl_base", "vl_nibble", "tl_nibble", "Δval"
    );
    println!("{}", "─".repeat(62));
    for ep in 0..N_EPOCHS {
        let vb = base_result.val_losses[ep];
        let tb = base_result.train_losses[ep];
        let vn = nibble_result.val_losses[ep];
        let tn = nibble_result.train_losses[ep];
        let dv = vb - vn;   // positivo = nibble melhor na val
        println!(
            "{:>5}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}  {:>+10.4}",
            ep + 1, vb, tb, vn, tn, dv
        );
    }

    // ---- Sumário ----
    let base_final  = base_result.val_losses.last().copied().unwrap_or(f32::NAN);
    let nibble_final = nibble_result.val_losses.last().copied().unwrap_or(f32::NAN);
    println!("\nVal final: baseline={:.4} nats ({:.3} bits)  nibble={:.4} nats ({:.3} bits)",
        base_final,   base_final   / std::f32::consts::LN_2,
        nibble_final, nibble_final / std::f32::consts::LN_2,
    );
    println!("Delta val final: {:+.4} nats  ({} nibble)",
        base_final - nibble_final,
        if nibble_final < base_final { "MELHOR" } else { "PIOR" }
    );

    // ---- Case generalization ----
    println!("\n=== Teste de case generalization (ao fim do treino) ===");
    println!("Sequência: 'RUST LANGUAGE...' (all-caps) → predicto 'R' vs 'r'\n");

    let (b_up_R, b_up_r, b_lo_r, b_lo_R) = base_result.case_gen;
    let (n_up_R, n_up_r, n_lo_r, n_lo_R) = nibble_result.case_gen;

    println!("  Contexto MAIÚSCULAS → P('R') vs P('r'):");
    println!("    baseline : P('R')={b_up_R:.4}  P('r')={b_up_r:.4}  ratio={:.2}",
        b_up_R / b_up_r.max(1e-9));
    println!("    nibble   : P('R')={n_up_R:.4}  P('r')={n_up_r:.4}  ratio={:.2}",
        n_up_R / n_up_r.max(1e-9));

    println!("\n  Contexto minúsculas → P('r') vs P('R'):");
    println!("    baseline : P('r')={b_lo_r:.4}  P('R')={b_lo_R:.4}  ratio={:.2}",
        b_lo_r / b_lo_R.max(1e-9));
    println!("    nibble   : P('r')={n_lo_r:.4}  P('R')={n_lo_R:.4}  ratio={:.2}",
        n_lo_r / n_lo_R.max(1e-9));

    println!("\n[nibble_l[1] partilhado entre 'R'(0x52) e 'r'(0x72) → inductive bias case grátis]");

    Ok(())
}
