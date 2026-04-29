//! Experimento 03 — Learnability: byte-positional vs ASCII-string
//!
//! Hipótese: streams de byte estruturado têm distribuição mais previsível
//! (lower perplexity em modelo Markov n-gram) que strings ASCII para tarefa
//! de arithmetic completion.

use blab::learn::{encode_ascii, encode_byte_positional, evaluate_encoding, generate_dataset};
use std::io::Write;

const N_PROBLEMS: usize = 5000;
const SEED: u64 = 9001;
const TRAIN_PCT: f64 = 0.8;

fn main() -> std::io::Result<()> {
    println!("[exp03] Generating {N_PROBLEMS} arithmetic problems (seed {SEED})...");
    let problems = generate_dataset(N_PROBLEMS, SEED);

    println!("[exp03] Encoding (ASCII)...");
    let ascii_seqs: Vec<Vec<u8>> = problems.iter().map(encode_ascii).collect();
    let r_ascii = evaluate_encoding("ASCII", ascii_seqs.clone(), TRAIN_PCT);

    println!("[exp03] Encoding (byte-positional / Path B)...");
    let bp_seqs: Vec<Vec<u8>> = problems.iter().map(encode_byte_positional).collect();
    let r_bp = evaluate_encoding("Byte-positional (Path B)", bp_seqs.clone(), TRAIN_PCT);

    println!("[exp03] Computing comparison metrics...");
    let total_ascii: usize = ascii_seqs.iter().map(Vec::len).sum();
    let total_bp: usize = bp_seqs.iter().map(Vec::len).sum();

    // Métrica adicional: entropia de Shannon empírica do stream completo
    let ascii_concat: Vec<u8> = ascii_seqs.iter().flatten().copied().collect();
    let bp_concat: Vec<u8> = bp_seqs.iter().flatten().copied().collect();

    let ascii_entropy = shannon_entropy(&ascii_concat);
    let bp_entropy = shannon_entropy(&bp_concat);

    let mut out = std::fs::File::create("results/exp03_learn.md")?;
    writeln!(out, "# Experimento 03 — Learnability: byte-positional vs ASCII-string\n")?;
    writeln!(out, "**N problemas:** {N_PROBLEMS} (operações + - × com inteiros 0-99)  ")?;
    writeln!(out, "**Seed:** {SEED}  ")?;
    writeln!(out, "**Train/test split:** {:.0}% / {:.0}%\n", TRAIN_PCT * 100.0, (1.0 - TRAIN_PCT) * 100.0)?;

    writeln!(out, "## Resultados\n")?;
    writeln!(out, "| Métrica | ASCII | Byte-positional | Razão (BP/ASCII) |")?;
    writeln!(out, "|---------|------:|----------------:|-----------------:|")?;
    writeln!(out, "| Total bytes | {} | {} | {:.2}× |", total_ascii, total_bp, total_bp as f64 / total_ascii as f64)?;
    writeln!(out, "| Bytes/problema (médio) | {:.2} | {:.2} | {:.2}× |",
        r_ascii.mean_bytes_per_sample, r_bp.mean_bytes_per_sample,
        r_bp.mean_bytes_per_sample / r_ascii.mean_bytes_per_sample)?;
    writeln!(out, "| Vocab size (símbolos distintos) | {} | {} | — |", r_ascii.vocab_size, r_bp.vocab_size)?;
    writeln!(out, "| Shannon entropy (bits/byte) | {:.3} | {:.3} | {:.2}× |",
        ascii_entropy, bp_entropy, bp_entropy / ascii_entropy)?;
    writeln!(out, "| Perplexity 2-gram | {:.2} | {:.2} | {:.2}× |",
        r_ascii.ngram_perplexity_n2, r_bp.ngram_perplexity_n2,
        r_bp.ngram_perplexity_n2 / r_ascii.ngram_perplexity_n2)?;
    writeln!(out, "| Perplexity 3-gram | {:.2} | {:.2} | {:.2}× |",
        r_ascii.ngram_perplexity_n3, r_bp.ngram_perplexity_n3,
        r_bp.ngram_perplexity_n3 / r_ascii.ngram_perplexity_n3)?;
    writeln!(out, "| Perplexity 4-gram | {:.2} | {:.2} | {:.2}× |",
        r_ascii.ngram_perplexity_n4, r_bp.ngram_perplexity_n4,
        r_bp.ngram_perplexity_n4 / r_ascii.ngram_perplexity_n4)?;

    writeln!(out, "\n## Interpretação\n")?;
    writeln!(out, "- **Perplexity menor = mais previsível para um modelo n-gram.**")?;
    writeln!(out, "- **Shannon entropy menor = stream mais redundante (mais aprendível).**\n")?;

    let bp_better_n3 = r_bp.ngram_perplexity_n3 < r_ascii.ngram_perplexity_n3;
    let bp_better_n4 = r_bp.ngram_perplexity_n4 < r_ascii.ngram_perplexity_n4;
    let bp_compact = total_bp < total_ascii;

    writeln!(out, "**Veredito empírico:**")?;
    writeln!(out, "- Compactness: {} ({} bytes vs {} bytes)",
        if bp_compact { "**byte-positional vence**" } else { "ASCII vence" },
        total_bp, total_ascii)?;
    writeln!(out, "- Perplexity 3-gram: {} ({:.2} vs {:.2})",
        if bp_better_n3 { "**byte-positional vence**" } else { "ASCII vence" },
        r_bp.ngram_perplexity_n3, r_ascii.ngram_perplexity_n3)?;
    writeln!(out, "- Perplexity 4-gram: {} ({:.2} vs {:.2})",
        if bp_better_n4 { "**byte-positional vence**" } else { "ASCII vence" },
        r_bp.ngram_perplexity_n4, r_ascii.ngram_perplexity_n4)?;

    writeln!(out, "\n## Observações importantes\n")?;
    writeln!(out, "- Esta tarefa (arithmetic completion com inteiros 0-99) tem vocab restrito; ASCII tem ~14 chars distintos, byte-positional ~16. Diferenças sutis em perplexity refletem ordem das sequências (mais que vocab).")?;
    writeln!(out, "- O ganho REAL de byte-positional aparece em tarefas onde POSIÇÃO do dígito importa para o cálculo (ex: aritmética com fracionais). N-gram não captura isso plenamente.")?;
    writeln!(out, "- Para validação completa da hipótese 'inteligência fala bits', precisamos modelo neural que possa explorar a estrutura posicional — fora do escopo desta lab inicial.")?;

    println!("[exp03] Done.");
    Ok(())
}

fn shannon_entropy(bytes: &[u8]) -> f64 {
    let mut counts = [0u32; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }
    let total = bytes.len() as f64;
    let mut h = 0.0_f64;
    for &c in &counts {
        if c > 0 {
            let p = f64::from(c) / total;
            h -= p * p.log2();
        }
    }
    h
}
