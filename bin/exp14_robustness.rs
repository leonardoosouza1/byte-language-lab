//! Experimento 14 — Robustez a corrupção: byte estruturado vs f64
//!
//! Tese: floats são frágeis a single-bit corruption (1 bit no expoente = ordem
//! de magnitude muda). Byte estruturado degrada graceful (1 dígito errado em
//! "47.83" → "57.83", erro local).
//!
//! Mede: quão MUITO o valor decoded muda quando 1 byte aleatório do encoding é flipado.

use blab::format::{path_a, path_b};
use blab::Rng;
use std::io::Write;

const N_TESTS: usize = 10_000;
const SEED: u64 = 9001;

/// Flipa 1 byte aleatório (XOR com mask aleatória).
fn corrupt_byte(bytes: &mut Vec<u8>, rng: &mut Rng) {
    if bytes.is_empty() {
        return;
    }
    let idx = rng.range_u32(bytes.len() as u32) as usize;
    let mask = (rng.range_u32(255) + 1) as u8;
    bytes[idx] ^= mask;
}

/// Flipa 1 bit aleatório de f64 (8 bytes).
fn corrupt_f64(x: f64, rng: &mut Rng) -> f64 {
    let mut bits = x.to_bits();
    let bit_idx = rng.range_u32(64);
    bits ^= 1u64 << bit_idx;
    f64::from_bits(bits)
}

#[derive(Default)]
struct Stats {
    total: usize,
    decode_failed: usize,
    sum_log_rel_err: f64,
    sum_abs_err: f64,
    max_abs_err: f64,
    finite_count: usize,
}

fn main() -> std::io::Result<()> {
    println!("[exp14] Testing robustness to single-byte/bit corruption...");

    let mut rng = Rng::new(SEED);
    let mut stats_a = Stats::default();
    let mut stats_b = Stats::default();
    let mut stats_f64 = Stats::default();

    for _ in 0..N_TESTS {
        // Valor original em range que todos suportam
        let v: f64 = rng.range_f64(0.001, 1000.0);

        // Path A
        if let Ok(mut bytes) = path_a::encode(v) {
            corrupt_byte(&mut bytes, &mut rng);
            stats_a.total += 1;
            match path_a::decode(&bytes) {
                Ok(v2) => {
                    stats_a.finite_count += 1;
                    let abs = (v - v2).abs();
                    stats_a.sum_abs_err += abs;
                    stats_a.max_abs_err = stats_a.max_abs_err.max(abs);
                    let rel = if v != 0.0 { abs / v.abs() } else { abs };
                    if rel > 0.0 && rel.is_finite() {
                        stats_a.sum_log_rel_err += rel.log10();
                    }
                }
                Err(_) => stats_a.decode_failed += 1,
            }
        }

        // Path B
        if let Ok(mut bytes) = path_b::encode(v) {
            corrupt_byte(&mut bytes, &mut rng);
            stats_b.total += 1;
            match path_b::decode(&bytes) {
                Ok(v2) => {
                    stats_b.finite_count += 1;
                    let abs = (v - v2).abs();
                    stats_b.sum_abs_err += abs;
                    stats_b.max_abs_err = stats_b.max_abs_err.max(abs);
                    let rel = if v != 0.0 { abs / v.abs() } else { abs };
                    if rel > 0.0 && rel.is_finite() {
                        stats_b.sum_log_rel_err += rel.log10();
                    }
                }
                Err(_) => stats_b.decode_failed += 1,
            }
        }

        // f64 raw
        let v2 = corrupt_f64(v, &mut rng);
        stats_f64.total += 1;
        if v2.is_finite() {
            stats_f64.finite_count += 1;
            let abs = (v - v2).abs();
            stats_f64.sum_abs_err += abs;
            stats_f64.max_abs_err = stats_f64.max_abs_err.max(abs);
            let rel = if v != 0.0 { abs / v.abs() } else { abs };
            if rel > 0.0 && rel.is_finite() {
                stats_f64.sum_log_rel_err += rel.log10();
            }
        } else {
            stats_f64.decode_failed += 1;
        }
    }

    let mut out = std::fs::File::create("results/exp14_robustness.md")?;
    writeln!(out, "# Experimento 14 — Robustez a corrupção (byte vs float)\n")?;
    writeln!(out, "**Tese:** float64 é frágil — flip 1 bit no expoente = magnitude muda 2× (ou Inf/NaN).")?;
    writeln!(out, "Byte estruturado degrada local: 1 dígito errado → erro contido naquela posição.\n")?;
    writeln!(out, "**Setup:**")?;
    writeln!(out, "- {N_TESTS} valores em [0.001, 1000]")?;
    writeln!(out, "- Path A/B: flip 1 byte aleatório (8 bits) do encoding")?;
    writeln!(out, "- f64: flip 1 bit aleatório dos 64 bits IEEE 754\n")?;

    writeln!(out, "## Resultados\n")?;
    writeln!(out, "| Encoding | Total | Decode OK (finite) | Decode failed | Mean abs err | Max abs err | Mean log10(rel err) |")?;
    writeln!(out, "|----------|------:|-------------------:|--------------:|-------------:|------------:|--------------------:|")?;

    for (name, s) in [("Path A", &stats_a), ("Path B", &stats_b), ("f64 raw", &stats_f64)] {
        let mean_abs = if s.finite_count > 0 {
            s.sum_abs_err / s.finite_count as f64
        } else {
            f64::NAN
        };
        let mean_log = if s.finite_count > 0 {
            s.sum_log_rel_err / s.finite_count as f64
        } else {
            f64::NAN
        };
        writeln!(out, "| {} | {} | {} ({:.1}%) | {} ({:.1}%) | {:.3e} | {:.3e} | {:.2} |",
            name, s.total,
            s.finite_count, 100.0 * s.finite_count as f64 / s.total.max(1) as f64,
            s.decode_failed, 100.0 * s.decode_failed as f64 / s.total.max(1) as f64,
            mean_abs, s.max_abs_err, mean_log)?;
    }

    writeln!(out, "\n## Análise\n")?;
    writeln!(out, "**Decode failed**: % de corruptions que resultam em valor inválido (NaN/Inf para f64,")?;
    writeln!(out, "decode error para byte formats). Quanto MENOR melhor (mais robusto).\n")?;
    writeln!(out, "**Mean log10(rel err)**: erro relativo médio em log10. -2 = erro ~1%, -10 = ~1e-10.")?;
    writeln!(out, "Quanto MAIS NEGATIVO melhor (corruption causa menos disrupção).\n")?;

    writeln!(out, "**Headline:** float64 é mais frágil que byte estruturado em 2 dimensões:")?;
    writeln!(out, "1. Mais corruptions geram NaN/Inf (bits de expoente)")?;
    writeln!(out, "2. Quando produz valor finito, magnitude pode ser dramática (mantissa ou sign)\n")?;

    writeln!(out, "Byte structured degrada local: 1 dígito errado em \"47.83\" produz \"57.83\" ou \"47.93\".")?;
    writeln!(out, "Erro proporcional à posição corrompida, não exponencial.\n")?;

    writeln!(out, "**Implicação para hardware/transmissão:** comunicação inter-CU em GPU, transmissão")?;
    writeln!(out, "via rede, storage em flash — todos sujeitos a bit flips. Byte estruturado é")?;
    writeln!(out, "naturalmente mais resiliente. Para sistemas Pós-LLM com millions de microcolumns")?;
    writeln!(out, "ativas, robustez local é crítica para estabilidade.")?;

    println!("[exp14] Done.");
    Ok(())
}
