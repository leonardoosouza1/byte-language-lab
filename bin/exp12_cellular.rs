//! Experimento 12 — Cellular automata como substrato de compute (Wolfram-inspired)
//!
//! Tese (Wolfram 2002, NKS): regras simples sobre estados discretos geram
//! complexidade arbitrária. Rule 110 é Turing-completa.
//!
//! Implementa CA elementar 1D sobre array de bytes (cada byte = 8 células).
//! Mede:
//! - Classes comportamentais (estável, periódico, complexo, caótico)
//! - Entropia da evolução
//! - Detectabilidade de patterns

use blab::Rng;
use std::io::Write;

const N_CELLS: usize = 256; // bits = células
const N_STEPS: usize = 256;
const N_RULES: usize = 8; // amostragem
const SEED: u64 = 9001;

/// Lê bit i do array de bytes.
fn get_bit(cells: &[u8], i: usize) -> u8 {
    (cells[i / 8] >> (i % 8)) & 1
}

/// Escreve bit i do array.
fn set_bit(cells: &mut [u8], i: usize, v: u8) {
    let mask = 1 << (i % 8);
    if v == 0 {
        cells[i / 8] &= !mask;
    } else {
        cells[i / 8] |= mask;
    }
}

/// Aplica 1 step de CA elementar com `rule` (8-bit lookup table).
fn ca_step(cells: &[u8], rule: u8, n_cells: usize) -> Vec<u8> {
    let mut next = vec![0_u8; cells.len()];
    for i in 0..n_cells {
        let l = get_bit(cells, (i + n_cells - 1) % n_cells);
        let c = get_bit(cells, i);
        let r = get_bit(cells, (i + 1) % n_cells);
        let pattern = (l << 2) | (c << 1) | r;
        let new_v = (rule >> pattern) & 1;
        set_bit(&mut next, i, new_v);
    }
    next
}

/// Conta total de 1s em uma generation.
fn density(cells: &[u8], n_cells: usize) -> usize {
    (0..n_cells).filter(|&i| get_bit(cells, i) == 1).count()
}

/// Entropia de Shannon da sequência de densidades.
fn evolution_entropy(densities: &[usize]) -> f64 {
    use std::collections::HashMap;
    let mut counts: HashMap<usize, u32> = HashMap::new();
    for &d in densities {
        *counts.entry(d).or_insert(0) += 1;
    }
    let total = densities.len() as f64;
    let mut h = 0.0;
    for &c in counts.values() {
        let p = f64::from(c) / total;
        h -= p * p.log2();
    }
    h
}

/// Detecta period (loop) na sequência de bytes.
fn detect_period(generations: &[Vec<u8>]) -> Option<usize> {
    let n = generations.len();
    for period in 1..=(n / 2) {
        let mut is_periodic = true;
        for i in (n - period)..n {
            if generations[i] != generations[i - period] {
                is_periodic = false;
                break;
            }
        }
        if is_periodic {
            return Some(period);
        }
    }
    None
}

/// Classifica regra (Wolfram class):
/// I  - estável (entropy ~0, density constante ou zero)
/// II - periódico (entropy moderada, period detectado)
/// III - caótico (entropy alta, sem period)
/// IV - complexo (intermediário; difícil distinguir empiricamente)
fn classify_rule(densities: &[usize], generations: &[Vec<u8>]) -> &'static str {
    let h = evolution_entropy(densities);
    let period = detect_period(generations);

    // Class I: maioria estável
    let last_density = densities.last().copied().unwrap_or(0);
    if h < 0.5 || last_density == 0 || last_density == N_CELLS {
        return "I (stable)";
    }
    // Class II: periódico claro
    if let Some(p) = period {
        if p < 16 {
            return "II (periodic)";
        }
    }
    // Class III/IV
    if h > 4.0 {
        "III (chaotic)"
    } else {
        "IV (complex)"
    }
}

fn main() -> std::io::Result<()> {
    println!("[exp12] Running cellular automata over {} bits, {} steps...", N_CELLS, N_STEPS);

    // Regras icônicas
    let rules: [(u8, &str); 8] = [
        (30, "Rule 30 (chaotic)"),
        (90, "Rule 90 (Sierpinski)"),
        (110, "Rule 110 (Turing-complete)"),
        (54, "Rule 54 (complex)"),
        (60, "Rule 60 (XOR)"),
        (184, "Rule 184 (traffic flow)"),
        (250, "Rule 250 (mostly 1s)"),
        (0, "Rule 0 (all zero)"),
    ];

    let mut out = std::fs::File::create("results/exp12_cellular.md")?;
    writeln!(out, "# Experimento 12 — Cellular automata como compute substrate\n")?;
    writeln!(out, "**Tese (Wolfram NKS):** estados discretos + regras simples = computação universal.")?;
    writeln!(out, "Rule 110 é provadamente Turing-completa. CA é o substrato MAIS simples possível para")?;
    writeln!(out, "computação arbitrária.\n")?;
    writeln!(out, "**Setup:**")?;
    writeln!(out, "- {N_CELLS} cells (= {} bytes), {N_STEPS} generations", N_CELLS / 8)?;
    writeln!(out, "- Estado inicial: bit central ativo")?;
    writeln!(out, "- {N_RULES} regras testadas (incluindo Rule 110, 30, 90)\n")?;

    writeln!(out, "## Classificação Wolfram empírica\n")?;
    writeln!(out, "| Rule | Density end | Density entropy | Period | Class |")?;
    writeln!(out, "|-----:|------------:|----------------:|:------:|:------|")?;

    for &(rule, name) in &rules {
        let mut cells = vec![0_u8; N_CELLS / 8];
        set_bit(&mut cells, N_CELLS / 2, 1); // bit central
        let mut generations = vec![cells.clone()];
        let mut densities = vec![density(&cells, N_CELLS)];

        for _ in 0..N_STEPS {
            cells = ca_step(&cells, rule, N_CELLS);
            generations.push(cells.clone());
            densities.push(density(&cells, N_CELLS));
        }

        let h = evolution_entropy(&densities);
        let period = detect_period(&generations);
        let class = classify_rule(&densities, &generations);
        let last_d = *densities.last().unwrap();

        writeln!(out, "| {} ({}) | {} | {:.2} | {} | {} |",
            rule, name, last_d, h,
            period.map_or("none".to_string(), |p| p.to_string()),
            class
        )?;
    }

    writeln!(out, "\n## Validação Rule 110 (mais importante)\n")?;
    writeln!(out, "Rule 110 é provadamente Turing-completa (Cook 2004). Aqui rodamos {N_STEPS}")?;
    writeln!(out, "generations em {N_CELLS} cells. O comportamento esperado é Class IV (complexo):")?;
    writeln!(out, "estruturas localizadas (\"gliders\") interagindo de forma não-trivial.\n")?;

    // Rule 110 sample dump (primeiras gens)
    let mut cells = vec![0_u8; N_CELLS / 8];
    set_bit(&mut cells, N_CELLS / 2, 1);
    writeln!(out, "**Primeiras 20 gens (subset central, ASCII):**")?;
    writeln!(out, "```")?;
    for _ in 0..20 {
        let s: String = (N_CELLS / 2 - 30..N_CELLS / 2 + 30)
            .map(|i| if get_bit(&cells, i) == 1 { '#' } else { '.' })
            .collect();
        writeln!(out, "{}", s)?;
        cells = ca_step(&cells, 110, N_CELLS);
    }
    writeln!(out, "```\n")?;

    writeln!(out, "## Implicação\n")?;
    writeln!(out, "- **Compute substrate viável:** Rule 110 mostra que cells de 1 BIT (cada um!) podem")?;
    writeln!(out, "  implementar qualquer computação dado tempo suficiente. Bytes têm 256× mais estados.")?;
    writeln!(out, "- **Conexão com fsm-cognitive-architecture:** docs/01-conceitos-fundamentais.md cita")?;
    writeln!(out, "  Wolfram NKS como justificação filosófica para FSM 8-bit.")?;
    writeln!(out, "- **Implementação trivial:** 1 byte por 8 cells, lookup table de 8 entradas (256 regras).")?;
    writeln!(out, "  Roda em qualquer hardware integer.\n")?;
    writeln!(out, "**Pós-LLM hipotético:** rede de FSMs de 8 bits onde regras de transição (LUT 4096)")?;
    writeln!(out, "são análogas a regras CA. Computação emerge da DINÂMICA, não dos pesos.")?;

    println!("[exp12] Done.");
    Ok(())
}
