//! byte_harmony.rs — analisa harmonia bit entre pares de bytes ASCII.
//!
//! Para cada par (a,b) ∈ {A..Z, a..z, 0..9}², computa:
//!   - Hamming distance
//!   - Shared bits per role (7=struct, 6=struct, 5=case, 4=group, 3..0=pos)
//!   - É "harmônico"? (Hamming ≤ 2 = vizinhos próximos)
//!
//! Output: matriz colorida + DOT graph + JSON sumário.

use std::fs::File;
use std::io::Write;

const BIT_ROLE: [&str; 8] = [
    "LSB",       // 0
    "pos1",      // 1
    "pos2",      // 2
    "nibble_hi", // 3
    "group",     // 4
    "case",      // 5
    "struct",    // 6
    "ascii/MSB", // 7
];

fn hamming(a: u8, b: u8) -> u32 { (a ^ b).count_ones() }

fn bit_diff_at(a: u8, b: u8, pos: u8) -> bool {
    (a >> pos) & 1 != (b >> pos) & 1
}

fn print_char(b: u8) -> String {
    if b.is_ascii_graphic() { format!("{}", b as char) }
    else { format!("\\x{:02X}", b) }
}

fn main() -> std::io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   BYTE HARMONY ANALYSIS — geometria bit dos caracteres ASCII         ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let chars: Vec<u8> = (b'A'..=b'Z').chain(b'a'..=b'z').chain(b'0'..=b'9').collect();
    let n = chars.len();
    println!("  {} bytes analisados (A-Z, a-z, 0-9)\n", n);

    // ── ASCII harmony matrix ──────────────────────────────────────────
    println!("▸ Matriz Hamming letras→letras (cor: ░ 0-2 verde, ▒ 3-4 amarelo, ▓ 5-8 vermelho):\n");
    print!("       ");
    for c in &chars {
        print!("{} ", *c as char);
    }
    println!();
    for a in &chars {
        print!("    {} ", *a as char);
        for b in &chars {
            let h = hamming(*a, *b);
            let cell = match h {
                0 => "\x1b[100m·\x1b[0m",   // self
                1..=2 => "\x1b[42m█\x1b[0m", // green: HARMONIA forte
                3..=4 => "\x1b[43m▓\x1b[0m", // yellow: meio
                _ => "\x1b[41m░\x1b[0m",    // red: longe
            };
            print!("{} ", cell);
        }
        println!();
    }

    // ── Análise por role ──────────────────────────────────────────────
    println!("\n▸ Distribuição de bit-role differences nos pares ALFA (A-Z + a-z):\n");
    let alpha: Vec<u8> = (b'A'..=b'Z').chain(b'a'..=b'z').collect();
    let mut diff_count = [0u32; 8];
    let mut total_pairs = 0u32;
    for (i, a) in alpha.iter().enumerate() {
        for b in &alpha[i+1..] {
            for pos in 0..8 {
                if bit_diff_at(*a, *b, pos) {
                    diff_count[pos as usize] += 1;
                }
            }
            total_pairs += 1;
        }
    }
    println!("  {:>10}  {:>6}  {:>6}  bit role", "bit", "diffs", "%");
    println!("  {}", "─".repeat(40));
    for pos in (0..8).rev() {
        let pct = 100.0 * diff_count[pos] as f32 / total_pairs as f32;
        let bar = "█".repeat((pct / 5.0) as usize);
        println!("  bit{:>2}      {:>6}  {:>5.1}%  {}  ← {}",
            pos, diff_count[pos], pct, bar, BIT_ROLE[pos]);
    }
    println!("  total pares: {}", total_pairs);

    // ── Top pares harmônicos (Hamming = 1) ────────────────────────────
    println!("\n▸ Pares com HARMONIA MÁXIMA (Hamming = 1, só 1 bit difere):\n");
    let mut h1_pairs = Vec::new();
    for (i, a) in chars.iter().enumerate() {
        for b in &chars[i+1..] {
            if hamming(*a, *b) == 1 {
                let xor = a ^ b;
                let diff_bit = xor.trailing_zeros() as u8;
                h1_pairs.push((*a, *b, diff_bit));
            }
        }
    }
    println!("  {:>6}  {:>6}  bit_diff  role", "A", "B");
    println!("  {}", "─".repeat(40));
    for (a, b, bit) in &h1_pairs {
        println!("  '{}' <-> '{}'   bit{:>2}    {}",
            print_char(*a), print_char(*b), bit, BIT_ROLE[*bit as usize]);
    }
    println!("  Total: {} pares Hamming=1", h1_pairs.len());

    // ── Bit-2 pairs (geometricamente próximos mas não idênticos) ─────
    println!("\n▸ Pares Hamming=2 (vizinhos próximos):\n");
    let mut h2_count = 0;
    let mut h2_examples = Vec::new();
    for (i, a) in chars.iter().enumerate() {
        for b in &chars[i+1..] {
            if hamming(*a, *b) == 2 {
                h2_count += 1;
                if h2_examples.len() < 30 {
                    h2_examples.push((*a, *b));
                }
            }
        }
    }
    println!("  Total Hamming=2: {} pares (showing {} primeiros):", h2_count, h2_examples.len());
    for chunk in h2_examples.chunks(5) {
        for (a, b) in chunk {
            print!("  '{}' <-> '{}'  ", print_char(*a), print_char(*b));
        }
        println!();
    }

    // ── Export DOT graph ──────────────────────────────────────────────
    println!("\n▸ Exportando grafo bit-harmonia → byte_harmony.dot");
    let mut f = File::create("/tmp/byte_harmony.dot")?;
    writeln!(f, "graph ByteHarmony {{")?;
    writeln!(f, "  layout=fdp; overlap=false; node [shape=circle, style=filled];")?;
    writeln!(f, "  // Nós: letras alpha")?;
    let alpha: Vec<u8> = (b'A'..=b'Z').chain(b'a'..=b'z').collect();
    for &c in &alpha {
        let color = if c.is_ascii_uppercase() { "lightblue" } else { "lightgreen" };
        writeln!(f, "  \"{}\" [fillcolor={}];", c as char, color)?;
    }
    writeln!(f, "  // Arestas: Hamming ≤ 2")?;
    for (i, a) in alpha.iter().enumerate() {
        for b in &alpha[i+1..] {
            let h = hamming(*a, *b);
            if h <= 2 {
                let pen = match h { 1 => 3, 2 => 1, _ => 0 };
                let col = match h { 1 => "green", _ => "gray70" };
                writeln!(f, "  \"{}\" -- \"{}\" [penwidth={}, color={}];",
                    *a as char, *b as char, pen, col)?;
            }
        }
    }
    writeln!(f, "}}")?;
    println!("  /tmp/byte_harmony.dot ({} nós, ~{} arestas Hamming≤2)", alpha.len(), h2_count + h1_pairs.iter().filter(|(a,b,_)| a.is_ascii_alphabetic() && b.is_ascii_alphabetic()).count());

    // ── Findings summary ──────────────────────────────────────────────
    println!("\n▸ SUMÁRIO DA TESE:\n");
    println!("  ✓ bit 5 (case): {}% das letras DIFEREM neste bit",
        100.0 * diff_count[5] as f32 / total_pairs as f32);
    println!("    → MAIS sensível (esperado: capa metade das letras alpha)");
    println!("  ✓ bit 7 (ASCII): {}% diferem (esperado: 0% — todos ASCII)",
        100.0 * diff_count[7] as f32 / total_pairs as f32);
    println!("  ✓ bit 6 (struct): {}% diferem (esperado: 0% — todos letras)",
        100.0 * diff_count[6] as f32 / total_pairs as f32);
    println!("  ✓ {} pares Hamming=1 (bit5=case flips!): {}",
        h1_pairs.iter().filter(|(_,_,b)| *b == 5).count(),
        h1_pairs.iter().filter(|(_,_,b)| *b == 5).take(5).map(|(a,b,_)| format!("{}↔{}", *a as char, *b as char)).collect::<Vec<_>>().join(" "));

    Ok(())
}
