//! iara_hdc_analogy.rs — testa álgebra vetorial em HDC.
//!
//! Tese HDC: HV(rei) ⊕ HV(homem) ⊕ HV(mulher) ≈ HV(rainha)
//!
//! Para cada palavra do corpus PT-BR, computa HV via bundle de bytes.
//! Testa analogias: A is to B as C is to ?
//!   D_estimado = bundle(palavras_corpus) com argmin Hamming(HV(C) ⊕ HV(B) ⊕ HV(A), HV(D))

use std::{collections::HashMap, fs, io};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const HV_BITS: usize = 1024;
const HV_U64:  usize = HV_BITS / 64;
type Hv = [u64; HV_U64];

const HV_CACHE: &str = "data/byte_hvs_qwen.bin";
const CORPUS: &str = "data/dialogues_augmented.txt";

fn casefold(b: u8) -> u8 { if b.is_ascii_alphabetic() { b | 0x20 } else { b } }
fn strip_diacritics(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0xC3 && i + 1 < bytes.len() {
            let m = match bytes[i+1] {
                0xA0..=0xA5 => b'a', 0xA8..=0xAB => b'e',
                0xAC..=0xAF => b'i', 0xB2..=0xB6 => b'o',
                0xB9..=0xBC => b'u', 0xA7 => b'c',
                0x80..=0x85 => b'A', 0x87 => b'C',
                0x88..=0x8B => b'E', 0x8C..=0x8F => b'I',
                0x92..=0x96 => b'O', 0x99..=0x9C => b'U',
                _ => { i += 2; continue; }
            };
            out.push(m); i += 2;
        } else { out.push(bytes[i]); i += 1; }
    }
    out
}
fn normalize(s: &str) -> Vec<u8> {
    let stripped = strip_diacritics(s.as_bytes());
    stripped.iter().map(|&b| {
        if b.is_ascii_alphanumeric() { casefold(b) }
        else { b' ' }
    }).filter(|&b| b != b' ').collect()
}

#[inline]
fn hamming(a: &Hv, b: &Hv) -> u32 {
    let mut h = 0u32;
    for i in 0..HV_U64 { h += (a[i] ^ b[i]).count_ones(); }
    h
}

fn bundle(hvs: &[Hv]) -> Hv {
    if hvs.is_empty() { return [0; HV_U64]; }
    let half = hvs.len() / 2;
    let mut counts = vec![0u16; HV_BITS];
    for hv in hvs {
        for w in 0..HV_U64 {
            for b in 0..64 {
                if (hv[w] >> b) & 1 == 1 { counts[w*64 + b] += 1; }
            }
        }
    }
    let mut out = [0u64; HV_U64];
    for p in 0..HV_BITS {
        if counts[p] as usize > half { out[p/64] |= 1u64 << (p%64); }
    }
    out
}

fn hv_xor(a: &Hv, b: &Hv) -> Hv {
    let mut o = [0u64; HV_U64];
    for i in 0..HV_U64 { o[i] = a[i] ^ b[i]; }
    o
}

fn load_byte_hvs() -> io::Result<[Hv; 256]> {
    let mut f = fs::File::open(HV_CACHE)?;
    let mut out = [[0u64; HV_U64]; 256];
    for h in &mut out {
        for w in h.iter_mut() {
            let mut bb = [0u8; 8];
            f.read_exact(&mut bb)?;
            *w = u64::from_le_bytes(bb);
        }
    }
    Ok(out)
}

fn word_hv(word: &str, byte_hvs: &[Hv; 256]) -> Hv {
    let norm = normalize(word);
    if norm.is_empty() { return [0; HV_U64]; }
    let bytes_hvs: Vec<Hv> = norm.iter().map(|&b| byte_hvs[b as usize]).collect();
    bundle(&bytes_hvs)
}

// Extract distinct words from corpus
fn extract_vocabulary(text: &str) -> Vec<String> {
    let mut set: std::collections::HashSet<String> = std::collections::HashSet::new();
    for line in text.lines() {
        for word in line.split(|c: char| !c.is_alphanumeric()) {
            let w = word.trim().to_lowercase();
            if w.len() >= 3 && w.len() <= 20 {
                set.insert(w);
            }
        }
    }
    let mut v: Vec<String> = set.into_iter().collect();
    v.sort();
    v
}

fn main() -> io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA HDC Analogy — vector arithmetic em bits               ║");
    println!("║   teste: HV(A) ⊕ HV(B) ⊕ HV(C) ≈ HV(D)?                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let byte_hvs = load_byte_hvs()?;

    let mut text = String::new();
    for p in &["data/dialogues_augmented.txt", "data/qwen_sustained.txt"] {
        if Path::new(p).exists() {
            text.push_str(&fs::read_to_string(p)?);
            text.push('\n');
        }
    }
    println!("  Corpus: {} bytes", text.len());

    let vocab = extract_vocabulary(&text);
    println!("  Vocabulário extraído: {} palavras únicas\n", vocab.len());

    // Pre-computa HVs de todas palavras
    let word_hvs: Vec<(String, Hv)> = vocab.iter()
        .map(|w| (w.clone(), word_hv(w, &byte_hvs)))
        .collect();

    // Test analogies
    let analogies: &[(&str, &str, &str, &str)] = &[
        // (a, b, c, expected_d): "A is to B as C is to D"
        ("homem", "mulher", "rei", "rainha"),
        ("brasil", "brasília", "frança", "paris"),
        ("brasil", "real", "frança", "euro"),
        ("dia", "noite", "claro", "escuro"),
        ("grande", "pequeno", "alto", "baixo"),
        ("rapaz", "moça", "menino", "menina"),
        ("amigo", "amiga", "professor", "professora"),
        ("brasil", "portugues", "frança", "frances"),
        ("computador", "tela", "carro", "para-brisas"),
        ("cozinhar", "cozido", "fritar", "frito"),
    ];

    println!("▸ Testando analogias (HV(c) ⊕ HV(b) ⊕ HV(a) → nearest):");
    println!("  {:>15} {:>15} {:>15} {:>15}  {:>15}  rank", "A", "B", "C", "esperado_D", "predito_D");
    println!("  {}", "─".repeat(90));

    let mut hits_top1 = 0;
    let mut hits_top5 = 0;
    let mut total = 0;

    for (a, b, c, expected) in analogies {
        let hv_a = word_hv(a, &byte_hvs);
        let hv_b = word_hv(b, &byte_hvs);
        let hv_c = word_hv(c, &byte_hvs);
        // D ≈ A ⊕ B ⊕ C (XOR like "subtraction" in HDC binary algebra)
        let target = hv_xor(&hv_xor(&hv_a, &hv_b), &hv_c);

        // Find nearest word in vocab (excluding inputs)
        let inputs = [a.to_lowercase(), b.to_lowercase(), c.to_lowercase()];
        let mut scored: Vec<(u32, &String)> = word_hvs.iter()
            .filter(|(w, _)| !inputs.contains(&w.to_lowercase()))
            .map(|(w, hv)| (hamming(&target, hv), w))
            .collect();
        scored.sort_by_key(|&(h, _)| h);

        let top1 = scored.first().map(|(_, w)| (*w).clone()).unwrap_or_else(|| String::new());
        let exp_l = expected.to_lowercase();
        let rank = scored.iter().position(|(_, w)| **w == exp_l)
            .map(|p| p + 1).unwrap_or(usize::MAX);
        println!("  {:>15} {:>15} {:>15} {:>15}  {:>15}  {}",
            a, b, c, expected, top1,
            if rank == usize::MAX { "miss".to_string() } else { format!("#{}", rank) });
        if top1.to_lowercase() == exp_l { hits_top1 += 1; }
        if rank <= 5 { hits_top5 += 1; }
        total += 1;
    }

    println!("\n  ★ Top-1 hits: {}/{} ({:.0}%)", hits_top1, total, 100.0 * hits_top1 as f32 / total as f32);
    println!("  ★ Top-5 hits: {}/{} ({:.0}%)", hits_top5, total, 100.0 * hits_top5 as f32 / total as f32);

    // Baseline comparison: pure random expected hit rate = 1/|vocab|
    let baseline = 100.0 / vocab.len() as f32;
    println!("  ★ Baseline (random): {:.4}%", baseline);

    // Bonus: cluster analysis - HVs próximos no espaço
    println!("\n▸ Top vizinhos pra 'brasil' (similaridade Hamming):");
    let hv_q = word_hv("brasil", &byte_hvs);
    let mut nb: Vec<(u32, &String)> = word_hvs.iter()
        .filter(|(w, _)| w.to_lowercase() != "brasil")
        .map(|(w, hv)| (hamming(&hv_q, hv), w))
        .collect();
    nb.sort_by_key(|&(h, _)| h);
    for (h, w) in nb.iter().take(10) {
        let sim = 1.0 - *h as f32 / HV_BITS as f32;
        println!("    sim={:.3}  {}", sim, w);
    }

    Ok(())
}
