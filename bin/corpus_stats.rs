//! corpus_stats.rs — estatísticas completas do corpus.

use std::fs;
use std::path::Path;
use std::collections::HashMap;

fn main() -> std::io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA Corpus Stats                                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let files = ["data/dialogues_pt.txt", "data/dialogues_augmented.txt",
                 "data/qwen_sustained.txt", "data/online_learning.txt"];
    let mut total_bytes = 0usize;
    let mut total_pairs = 0usize;
    let mut all_text = String::new();

    println!("▸ Por arquivo:\n  {:>40}  bytes      pairs", "file");
    println!("  {}", "-".repeat(70));
    for p in &files {
        if !Path::new(p).exists() { continue; }
        let t = fs::read_to_string(p)?;
        let pairs = t.matches("\nMODEL:").count() + (if t.starts_with("MODEL:") { 1 } else { 0 });
        println!("  {:>40}  {:>8}  {:>5}", p, t.len(), pairs);
        total_bytes += t.len();
        total_pairs += pairs;
        all_text.push_str(&t);
        all_text.push('\n');
    }
    println!("  {}", "-".repeat(70));
    println!("  {:>40}  {:>8}  {:>5}", "TOTAL", total_bytes, total_pairs);

    // Char frequency
    println!("\n▸ Distribuição de bytes (top 20):");
    let mut counts: HashMap<u8, u32> = HashMap::new();
    for &b in all_text.as_bytes() { *counts.entry(b).or_insert(0) += 1; }
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    for (b, c) in v.iter().take(20) {
        let ch = if b.is_ascii_graphic() { format!("'{}'", *b as char) }
                else if *b == b' ' { "SPC".to_string() }
                else if *b == b'\n' { "NL".to_string() }
                else { format!("\\x{:02X}", b) };
        let pct = 100.0 * *c as f32 / total_bytes as f32;
        let bar = "█".repeat((pct * 2.0) as usize);
        println!("  {:>5}  {:>8}  {:.2}%  {}", ch, c, pct, bar);
    }

    // Word frequency
    println!("\n▸ Top 25 palavras (post-normalization):");
    let mut wc: HashMap<String, u32> = HashMap::new();
    for line in all_text.lines() {
        for w in line.split(|c: char| !c.is_alphanumeric()) {
            let w = w.trim().to_lowercase();
            if w.len() >= 3 && w.len() <= 20 {
                *wc.entry(w).or_insert(0) += 1;
            }
        }
    }
    let mut vw: Vec<_> = wc.into_iter().collect();
    vw.sort_by(|a, b| b.1.cmp(&a.1));
    for chunk in vw.iter().take(25).collect::<Vec<_>>().chunks(5) {
        for (w, c) in chunk {
            print!("  {:<14} {:>5}", w, c);
        }
        println!();
    }

    // USER vs MODEL length distribution
    println!("\n▸ Tamanho de USER/MODEL (chars):");
    let mut user_lens = Vec::new();
    let mut model_lens = Vec::new();
    for line in all_text.lines() {
        let line = line.trim();
        if let Some(u) = line.strip_prefix("USER:") { user_lens.push(u.trim().chars().count()); }
        else if let Some(m) = line.strip_prefix("MODEL:") { model_lens.push(m.trim().chars().count()); }
    }
    let stat = |v: &[usize]| -> (usize, usize, usize, f32) {
        let mut sorted = v.to_vec();
        sorted.sort();
        let n = sorted.len();
        let avg = sorted.iter().sum::<usize>() as f32 / n.max(1) as f32;
        (sorted[0], sorted[n/2], sorted[n*9/10], avg)
    };
    if !user_lens.is_empty() {
        let (mn, p50, p90, avg) = stat(&user_lens);
        println!("  USER:  n={}, min={}, p50={}, p90={}, avg={:.1}", user_lens.len(), mn, p50, p90, avg);
    }
    if !model_lens.is_empty() {
        let (mn, p50, p90, avg) = stat(&model_lens);
        println!("  MODEL: n={}, min={}, p50={}, p90={}, avg={:.1}", model_lens.len(), mn, p50, p90, avg);
    }

    // Diacritics frequency
    let mut diac_count = 0;
    let mut multibyte = 0;
    for &b in all_text.as_bytes() {
        if b == 0xC3 { diac_count += 1; }
        if b >= 0x80 { multibyte += 1; }
    }
    println!("\n▸ UTF-8 multibyte:");
    println!("  bytes 0xC3 (PT-BR diacrítico prefix): {} ({:.2}%)", diac_count, 100.0 * diac_count as f32 / total_bytes as f32);
    println!("  total multibyte (>0x80):              {} ({:.2}%)", multibyte, 100.0 * multibyte as f32 / total_bytes as f32);

    // Bit distribution
    println!("\n▸ Distribuição de bits (frequência de cada posição = 1):");
    let mut bit_counts = [0u64; 8];
    for &b in all_text.as_bytes() {
        for i in 0..8 {
            if (b >> i) & 1 == 1 { bit_counts[i] += 1; }
        }
    }
    let total_bytes_u64 = total_bytes as u64;
    println!("  bit | {:>8} | {:>5}% | bar", "count", "");
    println!("  ----|----------|--------|--------------------");
    for i in (0..8).rev() {
        let pct = 100.0 * bit_counts[i] as f32 / total_bytes_u64 as f32;
        let bar = "█".repeat((pct as usize) / 2);
        println!("  {:>3} | {:>8} | {:>5.1}% | {}", i, bit_counts[i], pct, bar);
    }

    Ok(())
}
