//! byte_walker.rs — gerador de "palavras" walking o grafo byte-byte.
//!
//! Constrói transition matrix do corpus. Walk markoviano:
//!   - Estado: byte atual
//!   - Próximo: amostrado da distribuição P(next | current)
//!   - Para: quando gera espaço ou max length
//!
//! Gera "palavras" sintéticas que SOAM PT-BR sem rede neural.

use std::fs;
use std::path::Path;

const CORPUS_FILES: &[&str] = &[
    "data/dialogues_augmented.txt",
    "data/qwen_sustained.txt",
];

fn normalize_byte(b: u8) -> u8 {
    if b.is_ascii_alphabetic() { b | 0x20 }
    else if b.is_ascii_whitespace() { b' ' }
    else { b }
}

fn main() -> std::io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA Byte Walker — palavras sintéticas via grafo bigram    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Build bigram matrix from corpus
    let mut text = String::new();
    for p in CORPUS_FILES {
        if Path::new(p).exists() {
            text.push_str(&fs::read_to_string(p)?);
            text.push('\n');
        }
    }
    println!("  Corpus: {} bytes", text.len());

    let mut bigram = vec![vec![0u32; 256]; 256];
    let mut start_freq = vec![0u32; 256];   // probabilidades de início de palavra

    // Coleta bigrams + start frequencies
    for line in text.lines() {
        for word in line.split(|c: char| !c.is_alphanumeric()) {
            let bytes: Vec<u8> = word.bytes().map(normalize_byte).collect();
            if bytes.is_empty() { continue; }
            start_freq[bytes[0] as usize] += 1;
            for w in bytes.windows(2) {
                bigram[w[0] as usize][w[1] as usize] += 1;
            }
            // also bigram (last, space) to capture end
            bigram[bytes[bytes.len()-1] as usize][b' ' as usize] += 1;
        }
    }

    // Normalize start freq
    let start_total: u32 = start_freq.iter().sum();
    let start_dist: Vec<f32> = start_freq.iter().map(|&v| v as f32 / start_total.max(1) as f32).collect();

    // RNG
    let mut rng: u64 = 0xCAFEBABE;
    fn next_u32(s: &mut u64) -> u32 {
        *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
        (*s >> 16) as u32
    }
    fn next_f32(s: &mut u64) -> f32 {
        next_u32(s) as f32 / u32::MAX as f32
    }

    fn sample_dist(dist: &[f32], rng: &mut u64) -> usize {
        let r = next_f32(rng);
        let mut cum = 0.0f32;
        for (i, &p) in dist.iter().enumerate() {
            cum += p;
            if r < cum { return i; }
        }
        dist.len() - 1
    }

    // Generate N words at different temperatures
    for &temp in &[0.5f32, 0.8, 1.0, 1.2, 1.5] {
        println!("\n▸ Temperatura={:.1}: 20 palavras geradas:", temp);
        let mut words = Vec::new();
        for _ in 0..20 {
            // sample start byte
            let start = sample_dist(&start_dist, &mut rng) as u8;
            let mut word = vec![start];
            let max_len = 12;
            let mut cur = start;
            for _ in 1..max_len {
                let row: Vec<f32> = (0..256).map(|j| {
                    let v = bigram[cur as usize][j];
                    (v as f32).powf(1.0 / temp.max(0.1))
                }).collect();
                let total: f32 = row.iter().sum();
                if total < 1.0 { break; }
                let norm: Vec<f32> = row.iter().map(|x| x / total).collect();
                let next = sample_dist(&norm, &mut rng) as u8;
                if next == b' ' { break; }
                word.push(next);
                cur = next;
            }
            let s = String::from_utf8_lossy(&word).to_string();
            words.push(s);
        }
        // Print 4 per line
        for chunk in words.chunks(4) {
            print!("  ");
            for w in chunk {
                print!("{:<15} ", w);
            }
            println!();
        }
    }

    println!("\n▸ Gerando 5 \"frases\" via walking (10 palavras cada, temp 0.9):\n");
    for sn in 0..5 {
        let mut sentence = String::new();
        for _ in 0..10 {
            let start = sample_dist(&start_dist, &mut rng) as u8;
            let mut word = vec![start];
            let mut cur = start;
            for _ in 1..12 {
                let row: Vec<f32> = (0..256).map(|j| {
                    let v = bigram[cur as usize][j];
                    (v as f32).powf(1.0 / 0.9)
                }).collect();
                let total: f32 = row.iter().sum();
                if total < 1.0 { break; }
                let norm: Vec<f32> = row.iter().map(|x| x / total).collect();
                let next = sample_dist(&norm, &mut rng) as u8;
                if next == b' ' { break; }
                word.push(next);
                cur = next;
            }
            let w = String::from_utf8_lossy(&word);
            sentence.push_str(&w);
            sentence.push(' ');
        }
        println!("  {}: {}", sn+1, sentence.trim());
    }

    Ok(())
}
