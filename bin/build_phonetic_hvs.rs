//! build_phonetic_hvs.rs — gera byte HVs com features fonéticas PT-BR injetadas.
//!
//! Cada byte ASCII letra ganha 1024-bit HV onde:
//!   - 256 bits: feature fonética (vogal/cons, nasal, oclusiva...)
//!   - 768 bits: Qwen embedding sign (semantic)
//!
//! Bytes não-alfa: usam Qwen full ou aleatório como antes.

use std::{collections::HashMap, fs, io};
use std::io::{Read, Seek, SeekFrom, Write};

const HV_BITS: usize = 1024;
const HV_U64:  usize = HV_BITS / 64;
type Hv = [u64; HV_U64];

const QWEN: &str   = "/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b";
const HIDDEN: usize = 1536;
const VOCAB_N: usize = 151936;
const OUT_PATH: &str = "data/byte_hvs_phonetic.bin";

/// 8 features fonéticas binárias por letra alfabética PT-BR.
/// Ordem: vogal, nasal, oclusiva, fricativa, lateral, vibrante, voz, alta
fn phonetic_features(c: char) -> [u8; 8] {
    let c = c.to_ascii_lowercase();
    // [vogal, nasal, oclusiva, fricativa, lateral, vibrante, voz, alta]
    match c {
        'a' => [1, 0, 0, 0, 0, 0, 1, 0],   // vogal aberta
        'e' => [1, 0, 0, 0, 0, 0, 1, 1],   // vogal média alta
        'i' => [1, 0, 0, 0, 0, 0, 1, 1],   // vogal alta
        'o' => [1, 0, 0, 0, 0, 0, 1, 0],   // vogal média
        'u' => [1, 0, 0, 0, 0, 0, 1, 1],   // vogal alta
        // Nasais
        'm' => [0, 1, 1, 0, 0, 0, 1, 0],
        'n' => [0, 1, 1, 0, 0, 0, 1, 0],
        // Oclusivas surdas
        'p' => [0, 0, 1, 0, 0, 0, 0, 0],
        't' => [0, 0, 1, 0, 0, 0, 0, 0],
        'k' => [0, 0, 1, 0, 0, 0, 0, 0],
        'c' => [0, 0, 1, 0, 0, 0, 0, 0],   // duro = k
        'q' => [0, 0, 1, 0, 0, 0, 0, 0],
        // Oclusivas sonoras
        'b' => [0, 0, 1, 0, 0, 0, 1, 0],
        'd' => [0, 0, 1, 0, 0, 0, 1, 0],
        'g' => [0, 0, 1, 0, 0, 0, 1, 0],
        // Fricativas surdas
        'f' => [0, 0, 0, 1, 0, 0, 0, 0],
        's' => [0, 0, 0, 1, 0, 0, 0, 0],
        'x' => [0, 0, 0, 1, 0, 0, 0, 0],   // sh ou ks
        // Fricativas sonoras
        'v' => [0, 0, 0, 1, 0, 0, 1, 0],
        'z' => [0, 0, 0, 1, 0, 0, 1, 0],
        'j' => [0, 0, 0, 1, 0, 0, 1, 0],   // PT
        'h' => [0, 0, 0, 1, 0, 0, 0, 0],
        // Lateral
        'l' => [0, 0, 0, 0, 1, 0, 1, 0],
        // Vibrante
        'r' => [0, 0, 0, 0, 0, 1, 1, 0],
        // Aproximantes / outros
        'y' => [1, 0, 0, 0, 0, 0, 1, 1],   // i-like
        'w' => [1, 0, 0, 0, 0, 0, 1, 1],   // u-like
        _ => [0, 0, 0, 0, 0, 0, 0, 0],
    }
}

/// Expand 8-bit features into 256-bit HV via replication pattern + position encoding.
/// Each feature gets 32 bits (replicated 32×), so similar features share bits.
fn features_to_hv_segment(feats: [u8; 8]) -> [u64; 4] {  // 256 bits = 4 u64
    let mut out = [0u64; 4];
    for (i, &f) in feats.iter().enumerate() {
        if f == 1 {
            // Feature i = 1 → set 32 contiguous bits at position i*32
            let bit_start = i * 32;
            for b in 0..32 {
                let abs = bit_start + b;
                out[abs / 64] |= 1u64 << (abs % 64);
            }
        }
    }
    out
}

fn main() -> io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   build_phonetic_hvs — byte HVs com fonética PT-BR + Qwen    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ── Load Qwen vocab + embeddings ──────────────────────────────────
    println!("▸ Lendo Qwen vocab + embeddings...");
    let vocab: HashMap<String, u32> = serde_json::from_str(
        &fs::read_to_string(format!("{QWEN}/vocab.json"))?
    ).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    let mut f = fs::File::open(format!("{QWEN}/model.safetensors"))?;
    let mut buf8 = [0u8; 8];
    f.read_exact(&mut buf8)?;
    let hlen = u64::from_le_bytes(buf8) as usize;
    let mut hbuf = vec![0u8; hlen];
    f.read_exact(&mut hbuf)?;
    let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&hbuf)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let embed_off = header["model.embed_tokens.weight"]["data_offsets"][0]
        .as_u64().unwrap() as usize;
    let data_start = 8 + hlen;
    f.seek(SeekFrom::Start((data_start + embed_off) as u64))?;
    let mut raw = vec![0u8; VOCAB_N * HIDDEN * 2];
    f.read_exact(&mut raw)?;

    // Per byte centroid
    println!("▸ Computando centroides Qwen...");
    let mut centroids: Vec<Vec<f32>> = (0..256).map(|_| vec![0.0f32; HIDDEN]).collect();
    let mut counts: Vec<u32> = vec![0; 256];
    for (tok_str, &tid) in &vocab {
        let bs = tok_str.as_bytes();
        if bs.is_empty() { continue; }
        let raw_first = if bs.len() >= 2 && bs[0] == 0xC4 && bs[1] == 0xA0 {
            if bs.len() >= 3 { bs[2] } else { continue }
        } else { bs[0] };
        let normed = if raw_first.is_ascii_alphabetic() { raw_first | 0x20 } else { raw_first };
        let base = tid as usize * HIDDEN * 2;
        if base + HIDDEN * 2 > raw.len() { continue; }
        for i in 0..HIDDEN {
            let bits = u16::from_le_bytes([raw[base + i*2], raw[base + i*2 + 1]]);
            let v = f32::from_bits((bits as u32) << 16);
            centroids[normed as usize][i] += v;
        }
        counts[normed as usize] += 1;
    }
    drop(raw);

    println!("▸ Construindo HVs (256 bits fonética + 768 bits Qwen)...");
    let mut out_hvs = [[0u64; HV_U64]; 256];
    let mut rng: u64 = 0xBADC0FFEE0DDF00D;
    let mut n_phonetic = 0;
    let mut n_qwen_only = 0;
    let mut n_random = 0;

    for b_u16 in 0u16..256 {
        let b = b_u16 as u8;
        let mut hv = [0u64; HV_U64];

        // Bits 0..256: fonética se alfa
        if b.is_ascii_alphabetic() {
            let feats = phonetic_features(b as char);
            let seg = features_to_hv_segment(feats);
            for i in 0..4 { hv[i] = seg[i]; }
            n_phonetic += 1;
        }

        // Bits 256..1024 (= words 4..16, 768 bits): Qwen sign
        if counts[b as usize] > 0 {
            let inv = 1.0 / counts[b as usize] as f32;
            for i in 0..768 {
                let coord_idx = i.min(HIDDEN - 1);
                let v = centroids[b as usize][coord_idx] * inv;
                if v > 0.0 {
                    let abs_bit = 256 + i;
                    hv[abs_bit / 64] |= 1u64 << (abs_bit % 64);
                }
            }
            if !b.is_ascii_alphabetic() { n_qwen_only += 1; }
        } else {
            // Random pra bytes sem Qwen e sem fonética (e.g., bytes UTF-8 cont)
            for w in 4..HV_U64 {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                hv[w] = rng.wrapping_add(b as u64 * 0x9E3779B97F4A7C15);
            }
            if !b.is_ascii_alphabetic() { n_random += 1; }
        }

        out_hvs[b as usize] = hv;
    }

    println!("  {} alpha bytes com fonética", n_phonetic);
    println!("  {} bytes Qwen-only (não-alpha)", n_qwen_only);
    println!("  {} bytes random (sem tokens)", n_random);

    // Save
    let mut out_f = fs::File::create(OUT_PATH)?;
    for h in &out_hvs {
        for &w in h.iter() {
            out_f.write_all(&w.to_le_bytes())?;
        }
    }
    println!("\n▸ Salvo: {} ({} bytes)", OUT_PATH, 256 * HV_U64 * 8);

    // Verify phonetic harmony
    println!("\n▸ Verificação: phonetic-similar letras devem ter HV próximas:\n");
    let test_pairs = [
        ('a', 'e'), ('a', 'i'), ('a', 'o'), ('a', 'u'),  // vogais entre si
        ('p', 'b'), ('t', 'd'), ('k', 'g'),               // pares sonora/surda
        ('m', 'n'),                                       // nasais
        ('s', 'z'), ('f', 'v'),                          // fricativas pares
        ('a', 'p'), ('a', 's'),                          // vogal vs cons (esperado: distante)
        ('p', 's'),                                       // ocl vs fric (distante)
    ];
    for (a, b) in test_pairs {
        let hv_a = out_hvs[a as usize];
        let hv_b = out_hvs[b as usize];
        let mut hd = 0u32;
        for i in 0..HV_U64 { hd += (hv_a[i] ^ hv_b[i]).count_ones(); }
        let sim = 1.0 - hd as f32 / HV_BITS as f32;
        println!("  {} ↔ {}: hamming={:>4} sim={:.3}", a, b, hd, sim);
    }
    Ok(())
}
