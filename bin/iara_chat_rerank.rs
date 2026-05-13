//! iara_chat_rerank.rs — chat com pipeline em dois estágios:
//!   1. HDC top-K (fast, μs): candidate retrieval
//!   2. Levenshtein re-ranking (slow, ms but ON TOP-K only): final accuracy
//!
//! Combina velocidade HDC + precisão edit distance.

use std::{collections::HashMap, fs, io};
use std::io::{Read, Write, BufRead};
use std::path::Path;
use std::time::Instant;

const HV_BITS: usize = 1024;
const HV_U64:  usize = HV_BITS / 64;
type Hv = [u64; HV_U64];

const HV_CACHE: &str = "data/byte_hvs_qwen.bin";
const TOP_K: usize = 30;

// ── HV utilities (compact) ────────────────────────────────────────────────
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
fn normalize(s: &str) -> String {
    let stripped = strip_diacritics(s.as_bytes());
    let v: Vec<u8> = stripped.iter().map(|&b| {
        if b.is_ascii_alphanumeric() { casefold(b) }
        else if b.is_ascii_whitespace() { b' ' }
        else { b' ' }
    }).collect();
    let s = String::from_utf8_lossy(&v).to_string();
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn hamming(a: &Hv, b: &Hv) -> u32 {
    let mut h = 0u32;
    for i in 0..HV_U64 { h += (a[i] ^ b[i]).count_ones(); }
    h
}
fn hv_xor_new(a: &Hv, b: &Hv) -> Hv {
    let mut o = [0u64; HV_U64];
    for i in 0..HV_U64 { o[i] = a[i] ^ b[i]; }
    o
}
fn hv_rot1(hv: &Hv) -> Hv {
    let mut out = [0u64; HV_U64];
    for i in 0..HV_U64 { out[(i + 1) % HV_U64] = hv[i]; }
    out
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
fn load_hvs(path: &str) -> io::Result<[Hv; 256]> {
    let mut f = fs::File::open(path)?;
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

struct Sig { bag: Hv, bigram: Hv }

fn encode(s: &str, byte_hvs: &[Hv; 256]) -> Sig {
    let norm = normalize(s);
    let bytes = norm.as_bytes();
    let unigrams: Vec<Hv> = bytes.iter().map(|&b| byte_hvs[b as usize]).collect();
    let bag = bundle(&unigrams);
    let bigrams: Vec<Hv> = if bytes.len() >= 2 {
        bytes.windows(2).map(|w| {
            let a = byte_hvs[w[0] as usize];
            let b_rot = hv_rot1(&byte_hvs[w[1] as usize]);
            hv_xor_new(&a, &b_rot)
        }).collect()
    } else { unigrams.clone() };
    let bigram = bundle(&bigrams);
    Sig { bag, bigram }
}

fn sim(a: &Sig, b: &Sig) -> f32 {
    let h_bag = hamming(&a.bag, &b.bag);
    let h_big = hamming(&a.bigram, &b.bigram);
    let s_bag = 1.0 - h_bag as f32 / HV_BITS as f32;
    let s_big = 1.0 - h_big as f32 / HV_BITS as f32;
    0.6 * s_bag + 0.4 * s_big
}

// Levenshtein distance (rapid for short strings ≤ 100 chars)
fn levenshtein(a: &str, b: &str) -> usize {
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    let m = av.len();
    let n = bv.len();
    if m == 0 { return n; }
    if n == 0 { return m; }
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m { dp[i][0] = i; }
    for j in 0..=n { dp[0][j] = j; }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if av[i-1] == bv[j-1] { 0 } else { 1 };
            dp[i][j] = (dp[i-1][j] + 1).min(dp[i][j-1] + 1).min(dp[i-1][j-1] + cost);
        }
    }
    dp[m][n]
}

fn lev_sim(a: &str, b: &str) -> f32 {
    let a = normalize(a);
    let b = normalize(b);
    let d = levenshtein(&a, &b);
    let maxl = a.chars().count().max(b.chars().count()).max(1);
    1.0 - d as f32 / maxl as f32
}

struct QA { user_raw: String, user_norm: String, sig: Sig, model: String }

fn parse_corpus(text: &str, byte_hvs: &[Hv; 256]) -> Vec<QA> {
    let mut out = Vec::new();
    let mut cur: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(u) = line.strip_prefix("USER:") {
            cur = Some(u.trim().to_string());
        } else if let Some(m) = line.strip_prefix("MODEL:") {
            if let Some(u) = cur.take() {
                let model = m.trim().to_string();
                if !u.is_empty() && !model.is_empty() {
                    let sig = encode(&u, byte_hvs);
                    let user_norm = normalize(&u);
                    out.push(QA { user_raw: u, user_norm, sig, model });
                }
            }
        }
    }
    out
}

fn main() -> io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA Chat — HDC top-K + Levenshtein re-ranking             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let byte_hvs = load_hvs(HV_CACHE)?;
    let mut text = String::new();
    for p in &["data/dialogues_augmented.txt", "data/qwen_sustained.txt", "data/online_learning.txt"] {
        if Path::new(p).exists() {
            text.push_str(&fs::read_to_string(p)?);
            text.push('\n');
        }
    }
    let qa = parse_corpus(&text, &byte_hvs);
    println!("  Corpus: {} pares\n", qa.len());

    let stdin = io::stdin();
    print!("\x1b[1;32mUSER:\x1b[0m "); io::stdout().flush().ok();

    for line in stdin.lock().lines() {
        let line = line?;
        let s = line.trim();
        if s.is_empty() { print!("\x1b[1;32mUSER:\x1b[0m "); io::stdout().flush().ok(); continue; }
        if s == "/quit" || s == "/exit" { break; }

        // Stage 1: HDC top-K
        let t1 = Instant::now();
        let q_sig = encode(s, &byte_hvs);
        let mut hdc_scores: Vec<(f32, usize)> = qa.iter().enumerate()
            .map(|(i, item)| (sim(&q_sig, &item.sig), i))
            .collect();
        // Partial sort top-K
        hdc_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let topk = hdc_scores.into_iter().take(TOP_K).collect::<Vec<_>>();
        let t_hdc = t1.elapsed().as_micros();

        // Stage 2: Levenshtein re-ranking
        let t2 = Instant::now();
        let q_norm = normalize(s);
        let mut reranked: Vec<(f32, usize, f32, f32)> = topk.into_iter().map(|(hdc_s, idx)| {
            let item = &qa[idx];
            let lev_s = lev_sim(&q_norm, &item.user_norm);
            // Combined: 0.4 HDC + 0.6 Levenshtein
            let combined = 0.4 * hdc_s + 0.6 * lev_s;
            (combined, idx, hdc_s, lev_s)
        }).collect();
        reranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let t_rerank = t2.elapsed().as_micros();

        let (combined, idx, hdc_s, lev_s) = reranked[0];
        let item = &qa[idx];
        println!("\x1b[1;34mMODEL:\x1b[0m {}", item.model);
        println!("\x1b[2m[match: \"{}\" combined={:.3} (HDC={:.3} Lev={:.3}) | HDC {}μs + rerank {}μs]\x1b[0m",
                 item.user_raw, combined, hdc_s, lev_s, t_hdc, t_rerank);
        println!();
        print!("\x1b[1;32mUSER:\x1b[0m "); io::stdout().flush().ok();
    }
    Ok(())
}
