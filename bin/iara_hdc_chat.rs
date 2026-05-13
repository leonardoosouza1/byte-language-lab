//! iara_hdc_chat.rs — HDC chat (1024-bit hypervectors, Qwen-informed).
//!
//! Composição: signature(s) = XOR_i (byte_HV[s[i]] ⊕ pos_HV[i])
//!   - byte_HV[256][1024 bits]: sign(centroid of Qwen embeddings de tokens iniciando com byte b)
//!   - pos_HV[256][1024 bits]: vetores aleatórios determinísticos (uma por posição)
//!
//! Similarity: 1 - hamming(a, b) / 1024.   GPU/CPU pop count rápido.
//!
//! Sem rede neural em inferência. Só XOR + POPCOUNT puro.

use std::{collections::HashMap, fs, io};
use std::io::{Read, Seek, SeekFrom, Write, BufRead};
use std::path::Path;
use std::time::Instant;

const HV_BITS: usize = 1024;
const HV_U64:  usize = HV_BITS / 64; // 16

type Hv = [u64; HV_U64];

const QWEN: &str   = "/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b";
const HIDDEN: usize = 1536;
const VOCAB_N: usize = 151936;
const HV_CACHE: &str = "data/byte_hvs_qwen.bin";
const CORPUS: &str   = "data/dialogues_augmented.txt";

// ── Bit utilities ──────────────────────────────────────────────────────────
fn casefold(b: u8) -> u8 { if b.is_ascii_alphabetic() { b | 0x20 } else { b } }

fn strip_diacritics(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0xC3 && i + 1 < bytes.len() {
            let m = match bytes[i+1] {
                0xA0..=0xA5 => b'a',
                0xA8..=0xAB => b'e',
                0xAC..=0xAF => b'i',
                0xB2..=0xB6 => b'o',
                0xB9..=0xBC => b'u',
                0xA7        => b'c',
                0x80..=0x85 => b'A',
                0x87        => b'C',
                0x88..=0x8B => b'E',
                0x8C..=0x8F => b'I',
                0x92..=0x96 => b'O',
                0x99..=0x9C => b'U',
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
        else if b.is_ascii_whitespace() { b' ' }
        else { b' ' }
    }).collect()
}

#[inline]
fn hv_xor(a: &mut Hv, b: &Hv) { for i in 0..HV_U64 { a[i] ^= b[i]; } }

#[inline]
fn hamming(a: &Hv, b: &Hv) -> u32 {
    let mut h = 0u32;
    for i in 0..HV_U64 { h += (a[i] ^ b[i]).count_ones(); }
    h
}

fn hv_xor_combined(a: &Hv, b: &Hv) -> Hv {
    let mut out = [0u64; HV_U64];
    for i in 0..HV_U64 { out[i] = a[i] ^ b[i]; }
    out
}

// ── Generate position HVs (deterministic xorshift) ────────────────────────
fn build_pos_hvs() -> [Hv; 256] {
    let mut out = [[0u64; HV_U64]; 256];
    let mut s: u64 = 0xDEADBEEFCAFEBABE;
    for p in 0..256 {
        for w in 0..HV_U64 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            out[p][w] = s;
        }
    }
    out
}

// ── Build byte HVs from Qwen embedding centroids ──────────────────────────
fn build_byte_hvs() -> io::Result<[Hv; 256]> {
    println!("▸ Construindo byte HVs do Qwen (one-shot, salva cache)...");
    let vocab_raw = fs::read_to_string(format!("{QWEN}/vocab.json"))?;
    let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_raw)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    println!("  vocab: {} tokens", vocab.len());

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

    println!("  lendo 466MB BF16 embed table...");
    f.seek(SeekFrom::Start((data_start + embed_off) as u64))?;
    let mut raw = vec![0u8; VOCAB_N * HIDDEN * 2];
    f.read_exact(&mut raw)?;

    // Per byte (normalized first byte): sum embeddings + count
    let mut centroids: Vec<Vec<f32>> = (0..256).map(|_| vec![0.0f32; HIDDEN]).collect();
    let mut counts: Vec<u32> = vec![0; 256];

    println!("  agregando centroides por byte...");
    for (tok_str, &tid) in &vocab {
        let bs = tok_str.as_bytes();
        if bs.is_empty() { continue; }
        let raw_first = if bs.len() >= 2 && bs[0] == 0xC4 && bs[1] == 0xA0 {
            if bs.len() >= 3 { bs[2] } else { continue }
        } else { bs[0] };
        // Apply normalization same as query
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

    println!("  binarizando para HV 1024-bit (sign)...");
    let mut out = [[0u64; HV_U64]; 256];
    let mut nbins = 0;
    let mut nrandom = 0;
    let mut rng: u64 = 0xBADC0FFEE0DDF00D;
    for b in 0..256 {
        if counts[b] == 0 {
            // No tokens — random HV deterministico
            for w in 0..HV_U64 {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                out[b][w] = rng.wrapping_add(b as u64 * 0x9E3779B97F4A7C15);
            }
            nrandom += 1;
            continue;
        }
        let inv = 1.0 / counts[b] as f32;
        // Take 1024 of 1536 coords. Use stride pra cobrir espaço todo (every 1.5-th).
        // Simpler: pick coords 0..1024 directly.
        for i in 0..HV_BITS {
            let v = centroids[b][i] * inv;
            if v > 0.0 {
                out[b][i / 64] |= 1u64 << (i % 64);
            }
        }
        nbins += 1;
    }
    println!("  {} bytes com Qwen centroid, {} aleatórios (sem tokens)", nbins, nrandom);

    // Save cache
    let mut cf = fs::File::create(HV_CACHE)?;
    for h in &out {
        for &w in h.iter() {
            cf.write_all(&w.to_le_bytes())?;
        }
    }
    println!("  cache: {} ({} bytes)", HV_CACHE, 256 * HV_U64 * 8);

    Ok(out)
}

fn load_byte_hvs() -> io::Result<[Hv; 256]> {
    if Path::new(HV_CACHE).exists() {
        println!("▸ Carregando cache {}", HV_CACHE);
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
    } else {
        build_byte_hvs()
    }
}

// ── Encode string ──────────────────────────────────────────────────────────
fn encode(s: &str, byte_hvs: &[Hv; 256], pos_hvs: &[Hv; 256]) -> Hv {
    let norm = normalize(s);
    let mut acc = [0u64; HV_U64];
    for (i, &b) in norm.iter().enumerate() {
        let pos = i.min(255);
        let bound = hv_xor_combined(&byte_hvs[b as usize], &pos_hvs[pos]);
        hv_xor(&mut acc, &bound);
    }
    acc
}

// ── Corpus ─────────────────────────────────────────────────────────────────
struct QA { user_raw: String, hv: Hv, model: String }

fn parse_corpus(text: &str, byte_hvs: &[Hv; 256], pos_hvs: &[Hv; 256]) -> Vec<QA> {
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
                    let hv = encode(&u, byte_hvs, pos_hvs);
                    out.push(QA { user_raw: u, hv, model });
                }
            }
        }
    }
    out
}

fn main() -> io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA HDC Chat — Hyperdim 1024-bit  XOR/POPCOUNT only       ║");
    println!("║   byte HVs = sign(Qwen embedding centroid per byte)          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let t0 = Instant::now();
    let byte_hvs = load_byte_hvs()?;
    let pos_hvs  = build_pos_hvs();
    println!("  HVs prontas em {:.0}ms", t0.elapsed().as_millis());

    let text = fs::read_to_string(CORPUS)?;
    let t1 = Instant::now();
    let qa = parse_corpus(&text, &byte_hvs, &pos_hvs);
    println!("  corpus: {} pares indexados em {:.0}ms\n",
             qa.len(), t1.elapsed().as_millis());

    println!("Digite mensagens, /quit, /show N pra alternativas:\n");

    let mut show_n = 1usize;
    let stdin = io::stdin();
    print!("\x1b[1;32mUSER:\x1b[0m "); io::stdout().flush().ok();

    for line in stdin.lock().lines() {
        let line = line?;
        let s = line.trim();
        if s.is_empty() { print!("\x1b[1;32mUSER:\x1b[0m "); io::stdout().flush().ok(); continue; }
        if s == "/quit" || s == "/exit" { println!("bye"); break; }
        if let Some(n) = s.strip_prefix("/show ") {
            show_n = n.parse().unwrap_or(1);
            println!("[show_n={}]", show_n);
            print!("\x1b[1;32mUSER:\x1b[0m "); io::stdout().flush().ok();
            continue;
        }

        let tq = Instant::now();
        let q_hv = encode(s, &byte_hvs, &pos_hvs);
        let mut scored: Vec<(u32, &QA)> = qa.iter().map(|item| {
            (hamming(&q_hv, &item.hv), item)
        }).collect();
        scored.sort_by_key(|&(h, _)| h);

        let elapsed = tq.elapsed().as_micros();
        let (h, top) = &scored[0];
        let sim = 1.0 - *h as f32 / HV_BITS as f32;
        println!("\x1b[1;34mMODEL:\x1b[0m {}", top.model);
        println!("\x1b[2m[match: \"{}\" hamming={} sim={:.3} {}μs]\x1b[0m",
                 top.user_raw, h, sim, elapsed);

        if show_n > 1 {
            for (h, q) in scored.iter().take(show_n).skip(1) {
                let s2 = 1.0 - *h as f32 / HV_BITS as f32;
                let mtrunc = if q.model.len() > 60 { &q.model[..60] } else { &q.model[..] };
                println!("\x1b[2m  alt sim={:.3}: \"{}\" → {}\x1b[0m", s2, q.user_raw, mtrunc);
            }
        }
        println!();
        print!("\x1b[1;32mUSER:\x1b[0m "); io::stdout().flush().ok();
    }
    Ok(())
}
