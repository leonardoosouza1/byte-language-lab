//! iara_hdc_chat_v2.rs — HDC v2: dual-signature (bundle + sequence).
//!
//! v1 problema: XOR puro colapsa info em strings curtas (4 bytes podem cancelar).
//! v2: combina DUAS signatures por entrada:
//!   - bag_HV  = majority vote dos byte HVs (multiset, ordem irrelevante)
//!   - seq_HV  = XOR_i (byte_HV[i] ⊕ pos_HV[i]) (ordem importa, igual v1)
//!   - bigram_HV = bundle dos pares byte_HV[i] ⊕ ROT(byte_HV[i+1])
//! Similarity = 0.4·hamming(bag) + 0.4·hamming(bigram) + 0.2·hamming(seq)
//!
//! Bundling (majority) preserva semântica sem cancelamento XOR.

use std::{collections::HashMap, fs, io};
use std::io::{Read, Seek, SeekFrom, Write, BufRead};
use std::path::Path;
use std::time::Instant;

const HV_BITS: usize = 1024;
const HV_U64:  usize = HV_BITS / 64;
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
        else if b.is_ascii_whitespace() { b' ' }
        else { b' ' }
    }).collect()
}

#[inline] fn hv_xor(a: &mut Hv, b: &Hv) { for i in 0..HV_U64 { a[i] ^= b[i]; } }
#[inline]
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

/// Rotate HV left by 1 word (64 bits). Cheap, sufficient pra ordem em par.
fn hv_rot1(hv: &Hv) -> Hv {
    let mut out = [0u64; HV_U64];
    for i in 0..HV_U64 {
        out[(i + 1) % HV_U64] = hv[i];
    }
    out
}

/// Bundle via majority vote. tie → bit = 0.
fn bundle(hvs: &[Hv]) -> Hv {
    if hvs.is_empty() { return [0u64; HV_U64]; }
    let n = hvs.len();
    let half = n / 2;
    let mut counts = [[0u16; 64]; HV_U64];
    for hv in hvs {
        for w in 0..HV_U64 {
            for b in 0..64 {
                if (hv[w] >> b) & 1 == 1 {
                    counts[w][b] += 1;
                }
            }
        }
    }
    let mut out = [0u64; HV_U64];
    for w in 0..HV_U64 {
        for b in 0..64 {
            if counts[w][b] as usize > half {
                out[w] |= 1u64 << b;
            }
        }
    }
    out
}

// ── pos HVs ────────────────────────────────────────────────────────────────
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

// ── byte HVs from Qwen (load cache) ───────────────────────────────────────
fn load_byte_hvs() -> io::Result<[Hv; 256]> {
    if !Path::new(HV_CACHE).exists() {
        eprintln!("ERRO: cache {} não existe. Rode iara_hdc_chat primeiro.", HV_CACHE);
        std::process::exit(1);
    }
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

// ── Triple signature per string ───────────────────────────────────────────
struct Sig {
    bag:    Hv,    // multiset of bytes (bundle)
    bigram: Hv,    // multiset of (byte_i ⊕ rot(byte_i+1))
    seq:    Hv,    // XOR with position
}

fn encode_v2(s: &str, byte_hvs: &[Hv; 256], pos_hvs: &[Hv; 256]) -> Sig {
    let norm = normalize(s);

    // bag
    let unigrams: Vec<Hv> = norm.iter().map(|&b| byte_hvs[b as usize]).collect();
    let bag = bundle(&unigrams);

    // bigram bundle
    let bigrams: Vec<Hv> = if norm.len() >= 2 {
        norm.windows(2).map(|w| {
            let a = byte_hvs[w[0] as usize];
            let b_rot = hv_rot1(&byte_hvs[w[1] as usize]);
            hv_xor_new(&a, &b_rot)
        }).collect()
    } else { unigrams.clone() };
    let bigram = bundle(&bigrams);

    // seq with pos (v1 style)
    let mut seq = [0u64; HV_U64];
    for (i, &b) in norm.iter().enumerate() {
        let pos = i.min(255);
        let bound = hv_xor_new(&byte_hvs[b as usize], &pos_hvs[pos]);
        hv_xor(&mut seq, &bound);
    }

    Sig { bag, bigram, seq }
}

// ── Combined similarity ───────────────────────────────────────────────────
fn similarity(a: &Sig, b: &Sig) -> f32 {
    let h_bag = hamming(&a.bag, &b.bag);
    let h_big = hamming(&a.bigram, &b.bigram);
    let h_seq = hamming(&a.seq, &b.seq);
    let s_bag = 1.0 - h_bag as f32 / HV_BITS as f32;
    let s_big = 1.0 - h_big as f32 / HV_BITS as f32;
    let s_seq = 1.0 - h_seq as f32 / HV_BITS as f32;
    0.5 * s_bag + 0.35 * s_big + 0.15 * s_seq
}

// ── Corpus ─────────────────────────────────────────────────────────────────
struct QA { user_raw: String, sig: Sig, model: String }

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
                    let sig = encode_v2(&u, byte_hvs, pos_hvs);
                    out.push(QA { user_raw: u, sig, model });
                }
            }
        }
    }
    out
}

fn main() -> io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA HDC Chat v2 — bag + bigram + seq (1024-bit each)      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let t0 = Instant::now();
    let byte_hvs = load_byte_hvs()?;
    let pos_hvs  = build_pos_hvs();
    println!("  HVs prontas em {:.0}ms", t0.elapsed().as_millis());

    // Concatena TODOS corpus existentes pra cobertura máxima
    let corpus_candidates = [
        "data/dialogues_augmented.txt",
        "data/qwen_sustained.txt",
    ];
    let mut text = String::new();
    let mut sources = Vec::new();
    for p in &corpus_candidates {
        if Path::new(p).exists() {
            let t = fs::read_to_string(p)?;
            text.push_str(&t);
            text.push('\n');
            sources.push(*p);
        }
    }
    let corpus_path = sources.join(" + ");
    let t1 = Instant::now();
    let qa = parse_corpus(&text, &byte_hvs, &pos_hvs);
    println!("  corpus: {} ({} pares em {:.0}ms)\n",
             corpus_path, qa.len(), t1.elapsed().as_millis());

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
        let q_sig = encode_v2(s, &byte_hvs, &pos_hvs);
        let mut scored: Vec<(f32, &QA)> = qa.iter().map(|item| {
            (similarity(&q_sig, &item.sig), item)
        }).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let elapsed = tq.elapsed().as_micros();
        let (sim, top) = &scored[0];
        println!("\x1b[1;34mMODEL:\x1b[0m {}", top.model);
        println!("\x1b[2m[match: \"{}\" sim={:.3} {}μs]\x1b[0m", top.user_raw, sim, elapsed);

        if show_n > 1 {
            for (sim, q) in scored.iter().take(show_n).skip(1) {
                let mtrunc = if q.model.len() > 60 { &q.model[..60] } else { &q.model[..] };
                println!("\x1b[2m  alt sim={:.3}: \"{}\" → {}\x1b[0m", sim, q.user_raw, mtrunc);
            }
        }
        println!();
        print!("\x1b[1;32mUSER:\x1b[0m "); io::stdout().flush().ok();
    }
    Ok(())
}
