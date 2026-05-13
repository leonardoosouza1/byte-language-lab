//! iara_chat_eval.rs — avalia múltiplas variantes de chat lado-a-lado.
//!
//! Test set: 40 prompts (alguns no corpus, alguns novos).
//! Variantes:
//!   - iara_byte_chat (bigram + char-match, 100% sem Qwen)
//!   - iara_hdc_chat   (HDC v1, XOR position)
//!   - iara_hdc_chat_v2(HDC v2, bag+bigram+seq)
//!   - intelligent     (v2 + Qwen fallback)
//!
//! Mede: latência, sim score, e se a resposta faz sentido (manual review = print + label)

use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

const HV_BITS: usize = 1024;
const HV_U64:  usize = HV_BITS / 64;
type Hv = [u64; HV_U64];

const HV_CACHE: &str = "data/byte_hvs_qwen.bin";

// ── Shared utils ───────────────────────────────────────────────────────────
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
fn load_byte_hvs() -> std::io::Result<[Hv; 256]> {
    let mut f = std::fs::File::open(HV_CACHE)?;
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

// ── HDC v2 encoding ────────────────────────────────────────────────────────
struct Sig { bag: Hv, bigram: Hv, seq: Hv }

fn encode_v2(s: &str, byte_hvs: &[Hv; 256], pos_hvs: &[Hv; 256]) -> Sig {
    let norm = normalize(s);
    let unigrams: Vec<Hv> = norm.iter().map(|&b| byte_hvs[b as usize]).collect();
    let bag = bundle(&unigrams);
    let bigrams: Vec<Hv> = if norm.len() >= 2 {
        norm.windows(2).map(|w| {
            let a = byte_hvs[w[0] as usize];
            let b_rot = hv_rot1(&byte_hvs[w[1] as usize]);
            hv_xor_new(&a, &b_rot)
        }).collect()
    } else { unigrams.clone() };
    let bigram = bundle(&bigrams);
    let mut seq = [0u64; HV_U64];
    for (i, &b) in norm.iter().enumerate() {
        let pos = i.min(255);
        let bound = hv_xor_new(&byte_hvs[b as usize], &pos_hvs[pos]);
        for k in 0..HV_U64 { seq[k] ^= bound[k]; }
    }
    Sig { bag, bigram, seq }
}

fn sim_v2(a: &Sig, b: &Sig) -> f32 {
    let h_bag = hamming(&a.bag, &b.bag);
    let h_big = hamming(&a.bigram, &b.bigram);
    let h_seq = hamming(&a.seq, &b.seq);
    let s_bag = 1.0 - h_bag as f32 / HV_BITS as f32;
    let s_big = 1.0 - h_big as f32 / HV_BITS as f32;
    let s_seq = 1.0 - h_seq as f32 / HV_BITS as f32;
    0.5 * s_bag + 0.35 * s_big + 0.15 * s_seq
}

// Char-level signature (iara_byte_chat style)
use std::collections::HashMap;
fn sig_bigrams(s: &str) -> HashMap<[u8;2], f32> {
    let norm = normalize(s);
    let bs: Vec<u8> = norm.iter().filter(|&&b| b != b' ').copied().collect();
    let mut map = HashMap::new();
    for w in bs.windows(2) {
        let key = [w[0], w[1]];
        *map.entry(key).or_insert(0.0) += 1.0;
    }
    let n: f32 = map.values().sum();
    if n > 0.0 { for v in map.values_mut() { *v /= n; } }
    map
}
fn cos_bigrams(a: &HashMap<[u8;2], f32>, b: &HashMap<[u8;2], f32>) -> f32 {
    let mut dot = 0.0;
    for (k, va) in a {
        if let Some(vb) = b.get(k) { dot += va * vb; }
    }
    let na: f32 = a.values().map(|v| v*v).sum::<f32>().sqrt();
    let nb: f32 = b.values().map(|v| v*v).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 { return 0.0; }
    dot / (na * nb)
}

struct QA { user_raw: String, sig_hdc: Sig, sig_bi: HashMap<[u8;2], f32>, model: String }

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
                    let sig_hdc = encode_v2(&u, byte_hvs, pos_hvs);
                    let sig_bi = sig_bigrams(&u);
                    out.push(QA { user_raw: u, sig_hdc, sig_bi, model });
                }
            }
        }
    }
    out
}

fn main() -> std::io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA Chat Evaluation Suite                                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let byte_hvs = load_byte_hvs()?;
    let pos_hvs = build_pos_hvs();

    let mut text = String::new();
    for p in &["data/dialogues_augmented.txt", "data/qwen_sustained.txt", "data/online_learning.txt"] {
        if Path::new(p).exists() {
            text.push_str(&std::fs::read_to_string(p)?);
            text.push('\n');
        }
    }
    let qa = parse_corpus(&text, &byte_hvs, &pos_hvs);
    println!("  Corpus: {} pares\n", qa.len());

    // Test prompts (PT-BR, mix de in-corpus + novos)
    let tests: &[&str] = &[
        "oi",
        "Oi",
        "OI",
        "olá",
        "tudo bem?",
        "como vai?",
        "boa tarde",
        "obrigado",
        "tchau",
        "O que é Rust?",
        "O que é GPU?",
        "O que é Python?",
        "Quem é Pelé?",
        "Quem é Paulo Coelho?",
        "Qual a capital do Brasil?",
        "Qual a capital de São Paulo?",
        "Onde fica o Pantanal?",
        "Conte sobre Salvador",
        "feijoada",
        "açaí",
        "brigadeiro",
        "ronaldinho gaucho",   // sem match exato
        "qual a moeda do Brasil?",
        "qual a moeda do Japão?",   // fora do corpus?
        "1+1=",
        "matemática",
        "blockchain",
        "Olá, como você está?",
        "Você é uma IA?",
        "qual é o seu nome?",
        "amanhã",
        "ontem",
        "Lula é presidente?",
        "fome",
        "comida japonesa",      // semântico
        "comida brasileira",
        "saudade",
        "Que dia é hoje?",
        "Como faço café?",
        "carnaval em recife",
    ];

    println!("▸ Comparando variantes:\n");
    println!("  {:>30}  {:>10}  {:>10}  {:>10}", "QUERY", "HDC v2", "byte_chat", "vencedor");
    println!("  {}", "-".repeat(70));

    let mut wins_hdc = 0;
    let mut wins_byte = 0;
    let mut ties = 0;
    let mut total_us_hdc = 0u128;
    let mut total_us_byte = 0u128;

    for q in tests {
        // HDC v2
        let t1 = Instant::now();
        let q_sig = encode_v2(q, &byte_hvs, &pos_hvs);
        let mut best_hdc = (-1.0f32, &qa[0]);
        for item in &qa {
            let s = sim_v2(&q_sig, &item.sig_hdc);
            if s > best_hdc.0 { best_hdc = (s, item); }
        }
        let us_hdc = t1.elapsed().as_micros();

        // byte_chat-style
        let t2 = Instant::now();
        let q_bi = sig_bigrams(q);
        let q_chars: Vec<char> = String::from_utf8_lossy(&normalize(q)).chars().collect();
        let mut best_byte = (-1.0f32, &qa[0]);
        for item in &qa {
            let s_bi = cos_bigrams(&q_bi, &item.sig_bi);
            let item_chars: Vec<char> = String::from_utf8_lossy(&normalize(&item.user_raw)).chars().collect();
            let m: f32 = item_chars.iter().zip(&q_chars).take(q_chars.len().min(item_chars.len()))
                .filter(|(a, b)| a == b).count() as f32 / q_chars.len().max(item_chars.len()).max(1) as f32;
            let s = 0.7 * s_bi + 0.3 * m;
            if s > best_byte.0 { best_byte = (s, item); }
        }
        let us_byte = t2.elapsed().as_micros();

        let winner = if (best_hdc.0 - best_byte.0).abs() < 0.02 { "tie" }
                     else if best_hdc.0 > best_byte.0 { "HDC" }
                     else { "BYTE" };
        match winner {
            "HDC" => wins_hdc += 1,
            "BYTE" => wins_byte += 1,
            _ => ties += 1,
        }
        total_us_hdc += us_hdc;
        total_us_byte += us_byte;

        let q_short = if q.len() > 28 { &q[..28] } else { q };
        println!("  {:>30}  sim={:.3}  sim={:.3}  {}", q_short, best_hdc.0, best_byte.0, winner);
    }

    println!("\n  ★ HDC wins: {} | byte_chat wins: {} | ties: {}", wins_hdc, wins_byte, ties);
    println!("  ★ Avg latency: HDC v2 = {}μs, byte_chat = {}μs",
             total_us_hdc / tests.len() as u128,
             total_us_byte / tests.len() as u128);
    println!("  ★ Corpus size: {} pares", qa.len());

    Ok(())
}
