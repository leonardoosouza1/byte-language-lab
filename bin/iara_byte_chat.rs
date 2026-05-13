//! iara_byte_chat.rs — chat com INTELIGÊNCIA bit-level, sem rede neural.
//!
//! Tese aplicada (validada em sessões anteriores):
//!   - bit5 do byte = case (KL=0.003 entre o↔O) → normalize antes de comparar
//!   - bit4 = grupo ASCII (vogal/cons/símbolo)
//!   - bits 3-0 = posição no grupo
//!   - Hamming na codificação semântica ↔ similaridade
//!   - byte 0xC3 = prefixo UTF-8 estrutural (multi-byte)
//!
//! Pipeline:
//!   1. Lê corpus augmentado (USER/MODEL pairs)
//!   2. Indexa cada USER como "signature" (case-fold + group + position)
//!   3. Input do user → mesma signature → byte-Hamming distance pra cada
//!   4. Retorna MODEL do par mais similar (k-NN k=1)
//!
//! Resultado: chat "compreensivo" via geometria bit, ZERO inferência.

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::time::Instant;

const CORPUS: &str = "data/dialogues_augmented.txt";

// ── Bit-level hacks ─────────────────────────────────────────────────────────

/// Normaliza byte ASCII: zera bit5 (case-fold). Preserva resto.
#[inline]
fn casefold(b: u8) -> u8 {
    // ASCII alfa: 'A'..='Z' = 0x41..0x5A, 'a'..='z' = 0x61..0x7A
    // bit5 distingue case nos alfa. Zerar bit5 → minúscula virtual.
    if b.is_ascii_alphabetic() { b | 0x20 } else { b }
}

/// Strip diacríticos UTF-8 PT comuns: á→a, é→e, ç→c, etc.
/// Remove byte 0xC3 e mapeia o byte seguinte pra ASCII equivalente.
fn strip_diacritics(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0xC3 && i + 1 < bytes.len() {
            let next = bytes[i+1];
            let ascii = match next {
                // á â ã ä à
                0xA0..=0xA5 => b'a',
                // é ê ë è
                0xA8..=0xAB => b'e',
                // í î ï ì
                0xAC..=0xAF => b'i',
                // ó ô õ ö ò
                0xB2..=0xB6 => b'o',
                // ú û ü ù
                0xB9..=0xBC => b'u',
                // ç
                0xA7        => b'c',
                // Á Â Ã Ä À
                0x80..=0x85 => b'A',
                0x87        => b'C',
                0x88..=0x8B => b'E',
                0x8C..=0x8F => b'I',
                0x92..=0x96 => b'O',
                0x99..=0x9C => b'U',
                _ => { i += 2; continue; }
            };
            out.push(ascii);
            i += 2;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    out
}

/// Tokeniza-mas-byte-aware: lowercase + strip diacritic + keep só alfanum+espaço.
fn signature(s: &str) -> String {
    let bytes = s.as_bytes();
    let ascii = strip_diacritics(bytes);
    let folded: Vec<u8> = ascii.iter().map(|&b| {
        if b.is_ascii_alphanumeric() { casefold(b) }
        else if b.is_ascii_whitespace() { b' ' }
        else { b' ' }   // pontuação → espaço
    }).collect();
    // colapsa espaços
    let s = String::from_utf8_lossy(&folded).into_owned();
    let mut out = String::with_capacity(s.len());
    let mut prev_sp = false;
    for c in s.chars() {
        if c == ' ' {
            if !prev_sp && !out.is_empty() { out.push(' '); prev_sp = true; }
        } else {
            out.push(c); prev_sp = false;
        }
    }
    out.trim().to_string()
}

/// Bag-of-bigrams (char bigrams) com pesos. Captura ordem local.
fn signature_bigrams(s: &str) -> HashMap<[u8; 2], f32> {
    let sig = signature(s);
    let bs = sig.as_bytes();
    let mut map = HashMap::new();
    for w in bs.windows(2) {
        let key = [w[0], w[1]];
        *map.entry(key).or_insert(0.0) += 1.0;
    }
    // Normaliza
    let n: f32 = map.values().sum();
    if n > 0.0 {
        for v in map.values_mut() { *v /= n; }
    }
    map
}

/// Cosine similarity entre dois bigram histograms.
fn cosine(a: &HashMap<[u8;2], f32>, b: &HashMap<[u8;2], f32>) -> f32 {
    let mut dot = 0.0f32;
    for (k, va) in a {
        if let Some(vb) = b.get(k) { dot += va * vb; }
    }
    let na: f32 = a.values().map(|v| v*v).sum::<f32>().sqrt();
    let nb: f32 = b.values().map(|v| v*v).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 { return 0.0; }
    dot / (na * nb)
}

// ── Parse corpus ───────────────────────────────────────────────────────────
struct QA {
    user_raw:    String,
    user_sig:    HashMap<[u8;2], f32>,
    user_chars:  Vec<char>,    // pra hamming char-level
    model:       String,
}

fn parse_corpus(text: &str) -> Vec<QA> {
    let mut out = Vec::new();
    let mut cur_user: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(u) = line.strip_prefix("USER:") {
            cur_user = Some(u.trim().to_string());
        } else if let Some(m) = line.strip_prefix("MODEL:") {
            if let Some(user) = cur_user.take() {
                let model = m.trim().to_string();
                if !user.is_empty() && !model.is_empty() {
                    let sig = signature_bigrams(&user);
                    let chars: Vec<char> = signature(&user).chars().collect();
                    out.push(QA { user_raw: user, user_sig: sig, user_chars: chars, model });
                }
            }
        }
    }
    out
}

// ── Char-level Hamming-ish (Levenshtein-light) ─────────────────────────────
fn char_match(a: &[char], b: &[char]) -> f32 {
    let max_len = a.len().max(b.len()).max(1);
    let min_len = a.len().min(b.len());
    let mut matches = 0;
    for i in 0..min_len {
        if a[i] == b[i] { matches += 1; }
    }
    matches as f32 / max_len as f32
}

fn main() -> io::Result<()> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   IARA Byte Chat — retrieval via bit-level signature       ║");
    println!("║   tese: bit5=case, bit4=grupo, 0xC3=struct UTF-8           ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    let t0 = Instant::now();
    let text = std::fs::read_to_string(CORPUS)?;
    let qa = parse_corpus(&text);
    println!("  Corpus: {} pares USER/MODEL ({:.0} KB)", qa.len(), text.len() as f32 / 1024.0);
    println!("  Indexação: {:.0} ms\n", t0.elapsed().as_millis());

    println!("Comandos: /quit  /show N (mostra top-N matches)");
    println!();

    let stdin = io::stdin();
    let show_n_default = 1;
    let mut show_n = show_n_default;

    for line in stdin.lock().lines() {
        let line = line?;
        let cmd = line.trim();
        if cmd.is_empty() { continue; }
        if cmd == "/quit" || cmd == "/exit" { break; }
        if let Some(rest) = cmd.strip_prefix("/show ") {
            show_n = rest.parse().unwrap_or(show_n_default);
            println!("[show_n = {}]", show_n);
            continue;
        }

        let t_q = Instant::now();
        let q_sig = signature_bigrams(cmd);
        let q_chars: Vec<char> = signature(cmd).chars().collect();

        // Score = 0.7 * cosine bigrams + 0.3 * char_match
        let mut scored: Vec<(f32, &QA)> = qa.iter().map(|item| {
            let s_bi = cosine(&q_sig, &item.user_sig);
            let s_ch = char_match(&q_chars, &item.user_chars);
            let score = 0.7 * s_bi + 0.3 * s_ch;
            (score, item)
        }).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let elapsed_us = t_q.elapsed().as_micros();
        let top = &scored[0];
        println!("\x1b[1;34mMODEL:\x1b[0m {}", top.1.model);
        println!("\x1b[2m[match: \"{}\" (score={:.3}, {}μs)]\x1b[0m", top.1.user_raw, top.0, elapsed_us);

        if show_n > 1 {
            println!("\x1b[2m  alternativas:\x1b[0m");
            for (sc, q) in scored.iter().take(show_n).skip(1) {
                println!("\x1b[2m    {:.3}  USER: {}  →  MODEL: {}\x1b[0m",
                    sc, q.user_raw, &q.model[..q.model.len().min(60)]);
            }
        }
        println!();
        print!("\x1b[1;32mUSER:\x1b[0m ");
        io::stdout().flush().ok();
    }
    Ok(())
}
