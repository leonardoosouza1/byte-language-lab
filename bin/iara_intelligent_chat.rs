//! iara_intelligent_chat.rs — chat com:
//!   1. HDC retrieval (microsegundos, base de 5000+ pares)
//!   2. Multi-turn context (history HV decai com tempo)
//!   3. Qwen fallback quando sim < threshold (graceful)
//!   4. Online learning: cada turn adiciona ao corpus
//!
//! Resposta: SE HDC.sim ≥ 0.85 → corpus match (rápido, deterministic)
//!           SENÃO → /api/generate Qwen (slow, generative)

use std::{collections::HashMap, fs, io};
use std::io::{Read, Write, BufRead};
use std::path::Path;
use std::time::Instant;

const HV_BITS: usize = 1024;
const HV_U64:  usize = HV_BITS / 64;
type Hv = [u64; HV_U64];

const HV_CACHE: &str = "data/byte_hvs_qwen.bin";
const SIM_THRESHOLD: f32 = 0.85;
const HIST_DECAY: f32 = 0.5;  // peso da última turn vs anteriores
const ENGINE: &str = "http://localhost:3000";
const ONLINE_LOG: &str = "data/online_learning.txt";

// ── Bit utilities (mesmos de v2) ──────────────────────────────────────────
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

fn hv_xor(a: &mut Hv, b: &Hv) { for i in 0..HV_U64 { a[i] ^= b[i]; } }
fn hv_xor_new(a: &Hv, b: &Hv) -> Hv {
    let mut o = [0u64; HV_U64];
    for i in 0..HV_U64 { o[i] = a[i] ^ b[i]; }
    o
}
fn hamming(a: &Hv, b: &Hv) -> u32 {
    let mut h = 0u32;
    for i in 0..HV_U64 { h += (a[i] ^ b[i]).count_ones(); }
    h
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

// Bundle Hvs com pesos (peso > 0 = mais votos)
fn weighted_bundle(weighted: &[(Hv, f32)]) -> Hv {
    if weighted.is_empty() { return [0; HV_U64]; }
    let total_w: f32 = weighted.iter().map(|(_, w)| w).sum();
    let threshold = total_w / 2.0;
    let mut counts = vec![0.0f32; HV_BITS];
    for (hv, w) in weighted {
        for word in 0..HV_U64 {
            for b in 0..64 {
                if (hv[word] >> b) & 1 == 1 { counts[word*64 + b] += w; }
            }
        }
    }
    let mut out = [0u64; HV_U64];
    for p in 0..HV_BITS {
        if counts[p] > threshold { out[p/64] |= 1u64 << (p%64); }
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

struct Sig { bag: Hv, bigram: Hv, seq: Hv }

fn encode(s: &str, byte_hvs: &[Hv; 256], pos_hvs: &[Hv; 256]) -> Sig {
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
        hv_xor(&mut seq, &bound);
    }
    Sig { bag, bigram, seq }
}

fn similarity(a: &Sig, b: &Sig) -> f32 {
    let h_bag = hamming(&a.bag, &b.bag);
    let h_big = hamming(&a.bigram, &b.bigram);
    let h_seq = hamming(&a.seq, &b.seq);
    let s_bag = 1.0 - h_bag as f32 / HV_BITS as f32;
    let s_big = 1.0 - h_big as f32 / HV_BITS as f32;
    let s_seq = 1.0 - h_seq as f32 / HV_BITS as f32;
    0.5 * s_bag + 0.35 * s_big + 0.15 * s_seq
}

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
                    let sig = encode(&u, byte_hvs, pos_hvs);
                    out.push(QA { user_raw: u, sig, model });
                }
            }
        }
    }
    out
}

// ── Qwen fallback ──────────────────────────────────────────────────────────
fn qwen_generate(prompt: &str) -> io::Result<String> {
    let body = format!(
        "{{\"model\":\"qwen2.5-1.5b\",\"prompt\":{},\"max_tokens\":80,\"temperature\":0.7,\"top_k\":40,\"top_p\":0.9}}",
        serde_json::to_string(prompt).unwrap_or_default()
    );

    let req = ureq_post(&format!("{}/api/generate", ENGINE), &body)?;
    let json: serde_json::Value = serde_json::from_str(&req)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    Ok(json["response"].as_str().unwrap_or("").replace("<|im_end|>", "").trim().to_string())
}

/// Minimal POST via std (no ureq dep)
fn ureq_post(url: &str, body: &str) -> io::Result<String> {
    use std::io::{BufRead, BufReader};
    use std::net::TcpStream;
    let prefix = "http://";
    let url_strip = url.strip_prefix(prefix).unwrap_or(url);
    let parts: Vec<&str> = url_strip.splitn(2, '/').collect();
    let host = parts[0];
    let path = if parts.len() > 1 { format!("/{}", parts[1]) } else { "/".to_string() };

    let mut stream = TcpStream::connect(host)?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(60)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(60)))?;

    let req = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        path, host, body.len(), body
    );
    stream.write_all(req.as_bytes())?;
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let mut response_body = String::new();
    let mut headers_done = false;
    let mut content_length: usize = 0;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 { break; }
        let trimmed = line.trim();
        if !headers_done {
            if trimmed.is_empty() { headers_done = true; continue; }
            if let Some(v) = trimmed.to_lowercase().strip_prefix("content-length: ") {
                content_length = v.parse().unwrap_or(0);
            }
        }
    }
    let mut buf = vec![0u8; content_length];
    reader.read_exact(&mut buf)?;
    response_body.push_str(&String::from_utf8_lossy(&buf));
    Ok(response_body)
}

fn main() -> io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA Intelligent Chat                                       ║");
    println!("║   HDC retrieval (μs) + Qwen fallback (gen) + multi-turn      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let t0 = Instant::now();
    let byte_hvs = load_byte_hvs()?;
    let pos_hvs  = build_pos_hvs();

    let mut text = String::new();
    for p in &["data/dialogues_augmented.txt", "data/qwen_sustained.txt", ONLINE_LOG] {
        if Path::new(p).exists() {
            text.push_str(&fs::read_to_string(p)?);
            text.push('\n');
        }
    }
    let mut qa = parse_corpus(&text, &byte_hvs, &pos_hvs);
    println!("  Carregado: {} pares ({:.0}ms)", qa.len(), t0.elapsed().as_millis());
    println!("  Threshold HDC: {} (abaixo → Qwen fallback)", SIM_THRESHOLD);
    println!();

    let stdin = io::stdin();
    let mut history_hvs: Vec<(Hv, f32)> = Vec::new();   // (HV, weight)
    let mut hit_corpus = 0u32;
    let mut hit_qwen = 0u32;

    print!("\x1b[1;32mUSER:\x1b[0m "); io::stdout().flush().ok();
    for line in stdin.lock().lines() {
        let line = line?;
        let s = line.trim();
        if s.is_empty() { print!("\x1b[1;32mUSER:\x1b[0m "); io::stdout().flush().ok(); continue; }
        if s == "/quit" || s == "/exit" {
            println!("Sessão: {} corpus hits, {} Qwen fallbacks", hit_corpus, hit_qwen);
            break;
        }
        if s == "/stats" {
            println!("  Corpus hits: {}", hit_corpus);
            println!("  Qwen fallbacks: {}", hit_qwen);
            println!("  History length: {}", history_hvs.len());
            print!("\x1b[1;32mUSER:\x1b[0m "); io::stdout().flush().ok();
            continue;
        }
        if s == "/reset" {
            history_hvs.clear();
            println!("[history limpo]");
            print!("\x1b[1;32mUSER:\x1b[0m "); io::stdout().flush().ok();
            continue;
        }

        let tq = Instant::now();
        let q_sig = encode(s, &byte_hvs, &pos_hvs);

        // Boost query with conversational context (history HV)
        // Bundle current query with weighted history
        let mut all_hvs: Vec<(Hv, f32)> = vec![(q_sig.bag, 1.0)];
        all_hvs.extend(history_hvs.iter().cloned());
        let ctx_hv = weighted_bundle(&all_hvs);

        let mut scored: Vec<(f32, &QA)> = qa.iter().map(|item| {
            let sim_raw = similarity(&q_sig, &item.sig);
            // Tiny boost when history HV agrees with item.bag
            let h_ctx = hamming(&ctx_hv, &item.sig.bag);
            let ctx_sim = 1.0 - h_ctx as f32 / HV_BITS as f32;
            let boosted = 0.85 * sim_raw + 0.15 * ctx_sim;
            (boosted, item)
        }).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let elapsed_hdc = tq.elapsed().as_micros();

        let (top_sim, top) = &scored[0];

        let response: String;
        let source: &str;
        if *top_sim >= SIM_THRESHOLD {
            response = top.model.clone();
            source = "corpus";
            hit_corpus += 1;
            println!("\x1b[1;34mMODEL:\x1b[0m {}", response);
            println!("\x1b[2m[{} sim={:.3} match=\"{}\" {}μs]\x1b[0m",
                     source, top_sim, top.user_raw, elapsed_hdc);
        } else {
            // Fallback Qwen
            let tq2 = Instant::now();
            response = match qwen_generate(s) {
                Ok(r) => r,
                Err(e) => format!("[erro Qwen: {}]", e),
            };
            source = "qwen";
            hit_qwen += 1;
            let elapsed_qwen = tq2.elapsed().as_millis();
            println!("\x1b[1;35mMODEL:\x1b[0m {}", response);
            println!("\x1b[2m[{} (HDC sim={:.3} < {}) {}ms]\x1b[0m",
                     source, top_sim, SIM_THRESHOLD, elapsed_qwen);

            // Online learning: add (user, qwen_response) to corpus
            let sig_new = encode(s, &byte_hvs, &pos_hvs);
            qa.push(QA { user_raw: s.to_string(), sig: sig_new, model: response.clone() });
            // Append to online log file
            let mut f = fs::OpenOptions::new().create(true).append(true).open(ONLINE_LOG)?;
            writeln!(f, "USER: {}", s)?;
            writeln!(f, "MODEL: {}", response)?;
        }

        // Update history HV
        // Decay existing weights, add new
        for (_, w) in history_hvs.iter_mut() { *w *= HIST_DECAY; }
        history_hvs.push((q_sig.bag, 1.0));
        // Truncate history to last 5 turns
        if history_hvs.len() > 5 { history_hvs.remove(0); }

        println!();
        print!("\x1b[1;32mUSER:\x1b[0m "); io::stdout().flush().ok();
    }
    Ok(())
}
