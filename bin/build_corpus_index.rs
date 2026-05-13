//! build_corpus_index.rs — pre-computa e salva HV index do corpus em binário.
//!
//! Formato:
//!   [magic u32 = HDCX] [version u32 = 1] [n_entries u32]
//!   pra cada entry:
//!     [user_len u16] [user_bytes ...]
//!     [model_len u16] [model_bytes ...]
//!     [sig.bag    128 bytes]
//!     [sig.bigram 128 bytes]
//!     [sig.seq    128 bytes]
//!
//! Saved to: data/corpus_index.bin
//! Loading: O(n) read serial vs O(n × O(encode))

use std::{collections::HashMap, fs, io};
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

const HV_BITS: usize = 1024;
const HV_U64:  usize = HV_BITS / 64;
type Hv = [u64; HV_U64];

const HV_CACHE: &str = "data/byte_hvs_qwen.bin";
const OUT_PATH: &str = "data/corpus_index.bin";
const MAGIC: u32 = 0x48444358;  // "HDCX"

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
    let v: Vec<u8> = strip_diacritics(s.as_bytes()).iter().map(|&b| {
        if b.is_ascii_alphanumeric() { casefold(b) }
        else if b.is_ascii_whitespace() { b' ' }
        else { b' ' }
    }).collect();
    let s = String::from_utf8_lossy(&v).to_string();
    s.split_whitespace().collect::<Vec<_>>().join(" ")
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
    let mut seq = [0u64; HV_U64];
    for (i, &b) in bytes.iter().enumerate() {
        let pos = i.min(255);
        let bound = hv_xor_new(&byte_hvs[b as usize], &pos_hvs[pos]);
        for k in 0..HV_U64 { seq[k] ^= bound[k]; }
    }
    Sig { bag, bigram, seq }
}

fn write_hv(f: &mut fs::File, hv: &Hv) -> io::Result<()> {
    for &w in hv.iter() { f.write_all(&w.to_le_bytes())?; }
    Ok(())
}

fn main() -> io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   build_corpus_index — pre-encodes corpus pra HV binary       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let t0 = Instant::now();
    let byte_hvs = load_byte_hvs()?;
    let pos_hvs = build_pos_hvs();

    // Load all corpus files
    let mut text = String::new();
    let mut sources = Vec::new();
    for p in &["data/dialogues_augmented.txt", "data/qwen_sustained.txt", "data/online_learning.txt"] {
        if Path::new(p).exists() {
            text.push_str(&fs::read_to_string(p)?);
            text.push('\n');
            sources.push(*p);
        }
    }
    println!("  Sources: {}", sources.join(", "));
    println!("  Raw text: {} bytes", text.len());

    // Parse pairs
    let mut pairs: Vec<(String, String)> = Vec::new();
    let mut cur_user: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(u) = line.strip_prefix("USER:") {
            cur_user = Some(u.trim().to_string());
        } else if let Some(m) = line.strip_prefix("MODEL:") {
            if let Some(u) = cur_user.take() {
                let model = m.trim().to_string();
                if !u.is_empty() && !model.is_empty() && u.len() < 65535 && model.len() < 65535 {
                    pairs.push((u, model));
                }
            }
        }
    }
    println!("  Pares parseados: {}", pairs.len());

    // Encode + write
    let mut f = fs::File::create(OUT_PATH)?;
    f.write_all(&MAGIC.to_le_bytes())?;
    f.write_all(&1u32.to_le_bytes())?;   // version
    f.write_all(&(pairs.len() as u32).to_le_bytes())?;

    let t_encode = Instant::now();
    for (u, m) in &pairs {
        let sig = encode(u, &byte_hvs, &pos_hvs);
        let u_bytes = u.as_bytes();
        let m_bytes = m.as_bytes();
        f.write_all(&(u_bytes.len() as u16).to_le_bytes())?;
        f.write_all(u_bytes)?;
        f.write_all(&(m_bytes.len() as u16).to_le_bytes())?;
        f.write_all(m_bytes)?;
        write_hv(&mut f, &sig.bag)?;
        write_hv(&mut f, &sig.bigram)?;
        write_hv(&mut f, &sig.seq)?;
    }
    let elapsed_encode = t_encode.elapsed();

    let sz = fs::metadata(OUT_PATH)?.len();
    println!("\n  Encoded em {:.0}ms", elapsed_encode.as_millis());
    println!("  Salvo: {} ({} bytes, {:.1} KB)", OUT_PATH, sz, sz as f32 / 1024.0);
    println!("  Por entry: {:.1} bytes", sz as f32 / pairs.len() as f32);
    println!("  Total wall: {:.0}ms", t0.elapsed().as_millis());

    Ok(())
}
