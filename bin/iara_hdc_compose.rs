//! iara_hdc_compose.rs — testa composicionalidade via bundle de HVs.
//!
//! Tese: HV(food) + HV(brasileiro) ≈ HV(brigadeiro)?
//!
//! Usa byte HVs Qwen pra encode word → HV. Bundle (majority vote) compõe.

use std::{fs, io};
use std::io::Read;
use std::path::Path;

const HV_BITS: usize = 1024;
const HV_U64:  usize = HV_BITS / 64;
type Hv = [u64; HV_U64];

const HV_CACHE: &str = "data/byte_hvs_qwen.bin";

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
                _ => { i += 2; continue; }
            };
            out.push(m); i += 2;
        } else { out.push(bytes[i]); i += 1; }
    }
    out
}
fn normalize(s: &str) -> Vec<u8> {
    strip_diacritics(s.as_bytes()).iter().map(|&b| {
        if b.is_ascii_alphanumeric() { casefold(b) } else { b' ' }
    }).filter(|&b| b != b' ').collect()
}

fn hamming(a: &Hv, b: &Hv) -> u32 {
    let mut h = 0u32;
    for i in 0..HV_U64 { h += (a[i] ^ b[i]).count_ones(); }
    h
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

fn word_hv(w: &str, byte_hvs: &[Hv; 256]) -> Hv {
    let n = normalize(w);
    if n.is_empty() { return [0; HV_U64]; }
    let hvs: Vec<Hv> = n.iter().map(|&b| byte_hvs[b as usize]).collect();
    bundle(&hvs)
}

fn extract_vocab(text: &str) -> Vec<String> {
    let mut set: std::collections::HashSet<String> = std::collections::HashSet::new();
    for line in text.lines() {
        for word in line.split(|c: char| !c.is_alphabetic()) {
            let w = word.trim().to_lowercase();
            if w.chars().count() >= 3 && w.chars().count() <= 20 {
                set.insert(w);
            }
        }
    }
    let mut v: Vec<String> = set.into_iter().collect();
    v.sort();
    v
}

fn main() -> io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA HDC Compose — bundle de HVs para composição           ║");
    println!("║   Test: HV(A) + HV(B) ≈ HV(C composto)?                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let byte_hvs = load_byte_hvs()?;
    let mut text = String::new();
    for p in &["data/dialogues_augmented.txt", "data/qwen_sustained.txt"] {
        if Path::new(p).exists() {
            text.push_str(&fs::read_to_string(p)?);
            text.push('\n');
        }
    }
    let vocab = extract_vocab(&text);
    println!("  Vocabulário: {} palavras únicas", vocab.len());

    let word_hvs: Vec<(String, Hv)> = vocab.iter()
        .map(|w| (w.clone(), word_hv(w, &byte_hvs)))
        .collect();
    println!("  HVs computadas: {}\n", word_hvs.len());

    // Test composições
    let tests: &[(&[&str], &str)] = &[
        // (componentes, esperado_composto)
        (&["doce", "brasileiro"], "brigadeiro"),
        (&["comida", "tipica"], "feijoada"),
        (&["capital", "brasil"], "brasilia"),
        (&["cidade", "histórica"], "ouro preto"),
        (&["esporte", "popular"], "futebol"),
        (&["jogador", "futebol"], "pele"),
        (&["fruta", "amazonia"], "acai"),
        (&["dança", "brasileira"], "samba"),
        (&["linguagem", "programação"], "rust"),
        (&["sistema", "operacional"], "linux"),
    ];

    println!("▸ Testando composição (bundle de HVs → nearest no vocab):\n");
    let mut hits_top1 = 0;
    let mut hits_top10 = 0;
    let mut total = 0;
    for (parts, expected) in tests {
        let part_hvs: Vec<Hv> = parts.iter().map(|p| word_hv(p, &byte_hvs)).collect();
        let composed = bundle(&part_hvs);

        // Find nearest, excluding the input components
        let mut nearest: Vec<(u32, &String)> = word_hvs.iter()
            .filter(|(w, _)| !parts.iter().any(|p| w.to_lowercase() == p.to_lowercase()))
            .map(|(w, hv)| (hamming(&composed, hv), w))
            .collect();
        nearest.sort_by_key(|&(h, _)| h);
        let top5_str: Vec<String> = nearest.iter().take(5)
            .map(|(h, w)| {
                let sim = 1.0 - *h as f32 / HV_BITS as f32;
                format!("{}({:.2})", w, sim)
            }).collect();

        let exp_norm = expected.to_lowercase().replace(" ", "");
        let rank = nearest.iter().position(|(_, w)| {
            w.to_lowercase().contains(&exp_norm) || exp_norm.contains(&w.to_lowercase())
        });
        let r_str = match rank {
            Some(r) if r == 0 => { hits_top1 += 1; hits_top10 += 1; "#1 ★".to_string() }
            Some(r) if r < 10 => { hits_top10 += 1; format!("#{}", r + 1) }
            Some(r) => format!("#{}", r + 1),
            None => "miss".to_string(),
        };
        total += 1;
        println!("  [{:?}] → top5={}  esperava: {} ({})",
            parts.join(" + "), top5_str.join(" "), expected, r_str);
    }
    println!("\n  ★ Top-1 hits: {}/{} ({:.0}%)", hits_top1, total, 100.0*hits_top1 as f32/total as f32);
    println!("  ★ Top-10 hits: {}/{} ({:.0}%)", hits_top10, total, 100.0*hits_top10 as f32/total as f32);

    Ok(())
}
