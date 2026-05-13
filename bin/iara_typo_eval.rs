//! iara_typo_eval.rs — testa robustez a typos: HV Qwen-only vs HV fonético.
//!
//! Pares (query_correta, query_com_typo, expected_in_corpus):
//!   - "feijoada" vs "fejoada" → expected match "feijoada"
//!   - "obrigado" vs "obirgado" → expected match "obrigado"
//!   - etc.
//!
//! Compara: HV vanilla (sign Qwen) vs HV fonético (256 fonético + 768 Qwen).

use std::{collections::HashMap, fs, io};
use std::io::Read;
use std::path::Path;

const HV_BITS: usize = 1024;
const HV_U64:  usize = HV_BITS / 64;
type Hv = [u64; HV_U64];

const HV_QWEN: &str   = "data/byte_hvs_qwen.bin";
const HV_PHON: &str   = "data/byte_hvs_phonetic.bin";

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
    strip_diacritics(s.as_bytes()).iter().map(|&b| {
        if b.is_ascii_alphanumeric() { casefold(b) }
        else { b' ' }
    }).filter(|&b| b != b' ').collect()
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
    Sig { bag, bigram }
}

fn sim(a: &Sig, b: &Sig) -> f32 {
    let h_bag = hamming(&a.bag, &b.bag);
    let h_big = hamming(&a.bigram, &b.bigram);
    let s_bag = 1.0 - h_bag as f32 / HV_BITS as f32;
    let s_big = 1.0 - h_big as f32 / HV_BITS as f32;
    0.6 * s_bag + 0.4 * s_big
}

struct QA { user_raw: String, sig: Sig }

fn parse_corpus(text: &str, byte_hvs: &[Hv; 256]) -> Vec<QA> {
    let mut out = Vec::new();
    let mut cur: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(u) = line.strip_prefix("USER:") {
            cur = Some(u.trim().to_string());
        } else if let Some(_m) = line.strip_prefix("MODEL:") {
            if let Some(u) = cur.take() {
                if !u.is_empty() {
                    let sig = encode(&u, byte_hvs);
                    out.push(QA { user_raw: u, sig });
                }
            }
        }
    }
    out
}

fn main() -> io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA Typo Eval — robustez: Qwen-HV vs Phonetic-HV           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let hvs_q = load_hvs(HV_QWEN)?;
    let hvs_p = load_hvs(HV_PHON)?;
    println!("  HVs carregadas (Qwen vanilla, Phonetic enriched)");

    let mut text = String::new();
    for p in &["data/dialogues_augmented.txt", "data/qwen_sustained.txt"] {
        if Path::new(p).exists() {
            text.push_str(&fs::read_to_string(p)?);
            text.push('\n');
        }
    }
    let qa_q = parse_corpus(&text, &hvs_q);
    let qa_p = parse_corpus(&text, &hvs_p);
    println!("  Corpus: {} pares\n", qa_q.len());

    // Test cases: (clean, typo_versions, expected_corpus_user_substring)
    let tests: &[(&str, &[&str], &str)] = &[
        ("oi", &["Oi", "OI", "oii", "oie", "iio"], "Oi"),
        ("olá", &["ola", "olahh", "olla", "olhá"], "Olá"),
        ("obrigado", &["obirgado", "obrigao", "obrgiado", "brigado"], "obrigado"),
        ("feijoada", &["fejoada", "feijoa", "feijoda", "fejoda"], "feijoada"),
        ("brigadeiro", &["briagdeiro", "brigadero", "brigaderro"], "brigadeiro"),
        ("Pelé", &["Pele", "Peley", "Pleé", "Pellé"], "Pelé"),
        ("Brasil", &["Brazil", "Brasl", "Brsil", "Bralis"], "Brasil"),
        ("samba", &["sanba", "smaba", "samaba"], "samba"),
        ("matemática", &["matematica", "matmatica", "matemtica"], "matemática"),
        ("café", &["cafe", "caff", "caffe"], "café"),
    ];

    let mut wins_q = 0;
    let mut wins_p = 0;
    let mut ties = 0;
    let mut found_q_total = 0;
    let mut found_p_total = 0;
    let total_tests: usize = tests.iter().map(|(_, t, _)| t.len()).sum();

    println!("{:>15} → {:>15}  Qwen-HV    Phonetic-HV  match expected? (Q | P)", "ORIGINAL", "TYPO");
    println!("{}", "-".repeat(85));

    for (clean, typos, expected) in tests {
        for typo in *typos {
            let sig_q = encode(typo, &hvs_q);
            let sig_p = encode(typo, &hvs_p);

            // Best match Qwen HV
            let mut best_q = (-1.0f32, &qa_q[0]);
            for item in &qa_q {
                let s = sim(&sig_q, &item.sig);
                if s > best_q.0 { best_q = (s, item); }
            }
            // Best match Phonetic HV
            let mut best_p = (-1.0f32, &qa_p[0]);
            for item in &qa_p {
                let s = sim(&sig_p, &item.sig);
                if s > best_p.0 { best_p = (s, item); }
            }

            let q_hit = best_q.1.user_raw.to_lowercase().contains(&expected.to_lowercase());
            let p_hit = best_p.1.user_raw.to_lowercase().contains(&expected.to_lowercase());
            if q_hit { found_q_total += 1; }
            if p_hit { found_p_total += 1; }
            let winner = match (q_hit, p_hit) {
                (true, true) => { ties += 1; "tie" },
                (true, false) => { wins_q += 1; "Q" },
                (false, true) => { wins_p += 1; "P" },
                (false, false) => "both wrong",
            };
            let q_short: String = best_q.1.user_raw.chars().take(18).collect();
            let p_short: String = best_p.1.user_raw.chars().take(18).collect();
            println!("{:>15} → {:>15}  {:>5.3}/{:<18}  {:>5.3}/{:<18}  {}/{} ({})",
                clean, typo, best_q.0, q_short, best_p.0, p_short,
                if q_hit { "✓" } else { "✗" },
                if p_hit { "✓" } else { "✗" }, winner);
        }
    }

    println!("\n  ★ Total typos tested: {}", total_tests);
    println!("  ★ Qwen-HV correctly matched: {}/{} ({:.0}%)",
             found_q_total, total_tests, 100.0 * found_q_total as f32 / total_tests as f32);
    println!("  ★ Phonetic-HV correctly matched: {}/{} ({:.0}%)",
             found_p_total, total_tests, 100.0 * found_p_total as f32 / total_tests as f32);
    println!("  ★ Wins: Qwen={} | Phonetic={} | tie={}", wins_q, wins_p, ties);
    Ok(())
}
