//! Streaming, constant-RAM corpus builder (replaces find + cat + awk-dedup +
//! python-stats). Walks source dirs, concatenates matching files with `===FILE`
//! separators, deduplicates lines by **u64 hash** (≈8 bytes/unique-line instead
//! of the full string the awk approach kept in RAM), and prints stats — all in a
//! single streaming pass with a `BufWriter`.
//!
//! Run at idle priority so it never competes with your work:
//!   nice -19 ionice -c3 ./target/release/corpus_builder OUT EXTS MAX_KB DIR...
//!
//! Args: corpus_builder <out> <exts_csv> <max_file_kb> <dir>...
//! e.g.  corpus_builder data/corpus.txt rs,py,go,ts,md 400 /tmp/codecorpus

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// FNV-1a 64-bit — fast, no deps, good enough for line dedup (64-bit collision
/// probability is negligible at corpus scale).
#[inline]
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

const EXCLUDES: &[&str] = &[
    "/.git/", "/node_modules/", "/target/", "/.next/", "/dist/", "/build/",
    "/venv/", "/.venv/", "/site-packages/", "/__pycache__/", "/vendor/",
    "/ollama/", "llama.cpp",
];

struct Stats {
    files: u64,
    lines_in: u64,
    lines_kept: u64,
    bytes: u64,
    byte_counts: Box<[u64; 256]>,
}

fn walk(dir: &Path, exts: &[String], max_bytes: u64,
        seen: &mut HashSet<u64>, out: &mut impl Write, st: &mut Stats, root_len: usize) {
    let rd = match fs::read_dir(dir) { Ok(r) => r, Err(_) => return };
    for entry in rd.flatten() {
        let path = entry.path();
        let ps = path.to_string_lossy();
        if EXCLUDES.iter().any(|e| ps.contains(e)) { continue; }
        let ft = match entry.file_type() { Ok(t) => t, Err(_) => continue };
        if ft.is_dir() {
            walk(&path, exts, max_bytes, seen, out, st, root_len);
        } else if ft.is_file() {
            let ext_ok = path.extension()
                .and_then(|e| e.to_str())
                .map(|e| exts.iter().any(|x| x == e))
                .unwrap_or(false);
            if !ext_ok { continue; }
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.contains(".min.") { continue; }
            let meta = match entry.metadata() { Ok(m) => m, Err(_) => continue };
            if meta.len() > max_bytes { continue; }
            let f = match File::open(&path) { Ok(f) => f, Err(_) => continue };
            let _ = root_len;
            // 1-byte document separator (0x00): a clean boundary signal the model
            // learns as one byte, instead of the verbose `===FILE path===` line
            // (whose paths are noise the model wasted capacity memorizing).
            let _ = out.write_all(&[0u8]);
            st.byte_counts[0] += 1;
            st.bytes += 1;
            st.files += 1;
            // stream lines, dedup by hash
            let reader = BufReader::new(f);
            for line in reader.split(b'\n') {
                let mut line = match line { Ok(l) => l, Err(_) => break };
                st.lines_in += 1;
                let h = fnv1a(&line);
                if !seen.insert(h) { continue; }   // duplicate → skip
                line.push(b'\n');
                let _ = out.write_all(&line);
                for &c in &line { st.byte_counts[c as usize] += 1; }
                st.bytes += line.len() as u64;
                st.lines_kept += 1;
            }
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: corpus_builder <out> <exts_csv> <max_file_kb> <dir>...");
        std::process::exit(1);
    }
    let out_path = &a[1];
    let exts: Vec<String> = a[2].split(',').map(|s| s.to_string()).collect();
    let max_bytes: u64 = a[3].parse::<u64>().unwrap_or(400) * 1024;
    let dirs = &a[4..];

    let mut seen: HashSet<u64> = HashSet::new();
    let file = File::create(out_path).unwrap_or_else(|e| panic!("create {out_path}: {e}"));
    let mut out = BufWriter::with_capacity(1 << 20, file);   // 1MB buffer
    let mut st = Stats { files: 0, lines_in: 0, lines_kept: 0, bytes: 0, byte_counts: Box::new([0u64; 256]) };

    for d in dirs {
        let p = Path::new(d);
        let root_len = d.trim_end_matches('/').len() + 1;
        walk(p, &exts, max_bytes, &mut seen, &mut out, &mut st, root_len.min(d.len()));
    }
    out.flush().unwrap();

    // unigram entropy (the corpus "floor") — constant RAM
    let n = st.bytes as f64;
    let entropy: f64 = st.byte_counts.iter().filter(|&&c| c > 0)
        .map(|&c| { let p = c as f64 / n; -p * p.log2() }).sum();
    let uniq_pct = if st.lines_in > 0 { st.lines_kept * 100 / st.lines_in } else { 0 };
    println!("=== corpus_builder ===");
    println!("out: {out_path}");
    println!("files: {}", st.files);
    println!("lines: {} kept / {} read ({}% unique)", st.lines_kept, st.lines_in, uniq_pct);
    println!("size: {:.1} MB", st.bytes as f64 / 1e6);
    println!("unigram floor: {entropy:.3} bits/byte");
    println!("dedup RAM: ~{:.1} MB (u64 hashes)", seen.len() as f64 * 8.0 / 1e6);
}
