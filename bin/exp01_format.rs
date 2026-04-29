//! Experimento 01 — Formato byte: round-trip & compactness comparison
//!
//! Para cada path (A/B/C), mede:
//!   - taxa de erro de round-trip em 100K números aleatórios
//!   - bytes médios por número (compactness)
//!   - throughput de encode/decode (ns/op)
//!
//! Compara com f64 raw (8 bytes constante).

use blab::format::{path_a, path_b, path_c};
use blab::Rng;
use std::io::Write;
use std::time::Instant;

const N: usize = 100_000;
const SEED: u64 = 9001;

#[derive(Debug)]
struct PathStats {
    name: &'static str,
    encoded_ok: usize,
    decoded_ok: usize,
    roundtrip_match: usize,
    total_bytes: u64,
    encode_ns: u128,
    decode_ns: u128,
}

fn run_path_a(values: &[f64]) -> PathStats {
    let mut s = PathStats {
        name: "Path A (positional)",
        encoded_ok: 0, decoded_ok: 0, roundtrip_match: 0,
        total_bytes: 0, encode_ns: 0, decode_ns: 0,
    };
    let mut all_bytes = Vec::with_capacity(values.len());
    let t = Instant::now();
    for &v in values {
        if let Ok(b) = path_a::encode(v) {
            s.total_bytes += b.len() as u64;
            s.encoded_ok += 1;
            all_bytes.push(Some(b));
        } else { all_bytes.push(None); }
    }
    s.encode_ns = t.elapsed().as_nanos();

    let t = Instant::now();
    for (i, opt) in all_bytes.iter().enumerate() {
        if let Some(b) = opt {
            if let Ok(d) = path_a::decode(b) {
                s.decoded_ok += 1;
                if path_a::path_a_tolerance(values[i], d) {
                    s.roundtrip_match += 1;
                }
            }
        }
    }
    s.decode_ns = t.elapsed().as_nanos();
    s
}

fn run_path_b(values: &[f64]) -> PathStats {
    let mut s = PathStats {
        name: "Path B (BCD ordered)",
        encoded_ok: 0, decoded_ok: 0, roundtrip_match: 0,
        total_bytes: 0, encode_ns: 0, decode_ns: 0,
    };
    let mut all_bytes = Vec::with_capacity(values.len());
    let t = Instant::now();
    for &v in values {
        if let Ok(b) = path_b::encode(v) {
            s.total_bytes += b.len() as u64;
            s.encoded_ok += 1;
            all_bytes.push(Some(b));
        } else { all_bytes.push(None); }
    }
    s.encode_ns = t.elapsed().as_nanos();

    let t = Instant::now();
    for (i, opt) in all_bytes.iter().enumerate() {
        if let Some(b) = opt {
            if let Ok(d) = path_b::decode(b) {
                s.decoded_ok += 1;
                let v = values[i];
                let abs = (v - d).abs();
                let ok = if v == 0.0 { abs < 1e-12 } else { abs < 1e-12 || abs / v.abs() < 1e-9 };
                if ok { s.roundtrip_match += 1; }
            }
        }
    }
    s.decode_ns = t.elapsed().as_nanos();
    s
}

fn run_path_c(values: &[f64]) -> PathStats {
    let mut s = PathStats {
        name: "Path C (domain-tagged)",
        encoded_ok: 0, decoded_ok: 0, roundtrip_match: 0,
        total_bytes: 0, encode_ns: 0, decode_ns: 0,
    };
    let mut all_bytes = Vec::with_capacity(values.len());
    let t = Instant::now();
    for &v in values {
        if let Ok(b) = path_c::encode_number(v) {
            s.total_bytes += b.len() as u64;
            s.encoded_ok += 1;
            all_bytes.push(Some(b));
        } else { all_bytes.push(None); }
    }
    s.encode_ns = t.elapsed().as_nanos();

    let t = Instant::now();
    for (i, opt) in all_bytes.iter().enumerate() {
        if let Some(b) = opt {
            if let Ok(blocks) = path_c::decode_stream(b) {
                if let Some(blab::format::path_c::Block::Number(d)) = blocks.first() {
                    s.decoded_ok += 1;
                    if path_a::path_a_tolerance(values[i], *d) {
                        s.roundtrip_match += 1;
                    }
                }
            }
        }
    }
    s.decode_ns = t.elapsed().as_nanos();
    s
}

fn main() -> std::io::Result<()> {
    println!("[exp01] Generating {N} values (seed {SEED})...");
    let mut rng = Rng::new(SEED);
    let mut values = Vec::with_capacity(N);
    for _ in 0..N {
        let scale = (rng.range_u32(7) as i32) - 3;
        let raw = rng.range_f64(-1.0, 1.0) * (10f64).powi(scale);
        values.push(raw);
    }

    println!("[exp01] Running Path A...");
    let a = run_path_a(&values);
    println!("[exp01] Running Path B...");
    let b = run_path_b(&values);
    println!("[exp01] Running Path C...");
    let c = run_path_c(&values);

    // f64 raw baseline
    let t = Instant::now();
    let mut sink = 0.0_f64;
    for &v in &values {
        sink += f64::from_bits(v.to_bits());
    }
    std::hint::black_box(sink);
    let f64_ns = t.elapsed().as_nanos();

    let mut out = std::fs::File::create("results/exp01_format.md")?;
    writeln!(out, "# Experimento 01 — Formato byte: round-trip & compactness\n")?;
    writeln!(out, "**N samples:** {N}  ")?;
    writeln!(out, "**Seed:** {SEED}  ")?;
    writeln!(out, "**Range:** valores aleatórios em escalas 10^[-3..+3]\n")?;
    writeln!(out, "## Resultados consolidados\n")?;
    writeln!(out, "| Path | Encode OK | Decode OK | Round-trip Match | Avg bytes | Encode (ns/op) | Decode (ns/op) |")?;
    writeln!(out, "|------|----------:|----------:|-----------------:|----------:|---------------:|---------------:|")?;
    for s in [&a, &b, &c] {
        let avg = s.total_bytes as f64 / s.encoded_ok.max(1) as f64;
        let enc = s.encode_ns as f64 / s.encoded_ok.max(1) as f64;
        let dec = s.decode_ns as f64 / s.decoded_ok.max(1) as f64;
        writeln!(out, "| {} | {} | {} | {} | {:.2} | {:.0} | {:.0} |",
            s.name, s.encoded_ok, s.decoded_ok, s.roundtrip_match, avg, enc, dec)?;
    }
    writeln!(out, "| f64 raw (baseline) | {N} | {N} | {N} | 8.00 | ~{:.0} | ~{:.0} |",
        f64_ns as f64 / N as f64 / 2.0, f64_ns as f64 / N as f64 / 2.0)?;

    writeln!(out, "\n## Análise\n")?;
    writeln!(out, "- **Path A** ({:.2} B/num): BCD posição explícita. Operações byte-wise viáveis sem FP-ALU.", a.total_bytes as f64 / a.encoded_ok.max(1) as f64)?;
    writeln!(out, "- **Path B** ({:.2} B/num): BCD ordenado, posição implícita pela sequência. Decode stateful.", b.total_bytes as f64 / b.encoded_ok.max(1) as f64)?;
    writeln!(out, "- **Path C** ({:.2} B/num): header overhead, mas único multi-domain.", c.total_bytes as f64 / c.encoded_ok.max(1) as f64)?;
    writeln!(out, "- **f64 raw**: 8 bytes constante, opaco.")?;
    writeln!(out, "\n**Conclusão:** Bytes estruturados pagam ~{:.1}× mais memória que f64 raw — mas ganham interpretabilidade posicional e suportam operações sem FP-ALU.",
        (a.total_bytes as f64 / a.encoded_ok.max(1) as f64) / 8.0)?;

    println!("[exp01] Done.");
    Ok(())
}
