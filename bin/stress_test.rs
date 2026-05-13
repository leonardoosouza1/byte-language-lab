//! stress_test.rs — stress test do HDC server.
//!
//! Manda 1000 queries randômicas (extraídas + variações typo) via HTTP.
//! Calcula: throughput, latency p50/p95/p99, accuracy.

use std::io::{Read, Write, BufRead, BufReader};
use std::net::TcpStream;
use std::time::Instant;

const SERVER: &str = "127.0.0.1:3010";

fn json_extract_num(body: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{}\"", key);
    let pos = body.find(&pat)?;
    let after = &body[pos + pat.len()..];
    let colon = after.find(':')?;
    let after = after[colon+1..].trim_start();
    let end = after.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')?;
    after[..end].parse().ok()
}

fn ask(prompt: &str) -> Result<(f64, u128), Box<dyn std::error::Error>> {
    let mut stream = TcpStream::connect(SERVER)?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
    let body_payload = format!("{{\"prompt\":\"{}\"}}", prompt.replace('"', "\\\""));
    let req = format!(
        "POST /chat HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        SERVER, body_payload.len(), body_payload
    );
    stream.write_all(req.as_bytes())?;
    let mut reader = BufReader::new(stream);
    let mut content_length = 0;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 { break; }
        let t = line.trim();
        if t.is_empty() { break; }
        if let Some(v) = t.to_lowercase().strip_prefix("content-length: ") {
            content_length = v.parse().unwrap_or(0);
        }
    }
    let mut buf = vec![0u8; content_length];
    reader.read_exact(&mut buf)?;
    let body_str = String::from_utf8_lossy(&buf).to_string();
    let sim = json_extract_num(&body_str, "sim").unwrap_or(0.0);
    let lat_us = json_extract_num(&body_str, "latency_us").unwrap_or(0.0) as u128;
    Ok((sim, lat_us))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA Stress Test — 1000 queries via HTTP                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Pool de queries — mix diverso
    let pool = vec![
        "oi", "olá", "tudo bem", "como vai", "boa tarde", "obrigado", "tchau",
        "O que é Rust?", "O que é Python?", "O que é GPU?", "O que é AMD?",
        "Quem é Pelé?", "Quem é Paulo Coelho?", "Quem é Senna?",
        "Qual a capital do Brasil?", "Qual a capital de São Paulo?",
        "Onde fica o Pantanal?", "Onde fica a Amazônia?", "Onde fica Salvador?",
        "Conte sobre Salvador", "Conte sobre Brasília", "Conte sobre Recife",
        "feijoada", "açaí", "brigadeiro", "pão de queijo",
        "samba", "bossa nova", "MPB", "carnaval",
        "Lula", "Bolsonaro", "Tiradentes",
        "futebol", "Corinthians", "Flamengo",
        "Tropicália", "Drummond", "Machado de Assis",
        "Rust", "Python", "JavaScript", "Linux",
        "amor", "saudade", "alegria",
        // Variantes
        "Oi", "OI", "OII", "oie",
        "fejoada", "feijoda", "feijoa",
        "obirgado", "obrigao",
        "qual a moeda do Brasil?", "qual a moeda do Japão?",
        // Out-of-corpus
        "blockchain", "smart contract", "metaverse", "AI safety",
        "qual a temperatura do sol?",
        // Adversarial
        "!!!!", "asdfg", "🇧🇷", "🎵", "@@@@",
    ];

    let n_queries = 1000usize;
    let mut latencies = Vec::with_capacity(n_queries);
    let mut sims = Vec::with_capacity(n_queries);
    let mut rng: u64 = 0xDEADBEEF;
    let mut errors = 0;

    let t0 = Instant::now();
    for _ in 0..n_queries {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let idx = (rng as usize) % pool.len();
        let query = pool[idx];
        match ask(query) {
            Ok((sim, lat_us)) => {
                latencies.push(lat_us);
                sims.push(sim);
            }
            Err(_) => errors += 1,
        }
    }
    let total = t0.elapsed();

    latencies.sort();
    let n = latencies.len() as f64;
    let p50 = latencies[(n * 0.50) as usize];
    let p95 = latencies[(n * 0.95) as usize];
    let p99 = latencies[(n * 0.99) as usize];
    let avg = latencies.iter().sum::<u128>() / latencies.len() as u128;
    let max = *latencies.last().unwrap();

    let avg_sim: f64 = sims.iter().sum::<f64>() / sims.len() as f64;
    let frac_high = sims.iter().filter(|&&s| s >= 0.85).count() as f64 / sims.len() as f64;
    let frac_low = sims.iter().filter(|&&s| s < 0.5).count() as f64 / sims.len() as f64;

    println!("Resultados:");
    println!("  Queries: {}", n_queries);
    println!("  Errors: {}", errors);
    println!("  Wall time: {:.2}s", total.as_secs_f64());
    println!("  Throughput: {:.1} req/s", n_queries as f64 / total.as_secs_f64());
    println!();
    println!("Latency (server-side μs):");
    println!("  avg: {}μs", avg);
    println!("  p50: {}μs", p50);
    println!("  p95: {}μs", p95);
    println!("  p99: {}μs", p99);
    println!("  max: {}μs", max);
    println!();
    println!("Quality:");
    println!("  avg sim:     {:.3}", avg_sim);
    println!("  sim >= 0.85: {:.1}% (high-confidence corpus hits)", frac_high * 100.0);
    println!("  sim < 0.50:  {:.1}% (fallback 'não entendi')", frac_low * 100.0);

    Ok(())
}
