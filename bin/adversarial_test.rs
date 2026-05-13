//! adversarial_test.rs — testes adversariais contra o HDC chat server.
//!
//! Categorias:
//!   - Typos (correct word + 1-3 char errors)
//!   - Case variations
//!   - Diacritics (com/sem)
//!   - Out-of-corpus (deliberately new topics)
//!   - Long queries
//!   - Empty/edge cases
//!   - Unicode mixed
//!
//! Bate em http://127.0.0.1:3010/chat e calcula stats.

use std::io::{Read, Write, BufRead, BufReader};
use std::net::TcpStream;
use std::time::Instant;

const SERVER: &str = "127.0.0.1:3010";

fn json_extract_str(body: &str, key: &str) -> Option<String> {
    let pat = format!("\"{}\"", key);
    let pos = body.find(&pat)?;
    let after = &body[pos + pat.len()..];
    let colon = after.find(':')?;
    let after = after[colon+1..].trim_start();
    if !after.starts_with('"') { return None; }
    let mut s = String::new();
    let mut iter = after[1..].chars().peekable();
    while let Some(c) = iter.next() {
        if c == '\\' {
            if let Some(&nx) = iter.peek() { iter.next();
                match nx {
                    'n' => s.push('\n'), 't' => s.push('\t'),
                    'r' => s.push('\r'), '"' => s.push('"'),
                    '\\' => s.push('\\'), other => s.push(other),
                }
            }
        } else if c == '"' { break; } else { s.push(c); }
    }
    Some(s)
}
fn json_extract_num(body: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{}\"", key);
    let pos = body.find(&pat)?;
    let after = &body[pos + pat.len()..];
    let colon = after.find(':')?;
    let after = after[colon+1..].trim_start();
    let end = after.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')?;
    after[..end].parse().ok()
}

fn ask(prompt: &str) -> Result<(String, f64, u128, std::time::Duration), Box<dyn std::error::Error>> {
    let t0 = Instant::now();
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
    let elapsed = t0.elapsed();
    let response = json_extract_str(&body_str, "response").unwrap_or_default();
    let sim = json_extract_num(&body_str, "sim").unwrap_or(0.0);
    let lat_us = json_extract_num(&body_str, "latency_us").unwrap_or(0.0) as u128;
    Ok((response, sim, lat_us, elapsed))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA Adversarial Test Suite                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let tests: &[(&str, &str)] = &[
        // Category : Query
        ("hello",        "oi"),
        ("hello",        "Oi"),
        ("hello",        "OI"),
        ("hello",        "OII"),
        ("typo-1",       "ola"),
        ("typo-1",       "olha"),
        ("typo-2",       "feijoda"),
        ("typo-3",       "fjeoada"),
        ("typo-mix",     "ronaldinho gaucho"),
        ("typo-mix",     "ronald gaúcho"),
        ("case-mix",     "QUEM É PELÉ?"),
        ("case-mix",     "QuEm É pElÉ"),
        ("acc-strip",    "feijoada sem acentos"),
        ("acc-strip",    "voce é uma ia"),
        ("oov-topic",    "quantum entanglement"),
        ("oov-topic",    "como vai o mercado de NFTs"),
        ("long",         "Olá amigo me conta tudo sobre o Brasil agora por favor"),
        ("very-long",    "Pensando agora sobre a relação entre samba e bossa nova quero entender melhor"),
        ("very-short",   "a"),
        ("very-short",   "e"),
        ("empty-like",   "    "),
        ("punctuation",  "!!!!!!"),
        ("unicode-emoji","🇧🇷"),
        ("unicode-emoji","🎵"),
        ("mixed-lang",   "what é Brasil"),
        ("mixed-lang",   "how say bom dia"),
        ("numbers",      "1+1"),
        ("numbers",      "2 times 3"),
        ("greeting-mix", "oi tudo bem como vai"),
        ("question",     "quem inventou a internet?"),
        ("question",     "qual a capital de Portugal?"),
        ("verb-form",    "cozinhei feijoada ontem"),
        ("colloquial",   "tipo assim né cara"),
        ("formal",       "Prezado(a), gostaria de saber mais sobre"),
        ("imperative",   "me ensine python"),
        ("imperative",   "explique gravidade"),
    ];

    let mut by_category: std::collections::HashMap<&str, (u32, f64, u128)> = std::collections::HashMap::new();
    let mut total_lat_us = 0u128;
    let mut count = 0;

    println!("{:>14}  {:>40}  {:>5}  {:>5}  {:>4}", "category", "QUERY → match (truncated)", "sim", "us", "ok?");
    println!("{}", "-".repeat(90));
    for (cat, q) in tests {
        match ask(q) {
            Ok((resp, sim, lat_us, _wall)) => {
                let resp_short: String = resp.chars().take(45).collect();
                let ok = if sim >= 0.85 { "✓" } else { "?" };
                let entry = by_category.entry(*cat).or_insert((0, 0.0, 0));
                entry.0 += 1;
                entry.1 += sim;
                entry.2 += lat_us;
                total_lat_us += lat_us;
                count += 1;
                let q_short: String = q.chars().take(38).collect();
                println!("{:>14}  {:>40} → {:.3}  {:>5}μs  {}",
                    cat, q_short, sim, lat_us, ok);
                println!("                       └─ MODEL: {}", resp_short);
            }
            Err(e) => println!("{:>14}  {:>40}  ERROR: {}", cat, q, e),
        }
    }

    println!("\n  ★ Total queries: {}", count);
    println!("  ★ Avg latency: {}μs", total_lat_us / count.max(1) as u128);
    println!("\n  ★ Por categoria:\n");
    let mut cats: Vec<_> = by_category.iter().collect();
    cats.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    for (cat, (n, sum_sim, sum_lat)) in cats {
        println!("  {:>14}  n={}  avg_sim={:.3}  avg_lat={}μs", cat, n, sum_sim / *n as f64, sum_lat / *n as u128);
    }

    Ok(())
}
