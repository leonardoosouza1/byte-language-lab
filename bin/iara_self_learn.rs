//! iara_self_learn.rs — sistema aprende sozinho consultando Qwen.
//!
//! Pipeline:
//!   1. Lê test set de prompts (out-of-corpus prováveis)
//!   2. Query no server HDC (sim score?)
//!   3. Se sim < THRESHOLD: query Qwen como oracle
//!   4. Append (test_prompt, qwen_response) → online_learning.txt
//!   5. Pede ao watchdog rebuild (próximo ciclo 30min)
//!   6. Re-test e mede improvement
//!
//! Objetivo: corpus evolui sozinho. Próxima chamada da query similar = match exato.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Instant;

const HDC_SERVER: &str = "127.0.0.1:3010";
const QWEN_SERVER: &str = "127.0.0.1:3000";
const LEARN_THRESHOLD: f64 = 0.70;  // sim abaixo disso → aprende via Qwen
const ONLINE_LOG: &str = "data/online_learning.txt";

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

fn http_post(server: &str, path: &str, body: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut stream = TcpStream::connect(server)?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(60)))?;
    let req = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        path, server, body.len(), body
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
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn query_hdc(prompt: &str) -> Result<f64, Box<dyn std::error::Error>> {
    let body = format!("{{\"prompt\":{}}}", serde_json::to_string(prompt)?);
    let resp = http_post(HDC_SERVER, "/chat", &body)?;
    Ok(json_extract_num(&resp, "sim").unwrap_or(0.0))
}

fn query_qwen(prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    let body = format!(
        "{{\"model\":\"qwen2.5-1.5b\",\"prompt\":{},\"max_tokens\":80,\"temperature\":0.7,\"top_k\":40,\"top_p\":0.9}}",
        serde_json::to_string(prompt)?
    );
    let resp = http_post(QWEN_SERVER, "/api/generate", &body)?;
    let r = json_extract_str(&resp, "response").unwrap_or_default()
        .replace("<|im_end|>", "").trim().to_string();
    Ok(r)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA Self-Learn — sistema aprende sozinho via Qwen oracle  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Out-of-corpus prompts: topics novos PT-BR / técnicos
    let new_prompts: &[&str] = &[
        "qual a moeda da Argentina?",
        "quem foi Getúlio Vargas?",
        "o que é WebAssembly?",
        "o que é blockchain?",
        "qual a velocidade do som?",
        "qual o ponto de fusão da água?",
        "o que é gravidade?",
        "quem inventou a internet?",
        "qual a capital do Japão?",
        "qual a capital da Coreia do Sul?",
        "o que é DNA?",
        "quem foi Charles Darwin?",
        "o que é fotossíntese?",
        "quem foi Albert Einstein?",
        "o que significa relatividade?",
        "qual o maior oceano do mundo?",
        "quem foi Santos Dumont?",
        "o que é 5G?",
        "qual o maior país do mundo?",
        "quem foi Madame Curie?",
        "o que é quântica?",
        "quem foi Picasso?",
        "qual o significado de gestalt?",
        "o que é homeostase?",
        "quem foi Bach?",
        "o que é jazz?",
        "qual a maior pirâmide do mundo?",
        "quem foi Cleopatra?",
        "o que é renascimento?",
        "qual o livro mais vendido?",
        "quem inventou o avião?",
        "o que é mecânica clássica?",
        "qual a montanha mais alta?",
        "quem foi Steve Jobs?",
        "o que é hash criptográfico?",
        "qual a profundidade do oceano?",
        "quem pintou Mona Lisa?",
        "o que é metabolismo?",
        "qual a distância da Terra ao Sol?",
        "quem inventou a roda?",
        "como funciona um motor a combustão?",
        "o que é antimatéria?",
        "qual a origem do universo?",
        "quem foi Mahatma Gandhi?",
        "o que é diabetes?",
        "qual a velocidade da luz?",
        "quem foi Beethoven?",
        "o que é nuvem (cloud)?",
        "qual a função do fígado?",
        "quem escreveu Dom Casmurro?",
    ];

    println!("▸ Test set: {} prompts out-of-corpus", new_prompts.len());

    // Stage 1: test all, measure baseline
    println!("\n▸ Stage 1: medindo baseline (sim score por prompt):\n");
    let mut baseline: Vec<(String, f64)> = Vec::new();
    let mut needs_learn = 0;
    let mut already_good = 0;
    for p in new_prompts {
        match query_hdc(p) {
            Ok(sim) => {
                let status = if sim >= LEARN_THRESHOLD { already_good += 1; "✓ já tem" }
                            else { needs_learn += 1; "→ vai aprender" };
                println!("  {:.3}  {:.<40} {}", sim, p, status);
                baseline.push((p.to_string(), sim));
            }
            Err(e) => println!("  ERR  {}: {}", p, e),
        }
    }
    println!("\n  Baseline: {}/{} já tem cobertura adequada", already_good, new_prompts.len());

    if needs_learn == 0 {
        println!("\n  Nada para aprender. Saindo.");
        return Ok(());
    }

    // Stage 2: learn missing via Qwen
    println!("\n▸ Stage 2: consultando Qwen como oracle ({} prompts):\n", needs_learn);
    let mut learned = 0;
    let mut online_log = std::fs::OpenOptions::new().create(true).append(true).open(ONLINE_LOG)?;
    let t_learn = Instant::now();
    for (p, sim_prev) in &baseline {
        if *sim_prev >= LEARN_THRESHOLD { continue; }
        match query_qwen(p) {
            Ok(resp) if !resp.is_empty() => {
                writeln!(online_log, "USER: {}", p)?;
                writeln!(online_log, "MODEL: {}", resp)?;
                online_log.flush()?;
                learned += 1;
                let resp_short: String = resp.chars().take(60).collect();
                println!("  [{}] {:.<40} → {}", learned, p, resp_short);
            }
            Ok(_) => println!("  [skip] {} (resp vazio)", p),
            Err(e) => println!("  [err] {} ({})", p, e),
        }
    }
    let elapsed = t_learn.elapsed();
    println!("\n  Aprendeu {} prompts em {:.0}s ({:.1}s/prompt)",
             learned, elapsed.as_secs_f32(), elapsed.as_secs_f32() / learned.max(1) as f32);

    println!("\n▸ Stage 3: prompts salvos em {}. Watchdog vai rebuild index em até 30min.", ONLINE_LOG);
    println!("            Para forçar agora:");
    println!("            ./target/release/build_corpus_index");
    println!("            pkill -f iara_chat_server_indexed && nohup ./target/release/iara_chat_server_indexed &");

    Ok(())
}
