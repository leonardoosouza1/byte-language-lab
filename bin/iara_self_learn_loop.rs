//! iara_self_learn_loop.rs — loop contínuo de auto-aprendizado.
//!
//! Gera novos prompts (templates × tópicos diversos), testa cada um,
//! aprende os que falham, rebuild index, loop.
//!
//! Stop quando: target_iters atingido OU < 5% prompts precisam aprender.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Instant;

const HDC_SERVER: &str = "127.0.0.1:3011";  // usa indexed (atualizado)
const QWEN_SERVER: &str = "127.0.0.1:3000";
const LEARN_THRESHOLD: f64 = 0.85;   // mais agressivo (aprende quase tudo)
const ONLINE_LOG: &str = "data/online_learning.txt";
const MAX_ITERS: usize = 5;
const PROMPTS_PER_ITER: usize = 30;

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
                match nx { 'n'=>s.push('\n'),'t'=>s.push('\t'),'r'=>s.push('\r'),
                          '"'=>s.push('"'),'\\'=>s.push('\\'),o=>s.push(o), }
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
    Ok(json_extract_str(&resp, "response").unwrap_or_default()
       .replace("<|im_end|>", "").trim().to_string())
}

fn gen_prompts(seed: u64, n: usize) -> Vec<String> {
    let templates = [
        "Como funciona {}?", "O que é {}?", "Explique {}",
        "Conte sobre {}", "Para que serve {}", "Defina {}",
        "Por que {} é importante", "Qual a história de {}",
        "Onde fica {}", "Qual a função de {}",
    ];
    let topics = [
        // Tech expandido
        "Docker", "Kubernetes", "Vue.js", "React", "Django", "FastAPI",
        "GraphQL", "REST API", "TCP/IP", "DNS", "HTTPS", "SSH", "VPN",
        "RAID 0", "RAID 5", "RAM DDR5", "SSD NVMe", "HDD", "CPU cache L3", "memória virtual",
        "JIT compiler", "garbage collector", "lambda function", "closure JS",
        "Redis", "MongoDB", "PostgreSQL", "MySQL", "Cassandra", "Elasticsearch",
        "Kafka", "RabbitMQ", "WebSocket", "gRPC", "JWT", "OAuth", "OAuth2",
        "TensorFlow", "PyTorch", "JAX", "ONNX", "CUDA", "OpenCL", "Vulkan",
        "WebGPU", "WebAssembly", "Rust async", "Go routines", "C++ templates",
        // Físicos
        "elétron", "próton", "nêutron", "fóton", "neutrino",
        "Higgs", "quark", "léptons", "gluon", "boson",
        "termodinâmica", "entropia", "entalpia", "calor específico",
        "ondas eletromagnéticas", "espectro visível", "raios-X", "infravermelho",
        // Bio
        "mitose", "meiose", "tradução ribossomal", "transcrição",
        "neurônio", "sinapse", "neurotransmissor", "axônio",
        "imunidade", "linfócito", "anticorpo", "antígeno",
        "ciclo de Krebs", "respiração celular", "ATP", "glicose",
        // Brasileiríssimo
        "Açúcar", "Rita Lee", "Os Mutantes", "Pixinguinha", "Noel Rosa",
        "Mário de Andrade", "Oswald de Andrade", "Cecília Meireles", "Manuel Bandeira",
        "Castro Alves", "Olavo Bilac", "Vinicius de Moraes",
        "Sérgio Mendes", "Ivete Sangalo", "Anitta", "Pabllo Vittar",
        "Anita Garibaldi", "Zumbi dos Palmares", "Chiquinha Gonzaga", "Maria Bonita",
        "Diretas Já", "Maracanã", "Estádio do Pacaembu", "Mineirão",
        "Vinhedo", "Penedo", "Tiradentes MG", "São Luiz do Paraitinga",
        // Geo
        "Mariana", "Brumadinho", "Petrolina", "Juazeiro do Norte",
        "Cataratas do Iguaçu", "Lençóis Maranhenses", "Chapada Diamantina",
        "Serra do Cipó", "Pico da Bandeira", "Pico da Neblina",
        // Comida regional
        "rabanada", "paçoca", "rapadura", "cocada", "doce de leite",
        "buchada de bode", "carne seca", "carneiro do reino",
        "tucupi", "jambu", "pirarucu", "tambaqui",
        "guaraná Antarctica", "caipirinha", "caju", "graviola",
        // Personagens infantis
        "Mauricio de Sousa", "Monica", "Cebolinha", "Magali", "Cascão",
        "Bidu", "Franjinha", "Chico Bento",
        // Animais Amazônia
        "onça-pintada", "peixe-boi", "guariba", "macaco-prego",
        "anaconda", "sucuri-verde", "jacaré-açu", "piranha",
        "arraia", "poraquê", "candiru", "pirarucu adulto",
        // Tópicos cotidianos
        "atendimento ao cliente", "concurso público", "carteira de trabalho",
        "INSS", "FGTS", "PIS", "MEI", "ICMS", "imposto de renda",
        "vacinação infantil", "SUS", "cartão de saúde",
        "ENEM", "vestibular", "Sisu", "ProUni",
        // Eventos
        "rock in rio", "lollapalooza Brasil", "São Silvestre",
        "fórmula 1 brasileira", "stockcar",
    ];

    let mut rng = seed.max(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let t = templates[(rng as usize) % templates.len()];
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let topic = topics[(rng as usize) % topics.len()];
        out.push(t.replace("{}", topic));
    }
    out
}

fn rebuild_and_restart_server() -> Result<(), Box<dyn std::error::Error>> {
    // Build index
    let st = std::process::Command::new("./target/release/build_corpus_index")
        .output()?;
    if !st.status.success() {
        eprintln!("  rebuild fail: {}", String::from_utf8_lossy(&st.stderr));
        return Err("rebuild fail".into());
    }
    // Kill server
    let _ = std::process::Command::new("pkill")
        .args(["-KILL", "-f", "iara_chat_server_indexed"])
        .output();
    std::thread::sleep(std::time::Duration::from_secs(2));
    // Start
    let _ = std::process::Command::new("sh")
        .arg("-c")
        .arg("nohup ./target/release/iara_chat_server_indexed > /tmp/hdc_server_idx.log 2>&1 &")
        .spawn()?;
    std::thread::sleep(std::time::Duration::from_secs(4));
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA Self-Learn Loop — corpus evolui via Qwen oracle       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("  Server HDC: {}", HDC_SERVER);
    println!("  Server Qwen: {}", QWEN_SERVER);
    println!("  Threshold learn: {}", LEARN_THRESHOLD);
    println!("  Max iters: {}, Prompts/iter: {}", MAX_ITERS, PROMPTS_PER_ITER);
    println!();

    let online_log_path = ONLINE_LOG;
    let initial_size = std::fs::metadata(online_log_path).map(|m| m.len()).unwrap_or(0);
    let t_total = Instant::now();
    let mut total_learned = 0;

    for iter in 0..MAX_ITERS {
        let prompts = gen_prompts((iter as u64 + 100) * 99991, PROMPTS_PER_ITER);
        println!("\n━━━ Iter {} / {} ━━━", iter + 1, MAX_ITERS);
        println!("  Gerados {} prompts", prompts.len());

        let mut to_learn: Vec<String> = Vec::new();
        for p in &prompts {
            match query_hdc(p) {
                Ok(sim) if sim < LEARN_THRESHOLD => {
                    to_learn.push(p.clone());
                }
                Ok(_) => {}
                Err(_) => {}
            }
        }
        let pct_known = 100.0 * (prompts.len() - to_learn.len()) as f32 / prompts.len() as f32;
        println!("  {} já conhecidos ({:.0}%), {} para aprender",
                 prompts.len() - to_learn.len(), pct_known, to_learn.len());

        if to_learn.is_empty() {
            println!("  ✓ saturação: tudo já tem cobertura. parar.");
            break;
        }

        // Learn via Qwen
        let t_iter = Instant::now();
        let mut log_f = std::fs::OpenOptions::new().create(true).append(true).open(online_log_path)?;
        let mut learned_this = 0;
        for p in &to_learn {
            match query_qwen(p) {
                Ok(resp) if resp.len() > 10 => {
                    writeln!(log_f, "USER: {}", p)?;
                    writeln!(log_f, "MODEL: {}", resp)?;
                    learned_this += 1;
                }
                _ => {}
            }
        }
        log_f.flush()?;
        drop(log_f);
        total_learned += learned_this;
        println!("  Qwen ensinou {}/{} em {:.0}s",
                 learned_this, to_learn.len(), t_iter.elapsed().as_secs_f32());

        // Rebuild + restart if learned something
        if learned_this > 0 {
            print!("  ▸ rebuilding index + restart server... ");
            std::io::stdout().flush().ok();
            match rebuild_and_restart_server() {
                Ok(_) => println!("✓"),
                Err(e) => { println!("⚠ {}", e); break; }
            }
        }
    }

    let elapsed = t_total.elapsed();
    let final_size = std::fs::metadata(online_log_path).map(|m| m.len()).unwrap_or(0);
    println!("\n━━━ FIM ━━━");
    println!("  Total iterations: {}", MAX_ITERS);
    println!("  Total learned: {}", total_learned);
    println!("  Online learning file: {} → {} bytes (Δ={})",
             initial_size, final_size, final_size - initial_size);
    println!("  Wall: {:.1} min", elapsed.as_secs_f32() / 60.0);
    Ok(())
}
