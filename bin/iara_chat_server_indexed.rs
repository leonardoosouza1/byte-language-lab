//! iara_chat_server_indexed.rs — server que CARREGA pre-built corpus_index.bin.
//!
//! Boot ~10ms vs 300ms da versão que parse text. Same API endpoints.

use std::{fs, io};
use std::io::{Read, Write, BufRead, BufReader};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const HV_BITS: usize = 1024;
const HV_U64:  usize = HV_BITS / 64;
type Hv = [u64; HV_U64];

const HV_CACHE: &str = "data/byte_hvs_qwen.bin";
const INDEX_PATH: &str = "data/corpus_index.bin";
const PORT: u16 = 3011;
const TOP_K: usize = 30;

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

fn sim(a: &Sig, b: &Sig) -> f32 {
    let h_bag = hamming(&a.bag, &b.bag);
    let h_big = hamming(&a.bigram, &b.bigram);
    let h_seq = hamming(&a.seq, &b.seq);
    let s_bag = 1.0 - h_bag as f32 / HV_BITS as f32;
    let s_big = 1.0 - h_big as f32 / HV_BITS as f32;
    let s_seq = 1.0 - h_seq as f32 / HV_BITS as f32;
    0.5 * s_bag + 0.35 * s_big + 0.15 * s_seq
}

fn levenshtein(a: &str, b: &str) -> usize {
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    let m = av.len();
    let n = bv.len();
    if m == 0 { return n; }
    if n == 0 { return m; }
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m { dp[i][0] = i; }
    for j in 0..=n { dp[0][j] = j; }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if av[i-1] == bv[j-1] { 0 } else { 1 };
            dp[i][j] = (dp[i-1][j] + 1).min(dp[i][j-1] + 1).min(dp[i-1][j-1] + cost);
        }
    }
    dp[m][n]
}
fn lev_sim(a: &str, b: &str) -> f32 {
    let d = levenshtein(a, b);
    let maxl = a.chars().count().max(b.chars().count()).max(1);
    1.0 - d as f32 / maxl as f32
}

struct QA { user_raw: String, user_norm: String, sig: Sig, model: String }

fn read_hv(f: &mut fs::File) -> io::Result<Hv> {
    let mut out = [0u64; HV_U64];
    for w in out.iter_mut() {
        let mut bb = [0u8; 8];
        f.read_exact(&mut bb)?;
        *w = u64::from_le_bytes(bb);
    }
    Ok(out)
}

fn load_index(path: &str) -> io::Result<Vec<QA>> {
    let mut f = fs::File::open(path)?;
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)?;
    let magic = u32::from_le_bytes(buf4);
    if magic != 0x48444358 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "wrong magic"));
    }
    f.read_exact(&mut buf4)?;  // version
    f.read_exact(&mut buf4)?;
    let n = u32::from_le_bytes(buf4) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut buf2 = [0u8; 2];
        f.read_exact(&mut buf2)?;
        let ul = u16::from_le_bytes(buf2) as usize;
        let mut ub = vec![0u8; ul];
        f.read_exact(&mut ub)?;
        let user_raw = String::from_utf8_lossy(&ub).to_string();

        f.read_exact(&mut buf2)?;
        let ml = u16::from_le_bytes(buf2) as usize;
        let mut mb = vec![0u8; ml];
        f.read_exact(&mut mb)?;
        let model = String::from_utf8_lossy(&mb).to_string();

        let bag = read_hv(&mut f)?;
        let bigram = read_hv(&mut f)?;
        let seq = read_hv(&mut f)?;
        let user_norm = normalize(&user_raw);
        out.push(QA { user_raw, user_norm, sig: Sig { bag, bigram, seq }, model });
    }
    Ok(out)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"), '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
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

fn search(query: &str, byte_hvs: &[Hv; 256], pos_hvs: &[Hv; 256], qa: &[QA]) -> (f32, usize, u128) {
    let t0 = Instant::now();
    let sig = encode(query, byte_hvs, pos_hvs);
    let mut hdc_scores: Vec<(f32, usize)> = qa.iter().enumerate()
        .map(|(i, item)| (sim(&sig, &item.sig), i))
        .collect();
    hdc_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let topk: Vec<_> = hdc_scores.into_iter().take(TOP_K).collect();
    let q_norm = normalize(query);
    let mut reranked: Vec<(f32, usize)> = topk.into_iter().map(|(hdc_s, idx)| {
        let lev_s = lev_sim(&q_norm, &qa[idx].user_norm);
        (0.4 * hdc_s + 0.6 * lev_s, idx)
    }).collect();
    reranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let (s, i) = reranked[0];
    (s, i, t0.elapsed().as_micros())
}

fn handle(mut stream: TcpStream, byte_hvs: Arc<[Hv; 256]>, pos_hvs: Arc<[Hv; 256]>, qa: Arc<Vec<QA>>) -> io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let request_line = request_line.trim().to_string();

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 { break; }
        let t = line.trim().to_string();
        if t.is_empty() { break; }
        if let Some(v) = t.to_lowercase().strip_prefix("content-length: ") {
            content_length = v.parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 { reader.read_exact(&mut body)?; }
    let body_str = String::from_utf8_lossy(&body).to_string();

    let parts: Vec<&str> = request_line.split_whitespace().collect();
    let method = parts.get(0).copied().unwrap_or("");
    let path = parts.get(1).copied().unwrap_or("/");

    let (status, response_body) = match (method, path) {
        ("GET", "/health") => (
            "200 OK",
            format!("{{\"status\":\"ok\",\"corpus_size\":{},\"version\":\"hdc-1.1-indexed\"}}", qa.len())
        ),
        ("POST", "/chat") | ("POST", "/api/generate") => {
            let prompt = json_extract_str(&body_str, "prompt").unwrap_or_default();
            if prompt.is_empty() {
                ("400 Bad Request", "{\"error\":\"missing prompt\"}".to_string())
            } else {
                let (sim_score, idx, lat_us) = search(&prompt, &byte_hvs, &pos_hvs, &qa);
                let item = &qa[idx];
                let (response_text, match_text) = if sim_score < 0.50 {
                    ("Desculpe, não entendi sua mensagem ou não tenho contexto sobre isso. Pode reformular?".to_string(),
                     format!("[no-match sim={:.3}]", sim_score))
                } else if sim_score < 0.70 {
                    (format!("Não tenho certeza, mas talvez: {}", item.model), item.user_raw.clone())
                } else {
                    (item.model.clone(), item.user_raw.clone())
                };
                ("200 OK", format!(
                    "{{\"model\":\"hdc-iara-indexed\",\"response\":\"{}\",\"match\":\"{}\",\"sim\":{:.4},\"latency_us\":{},\"done\":true,\"tokens_per_second\":{:.1}}}",
                    json_escape(&response_text), json_escape(&match_text), sim_score, lat_us,
                    response_text.len() as f64 * 1_000_000.0 / lat_us as f64
                ))
            }
        }
        _ => ("404 Not Found", "{\"error\":\"not found\"}".to_string())
    };

    let response = format!(
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
        status, response_body.len(), response_body
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn main() -> io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA Chat Server INDEXED — load pre-built corpus binary    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let t0 = Instant::now();
    let byte_hvs = load_byte_hvs()?;
    let pos_hvs = build_pos_hvs();
    println!("  HVs: {}ms", t0.elapsed().as_millis());

    if !Path::new(INDEX_PATH).exists() {
        eprintln!("ERRO: {} não existe.", INDEX_PATH);
        eprintln!("  Rode: ./target/release/build_corpus_index");
        std::process::exit(1);
    }

    let t1 = Instant::now();
    let qa = load_index(INDEX_PATH)?;
    println!("  Index: {} pares carregados em {}ms ★", qa.len(), t1.elapsed().as_millis());
    println!("  Boot total: {}ms (vs ~300ms da versão sem index)\n", t0.elapsed().as_millis());

    let byte_hvs = Arc::new(byte_hvs);
    let pos_hvs = Arc::new(pos_hvs);
    let qa = Arc::new(qa);

    let listener = TcpListener::bind(("127.0.0.1", PORT))?;
    println!("  Listening em http://127.0.0.1:{}\n", PORT);

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let b = Arc::clone(&byte_hvs);
                let p = Arc::clone(&pos_hvs);
                let q = Arc::clone(&qa);
                std::thread::spawn(move || {
                    if let Err(e) = handle(s, b, p, q) { eprintln!("err: {}", e); }
                });
            }
            Err(e) => eprintln!("conn: {}", e),
        }
    }
    Ok(())
}
