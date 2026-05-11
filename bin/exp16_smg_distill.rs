//! Experimento 16 — SMG-tiny distillation from Qwen 2.5 1.5B
//!
//! ## Tese
//! O modelo aprende **o bit**, não a semântica. Pegamos a saída byte-a-byte do
//! Qwen 2.5 1.5B (teacher) para 100 prompts, codificamos em Path C, e treinamos
//! um modelo minúsculo (SMG-tiny baseline = MLP, ver `docs/smg-design.md`) em
//! next-byte prediction.
//!
//! ## Por quê isso valida a destilação
//! Se um MLP minúsculo já consegue reproduzir os bytes do Qwen com
//! agreement ≥ 90%, então o sinal de supervisão "(prompt-bytes → response-bytes
//! da teacher)" é aprendível. A SMG real (microcoluna FSM + bus 32×32, exp17+)
//! herda esse mesmo sinal — apenas troca a estrutura interna.
//!
//! ## Critérios (definidos em docs/smg-design.md §6)
//! - **Sucesso M1:** byte-exact agreement ≥ 90%
//! - **Sucesso parcial:** prefix-agreement ≥ 70% (primeiros 8 bytes batem)
//! - **Falha clara:** loss converge mas agreement < 50% — design tem falha conceitual
//!
//! ## Pré-requisito
//! Arquivo `data/qwen_traces.bin` gerado por:
//! ```bash
//! cargo run --release -p iara-byte-bench --bin dump_qwen_traces -- \
//!   --prompts ../byte-language-lab/data/exp16_prompts.txt \
//!   --out    ../byte-language-lab/data/qwen_traces.bin
//! ```

use blab::Rng;
use blab::format::path_c;
use blab::learn::mlp::{Mlp, MlpConfig};
use std::collections::{BTreeSet, HashMap};
use std::io::{Read, Write};
use std::path::Path;

const SEED: u64 = 9001;
const WINDOW: usize = 48;
const HIDDEN: usize = 512;
/// LR para mini-batch SGD paralelo. Equivalente ~ per-sample SGD lr=0.001
/// dividido pelo batch (batch=64 → effective per-sample lr ≈ 0.064 / 64).
const LR: f32 = 0.064;
const EPOCHS: usize = 30;
const REPS_PER_PAIR: usize = 24;
/// Mini-batch para parallel training.
const BATCH_SIZE: usize = 64;
const PAD_BYTE: u8 = 0xFE; // EOB / pad
/// Trunca **resposta** do teacher a este tamanho (bytes ASCII pós-sanitize).
/// Prompts NÃO são truncados — precisam preservar o nome do país/objeto.
const MAX_RESPONSE_BYTES: usize = 32;
/// Trunca prompt no máximo isto (raramente acionado; prompts são curtos).
const MAX_PROMPT_BYTES: usize = 40;

const TRACES_PATH: &str = "data/qwen_traces.bin";
const RESULTS_PATH: &str = "results/exp16_smg_distill.md";
const PROMPTS_PATH: &str = "data/exp16_prompts.txt";

/// Record vindo do Qwen teacher: (prompt UTF-8, response UTF-8).
#[derive(Debug, Clone)]
struct TeacherRecord {
    prompt: String,
    response: String,
}

/// Lê arquivo binário formato QTRC v1.
/// Magic 4B + version u32 + n u32 + (plen u32 + p + rlen u32 + r)*
fn read_traces<P: AsRef<Path>>(p: P) -> std::io::Result<Vec<TeacherRecord>> {
    let mut f = std::fs::File::open(p)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;

    if buf.len() < 12 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "file too small"));
    }
    if &buf[0..4] != b"QTRC" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
    }
    let mut cursor = 4;
    let take_u32 = |cursor: &mut usize| -> std::io::Result<u32> {
        if *cursor + 4 > buf.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "u32 overrun"));
        }
        let v = u32::from_le_bytes([buf[*cursor], buf[*cursor + 1], buf[*cursor + 2], buf[*cursor + 3]]);
        *cursor += 4;
        Ok(v)
    };

    let version = take_u32(&mut cursor)?;
    if version != 1 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unsupported version {version}"),
        ));
    }
    let n = take_u32(&mut cursor)? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let plen = take_u32(&mut cursor)? as usize;
        if cursor + plen > buf.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "prompt overrun"));
        }
        let prompt = String::from_utf8_lossy(&buf[cursor..cursor + plen]).into_owned();
        cursor += plen;
        let rlen = take_u32(&mut cursor)? as usize;
        if cursor + rlen > buf.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "response overrun"));
        }
        let response = String::from_utf8_lossy(&buf[cursor..cursor + rlen]).into_owned();
        cursor += rlen;
        out.push(TeacherRecord { prompt, response });
    }
    Ok(out)
}

/// Encoda (prompt, response) como Path C concatenado, ASCII-normalizado e
/// trimmed para 1 linha. Layout do stream:
///
/// ```text
/// [PAD * pad_left]  [Text-block prompt]  [Text-block response]  0xFE
/// ```
///
/// Path C `encode_text` produz `[header(0x2X)] payload [EOB?]`.
/// Para blocos < 16B usa header com length. > 16B usa header(0x20) + EOB.
fn encode_record(prompt: &str, response: &str) -> Vec<u8> {
    let p = sanitize_prompt(prompt);
    let r = sanitize(response);
    let pblock = path_c::encode_text(&p).expect("text block");
    let rblock = path_c::encode_text(&r).expect("text block");

    // Janela mínima para inferência = pad+pblock alinhados em WINDOW.
    let pad = WINDOW.saturating_sub(pblock.len());
    let mut out = vec![PAD_BYTE; pad];
    out.extend_from_slice(&pblock);
    out.extend_from_slice(&rblock);
    out.push(PAD_BYTE);
    out
}

/// Janela exata do prompt para inferência (pad + bloco prompt = WINDOW bytes).
fn prompt_window(prompt: &str) -> Vec<u8> {
    let p = sanitize_prompt(prompt);
    let pblock = path_c::encode_text(&p).expect("text block");
    let pad = WINDOW.saturating_sub(pblock.len());
    let mut out = vec![PAD_BYTE; pad];
    out.extend_from_slice(&pblock);
    out
}

/// Strip Qwen2 chat special tokens da forma `<|...|>` (`<|im_end|>`, `<|endoftext|>`, etc).
/// Char-safe: opera sobre `char_indices` para preservar UTF-8 multi-byte.
fn strip_special_tokens(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut byte_i = 0usize;
    while byte_i < s.len() {
        // janela ASCII `<|...|>` é segura em bytes (todos os 4 chars são ASCII)
        if byte_i + 1 < bytes.len() && bytes[byte_i] == b'<' && bytes[byte_i + 1] == b'|' {
            if let Some(end_off) = bytes[byte_i + 2..].windows(2).position(|w| w == b"|>") {
                byte_i += 2 + end_off + 2;
                continue;
            }
        }
        // copia 1 char UTF-8 inteiro (1-4 bytes)
        let ch_end = (byte_i + 1..=byte_i + 4)
            .find(|&j| j == s.len() || s.is_char_boundary(j))
            .unwrap_or(byte_i + 1);
        out.push_str(&s[byte_i..ch_end]);
        byte_i = ch_end;
    }
    out
}

/// Trunca string ANTES do primeiro `.`, `\n`, `:`, `;` ou `,` — pega 1ª clause.
/// Char-safe: opera sobre `char_indices` para nunca quebrar UTF-8 multi-byte.
fn first_clause(s: &str) -> &str {
    let cap_bytes = MAX_RESPONSE_BYTES * 4;
    // primeiro, descobre o ponto de corte char-aligned ≤ cap_bytes
    let scan_end = s
        .char_indices()
        .take_while(|(i, _)| *i < cap_bytes)
        .last()
        .map_or(0, |(i, c)| i + c.len_utf8())
        .min(s.len());
    let scan = &s[..scan_end];
    if let Some(pos) = scan.find(|c: char| matches!(c, '.' | '\n' | ':' | ';')) {
        &s[..pos]
    } else if let Some(pos) = scan.rfind(',') {
        &s[..pos]
    } else {
        scan
    }
}

/// Normalização compartilhada: lowercase + diacríticos + filter ASCII alphanum + collapse spaces.
/// `max_bytes` controla truncate final; `take_first_clause` decide se cortar em `.`/`\n`.
fn sanitize_inner(s: &str, max_bytes: usize, take_first_clause_flag: bool) -> String {
    let stripped = strip_special_tokens(s);
    let clause = if take_first_clause_flag { first_clause(&stripped) } else { &stripped[..] };
    let lowered = clause.to_lowercase();
    let mut out = String::with_capacity(lowered.len());
    let mut last_was_space = true;
    for c in lowered.chars() {
        let mapped = match c {
            'á' | 'à' | 'â' | 'ã' | 'ä' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'í' | 'ì' | 'î' | 'ï' => 'i',
            'ó' | 'ò' | 'ô' | 'õ' | 'ö' => 'o',
            'ú' | 'ù' | 'û' | 'ü' => 'u',
            'ç' => 'c',
            'ñ' => 'n',
            _ => c,
        };
        if mapped.is_ascii_alphanumeric() {
            out.push(mapped);
            last_was_space = false;
        } else if mapped.is_whitespace() || mapped == ',' {
            if !last_was_space && !out.is_empty() {
                out.push(' ');
                last_was_space = true;
            }
        }
        // Skip qualquer outro: . - ? ! ( ) etc. Mantém só [a-z0-9 ].
    }
    while out.ends_with(' ') {
        out.pop();
    }
    if out.len() > max_bytes {
        out.truncate(max_bytes);
        if let Some(sp) = out.rfind(' ') {
            if out.len() - sp <= 8 {
                out.truncate(sp);
            }
        }
    }
    out.trim().to_string()
}

/// Sanitiza prompt (preserva nome do país/objeto, sem cortar em `.`).
fn sanitize_prompt(s: &str) -> String {
    sanitize_inner(s, MAX_PROMPT_BYTES, false)
}

/// Sanitiza resposta do teacher (1ª clause, truncada).
fn sanitize(s: &str) -> String {
    sanitize_inner(s, MAX_RESPONSE_BYTES, true)
}

/// Mapa estável byte ↔ id.
struct VocabMap {
    byte_to_id: HashMap<u8, u32>,
    id_to_byte: Vec<u8>,
}

impl VocabMap {
    fn build(streams: &[Vec<u8>]) -> Self {
        let mut byte_to_id = HashMap::new();
        let unique: BTreeSet<u8> = streams.iter().flatten().copied().collect();
        let id_to_byte: Vec<u8> = unique.into_iter().collect();
        for (i, &b) in id_to_byte.iter().enumerate() {
            byte_to_id.insert(b, i as u32);
        }
        Self { byte_to_id, id_to_byte }
    }
    fn encode(&self, bytes: &[u8]) -> Vec<u32> {
        bytes.iter().filter_map(|b| self.byte_to_id.get(b).copied()).collect()
    }
    fn decode(&self, ids: &[u32]) -> Vec<u8> {
        ids.iter().map(|&i| *self.id_to_byte.get(i as usize).unwrap_or(&b'?')).collect()
    }
    fn vocab_size(&self) -> usize {
        self.id_to_byte.len()
    }
    fn pad_id(&self) -> Option<u32> {
        self.byte_to_id.get(&PAD_BYTE).copied()
    }
}

fn levenshtein(a: &[u8], b: &[u8]) -> usize {
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

fn main() -> std::io::Result<()> {
    println!("[exp16] SMG-tiny distillation from Qwen 2.5 1.5B\n");

    // ── Carrega traces ─────────────────────────────────────────────────────
    let traces = match read_traces(TRACES_PATH) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "[exp16] Não consegui ler {TRACES_PATH}: {e}\n\
                 \n\
                 Antes de rodar exp16, gere as traces do Qwen:\n\
                 \n\
                 1. Em outro terminal, suba o iara-engine + carregue qwen2.5-1.5b\n\
                    cd ../llm-lab && ./scripts/run_iara.sh release\n\
                    curl -X POST http://localhost:3000/api/load-model \\\n\
                         -H 'content-type: application/json' \\\n\
                         -d '{{\"name\":\"qwen2.5-1.5b\"}}'\n\
                 \n\
                 2. Dump traces (1 vez):\n\
                    cd ../llm-lab/iara-byte-bench && \\\n\
                    cargo run --release --bin dump_qwen_traces -- \\\n\
                      --prompts ../../byte-language-lab/{PROMPTS_PATH} \\\n\
                      --out     ../../byte-language-lab/{TRACES_PATH}\n\
                 \n\
                 3. Rodar de novo: cargo run --release --bin exp16_smg_distill\n"
            );
            std::process::exit(1);
        }
    };
    println!("[exp16] {} (prompt, response) records carregados de {TRACES_PATH}", traces.len());
    if traces.is_empty() {
        eprintln!("[exp16] zero records — abort.");
        std::process::exit(1);
    }

    // ── Sanitiza & encoda ───────────────────────────────────────────────────
    let sanitized: Vec<(String, String)> = traces
        .iter()
        .map(|r| (sanitize_prompt(&r.prompt), sanitize(&r.response)))
        .filter(|(p, r)| !p.is_empty() && !r.is_empty())
        .collect();
    println!("[exp16] {} records após sanitize (drop vazios)", sanitized.len());
    println!("[exp16] preview (10 primeiros):");
    for (p, r) in sanitized.iter().take(10) {
        println!("        {p:?} → {r:?}");
    }

    // Constrói streams de treino (1 stream por record × REPS_PER_PAIR repetições)
    let mut streams = Vec::with_capacity(sanitized.len() * REPS_PER_PAIR);
    for _ in 0..REPS_PER_PAIR {
        for (p, r) in &sanitized {
            streams.push(encode_record(p, r));
        }
    }

    let vocab = VocabMap::build(&streams);
    println!("[exp16] vocab_size = {} bytes únicos", vocab.vocab_size());
    println!("[exp16] avg bytes/stream = {:.1}",
        streams.iter().map(Vec::len).sum::<usize>() as f64 / streams.len() as f64);

    // Sliding-window training pairs
    let id_streams: Vec<Vec<u32>> = streams.iter().map(|s| vocab.encode(s)).collect();
    let mut pairs: Vec<(Vec<u32>, u32)> = Vec::new();
    for stream in &id_streams {
        if stream.len() <= WINDOW {
            continue;
        }
        for i in 0..stream.len() - WINDOW {
            pairs.push((stream[i..i + WINDOW].to_vec(), stream[i + WINDOW]));
        }
    }
    println!("[exp16] training pairs (window→target) = {}", pairs.len());

    // ── Treino ──────────────────────────────────────────────────────────────
    let cfg = MlpConfig {
        vocab_size: vocab.vocab_size(),
        window: WINDOW,
        hidden: HIDDEN,
        learning_rate: LR,
        seed: SEED,
    };
    let mut mlp = Mlp::new(cfg);
    let mut rng = Rng::new(SEED);
    let mut last_loss = 0.0f32;
    let mut loss_history: Vec<(usize, f32)> = Vec::new();

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    println!("[exp16] training in parallel: {n_threads} threads, batch_size={BATCH_SIZE}");

    let t_train = std::time::Instant::now();
    let initial_loss_target = (vocab.vocab_size() as f32).ln(); // -ln(1/V) ≈ random init
    let divergence_threshold = initial_loss_target * 3.0; // > 3× random = explodiu
    let mut diverged = false;

    for epoch in 0..EPOCHS {
        // Fisher-Yates shuffle
        for i in (1..pairs.len()).rev() {
            let j = rng.range_u32(i as u32 + 1) as usize;
            pairs.swap(i, j);
        }
        last_loss = mlp.train_epoch_parallel(&pairs, n_threads, BATCH_SIZE);
        if epoch == 0 || (epoch + 1) % 5 == 0 || epoch == EPOCHS - 1 {
            loss_history.push((epoch, last_loss));
            println!("  epoch {:3}: loss = {:.4}", epoch + 1, last_loss);
        }
        if !last_loss.is_finite() || last_loss > divergence_threshold {
            eprintln!("[exp16] DIVERGED em epoch {} (loss={:.2} > {:.2}). \
                Abortando — reduzir LR (atual {LR}) ou capacidade.",
                epoch + 1, last_loss, divergence_threshold);
            diverged = true;
            break;
        }
    }
    let train_secs = t_train.elapsed().as_secs_f64();
    let train_acc = if diverged { 0.0 } else { mlp.eval_accuracy(&pairs) };
    println!("[exp16] train done in {:.1}s — final loss = {:.4}, next-byte acc = {:.1}%{}",
        train_secs, last_loss, train_acc * 100.0,
        if diverged { " (DIVERGED)" } else { "" });

    // ── Avaliação: byte-agreement com teacher ───────────────────────────────
    let pad_id = vocab.pad_id();
    let mut exact_match = 0usize;
    let mut prefix8_match = 0usize;
    let mut byte_agreement_total = 0.0f64;
    let mut row_results: Vec<(String, String, String, usize, usize, bool)> = Vec::new();

    for (p, expected_r) in &sanitized {
        let prompt_bytes = prompt_window(p);
        let mut gen_ids: Vec<u32> = vocab.encode(&prompt_bytes);
        let initial_len = gen_ids.len();
        let max_gen = expected_r.len() as i32 + 16; // pequena folga
        let max_gen = max_gen.max(8) as usize;

        for _ in 0..max_gen {
            let window: Vec<u32> = if gen_ids.len() >= WINDOW {
                gen_ids[gen_ids.len() - WINDOW..].to_vec()
            } else {
                let mut w = vec![pad_id.unwrap_or(0); WINDOW - gen_ids.len()];
                w.extend(&gen_ids);
                w
            };
            let next = mlp.predict(&window);
            gen_ids.push(next);
            if Some(next) == pad_id {
                break;
            }
        }
        let payload_ids = &gen_ids[initial_len..];
        let payload = vocab.decode(payload_ids);

        // Strip header byte do bloco Text + EOB final, se presentes
        let mut gen_str = String::new();
        for &b in &payload {
            if b.is_ascii_alphanumeric() || b == b' ' {
                gen_str.push(b as char);
            }
        }
        let gen_str = gen_str.trim().to_string();

        // Comparações
        let exp_b = expected_r.as_bytes();
        let out_b = gen_str.as_bytes();
        let edit = levenshtein(exp_b, out_b);
        let max_len = exp_b.len().max(out_b.len()).max(1);
        let agreement = 1.0 - (edit as f64 / max_len as f64);
        byte_agreement_total += agreement;

        let exact = exp_b == out_b;
        if exact {
            exact_match += 1;
        }
        let prefix_n = 8.min(exp_b.len()).min(out_b.len());
        if prefix_n > 0 && exp_b[..prefix_n] == out_b[..prefix_n] {
            prefix8_match += 1;
        }

        row_results.push((p.clone(), expected_r.clone(), gen_str, edit, max_len, exact));
    }

    let n = sanitized.len() as f64;
    let avg_agreement = byte_agreement_total / n;

    // ── Markdown report ─────────────────────────────────────────────────────
    std::fs::create_dir_all("results")?;
    let mut out = std::fs::File::create(RESULTS_PATH)?;
    writeln!(out, "# Experimento 16 — SMG-tiny distillation from Qwen 2.5 1.5B\n")?;
    writeln!(out, "**Tese:** o modelo aprende o **bit**, não a semântica. Supervisão = bytes do Qwen.\n")?;
    writeln!(out, "Ver `docs/smg-design.md` §6 para critérios de sucesso/falha.\n")?;

    writeln!(out, "## Setup\n")?;
    writeln!(out, "- Teacher: **Qwen 2.5 1.5B** (greedy, T=0, top_k=1) via iara-engine HTTP.")?;
    writeln!(out, "- Student: SMG-tiny baseline (MLP 1-hidden, vocab={}, window={}, hidden={}).",
        vocab.vocab_size(), WINDOW, HIDDEN)?;
    writeln!(out, "- Encoding: Path C tagged. `[T:N prompt][T:M response]\\xFE`")?;
    writeln!(out, "- Treino: {} prompts × {} reps = {} streams, {} pares window→target, {} épocas.",
        sanitized.len(), REPS_PER_PAIR, streams.len(), pairs.len(), EPOCHS)?;
    writeln!(out, "- SGD paralelo: {} threads, batch_size={}, lr={}.",
        n_threads, BATCH_SIZE, LR)?;
    writeln!(out, "- Tempo: {:.1}s.", train_secs)?;
    writeln!(out)?;

    writeln!(out, "## Loss curve\n")?;
    writeln!(out, "| Epoch | Loss |")?;
    writeln!(out, "|------:|-----:|")?;
    for (e, l) in &loss_history {
        writeln!(out, "| {} | {:.4} |", e + 1, l)?;
    }
    writeln!(out, "\n- Final loss: **{:.4}**", last_loss)?;
    writeln!(out, "- Train next-byte accuracy: **{:.1}%**", train_acc * 100.0)?;
    writeln!(out)?;

    writeln!(out, "## Byte-agreement com teacher\n")?;
    writeln!(out, "| Prompt | Qwen (teacher) | SMG-tiny (student) | edit/len | exact |")?;
    writeln!(out, "|--------|----------------|--------------------|---------:|:-----:|")?;
    for (p, exp_s, out_s, edit, mlen, ok) in &row_results {
        let mark = if *ok { "✓" } else { "✗" };
        let exp_disp = if exp_s.len() > 40 { format!("{}…", &exp_s[..40]) } else { exp_s.clone() };
        let out_disp = if out_s.len() > 40 { format!("{}…", &out_s[..40]) } else { out_s.clone() };
        writeln!(out, "| {} | `{}` | `{}` | {}/{} | {} |",
            p, exp_disp, out_disp, edit, mlen, mark)?;
    }
    writeln!(out)?;

    writeln!(out, "## Métricas finais\n")?;
    writeln!(out, "| Métrica | Valor |")?;
    writeln!(out, "|---------|------:|")?;
    writeln!(out, "| Records avaliados | {} |", sanitized.len())?;
    writeln!(out, "| Byte-exact match | {}/{} = {:.1}% |", exact_match, sanitized.len(),
        100.0 * exact_match as f64 / n)?;
    writeln!(out, "| Prefix-8 match  | {}/{} = {:.1}% |", prefix8_match, sanitized.len(),
        100.0 * prefix8_match as f64 / n)?;
    writeln!(out, "| Avg byte-agreement (1 - edit/len) | **{:.1}%** |", avg_agreement * 100.0)?;
    writeln!(out)?;

    // ── Verdict alinhado ao spec §6 ─────────────────────────────────────────
    let pct = avg_agreement * 100.0;
    let exact_pct = 100.0 * exact_match as f64 / n;
    writeln!(out, "## Veredicto (vs `docs/smg-design.md` §6)\n")?;
    let verdict = if exact_pct >= 90.0 || pct >= 90.0 {
        "**✓ M1 ATINGIDO** — destilação byte-a-byte do Qwen é aprendível por modelo minúsculo.\n\
         Liberado para exp17 (substituir MLP por estrutura microcoluna FSM real)."
    } else if pct >= 70.0 {
        "**~ Parcial** — sinal está aprendível mas com erro. Investigar capacidade do modelo, \
         tamanho de janela ou normalização de prompts antes de exp17."
    } else if pct >= 50.0 {
        "**? Cinza** — destilação parcial. Provável que sanitize esteja agressivo demais ou \
         o vocab esteja truncando informação. Revisar antes de avançar."
    } else {
        "**✗ FALHA CONCEITUAL** — modelo não converge para os bytes do teacher. Hipótese \
         'destilação resolve treino' precisa ser reavaliada antes de qualquer SMG estrutural."
    };
    writeln!(out, "{verdict}")?;
    writeln!(out)?;

    println!("\n[exp16] avg byte-agreement = {:.1}%, exact = {:.1}%, prefix8 = {:.1}%",
        pct, exact_pct, 100.0 * prefix8_match as f64 / n);
    println!("[exp16] resultados gravados em {RESULTS_PATH}");
    Ok(())
}
