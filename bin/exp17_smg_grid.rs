//! Experimento 17 — SMG-grid: primeira microcoluna real, com teste de generalização.
//!
//! Ver `docs/exp17-design.md` para o spec comitado.
//!
//! ## O que muda vs exp16
//!
//! | Aspecto | exp16 (MLP) | exp17 (SMG-grid) |
//! |---|---|---|
//! | Modelo | MLP plano window→logits | 4×4 grade de microcolunas FSM |
//! | Estado | nenhum (stateless) | recorrente K-dim por FSM, BPTT pelos 48 ticks |
//! | Params | ~3M | ~92K (32× menor) |
//! | Split | nenhum (100/0) | **80 train / 20 test, ambos rodam MLP e microcoluna** |
//!
//! ## Pergunta
//!
//! No mesmo train/test split, com prompts não vistos no test, a microcoluna
//! generaliza melhor que o MLP?

use blab::Rng;
use blab::format::path_c;
use blab::learn::microcolumn::{GridConfig, SmgGrid};
use blab::learn::mlp::{Mlp, MlpConfig};
use std::collections::{BTreeSet, HashMap};
use std::io::{Read, Write};
use std::path::Path;

const SEED: u64 = 9001;
const SPLIT_SEED: u64 = 4242;
const WINDOW: usize = 48;
const HIDDEN: usize = 512; // só pro MLP baseline
const LR_MLP: f32 = 0.064;
const LR_GRID: f32 = 0.05;
const EPOCHS: usize = 30;
const REPS_PER_PAIR: usize = 24;
const BATCH_SIZE: usize = 64;
const PAD_BYTE: u8 = 0xFE;
const MAX_RESPONSE_BYTES: usize = 32;
const MAX_PROMPT_BYTES: usize = 40;
const TRAIN_FRAC: f64 = 0.80;

const TRACES_PATH: &str = "data/qwen_traces.bin";
const RESULTS_PATH: &str = "results/exp17_smg_grid.md";

#[derive(Debug, Clone)]
struct TeacherRecord {
    prompt: String,
    response: String,
}

fn read_traces<P: AsRef<Path>>(p: P) -> std::io::Result<Vec<TeacherRecord>> {
    let mut f = std::fs::File::open(p)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    if buf.len() < 12 || &buf[0..4] != b"QTRC" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad header"));
    }
    let mut c = 4;
    let mut take = |n: &mut usize| -> std::io::Result<u32> {
        if *n + 4 > buf.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "u32"));
        }
        let v = u32::from_le_bytes([buf[*n], buf[*n + 1], buf[*n + 2], buf[*n + 3]]);
        *n += 4;
        Ok(v)
    };
    let _ver = take(&mut c)?;
    let n = take(&mut c)? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let pl = take(&mut c)? as usize;
        let prompt = String::from_utf8_lossy(&buf[c..c + pl]).into_owned();
        c += pl;
        let rl = take(&mut c)? as usize;
        let response = String::from_utf8_lossy(&buf[c..c + rl]).into_owned();
        c += rl;
        out.push(TeacherRecord { prompt, response });
    }
    Ok(out)
}

fn strip_special_tokens(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < s.len() {
        if i + 1 < bytes.len() && bytes[i] == b'<' && bytes[i + 1] == b'|' {
            if let Some(end) = bytes[i + 2..].windows(2).position(|w| w == b"|>") {
                i += 2 + end + 2;
                continue;
            }
        }
        let ch_end = (i + 1..=i + 4)
            .find(|&j| j == s.len() || s.is_char_boundary(j))
            .unwrap_or(i + 1);
        out.push_str(&s[i..ch_end]);
        i = ch_end;
    }
    out
}

fn first_clause(s: &str) -> &str {
    let cap = MAX_RESPONSE_BYTES * 4;
    let scan_end = s
        .char_indices()
        .take_while(|(i, _)| *i < cap)
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

fn sanitize_inner(s: &str, max_bytes: usize, take_clause: bool) -> String {
    let stripped = strip_special_tokens(s);
    let clause = if take_clause { first_clause(&stripped) } else { &stripped[..] };
    let lowered = clause.to_lowercase();
    let mut out = String::with_capacity(lowered.len());
    let mut last_space = true;
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
            last_space = false;
        } else if mapped.is_whitespace() || mapped == ',' {
            if !last_space && !out.is_empty() {
                out.push(' ');
                last_space = true;
            }
        }
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

fn sanitize_prompt(s: &str) -> String {
    sanitize_inner(s, MAX_PROMPT_BYTES, false)
}

fn sanitize(s: &str) -> String {
    sanitize_inner(s, MAX_RESPONSE_BYTES, true)
}

fn encode_record(prompt: &str, response: &str) -> Vec<u8> {
    let p = sanitize_prompt(prompt);
    let r = sanitize(response);
    let pblock = path_c::encode_text(&p).expect("text block");
    let rblock = path_c::encode_text(&r).expect("text block");
    let pad = WINDOW.saturating_sub(pblock.len());
    let mut out = vec![PAD_BYTE; pad];
    out.extend_from_slice(&pblock);
    out.extend_from_slice(&rblock);
    out.push(PAD_BYTE);
    out
}

fn prompt_window_bytes(prompt: &str) -> Vec<u8> {
    let p = sanitize_prompt(prompt);
    let pblock = path_c::encode_text(&p).expect("text block");
    let pad = WINDOW.saturating_sub(pblock.len());
    let mut out = vec![PAD_BYTE; pad];
    out.extend_from_slice(&pblock);
    out
}

struct VocabMap {
    byte_to_id: HashMap<u8, u32>,
    id_to_byte: Vec<u8>,
}

impl VocabMap {
    fn build(streams: &[Vec<u8>]) -> Self {
        let unique: BTreeSet<u8> = streams.iter().flatten().copied().collect();
        let id_to_byte: Vec<u8> = unique.into_iter().collect();
        let mut byte_to_id = HashMap::new();
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
    let mut curr = vec![0_usize; n + 1];
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

/// Build training pairs from streams using sliding window.
fn build_pairs(streams: &[Vec<u32>]) -> Vec<(Vec<u32>, u32)> {
    let mut pairs = Vec::new();
    for s in streams {
        if s.len() <= WINDOW {
            continue;
        }
        for i in 0..s.len() - WINDOW {
            pairs.push((s[i..i + WINDOW].to_vec(), s[i + WINDOW]));
        }
    }
    pairs
}

/// Greedy autoregressive generation (works for both Mlp and SmgGrid through closure).
fn generate<F: Fn(&[u32]) -> u32>(
    predict: F,
    initial: Vec<u32>,
    pad_id: Option<u32>,
    max_steps: usize,
) -> Vec<u32> {
    let mut buf = initial;
    let initial_len = buf.len();
    for _ in 0..max_steps {
        let window: Vec<u32> = if buf.len() >= WINDOW {
            buf[buf.len() - WINDOW..].to_vec()
        } else {
            let mut w = vec![pad_id.unwrap_or(0); WINDOW - buf.len()];
            w.extend(&buf);
            w
        };
        let next = predict(&window);
        buf.push(next);
        if Some(next) == pad_id {
            break;
        }
    }
    buf[initial_len..].to_vec()
}

fn ids_to_string(decoded: &[u8]) -> String {
    let mut out = String::new();
    for &b in decoded {
        if b.is_ascii_alphanumeric() || b == b' ' {
            out.push(b as char);
        }
    }
    out.trim().to_string()
}

#[derive(Debug, Clone)]
struct EvalRow {
    prompt: String,
    expected: String,
    out_mlp: String,
    out_grid: String,
    edit_mlp: usize,
    edit_grid: usize,
    max_len: usize,
}

fn eval_on_split(
    sanitized: &[(String, String)],
    vocab: &VocabMap,
    mlp: &Mlp,
    grid: &SmgGrid,
) -> (Vec<EvalRow>, f64, f64, usize, usize) {
    let pad_id = vocab.pad_id();
    let mut rows = Vec::new();
    let mut sum_agree_mlp = 0.0_f64;
    let mut sum_agree_grid = 0.0_f64;
    let mut exact_mlp = 0_usize;
    let mut exact_grid = 0_usize;

    for (p, expected) in sanitized {
        let pwin = prompt_window_bytes(p);
        let initial = vocab.encode(&pwin);
        let max_steps = (expected.len() + 16).max(8);

        let gen_mlp_ids = generate(|w| mlp.predict(w), initial.clone(), pad_id, max_steps);
        let gen_grid_ids = generate(|w| grid.predict(w), initial.clone(), pad_id, max_steps);

        let s_mlp = ids_to_string(&vocab.decode(&gen_mlp_ids));
        let s_grid = ids_to_string(&vocab.decode(&gen_grid_ids));

        let exp_b = expected.as_bytes();
        let edit_mlp = levenshtein(exp_b, s_mlp.as_bytes());
        let edit_grid = levenshtein(exp_b, s_grid.as_bytes());
        let max_mlp = exp_b.len().max(s_mlp.len()).max(1);
        let max_grid = exp_b.len().max(s_grid.len()).max(1);
        let max_len = max_mlp.max(max_grid);
        sum_agree_mlp += 1.0 - edit_mlp as f64 / max_mlp as f64;
        sum_agree_grid += 1.0 - edit_grid as f64 / max_grid as f64;
        if exp_b == s_mlp.as_bytes() {
            exact_mlp += 1;
        }
        if exp_b == s_grid.as_bytes() {
            exact_grid += 1;
        }
        rows.push(EvalRow {
            prompt: p.clone(),
            expected: expected.clone(),
            out_mlp: s_mlp,
            out_grid: s_grid,
            edit_mlp,
            edit_grid,
            max_len,
        });
    }
    let n = sanitized.len() as f64;
    (rows, sum_agree_mlp / n, sum_agree_grid / n, exact_mlp, exact_grid)
}

fn main() -> std::io::Result<()> {
    println!("[exp17] SMG-grid (4×4 microcolunas FSM) vs MLP, com generalização\n");

    let traces = match read_traces(TRACES_PATH) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[exp17] Não consegui ler {TRACES_PATH}: {e}\n\
                Rode primeiro o pipeline de exp16 (ver docs/exp16-runbook.md).");
            std::process::exit(1);
        }
    };
    println!("[exp17] {} traces carregados", traces.len());

    // ── Sanitiza ───────────────────────────────────────────────────────────
    let sanitized: Vec<(String, String)> = traces
        .iter()
        .map(|r| (sanitize_prompt(&r.prompt), sanitize(&r.response)))
        .filter(|(p, r)| !p.is_empty() && !r.is_empty())
        .collect();
    println!("[exp17] {} records após sanitize", sanitized.len());

    // ── Split 80/20 determinístico ──────────────────────────────────────────
    let mut indices: Vec<usize> = (0..sanitized.len()).collect();
    let mut rng = Rng::new(SPLIT_SEED);
    for i in (1..indices.len()).rev() {
        let j = rng.range_u32(i as u32 + 1) as usize;
        indices.swap(i, j);
    }
    let split = (sanitized.len() as f64 * TRAIN_FRAC) as usize;
    let train_idx: Vec<usize> = indices[..split].to_vec();
    let test_idx: Vec<usize> = indices[split..].to_vec();
    let train_set: Vec<(String, String)> = train_idx.iter().map(|&i| sanitized[i].clone()).collect();
    let test_set: Vec<(String, String)> = test_idx.iter().map(|&i| sanitized[i].clone()).collect();
    println!("[exp17] split: {} train / {} test", train_set.len(), test_set.len());

    // ── Encoda streams (vocab construído sobre TRAIN apenas; test pode ter \
    //    bytes não vistos, mas no nosso domínio sanitizado tudo é [a-z0-9 ]) ──
    let mut all_streams: Vec<Vec<u8>> = Vec::with_capacity(train_set.len() * REPS_PER_PAIR);
    for _ in 0..REPS_PER_PAIR {
        for (p, r) in &train_set {
            all_streams.push(encode_record(p, r));
        }
    }
    // Garante que vocab inclua bytes que possam aparecer só no test (por sanidade).
    // Adiciona 1 stream de cada test pair (sem usar para treino).
    let mut vocab_seed = all_streams.clone();
    for (p, r) in &test_set {
        vocab_seed.push(encode_record(p, r));
    }

    let vocab = VocabMap::build(&vocab_seed);
    println!("[exp17] vocab_size = {} bytes únicos", vocab.vocab_size());

    let id_train_streams: Vec<Vec<u32>> = all_streams.iter().map(|s| vocab.encode(s)).collect();
    let mut pairs = build_pairs(&id_train_streams);
    println!("[exp17] training pairs = {}", pairs.len());

    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    println!("[exp17] {} threads paralelos\n", n_threads);

    // ── Treina MLP baseline ────────────────────────────────────────────────
    println!("[exp17] === Treinando MLP baseline ===");
    let mlp_cfg = MlpConfig {
        vocab_size: vocab.vocab_size(),
        window: WINDOW,
        hidden: HIDDEN,
        learning_rate: LR_MLP,
        seed: SEED,
    };
    let mut mlp = Mlp::new(mlp_cfg.clone());

    let t0 = std::time::Instant::now();
    let mut shuffler = Rng::new(SEED);
    let mut mlp_loss_curve: Vec<(usize, f32)> = Vec::new();
    for epoch in 0..EPOCHS {
        for i in (1..pairs.len()).rev() {
            let j = shuffler.range_u32(i as u32 + 1) as usize;
            pairs.swap(i, j);
        }
        let l = mlp.train_epoch_parallel(&pairs, n_threads, BATCH_SIZE);
        if epoch == 0 || (epoch + 1) % 5 == 0 || epoch == EPOCHS - 1 {
            mlp_loss_curve.push((epoch, l));
            println!("  MLP  epoch {:3}: loss = {:.4}", epoch + 1, l);
        }
    }
    let mlp_time = t0.elapsed().as_secs_f64();
    println!("[exp17] MLP treinado em {:.1}s\n", mlp_time);

    // ── Treina SMG-grid ────────────────────────────────────────────────────
    println!("[exp17] === Treinando SMG-grid ===");
    let grid_cfg = GridConfig {
        vocab_size: vocab.vocab_size(),
        window: WINDOW,
        learning_rate: LR_GRID,
        seed: SEED,
    };
    let mut grid = SmgGrid::new(grid_cfg);
    println!("[exp17] grid params = {}", grid.n_params());

    let t1 = std::time::Instant::now();
    let mut shuffler = Rng::new(SEED + 1);
    let mut grid_loss_curve: Vec<(usize, f32)> = Vec::new();
    let mut grid_diverged = false;
    let div_thr = (vocab.vocab_size() as f32).ln() * 3.0;
    for epoch in 0..EPOCHS {
        for i in (1..pairs.len()).rev() {
            let j = shuffler.range_u32(i as u32 + 1) as usize;
            pairs.swap(i, j);
        }
        let l = grid.train_epoch_parallel(&pairs, n_threads, BATCH_SIZE);
        if epoch == 0 || (epoch + 1) % 5 == 0 || epoch == EPOCHS - 1 {
            grid_loss_curve.push((epoch, l));
            println!("  GRID epoch {:3}: loss = {:.4}", epoch + 1, l);
        }
        if !l.is_finite() || l > div_thr {
            eprintln!("[exp17] GRID DIVERGED em epoch {}: loss={:.2}", epoch + 1, l);
            grid_diverged = true;
            break;
        }
    }
    let grid_time = t1.elapsed().as_secs_f64();
    println!("[exp17] SMG-grid treinado em {:.1}s\n", grid_time);

    // ── Avaliação no train e test ──────────────────────────────────────────
    println!("[exp17] === Avaliando no train e test ===");
    let (train_rows, train_agree_mlp, train_agree_grid, train_exact_mlp, train_exact_grid) =
        eval_on_split(&train_set, &vocab, &mlp, &grid);
    let (test_rows, test_agree_mlp, test_agree_grid, test_exact_mlp, test_exact_grid) =
        eval_on_split(&test_set, &vocab, &mlp, &grid);

    println!("[exp17] TRAIN  MLP agree={:.1}% exact={}/{}, GRID agree={:.1}% exact={}/{}",
        train_agree_mlp * 100.0, train_exact_mlp, train_set.len(),
        train_agree_grid * 100.0, train_exact_grid, train_set.len());
    println!("[exp17] TEST   MLP agree={:.1}% exact={}/{}, GRID agree={:.1}% exact={}/{}",
        test_agree_mlp * 100.0, test_exact_mlp, test_set.len(),
        test_agree_grid * 100.0, test_exact_grid, test_set.len());

    // ── Markdown ──────────────────────────────────────────────────────────
    std::fs::create_dir_all("results")?;
    let mut out = std::fs::File::create(RESULTS_PATH)?;
    writeln!(out, "# Experimento 17 — SMG-grid vs MLP, generalização\n")?;
    writeln!(out, "Pergunta: substituindo MLP plano por grade 4×4 de microcolunas FSM (\
        ~32× menos parâmetros), a estrutura **generaliza melhor** para prompts não vistos?\n")?;
    writeln!(out, "Ver `docs/exp17-design.md` §1 para critério.\n")?;

    writeln!(out, "## Setup\n")?;
    writeln!(out, "- Teacher: Qwen 2.5 1.5B (mesmas 100 traces de exp16).")?;
    writeln!(out, "- Sanitize idêntico: prompt {MAX_PROMPT_BYTES}b max, response {MAX_RESPONSE_BYTES}b max.")?;
    writeln!(out, "- Split: {} train / {} test, seed={SPLIT_SEED}.", train_set.len(), test_set.len())?;
    writeln!(out, "- Encoding: Path C tagged `[T:N prompt][T:M response]\\xFE`. Vocab={}.", vocab.vocab_size())?;
    writeln!(out, "- Window={WINDOW}, batch={BATCH_SIZE}, epochs={EPOCHS}, reps={REPS_PER_PAIR}, threads={n_threads}.")?;
    writeln!(out, "- Pairs de treino: {}.", pairs.len())?;
    writeln!(out)?;

    writeln!(out, "### Modelos\n")?;
    writeln!(out, "| | params | LR | tempo |")?;
    writeln!(out, "|---|---:|---:|---:|")?;
    let mlp_params = mlp_cfg.window * mlp_cfg.vocab_size * mlp_cfg.hidden
        + mlp_cfg.hidden
        + mlp_cfg.hidden * mlp_cfg.vocab_size
        + mlp_cfg.vocab_size;
    writeln!(out, "| MLP | {} | {LR_MLP} | {:.1}s |", mlp_params, mlp_time)?;
    writeln!(out, "| SMG-grid | {} | {LR_GRID} | {:.1}s |", grid.n_params(), grid_time)?;
    writeln!(out)?;

    writeln!(out, "## Loss curves\n")?;
    writeln!(out, "| Epoch | MLP loss | Grid loss |")?;
    writeln!(out, "|------:|---------:|----------:|")?;
    for ((em, lm), (eg, lg)) in mlp_loss_curve.iter().zip(grid_loss_curve.iter()) {
        let _ = em;
        writeln!(out, "| {} | {:.4} | {:.4} |", eg + 1, lm, lg)?;
    }
    if grid_diverged {
        writeln!(out, "\n**SMG-grid divergiu durante treino.**\n")?;
    }
    writeln!(out)?;

    writeln!(out, "## Métricas finais\n")?;
    writeln!(out, "| | byte-agreement train | exact train | byte-agreement TEST | exact TEST |")?;
    writeln!(out, "|---|---:|---:|---:|---:|")?;
    writeln!(out, "| MLP | {:.1}% | {}/{} | **{:.1}%** | {}/{} |",
        train_agree_mlp * 100.0, train_exact_mlp, train_set.len(),
        test_agree_mlp * 100.0, test_exact_mlp, test_set.len())?;
    writeln!(out, "| SMG-grid | {:.1}% | {}/{} | **{:.1}%** | {}/{} |",
        train_agree_grid * 100.0, train_exact_grid, train_set.len(),
        test_agree_grid * 100.0, test_exact_grid, test_set.len())?;
    writeln!(out)?;

    writeln!(out, "## Test set (prompts não vistos)\n")?;
    writeln!(out, "| Prompt | Qwen (gold) | MLP | SMG-grid | edit MLP | edit Grid |")?;
    writeln!(out, "|--------|-------------|-----|----------|---------:|----------:|")?;
    for r in &test_rows {
        let p_disp = if r.prompt.len() > 36 { format!("{}…", &r.prompt[..36]) } else { r.prompt.clone() };
        let e_disp = if r.expected.len() > 36 { format!("{}…", &r.expected[..36]) } else { r.expected.clone() };
        let m_disp = if r.out_mlp.len() > 36 { format!("{}…", &r.out_mlp[..36]) } else { r.out_mlp.clone() };
        let g_disp = if r.out_grid.len() > 36 { format!("{}…", &r.out_grid[..36]) } else { r.out_grid.clone() };
        writeln!(out, "| `{p_disp}` | `{e_disp}` | `{m_disp}` | `{g_disp}` | {} | {} |",
            r.edit_mlp, r.edit_grid)?;
    }
    writeln!(out)?;

    writeln!(out, "## Train set (sanity check, prompts vistos)\n")?;
    writeln!(out, "<details><summary>{} prompts (clicar para expandir)</summary>\n", train_rows.len())?;
    writeln!(out, "| Prompt | Qwen (gold) | MLP | SMG-grid | edit MLP | edit Grid |")?;
    writeln!(out, "|--------|-------------|-----|----------|---------:|----------:|")?;
    for r in &train_rows {
        let p_disp = if r.prompt.len() > 36 { format!("{}…", &r.prompt[..36]) } else { r.prompt.clone() };
        let e_disp = if r.expected.len() > 36 { format!("{}…", &r.expected[..36]) } else { r.expected.clone() };
        let m_disp = if r.out_mlp.len() > 36 { format!("{}…", &r.out_mlp[..36]) } else { r.out_mlp.clone() };
        let g_disp = if r.out_grid.len() > 36 { format!("{}…", &r.out_grid[..36]) } else { r.out_grid.clone() };
        writeln!(out, "| `{p_disp}` | `{e_disp}` | `{m_disp}` | `{g_disp}` | {} | {} |",
            r.edit_mlp, r.edit_grid)?;
    }
    writeln!(out, "</details>\n")?;

    // ── Veredicto ──────────────────────────────────────────────────────────
    let test_diff = test_agree_grid - test_agree_mlp;
    writeln!(out, "## Veredicto\n")?;
    let verdict = if grid_diverged {
        "**? Inválido** — SMG-grid divergiu. Reduzir LR ou inicializar diferente.".to_string()
    } else if test_diff > 0.05 {
        format!(
            "**✓ Estrutura agrega valor** — SMG-grid (~32× menos params) supera MLP em \
             test por +{:.1}pp byte-agreement. Avançar para exp18 (hierarquia 6 níveis + bus).",
            test_diff * 100.0
        )
    } else if test_diff.abs() <= 0.05 {
        format!(
            "**~ Estrutura é neutra** — diferença em test = {:+.1}pp, dentro do ruído. \
             Estrutura por si só não generaliza melhor; valor virá da hierarquia + bus (exp18-19) \
             ou é necessário rever a tese.",
            test_diff * 100.0
        )
    } else {
        format!(
            "**✗ Estrutura piora** — SMG-grid perde {:.1}pp para MLP em test. \
             Possíveis causas: BPTT instável, K muito pequeno, ou tese 'byte-language gera \
             estrutura' precisa ser repensada.",
            -test_diff * 100.0
        )
    };
    writeln!(out, "{verdict}")?;
    writeln!(out)?;

    println!("\n[exp17] resultados gravados em {RESULTS_PATH}");
    println!("[exp17] {verdict}");
    Ok(())
}
