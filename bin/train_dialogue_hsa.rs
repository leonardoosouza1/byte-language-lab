//! train_dialogue_hsa — Treino byte-level em corpus de diálogo PT-BR.
//!
//! Treina SMG v3 num corpus de diálogo em formato USER/MODEL e guarda
//! os pesos em `dialogue_weights.blab` para uso posterior pelo `chat_hsa`.
//!
//! Formato do corpus (cada troca numa linha separada por `\n\n`):
//! ```
//! USER: OI
//! MODEL: Olá! Como posso te ajudar?
//!
//! USER: Tudo bem?
//! MODEL: Sim, muito bem! E você?
//! ```
//!
//! Depois do treino, o binário:
//!   1. Reporta loss de treino/validação por época.
//!   2. Gera samples autoregressivos a partir de "USER: OI\nMODEL: ".
//!   3. Salva `dialogue_weights.blab` (usado por `chat_hsa`).
//!
//! ```bash
//! cargo run --release --features gpu-aql --bin train_dialogue_hsa
//! cargo run --release --features gpu-aql --bin train_dialogue_hsa -- corpus.txt pesos.blab
//! ```

#![cfg(feature = "gpu-aql")]
#![allow(unsafe_code)]

use std::io::{Read as _, Write as _};
use std::time::Instant;

use blab::{
    gpu_aql::{SmgGridHsa, WeightsHost},
    learn::microcolumn_v3::{GridConfig, SmgGrid},
};

// ---- Hiperparâmetros ----
const WINDOW:          usize = 48;
const VOCAB:           usize = 256;
const LR:              f32   = 0.003;
const BATCH:           usize = 64;
const N_EPOCHS:        usize = 60;
const STEPS_PER_EPOCH: usize = 200;
const VAL_STEPS:       usize = 40;
const SEED:            u64   = 7;
const GEN_LEN:         usize = 96;   // bytes para demo de geração

// ---- Corpus PT-BR embutido ----
const EMBEDDED_CORPUS: &str = "\
USER: OI
MODEL: Olá! Como posso te ajudar hoje?

USER: Tudo bem?
MODEL: Sim, estou ótimo! E você, como está?

USER: Boa tarde
MODEL: Boa tarde! Em que posso te ajudar?

USER: Boa noite
MODEL: Boa noite! Como posso ser útil?

USER: Olá
MODEL: Olá! Que bom falar com você. O que você precisa?

USER: O que você é?
MODEL: Sou um assistente de inteligência artificial treinado para conversar.

USER: Você é uma IA?
MODEL: Sim, sou um modelo de linguagem baseado em bytes, rodando em GPU AMD.

USER: Como você funciona?
MODEL: Aprendo padrões de texto byte a byte usando redes neurais em GPU.

USER: O que você sabe fazer?
MODEL: Posso conversar, responder perguntas e gerar texto em português.

USER: Fale sobre programação
MODEL: Programação é a arte de criar instruções para computadores executarem tarefas.

USER: O que é Rust?
MODEL: Rust é uma linguagem de programação focada em segurança e desempenho.

USER: O que é Python?
MODEL: Python é uma linguagem de programação de alto nível, fácil de aprender.

USER: O que é GPU?
MODEL: GPU é uma unidade de processamento gráfico, ótima para cálculos paralelos.

USER: O que é inteligência artificial?
MODEL: Inteligência artificial é um campo da computação que simula raciocínio humano.

USER: O que é aprendizado de máquina?
MODEL: Aprendizado de máquina permite que sistemas aprendam com dados automaticamente.

USER: Você gosta de conversar?
MODEL: Sim! Adoro conversar e ajudar as pessoas com suas dúvidas.

USER: Qual é a sua linguagem favorita?
MODEL: Como fui feito em Rust, tenho um carinho especial por essa linguagem!

USER: Você pode me ajudar com código?
MODEL: Claro! Posso ajudar com dúvidas sobre código, algoritmos e lógica.

USER: Como vai o tempo?
MODEL: Não tenho acesso a dados meteorológicos, mas espero que esteja bom!

USER: Qual é a capital do Brasil?
MODEL: A capital do Brasil é Brasília, cidade planejada inaugurada em 1960.

USER: Quantos estados tem o Brasil?
MODEL: O Brasil tem 26 estados mais o Distrito Federal, totalizando 27 unidades.

USER: O que é ROCm?
MODEL: ROCm é a plataforma de computação GPU da AMD, usada para acelerar IA.

USER: O que é HSA?
MODEL: HSA é a arquitetura heterogênea que permite CPU e GPU trabalharem juntos.

USER: O que é RDNA2?
MODEL: RDNA2 é a arquitetura de GPU da AMD da segunda geração, muito eficiente.

USER: O que é aprendizado por reforço?
MODEL: É um tipo de aprendizado onde um agente aprende por tentativa e erro.

USER: Me conta uma curiosidade
MODEL: Sabia que o primeiro computador pessoal foi lançado nos anos 1970?

USER: O que é byte?
MODEL: Um byte é uma unidade de dados composta por 8 bits, podendo representar 256 valores.

USER: O que é bit?
MODEL: Um bit é a menor unidade de informação, podendo ser 0 ou 1.

USER: Você tem memória?
MODEL: Tenho memória limitada ao contexto da conversa atual de 48 bytes.

USER: Você aprende com a conversa?
MODEL: Aprendo durante o treino, não em tempo real durante a conversa.

USER: Como posso melhorar em programação?
MODEL: Pratique diariamente, leia código de outros, e construa projetos próprios.

USER: O que é algoritmo?
MODEL: Um algoritmo é uma sequência de passos para resolver um problema.

USER: O que é compilador?
MODEL: Um compilador converte código fonte em código executável pela máquina.

USER: O que é kernel?
MODEL: Em GPU, kernel é uma função executada em paralelo por muitos núcleos.

USER: O que é paralelismo?
MODEL: Paralelismo é executar múltiplas operações ao mesmo tempo, em vários núcleos.

USER: Obrigado
MODEL: De nada! Fico feliz em ajudar. Até logo!

USER: Até logo
MODEL: Até logo! Foi um prazer conversar com você. Volte sempre!

USER: Tchau
MODEL: Tchau! Tenha um ótimo dia!

USER: Muito obrigado
MODEL: Por nada! Estou sempre aqui para ajudar quando precisar.

USER: Você é inteligente?
MODEL: Sou um modelo simples, mas faço o meu melhor para ser útil!

USER: Qual é o seu nome?
MODEL: Pode me chamar de IARA, um assistente de linguagem em Rust.

USER: De onde você vem?
MODEL: Fui criado num projeto de pesquisa de modelos de linguagem byte-level.

USER: Você fala outros idiomas?
MODEL: Fui treinado principalmente em português, mas entendo algum inglês.

USER: O que é transformador?
MODEL: Transformer é uma arquitetura de rede neural usada em modelos de linguagem.

USER: O que é atenção em IA?
MODEL: Mecanismo de atenção permite ao modelo focar em partes relevantes do texto.

USER: O que é embeddings?
MODEL: Embeddings são representações numéricas de palavras ou bytes em vetores.

USER: O que é gradiente?
MODEL: Gradiente indica a direção de maior crescimento de uma função, usado no treino.

USER: O que é backpropagation?
MODEL: Backpropagation é o algoritmo que calcula gradientes para treinar redes neurais.

USER: Como você foi treinado?
MODEL: Fui treinado com gradiente descendente estocástico em corpus de texto byte a byte.

USER: OI
MODEL: Olá de novo! Como posso te ajudar?

USER: oi tudo bem
MODEL: Tudo ótimo, obrigado! E com você, como está?

USER: oi como vai
MODEL: Vou muito bem! Em que posso ser útil hoje?

USER: Preciso de ajuda
MODEL: Claro! Estou aqui. Me diga o que você precisa.

USER: Pode me explicar isso?
MODEL: Com prazer! Me explique melhor o que você gostaria de entender.

USER: Não entendi
MODEL: Sem problema! Vou tentar explicar de outra forma. O que não ficou claro?

USER: Pode repetir?
MODEL: Claro! Vou repetir de forma mais simples para facilitar o entendimento.

USER: Sim
MODEL: Ótimo! Posso continuar com mais detalhes se precisar.

USER: Não
MODEL: Tudo bem! Se precisar de algo diferente, é só perguntar.

USER: Talvez
MODEL: Entendo! Se quiser mais informações para decidir, estou à disposição.

";

// ---------------------------------------------------------------------------
// Formato .blab — magic + version + 9 tensores (u32 length + f32 data)
// ---------------------------------------------------------------------------

const BLAB_MAGIC:   u32 = 0x424C4142;   // 'BLAB'
const BLAB_VERSION: u32 = 1;

fn save_weights(path: &str, wh: &WeightsHost) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;

    // Cabeçalho: magic + version
    f.write_all(&BLAB_MAGIC.to_le_bytes())?;
    f.write_all(&BLAB_VERSION.to_le_bytes())?;

    // Cada tensor: length (u32) + dados (f32 × length)
    for slice in [
        wh.t_f.as_slice(),
        wh.i_f.as_slice(),
        wh.nibble_h.as_slice(),
        wh.nibble_l.as_slice(),
        wh.b_f.as_slice(),
        wh.pos_bias.as_slice(),
        wh.lat_w.as_slice(),
        wh.w_out.as_slice(),
        wh.b_out.as_slice(),
    ] {
        f.write_all(&(slice.len() as u32).to_le_bytes())?;
        let bytes = unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4)
        };
        f.write_all(bytes)?;
    }

    f.flush()?;
    Ok(())
}

fn load_weights(path: &str) -> std::io::Result<WeightsHost> {
    let data = std::fs::read(path)?;
    let mut cur = 0usize;

    let read_u32 = |cur: &mut usize, data: &[u8]| -> std::io::Result<u32> {
        if *cur + 4 > data.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "EOF"));
        }
        let v = u32::from_le_bytes(data[*cur..*cur+4].try_into().unwrap());
        *cur += 4;
        Ok(v)
    };

    let magic   = read_u32(&mut cur, &data)?;
    let version = read_u32(&mut cur, &data)?;
    if magic != BLAB_MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("magic inválido: 0x{:08X}", magic)));
    }
    if version != BLAB_VERSION {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("versão não suportada: {}", version)));
    }

    let read_tensor = |cur: &mut usize, data: &[u8]| -> std::io::Result<Vec<f32>> {
        let n = read_u32(cur, data)? as usize;
        let byte_len = n * 4;
        if *cur + byte_len > data.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "EOF tensor"));
        }
        let mut v = vec![0.0f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                data[*cur..].as_ptr(),
                v.as_mut_ptr() as *mut u8,
                byte_len,
            );
        }
        *cur += byte_len;
        Ok(v)
    };

    Ok(WeightsHost {
        t_f:      read_tensor(&mut cur, &data)?,
        i_f:      read_tensor(&mut cur, &data)?,
        nibble_h: read_tensor(&mut cur, &data)?,
        nibble_l: read_tensor(&mut cur, &data)?,
        b_f:      read_tensor(&mut cur, &data)?,
        pos_bias: read_tensor(&mut cur, &data)?,
        lat_w:    read_tensor(&mut cur, &data)?,
        w_out:    read_tensor(&mut cur, &data)?,
        b_out:    read_tensor(&mut cur, &data)?,
    })
}

// ---------------------------------------------------------------------------
// Utilitários
// ---------------------------------------------------------------------------

fn shuffle<T>(v: &mut Vec<T>, seed: u64) {
    let mut rng = seed;
    for i in (1..v.len()).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng >> 33) as usize % (i + 1);
        v.swap(i, j);
    }
}

fn softmax_sample(logits: &[f32], temperature: f32, rng: &mut u64) -> usize {
    if temperature == 0.0 {
        return logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
    }
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    let mx = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - mx).exp()).collect();
    let s: f32 = exps.iter().sum();
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let u = (*rng >> 11) as f32 / (1u64 << 53) as f32;
    let mut cdf = 0.0f32;
    for (i, &e) in exps.iter().enumerate() {
        cdf += e / s;
        if u <= cdf { return i; }
    }
    exps.len() - 1
}

fn make_pairs(data: &[u8]) -> Vec<(Vec<u32>, u32)> {
    if data.len() <= WINDOW { return Vec::new(); }
    data.windows(WINDOW + 1)
        .map(|w| {
            let win: Vec<u32> = w[..WINDOW].iter().map(|&b| b as u32).collect();
            (win, w[WINDOW] as u32)
        })
        .collect()
}

fn val_loss(hsa: &SmgGridHsa, val_pairs: &[(Vec<u32>, u32)]) -> f32 {
    let n = (VAL_STEPS * BATCH).min(val_pairs.len());
    if n == 0 { return f32::NAN; }
    let mut total = 0.0f32;
    let mut nb    = 0usize;
    for chunk in val_pairs[..n].chunks(BATCH) {
        if chunk.is_empty() { continue; }
        let windows: Vec<Vec<u32>> = chunk.iter().map(|(w,_)| w.clone()).collect();
        let targets: Vec<u32>      = chunk.iter().map(|(_,t)| *t).collect();
        let logits_batch = hsa.forward_batch(&windows);
        let mut loss_batch = 0.0f32;
        for (logits, &tgt) in logits_batch.iter().zip(&targets) {
            let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&x| (x-mx).exp()).collect();
            let s: f32 = exps.iter().sum();
            loss_batch -= (exps[tgt as usize] / s).max(1e-20).ln();
        }
        total += loss_batch / chunk.len() as f32;
        nb    += 1;
    }
    if nb > 0 { total / nb as f32 } else { f32::NAN }
}

/// Geração autoregressiva a partir de um prompt de texto.
fn generate_text(hsa: &SmgGridHsa, prompt: &str, max_len: usize,
                 temperature: f32, stop_at_newline: bool, rng: &mut u64) -> String {
    let pbytes = prompt.as_bytes();
    let mut window: Vec<u32> = if pbytes.len() >= WINDOW {
        pbytes[pbytes.len() - WINDOW..].iter().map(|&b| b as u32).collect()
    } else {
        let pad = WINDOW - pbytes.len();
        std::iter::repeat(b'\n' as u32).take(pad)
            .chain(pbytes.iter().map(|&b| b as u32))
            .collect()
    };

    let mut out = String::new();
    for _ in 0..max_len {
        let logits = &hsa.forward_batch(&[window.clone()])[0];
        let next = softmax_sample(logits, temperature, rng) as u8;
        if stop_at_newline && next == b'\n' { break; }
        if next.is_ascii_graphic() || next == b' ' {
            out.push(next as char);
        } else if next == b'\n' {
            out.push('\n');
        } else {
            out.push('·');
        }
        window.remove(0);
        window.push(next as u32);
    }
    out
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== train_dialogue_hsa: treino de diálogo PT-BR em RDNA2 ===");
    println!("  window={WINDOW}  vocab={VOCAB}  batch={BATCH}  lr={LR:.4}");
    println!("  {N_EPOCHS} epochs × {STEPS_PER_EPOCH} steps/epoch\n");

    // ---- Argumentos ----
    let args: Vec<String> = std::env::args().collect();
    let corpus_path  = args.get(1).map(|s| s.as_str());
    let weights_path = args.get(2).map(|s| s.as_str()).unwrap_or("dialogue_weights.blab");

    // ---- Corpus ----
    let corpus_text: String = match corpus_path {
        Some(path) => {
            println!("Corpus: {}", path);
            let mut s = String::new();
            std::fs::File::open(path)?.read_to_string(&mut s)?;
            s
        }
        None => {
            println!("Corpus: diálogo PT-BR embutido ({} bytes)", EMBEDDED_CORPUS.len());
            // Repete o corpus várias vezes para ter mais pares de treino
            EMBEDDED_CORPUS.repeat(40)
        }
    };
    let data = corpus_text.as_bytes().to_vec();
    println!("  {} bytes totais\n", data.len());

    let mut all_pairs = make_pairs(&data);
    if all_pairs.len() < WINDOW * 2 {
        eprintln!("Corpus demasiado pequeno. Use um ficheiro maior.");
        std::process::exit(1);
    }
    let n_val   = (all_pairs.len() / 8).max(VAL_STEPS * BATCH);
    let n_train = all_pairs.len() - n_val;
    let val_pairs:    Vec<_> = all_pairs.split_off(n_train);
    let mut train_pairs      = all_pairs;
    println!("Pares: {} treino | {} validação\n", train_pairs.len(), val_pairs.len());

    // ---- Modelo ----
    let cfg = GridConfig {
        vocab_size:    VOCAB,
        window:        WINDOW,
        learning_rate: LR,
        seed:          SEED,
    };
    let cpu_init = SmgGrid::new(cfg.clone());
    let mut hsa  = SmgGridHsa::from_cpu(&cpu_init, BATCH)?;
    println!("Modelo: {} parâmetros\n", hsa.n_params());

    // ---- Treino ----
    println!("{:>5}  {:>10}  {:>10}  {:>8}", "época", "train_loss", "val_loss", "tempo");
    println!("{}", "─".repeat(42));

    let t_train = Instant::now();
    let mut rng_seed = SEED;

    for ep in 0..N_EPOCHS {
        // LR schedule: warm-up primeiras 10%, plateau até 75%, cosine decay até fim
        let warmup_end = (N_EPOCHS / 10).max(1);
        let decay_start = (N_EPOCHS * 3 / 4).max(warmup_end + 1);
        let lr = if ep < warmup_end {
            LR * (ep + 1) as f32 / warmup_end as f32
        } else if ep < decay_start {
            LR
        } else {
            let t = (ep - decay_start) as f32 / (N_EPOCHS - decay_start).max(1) as f32;
            (LR * 0.5 * (1.0 + (std::f32::consts::PI * t).cos())).max(LR * 0.05)
        };
        hsa.set_lr(lr);

        rng_seed = rng_seed.wrapping_add(ep as u64 * 0x9e3779b97f4a7c15);
        shuffle(&mut train_pairs, rng_seed);

        let epoch_pairs: Vec<_> = train_pairs
            .iter().take(STEPS_PER_EPOCH * BATCH).cloned().collect();

        let t_ep = Instant::now();
        let tl   = hsa.train_epoch(&epoch_pairs, BATCH);
        let vl   = val_loss(&hsa, &val_pairs);
        let dur  = t_ep.elapsed();

        println!("{:>5}  {:>10.4}  {:>10.4}  {:>7.2?}", ep + 1, tl, vl, dur);
    }

    let train_elapsed = t_train.elapsed();
    println!("\nTreino total: {:.1?}", train_elapsed);

    let final_vl = val_loss(&hsa, &val_pairs);
    println!("Val final: {:.4} nats ({:.3} bits/byte)\n",
        final_vl, final_vl / std::f32::consts::LN_2);

    // ---- Demonstração de geração ----
    println!("━━━ Demonstração de Geração ━━━\n");
    let mut gen_rng = SEED ^ 0xf00dcafe;

    let demo_prompts = [
        "USER: OI\nMODEL: ",
        "USER: Tudo bem?\nMODEL: ",
        "USER: O que você é?\nMODEL: ",
        "USER: Qual é o seu nome?\nMODEL: ",
        "USER: O que é Rust?\nMODEL: ",
        "USER: Obrigado\nMODEL: ",
    ];

    for prompt in &demo_prompts {
        let resp = generate_text(&hsa, prompt, GEN_LEN, 0.7, true, &mut gen_rng);
        let user_part = prompt.trim_end_matches("MODEL: ").trim_start_matches('\n');
        println!("  {}", user_part);
        println!("  → {}", resp.trim());
        println!();
    }

    // ---- Salvar pesos ----
    println!("━━━ Salvando pesos em '{}' ━━━", weights_path);
    let wh = hsa.weights_to_host();
    save_weights(weights_path, &wh)?;
    let file_size = std::fs::metadata(weights_path)?.len();
    println!("  Salvo: {:.2} MB\n", file_size as f64 / 1_048_576.0);

    // ---- Verificação de round-trip ----
    print!("  Verificando round-trip load → upload... ");
    let wh2 = load_weights(weights_path)?;
    hsa.upload_weights(&wh2)?;
    let vl_after = val_loss(&hsa, &val_pairs);
    println!(
        "val={:.4} (delta={:+.6})",
        vl_after, vl_after - final_vl
    );

    println!("\n=== Concluído. Use 'cargo run --features gpu-aql --bin chat_hsa' para conversar. ===");
    Ok(())
}
