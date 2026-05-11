//! chat_hsa — CLI de chat interactivo usando SMG v3 em RDNA2.
//!
//! Carrega pesos de `dialogue_weights.blab` (gerado por `train_dialogue_hsa`).
//! Se o ficheiro não existir, treina um novo modelo inline antes de iniciar o chat.
//!
//! Formato de conversa:
//! ```
//! USER: OI
//! MODEL: Olá! Como posso te ajudar hoje?
//! USER: O que você é?
//! MODEL: Sou um assistente de linguagem...
//! ```
//!
//! O modelo mantém as últimas 48 bytes de contexto (WINDOW bytes).
//! O histórico é truncado automaticamente ao tamanho da janela.
//!
//! Comandos especiais:
//!   /reset   — limpa o histórico de conversa
//!   /quit    — sai do chat
//!   /temp N  — muda temperatura (ex: /temp 0.5)
//!   /greedy  — atalho para temperatura=0.0 (determinístico)
//!
//! ```bash
//! cargo run --release --features gpu-aql --bin chat_hsa
//! cargo run --release --features gpu-aql --bin chat_hsa -- dialogue_weights.blab
//! ```

#![cfg(feature = "gpu-aql")]
#![allow(unsafe_code)]

use std::io::{Read as _, Write as _, BufRead as _};
use std::time::Instant;

use blab::{
    gpu_aql::{SmgGridHsa, WeightsHost},
    learn::microcolumn_v3::{GridConfig, SmgGrid},
};

// ---- Configuração ----
const WINDOW:    usize = 48;
const VOCAB:     usize = 256;
const SEED:      u64   = 7;
const MAX_RESP:  usize = 200;   // comprimento máximo da resposta em bytes

// ---- Hiperparâmetros de treino inline (fallback) ----
const LR_INLINE:              f32   = 0.003;
const BATCH_INLINE:           usize = 64;
const N_EPOCHS_INLINE:        usize = 60;
const STEPS_PER_EPOCH_INLINE: usize = 200;

// ---- Corpus inline (idêntico ao de train_dialogue_hsa) ----
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
// Formato .blab
// ---------------------------------------------------------------------------

const BLAB_MAGIC:   u32 = 0x424C4142;
const BLAB_VERSION: u32 = 1;

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
// Treino inline (fallback quando não há pesos)
// ---------------------------------------------------------------------------

fn train_inline(cfg: &GridConfig) -> Result<SmgGridHsa, Box<dyn std::error::Error>> {
    eprintln!("[chat_hsa] Pesos não encontrados. Treinando modelo inline...");
    eprintln!("[chat_hsa] (use 'cargo run --bin train_dialogue_hsa' para pré-treinar)");
    eprintln!();

    let data = EMBEDDED_CORPUS.repeat(40).into_bytes();
    let mut all_pairs: Vec<(Vec<u32>, u32)> = if data.len() > WINDOW {
        data.windows(WINDOW + 1)
            .map(|w| {
                let win: Vec<u32> = w[..WINDOW].iter().map(|&b| b as u32).collect();
                (win, w[WINDOW] as u32)
            })
            .collect()
    } else { Vec::new() };

    let n_val = (all_pairs.len() / 8).max(BATCH_INLINE);
    let val_pairs: Vec<_> = all_pairs.split_off(all_pairs.len() - n_val);
    let mut train_pairs = all_pairs;

    let cpu = SmgGrid::new(cfg.clone());
    let mut hsa = SmgGridHsa::from_cpu(&cpu, BATCH_INLINE)?;

    let mut rng_seed = SEED;
    for ep in 0..N_EPOCHS_INLINE {
        let warmup_end   = (N_EPOCHS_INLINE / 10).max(1);
        let decay_start  = (N_EPOCHS_INLINE * 3 / 4).max(warmup_end + 1);
        let lr = if ep < warmup_end {
            LR_INLINE * (ep + 1) as f32 / warmup_end as f32
        } else if ep < decay_start {
            LR_INLINE
        } else {
            let t = (ep - decay_start) as f32 / (N_EPOCHS_INLINE - decay_start).max(1) as f32;
            (LR_INLINE * 0.5 * (1.0 + (std::f32::consts::PI * t).cos())).max(LR_INLINE * 0.05)
        };
        hsa.set_lr(lr);

        rng_seed = rng_seed.wrapping_add(ep as u64 * 0x9e3779b97f4a7c15);
        {
            let mut rng = rng_seed;
            for i in (1..train_pairs.len()).rev() {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = (rng >> 33) as usize % (i + 1);
                train_pairs.swap(i, j);
            }
        }

        let epoch_pairs: Vec<_> = train_pairs.iter()
            .take(STEPS_PER_EPOCH_INLINE * BATCH_INLINE).cloned().collect();

        let _tl = hsa.train_epoch(&epoch_pairs, BATCH_INLINE);

        // Progresso mínimo (uma linha por 10 épocas)
        if (ep + 1) % 10 == 0 {
            // Calcula val loss de forma amostral (sem função auxiliar)
            let n_val_sample = val_pairs.len().min(BATCH_INLINE);
            let windows: Vec<Vec<u32>> = val_pairs[..n_val_sample]
                .iter().map(|(w,_)| w.clone()).collect();
            let targets: Vec<u32> = val_pairs[..n_val_sample]
                .iter().map(|(_,t)| *t).collect();
            let logits_batch = hsa.forward_batch(&windows);
            let mut vl = 0.0f32;
            for (logits, &tgt) in logits_batch.iter().zip(&targets) {
                let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = logits.iter().map(|&x| (x-mx).exp()).collect();
                let s: f32 = exps.iter().sum();
                vl -= (exps[tgt as usize]/s).max(1e-20).ln();
            }
            vl /= n_val_sample as f32;
            eprintln!("[chat_hsa] treino {}/{}: val={:.3} nats ({:.3} bits)",
                ep + 1, N_EPOCHS_INLINE, vl, vl / std::f32::consts::LN_2);
        }
    }

    eprintln!("[chat_hsa] Treino concluído.\n");
    Ok(hsa)
}

// ---------------------------------------------------------------------------
// Geração autoregressiva (char a char até \n ou MAX_RESP)
// ---------------------------------------------------------------------------

fn softmax_sample(logits: &[f32], temperature: f32, rng: &mut u64) -> usize {
    if temperature == 0.0 {
        return logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i,_)| i).unwrap_or(0);
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

/// Gera a resposta do modelo dado o contexto atual (últimos WINDOW bytes).
fn respond(
    hsa: &SmgGridHsa,
    context: &[u8],        // histórico + "USER: X\nMODEL: "
    temperature: f32,
    rng: &mut u64,
) -> String {
    // Janela de contexto (últimos WINDOW bytes)
    let window_bytes = if context.len() >= WINDOW {
        &context[context.len() - WINDOW..]
    } else {
        context
    };

    // Pré-preenche com newlines se contexto < WINDOW
    let mut window: Vec<u32> = {
        let pad = WINDOW.saturating_sub(window_bytes.len());
        std::iter::repeat(b'\n' as u32).take(pad)
            .chain(window_bytes.iter().map(|&b| b as u32))
            .collect()
    };

    let mut response = String::new();
    for _ in 0..MAX_RESP {
        let logits = &hsa.forward_batch(&[window.clone()])[0];
        let next = softmax_sample(logits, temperature, rng) as u8;

        // Para no próximo \n (fim da resposta do MODEL)
        if next == b'\n' { break; }

        if next.is_ascii_graphic() || next == b' ' {
            response.push(next as char);
        } else {
            response.push('?');
        }

        window.remove(0);
        window.push(next as u32);
    }
    response
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Banner
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║          IARA Chat — SMG v3 (RDNA2 / ROCm HSA)          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("  Comandos: /reset  /quit  /temp N  /greedy");
    println!("  (modelo byte-level com janela de {} bytes)\n", WINDOW);

    // ---- Argumentos ----
    let args: Vec<String> = std::env::args().collect();
    let weights_path = args.get(1).map(|s| s.as_str()).unwrap_or("dialogue_weights.blab");

    // ---- Config ----
    let cfg = GridConfig {
        vocab_size:    VOCAB,
        window:        WINDOW,
        learning_rate: LR_INLINE,
        seed:          SEED,
    };

    // ---- Carrega ou treina ----
    let mut hsa = match load_weights(weights_path) {
        Ok(wh) => {
            println!("  Pesos carregados de '{}'\n", weights_path);
            let cpu = SmgGrid::new(cfg.clone());
            let mut h = SmgGridHsa::from_cpu(&cpu, 1)?;  // batch=1 para inferência
            h.upload_weights(&wh)?;
            h
        }
        Err(e) => {
            eprintln!("  [aviso] Não foi possível carregar '{}': {}", weights_path, e);
            let mut h = train_inline(&cfg)?;
            // Recria com batch=1 para inferência eficiente
            let wh = h.weights_to_host();
            let cpu = SmgGrid::new(cfg.clone());
            let mut h2 = SmgGridHsa::from_cpu(&cpu, 1)?;
            h2.upload_weights(&wh)?;
            h2
        }
    };

    let mut temperature = 0.7f32;
    let mut rng = SEED ^ 0xcafef00d;

    // ---- Histórico: mantém o contexto completo (truncado a WINDOW ao inferir) ----
    let mut history: Vec<u8> = Vec::new();

    println!("  Pronto! Digite sua mensagem ou um comando.\n");

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();

    loop {
        // Prompt do usuário
        {
            let mut out = stdout.lock();
            out.write_all(b"\x1b[1;32mUSER\x1b[0m: ")?;
            out.flush()?;
        }

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            println!("\nAté logo!");
            break;
        }
        let input = line.trim();

        if input.is_empty() { continue; }

        // ---- Comandos especiais ----
        if input == "/quit" || input == "/exit" {
            println!("Até logo!");
            break;
        }
        if input == "/reset" {
            history.clear();
            println!("  [histórico limpo]\n");
            continue;
        }
        if input == "/greedy" {
            temperature = 0.0;
            println!("  [temperatura → 0.0 (greedy)]\n");
            continue;
        }
        if let Some(rest) = input.strip_prefix("/temp ") {
            if let Ok(t) = rest.trim().parse::<f32>() {
                temperature = t.clamp(0.0, 2.0);
                println!("  [temperatura → {:.2}]\n", temperature);
            } else {
                println!("  [uso: /temp 0.7]\n");
            }
            continue;
        }

        // ---- Formata prompt USER + MODEL: ----
        // Append to history: "USER: {input}\nMODEL: "
        let user_turn = format!("USER: {}\nMODEL: ", input);
        history.extend_from_slice(user_turn.as_bytes());

        // ---- Geração ----
        let t_gen = Instant::now();
        let resp = respond(&hsa, &history, temperature, &mut rng);
        let elapsed = t_gen.elapsed();

        // Append response to history + newline
        history.extend_from_slice(resp.as_bytes());
        history.push(b'\n');

        // Trunca histórico para não crescer indefinidamente (mantém últimos 4×WINDOW)
        let max_history = WINDOW * 4;
        if history.len() > max_history {
            let start = history.len() - max_history;
            // Ajusta para fronteira de linha
            let adj = history[start..].iter().position(|&b| b == b'\n')
                .map(|i| start + i + 1).unwrap_or(start);
            history = history[adj..].to_vec();
        }

        // Imprime resposta
        let bits_per_token = if resp.is_empty() {
            0.0
        } else {
            (elapsed.as_secs_f32() * 1000.0) / resp.len() as f32
        };

        print!("\x1b[1;34mMODEL\x1b[0m: {}", resp.trim());
        println!(
            "  \x1b[2m[{} chars, {:.0}ms, {:.1}ms/char, temp={:.2}]\x1b[0m",
            resp.len(),
            elapsed.as_secs_f32() * 1000.0,
            bits_per_token,
            temperature
        );
        println!();
    }

    Ok(())
}
