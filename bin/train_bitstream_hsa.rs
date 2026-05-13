//! train_bitstream_hsa.rs — bit-stream LM.
//!
//! IDEIA: corpus = stream contínuo de bits (1 byte → 8 bits). Modelo prediz
//! próximo bit dado os últimos N bits. Vocab = 2.
//!
//! Aproveita 100% kernels SMG existentes (HSA AQL): forward + backward_bptt +
//! grad_acc + sgd. Único parâmetro que muda: `vocab_size = 2` e janela em bits.
//!
//! Geração: amostra bit-a-bit, mash 8 bits em byte, mostra como string.

use std::fs::File;
use std::io::{Read, Write as _};

use blab::gpu_aql::SmgGridHsa;
use blab::learn::microcolumn_v3::{GridConfig, SmgGrid};

// ── Hiperparams ────────────────────────────────────────────────────────────
const WINDOW:        usize = 256;   // bits (= 32 bytes context)
const VOCAB:         usize = 2;     // bit binário
const LR:            f32   = 0.005;
const BATCH:         usize = 32;
const N_EPOCHS:      usize = 30;
const STEPS_PER_EP:  usize = 400;
const SEED:          u64   = 7;
const CORPUS:        &str  = "data/dialogues_pt.txt";
const OUT_WEIGHTS:   &str  = "bitstream_weights.blab";

// ── .blab serializer ──────────────────────────────────────────────────────
const BLAB_MAGIC:   u32 = 0x424C4142;
const BLAB_VERSION: u32 = 1;

fn save_weights(path: &str, hsa: &SmgGridHsa) -> std::io::Result<()> {
    let wh = hsa.weights_to_host();
    let mut f = File::create(path)?;
    f.write_all(&BLAB_MAGIC.to_le_bytes())?;
    f.write_all(&BLAB_VERSION.to_le_bytes())?;
    let tensors = [
        &wh.t_f, &wh.i_f, &wh.nibble_h, &wh.nibble_l,
        &wh.b_f, &wh.pos_bias, &wh.lat_w, &wh.w_out, &wh.b_out,
    ];
    for t in tensors {
        f.write_all(&(t.len() as u32).to_le_bytes())?;
        for &x in t { f.write_all(&x.to_le_bytes())?; }
    }
    Ok(())
}

// ── Bit stream helpers ─────────────────────────────────────────────────────
/// Converte bytes em stream de bits (MSB first per byte).
fn bytes_to_bits(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 8);
    for &b in bytes {
        for i in (0..8).rev() {
            out.push((b >> i) & 1);
        }
    }
    out
}

/// Converte bit stream em bytes (MSB first).
fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.len() / 8);
    for chunk in bits.chunks_exact(8) {
        let mut b = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            if bit == 1 { b |= 1 << (7 - i); }
        }
        out.push(b);
    }
    out
}

// ── RNG ────────────────────────────────────────────────────────────────────
fn xorshift(s: &mut u64) -> u32 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s >> 16) as u32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   train_bitstream_hsa — Bit-stream LM (vocab=2, GPU/HSA)   ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("  WINDOW={} bits  VOCAB={}  LR={}  BATCH={}  EPOCHS={}  STEPS/ep={}",
             WINDOW, VOCAB, LR, BATCH, N_EPOCHS, STEPS_PER_EP);

    // ── Read corpus + convert to bit stream ───────────────────────────
    let bytes = std::fs::read(CORPUS)?;
    println!("  Corpus: {} bytes", bytes.len());
    let bits = bytes_to_bits(&bytes);
    println!("  Bit stream: {} bits ({:.1} MB)", bits.len(), bits.len() as f32 / 8e6);

    // ── Init GPU model ────────────────────────────────────────────────
    let cfg = GridConfig {
        vocab_size:    VOCAB,
        window:        WINDOW,
        learning_rate: LR,
        seed:          SEED,
    };
    let cpu = SmgGrid::new(cfg.clone());
    let mut hsa = SmgGridHsa::from_cpu(&cpu, BATCH)?;
    println!("  Modelo: {} params (vocab tiny, mas FLAT_DIM=4096 do core)\n", hsa.n_params());

    println!("{:>5}  {:>10}  {:>5}  tempo", "epoch", "loss", "bpb");
    println!("{}", "─".repeat(40));

    let mut rng = SEED.max(1);
    let n_bits = bits.len();
    let total_t0 = std::time::Instant::now();

    for epoch in 0..N_EPOCHS {
        let ep0 = std::time::Instant::now();

        // LR schedule warmup + cosine
        let warm = (N_EPOCHS / 10).max(1);
        let lr = if epoch < warm {
            LR * (epoch + 1) as f32 / warm as f32
        } else {
            let p = (epoch - warm) as f32 / (N_EPOCHS - warm) as f32;
            LR * 0.5 * (1.0 + (std::f32::consts::PI * p).cos()).max(0.05)
        };
        hsa.set_lr(lr);

        let mut loss_sum = 0.0;
        let mut n_batches = 0;
        for _ in 0..STEPS_PER_EP {
            // Sample BATCH random windows from bit stream
            let mut batch = Vec::with_capacity(BATCH);
            for _ in 0..BATCH {
                let start = (xorshift(&mut rng) as usize) % (n_bits - WINDOW - 1);
                let win: Vec<u32> = bits[start..start + WINDOW]
                    .iter().map(|&b| b as u32).collect();
                let tgt = bits[start + WINDOW] as u32;
                batch.push((win, tgt));
            }
            loss_sum += hsa.train_step(&batch);
            n_batches += 1;
        }

        let loss = loss_sum / n_batches as f32;
        // bits per bit (entropy per bit) — meta de modelo bom é < 0.5 bpb
        let bpb = loss / std::f32::consts::LN_2;
        println!("  {:>3}    {:>9.4}  {:>5.3}  {:.1}s",
                 epoch, loss, bpb, ep0.elapsed().as_secs_f32());
        std::io::stdout().flush().ok();
    }

    let elapsed = total_t0.elapsed().as_secs_f32();
    println!("\n━━━ Wall: {:.0}s ━━━\n", elapsed);

    // ── Save weights ─────────────────────────────────────────────────
    println!("▸ Salvando {}", OUT_WEIGHTS);
    save_weights(OUT_WEIGHTS, &hsa)?;
    let sz = std::fs::metadata(OUT_WEIGHTS)?.len();
    println!("  {} bytes ({:.2} MB)", sz, sz as f32 / 1e6);

    // ── Geração demo ──────────────────────────────────────────────────
    println!("\n▸ Geração demo (greedy + temp):");
    let prompts: &[&str] = &["USER: oi\nMODEL: ", "USER: bom dia\nMODEL: "];
    for p in prompts {
        let prompt_bits = bytes_to_bits(p.as_bytes());
        let n_gen_bits = 30 * 8; // 30 bytes
        let mut gen_bits: Vec<u8> = Vec::new();

        for mode in &[(0.001_f32, "greedy"), (0.7_f32, "t=0.7")] {
            let (temp, label) = *mode;
            gen_bits.clear();
            let mut rng_g = SEED ^ 0xdeadbeef;
            for _ in 0..n_gen_bits {
                // Build window: last WINDOW bits from (prompt + generated)
                let combined: Vec<u8> = prompt_bits.iter()
                    .chain(gen_bits.iter())
                    .copied().collect();
                let start = if combined.len() >= WINDOW {
                    combined.len() - WINDOW
                } else { 0 };
                let mut win: Vec<u32> = combined[start..].iter().map(|&b| b as u32).collect();
                // Pad if needed
                while win.len() < WINDOW {
                    win.insert(0, 0);
                }

                // Forward to get logits [2]
                let logits = &hsa.forward_batch(&[win])[0];
                let p0 = logits[0];
                let p1 = logits[1];
                // softmax over 2 logits
                let mx = p0.max(p1);
                let e0 = (p0 - mx).exp();
                let e1 = (p1 - mx).exp();
                let pr1 = e1 / (e0 + e1);
                let pr1_t = if temp < 0.01 {
                    if pr1 > 0.5 { 1.0 } else { 0.0 }
                } else {
                    // temperature on logit difference
                    let diff = (p1 - p0) / temp;
                    1.0 / (1.0 + (-diff).exp())
                };
                let r = (xorshift(&mut rng_g) as f32) / (u32::MAX as f32);
                let bit: u8 = if r < pr1_t { 1 } else { 0 };
                gen_bits.push(bit);
            }
            let out_bytes = bits_to_bytes(&gen_bits);
            let s = String::from_utf8_lossy(&out_bytes);
            let hex: String = out_bytes.iter().take(20)
                .map(|b| format!("{:02X}", b)).collect::<Vec<_>>().join(" ");
            println!("  [{}] {:?}", label, p);
            println!("      → {:?}", s);
            println!("      bytes: [{}...]", hex);
        }
    }

    Ok(())
}
