//! quantize_ternary_hsa.rs — quantiza pesos .blab pra ternário {-1, 0, +1}.
//!
//! BitNet 1.58 style: threshold τ = α * mean(|W|). Cada peso vira -1/0/+1 escalado.
//!
//! Mede KL divergence pre vs post quantização em amostras de validação.
//! Demonstra: pesos f32 → 2 bits por peso (3 estados) preservando função.

use std::fs::File;
use std::io::{Read, Write as _};

use blab::gpu_aql::SmgGridHsa;
use blab::gpu_aql::WeightsHost;
use blab::learn::microcolumn_v3::{GridConfig, SmgGrid};

const VOCAB: usize = 256;
const WINDOW: usize = 96;
const LR: f32 = 0.003;
const SEED: u64 = 7;

const BLAB_MAGIC:   u32 = 0x424C4142;
const BLAB_VERSION: u32 = 1;

fn load_weights(path: &str) -> std::io::Result<WeightsHost> {
    let mut f = File::open(path)?;
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)?;
    assert_eq!(u32::from_le_bytes(buf4), BLAB_MAGIC, "MAGIC");
    f.read_exact(&mut buf4)?;
    assert_eq!(u32::from_le_bytes(buf4), BLAB_VERSION, "VERSION");

    let read_tensor = |f: &mut File| -> std::io::Result<Vec<f32>> {
        let mut bb = [0u8; 4];
        f.read_exact(&mut bb)?;
        let n = u32::from_le_bytes(bb) as usize;
        let mut v = vec![0f32; n];
        for x in v.iter_mut() {
            let mut bb2 = [0u8; 4];
            f.read_exact(&mut bb2)?;
            *x = f32::from_le_bytes(bb2);
        }
        Ok(v)
    };

    Ok(WeightsHost {
        t_f:      read_tensor(&mut f)?,
        i_f:      read_tensor(&mut f)?,
        nibble_h: read_tensor(&mut f)?,
        nibble_l: read_tensor(&mut f)?,
        b_f:      read_tensor(&mut f)?,
        pos_bias: read_tensor(&mut f)?,
        lat_w:    read_tensor(&mut f)?,
        w_out:    read_tensor(&mut f)?,
        b_out:    read_tensor(&mut f)?,
    })
}

fn save_weights(path: &str, wh: &WeightsHost) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(&BLAB_MAGIC.to_le_bytes())?;
    f.write_all(&BLAB_VERSION.to_le_bytes())?;
    let tensors: [&Vec<f32>; 9] = [
        &wh.t_f, &wh.i_f, &wh.nibble_h, &wh.nibble_l,
        &wh.b_f, &wh.pos_bias, &wh.lat_w, &wh.w_out, &wh.b_out,
    ];
    for t in tensors {
        f.write_all(&(t.len() as u32).to_le_bytes())?;
        for &x in t { f.write_all(&x.to_le_bytes())?; }
    }
    Ok(())
}

/// Quantiza tensor para ternário com escala.
/// τ = α * mean(|w|). w' = sign(w) * scale se |w| > τ, senão 0.
/// scale = mean(|w| | |w|>τ) — preserva norma média.
fn quantize_ternary(w: &[f32], alpha: f32) -> (Vec<f32>, f32, usize, usize) {
    let n = w.len();
    let mean_abs: f32 = w.iter().map(|x| x.abs()).sum::<f32>() / n as f32;
    let tau = alpha * mean_abs;

    let mut sum_active = 0.0f32;
    let mut count_active = 0usize;
    let mut count_pos = 0usize;
    let mut count_neg = 0usize;
    for &x in w {
        if x.abs() > tau {
            sum_active += x.abs();
            count_active += 1;
            if x > 0.0 { count_pos += 1; } else { count_neg += 1; }
        }
    }
    let scale = if count_active > 0 { sum_active / count_active as f32 } else { mean_abs };

    let out: Vec<f32> = w.iter().map(|&x| {
        if x > tau { scale }
        else if x < -tau { -scale }
        else { 0.0 }
    }).collect();

    (out, scale, count_active, n)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let in_path  = args.get(1).map(|s| s.as_str()).unwrap_or("dialogue_weights.blab");
    let out_path = args.get(2).map(|s| s.as_str()).unwrap_or("dialogue_weights_ternary.blab");
    let alpha: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.7);

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   quantize_ternary_hsa — pesos f32 → {{-1,0,+1}} × scale     ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("  Input:  {}", in_path);
    println!("  Output: {}", out_path);
    println!("  α (threshold = α × mean|w|): {}", alpha);

    let wh = load_weights(in_path)?;
    println!("\n  Tensors carregados:");
    let names = ["t_f", "i_f", "nibble_h", "nibble_l", "b_f", "pos_bias", "lat_w", "w_out", "b_out"];
    let tensors_orig = [&wh.t_f, &wh.i_f, &wh.nibble_h, &wh.nibble_l, &wh.b_f, &wh.pos_bias, &wh.lat_w, &wh.w_out, &wh.b_out];
    let mut total_f32 = 0usize;
    for (n, t) in names.iter().zip(&tensors_orig) {
        println!("    {:>10}  len={:>7}  ({:.1} KB)", n, t.len(), t.len() as f32 * 4.0 / 1024.0);
        total_f32 += t.len();
    }
    println!("  Total: {} params = {:.1} MB (f32)", total_f32, total_f32 as f32 * 4.0 / 1e6);

    // Quantize
    println!("\n▸ Quantização:");
    println!("  {:>10}  {:>9}  {:>9}  {:>7}   {}", "tensor", "active%", "scale", "mean|w|", "(active/total)");
    let mut q = WeightsHost {
        t_f:      vec![], i_f:      vec![], nibble_h: vec![], nibble_l: vec![],
        b_f:      vec![], pos_bias: vec![], lat_w:    vec![],
        w_out:    vec![], b_out:    vec![],
    };
    let mut total_active = 0usize;
    let mut quantized_slots: Vec<Vec<f32>> = vec![];
    for (n, t) in names.iter().zip(&tensors_orig) {
        let mean_abs: f32 = t.iter().map(|x| x.abs()).sum::<f32>() / t.len() as f32;
        let (qt, scale, active, total) = quantize_ternary(t, alpha);
        let pct = 100.0 * active as f32 / total as f32;
        total_active += active;
        println!("  {:>10}  {:>8.1}%  {:>9.5}  {:>7.5}   ({}/{})", n, pct, scale, mean_abs, active, total);
        quantized_slots.push(qt);
    }
    q.t_f      = quantized_slots[0].clone();
    q.i_f      = quantized_slots[1].clone();
    q.nibble_h = quantized_slots[2].clone();
    q.nibble_l = quantized_slots[3].clone();
    q.b_f      = quantized_slots[4].clone();
    q.pos_bias = quantized_slots[5].clone();
    q.lat_w    = quantized_slots[6].clone();
    q.w_out    = quantized_slots[7].clone();
    q.b_out    = quantized_slots[8].clone();

    // Storage delta
    let bits_orig    = total_f32 * 32;
    let bits_ternary = total_f32 * 2;  // 2 bits/peso teórico
    println!("\n  Storage:");
    println!("    f32:     {} bits = {:.2} MB", bits_orig, bits_orig as f32 / 8e6);
    println!("    ternary: {} bits = {:.2} MB  (compressão = {:.1}×)",
             bits_ternary, bits_ternary as f32 / 8e6, bits_orig as f32 / bits_ternary as f32);
    println!("    sparsity = {:.1}% pesos = 0 (zero entries grátis)",
             100.0 * (total_f32 - total_active) as f32 / total_f32 as f32);

    // ── Compara logits: original vs ternary em batch teste ─────────────
    println!("\n▸ Comparação logits (original vs ternary) em 16 amostras teste:");

    // Sample windows from corpus
    let corpus = std::fs::read("data/dialogues_pt.txt")?;
    let n_samp = 16usize;
    let mut samples: Vec<Vec<u32>> = Vec::with_capacity(n_samp);
    let mut rng = SEED;
    for _ in 0..n_samp {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let start = (rng as usize) % (corpus.len() - WINDOW - 1);
        samples.push(corpus[start..start+WINDOW].iter().map(|&b| b as u32).collect());
    }

    let cfg = GridConfig { vocab_size: VOCAB, window: WINDOW, learning_rate: LR, seed: SEED };
    let cpu = SmgGrid::new(cfg.clone());

    let mut hsa = SmgGridHsa::from_cpu(&cpu, n_samp)?;

    // Original
    hsa.upload_weights(&wh)?;
    let logits_orig = hsa.forward_batch(&samples);

    // Ternary
    hsa.upload_weights(&q)?;
    let logits_tern = hsa.forward_batch(&samples);

    // Compute KL div between softmaxes
    let mut total_kl = 0.0f32;
    let mut top1_match = 0;
    for (lo, lt) in logits_orig.iter().zip(&logits_tern) {
        let mxo = lo.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mxt = lt.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let eo: Vec<f32> = lo.iter().map(|&x| (x - mxo).exp()).collect();
        let et: Vec<f32> = lt.iter().map(|&x| (x - mxt).exp()).collect();
        let so: f32 = eo.iter().sum();
        let st: f32 = et.iter().sum();
        let po: Vec<f32> = eo.iter().map(|&x| x/so).collect();
        let pt: Vec<f32> = et.iter().map(|&x| x/st).collect();
        let mut kl = 0.0f32;
        for j in 0..VOCAB {
            if po[j] > 1e-9 {
                kl += po[j] * (po[j].max(1e-20).ln() - pt[j].max(1e-20).ln());
            }
        }
        total_kl += kl;
        let argmax_o = po.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let argmax_t = pt.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        if argmax_o == argmax_t { top1_match += 1; }
    }
    let avg_kl = total_kl / n_samp as f32;
    let acc_top1 = 100.0 * top1_match as f32 / n_samp as f32;

    println!("    avg KL(orig || ternary): {:.4}  ({})",
             avg_kl,
             if avg_kl < 0.1 { "✓ pesos comprimem bem" } else { "⚠ alta divergência" });
    println!("    top-1 match: {}/{} = {:.0}%",
             top1_match, n_samp, acc_top1);

    // Save
    save_weights(out_path, &q)?;
    println!("\n▸ Salvo: {}", out_path);
    let sz_orig = std::fs::metadata(in_path)?.len();
    let sz_q    = std::fs::metadata(out_path)?.len();
    println!("    (file ainda f32 pra compat com chat_hsa: {} bytes vs orig {} bytes)", sz_q, sz_orig);
    println!("    Mas valores ∈ {{-scale, 0, +scale}} — quantizado de fato.");
    println!("    Pra storage real: 2 bits/peso → {:.2} MB", total_f32 as f32 * 2.0 / 8e6);
    Ok(())
}
