//! Cheap CPU proxy for "does QK-norm accelerate the attention bootstrap?".
//! Trains the same tiny model on a repeating pattern (memorization needs
//! attention) with qk_norm off vs on, printing loss vs step. If QK-norm wakes
//! attention faster (lower loss earlier), it should shorten the unigram-floor
//! plateau on the real corpus — justifying the device-side implementation.

use blab::gpu::transformer::{Config, Model};
use blab::gpu::GpuContext;

fn run(ctx: &GpuContext, qk_norm: bool, alibi_slope: f32, toks: &[u8], tgts: &[u8], steps: usize, lr: f32) -> Vec<(usize, f32)> {
    let cfg = Config { vocab: 256, d_model: 48, n_layers: 2, d_ff: 192, max_seq: toks.len(),
                       n_predict: 1, qk_norm, alibi_slope };
    let mut m = Model::init(cfg, 7);
    let mut tr = Vec::new();
    for s in 0..=steps {
        let l = m.train_step_clip(ctx, toks, tgts, lr, 1.0);
        if s % 50 == 0 { tr.push((s, l)); }
    }
    tr
}

fn main() {
    let ctx = GpuContext::new().expect("gpu");
    // a longer, structured repeating string so beating unigram REQUIRES context.
    let data = b"the quick brown fox. the quick brown fox. the quick brown fox!";
    let toks: Vec<u8> = data[..data.len() - 1].to_vec();
    let tgts: Vec<u8> = data[1..].to_vec();
    let steps = 600usize;
    let lr = 0.2f32;
    println!("GPU: {}  seq={}", ctx.device_name(), toks.len());

    let off   = run(&ctx, false, 0.0, &toks, &tgts, steps, lr);
    let qkn   = run(&ctx, true,  0.0, &toks, &tgts, steps, lr);
    let alibi = run(&ctx, false, 0.5, &toks, &tgts, steps, lr);
    println!("\n step |   baseline | qk_norm  | alibi(0.5)");
    for (((s, lo), (_, ln)), (_, la)) in off.iter().zip(&qkn).zip(&alibi) {
        println!("{s:5} | {lo:9.4} | {ln:8.4} | {la:.4}");
    }
}
