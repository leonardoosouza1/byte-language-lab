//! Diagnostic: is the unigram-floor plateau caused by vanishing attention
//! gradients? Measures per-parameter-group L2 grad norms at init (and after a
//! few SGD steps) for the d=256 model. Hypothesis: ‖dWq‖,‖dWk‖ ≪ ‖dWv‖,‖dWo‖
//! because dWq ∝ K ∝ Wk and the QK projections bootstrap each other from ~0.
//!
//! Usage: diag_attn_grad [corpus] [d_model] [n_layers] [seq] [warm_steps]

use blab::gpu::transformer::{Config, Model};
use blab::gpu::GpuContext;

fn l2(v: &[f32]) -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() }

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).cloned().unwrap_or_else(|| "data/corpus_local.txt".into());
    let d: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let nl: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);
    let seq: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(128);
    let warm: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);

    let bytes = std::fs::read(&path).unwrap();
    let start = bytes.len() / 2;
    let toks = &bytes[start..start + seq];
    let tgts = &bytes[start + 1..start + seq + 1];

    let qk_scale: f32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let qk_norm: bool = a.get(7).map(|s| s == "1" || s == "true").unwrap_or(false);

    let ctx = GpuContext::new().expect("gpu");
    let cfg = Config { vocab: 256, d_model: d, n_layers: nl, d_ff: d * 4, max_seq: seq, n_predict: 1, qk_norm, alibi_slope: 0.0 };
    let mut model = Model::init(cfg, 1234);
    // optional: shrink the QK projection init to de-saturate attention scores.
    if (qk_scale - 1.0).abs() > 1e-6 {
        for ly in model.layers.iter_mut() {
            for w in ly.wq.iter_mut() { *w *= qk_scale; }
            for w in ly.wk.iter_mut() { *w *= qk_scale; }
        }
    }
    println!("GPU: {}  d={d} layers={nl} seq={seq} qk_scale={qk_scale}", ctx.device_name());

    // optional warmup with plain SGD so we can also see grads mid-plateau
    for _ in 0..warm {
        let (_l, g) = model.backward(&ctx, toks, tgts);
        let lr = 0.05;
        for (li, ly) in model.layers.iter_mut().enumerate() {
            for (w, dw) in ly.wq.iter_mut().zip(&g.dwq[li]) { *w -= lr * dw; }
            for (w, dw) in ly.wk.iter_mut().zip(&g.dwk[li]) { *w -= lr * dw; }
            for (w, dw) in ly.wv.iter_mut().zip(&g.dwv[li]) { *w -= lr * dw; }
            for (w, dw) in ly.wo.iter_mut().zip(&g.dwo[li]) { *w -= lr * dw; }
            for (w, dw) in ly.w1.iter_mut().zip(&g.dw1[li]) { *w -= lr * dw; }
            for (w, dw) in ly.w2.iter_mut().zip(&g.dw2[li]) { *w -= lr * dw; }
        }
    }

    let (loss, g) = model.backward(&ctx, toks, tgts);
    println!("loss = {loss:.4} nats ({:.3} bits/byte)  [after {warm} warm steps]",
             loss / std::f32::consts::LN_2);
    println!("\n  layer |    dWq    dWk  |    dWv    dWo  |    dW1    dW2  | ratio Wv/Wq");
    for l in 0..nl {
        let (qn, kn) = (l2(&g.dwq[l]), l2(&g.dwk[l]));
        let (vn, on) = (l2(&g.dwv[l]), l2(&g.dwo[l]));
        let (n1, n2) = (l2(&g.dw1[l]), l2(&g.dw2[l]));
        println!("  L{l:<4} | {qn:9.3e} {kn:9.3e} | {vn:9.3e} {on:9.3e} | {n1:9.3e} {n2:9.3e}");
    }
    println!("\n  ‖d_embed‖ = {:.4}", l2(&g.d_embed));

    // also report attention-score magnitude at init: small scores ⇒ flat softmax
    // ⇒ centered grad ⇒ vanish. compute Q,K for layer 0 on the host.
    // xn = rmsnorm(embed[tok]+pos) with layer-0 norm1 weights.
    let mut xn = vec![0.0f32; seq * d];
    for (i, &t) in toks.iter().enumerate() {
        let mut ss = 0.0f32;
        for j in 0..d {
            let v = model.embed[(t as usize) * d + j] + model.pos[i * d + j];
            xn[i * d + j] = v; ss += v * v;
        }
        let rms = (ss / d as f32 + model.eps).sqrt();
        for j in 0..d { xn[i * d + j] = xn[i * d + j] / rms * model.layers[0].norm1[j]; }
    }
    let (qstd, kstd, sstd, smax) = score_stats(&model.layers[0].wq, &model.layers[0].wk, &xn, seq, d);
    println!("  layer0 init: Q.std={qstd:.4} K.std={kstd:.4} | scores.std={sstd:.4} max|score|={smax:.4}");
    println!("  (uniform softmax when |score| ≪ 1 ⇒ attn grad centered+tiny)");
}

fn score_stats(wq: &[f32], wk: &[f32], xn: &[f32], seq: usize, d: usize) -> (f32, f32, f32, f32) {
    let mm = |x: &[f32], w: &[f32]| -> Vec<f32> {
        let mut o = vec![0.0f32; seq * d];
        for i in 0..seq { for j in 0..d {
            let mut s = 0.0; for k in 0..d { s += x[i * d + k] * w[k * d + j]; }
            o[i * d + j] = s;
        }}
        o
    };
    let q = mm(xn, wq); let k = mm(xn, wk);
    let std = |v: &[f32]| { let m = v.iter().sum::<f32>() / v.len() as f32;
        (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32).sqrt() };
    let scale = 1.0 / (d as f32).sqrt();
    let mut sc = Vec::new(); let mut smax = 0.0f32;
    for i in 0..seq { for j in 0..=i {
        let mut s = 0.0; for c in 0..d { s += q[i * d + c] * k[j * d + c]; }
        s *= scale; smax = smax.max(s.abs()); sc.push(s);
    }}
    (std(&q), std(&k), std(&sc), smax)
}
