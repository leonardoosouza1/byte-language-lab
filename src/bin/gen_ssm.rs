//! Interactive generation from a trained liquid/SSM checkpoint.
//! Usage: gen_ssm [checkpoint] [prompt] [n_bytes]
//!   gen_ssm data/manjou_ssm.ssmc "fn main" 160

use blab::gpu::ssm::DevSsm;
use blab::gpu::GpuContext;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ckpt = a.get(1).cloned().unwrap_or_else(|| "data/manjou_ssm.ssmc".into());
    let prompt = a.get(2).cloned().unwrap_or_else(|| "fn ".into());
    let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(140);

    let ctx = match GpuContext::new() { Ok(c) => c, Err(e) => { eprintln!("no GPU: {e}"); return; } };
    let dev = DevSsm::load(&ctx, &ckpt).unwrap_or_else(|e| panic!("load {ckpt}: {e}"));
    println!("model: d={} ff={} | checkpoint {ckpt}\nprompt: {prompt:?}\n", dev.m.d, dev.ff);

    let p = prompt.as_bytes();
    let greedy = dev.generate(&ctx, p, n);
    println!("── greedy ──\n{prompt}{}", String::from_utf8_lossy(&greedy));
    for &t in &[0.5f32, 0.8] {
        let g = dev.generate_temp(&ctx, p, n, t, 0xABCD ^ t.to_bits() as u64);
        println!("\n── temp {t} ──\n{prompt}{}", String::from_utf8_lossy(&g));
    }
}
