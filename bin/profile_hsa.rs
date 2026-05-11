//! Perfil por fase do train_step HSA.
//!
//! Mede separadamente: forward, backward, SGD, hip_d2h.
//! Objetivo: identificar qual fase é o gargalo (CAS contention vs banco LDS).
//!
//! ```bash
//! ROCM_PATH=/opt/rocm-7.2.1 cargo run --release --features gpu-aql --bin profile_hsa
//! ```

#![cfg(feature = "gpu-aql")]
#![allow(unsafe_code)]

use std::time::{Duration, Instant};
use blab::{
    gpu_aql::SmgGridHsa,
    learn::microcolumn_v3::{GridConfig, SmgGrid, FLAT_DIM},
};
use hip_ffi::safe::{DeviceBuffer, hip_h2d, hip_d2h};
use blab::gpu_aql::dispatch::{SmgRuntime, ParamsFwd, ParamsBwd};

const BATCH: usize = 64;
const T:     usize = 48;
const VOCAB: usize = 256;
const LR:    f32   = 0.1;
const REPS:  usize = 5;

fn make_batch(batch: usize, t: usize, vocab: usize) -> Vec<(Vec<u32>, u32)> {
    (0..batch).map(|b| {
        let win: Vec<u32> = (0..t).map(|step| ((b * 7 + step * 3 + 1) % vocab) as u32).collect();
        let tgt = ((b + 1) * 11 % vocab) as u32;
        (win, tgt)
    }).collect()
}

fn mean_ms(times: &[Duration]) -> f64 {
    let sum: f64 = times.iter().map(|d| d.as_secs_f64() * 1000.0).sum();
    sum / times.len() as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== profile_hsa: per-phase timing ===");
    println!("batch={BATCH}, T={T}, vocab={VOCAB}, reps={REPS}\n");

    let cfg = GridConfig { vocab_size: VOCAB, window: T, learning_rate: LR, seed: 42 };
    let cpu = SmgGrid::new(cfg.clone());
    let batch_data = make_batch(BATCH, T, VOCAB);

    // ---- Warm-up completo ----
    println!("[warmup] 1 train_step...");
    let mut hsa = SmgGridHsa::from_cpu(&cpu, BATCH)?;
    hsa.train_step(&batch_data);
    println!("  warmup ok\n");

    // ---- Medir train_step total ----
    println!("[TOTAL train_step] {} reps:", REPS);
    let mut total_times = Vec::new();
    for _ in 0..REPS {
        let t0 = Instant::now();
        hsa.train_step(&batch_data);
        total_times.push(t0.elapsed());
    }
    println!("  mean = {:.3} ms\n", mean_ms(&total_times));

    // ---- Micro-benchmark: hip_h2d small ----
    println!("[hip_h2d 12KB (win)] {} reps:", REPS * 4);
    let win_flat: Vec<u32> = batch_data.iter().flat_map(|(w, _)| w.iter().copied()).collect();
    let targets: Vec<u32>  = batch_data.iter().map(|(_, t)| *t).collect();
    let mut buf_win = DeviceBuffer::<u32>::alloc(BATCH * T)?;
    let mut buf_tgt = DeviceBuffer::<u32>::alloc(BATCH)?;
    let mut h2d_times = Vec::new();
    for _ in 0..(REPS * 4) {
        let t0 = Instant::now();
        unsafe {
            hip_h2d(buf_win.as_mut_ptr(), &win_flat).unwrap();
            hip_h2d(buf_tgt.as_mut_ptr(), &targets).unwrap();
        }
        h2d_times.push(t0.elapsed());
    }
    println!("  mean 2×h2d = {:.3} ms\n", mean_ms(&h2d_times));

    // ---- Micro-benchmark: hip_d2h 64KB (logits) ----
    println!("[hip_d2h 64KB (logits)] {} reps:", REPS * 4);
    let buf_log = DeviceBuffer::<f32>::alloc(BATCH * VOCAB)?;
    let mut logits_host = vec![0.0f32; BATCH * VOCAB];
    let mut d2h_times = Vec::new();
    for _ in 0..(REPS * 4) {
        let t0 = Instant::now();
        unsafe { hip_d2h(&mut logits_host, buf_log.as_ptr()).unwrap(); }
        d2h_times.push(t0.elapsed());
    }
    println!("  mean d2h = {:.3} ms\n", mean_ms(&d2h_times));

    // ---- Micro-benchmark: drain() sozinho ----
    println!("[drain() sozinho (barrier+wait)] {} reps:", REPS * 4);
    // SmgRuntime não tem `drain` isolado — usar SmgGridHsa como proxy.
    // Medir diferença: step_total - (forward) - (backward) - (sgd).
    // Em vez disso, medimos via SmgRuntime público.
    {
        let rt = SmgRuntime::new()?;
        let mut drain_times = Vec::new();
        for _ in 0..(REPS * 4) {
            let t0 = Instant::now();
            rt.drain().unwrap();
            drain_times.push(t0.elapsed());
        }
        println!("  mean drain = {:.3} ms\n", mean_ms(&drain_times));
    }

    // ---- Micro-benchmark: forward kernel + drain ----
    println!("[forward_main dispatch + drain] {} reps:", REPS);
    {
        let rt = SmgRuntime::new()?;
        // Allocate persistent bufs
        let buf_t_f   = DeviceBuffer::<f32>::from_slice(cpu.weights_t_f())?;
        let buf_i_f   = DeviceBuffer::<f32>::from_slice(cpu.weights_i_f())?;
        let buf_b_f   = DeviceBuffer::<f32>::from_slice(cpu.weights_b_f())?;
        let buf_pos   = DeviceBuffer::<f32>::from_slice(cpu.weights_pos_bias())?;
        let buf_lat   = DeviceBuffer::<f32>::from_slice(cpu.weights_lat_w())?;
        let buf_w_out = DeviceBuffer::<f32>::from_slice(cpu.weights_w_out())?;
        let buf_b_out = DeviceBuffer::<f32>::from_slice(cpu.weights_b_out())?;
        let mut buf_log2 = DeviceBuffer::<f32>::alloc(BATCH * VOCAB)?;
        let mut buf_st   = DeviceBuffer::<f32>::alloc(BATCH * (T + 1) * FLAT_DIM)?;

        unsafe { hip_h2d(buf_win.as_mut_ptr(), &win_flat).unwrap(); }

        let params = ParamsFwd {
            vocab_size: VOCAB as u32,
            t_steps:    T as u32,
            batch_size: BATCH as u32,
            cache_states: 1,
        };

        // Warmup
        unsafe {
            rt.dispatch_forward(
                params,
                buf_t_f.as_ptr() as u64, buf_i_f.as_ptr() as u64,
                buf_b_f.as_ptr() as u64, buf_pos.as_ptr() as u64,
                buf_lat.as_ptr() as u64, buf_w_out.as_ptr() as u64,
                buf_b_out.as_ptr() as u64, buf_win.as_ptr() as u64,
                buf_log2.as_mut_ptr() as u64, buf_st.as_mut_ptr() as u64,
            ).unwrap();
            rt.drain().unwrap();
        }

        let mut fwd_times = Vec::new();
        for _ in 0..REPS {
            let t0 = Instant::now();
            unsafe {
                rt.dispatch_forward(
                    params,
                    buf_t_f.as_ptr() as u64, buf_i_f.as_ptr() as u64,
                    buf_b_f.as_ptr() as u64, buf_pos.as_ptr() as u64,
                    buf_lat.as_ptr() as u64, buf_w_out.as_ptr() as u64,
                    buf_b_out.as_ptr() as u64, buf_win.as_ptr() as u64,
                    buf_log2.as_mut_ptr() as u64, buf_st.as_mut_ptr() as u64,
                ).unwrap();
                rt.drain().unwrap();
            }
            fwd_times.push(t0.elapsed());
        }
        println!("  mean forward+drain = {:.3} ms\n", mean_ms(&fwd_times));

        // ---- backward kernel + drain ----
        println!("[backward_main dispatch + drain] {} reps:", REPS);
        let n_t_f   = (8 * 32 * 32_usize) as usize;
        let n_i_f   = (8 * VOCAB * 32_usize) as usize;
        let n_b_f   = (8 * 32_usize) as usize;
        let n_pos   = (16 * 32_usize) as usize;
        let n_lat_w = (8 * 32 * 32_usize) as usize;
        let n_w_out = (FLAT_DIM * VOCAB) as usize;
        let n_b_out = VOCAB;

        let mut g_t_f   = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_t_f])?;
        let mut g_i_f   = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_i_f])?;
        let mut g_b_f   = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_b_f])?;
        let mut g_pos   = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_pos])?;
        let mut g_lat_w = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_lat_w])?;
        let mut g_w_out = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_w_out])?;
        let mut g_b_out = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_b_out])?;

        let params_bwd = ParamsBwd {
            vocab_size: VOCAB as u32,
            t_steps:    T as u32,
            batch_size: BATCH as u32,
            _pad: 0,
        };

        // Warmup
        unsafe {
            rt.dispatch_backward(
                params_bwd,
                buf_t_f.as_ptr() as u64, buf_lat.as_ptr() as u64,
                buf_w_out.as_ptr() as u64, buf_st.as_ptr() as u64,
                buf_log2.as_ptr() as u64, buf_win.as_ptr() as u64,
                buf_tgt.as_ptr() as u64,
                g_t_f.as_mut_ptr() as u64, g_i_f.as_mut_ptr() as u64,
                g_b_f.as_mut_ptr() as u64, g_pos.as_mut_ptr() as u64,
                g_lat_w.as_mut_ptr() as u64, g_w_out.as_mut_ptr() as u64,
                g_b_out.as_mut_ptr() as u64,
            ).unwrap();
            rt.drain().unwrap();
        }

        let mut bwd_times = Vec::new();
        for _ in 0..REPS {
            let t0 = Instant::now();
            unsafe {
                rt.dispatch_backward(
                    params_bwd,
                    buf_t_f.as_ptr() as u64, buf_lat.as_ptr() as u64,
                    buf_w_out.as_ptr() as u64, buf_st.as_ptr() as u64,
                    buf_log2.as_ptr() as u64, buf_win.as_ptr() as u64,
                    buf_tgt.as_ptr() as u64,
                    g_t_f.as_mut_ptr() as u64, g_i_f.as_mut_ptr() as u64,
                    g_b_f.as_mut_ptr() as u64, g_pos.as_mut_ptr() as u64,
                    g_lat_w.as_mut_ptr() as u64, g_w_out.as_mut_ptr() as u64,
                    g_b_out.as_mut_ptr() as u64,
                ).unwrap();
                rt.drain().unwrap();
            }
            bwd_times.push(t0.elapsed());
        }
        println!("  mean backward+drain = {:.3} ms\n", mean_ms(&bwd_times));

        // ---- backward com batch=1 (zero inter-WG contention) ----
        println!("[backward_main batch=1 (zero contention)] {} reps:", REPS);
        let params_bwd1 = ParamsBwd {
            vocab_size: VOCAB as u32,
            t_steps:    T as u32,
            batch_size: 1,
            _pad: 0,
        };
        let buf_win1 = DeviceBuffer::<u32>::from_slice(&win_flat[..T])?;
        let buf_tgt1 = DeviceBuffer::<u32>::from_slice(&targets[..1])?;
        let mut buf_st1  = DeviceBuffer::<f32>::alloc((T + 1) * FLAT_DIM)?;
        // need to populate buf_st1 with forward
        {
            let params1 = ParamsFwd { vocab_size: VOCAB as u32, t_steps: T as u32, batch_size: 1, cache_states: 1 };
            unsafe {
                rt.dispatch_forward(params1,
                    buf_t_f.as_ptr() as u64, buf_i_f.as_ptr() as u64,
                    buf_b_f.as_ptr() as u64, buf_pos.as_ptr() as u64,
                    buf_lat.as_ptr() as u64, buf_w_out.as_ptr() as u64,
                    buf_b_out.as_ptr() as u64, buf_win1.as_ptr() as u64,
                    buf_log2.as_mut_ptr() as u64, buf_st1.as_mut_ptr() as u64,
                ).unwrap();
                rt.drain().unwrap();
            }
        }
        let mut g_t_f1   = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_t_f])?;
        let mut g_i_f1   = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_i_f])?;
        let mut g_b_f1   = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_b_f])?;
        let mut g_pos1   = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_pos])?;
        let mut g_lat1   = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_lat_w])?;
        let mut g_w_out1 = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_w_out])?;
        let mut g_b_out1 = DeviceBuffer::<u32>::from_slice(&vec![0u32; n_b_out])?;
        // warmup
        unsafe {
            rt.dispatch_backward(params_bwd1,
                buf_t_f.as_ptr() as u64, buf_lat.as_ptr() as u64,
                buf_w_out.as_ptr() as u64, buf_st1.as_ptr() as u64,
                buf_log2.as_ptr() as u64, buf_win1.as_ptr() as u64,
                buf_tgt1.as_ptr() as u64,
                g_t_f1.as_mut_ptr() as u64, g_i_f1.as_mut_ptr() as u64,
                g_b_f1.as_mut_ptr() as u64, g_pos1.as_mut_ptr() as u64,
                g_lat1.as_mut_ptr() as u64, g_w_out1.as_mut_ptr() as u64,
                g_b_out1.as_mut_ptr() as u64,
            ).unwrap();
            rt.drain().unwrap();
        }
        let mut bwd1_times = Vec::new();
        for _ in 0..REPS {
            // Reset grads
            unsafe { hip_h2d(g_t_f1.as_mut_ptr(), &vec![0u32; n_t_f]).unwrap(); }
            let t0 = Instant::now();
            unsafe {
                rt.dispatch_backward(params_bwd1,
                    buf_t_f.as_ptr() as u64, buf_lat.as_ptr() as u64,
                    buf_w_out.as_ptr() as u64, buf_st1.as_ptr() as u64,
                    buf_log2.as_ptr() as u64, buf_win1.as_ptr() as u64,
                    buf_tgt1.as_ptr() as u64,
                    g_t_f1.as_mut_ptr() as u64, g_i_f1.as_mut_ptr() as u64,
                    g_b_f1.as_mut_ptr() as u64, g_pos1.as_mut_ptr() as u64,
                    g_lat1.as_mut_ptr() as u64, g_w_out1.as_mut_ptr() as u64,
                    g_b_out1.as_mut_ptr() as u64,
                ).unwrap();
                rt.drain().unwrap();
            }
            bwd1_times.push(t0.elapsed());
        }
        let bwd1_ms = mean_ms(&bwd1_times);
        let bwd64_ms = mean_ms(&bwd_times);
        println!("  mean backward+drain (batch=1)  = {:.3} ms", bwd1_ms);
        println!("  mean backward+drain (batch=64) = {:.3} ms", bwd64_ms);
        println!("  contention ratio: {:.1}×  (expected: 1 if no inter-WG conflict)", bwd64_ms / bwd1_ms);

        // ---- Resumo ----
        let fwd = mean_ms(&fwd_times);
        let bwd = mean_ms(&bwd_times);
        let h2d = mean_ms(&h2d_times);
        let d2h = mean_ms(&d2h_times);
        let total = mean_ms(&total_times);
        let other = total - fwd - bwd - h2d - d2h;

        println!("=== BREAKDOWN ===");
        println!("  forward+drain  = {:8.3} ms  ({:.1}%)", fwd, fwd/total*100.0);
        println!("  backward+drain = {:8.3} ms  ({:.1}%)", bwd, bwd/total*100.0);
        println!("  2×h2d          = {:8.3} ms  ({:.1}%)", h2d, h2d/total*100.0);
        println!("  d2h logits     = {:8.3} ms  ({:.1}%)", d2h, d2h/total*100.0);
        println!("  other (SGD+...) = {:7.3} ms  ({:.1}%)", other, other/total*100.0);
        println!("  TOTAL measured = {:8.3} ms", total);
    }

    Ok(())
}
