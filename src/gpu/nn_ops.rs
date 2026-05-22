//! Transformer element/row ops on wgpu/WGSL — the non-GEMM kernels the
//! byte-transformer needs: row-wise softmax (attention), RMSNorm (Qwen/Llama
//! normalization), and SiLU (SwiGLU activation). All size-parametric.
//!
//! Row ops use one workgroup per row with a 256-thread tree reduction in
//! shared memory; elementwise ops use one thread per element. Bounds-checked
//! so any [rows, cols] / length works.

use crate::gpu::GpuContext;
use wgpu::util::DeviceExt;

const ROW_WGSL: &str = r#"
struct Dims { rows: u32, cols: u32, eps: f32, op: u32 };
@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read>       wgt: array<f32>;   // RMSNorm weight (cols), unused for softmax
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

var<workgroup> red: array<f32, 256>;

// op 0 = softmax (row), op 1 = rmsnorm (row, * wgt), op 2 = causal softmax
// (row r attends only to cols j <= r — attention causal mask)
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    if (row >= d.rows) { return; }
    let base = row * d.cols;
    // causal: a column j is valid only if j <= row
    let limit = select(d.cols, row + 1u, d.op == 2u);

    if (d.op == 0u || d.op == 2u) {
        // ---- softmax: max, then exp-sum, then normalize ----
        var local_max: f32 = -3.0e38;
        var i: u32 = tid;
        loop {
            if (i >= limit) { break; }
            local_max = max(local_max, inp[base + i]);
            i = i + 256u;
        }
        red[tid] = local_max;
        workgroupBarrier();
        var s: u32 = 128u;
        loop {
            if (s == 0u) { break; }
            if (tid < s) { red[tid] = max(red[tid], red[tid + s]); }
            workgroupBarrier();
            s = s / 2u;
        }
        let row_max = red[0];
        workgroupBarrier();

        var local_sum: f32 = 0.0;
        i = tid;
        loop {
            if (i >= d.cols) { break; }
            local_sum = local_sum + exp(inp[base + i] - row_max);
            i = i + 256u;
        }
        red[tid] = local_sum;
        workgroupBarrier();
        s = 128u;
        loop {
            if (s == 0u) { break; }
            if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
            workgroupBarrier();
            s = s / 2u;
        }
        let row_sum = red[0];
        workgroupBarrier();

        i = tid;
        loop {
            if (i >= d.cols) { break; }
            out[base + i] = exp(inp[base + i] - row_max) / row_sum;
            i = i + 256u;
        }
    } else {
        // ---- rmsnorm: x / sqrt(mean(x^2)+eps) * weight ----
        var local_ss: f32 = 0.0;
        var i: u32 = tid;
        loop {
            if (i >= d.cols) { break; }
            let v = inp[base + i];
            local_ss = local_ss + v * v;
            i = i + 256u;
        }
        red[tid] = local_ss;
        workgroupBarrier();
        var s: u32 = 128u;
        loop {
            if (s == 0u) { break; }
            if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
            workgroupBarrier();
            s = s / 2u;
        }
        let mean_ss = red[0] / f32(d.cols);
        let inv = 1.0 / sqrt(mean_ss + d.eps);
        workgroupBarrier();

        i = tid;
        loop {
            if (i >= d.cols) { break; }
            out[base + i] = inp[base + i] * inv * wgt[i];
            i = i + 256u;
        }
    }
}
"#;

const ELEM_WGSL: &str = r#"
struct Dims { n: u32, op: u32 };
@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             d: Dims;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= d.n) { return; }
    let x = inp[i];
    if (d.op == 0u) {
        out[i] = x / (1.0 + exp(-x));            // SiLU
    } else {
        // GELU tanh approx
        let c = 0.7978845608 * (x + 0.044715 * x * x * x);
        out[i] = 0.5 * x * (1.0 + tanh(c));
    }
}
"#;

fn run_row(ctx: &GpuContext, inp: &[f32], wgt: &[f32], rows: usize, cols: usize,
           eps: f32, op: u32) -> Vec<f32> {
    let dev = &ctx.device; let q = &ctx.queue;
    let ib = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(inp), usage: wgpu::BufferUsages::STORAGE });
    let wb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(wgt), usage: wgpu::BufferUsages::STORAGE });
    let sz = (rows * cols * 4) as u64;
    let ob = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: sz,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let dims = [rows as u32, cols as u32, eps.to_bits(), op];
    let db = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&dims), usage: wgpu::BufferUsages::UNIFORM });
    let p = ctx.pipeline("nn_row", ROW_WGSL);
    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor { label: None,
        layout: &p.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ib.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ob.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: db.as_entire_binding() }]});
    let mut e = dev.create_command_encoder(&Default::default());
    { let mut pass = e.begin_compute_pass(&Default::default());
      pass.set_pipeline(&p); pass.set_bind_group(0, &bg, &[]);
      pass.dispatch_workgroups(rows as u32, 1, 1); }
    read_back(ctx, &mut e, &ob, sz)
}

fn read_back(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, src: &wgpu::Buffer, sz: u64) -> Vec<f32> {
    let dev = &ctx.device;
    let stg = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: sz,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    enc.copy_buffer_to_buffer(src, 0, &stg, 0, sz);
    // encoder finished by caller's submit — but we own it here:
    let enc2 = std::mem::replace(enc, dev.create_command_encoder(&Default::default()));
    ctx.queue.submit(Some(enc2.finish()));
    let sl = stg.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    sl.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    dev.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let data = sl.get_mapped_range();
    let out = bytemuck::cast_slice(&data).to_vec();
    drop(data); stg.unmap();
    out
}

/// Row-wise softmax over the last dim of an [rows, cols] row-major matrix.
pub fn softmax(ctx: &GpuContext, inp: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    run_row(ctx, inp, &[0.0], rows, cols, 0.0, 0)
}

/// Causal row-wise softmax: row r attends only to cols j <= r (attention mask).
/// Cols j > r get probability 0. Input/output [rows, cols], rows==cols typical.
pub fn causal_softmax(ctx: &GpuContext, inp: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    run_row(ctx, inp, &[0.0], rows, cols, 0.0, 2)
}

/// Row-wise RMSNorm: out[r,c] = x/sqrt(mean(x^2)+eps) * weight[c].
pub fn rmsnorm(ctx: &GpuContext, inp: &[f32], weight: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    run_row(ctx, inp, weight, rows, cols, eps, 1)
}

/// Elementwise SiLU (op 0) or GELU (op 1).
fn run_elem(ctx: &GpuContext, inp: &[f32], op: u32) -> Vec<f32> {
    let dev = &ctx.device;
    let n = inp.len();
    let ib = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(inp), usage: wgpu::BufferUsages::STORAGE });
    let sz = (n * 4) as u64;
    let ob = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: sz,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let dims = [n as u32, op];
    let db = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&dims), usage: wgpu::BufferUsages::UNIFORM });
    let p = ctx.pipeline("nn_elem", ELEM_WGSL);
    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor { label: None,
        layout: &p.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ib.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: ob.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: db.as_entire_binding() }]});
    let mut e = dev.create_command_encoder(&Default::default());
    { let mut pass = e.begin_compute_pass(&Default::default());
      pass.set_pipeline(&p); pass.set_bind_group(0, &bg, &[]);
      pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1); }
    read_back(ctx, &mut e, &ob, sz)
}

pub fn silu(ctx: &GpuContext, inp: &[f32]) -> Vec<f32> { run_elem(ctx, inp, 0) }
pub fn gelu(ctx: &GpuContext, inp: &[f32]) -> Vec<f32> { run_elem(ctx, inp, 1) }

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f32 / u32::MAX as f32) * 4.0 - 2.0
    }

    #[test]
    fn softmax_parity() {
        let ctx = match GpuContext::new() { Ok(c) => c, Err(_) => return };
        let mut s = 1u64;
        for &(rows, cols) in &[(4, 10), (8, 300), (1, 1000), (33, 7)] {
            let inp: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut s)).collect();
            let gpu = softmax(&ctx, &inp, rows, cols);
            // cpu ref
            let mut cpu = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let row = &inp[r * cols..(r + 1) * cols];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = row.iter().map(|x| (x - mx).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for c in 0..cols { cpu[r * cols + c] = exps[c] / sum; }
            }
            let err = gpu.iter().zip(&cpu).map(|(g, c)| (g - c).abs()).fold(0.0f32, f32::max);
            assert!(err < 1e-4, "softmax {rows}x{cols} err {err}");
        }
    }

    #[test]
    fn rmsnorm_parity() {
        let ctx = match GpuContext::new() { Ok(c) => c, Err(_) => return };
        let mut s = 99u64;
        let eps = 1e-6;
        for &(rows, cols) in &[(4, 16), (8, 256), (2, 513)] {
            let inp: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut s)).collect();
            let w: Vec<f32> = (0..cols).map(|_| lcg(&mut s)).collect();
            let gpu = rmsnorm(&ctx, &inp, &w, rows, cols, eps);
            let mut cpu = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let row = &inp[r * cols..(r + 1) * cols];
                let ms: f32 = row.iter().map(|x| x * x).sum::<f32>() / cols as f32;
                let inv = 1.0 / (ms + eps).sqrt();
                for c in 0..cols { cpu[r * cols + c] = row[c] * inv * w[c]; }
            }
            let err = gpu.iter().zip(&cpu).map(|(g, c)| (g - c).abs()).fold(0.0f32, f32::max);
            assert!(err < 1e-3, "rmsnorm {rows}x{cols} err {err}");
        }
    }

    #[test]
    fn silu_gelu_parity() {
        let ctx = match GpuContext::new() { Ok(c) => c, Err(_) => return };
        let mut s = 7u64;
        let inp: Vec<f32> = (0..1000).map(|_| lcg(&mut s)).collect();
        let g = silu(&ctx, &inp);
        let err = g.iter().zip(&inp).map(|(o, x)| (o - x / (1.0 + (-x).exp())).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-4, "silu err {err}");
        let gg = gelu(&ctx, &inp);
        let errg = gg.iter().zip(&inp).map(|(o, x)| {
            let c = 0.7978845608 * (x + 0.044715 * x * x * x);
            (o - 0.5 * x * (1.0 + c.tanh())).abs()
        }).fold(0.0f32, f32::max);
        assert!(errg < 1e-4, "gelu err {errg}");
    }
}
