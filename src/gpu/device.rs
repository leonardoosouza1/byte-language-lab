//! Device-resident tensors + op chaining — speed phase 2.
//!
//! The per-op `gemm`/`nn_ops` helpers each upload inputs, dispatch, and read
//! the result back to the host. In a chained forward (QKV → scores → softmax →
//! attn → FFN → ...) every intermediate makes a round-trip and a full
//! `poll(Wait)` sync. That host↔device traffic — not the math — dominates.
//!
//! `Tensor` keeps data in a persistent GPU buffer. `gemm_rec` RECORDS a matmul
//! into a shared command encoder (no submit, no readback) reusing the cached
//! "gemm" pipeline. A whole chain is built in one encoder, submitted once, and
//! only the final tensor is downloaded. Weights are uploaded once and reused
//! across steps. This is the foundation the batched 1GB-corpus trainer needs.

use crate::gpu::{gemm, GpuContext};
use wgpu::util::DeviceExt;

/// A matrix [rows, cols] resident in GPU memory (row-major, f32).
pub struct Tensor {
    pub buf: wgpu::Buffer,
    pub rows: usize,
    pub cols: usize,
}

impl GpuContext {
    /// Upload host data into a persistent STORAGE buffer (readable as GEMM input
    /// and writable as output, so intermediates can be reused in place).
    pub fn upload(&self, data: &[f32], rows: usize, cols: usize) -> Tensor {
        assert_eq!(data.len(), rows * cols, "shape mismatch");
        let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tensor.upload"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        Tensor { buf, rows, cols }
    }

    /// Allocate an uninitialized device tensor (for op outputs).
    pub fn alloc(&self, rows: usize, cols: usize) -> Tensor {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tensor.alloc"),
            size: (rows * cols * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Tensor { buf, rows, cols }
    }

    /// Download a device tensor to the host (one map/poll — use sparingly).
    pub fn download(&self, t: &Tensor) -> Vec<f32> {
        let sz = (t.rows * t.cols * 4) as u64;
        let stg = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tensor.staging"), size: sz,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&t.buf, 0, &stg, 0, sz);
        self.queue.submit(Some(enc.finish()));
        let sl = stg.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        sl.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = sl.get_mapped_range();
        let out = bytemuck::cast_slice(&data).to_vec();
        drop(data); stg.unmap();
        out
    }
}

/// Record `out = a · b` into `enc` (no submit, no readback). Shapes:
/// a [m,k], b [k,n], out [m,n]. Reuses the cached "gemm" pipeline.
pub fn gemm_rec(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder,
                a: &Tensor, b: &Tensor, out: &Tensor) {
    assert_eq!(a.cols, b.rows, "inner dim mismatch");
    assert_eq!(out.rows, a.rows);
    assert_eq!(out.cols, b.cols);
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let dims = [m as u32, n as u32, k as u32, 0u32];
    let dim_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gemm_rec.dims"), contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let pipeline = ctx.pipeline("gemm", gemm::WGSL);
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gemm_rec.bind"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dim_buf.as_entire_binding() },
        ],
    });
    let mut pass = enc.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1);
}

/// `out[m,n] = A[m,k] · B[n,k]^T` — A times B-transposed, used for scores
/// (q·kᵀ) and tied lm_head (xf·embedᵀ) without a separate transpose op.
const WGSL_NT: &str = r#"
struct Dims { m: u32, n: u32, k: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>        a: array<f32>;
@group(0) @binding(1) var<storage, read>        b: array<f32>;
@group(0) @binding(2) var<storage, read_write>  c: array<f32>;
@group(0) @binding(3) var<uniform>              d: Dims;
var<workgroup> tileA: array<f32, 256>;
var<workgroup> tileB: array<f32, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>,
        @builtin(workgroup_id)         wid: vec3<u32>) {
    let row = gid.y; let col = gid.x;
    let lr = lid.y;  let lc = lid.x;
    var acc: f32 = 0.0;
    let n_tiles = (d.k + 15u) / 16u;
    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        let a_col = t * 16u + lc;
        if (row < d.m && a_col < d.k) { tileA[lr*16u+lc] = a[row*d.k + a_col]; }
        else { tileA[lr*16u+lc] = 0.0; }
        let b_col = wid.x * 16u + lc;   // n-index
        let b_k   = t * 16u + lr;
        if (b_col < d.n && b_k < d.k) { tileB[lr*16u+lc] = b[b_col*d.k + b_k]; }
        else { tileB[lr*16u+lc] = 0.0; }
        workgroupBarrier();
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            acc = acc + tileA[lr*16u+i] * tileB[i*16u+lc];
        }
        workgroupBarrier();
    }
    if (row < d.m && col < d.n) { c[row*d.n + col] = acc; }
}
"#;

/// Record `out = a · bᵀ` (b is [n,k]) into `enc`.
pub fn gemm_nt_rec(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder,
                   a: &Tensor, b: &Tensor, out: &Tensor) {
    assert_eq!(a.cols, b.cols, "shared dim k mismatch");
    assert_eq!(out.rows, a.rows);
    assert_eq!(out.cols, b.rows);
    let (m, k, n) = (a.rows, a.cols, b.rows);
    let dims = [m as u32, n as u32, k as u32, 0u32];
    let dim_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("nt.dims"), contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM });
    let pipeline = ctx.pipeline("gemm_nt", WGSL_NT);
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nt.bind"), layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dim_buf.as_entire_binding() }]});
    let mut pass = enc.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1);
}

/// Record a row op (op 0=softmax, 1=rmsnorm, 2=causal softmax) into `enc`.
/// `wgt` is the RMSNorm weight [cols]; pass any tensor for softmax (unused).
pub fn row_rec(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder,
               inp: &Tensor, wgt: &Tensor, out: &Tensor, eps: f32, op: u32) {
    let dims = [inp.rows as u32, inp.cols as u32, eps.to_bits(), op];
    let dim_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("row.dims"), contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM });
    let pipeline = ctx.pipeline("nn_row", crate::gpu::nn_ops::ROW_WGSL);
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("row.bind"), layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: inp.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgt.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dim_buf.as_entire_binding() }]});
    let mut pass = enc.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups(inp.rows as u32, 1, 1);
}

/// Record an elementwise op (op 0=SiLU, 1=GELU) into `enc`. out may equal inp.
pub fn elem_rec(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder,
                inp: &Tensor, out: &Tensor, op: u32) {
    let n = inp.rows * inp.cols;
    let dims = [n as u32, op];
    let dim_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("elem.dims"), contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM });
    let pipeline = ctx.pipeline("nn_elem", crate::gpu::nn_ops::ELEM_WGSL);
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("elem.bind"), layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: inp.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dim_buf.as_entire_binding() }]});
    let mut pass = enc.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
}

/// Record `out[i] = a[i] + b[i]` (residual add) into `enc`.
const WGSL_ADD: &str = r#"
struct Dims { n: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;
@compute @workgroup_size(256,1,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= d.n) { return; }
    o[i] = a[i] + b[i];
}
"#;

/// Record element-wise `out = a + b` into `enc`.
pub fn add_rec(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder,
               a: &Tensor, b: &Tensor, out: &Tensor) {
    let n = a.rows * a.cols;
    let dims = [n as u32, 0u32];
    let dim_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("add.dims"), contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM });
    let pipeline = ctx.pipeline("add", WGSL_ADD);
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("add.bind"), layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dim_buf.as_entire_binding() }]});
    let mut pass = enc.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    /// A chain of GEMMs recorded into ONE encoder, submitted once, downloaded
    /// once — must equal the same chain done with the per-op readback gemm.
    /// Proves device-resident chaining is correct and quantifies the win.
    #[test]
    fn device_chain_parity_and_speed() {
        let ctx = match GpuContext::new() { Ok(c) => c, Err(_) => return };
        let (m, k, n) = (64, 64, 64);
        let mut s = 99u64;
        let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
        let b: Vec<f32> = (0..k * n).map(|_| lcg(&mut s)).collect();
        let c: Vec<f32> = (0..k * n).map(|_| lcg(&mut s)).collect();

        // reference: per-op readback gemm, A·B then ·C
        let ref1 = gemm::gemm(&ctx, &a, &b, m, k, n);
        let refr = gemm::gemm(&ctx, &ref1, &c, m, k, n);

        // device-resident: upload once, record both gemms, one submit, one download
        let ta = ctx.upload(&a, m, k);
        let tb = ctx.upload(&b, k, n);
        let tc = ctx.upload(&c, k, n);
        let t1 = ctx.alloc(m, n);
        let t2 = ctx.alloc(m, n);
        let t0 = std::time::Instant::now();
        let iters = 200;
        for _ in 0..iters {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            gemm_rec(&ctx, &mut enc, &ta, &tb, &t1);
            gemm_rec(&ctx, &mut enc, &t1, &tc, &t2);
            ctx.queue.submit(Some(enc.finish()));
        }
        ctx.device.poll(wgpu::Maintain::Wait);
        let dev_chain = ctx.download(&t2);
        let dev_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // per-op timing for comparison
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let r1 = gemm::gemm(&ctx, &a, &b, m, k, n);
            let _ = gemm::gemm(&ctx, &r1, &c, m, k, n);
        }
        let perop_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let max_err = dev_chain.iter().zip(&refr)
            .map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        eprintln!("device-chain {dev_ms:.3} ms/iter  vs per-op {perop_ms:.3} ms/iter  \
                   ({:.1}x)  max_err {max_err:.2e}", perop_ms / dev_ms);
        assert!(max_err < 1e-3, "device chain disagrees: {max_err}");
    }
}
