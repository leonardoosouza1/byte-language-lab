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

/// `out[k,n] = A[m,k]^T · B[m,n]` — A-transposed times B, used for weight
/// gradients (dW = Xᵀ·dY) in the device-resident backward.
const WGSL_TN: &str = r#"
struct Dims { m: u32, n: u32, k: u32, _pad: u32 };   // A is [m,k], B is [m,n], out is [k,n]
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
    let row = gid.y;   // 0..k  (output row)
    let col = gid.x;   // 0..n
    let lr = lid.y; let lc = lid.x;
    var acc: f32 = 0.0;
    let n_tiles = (d.m + 15u) / 16u;        // contract over m
    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        // need A[t*16+i, row] and B[t*16+i, col] for i in 0..16
        let a_m = t * 16u + lr;             // m-index for tileA staging
        let a_row = wid.y * 16u + lc;       // k-index (= output row range)
        if (a_m < d.m && a_row < d.k) { tileA[lr*16u+lc] = a[a_m*d.k + a_row]; }
        else { tileA[lr*16u+lc] = 0.0; }
        let b_m = t * 16u + lr;
        if (b_m < d.m && col < d.n) { tileB[lr*16u+lc] = b[b_m*d.n + col]; }
        else { tileB[lr*16u+lc] = 0.0; }
        workgroupBarrier();
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            acc = acc + tileA[i*16u+lr] * tileB[i*16u+lc];
        }
        workgroupBarrier();
    }
    if (row < d.k && col < d.n) { c[row*d.n + col] = acc; }
}
"#;

/// Record `out = aᵀ · b` (a is [m,k], b is [m,n], out is [k,n]) into `enc`.
pub fn gemm_tn_rec(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder,
                   a: &Tensor, b: &Tensor, out: &Tensor) {
    assert_eq!(a.rows, b.rows, "contract dim m mismatch");
    assert_eq!(out.rows, a.cols);
    assert_eq!(out.cols, b.cols);
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let dims = [m as u32, n as u32, k as u32, 0u32];
    let dim_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tn.dims"), contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM });
    let pipeline = ctx.pipeline("gemm_tn", WGSL_TN);
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tn.bind"), layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dim_buf.as_entire_binding() }]});
    let mut pass = enc.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((n as u32).div_ceil(16), (k as u32).div_ceil(16), 1);
}

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

// ======================= backward-pass device kernels =======================
// Each mirrors a CPU helper in transformer.rs and is parity-checked against it.

/// dh = dha * silu'(z),  silu'(z) = s + z·s·(1-s),  s = sigmoid(z),  z = h.
const WGSL_SILU_BWD: &str = r#"
struct D { n: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       dha: array<f32>;
@group(0) @binding(1) var<storage, read>       hh:  array<f32>;
@group(0) @binding(2) var<storage, read_write> o:   array<f32>;
@group(0) @binding(3) var<uniform>             d:   D;
@compute @workgroup_size(256,1,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= d.n) { return; }
    let z = hh[i]; let s = 1.0 / (1.0 + exp(-z));
    o[i] = dha[i] * (s + z * s * (1.0 - s));
}
"#;

pub fn silu_bwd_rec(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder,
                    dha: &Tensor, h: &Tensor, out: &Tensor) {
    let n = h.rows * h.cols;
    elem2(ctx, enc, "silu_bwd", WGSL_SILU_BWD, dha, h, out, &[n as u32, 0]);
}

/// In-place scale: t *= scalar. (single binding so the same buffer is never
/// both read and read_write in one dispatch — wgpu forbids that aliasing.)
const WGSL_SCALE: &str = r#"
struct D { n: u32, s: f32 };
@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<uniform>             d: D;
@compute @workgroup_size(256,1,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= d.n) { return; }
    w[i] = w[i] * d.s;
}
"#;

/// Record in-place `t *= s` into `enc`.
pub fn scale_rec(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, t: &Tensor, s: f32) {
    let n = t.rows * t.cols;
    let dims = [n as u32, s.to_bits()];
    let dim_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scale.dims"), contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM });
    let pipeline = ctx.pipeline("scale", WGSL_SCALE);
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: t.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dim_buf.as_entire_binding() }]});
    let mut pass = enc.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
}

/// Causal softmax backward: ds[r,j<=r] = p[r,j]*(g[r,j] - sum_k p[r,k]g[r,k]).
const WGSL_CSM_BWD: &str = r#"
struct D { rows: u32, cols: u32, _e: u32, _o: u32 };
@group(0) @binding(0) var<storage, read>       g: array<f32>;   // dprobs
@group(0) @binding(1) var<storage, read>       p: array<f32>;   // probs
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform>             d: D;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256,1,1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = wid.x; let tid = lid.x;
    if (r >= d.rows) { return; }
    let base = r * d.cols; let lim = r + 1u;
    var ld: f32 = 0.0; var i = tid;
    loop { if (i >= lim) { break; } ld = ld + p[base+i]*g[base+i]; i = i + 256u; }
    red[tid] = ld; workgroupBarrier();
    var s: u32 = 128u;
    loop { if (s == 0u) { break; } if (tid < s) { red[tid] = red[tid] + red[tid+s]; } workgroupBarrier(); s = s/2u; }
    let dot = red[0]; workgroupBarrier();
    i = tid;
    loop { if (i >= d.cols) { break; }
        if (i < lim) { o[base+i] = p[base+i]*(g[base+i]-dot); } else { o[base+i] = 0.0; }
        i = i + 256u; }
}
"#;

pub fn csm_bwd_rec(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder,
                   dprobs: &Tensor, probs: &Tensor, out: &Tensor) {
    row2(ctx, enc, "csm_bwd", WGSL_CSM_BWD, dprobs, probs, out,
         &[dprobs.rows as u32, dprobs.cols as u32, 0, 0], dprobs.rows);
}

/// RMSNorm input grad: dx_j = inv·w_j·dy_j - coef·x_j,
/// inv=1/sqrt(mean(x²)+eps),  coef = inv³/cols · sum_j dy_j·w_j·x_j.
const WGSL_RMS_IN_BWD: &str = r#"
struct D { rows: u32, cols: u32, eps: f32, _o: u32 };
@group(0) @binding(0) var<storage, read>       dy: array<f32>;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       w:  array<f32>;
@group(0) @binding(3) var<storage, read_write> o:  array<f32>;
@group(0) @binding(4) var<uniform>             d:  D;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256,1,1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = wid.x; let tid = lid.x;
    if (r >= d.rows) { return; }
    let base = r * d.cols;
    // mean square
    var ss: f32 = 0.0; var i = tid;
    loop { if (i >= d.cols) { break; } let v = x[base+i]; ss = ss + v*v; i = i + 256u; }
    red[tid] = ss; workgroupBarrier();
    var s: u32 = 128u;
    loop { if (s==0u){break;} if (tid<s){red[tid]=red[tid]+red[tid+s];} workgroupBarrier(); s=s/2u; }
    let inv = 1.0 / sqrt(red[0]/f32(d.cols) + d.eps); workgroupBarrier();
    // dot = sum dy*w*x
    var dt: f32 = 0.0; i = tid;
    loop { if (i >= d.cols) { break; } dt = dt + dy[base+i]*w[i]*x[base+i]; i = i + 256u; }
    red[tid] = dt; workgroupBarrier();
    s = 128u;
    loop { if (s==0u){break;} if (tid<s){red[tid]=red[tid]+red[tid+s];} workgroupBarrier(); s=s/2u; }
    let coef = inv*inv*inv/f32(d.cols)*red[0]; workgroupBarrier();
    i = tid;
    loop { if (i >= d.cols) { break; }
        o[base+i] = inv*w[i]*dy[base+i] - coef*x[base+i]; i = i + 256u; }
}
"#;

pub fn rms_in_bwd_rec(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder,
                      dy: &Tensor, x: &Tensor, w: &Tensor, out: &Tensor, eps: f32) {
    let dims = [dy.rows as u32, dy.cols as u32, eps.to_bits(), 0];
    let dim_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rmsin.dims"), contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM });
    let pipeline = ctx.pipeline("rms_in_bwd", WGSL_RMS_IN_BWD);
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dy.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: w.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: dim_buf.as_entire_binding() }]});
    let mut pass = enc.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups(dy.rows as u32, 1, 1);
}

/// inv[r] = 1/sqrt(mean(x[r]²)+eps) — per-row, feeds the weight grad.
const WGSL_RMS_INV: &str = r#"
struct D { rows: u32, cols: u32, eps: f32, _o: u32 };
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;   // [rows]
@group(0) @binding(2) var<uniform>             d: D;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256,1,1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = wid.x; let tid = lid.x;
    if (r >= d.rows) { return; }
    let base = r * d.cols;
    var ss: f32 = 0.0; var i = tid;
    loop { if (i >= d.cols) { break; } let v = x[base+i]; ss = ss + v*v; i = i + 256u; }
    red[tid] = ss; workgroupBarrier();
    var s: u32 = 128u;
    loop { if (s==0u){break;} if (tid<s){red[tid]=red[tid]+red[tid+s];} workgroupBarrier(); s=s/2u; }
    if (tid == 0u) { o[r] = 1.0 / sqrt(red[0]/f32(d.cols) + d.eps); }
}
"#;

/// RMSNorm weight grad: dw[c] = sum_r dy[r,c]·x[r,c]·inv[r].
const WGSL_RMS_W_BWD: &str = r#"
struct D { rows: u32, cols: u32, _e: u32, _o: u32 };
@group(0) @binding(0) var<storage, read>       dy:  array<f32>;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       inv: array<f32>;
@group(0) @binding(3) var<storage, read_write> o:   array<f32>;
@group(0) @binding(4) var<uniform>             d:   D;
@compute @workgroup_size(256,1,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x; if (c >= d.cols) { return; }
    var acc: f32 = 0.0;
    for (var r: u32 = 0u; r < d.rows; r = r + 1u) {
        acc = acc + dy[r*d.cols+c]*x[r*d.cols+c]*inv[r];
    }
    o[c] = acc;
}
"#;

pub fn rms_w_bwd_rec(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder,
                     dy: &Tensor, x: &Tensor, inv_scratch: &Tensor, out: &Tensor, eps: f32) {
    let (rows, cols) = (dy.rows, dy.cols);
    // 1) inv[r]
    let dims = [rows as u32, cols as u32, eps.to_bits(), 0];
    let db = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rmsinv.dims"), contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM });
    let p1 = ctx.pipeline("rms_inv", WGSL_RMS_INV);
    let bg1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &p1.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: x.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: inv_scratch.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: db.as_entire_binding() }]});
    { let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&p1); pass.set_bind_group(0, &bg1, &[]);
      pass.dispatch_workgroups(rows as u32, 1, 1); }
    // 2) dw[c]
    let db2 = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rmsw.dims"), contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM });
    let p2 = ctx.pipeline("rms_w_bwd", WGSL_RMS_W_BWD);
    let bg2 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &p2.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dy.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: inv_scratch.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: db2.as_entire_binding() }]});
    let mut pass = enc.begin_compute_pass(&Default::default());
    pass.set_pipeline(&p2); pass.set_bind_group(0, &bg2, &[]);
    pass.dispatch_workgroups((cols as u32).div_ceil(256), 1, 1);
}

/// In-place SGD: w -= lr · g.
const WGSL_SGD: &str = r#"
struct D { n: u32, lr: f32 };
@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<storage, read>       g: array<f32>;
@group(0) @binding(2) var<uniform>             d: D;
@compute @workgroup_size(256,1,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= d.n) { return; }
    w[i] = w[i] - d.lr * g[i];
}
"#;

/// Record `w -= lr · g` into `enc` (both device tensors, same length).
pub fn sgd_rec(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder,
               w: &Tensor, g: &Tensor, lr: f32) {
    let n = w.rows * w.cols;
    let dims = [n as u32, lr.to_bits()];
    let dim_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sgd.dims"), contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM });
    let pipeline = ctx.pipeline("sgd", WGSL_SGD);
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: w.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: g.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dim_buf.as_entire_binding() }]});
    let mut pass = enc.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
}

// shared helper: elementwise op with two read inputs + one output.
fn elem2(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, key: &'static str, wgsl: &str,
         a: &Tensor, b: &Tensor, out: &Tensor, dims: &[u32]) {
    let dim_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(dims), usage: wgpu::BufferUsages::UNIFORM });
    let pipeline = ctx.pipeline(key, wgsl);
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dim_buf.as_entire_binding() }]});
    let n = a.rows * a.cols;
    let mut pass = enc.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
}

// shared helper: row op with two read inputs + one output, one workgroup/row.
fn row2(ctx: &GpuContext, enc: &mut wgpu::CommandEncoder, key: &'static str, wgsl: &str,
        a: &Tensor, b: &Tensor, out: &Tensor, dims: &[u32], rows: usize) {
    let dim_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(dims), usage: wgpu::BufferUsages::UNIFORM });
    let pipeline = ctx.pipeline(key, wgsl);
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dim_buf.as_entire_binding() }]});
    let mut pass = enc.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups(rows as u32, 1, 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    /// Every backward kernel (silu', causal-softmax grad, rmsnorm input &
    /// weight grads, scale) must match the CPU formula it ports.
    #[test]
    fn backward_kernels_parity() {
        let ctx = match GpuContext::new() { Ok(c) => c, Err(_) => return };
        let mut s = 3u64;
        let (rows, cols) = (16usize, 24usize);
        let eps = 1e-6f32;
        let mk = |s: &mut u64, n: usize| (0..n).map(|_| lcg(s)).collect::<Vec<f32>>();
        let mxerr = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x,y)|(x-y).abs()).fold(0.0f32,f32::max);

        // --- silu_bwd ---
        let dha = mk(&mut s, rows*cols); let hh = mk(&mut s, rows*cols);
        let mut ref_silu = vec![0.0f32; rows*cols];
        for i in 0..rows*cols { let z = hh[i]; let sg = 1.0/(1.0+(-z).exp());
            ref_silu[i] = dha[i]*(sg + z*sg*(1.0-sg)); }
        let (tda, thh, to) = (ctx.upload(&dha,rows,cols), ctx.upload(&hh,rows,cols), ctx.alloc(rows,cols));
        let mut e = ctx.device.create_command_encoder(&Default::default());
        silu_bwd_rec(&ctx,&mut e,&tda,&thh,&to); ctx.queue.submit(Some(e.finish()));
        let g_silu = ctx.download(&to);

        // --- scale (in place) ---
        let aa = mk(&mut s, rows*cols);
        let ref_sc: Vec<f32> = aa.iter().map(|x| x*0.125).collect();
        let taa = ctx.upload(&aa,rows,cols);
        let mut e = ctx.device.create_command_encoder(&Default::default());
        scale_rec(&ctx,&mut e,&taa,0.125); ctx.queue.submit(Some(e.finish()));
        let g_sc = ctx.download(&taa);

        // --- causal softmax bwd (rows==cols square) ---
        let n = 20usize;
        let probs = mk(&mut s, n*n); let dprobs = mk(&mut s, n*n);
        let mut ref_csm = vec![0.0f32; n*n];
        for r in 0..n { let lim = r+1; let mut dot = 0.0;
            for j in 0..lim { dot += probs[r*n+j]*dprobs[r*n+j]; }
            for j in 0..lim { ref_csm[r*n+j] = probs[r*n+j]*(dprobs[r*n+j]-dot); } }
        let (tp, tg, tcsm) = (ctx.upload(&probs,n,n), ctx.upload(&dprobs,n,n), ctx.alloc(n,n));
        let mut e = ctx.device.create_command_encoder(&Default::default());
        csm_bwd_rec(&ctx,&mut e,&tg,&tp,&tcsm); ctx.queue.submit(Some(e.finish()));
        let g_csm = ctx.download(&tcsm);

        // --- rmsnorm input grad ---
        let dy = mk(&mut s, rows*cols); let x = mk(&mut s, rows*cols); let w = mk(&mut s, cols);
        let mut ref_in = vec![0.0f32; rows*cols];
        for r in 0..rows {
            let ms: f32 = (0..cols).map(|j| x[r*cols+j]*x[r*cols+j]).sum::<f32>()/cols as f32;
            let inv = 1.0/(ms+eps).sqrt();
            let mut dot = 0.0; for j in 0..cols { dot += dy[r*cols+j]*w[j]*x[r*cols+j]; }
            let coef = inv*inv*inv/cols as f32*dot;
            for j in 0..cols { ref_in[r*cols+j] = inv*w[j]*dy[r*cols+j] - coef*x[r*cols+j]; }
        }
        let (tdy,tx,tw,tin) = (ctx.upload(&dy,rows,cols), ctx.upload(&x,rows,cols),
                               ctx.upload(&w,1,cols), ctx.alloc(rows,cols));
        let mut e = ctx.device.create_command_encoder(&Default::default());
        rms_in_bwd_rec(&ctx,&mut e,&tdy,&tx,&tw,&tin,eps); ctx.queue.submit(Some(e.finish()));
        let g_in = ctx.download(&tin);

        // --- rmsnorm weight grad ---
        let mut ref_w = vec![0.0f32; cols];
        for r in 0..rows {
            let ms: f32 = (0..cols).map(|j| x[r*cols+j]*x[r*cols+j]).sum::<f32>()/cols as f32;
            let inv = 1.0/(ms+eps).sqrt();
            for j in 0..cols { ref_w[j] += dy[r*cols+j]*x[r*cols+j]*inv; }
        }
        let (tinv, twg) = (ctx.alloc(rows,1), ctx.alloc(1,cols));
        let mut e = ctx.device.create_command_encoder(&Default::default());
        rms_w_bwd_rec(&ctx,&mut e,&tdy,&tx,&tinv,&twg,eps); ctx.queue.submit(Some(e.finish()));
        let g_w = ctx.download(&twg);

        let errs = [mxerr(&g_silu,&ref_silu), mxerr(&g_sc,&ref_sc), mxerr(&g_csm,&ref_csm),
                    mxerr(&g_in,&ref_in), mxerr(&g_w,&ref_w)];
        eprintln!("backward kernel max_err: silu={:.2e} scale={:.2e} csm={:.2e} rms_in={:.2e} rms_w={:.2e}",
                  errs[0],errs[1],errs[2],errs[3],errs[4]);
        for (i,&er) in errs.iter().enumerate() { assert!(er < 1e-3, "kernel {i} err {er}"); }
    }

    /// NT (A·Bᵀ) and TN (Aᵀ·B) device kernels must match a CPU reference —
    /// they're the operand-transpose GEMMs the forward (scores, lm_head) and
    /// backward (weight grads) rely on.
    #[test]
    fn nt_tn_parity() {
        let ctx = match GpuContext::new() { Ok(c) => c, Err(_) => return };
        let mut s = 7u64;
        // NT: a[m,k]·b[n,k]ᵀ = out[m,n]
        let (m, k, n) = (12, 20, 7);
        let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
        let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut s)).collect();
        let mut nt_ref = vec![0.0f32; m * n];
        for i in 0..m { for j in 0..n {
            let mut acc = 0.0; for p in 0..k { acc += a[i*k+p] * b[j*k+p]; }
            nt_ref[i*n+j] = acc;
        }}
        let ta = ctx.upload(&a, m, k); let tb = ctx.upload(&b, n, k);
        let to = ctx.alloc(m, n);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        gemm_nt_rec(&ctx, &mut enc, &ta, &tb, &to);
        ctx.queue.submit(Some(enc.finish()));
        let nt = ctx.download(&to);
        let nt_err = nt.iter().zip(&nt_ref).map(|(x,y)|(x-y).abs()).fold(0.0f32,f32::max);

        // TN: a[m,k]ᵀ·b[m,n] = out[k,n]
        let (m2, k2, n2) = (15, 9, 6);
        let a2: Vec<f32> = (0..m2 * k2).map(|_| lcg(&mut s)).collect();
        let b2: Vec<f32> = (0..m2 * n2).map(|_| lcg(&mut s)).collect();
        let mut tn_ref = vec![0.0f32; k2 * n2];
        for i in 0..k2 { for j in 0..n2 {
            let mut acc = 0.0; for p in 0..m2 { acc += a2[p*k2+i] * b2[p*n2+j]; }
            tn_ref[i*n2+j] = acc;
        }}
        let ta2 = ctx.upload(&a2, m2, k2); let tb2 = ctx.upload(&b2, m2, n2);
        let to2 = ctx.alloc(k2, n2);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        gemm_tn_rec(&ctx, &mut enc, &ta2, &tb2, &to2);
        ctx.queue.submit(Some(enc.finish()));
        let tn = ctx.download(&to2);
        let tn_err = tn.iter().zip(&tn_ref).map(|(x,y)|(x-y).abs()).fold(0.0f32,f32::max);

        eprintln!("NT max_err {nt_err:.2e}  TN max_err {tn_err:.2e}");
        assert!(nt_err < 1e-3, "NT mismatch {nt_err}");
        assert!(tn_err < 1e-3, "TN mismatch {tn_err}");
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
