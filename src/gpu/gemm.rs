//! Tiled GEMM on wgpu/WGSL — `C[M,N] = A[M,K] · B[K,N]` (row-major, f32).
//!
//! First foundational kernel of the byte-transformer student: every linear
//! layer (QKV proj, attention output, FFN up/down, lm_head) is a GEMM. Unlike
//! the SMG kernels this is **size-parametric** — M, N, K are runtime uniforms,
//! the workgroup is a fixed 16×16 tile, and bounds are checked so any shape
//! works (no compile-time constants, no LDS-bound ceiling).
//!
//! Shared-memory tiling: each 16×16 workgroup cooperatively stages a 16×16
//! tile of A and of B into workgroup memory, syncs, accumulates, advances K.
//! This is the standard cache-blocked matmul; on RDNA2 (wave32) a 16×16 tile
//! = 8 waves/workgroup, 2 KB shared per tile (well under 64 KB LDS).

use crate::gpu::GpuContext;
use wgpu::util::DeviceExt;

const TILE: u32 = 16;

const WGSL: &str = r#"
struct Dims { m: u32, n: u32, k: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>        a: array<f32>;
@group(0) @binding(1) var<storage, read>        b: array<f32>;
@group(0) @binding(2) var<storage, read_write>  c: array<f32>;
@group(0) @binding(3) var<uniform>              d: Dims;

var<workgroup> tileA: array<f32, 256>;  // 16*16
var<workgroup> tileB: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>,
        @builtin(workgroup_id)         wid: vec3<u32>) {
    let row = gid.y;          // 0..M
    let col = gid.x;          // 0..N
    let lr  = lid.y;          // 0..15
    let lc  = lid.x;          // 0..15

    var acc: f32 = 0.0;
    let n_tiles = (d.k + 15u) / 16u;

    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        // stage A[row, t*16 + lc]
        let a_col = t * 16u + lc;
        if (row < d.m && a_col < d.k) {
            tileA[lr * 16u + lc] = a[row * d.k + a_col];
        } else {
            tileA[lr * 16u + lc] = 0.0;
        }
        // stage B[t*16 + lr, col]
        let b_row = t * 16u + lr;
        if (b_row < d.k && col < d.n) {
            tileB[lr * 16u + lc] = b[b_row * d.n + col];
        } else {
            tileB[lr * 16u + lc] = 0.0;
        }
        workgroupBarrier();

        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            acc = acc + tileA[lr * 16u + i] * tileB[i * 16u + lc];
        }
        workgroupBarrier();
    }

    if (row < d.m && col < d.n) {
        c[row * d.n + col] = acc;
    }
}
"#;

/// GPU GEMM. `a` is M×K, `b` is K×N, returns C = A·B (M×N), row-major.
pub fn gemm(ctx: &GpuContext, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "a must be M*K");
    assert_eq!(b.len(), k * n, "b must be K*N");
    let dev = &ctx.device;
    let q = &ctx.queue;

    let a_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gemm.a"), contents: bytemuck::cast_slice(a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gemm.b"), contents: bytemuck::cast_slice(b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let c_size = (m * n * 4) as u64;
    let c_buf = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemm.c"), size: c_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let dims = [m as u32, n as u32, k as u32, 0u32];
    let dim_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gemm.dims"), contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let pipeline = ctx.pipeline("gemm", WGSL);
    let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gemm.bind"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: c_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dim_buf.as_entire_binding() },
        ],
    });

    let mut enc = dev.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        let gx = (n as u32).div_ceil(TILE);
        let gy = (m as u32).div_ceil(TILE);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    // readback
    let staging = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemm.staging"), size: c_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&c_buf, 0, &staging, 0, c_size);
    q.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    dev.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    out
}

/// CPU reference GEMM — ground truth for parity tests.
pub fn gemm_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += av * b[p * n + j];
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    #[test]
    fn gemm_parity_various_shapes() {
        let ctx = match GpuContext::new() { Ok(c) => c, Err(_) => return };
        let mut s = 12345u64;
        for &(m, k, n) in &[(16, 16, 16), (17, 33, 9), (64, 128, 32), (1, 256, 256), (100, 1, 100)] {
            let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
            let b: Vec<f32> = (0..k * n).map(|_| lcg(&mut s)).collect();
            let gpu = gemm(&ctx, &a, &b, m, k, n);
            let cpu = gemm_cpu(&a, &b, m, k, n);
            let max_err = gpu.iter().zip(&cpu).map(|(g, c)| (g - c).abs()).fold(0.0f32, f32::max);
            assert!(max_err < 1e-3, "shape {m}x{k}x{n} max_err {max_err}");
        }
    }
}
