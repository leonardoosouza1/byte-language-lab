// smg_norm.hip — Gradient norm (sum-of-squares) para gradient clipping.
//
// Port de `src/gpu/shaders/microcolumn_v3_norm.wgsl`.
// Alvo: RDNA2 gfx1031. Compilar com hipcc --genco --offload-arch=gfx1031 ...
//
// Dois entry points:
//   sumsq_main  — acumula sum(g[i]^2) em norm_sq_acc via LDS-reduce + CAS global
//   zero_acc    — zera norm_sq_acc[0]
//
// Kernarg layout (sumsq_main):
//   offset  0: ParamsNorm  (16 B, 4×u32)
//   offset 16: g           (8 B ptr, uint32_t* — gradientes como bit-pattern f32)
//   offset 24: norm_sq_acc (8 B ptr, uint32_t* — accumulator global)
//   explicit: 32 B + hidden args ≈ 88 B

#include <hip/hip_runtime.h>
#include <math.h>

struct ParamsNorm {
    uint32_t length;
    uint32_t _pad0, _pad1, _pad2;
};

__device__ __forceinline__ float u32_as_f32(uint32_t u) {
    float f; __builtin_memcpy(&f, &u, sizeof(float)); return f;
}
__device__ __forceinline__ uint32_t f32_as_u32(float f) {
    uint32_t u; __builtin_memcpy(&u, &f, sizeof(uint32_t)); return u;
}

__device__ __forceinline__ void atomic_add_f32(uint32_t* __restrict__ addr, float val) {
    if (val == 0.0f) return;
    uint32_t old = *addr, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, f32_as_u32(u32_as_f32(assumed) + val));
    } while (assumed != old);
}

// ---------------------------------------------------------------------------
// sumsq_main: sum(g[i]^2), LDS tree-reduce, então 1 CAS-add ao accumulator.
// ---------------------------------------------------------------------------
extern "C" __global__ void
__launch_bounds__(256)
sumsq_main(
    ParamsNorm               params,
    const uint32_t* __restrict__ g,            // grad buffer (bit-pattern f32)
    uint32_t* __restrict__       norm_sq_acc   // 1-element global accumulator
) {
    __shared__ float partial[256];

    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x * blockDim.x + tid;

    float local_sum = 0.0f;
    if (gid < params.length) {
        const float v = u32_as_f32(g[gid]);
        local_sum = v * v;
    }
    partial[tid] = local_sum;
    __syncthreads();

    // Redução binária em LDS: 256 → 128 → ... → 1.
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    // Thread 0 atomic-add ao accumulator global.
    if (tid == 0u) {
        const float val = partial[0];
        if (val != 0.0f) atomic_add_f32(norm_sq_acc, val);
    }
}

// ---------------------------------------------------------------------------
// zero_acc: zera o accumulator global.
// ---------------------------------------------------------------------------
extern "C" __global__ void
__launch_bounds__(1)
zero_acc(
    uint32_t* __restrict__ norm_sq_acc
) {
    *norm_sq_acc = 0u;
}
