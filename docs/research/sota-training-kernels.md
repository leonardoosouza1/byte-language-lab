# State of the Art in GPU Training Kernels for AMD RDNA2

**Target hardware:** AMD Radeon RX 6700 XT (gfx1031, RDNA2), 40 CUs, 96 MB Infinity Cache, 384 GB/s GDDR6, 13.21 TFLOPS FP32 boost. Same patterns generally apply to gfx1030 (RX 6800/6900) and other gfx103x parts.

**Scope.** This document surveys training-relevant GPU kernel patterns: forward passes, backward passes with gradient accumulation, reductions, atomics, and SGD updates. The emphasis is on patterns that survive on RDNA2 — a wave32 architecture with native packed dot products (`v_dot2_*`) but no Matrix/Tensor cores and only emulated bf16. We extract patterns from llama.cpp/ggml, CubeCL/Burn, PyTorch ROCm, Triton-on-AMD, rocBLAS/hipBLASLt, and several smaller research kernels, then synthesize a reusable cookbook (sections A–I).

**Audience.** Engineers writing custom HIP / WGSL / CubeCL kernels for training tiny-to-medium models on a single consumer RDNA2 card. The implicit baseline is Wave64 + naive global atomic accumulation; the ceiling is hand-tuned Wave32 with LDS staging, packed-dp4a, and a hipGraph-captured training step.

---

## 1. RDNA2 baseline: what the hardware actually wants

Before discussing kernels we have to be honest about what RDNA2 is:

- **Native wavefront is 32**, not 64. RDNA can run wave64 mode but it's emulated as 2× wave32 issue under the hood, and most cross-lane intrinsics (`__shfl_xor`, DPP) generate cleaner code at wave32. ROCm historically defaulted to wave64 for HIP on RDNA, which is wrong for compute kernels — every llama.cpp HIP perf PR for gfx10/11 starts with forcing `__AMDGCN_WAVEFRONT_SIZE=32`.
- **No matrix cores on RDNA2.** WMMA shows up first on RDNA3 (gfx1100). On gfx1030/1031 the best you can do for ML matmul is `v_dot4_i8_iu8` / `v_dot2_f32_f16` / `v_dot2c_f32_f16` (packed-dp4a-style) plus aggressive LDS tiling.
- **bf16 is essentially absent on RDNA2.** There is no native `v_dot2_bf16_bf16` on gfx1030; you have to fake it with FP16 + scaling tricks or just store BF16 and convert on load.
- **96 MB Infinity Cache** changes the bandwidth picture: peak DRAM is 384 GB/s, but for working sets that fit in IC the effective bandwidth is ~1.2–1.3 TB/s. Kernels that iterate over a small working set (e.g., a 256×768 weight matrix during BPTT through 32 steps) live in IC and look memory-bound at IC speed, not DRAM speed.
- **Atomic FP semantics.** RDNA2 has hardware `global_atomic_add_f32` and `ds_add_f32` (LDS), but on coarse-grained allocations only. Fine-grained host-mapped memory falls back to CAS. This is the source of the "f32 CAS-loop fallback" that everyone writes — it is the *portable* version, the hardware path is only available with `-munsafe-fp-atomics` or `__builtin_amdgcn_global_atomic_fadd_f32`.

Everything below is shaped by these four facts.

---

## 2. llama.cpp / ggml — the canonical HIP-portable inference codebase

The ggml CUDA backend (`ggml/src/ggml-cuda/`) is the most thoroughly hand-tuned HIP-portable kernel codebase in the open-source LLM world. It is an *inference* codebase, but the kernel patterns are directly relevant for training small models, especially the GEMV (matrix × vector) case which dominates per-token compute and shows up unchanged in BPTT inner loops.

### 2.1 MMVQ — quantized GEMV with per-architecture tuning

**File:** [`ggml/src/ggml-cuda/mmvq.cu`](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu)

MMVQ ("matrix-multiply vector quantized") is the kernel called every decode step for `Q4_K × FP16` etc. The pattern is:

1. One CUDA block produces `rows_per_cuda_block` output rows.
2. Each warp in the block streams a slab of K columns, computes a partial dot product per row using the per-quant `vec_dot_<type>_q8_1(...)` intrinsic.
3. Accumulator lives in registers; warps reduce via `warp_reduce_sum<warp_size>(...)`; warp-leaders write to LDS.
4. A final warp reduces the LDS partials and writes the row outputs.

The interesting bit for us is the per-arch tuning table:

```cpp
// ggml/src/ggml-cuda/mmvq.cu (simplified)
static constexpr __host__ __device__ int calc_rows_per_block(
    int ncols_dst, int table_id, bool small_k=false, int nwarps=1)
{
    if (table_id == MMVQ_PARAMETERS_GENERIC ||
        table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:  return small_k ? nwarps : 1;
            case 2: case 3: case 4: case 5:
            case 6: case 7: case 8: return 2;
            default: return 1;
        }
    }
    return 1;  // RDNA2 / RDNA3 path
}
```

Why does RDNA always return 1? Because on RDNA, register pressure dominates: a 32-wide wave with wider rows-per-block runs out of VGPRs and tanks occupancy. GCN had 64-wide waves and a much larger register file *per wave*, so it could afford `rows_per_cuda_block = 2`. The lesson: **RDNA2 wants narrower, more-occupant blocks; GCN wants wider, fewer.**

The reduction at the end is the textbook shuffle pattern:

```cpp
tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
```

Where `warp_reduce_sum<32>` expands to:

```cpp
template <int W>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int s = W/2; s > 0; s >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, s, W);
    }
    return x;
}
```

On RDNA2 with `warp_size=32` this compiles to 5 `ds_bpermute_b32` (or `v_permlane*`) instructions plus 5 adds — tight, but note that `ds_bpermute` goes through LDS, so the throughput limit is one cross-lane swap per 4 cycles. DPP (`v_add_f32 ... dpp ctrl:row_xmask`) is 1 cycle but HIP doesn't always emit it from `__shfl_xor`; aggressive inner loops sometimes hand-roll DPP intrinsics.

### 2.2 Wave32 specifics for HIP

The HIP build sets `__AMDGCN_WAVEFRONT_SIZE` based on `mwavefrontsize64`. ggml's HIP compatibility shim in `ggml/src/ggml-cuda/vendors/hip.h` redefines `WARP_SIZE` to 32 on RDNA. This is non-negotiable — every shuffle width, every LDS-pad calculation, every "rows per warp" constant has to switch with it. A bug here is silent (it still runs, just slower or wrong-on-edges).

The derivative project [`carlosfundora/llama.cpp-1-bit-turbo`](https://github.com/carlosfundora/llama.cpp-1-bit-turbo) is a useful reference: it's a HIP fork specifically optimized for gfx1030 with 1-bit quant and forced wave32 throughout. Worth reading their diffs for "what does *real* RDNA2 tuning look like" — they touch every kernel that had a warp-size constant.

---

## 3. CubeCL / Burn — the Rust-native compute language

[`tracel-ai/cubecl`](https://github.com/tracel-ai/cubecl) is a Rust eDSL that compiles to CUDA, ROCm/HIP, Metal, Vulkan and WGPU backends. [`tracel-ai/burn`](https://github.com/tracel-ai/burn) sits on top of it as the deep-learning framework. The relevant pieces for us:

- **`cubecl-hip`** (`crates/cubecl-hip`) — the HIP backend, currently aimed primarily at gfx1100/CDNA. RDNA2 (gfx1030) works but matrix multiply acceleration is gated on rocWMMA which doesn't exist there, so you fall back to a software tiled matmul.
- **`burn-autodiff`** — reverse-mode AD as a backend decorator. Crucially, autodiff is *not* a kernel concern in Burn: it builds a tape of ops, and gradient kernels are launched via the same `CubeRuntime` as forward kernels. The tape decides who reads what; the kernels just write to gradient buffers passed in as outputs.
- **Atomic add for gradient accumulation.** CubeCL exposes `AtomicAdd` as a first-class operation in the IR and emits the right backend op (`atomicAdd` / `OpAtomicFAddEXT` / `__atomic_fetch_add`). The 0.19 release notes explicitly call out "support for full atomic sum operations" — this is what enables backward passes that scatter into a shared gradient tensor.

The takeaway for handwritten kernels: CubeCL's design implicitly answers "how do you do gradient accumulation?" with **one big atomic-add buffer per parameter, and let the runtime sequence the writers**. There is no clever multi-buffer reduction, no distributed accumulator. The simplicity is the point — atomics on coarse-grained device memory are fast on modern AMD hardware (the f32 CAS fallback only triggers in pathological allocator paths).

If you're hand-writing a Burn-style training loop in HIP for RDNA2, replicate this:

1. Allocate one `grad` buffer per parameter, same shape as the param.
2. Forward kernel writes activations to a stash buffer.
3. Backward kernels each `atomic_add_f32` into `grad`.
4. SGD kernel reads `param`, `grad`, writes back `param`, optionally zeroes `grad`.

This is exactly the pattern in section E below.

---

## 4. PyTorch ROCm — the production atomic story

PyTorch's atomic abstraction lives in [`aten/src/ATen/cuda/Atomic.cuh`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/Atomic.cuh) and the kernel-side helper [`aten/src/ATen/native/cuda/KernelUtils.cuh`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/KernelUtils.cuh). The float case for ROCm is delegated:

```cpp
// aten/src/ATen/cuda/Atomic.cuh
inline __device__ float gpuAtomicAdd(float *address, float val) {
    return atomicAdd(address, val);
}

#if defined(USE_ROCM)
inline __device__ void gpuAtomicAddNoReturn(float *address, float val) {
    // gfx908 (CDNA1) is special: needs atomicAddNoRet for performance.
    if (__builtin_amdgcn_processor_is("gfx908"))
        return atomicAddNoRet(address, val);
    (void)unsafeAtomicAdd(address, val);
}
#endif
```

`unsafeAtomicAdd` is the magic word: it tells the HIP compiler to emit `global_atomic_add_f32` directly (the hardware RMW instruction), bypassing the CAS loop. On RDNA2 with coarse-grained device memory this is ~5–10× faster than the CAS path under contention. The "unsafe" in the name refers to the wrong-rounding-on-fine-grained-memory issue mentioned in section 1; for normal `hipMalloc` allocations it's safe.

For half-precision PyTorch uses `fastAtomicAdd` (in `KernelUtils.cuh`), which casts to `__half2*` and does a packed atomic add of two halves at once when alignment allows. This is *not* available for FP32 — there is no packed FP32 atomic.

### 4.1 The CAS-loop fallback (the canonical form)

This is the snippet you put in your headers and forget about. It is correct on every HIP target including weird memory:

```cpp
__device__ inline void atomic_add_f32_cas(float* p, float v) {
    unsigned int* up = reinterpret_cast<unsigned int*>(p);
    unsigned int old = __atomic_load_n(up, __ATOMIC_RELAXED);
    unsigned int assumed;
    do {
        assumed = old;
        float new_val = __uint_as_float(assumed) + v;
        old = atomicCAS(up, assumed, __float_as_uint(new_val));
    } while (assumed != old);
}
```

Variants worth knowing:

- **Fast-skip on zero.** `if (v == 0.0f) return;` at the top saves a CAS round when sparse gradients are common. Only worth it if you measure ≥10% zero rate.
- **Release-store on first attempt.** Some kernels precompute the new value once and only enter the loop on contention:
  ```cpp
  unsigned int old = __atomic_load_n(up, __ATOMIC_RELAXED);
  unsigned int desired = __float_as_uint(__uint_as_float(old) + v);
  if (atomicCAS(up, old, desired) == old) return;
  // contended path
  do { ... } while (...);
  ```
- **`unsafeAtomicAdd` shortcut.** For coarse-grained memory just write `unsafeAtomicAdd(p, v);` and the compiler emits the hardware op.

### 4.2 Backward op lowering

`torch.autograd.Function` ops in the ROCm path lower to `aten::*_backward` kernels in `aten/src/ATen/native/cuda/`. The pattern across them is uniform:

- `embedding_backward_kernel` and `index_select_backward_kernel` use `fastAtomicAdd` because keys repeat.
- `scatter_add_kernel` uses `gpuAtomicAddNoReturn`.
- Conv/linear backward go through rocBLAS GEMM (`rocblas_sgemm` / `hipblaslt_matmul`), no atomics.
- `accumulate_grad` itself is just a fused `t.add_(g)` — not atomic on its own; it's the per-op backward that uses atomics, then the accumulated grad is added cleanly.

Contention mitigation in PyTorch ROCm:

1. **Coalesce indices first.** `embedding_backward` has a "dense" path that sorts indices and segments them, so each segment gets one writer per slot — no contention.
2. **Use `__half2` packed atomics for FP16.** Halves the number of atomic ops for transformer embedding tables.
3. **Per-block partial reductions in LDS** before global atomic. This is the dominant pattern in `*_reduce` kernels — see `Reduce.cuh`.

---

## 5. Triton on AMD — current status (May 2026)

Upstream Triton ([`triton-lang/triton`](https://github.com/triton-lang/triton)) has had an AMD backend on the `triton-mlir` track for a couple of years. As of May 2026:

- **gfx1100 (RDNA3)** is the well-supported consumer target — WMMA lowering, async copy, decent matmul perf.
- **gfx1030 (RDNA2)** compiles but is in a "best-effort" tier. The MLIR pipeline (`TritonAMDGPUPipeline`) has known issues for several patterns — see [issue #24 on triton-windows](https://github.com/triton-lang/triton-windows/issues/24) for SageAttention v1 failing on gfx1030.
- **gfx1250 (RDNA3.5/4)** is where new development is happening — TDM, async copy, WMMA refresh.

The implication: **don't rely on Triton for production RDNA2 training kernels in 2026.** Use it for prototyping, but have a HIP fallback. The block-level abstractions (`tl.load`, `tl.dot`, `tl.atomic_add`) are still useful as a mental model — Triton's `tl.atomic_add` on AMD lowers to `unsafeAtomicAdd` for coarse memory, exactly what you'd write by hand.

---

## 6. rocBLAS / hipBLASLt — when to call vendor and when to write your own

Rule of thumb on RDNA2:

| Shape | Best path |
|---|---|
| Square or near-square M=N=K ≥ 512 | rocBLAS / hipBLASLt |
| Tall-skinny (M≫N or K≫M) | Custom kernel, almost always |
| GEMV (N=1) | Custom MMVQ-style kernel |
| Batched GEMM with B≥32 | rocBLAS strided-batched |
| Batched GEMM with B<8, small K | Custom persistent kernel |
| Quantized weights | Custom (rocBLAS doesn't help) |

[hipBLASLt's TensileLite tuning](https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html) lets you generate a custom kernel and inject it into the dispatcher. For production this is the right move when one shape is hot. For research/iteration, hand-writing a kernel is usually faster end-to-end — TensileLite tuning runs are O(hours) per shape.

The reasons custom kernels win on irregular shapes:

1. **rocBLAS partitions assume rectangular tile geometries.** Tall-skinny falls off the heuristic table.
2. **No persistent-kernel mode for tiny batches.** rocBLAS launches one block per output tile; if you only have 8 tiles, 32 of 40 CUs sit idle. A persistent kernel keeps all 40 CUs busy by streaming work.
3. **No fused epilogues for arbitrary ops.** rocBLAS knows `gemm + bias + relu`. If you want `gemm + softplus + ternary_quantize` you write your own.
4. **rocBLAS is FP32/FP16/BF16 only.** Q4_K, Q8_1, NF4, packed-ternary all need custom kernels.

For training a small model where you control every kernel, the calculus is even more lopsided: rocBLAS gives you 80–90% of peak on big GEMMs, but the inner loop of BPTT through a 64×128 hidden-state matrix runs entirely in IC and a hand-rolled persistent kernel gets the same throughput with zero launch overhead and no tile-residue dropouts.

---

## 7. Reusable patterns (the cookbook)

Each of these is the version that should ship in your `kernels/common.h` (or WGSL equivalent). Sources cited inline.

### A. Atomic FP32 add helper

The CAS-loop fallback (portable):

```cpp
__device__ inline void atomic_add_f32(float* p, float v) {
    if (v == 0.0f) return;  // optional fast-skip
    unsigned int* up = reinterpret_cast<unsigned int*>(p);
    unsigned int old = __atomic_load_n(up, __ATOMIC_RELAXED);
    unsigned int assumed;
    do {
        assumed = old;
        float new_val = __uint_as_float(assumed) + v;
        old = atomicCAS(up, assumed, __float_as_uint(new_val));
    } while (assumed != old);
}
```

Hardware-fast version on RDNA2 with coarse-grained memory:

```cpp
__device__ inline void atomic_add_f32_fast(float* p, float v) {
    // Compiles to global_atomic_add_f32 on gfx1030+
    (void)unsafeAtomicAdd(p, v);
}
```

Source: derived from PyTorch [`Atomic.cuh`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/Atomic.cuh), HIP [Atomic operations: Histogram tutorial](https://rocm.docs.amd.com/projects/HIP/en/develop/tutorial/programming-patterns/atomic_operations_histogram.html).

WGSL/wgpu equivalent (no FP32 atomic, must do CAS in u32 manually):

```wgsl
fn atomic_add_f32(p: ptr<storage, atomic<u32>, read_write>, v: f32) {
    var old = atomicLoad(p);
    loop {
        let new_val = bitcast<u32>(bitcast<f32>(old) + v);
        let res = atomicCompareExchangeWeak(p, old, new_val);
        if (res.exchanged) { break; }
        old = res.old_value;
    }
}
```

### B. Workgroup reduction in LDS (halving stride)

Classical pattern, padded for 32-bank LDS:

```cpp
__device__ inline float block_reduce_sum_lds(float v, float* lds /* size = blockDim.x */) {
    int tid = threadIdx.x;
    lds[tid] = v;
    __syncthreads();
#pragma unroll
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) lds[tid] += lds[tid + s];
        __syncthreads();
    }
    return lds[0];
}
```

When LDS atomics are available (RDNA2 has `ds_add_f32`):

```cpp
__device__ inline void block_atomic_add_lds(float* lds_acc, float v) {
    // Single instruction on RDNA2
    atomicAdd(lds_acc, v);  // compiles to ds_add_f32 for __shared__ ptr
}
```

Bank-conflict avoidance: on RDNA2 LDS has 32 banks of 4 bytes, so a stride-1 read pattern is conflict-free for FP32. If your `lds` array is FP32 with 32-thread reductions, you don't need padding. If you're reducing over wave64 with wave64 mode you need `lds[tid + tid/32]` padding.

### C. Wave32 cross-lane reduction with shfl_xor

```cpp
__device__ inline float warp_reduce_sum_w32(float v) {
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        v += __shfl_xor(v, s, 32);
    }
    return v;
}
```

This is what `ggml/src/ggml-cuda/common.cuh` defines as `warp_reduce_sum<32>`. For sums and products it's the right primitive. For min/max use `__shfl_xor` with the appropriate op. Note that on RDNA2 `__shfl_xor` with non-default `width` arg can compile to suboptimal `ds_bpermute` sequences — verify the assembly with `--save-temps` or rocgdb if you need the last 5%. DPP fallback for the hot path:

```cpp
// Hand-rolled DPP, fastest on RDNA2 for stride-{1,2,4,8,16}
__device__ inline float warp_reduce_sum_dpp(float v) {
    v += __builtin_amdgcn_mov_dpp(v, 0xb1, 0xf, 0xf, false);  // xmask:1
    v += __builtin_amdgcn_mov_dpp(v, 0xe4, 0xf, 0xf, false);  // xmask:2
    v += __builtin_amdgcn_mov_dpp(v, 0x4e, 0xf, 0xf, false);  // xmask:4
    v += __builtin_amdgcn_mov_dpp(v, 0x114, 0xf, 0xf, false); // xmask:8
    v += __builtin_amdgcn_mov_dpp(v, 0x118, 0xf, 0xf, false); // xmask:16
    return v;
}
```

(See AMD's [Optimizing for the RDNA Architecture deck](https://gpuopen.com/wp-content/uploads/slides/GPUOpen_Let%E2%80%99sBuild2020_Optimizing%20for%20the%20Radeon%20RDNA%20Architecture.pdf) for the DPP control byte encoding.)

### D. Two-stage global reduction

The pattern: per-WG partial in registers/LDS, atomic add to a per-WG slot in global, optional second kernel to sum slots. Avoids the single-accumulator hot spot.

```cpp
// Pass 1: each WG computes its partial, writes to per-WG slot
__global__ void reduce_pass1(const float* __restrict__ x, float* __restrict__ partials, int N) {
    int tid = threadIdx.x, bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;
    float v = (gid < N) ? x[gid] : 0.0f;
    v = warp_reduce_sum_w32(v);
    __shared__ float lds[32];
    if ((tid & 31) == 0) lds[tid >> 5] = v;
    __syncthreads();
    if (tid < 32) {
        v = (tid < blockDim.x / 32) ? lds[tid] : 0.0f;
        v = warp_reduce_sum_w32(v);
        if (tid == 0) partials[bid] = v;
    }
}

// Pass 2: single block reduces all partials
__global__ void reduce_pass2(const float* __restrict__ partials, float* __restrict__ out, int K) {
    /* same shape, K elements, write out[0] */
}
```

This beats single-accumulator atomic-add when K (number of WGs) > ~64. Below that, just do one global `atomic_add_f32` per WG — the contention is fine.

### E. Backprop-style gradient accumulation

The critical question: when N kernels write to the same gradient tensor, how do you avoid serializing them?

**Option 1 (simplest, what CubeCL/Burn do):** one gradient buffer, every backward kernel does `atomic_add_f32`. On RDNA2 with `unsafeAtomicAdd` and coarse-grained memory this is ~10–20 ns per atomic uncontended, ~50–200 ns under heavy contention. For most training shapes this is fine.

**Option 2 (when you have a hot accumulator):** per-block partial sums in LDS, then one atomic per block:

```cpp
__global__ void embedding_backward(const float* __restrict__ dout,
                                    const int* __restrict__ idx,
                                    float* __restrict__ dW, int D, int B) {
    extern __shared__ float lds_partial[];  // size D
    int tid = threadIdx.x;
    for (int d = tid; d < D; d += blockDim.x) lds_partial[d] = 0.0f;
    __syncthreads();

    // each thread handles one (batch, dim) pair
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        for (int d = tid; d < D; d += blockDim.x) {
            atomicAdd(&lds_partial[d], dout[b*D + d]);  // ds_add_f32
        }
    }
    __syncthreads();

    // flush LDS partial to global
    int row = idx[blockIdx.x];  // assumed unique per block; if not, sort first
    for (int d = tid; d < D; d += blockDim.x) {
        atomic_add_f32(&dW[row*D + d], lds_partial[d]);
    }
}
```

**Option 3 (when even option 2 contends):** per-replica gradient buffers. Kernel `i` writes to `grad_replica[i % R]`; a final reduction sums replicas. This is the same idea as a "deterministic" backward in PyTorch.

### F. SGD step kernel

The trivial form:

```cpp
__global__ void sgd_step(float* __restrict__ w, const float* __restrict__ g, float lr, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) w[i] -= lr * g[i];
}
```

Vectorized + fused with grad zeroing:

```cpp
__global__ void sgd_step_v4(float4* __restrict__ w, float4* __restrict__ g, float lr, int N4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;
    float4 wv = w[i];
    float4 gv = g[i];
    wv.x -= lr * gv.x; wv.y -= lr * gv.y; wv.z -= lr * gv.z; wv.w -= lr * gv.w;
    w[i] = wv;
    g[i] = make_float4(0,0,0,0);  // fused zeroing
}
```

For an RX 6700 XT this saturates DRAM at ~340 GB/s when N is large enough to cover launch overhead (N ≥ ~1M floats). For smaller weight tensors, fuse multiple SGD steps into a single kernel via a grid-stride loop and pass the param array as a buffer of pointers.

### G. Loss computation on GPU (no CPU readback)

Single-accumulator pattern for cross-entropy sum:

```cpp
__global__ void ce_loss_accumulate(const float* __restrict__ logits,
                                    const int* __restrict__ targets,
                                    float* __restrict__ loss_acc, int B, int V) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    // 1. find max for numerical stability
    float m = -INFINITY;
    for (int v = tid; v < V; v += blockDim.x) m = fmaxf(m, logits[b*V + v]);
    m = warp_reduce_max_w32(m);
    __shared__ float s_m, s_z;
    if (tid == 0) s_m = m;
    __syncthreads();
    // 2. partition function
    float z = 0.0f;
    for (int v = tid; v < V; v += blockDim.x) z += expf(logits[b*V + v] - s_m);
    z = warp_reduce_sum_w32(z);
    if (tid == 0) s_z = z;
    __syncthreads();
    // 3. -log p(target) into accumulator
    if (tid == 0) {
        int t = targets[b];
        float nll = logf(s_z) + s_m - logits[b*V + t];
        atomic_add_f32(loss_acc, nll);
    }
}
```

Per-batch loss buffer with offsetted writes is the alternative when you need per-sample loss: `loss_buf[b] = nll;` and reduce later. Both fit our needs; the accumulator version is one extra atomic per batch, no follow-up kernel.

### H. BPTT-specific patterns

Backprop-Through-Time has a specific shape: forward writes T snapshots of state, backward reads them in reverse. Three knobs:

1. **Stash everything (memory-heavy).** For a hidden state of size H and T timesteps, you store T×H floats per sample. Cheap on compute, expensive on memory. For our typical configs (H=128, T=64, B=32) that's 32×64×128×4 = 1 MB — fits comfortably in IC.
2. **Recompute (compute-heavy).** Store every K-th timestep, recompute the rest on the way back. Cuts memory by K×, costs ~2× forward compute.
3. **Single-WG-per-sample vs many-WG-per-sample.** Single-WG keeps all timestep state in registers/LDS — no global trips for state, only for inputs/outputs. Limits batch size per launch but maximizes locality. Many-WG-per-sample parallelizes across timesteps (only valid for non-recurrent layers) — relevant for parallel scan / chunked attention but not for vanilla RNN BPTT.

For our LSH/Cortex case (single-WG-per-sample, H≤256, T≤128), **stash everything in LDS** is the correct choice: 128×256×4 = 128 KB which exceeds RDNA2's 64 KB LDS per CU. So stash in registers + global, with a "spill to global when LDS is full" policy. This is exactly the pattern in Mamba's reference HIP kernels and in [llm.c](https://github.com/karpathy/llm.c)'s recurrent variants.

### I. Mixed-precision considerations on RDNA2

- **FP16 storage + FP32 compute** is the sweet spot. RDNA2 does `v_dot2_f32_f16` natively (1 cycle, 2 muls + 1 add). Use FP16 for activations and weights, FP32 for accumulators and gradients.
- **Loss scaling** is required: scale the loss by 2^k before backward, unscale gradients before the SGD step. Standard recipe; PyTorch's `GradScaler` is the reference.
- **bf16 on RDNA2: avoid.** No native dot product, no atomic. If you need bf16-like dynamic range, use FP16 with loss scaling or just stay in FP32.
- **Atomics on FP16.** `atomic_add_f16` does not exist on RDNA2 hardware. Use `__half2` packed atomic on 32-bit-aligned pairs (PyTorch's `fastAtomicAdd`), or do gradient accumulation in FP32 and cast to FP16 only for forward-pass storage.

---

## 8. Performance benchmarks on RX 6700 XT (gfx1031)

Numbers below are order-of-magnitude from public benchmarks and our own runs. Treat as priors, not guarantees — the IC behavior makes microbenchmarks deceptive.

| Op | Throughput | Notes |
|---|---|---|
| FP32 GEMM 1024×1024×1024 (rocBLAS) | ~9 TFLOPS | ~70% of peak 13.2 TFLOPS |
| FP32 GEMV 1×4096×4096 | ~280 GB/s | DRAM-bound, ~75% of peak BW |
| FP16 GEMM 1024×1024×1024 (rocBLAS) | ~12 TFLOPS | dot2_f32_f16 path |
| Q4_K GEMV (llama.cpp MMVQ) | ~30 GB/s effective | weights expand on the fly |
| `unsafeAtomicAdd` uncontended | ~10 ns | one CU writing |
| `unsafeAtomicAdd` 8 lanes contending | ~25 ns | within one wave |
| `unsafeAtomicAdd` 64 lanes contending | ~120 ns | 2 waves on same CU |
| `unsafeAtomicAdd` 256 lanes contending | ~500 ns | scaling roughly linear |
| CAS-loop atomic 256 contending | ~1.5–3 µs | retry storm |
| `ds_add_f32` (LDS) uncontended | 1 cycle | CU-local, ~0.5 ns |
| Empty kernel launch | ~5 µs | the lower bound for any kernel |
| HipGraph-replayed kernel | ~1 µs | 5× reduction, key for tight loops |
| BPTT step (H=128, T=64, B=32) | ~150 µs | mostly LDS+register, IC-friendly |

The single most important number: **CAS-loop atomic contention scales worse than linearly past ~32 contending lanes.** This is the reason every serious training kernel has LDS staging — you want to land at most one global atomic per WG per gradient slot.

---

## 9. Anti-patterns to avoid

1. **Per-element global atomic without LDS staging.** A backward kernel that does `atomic_add_f32(&dW[i], v)` from every thread will hit the contention cliff above. Always stage to LDS first, then one atomic per WG (or use the index-coalesced pattern from PyTorch).
2. **Naive reduction without shuffle.** A `for s = blockDim.x/2; ...; lds[tid] += lds[tid+s]` reduction is fine but ~3× slower than the warp-shuffle version. There's no excuse on RDNA2.
3. **Wave-divergent branching where wave-uniform would suffice.** RDNA2 is wave32 — a single divergent thread costs the whole wave a cycle. Restructure to make branches wave-uniform: e.g., process the same row across a whole wave instead of letting each thread pick its own.
4. **Wave64 mode for compute kernels.** Just don't. Set `__AMDGCN_WAVEFRONT_SIZE=32` and force `mwavefrontsize64=false` everywhere.
5. **Forgetting the `__threadfence()` between LDS reduction and global atomic.** RDNA2's memory model is relaxed; without the fence, the LDS write may not be visible to the thread issuing the global atomic on the same CU.
6. **Allocating gradient buffers with fine-grained host allocator and wondering why atomics are slow.** Use `hipMalloc`, not `hipHostMalloc`, for gradient buffers.
7. **Calling rocBLAS for a 64×64 matmul.** Launch overhead alone exceeds the math; write a one-block kernel.
8. **Synchronous `hipDeviceSynchronize()` per training step for loss readback.** Use a pinned-memory copy + a single async event, or better, keep the loss accumulator on-device and copy once per N steps.

---

## 10. Tooling for kernel iteration speed

1. **rocprof + rocprofv2.** Counters for VALU utilization, LDS bank conflicts, atomic-stall cycles. Check `SQ_INSTS_VALU`, `SQ_WAVES`, `TA_BUSY` for memory-bound diagnosis.
2. **rocgdb + `--save-temps`.** Inspect generated AMDGCN ASM — confirms whether `__shfl_xor` lowered to `ds_bpermute` (slow path) or DPP (fast path).
3. **hipBench-style microbenchmark harness.** Wrap each candidate kernel in:
   ```cpp
   auto t0 = std::chrono::steady_clock::now();
   for (int r = 0; r < REPS; ++r) my_kernel<<<g, b>>>(...);
   hipDeviceSynchronize();
   auto t1 = std::chrono::steady_clock::now();
   double us = (t1-t0).count() / 1000.0 / REPS;
   ```
   For latency: `REPS=1, blocks=1`. For throughput: `REPS=100, blocks=enough_to_fill_GPU`.
4. **HipGraph capture for repetitive launches.** Inner training loop with N kernels per step, captured once and replayed:
   ```cpp
   hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
   forward<<<...,stream>>>(...);
   backward<<<...,stream>>>(...);
   sgd<<<...,stream>>>(...);
   hipStreamEndCapture(stream, &graph);
   hipGraphInstantiate(&exec, graph, ...);
   for (step) hipGraphLaunch(exec, stream);
   ```
   Cuts launch overhead from ~5 µs/kernel to ~1 µs/kernel — the difference between "GPU 30% utilized" and "GPU 90% utilized" for small models.
5. **A/B harness with seeded RNG and bit-exact diff.** When porting between WGSL and HIP, fix the seed and diff outputs at FP32 ULP level. Atomic ordering differences will show up here — they're benign for training but you want to know they exist.
6. **Cortex-style replay logs.** If you hit a perf regression in CI, capture the kernel arguments + buffer hashes and replay locally. Saves a day of "works on my machine".

---

## 11. References

### Primary repos
- **llama.cpp / ggml** — [`ggml/src/ggml-cuda/mmvq.cu`](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu), [`ggml/src/ggml-cuda/common.cuh`](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh), [`ggml/src/ggml-cuda/vendors/hip.h`](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cuda/vendors/hip.h).
- **PyTorch ATen** — [`aten/src/ATen/cuda/Atomic.cuh`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/Atomic.cuh), [`aten/src/ATen/native/cuda/KernelUtils.cuh`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/KernelUtils.cuh), [PR #84981 (bf16 atomicAdd)](https://github.com/pytorch/pytorch/pull/84981), [issue #62695 (FP16 fastAtomicAdd)](https://github.com/pytorch/pytorch/issues/62695).
- **CubeCL / Burn** — [`tracel-ai/cubecl`](https://github.com/tracel-ai/cubecl), [`crates/cubecl-hip`](https://github.com/tracel-ai/cubecl/tree/main/crates/cubecl-hip), [`tracel-ai/burn`](https://github.com/tracel-ai/burn), [Burn 0.19.0 release notes](https://burn.dev/blog/release-0.19.0/).
- **Triton AMD** — [`triton-lang/triton`](https://github.com/triton-lang/triton), [issue #1073 (AMDGPU MLIR)](https://github.com/triton-lang/triton/issues/1073).
- **rocBLAS / hipBLASLt** — [hipBLASLt TensileLite tuning](https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html), [GEMM Kernel Optimization for AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/gemm_blog/README.html).
- **HIP samples** — [`ROCm/HIP-Examples`](https://github.com/ROCm/HIP-Examples), [`ROCm/rocm-examples`](https://github.com/ROCm/rocm-examples), [`KernelTuner/kernel_float`](https://github.com/KernelTuner/kernel_float).
- **RDNA2-specific HIP fork** — [`carlosfundora/llama.cpp-1-bit-turbo`](https://github.com/carlosfundora/llama.cpp-1-bit-turbo) — gfx1030 + wave32 throughout.

### AMD architecture docs
- [RDNA2 ISA Reference Guide (PDF)](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf).
- [Optimizing for the RDNA Architecture (GPUOpen Let's Build 2020)](https://gpuopen.com/wp-content/uploads/slides/GPUOpen_Let%E2%80%99sBuild2020_Optimizing%20for%20the%20Radeon%20RDNA%20Architecture.pdf).
- [LLVM AMDGPU GFX10 RDNA2 instructions](https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX1030.html).
- [HIP C++ language extensions (atomics, intrinsics)](https://rocm.docs.amd.com/projects/HIP/en/develop/reference/kernel_language.html).
- [HIP Atomic Operations Histogram tutorial](https://rocm.docs.amd.com/projects/HIP/en/develop/tutorial/programming-patterns/atomic_operations_histogram.html).
- [HIP graphs](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html), [HIP Graph API tutorial](https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/graph_api.html).

### Background / theory
- Mark Harris, [Optimizing Parallel Reduction in CUDA (PDF)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) — still the canonical reduction reference.
- [Faster Parallel Reductions on Kepler (NVIDIA blog)](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/) — shuffle-based reductions explained.
- [GPU MODE Lecture 9: Reductions](https://christianjmills.com/posts/cuda-mode-notes/lecture-009/) — modern walkthrough.
- [siboehm: How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) — applies cleanly to HIP.
- Interplay of Light, [Optimizing for the RDNA Architecture: presentation notes](https://interplayoflight.wordpress.com/2020/05/23/optimizing-for-the-rdna-architecture-presentation-notes/).
- Chester Lam (Chips and Cheese), [AMD's RDNA 2: Shooting For the Top](https://chipsandcheese.com/p/amds-rdna-2-shooting-for-the-top) — microarchitecture deep dive.

### Benchmarks
- [AMD Radeon RX 6700 XT specs](https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6700-xt.html) — 13.21 TFLOPS FP32, 384 GB/s, 96 MB IC.
- [Performance of llama.cpp on AMD ROCm (HIP) — discussion #15021](https://github.com/ggml-org/llama.cpp/discussions/15021).

---

## 12. Recommendations for byte-language-lab

Concrete next steps given our current stack (Rust + WGSL on wgpu, HIP path planned):

1. **Adopt the canonical CAS atomic add helper in `kernels/common.wgsl` now.** It's portable across wgpu and HIP-WGSL, and the perf gap to `unsafeAtomicAdd` is small at our contention levels.
2. **Force wave32 in any HIP kernel we compile.** Set `-mwavefrontsize64=false` and `__AMDGCN_WAVEFRONT_SIZE=32`, mirror llama.cpp's `vendors/hip.h` shim.
3. **Use the two-stage reduction (D) for any loss/metric accumulation.** Single-accumulator atomic-add is fine until B > 64 in our shapes.
4. **Stage gradients through LDS** before global atomic-add, especially for embedding-style backward where many threads target the same row.
5. **Capture the training step in a hipGraph** once we move to HIP. The launch-overhead reduction is ~5× and matters for our small per-step compute.
6. **Skip Triton on RDNA2.** Prototype in WGSL via CubeCL-style abstractions or plain compute shaders; if/when we move to RDNA3, reassess.
7. **Don't use bf16.** Stick with FP32 master / FP16 storage. Our IC is large enough that the storage savings of FP16 are not the binding constraint; correctness is.
8. **Build the microbenchmark harness early.** Section 10's hipBench pattern, exposed as a `cargo bench` target, will catch perf regressions that microbenchmarks of individual ops miss.

The single biggest perf lever for our workload remains **fitting the BPTT working set in Infinity Cache** — once that holds, everything else is second-order.
