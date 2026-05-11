# GPU Memory Model & Atomics on AMD RDNA2 (gfx1030/gfx1031)

A deep technical reference for backprop training kernel authors. Target audience: somebody who needs to implement a tight reduction (loss, gradient norm, sumsq, accumulators) and a multi-kernel pipeline on a Radeon RX 6700/6800/6900 (gfx1030) or 6700M / W6800 (gfx1031), and who needs to know exactly which instructions the compiler emits, what the hardware actually does, and where the silent performance cliffs live.

Status: 2026-05. ROCm 6.x, LLVM 18+, HIP runtime ≥ 6.0. Verified against `clang --target=amdgcn-amd-amdhsa -mcpu=gfx1030`.

---

## 1. HSA Memory Model — the formal substrate

The HIP / OpenCL / SYCL memory models all reduce to the **HSA Platform System Architecture Specification v1.2**, which is itself a refinement of "HRF-indirect with scope inclusion" (Heterogeneous-Race-Free, indirect inclusion variant). LLVM's AMDGPU backend implements *exactly* this model.

### 1.1 Memory scopes (HSA PRM §6, table 6-1)

A scope is a set of execution units that share an ordering relation. The HSA scopes, from narrowest to broadest:

| Scope | LLVM `syncscope("...")` | Set of agents synchronized |
|---|---|---|
| `WORKITEM` | `singlethread` | one lane |
| `WAVEFRONT` | `wavefront` | one wavefront (32 lanes on RDNA2 wave32, 64 on wave64) |
| `WORKGROUP` | `workgroup` | one CU's worth of waves cooperating in a workgroup (≤1024 lanes) |
| `AGENT` | `agent` | one GPU device (the entire RX 6800) |
| `SYSTEM` | `` (default) / `system` | every coherent agent: GPU + CPU + peer GPUs over xGMI |

The narrower the scope, the cheaper the synchronization — `WORKGROUP` only needs L0 (per-WGP) cache discipline and `s_waitcnt lgkmcnt(0)` for LDS, while `SYSTEM` needs a full L2 writeback plus PCIe/Infinity Fabric round trips.

**Backprop relevance.** Per-block partial sums only need `WORKGROUP`. Cross-block accumulators (the global gradient norm, the global loss) need `AGENT`. `SYSTEM` is needed only when the host CPU directly observes a GPU pointer through HMM / fine-grained shared memory; do not pay for `SYSTEM` by accident.

### 1.2 Memory orderings (HSA PRM §6.3)

| C++/HIP ordering | LLVM | Meaning |
|---|---|---|
| `memory_order_relaxed` | `monotonic` | Atomicity but no ordering. Cheapest. |
| `memory_order_acquire` | `acquire` | All later reads/writes happen-after this load. |
| `memory_order_release` | `release` | All earlier reads/writes happen-before this store. |
| `memory_order_acq_rel` | `acq_rel` | Both, on an RMW. |
| `memory_order_seq_cst` | `seq_cst` | Total order with all other `seq_cst` ops at the same scope. |

In HSA, fences come paired with signals: `hsa_signal_store_screlease(...)` publishes a value and acts as a release fence; `hsa_signal_wait_scacquire(...)` waits and acts as an acquire fence. Inside a kernel you don't call those; the compiler synthesizes the equivalent fences from your `__atomic_*` calls and from `__threadfence*()`.

### 1.3 The release/acquire idiom

The canonical producer-consumer pattern at any scope:

```
producer wave: write payload; release-store(flag, 1)
consumer wave: while (acquire-load(flag) == 0); read payload
```

The release-store + acquire-load pair guarantees that everything the producer wrote *before* the flag is visible to the consumer *after* it sees the flag. On AMDGPU this lowers to flush + waitcnt on the producer and waitcnt + invalidate on the consumer; see §2.

References: HSA PRM v1.2 §6 (Memory Model), §4 (AQL packets, signals).

---

## 2. AMDGPU Implementation — gfx1030 cache hierarchy and scope mapping

### 2.1 Cache hierarchy on RDNA2

Hardware layout (Navi 21 / Navi 22, gfx1030 / gfx1031):

```
            +----------------------------------------------+
            |           HBM/GDDR6 (memory controllers)     |
            +----------------------------------------------+
                                 ↑
            +----------------------------------------------+
            |   Infinity Cache (L3)  — 128 MB on Navi 21,  |
            |                           96 MB on Navi 22   |
            +----------------------------------------------+
                                 ↑
            +----------------------------------------------+
            |               L2  (~4 MB, GPU-wide)          |
            +----------------------------------------------+
                                 ↑       ↑
                          +------+       +------+
                          ↓                     ↓
              +--------------------+  +--------------------+
              |  L1 per Shader     |  |  L1 per Shader     |
              |  Array (128 KB)    |  |  Array (128 KB)    |
              +--------------------+  +--------------------+
                          ↑                     ↑
                   per-WGP L0  ...        per-WGP L0
                   (16-32 KB vector L0,
                    16 KB scalar L0)
```

Important: on RDNA2, **Infinity Cache is on the memory side of L2** — it's a victim/aggregator cache for the memory controllers and is transparent to the ISA. The compiler never targets it directly; the only thing under your control is L0/L1/L2 via the cache-control bits.

Scope → minimum cache discipline:

| Source scope | Acquire (consumer) | Release (producer) |
|---|---|---|
| WAVEFRONT | nothing — same L0 + in-order vmcnt | nothing |
| WORKGROUP | `buffer_gl0_inv` | `s_waitcnt vmcnt(0) vscnt(0)` |
| AGENT | `buffer_gl0_inv; buffer_gl1_inv` | `s_waitcnt vmcnt(0) vscnt(0)` (L2 is GPU-coherent) |
| SYSTEM | `buffer_gl0_inv; buffer_gl1_inv` plus L2 invalidate (`buffer_inv`) | full waitcnt + L2 writeback |

This is exactly what's in the LLVM `AMDGPUUsage.rst` table "Memory Model GFX10–GFX11 Code Sequences", which the AMDGPU backend uses as its lowering spec.

### 2.2 Per-instruction cache flags: GLC, DLC, SLC, SCC

RDNA `BUFFER_*`, `FLAT_*`, `GLOBAL_*` memory instructions carry these single-bit modifiers:

- **GLC — Globally Coherent.** On a *load*, GLC=1 forces the L0 to miss and reload from L1 (or further), giving the most recent value visible to the hit-under-miss machinery; on a *store*, GLC=1 means the store does not allocate or update L0 (write-bypass behavior). Used for atomics-with-return and for acquire-loads.
- **DLC — Device-Level Coherent (RDNA-specific, not present on GCN).** Forces bypass of the per-Shader-Array L1. DLC=1 reads bypass L1 and go to L2; this is needed when the data was written by another Shader Array (i.e. a different workgroup that happened to land on the *other* SA). Set together with GLC for cross-workgroup acquire.
- **SLC — System-Level Coherent.** Marks the access as non-cacheable in L2, so it lands in the system fabric / Infinity Cache fabric and is visible to peer agents (CPU through PCIe ATS, peer GPU through xGMI). Used for SYSTEM-scope loads/stores.
- **SCC — Scalar Cache Coherent (sometimes called "system coherent" in S-mem context).** Applies to scalar `s_load_*` / `s_buffer_load_*`. Tells the scalar L0 to bypass.

Lowering rules of thumb on gfx1030 (from `AMDGPUUsage`):

- `load atomic acquire WORKGROUP` → `global_load_dword v, ... glc; s_waitcnt vmcnt(0); buffer_gl0_inv`
- `load atomic acquire AGENT` → `global_load_dword v, ... glc dlc; s_waitcnt vmcnt(0); buffer_gl0_inv; buffer_gl1_inv`
- `store atomic release AGENT` → `s_waitcnt vmcnt(0) vscnt(0); global_store_dword ...`
- `atomicrmw acq_rel AGENT` → `s_waitcnt vmcnt(0) vscnt(0); global_atomic_*  ... glc; s_waitcnt vmcnt(0); buffer_gl0_inv; buffer_gl1_inv`
- `fence acquire WORKGROUP` → `s_waitcnt lgkmcnt(0)` (if LDS-only) else `s_waitcnt vmcnt(0); buffer_gl0_inv`

### 2.3 The s_waitcnt counters

`s_waitcnt` is the memory ordering primitive. It blocks the wave until the named counters drop to ≤ a constant.

| Counter | Tracks |
|---|---|
| `vmcnt` | Outstanding vector memory **loads** (returns are in-order per wave) |
| `vscnt` | Outstanding vector memory **stores** (RDNA-only counter; GCN had only vmcnt) |
| `lgkmcnt` | LDS, GDS, scalar memory, kernarg loads/stores combined |
| `expcnt` | Exports (graphics; ignore in compute) |

`vmcnt(0) vscnt(0)` is "all my pending memory traffic has retired". `lgkmcnt(0)` is "all my LDS/scalar traffic has retired". The compiler inserts these at every release point and at the boundaries of an acquire sequence.

References: AMDGPUUsage §"Memory Model GFX10-GFX11", RDNA2 ISA Reference §3 (data flow, cache).

---

## 3. C++/HIP Atomic API → ISA mapping on gfx1030

### 3.1 The HIP frontend

HIP exposes three layers of atomics:

1. **CUDA-compatible** `atomicAdd`, `atomicCAS`, `atomicMin`, etc. — fixed scope (effectively AGENT) and `relaxed` ordering.
2. **C++11 std-style** `__hip_atomic_load`, `__hip_atomic_store`, `__hip_atomic_compare_exchange_strong`, `__hip_atomic_fetch_add` taking `(ptr, value, memory_order, scope)`. This is the modern API; use it.
3. **Builtins** `__atomic_load_n`, `__atomic_compare_exchange_n` from libstdc++ / clang. These work but always pick SYSTEM scope — too expensive.

The HIP scope enum:

```cpp
__HIP_MEMORY_SCOPE_SINGLETHREAD = 1,  // workitem
__HIP_MEMORY_SCOPE_WAVEFRONT    = 2,
__HIP_MEMORY_SCOPE_WORKGROUP    = 3,
__HIP_MEMORY_SCOPE_AGENT        = 4,
__HIP_MEMORY_SCOPE_SYSTEM       = 5,
```

### 3.2 Integer atomics — fast path

`atomicAdd(int*, int)` lowers in one instruction:

```
global_atomic_add v[returnval], v[addr_lo:addr_hi], v[increment], off
```

Native, single-cycle issue per active lane (with bank-level serialization at the L2 atomic ALU when many lanes hit the same line). `atomicCAS(unsigned int*, ...)` → `global_atomic_cmpswap_b32`. `atomicMin / atomicMax / atomicAnd / atomicOr / atomicXor` are all native b32/b64.

### 3.3 Float atomics — the cliff

This is where RDNA2 hurts. **There is no `global_atomic_add_f32` on gfx1030/gfx1031.** It exists on:

- gfx908 (CDNA1, MI100) — yes, with `-munsafe-fp-atomics`
- gfx90a (CDNA2, MI210/250) — yes
- gfx940/941/942 (CDNA3) — yes
- gfx11xx (RDNA3) — yes, finally added to global memory
- **gfx1030/gfx1031/gfx1032 (RDNA2) — NO**

Therefore `atomicAdd(float*, float)` on RDNA2 lowers to a CAS loop. The compiler does this for you, but the perf hit is real: instead of 1 instruction it's a small loop (5–8 instructions per iteration) that retries on contention. Under heavy contention every retry doubles the cost.

`-munsafe-fp-atomics` does **not** rescue you on gfx1030 — there is no instruction to fall back to. It only helps on gfx908+ where a buffer/global atomic FP add exists but is technically not IEEE-compliant for denormals/NaN.

LDS is different: **`ds_add_f32` is native on RDNA2.** The DS unit has had a float-add ALU since GCN5/Vega. So intra-workgroup f32 reductions through `__shared__` are fast.

### 3.4 The CAS loop in IR and ISA

HIP source:

```cpp
__device__ inline float atomic_add_f32_relaxed(float* addr, float val) {
    unsigned int* p   = reinterpret_cast<unsigned int*>(addr);
    unsigned int  old = __atomic_load_n(p, __ATOMIC_RELAXED);
    unsigned int  assumed;
    do {
        assumed = old;
        float    new_f = __uint_as_float(assumed) + val;
        unsigned new_u = __float_as_uint(new_f);
        old = atomicCAS(p, assumed, new_u);   // global_atomic_cmpswap_b32
    } while (assumed != old);
    return __uint_as_float(old);
}
```

Compiled with `clang -O3 --target=amdgcn-amd-amdhsa -mcpu=gfx1030`, the inner loop in `llvm-objdump -d` (gfx1030, abridged):

```
.LBB_loop:
    v_add_f32_e32   v3, v2, v0          ; new_f = old_f + val
    v_mov_b32_e32   v4, v2              ; assumed = old (kept in v2)
    global_atomic_cmpswap_b32 v3, v[5:6], v[3:4], off glc
    s_waitcnt       vmcnt(0)
    v_cmp_eq_u32_e32 vcc_lo, v3, v2     ; assumed == old?
    s_or_b32        s0, vcc_lo, s0
    v_mov_b32_e32   v2, v3              ; old = returned value
    s_andn2_b32     exec_lo, exec_lo, s0 ; mask off lanes that succeeded
    s_cbranch_execnz .LBB_loop
    s_or_b32        exec_lo, exec_lo, s0 ; restore EXEC
```

Notice three things:

1. The atomic carries **`glc`** because it returns a value (atomics-with-return must be GLC).
2. The loop uses **EXEC masking**: lanes that succeed drop out of EXEC, the loop iterates only for losers. This is the RDNA wave-divergence pattern; it is correct but costs lockstep cycles.
3. There is a **`s_waitcnt vmcnt(0)`** on every iteration because the compiler must observe the returned value before comparing.

### 3.5 Wavefront contention behavior

When all 32 lanes of a wave atomic-add the same address:

- Hardware does **not** coalesce them at the SIMD. Each lane issues its own atomic into the same cache line.
- The L2 atomic ALU serializes them per cache line. Throughput becomes ≈ 1 atomic per L2 ALU cycle for that line.
- For the CAS-loop f32 case: lane 0 wins on iteration 0 with probability 1/32, the other 31 retry. On iteration 1, one of 31 wins, etc. Expected total iterations across the wave is ≈ ln(32) ≈ 3.5× the un-contended cost — but the wall time is dominated by serialization at L2.
- Across a workgroup of 4 waves, expected total iterations ≈ ln(128) ≈ 4.85.

Empirical numbers from public benchmarks (e.g. fsword73's HIP perf repo, llama.cpp ggml-rocm logs): a fully-contended `atomicAdd(float*, ...)` to a single global address on gfx1030 costs **~50–200 ns per workgroup**, vs ~5 ns for the same `atomicAdd` on a per-lane address (no contention) and ~2 ns for the integer native path.

### 3.6 Mitigations (in order of effectiveness)

1. **Per-lane local accumulation** + wave reduction + one atomic per wave. Reduces global atomic traffic by 32×.
2. **Per-workgroup LDS accumulation** with `ds_add_f32` (native!) + one atomic per workgroup. Reduces by another 8–32×.
3. **Per-workgroup slot in a global scratch array**, then a second small kernel reduces. Avoids global atomics entirely. Best for gradient-norm-type reductions where the second pass is cheap.

A typical training kernel that needs one global f32 accumulator: do steps 1+2; live with one CAS-loop atomic per workgroup. For 256 workgroups that is 256 CAS attempts on one cache line — about 5–10 µs total, fine.

---

## 4. Reduction patterns

### 4.1 Wave-level reduction (wave32)

On RDNA2 with `-mwavefrontsize32` (HIP default for gfx10+), `__shfl_xor` lowers to `ds_bpermute_b32` or `ds_swizzle_b32` depending on the lane pattern.

```cpp
__device__ __forceinline__ float wave_reduce_sum_f32(float v) {
    // wave32: 5 butterfly steps
    v += __shfl_xor(v, 1);
    v += __shfl_xor(v, 2);
    v += __shfl_xor(v, 4);
    v += __shfl_xor(v, 8);
    v += __shfl_xor(v, 16);
    return v;  // every lane holds the wave sum
}
```

The ISA for one of those `__shfl_xor`s (gfx1030):

```
v_xor_b32_e32     v2, 0x10, v1        ; src lane id = lane ^ 16
ds_bpermute_b32   v3, v2, v0          ; v3 = v0 from src lane
s_waitcnt         lgkmcnt(0)
v_add_f32_e32     v0, v0, v3
```

Five rounds × 4 instructions = 20 instructions for a full wave32 reduction. About 20 cycles per SIMD32, fully internal to the WGP — no L1, no L2.

If you write `__shfl_xor_sync` (CUDA-style with a mask), HIP accepts it but the mask is ignored on AMD; the wave is the implicit synchronization domain.

DPP-mode (`v_add_f32_dpp` with row_xmask / wave_shr modifiers) is even faster (no DS unit), but requires inline assembly or `__builtin_amdgcn_mov_dpp`. For most training kernels the bpermute version is fine.

### 4.2 Workgroup-level reduction in LDS

Pattern: each wave does its wave reduction → wave 0's value goes into LDS slot 0, wave 1's into slot 1, etc. → wave 0 reduces the slots.

```cpp
__device__ float workgroup_reduce_sum_f32(float v) {
    __shared__ float buf[32];  // up to 32 waves per WG
    const int lane = __lane_id();             // 0..31
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
    v = wave_reduce_sum_f32(v);
    if (lane == 0) buf[wave] = v;
    __syncthreads();
    int n_waves = (blockDim.x + 31) / 32;
    if (wave == 0) {
        v = (lane < n_waves) ? buf[lane] : 0.0f;
        v = wave_reduce_sum_f32(v);
    }
    return v;  // lane 0 of wave 0 holds it
}
```

The store `buf[wave] = v` lowers to `ds_write_b32`. The reload is `ds_read_b32`. The `__syncthreads()` is `s_waitcnt lgkmcnt(0); s_barrier`. No global memory. Latency is on the order of 30–60 cycles for a 256-thread workgroup.

If you don't need the partial sum back in a register but want a running accumulator, replace the store with `atomicAdd(&buf[0], v)` — that lowers to **`ds_add_f32`** (native!) and is a single-instruction LDS atomic. Useful for sparse reductions where lane participation is irregular.

### 4.3 Cross-workgroup reduction

Two strategies:

**A. Single global atomic.**

```cpp
__global__ void reduce_to_scalar(const float* in, float* out, int n) {
    float v = (idx < n) ? in[idx] : 0.0f;
    v = workgroup_reduce_sum_f32(v);
    if (threadIdx.x == 0) atomicAdd(out, v);   // CAS loop on gfx1030
}
```

Cost: 1 CAS-loop atomic per workgroup. For ≤1024 workgroups, fine.

**B. Two-pass.**

```cpp
__global__ void reduce_pass1(const float* in, float* partials, int n) {
    float v = (idx < n) ? in[idx] : 0.0f;
    v = workgroup_reduce_sum_f32(v);
    if (threadIdx.x == 0) partials[blockIdx.x] = v;
}
__global__ void reduce_pass2(const float* partials, float* out, int n_partials) {
    // launched as one workgroup
    float v = (idx < n_partials) ? partials[idx] : 0.0f;
    // serial loop if n_partials > blockDim.x
    v = workgroup_reduce_sum_f32(v);
    if (threadIdx.x == 0) *out = v;
}
```

Cost: zero global atomics. Two kernel launches, but the second is ~5 µs.

**Rule of thumb.** Below ~1024 workgroups: strategy A. Above: strategy B. Always B for gradient-norm reductions across millions of params.

---

## 5. Producer-Consumer Between Kernels — AQL queue semantics

This is where most "phantom bug" stories come from. The rule:

> Two kernels enqueued **on the same HSA queue, in order, with default packet header bits**, see each other's writes in agent scope. You do not need explicit fences in the kernel.

The mechanism: each AQL Kernel Dispatch packet has an `acquire_fence_scope` and a `release_fence_scope` field in its 16-bit header. The HIP runtime sets both to `HSA_FENCE_SCOPE_AGENT` by default.

- Before kernel B starts, the packet processor issues an **acquire fence at agent scope**: all caches (L0, L1) are invalidated for that agent. The dispatch sees memory as it was committed by anyone in the agent.
- After kernel A finishes, the packet processor issues a **release fence at agent scope**: L0/L1 are flushed to L2. (L2 is GPU-coherent on RDNA2 so it's already global within the agent.)

Therefore: kernel A writes a tensor; kernel B on the same queue reads it. No explicit synchronization needed.

Caveats:

- **Different queues.** No automatic ordering. You must either (a) have the runtime insert a barrier-AND packet between them, (b) use `hipStreamSynchronize`/`hipEventRecord` to serialize, or (c) add an explicit `__threadfence_system()` and a host-visible flag.
- **Same queue but `barrier=0` and you packed the queue manually.** Don't. Always use HIP/HSA APIs.
- **Peer-to-peer GPU traffic.** Needs `system` scope on the AQL fence and `SLC=1` on the writes; expect Infinity Fabric latency.

References: HSA PRM v1.2 §4.5 (AQL packet header), §6.6 (fence scopes); ROCR-Runtime `amd_aql_queue.cpp`.

---

## 6. Buffer vs Flat vs Global Addressing

RDNA2 has three families of memory instructions:

| Family | Addressing | Where used |
|---|---|---|
| `BUFFER_*` | (V#, soffset, voffset) — needs a 4-dword V# resource descriptor in SGPRs | Graphics, structured-buffer access. HIP rarely emits these except for kernarg loads via `s_buffer_load`. |
| `FLAT_*` | 64-bit virtual address — hardware decides if it's global / scratch / LDS by aperture | Pointers whose address space the compiler cannot prove statically. **Slowest** because of aperture check. |
| `GLOBAL_*` | 64-bit GVA, but the compiler has proven it points to global | The fast path for `__global__` pointers. |

The compiler emits `global_*` whenever it can prove the address space (via `__restrict__`, address-space attributes, or alias analysis). It falls back to `flat_*` for generic pointers and for pointers passed through `__shared__` casts or `void*`. **Audit your kernels with `llvm-objdump -d`**: any unexpected `flat_*` is a red flag — usually fixed by adding `__restrict__` to the kernel signature.

`s_load_dword` (scalar load) fires when the compiler can prove the address is **uniform across the wave** and the value is **read-only for the kernel's lifetime**. Kernarg blocks always go through scalar loads. Constant tables also. Anything that becomes lane-divergent goes vector. A sign of trouble: if you index a small const table by `threadIdx.x`, you'll get vector loads — manually copy to LDS at kernel start.

---

## 7. Memory aperture, hidden args, address translation

The GPU virtual address space on RDNA2 is 48 bits. It is partitioned into windows:

- **Global aperture** — backed by VRAM, system memory (through ATS), or peer GPU memory.
- **Private (scratch) aperture** — per-wave spill space. Hidden argument `private_segment_buffer` and base in `s_private_base`.
- **Shared (LDS) aperture** — per-workgroup. Hidden arg `group_segment_size`, base in `s_shared_base`.
- **Kernarg** — read-only argument block, base passed in `s[0:1]`.
- **Constant** — folded into either kernarg loads or read-only global.

The hidden arguments include `private_base` and `shared_base` precisely so that `flat_*` instructions can compare a 64-bit pointer against these bases and route the access to LDS / scratch / global. If the compiler can't prove the aperture, `flat_*` does that runtime check — costing ~10 cycles vs ~2 for `global_*`.

**ATS (Address Translation Services).** When fine-grained shared memory is enabled (`hipMallocManaged` with HMM), GPU page faults can route into the CPU page tables through PCIe ATS. This is convenient but adds milliseconds of latency per fault. Disable it for training: pin everything in VRAM.

**Large pages.** On Linux, transparent hugepages don't help GPU TLBs directly, but `hipMalloc` allocates from a pool that uses 2 MB GPU pages by default on ROCm 5.7+. This matters when the model weights are larger than the GPU TLB reach (~hundreds of MB on RDNA2).

---

## 8. `__threadfence` semantics in HIP

| Builtin | LLVM | gfx1030 lowering |
|---|---|---|
| `__threadfence_block()` | `fence syncscope("workgroup") seq_cst` | `s_waitcnt vmcnt(0) vscnt(0) lgkmcnt(0); buffer_gl0_inv` |
| `__threadfence()` | `fence syncscope("agent") seq_cst` | `s_waitcnt vmcnt(0) vscnt(0) lgkmcnt(0); buffer_gl0_inv; buffer_gl1_inv` |
| `__threadfence_system()` | `fence syncscope("") seq_cst` | full waitcnt + L2 writeback (`buffer_wbinvl1_vol` / `buffer_wbl2`) + `buffer_gl0_inv; buffer_gl1_inv; buffer_inv` |

You almost never need `__threadfence_system()`. You sometimes need `__threadfence()` when implementing a hand-rolled cross-workgroup spinlock or a one-shot barrier (the "last workgroup wins" idiom). You rarely need `__threadfence_block()` because `__syncthreads()` already includes a workgroup-scope acq_rel fence.

---

## 9. Concrete code snippets — drop-in helpers

### 9.1 Float atomic add (CAS loop)

```cpp
__device__ __forceinline__
float atomic_add_f32(float* addr, float val) {
    unsigned int* p = reinterpret_cast<unsigned int*>(addr);
    unsigned int  old = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (true) {
        float    new_f = __uint_as_float(old) + val;
        unsigned new_u = __float_as_uint(new_f);
        unsigned prev  = atomicCAS(p, old, new_u);
        if (prev == old) return __uint_as_float(prev);
        old = prev;
    }
}
```

Use this everywhere you'd write `atomicAdd(float*, float)`. They lower identically; this version makes the cost visible.

### 9.2 Wave32 sum reduction

```cpp
__device__ __forceinline__
float wave_reduce_sum_f32(float v) {
    v += __shfl_xor(v, 1);
    v += __shfl_xor(v, 2);
    v += __shfl_xor(v, 4);
    v += __shfl_xor(v, 8);
    v += __shfl_xor(v, 16);
    return v;
}
```

### 9.3 Workgroup sum reduction with native LDS atomic

```cpp
__device__ float workgroup_reduce_sum_f32_lds(float v) {
    __shared__ float scratch;
    if (threadIdx.x == 0) scratch = 0.0f;
    __syncthreads();
    v = wave_reduce_sum_f32(v);            // collapse wave first
    if ((threadIdx.x & 31) == 0)
        atomicAdd(&scratch, v);            // ds_add_f32 — native on RDNA2
    __syncthreads();
    return scratch;
}
```

`atomicAdd` on `__shared__` lowers to `ds_add_f32`. Single instruction, ~10-cycle latency. Much cheaper than the buf[N_WAVES] tree when you only need the reduced scalar.

### 9.4 Two-stage global reduction

```cpp
__global__ void reduce_pass1(const float* __restrict__ in,
                             float* __restrict__ partials,
                             int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (gid < n) ? in[gid] : 0.0f;
    v = workgroup_reduce_sum_f32_lds(v);
    if (threadIdx.x == 0) partials[blockIdx.x] = v;
}

__global__ void reduce_pass2(const float* __restrict__ partials,
                             float* __restrict__ out,
                             int n_partials) {
    // launch with one workgroup of 1024 threads
    float v = 0.0f;
    for (int i = threadIdx.x; i < n_partials; i += blockDim.x)
        v += partials[i];
    v = workgroup_reduce_sum_f32_lds(v);
    if (threadIdx.x == 0) *out = v;
}
```

Zero global atomics. Two launches. Use for large reductions (>1024 workgroups in pass 1).

---

## 10. Pitfalls

1. **f32 atomic on RDNA2 is a CAS loop. Always.** Your code "works" on a CDNA test machine and falls off a cliff on Navi 21. Microbenchmark every f32 atomic before committing.

2. **AQL release fence is agent-scope only.** Cross-process or peer-GPU IPC needs an explicit `__threadfence_system()` plus the right HIP IPC handle. Don't trust the queue alone.

3. **`volatile` does not insert fences in HIP.** It only suppresses register caching / re-ordering at the LLVM IR level. To get a fence you must call `__threadfence*()` or use an `__atomic_*` op with the right ordering. `volatile float*` to communicate between waves is a silent bug.

4. **Wave divergence inside a CAS loop.** The loop above masks out winning lanes via EXEC. If the body of the loop has *another* divergent control flow (e.g. an `if (val > 0) atomicAdd(...)` where some lanes don't even want to add), make sure the CAS is only entered by participating lanes — otherwise idle lanes still issue cmpswap with random garbage. The standard pattern (`if (cond) atomic_add_f32(...)`) is correct because EXEC already masks the cmpswap; just verify the disassembly the first time.

5. **Wave64 mode.** If you (mis)compile with `-mwavefrontsize64`, your `__shfl_xor` and your "every wave does one atomic" assumption break — wave is now 64 lanes, contention doubles, and L0 lookup patterns change. RDNA2 native is wave32; only force wave64 with measurement.

6. **Forgetting `glc` on a hand-written acquire load.** If you write inline asm `global_load_dword` without `glc`, the L0 may give you a stale value. Compiler-emitted `__hip_atomic_load(..., __ATOMIC_ACQUIRE, scope)` always sets the right bits; do not bypass it.

7. **L1 cross-workgroup invisibility.** Two workgroups on different Shader Arrays do not share L1. A non-atomic write by one workgroup is not visible to the other until L1 is invalidated (i.e., until an acquire load with `dlc`, or a `buffer_gl1_inv` in a fence, or the kernel ends). Hand-rolled spinlocks across workgroups must use `agent`-scope atomics.

8. **`__shared__` zero-init is not free.** LDS is uninitialized at workgroup start (RDNA2 does not zero LDS like some older ISAs). Either initialize explicitly (one thread per slot + `__syncthreads()`) or rely on every slot being written before read.

9. **`atomicAdd(float*)` to `__shared__` is fast; to `__global__` is slow.** A common refactoring bug: moving an LDS-resident accumulator to global "for convenience" silently 10×'s your reduction time on RDNA2.

10. **Counting on `seq_cst`.** All `seq_cst` stores get a heavyweight fence. If you wrote `__hip_atomic_store(..., __ATOMIC_SEQ_CST, AGENT)` because that's what you'd write on x86, you paid for global ordering you don't need. Use `release` for producer, `acquire` for consumer.

---

## 11. Backprop training kernel checklist

For a typical layer-fused backward kernel on RDNA2, the memory-model checklist:

- [ ] All inputs `__restrict__`'d so loads emit `global_*` not `flat_*`.
- [ ] Weight gradients accumulated **per-thread → per-wave (`__shfl_xor`) → per-workgroup (LDS, `ds_add_f32`)** before any global atomic.
- [ ] Single `atomic_add_f32` per workgroup per output address. No naïve `atomicAdd(grad_ptr + i, x)` in the inner loop.
- [ ] Loss / norm reductions use the two-pass kernel pattern when ≥1024 workgroups.
- [ ] Forward and backward kernels enqueued on the same HIP stream → AQL agent fence handles visibility, no `__threadfence` in the kernel.
- [ ] `hipMalloc` for everything; no `hipMallocManaged` unless you've measured ATS overhead.
- [ ] Disassembly reviewed: no `flat_*`, no surprise CAS loops on int paths, `s_waitcnt` counts look reasonable (not `vmcnt(0)` after every load).

---

## References

- LLVM AMDGPUUsage — Memory Model GFX10–GFX11: https://llvm.org/docs/AMDGPUUsage.html
- LLVM AMDGPU GFX1030 instruction reference: https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX1030.html
- LLVM `s_waitcnt` / counter docs: https://llvm.org/docs/AMDGPU/gfx9_waitcnt.html
- AMD "RDNA 2" Instruction Set Architecture Reference Guide (Nov 2020): https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf
- HSA Platform System Architecture Specification v1.2 §6 (memory model), §4 (AQL): http://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf
- HIP C++ Language Extensions (atomics, scopes, builtins): https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html
- HIP atomic histogram tutorial: https://rocm.docs.amd.com/projects/HIP/en/develop/tutorial/programming-patterns/atomic_operations_histogram.html
- ROCR-Runtime `amd_aql_queue.cpp` (AQL header bit handling): https://github.com/ROCm/ROCR-Runtime
- AMD GPUOpen — GCN/RDNA Cross-Lane Operations: https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/
- ROCm AMDGCN ISA blog (reading objdump output): https://rocm.blogs.amd.com/software-tools-optimization/amdgcn-isa/README.html
- HIP-Performance-Optimization-on-VEGA64 — `atomic-float-add-hip.cpp` (canonical CAS-loop reference): https://github.com/fsword73/HIP-Performance-Optmization-on-VEGA64
- ROCm hipamd issue #19 — `atomicAddNoRet` status (CAS-vs-native float atomic discussion): https://github.com/ROCm/hipamd/issues/19
- "Examining AMD's RDNA 4 Changes in LLVM" — Chester Lam, Chips and Cheese (context on what RDNA2 lacks): https://chipsandcheese.com/p/examining-amds-rdna-4-changes-in-llvm
- llama.cpp ggml-rocm reduction kernels (real-world reference): https://github.com/ggerganov/llama.cpp/tree/master/ggml/src/ggml-cuda
