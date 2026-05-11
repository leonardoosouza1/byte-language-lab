# AMD RDNA 2 Microarchitecture — Deep Reference for HIP Compute (gfx1030/gfx1031)

> Target hardware: **Radeon RX 6700 XT / 6750 XT** (Navi 22, gfx1031) and Navi 21 (gfx1030).
> Audience: a Rust + HIP project doing **forward and backward training** with atomic
> gradient accumulation. Wherever a hardware feature affects backprop or matmul kernels,
> it is called out explicitly.
>
> **Primary sources** (cited inline by `[N]`):
>
> 1. AMD, *“RDNA 2” Instruction Set Architecture Reference Guide* (Nov 2020). [PDF](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
> 2. AMD, *RDNA Architecture Whitepaper* (RDNA 1, 2019). [PDF](https://gpuopen.com/download/RDNA_Architecture_public.pdf)
> 3. LLVM, *AMDGPUUsage* — backend user guide. [HTML](https://llvm.org/docs/AMDGPUUsage.html)
> 4. LLVM, *Syntax of GFX10 RDNA2 Instructions* (`AMDGPUAsmGFX1030`). [HTML](https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX1030.html)
> 5. AMD ROCm, *GPU hardware specifications*. [HTML](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
> 6. AMD ROCm, *Data types and precision support*. [HTML](https://rocm.docs.amd.com/en/latest/reference/precision-support.html)
> 7. AMD GPUOpen, *Occupancy explained*. [HTML](https://gpuopen.com/learn/occupancy-explained/)
> 8. AMD GPUOpen, *RDNA Performance Guide*. [HTML](https://gpuopen.com/learn/rdna-performance-guide/)
> 9. AMD GPUOpen, *AMD GCN Cross-Lane Operations*. [HTML](https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/)
> 10. AMD ROCm Blogs, *Avoiding LDS Bank Conflicts on AMD GPUs*. [HTML](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html)
> 11. Chips and Cheese, *AMD’s RDNA 2: Shooting For the Top*. [HTML](https://chipsandcheese.com/p/amds-rdna-2-shooting-for-the-top)
> 12. AMD ROCm, *HIP Hardware Implementation*. [HTML](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)
> 13. AMD, *“RDNA 3” ISA Reference Guide* (Feb 2023). [PDF](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)
> 14. Interplay of Light, *Optimizing for the RDNA Architecture*. [HTML](https://interplayoflight.wordpress.com/2020/05/23/optimizing-for-the-rdna-architecture-presentation-notes/)

---

## 1. High-Level CU / WGP Layout

RDNA 2 reorganizes the GCN-era “Compute Unit” into a **Workgroup Processor (WGP)**
that pairs **two CUs** sharing **128 KB of LDS** and a single L0 instruction cache
front-end [1, ch. 2]. This is the fundamental scheduling unit the driver targets.

```
                  ┌──────────────────────────────────────────────┐
                  │                  WGP  (×20 on RX 6700 XT)    │
                  │  ┌───────────────┐    ┌───────────────┐      │
                  │  │     CU 0      │    │     CU 1      │      │
                  │  │ ┌───┐ ┌───┐   │    │ ┌───┐ ┌───┐   │      │
                  │  │ │S0 │ │S1 │   │    │ │S0 │ │S1 │   │      │
                  │  │ │32 │ │32 │   │    │ │32 │ │32 │   │      │
                  │  │ └───┘ └───┘   │    │ └───┘ └───┘   │      │
                  │  │ ┌───┐ ┌───┐   │    │ ┌───┐ ┌───┐   │      │
                  │  │ │S2 │ │S3 │   │    │ │S2 │ │S3 │   │      │
                  │  │ └───┘ └───┘   │    │ └───┘ └───┘   │      │
                  │  │ L0$ 16KB      │    │ L0$ 16KB      │      │
                  │  └───────────────┘    └───────────────┘      │
                  │            128 KB  shared  LDS               │
                  │            32 KB   scalar  cache             │
                  └──────────────────────────────────────────────┘
```

Concrete numbers for **Navi 22 / RX 6700 XT** [11]:

| Quantity                                    | Value             |
| ------------------------------------------- | ----------------- |
| Compute Units                               | **40**            |
| Workgroup Processors                        | **20** (40 ÷ 2)   |
| SIMDs per CU                                | **4**             |
| Lanes per SIMD                              | **32**            |
| Total SIMDs                                 | 160               |
| Total lanes (FP32 ALUs)                     | 5 120             |
| LDS per WGP                                 | 128 KB            |
| L0 vector cache per CU                      | 16 KB             |
| L1 cache per Shader Array                   | 128 KB            |
| L2 cache (whole chip)                       | 4 MB              |
| Infinity Cache (whole chip)                 | 96 MB on 6700 XT  |
| Memory bus width                            | 192-bit GDDR6     |
| Peak DRAM bandwidth                         | 384 GB/s @ 16 Gbps |

> Note: ROCm’s spec table [5] groups the whole gfx1030 family in one row and lists the
> *maximum* configuration (Navi 21 / RX 6900 XT: 80 CU, 128 MB Infinity Cache). Navi 22
> on the 6700 XT trims to 40 CU and 96 MB IC [11].

### 1.1 Wave32 vs Wave64 issue paths

Each SIMD is 32 lanes wide. RDNA introduced **wave32** as the native size: a wave32
wavefront is issued in **one cycle** to one SIMD [2, §3]. Wave64 still works for
backward compatibility — the SIMD executes a wave64 instruction in **two consecutive
cycles** over the same 32-lane unit [2, §3.1].

* `gfx10+` defaults to **wave32** for compute kernels (`-mwavefrontsize32`) [3].
* The `wavefrontsize64` target feature is also available; the driver/compiler picks
  per kernel. For this project (training), **always force wave32**: it has lower
  divergence cost, lower register-allocation rounding, and matches one issue cycle.

When does wave64 win?

* Pixel shaders that benefit from instruction-cache footprint compression.
* Kernels where halving the issue-rate exactly hides VALU latency (rare on RDNA 2;
  becomes more attractive on RDNA 3 because of dual-issue, see §12).

For HIP compute on gfx1030, the recommendation is **wave32 + WGP mode** [3, §10].

---

## 2. Register Files and Occupancy

### 2.1 Sizes

| Resource                  | Per SIMD            | Per wave (max)         |
| ------------------------- | ------------------- | ---------------------- |
| VGPR file                 | **128 KB** = 1 024 × 32 lanes × 4 B | **256 VGPRs** addressable per lane [1, §3.6.1] |
| SGPR file                 | shared, 32 KB across CU [5] | **106 SGPRs** addressable per wave [1, §3.6.2] |
| Wave slots per SIMD       | **16** (RDNA 2/3)   | —                     |
| LDS                       | 128 KB / WGP        | per workgroup         |

Note that the per-lane VGPR limit is 256; with 32 lanes, a SIMD with 1 024 architectural
VGPRs can hold **8 waves × 128 VGPRs** or **16 waves × 64 VGPRs** simultaneously. The
hardware can *track* up to 16 waves but the register file caps you well before that for
register-heavy kernels. Source [11] phrases it as: a SIMD “can track up to 16 wavefronts”
on RDNA 2 (down from 20 on RDNA 1).

### 2.2 Allocation granularity

For wave32 on RDNA 2, the LLVM AMDGPU backend allocates VGPRs in **blocks of 24** (from
`AMDGPUSubtarget` heuristics; visible in `-Rpass-analysis=kernel-resource-usage`
output). SGPRs allocate in blocks of 16 within the 106-SGPR ceiling. Practical takeaway:
asking for 25 VGPRs costs you the slot for 48; rounding up the *intended* register
count to a multiple of 24 lets you reason about occupancy honestly.

### 2.3 Occupancy formula

The number of waves resident per SIMD is the minimum across four resources:

```
waves_per_SIMD =
    min( 16,                                              // hardware slots
         floor(1024 / round_up(VGPRs_per_wave, 24)),      // VGPR file
         floor(106  / round_up(SGPRs_per_wave, 16)),      // SGPR file
         floor(LDS_per_WGP_total / LDS_per_workgroup),    // LDS (×WG → per WGP)
         scratch_capacity_per_SIMD / scratch_per_wave )   // private scratch
```

Then `waves_per_WGP = 4 SIMDs × waves_per_SIMD` (subject to LDS sharing across the WGP).

Maximum theoretical: **8 waves/SIMD = 32 waves/WGP** at ≤128 VGPR/wave (wave32),
because at >128 VGPRs/wave you halve again. The VGPR cliffs occur at 128, 96, 80, 64,
56, 48, 40, 32 (wave32). [7]

> The VGPR file is 128 KB / SIMD = 32 768 dwords. With 32 lanes that is 1 024
> dwords/lane = the 1 024-VGPR pool divided among up to 16 waves. The ceiling
> per-wave is 256 VGPRs/lane regardless [1, §3.6.1].

### 2.4 Querying occupancy from binaries

Two primary tools, both work with HIP-compiled `.hsaco`:

* `roc-obj-ls` and `roc-obj-extract` to pull the kernel descriptor from a fat binary
  (ships with ROCm).
* `llvm-objdump --disassemble --mcpu=gfx1030 kernel.hsaco` shows the
  `kernel_descriptor` block which contains:
  * `.granulated_workitem_vgpr_count`
  * `.granulated_wavefront_sgpr_count`
  * `.kernarg_segment_byte_size`
  * `.workgroup_group_segment_byte_size` (LDS)

Compile with `-Rpass-analysis=kernel-resource-usage` to get a one-line summary per
kernel at compile time. Couple this with `-save-temps` to retain the assembly.

---

## 3. Pipeline Depth and Latency

### 3.1 VALU

* **Wave32 VALU latency: 5 cycles** for most ALU ops. Back-to-back dependent
  instructions stall for 4 wait-states unless an unrelated wave/instruction fills
  the bubble [1, §4.2 “Wait states”]. RDNA 2 reduces the GCN-era 4-cycle pipeline to
  a single-cycle issue, but the **pipeline depth is still 5** for VALU-to-VALU
  dependence resolution.
* In practice, the compiler interleaves independent VALU instructions; with ≥2 waves
  per SIMD the SIMD runs back-to-back issues every cycle (the second wave fills the
  shadow of the first).
* Transcendentals (`v_rcp`, `v_rsq`, `v_sqrt`, `v_exp`, `v_log`, `v_sin`, `v_cos`)
  execute at **¼ rate** on the dedicated SFU [8].

### 3.2 SALU

* Most scalar ALU ops are **single cycle** [1, §4.1].
* `s_waitcnt` itself is 1 cycle but the resulting stall depends on which counter is
  being drained.

### 3.3 Memory

| Path                | Typical latency (cycles)                      |
| ------------------- | --------------------------------------------- |
| LDS (`ds_*`)        | ~20 cycles (≈19.5 ns at 2.5 GHz) [11]         |
| L0 vector hit (VMEM)| 80–120 cycles                                 |
| L1 hit              | ~120–180 cycles                               |
| L2 hit              | ~180–250 cycles                               |
| Infinity Cache hit  | ~300–400 cycles                               |
| DRAM                | ~400–600 cycles                               |
| SMEM (constant)     | ~16 cycles via L0 scalar cache                |

These are order-of-magnitude figures gathered from microbenchmarks [11]; AMD does not
publish official latency tables. The mitigation is the same in every case: have enough
in-flight waves so the scheduler can find an issuable instruction every cycle. With
wave32 and ≥4 waves/SIMD you typically hide L0/L1 hits; you need ≥8 waves/SIMD for
DRAM-bound code unless you cover with vectorized loads (§9).

### 3.4 Branch divergence

Divergent branches force the SIMD to mask off lanes via the **EXEC** register and
execute both paths serially. The cost is exactly the work of the under-utilized lanes
plus the branch instructions themselves. Wave32 halves the worst-case divergence
penalty vs. wave64 because the wave is only 32 lanes wide.

* `s_cbranch_exec[n]z` — branch when EXEC is (non-)zero.
* `s_andn2_b32 exec, exec, vcc` — temporarily disable lanes for an `if`.
* `s_or_saveexec` / `s_andn2_saveexec` — save/restore EXEC across regions.

For backprop, the divergence cost shows up in (a) ReLU/GELU branches on gradient
masks and (b) variable-length sequence kernels. In both cases prefer
**predication** (multiply by 0/1) to actual branches when the work per lane is small.

---

## 4. LDS (Local Data Share)

### 4.1 Layout

* **128 KB per WGP** (shared between the two CUs in WGP mode). In CU mode it splits
  into 2 × 64 KB [3, §10].
* **32 banks**, each **4 bytes** wide (1 dword) [10]. Stride-1 dword access by 32
  consecutive lanes = bank-conflict-free. Stride that is a multiple of 32 dwords
  (128 bytes) hits the same bank → maximum conflict.

### 4.2 Bank-conflict patterns

For wave32:

* `ds_read_b32 v, v[idx]`: each of 32 lanes reads a dword. Bank = `(byte_addr / 4) mod 32`.
* Conflict-free shapes:
  * Stride 1 dword (the obvious row).
  * Stride coprime with 32 (e.g., stride 33 — the classic “add 1 padding word per row”
    trick for 32-wide tiles).
  * XOR-swizzled column index `j_phys = j ^ row_base_xor` so that successive rows
    rotate the bank pattern [10].
* Worst case: 32-way conflict serializes into 32 cycles instead of 1.

Wider accesses (`ds_read_b64`, `ds_read2_b32`, `ds_read_b128`, `ds_read2st64_b32`) are
documented in [10] with their per-lane bank-mapping rules. For RDNA 2 wave32, the
useful form for matmul tiles is `ds_read_b128` (4 dwords/lane → 4 lanes per bank
group) which **must** be conflict-free or it serializes into 4 cycles.

### 4.3 LDS instructions (subset)

| Instruction                | What it does                                        |
| -------------------------- | --------------------------------------------------- |
| `ds_read_b32` / `_b64` / `_b128` | LDS load, 4 / 8 / 16 bytes per lane            |
| `ds_write_b32` / `_b128`   | LDS store                                           |
| `ds_read2_b32 / b64`       | Two strided loads in one instruction                |
| `ds_add_u32`               | Atomic integer add to LDS                           |
| **`ds_add_f32`**           | **Atomic FP32 add to LDS — native on RDNA 2** [1, §10.2 / §10.7] |
| `ds_min_f32` / `ds_max_f32` | Native LDS atomic FP32 min/max                     |
| `ds_swizzle_b32`           | Cross-lane permute (offset-encoded pattern) [9]     |
| `ds_permute_b32` / `_bpermute_b32` | Forward / reverse cross-lane gather using LDS hardware path [9] |

### 4.4 `ds_add_f32` is native — confirm

The RDNA 2 ISA explicitly lists `DS_ADD_F32` and `DS_ADD_RTN_F32` as supported
opcodes for LDS [1, §10.7 “DS Instructions”]. This is **not** a compiler emulation;
the LDS hardware contains the FP32 adder. Practical implications for backprop:

* You can do **per-workgroup gradient accumulation in LDS** with native f32 atomics.
* Then a single thread per workgroup performs the **global** atomic via the
  CAS-loop (next section) or does `atomicAdd` on a packed integer counter.

This is the single most important micro-architectural fact for the training kernels:
*native FP32 atomic add exists in LDS but not in global memory* on RDNA 2.

### 4.5 LDS as a workgroup-scope reduction tool

Pattern (write-once, no false-sharing):

```text
1. Each lane computes its partial gradient -> VGPR
2. Wave-level reduction with DPP (no LDS, see §7)
3. Lane 0 of each wave: ds_add_f32 LDS[bin]   ← native
4. s_barrier
5. One lane per workgroup: global_atomic_cmpswap_b32 loop on global gradient
```

This combines a free fast path (DPP), one cheap LDS atomic, and exactly one
contended global atomic per workgroup per bin.

---

## 5. Caches

### 5.1 Hierarchy on Navi 22

```
            ┌────────────────────────────────────────────────────────┐
   per CU   │  L0 vector $  16 KB     L0 scalar $  16 KB             │
            │  cache line: 128 B                                     │
            └──────────────┬─────────────────────────────────────────┘
                           │
            ┌──────────────▼─────────────────────────────────────────┐
   per SA   │  L1 cache  128 KB   (shared by all WGPs in a SA)       │
            └──────────────┬─────────────────────────────────────────┘
                           │
            ┌──────────────▼─────────────────────────────────────────┐
   chip     │  L2 cache  4 MB                                        │
            └──────────────┬─────────────────────────────────────────┘
                           │
            ┌──────────────▼─────────────────────────────────────────┐
   chip     │  Infinity Cache 96 MB (Navi 22) / 128 MB (Navi 21)     │
            └──────────────┬─────────────────────────────────────────┘
                           │
                       GDDR6 192-bit @ 16 Gbps  → 384 GB/s
```

[11], [5]. RDNA introduced **128-byte cache lines** (vs. 64 B on GCN) [14]: align
your tile loads to 128 B and prefer `global_load_dwordx4` (16 B/lane × 32 lanes =
512 B = 4 lines) so a single instruction touches a whole-aligned region.

### 5.2 Coherence and scope flags

LDS is workgroup-scope; global memory is system-scope but the caches are not all
coherent at every level. HIP exposes scope via `__atomic_*_n` builtins or the
`hip::thread_scope_*` enum:

* `thread_scope_block` — workgroup. LDS atomics or coherent through L0/L1 within
  the WGP.
* `thread_scope_device` — agent / GPU. Requires invalidation through L2; the
  compiler emits the appropriate `glc`/`dlc` flags on `global_*` and a
  `buffer_gl0_inv` / `buffer_gl1_inv` after.
* `thread_scope_system` — host-coherent. Uses MTYPE = uncached or system-scope
  bypass; correct for GPU↔CPU shared atomics but very slow.

For training:

* Forward and backward kernels within a single launch ⇒ **agent scope**.
* Gradient buffer reused across multiple kernel launches ⇒ a `hipDeviceSynchronize`
  or a stream barrier flushes the L2. No need for system-scope atomics.
* Avoid system-scope unless explicitly mixing CPU writes with running kernels.

---

## 6. Atomic Operations — CRITICAL for Backprop

This is the section that most influences kernel design on this hardware.

### 6.1 Support matrix on gfx1030 (RDNA 2)

| Operation                                  | Global memory      | LDS                |
| ------------------------------------------ | ------------------ | ------------------ |
| `atomic_add` u32 / u64                     | ✅ native          | ✅ native (`ds_add_u32`) |
| `atomic_cmpswap` b32 / b64                 | ✅ `global_atomic_cmpswap[_x2]` | ✅ `ds_cmpst_b32` |
| `atomic_min/max` i32 / u32                 | ✅                 | ✅                 |
| `atomic_min/max` **f32**                   | ✅ `global_atomic_fmin/fmax` [4] | ✅ `ds_min/max_f32` |
| **`atomic_add` f32**                       | ❌ **NOT in HW** [6] | ✅ **`ds_add_f32` native** [1] |
| `atomic_add` f64                           | ❌                 | ❌                 |
| `atomic_pk_add` 2×f16 / 2×bf16             | ❌ (RDNA 3+)       | ❌ (RDNA 3+) [6]   |

The single defining constraint: **there is no `global_atomic_add_f32` on RDNA 2.**
LLVM and HIP will emit a **CAS loop** transparently for `atomicAdd(float*, float)`
on `global` memory. The CAS loop is:

```
loop:
   v_old = global_load_dword [addr]                    ; read
   v_new = v_old + delta                               ; compute
   v_cas = global_atomic_cmpswap_b32 [addr], v_old, v_new   ; try
   v_ok  = (v_cas == v_old)
   s_cbranch_exec ! v_ok, loop
```

`global_atomic_cmpswap_b32` is fully supported [4]. `global_atomic_min_f32` and
`global_atomic_max_f32` are also natively supported (handy for grad-clip style
operators that only need a max).

### 6.2 Performance characteristic of the CAS-loop f32 add

Under contention:

* **Uncontended** (each lane writes a unique cache line): ≈ 1 round-trip to L2,
  ~250 cycles. Effective throughput ≈ DRAM-bound for >L2-resident gradients.
* **Lightly contended** (a few waves, 1–2 retries average): 2–3× the uncontended
  cost.
* **Hot bin** (every workgroup hammers the same f32): retry storms; we have
  observed effective throughput collapsing by 5–20× vs the integer case.

> The Intel oneAPI guide on atomics [Intel docs] notes that software-emulated atomics
> have a “high negative performance impact when they frequently access the same
> memory address.” The same logic applies on RDNA 2; the CAS loop *is* the emulation.

### 6.3 Best practice: hierarchical reduction

For a backward pass that accumulates gradients into a global tensor `g[]`:

```text
Stage 1 — wave-local
  Each lane has dx_i. Reduce across the wave with DPP fan-in (5 ops, see §7).
  Result lives in lane 0 of the wave.

Stage 2 — workgroup-local
  Lane 0 of each wave executes ds_add_f32 LDS[bin].   // native, fast
  __syncthreads() / s_barrier.

Stage 3 — global
  One lane per workgroup reads LDS[bin] and emits one atomicAdd(&g[bin], v),
  which the compiler lowers to the global CAS loop.
```

This converts hundreds of contending global atomics per workgroup into **one**.
For dense Linear / Conv backward on RDNA 2 this is consistently the fastest pattern.

### 6.4 Useful global atomic instructions on gfx1030 [4]

```
global_atomic_add_x2          // u64 add
global_atomic_and / or / xor  // bitwise
global_atomic_cmpswap[_x2]    // f32-via-CAS lives here
global_atomic_inc / _dec      // counter
global_atomic_min / _max      // signed/unsigned, also _f32
global_atomic_smin / _smax / _umin / _umax
global_atomic_swap[_x2]
```

Note the absence of `global_atomic_add_f32` and `global_atomic_pk_add_f16` —
those land in `gfx11` / RDNA 3 [13].

---

## 7. Wave32 Specifics

A wavefront is 32 lanes; one issue cycle per wave32 instruction per SIMD.

### 7.1 Cross-lane operations [9]

| Instruction          | Purpose                                                          |
| -------------------- | ---------------------------------------------------------------- |
| `v_readfirstlane_b32` | Move the value of the first active lane to an SGPR              |
| `v_readlane_b32 sd, vsrc, lane` | Read a specific lane to scalar                       |
| `v_writelane_b32 vdst, src, lane` | Write a scalar into one lane of a VGPR             |
| `ds_swizzle_b32`     | Limited cross-lane permute, pattern in instruction offset (no extra VGPR) |
| `ds_permute_b32`     | "send my data to lane i" (forward)                              |
| `ds_bpermute_b32`    | "fetch data from lane i" (backward — most useful)               |

DPP modifiers attach to a normal VOP1/VOP2 to read-and-permute in one instruction,
at full ALU throughput [9]:

* `quad_perm` — full crossbar within groups of 4 lanes
* `row_shl/shr/ror N` — shift within a 16-lane row
* `row_mirror`, `row_half_mirror`
* `wave_shl/shr/ror` — within the whole wave
* `row_bcast15`, `row_bcast31` — broadcast lane 15 or 31 across rows

The DPP path has hazards: 2 wait-states if the prior VALU wrote the input VGPR; 5
wait-states if the prior op modified EXEC [9].

### 7.2 HIP subgroup intrinsics (compile to the above)

```cpp
__shfl(value, lane)              // ds_bpermute_b32 / DPP
__shfl_xor(value, mask)          // ds_swizzle_b32 with XOR pattern
__shfl_up(value, delta)
__shfl_down(value, delta)
__ballot(predicate)              // returns u64; on wave32 only low 32 bits are valid
__any(pred), __all(pred)
```

Caveat: `__ballot` returns u64 even in wave32 mode; on gfx1030 use the lower 32 bits.
`__activemask()` returns the EXEC mask of the current wave.

### 7.3 Wave-level reduction pattern (sum, wave32)

5 DPP shuffles to reduce 32 lanes into lane 0:

```cpp
float v = x;
v += __shfl_xor(v,  1);
v += __shfl_xor(v,  2);
v += __shfl_xor(v,  4);
v += __shfl_xor(v,  8);
v += __shfl_xor(v, 16);
// every lane now holds the wave sum
```

This is **5 VALU instructions** with no LDS traffic. Compare with an LDS-based
reduction (1 LDS write + barrier + 5 reads): the DPP version is ~3× faster on
RDNA 2.

---

## 8. Dual-Issue and Instruction Throughput

### 8.1 Dual-issue on RDNA 2

The frequently repeated claim that RDNA 2 has VOPD dual-issue is **incorrect**.
The dedicated **VOPD encoding** ("V_DUAL_*") for paired VALU instructions is an
RDNA 3 (`gfx11`) feature [13, ch. 7]. On RDNA 2:

* The SIMD has its primary FMA/MUL/ADD ALU plus a **separate transcendental SFU**.
* The two can co-issue: a VALU op and a transcendental can be in flight in the
  same cycle from the same SIMD ("co-execute transcendentals at quarter rate" [8]).
* There is **no general-purpose dual-issue** of two FMAs per cycle on RDNA 2.

Practical: code that mixes a hot FMA loop with `rcp`/`rsq` gets the transcendentals
"for free" because they fill the SFU bubble. Code with only FMAs is single-issue
per SIMD per cycle.

### 8.2 Throughput per SIMD per cycle (32 lanes)

| Op                                        | Ops per SIMD-cycle (wave32) | Notes |
| ----------------------------------------- | ---------------------------- | ----- |
| `v_mul_f32`, `v_add_f32`                  | 32                          | 1× rate |
| `v_fma_f32` / `v_mac_f32`                 | 32 (2 FLOPs each → 64 FLOPs) | This is the GEMM peak. |
| `v_pk_fma_f16` / `v_pk_mul_f16`           | 64 f16 FMA (128 FLOPs)       | Packed 2×f16. RDNA 2 has full v_pk_*. |
| `v_dot4_i32_i8`                           | 32 dot4-int8 (4 MACs each → 128 MAC/cycle) | The int8 GEMM peak. |
| `v_dot8_i32_i4`                           | 32 dot8-int4 (8 MACs each → 256 MAC/cycle) | int4 GEMM. |
| `v_rcp_f32` / `v_sqrt_f32`                | 8 (¼ rate)                  | SFU. |
| `v_cvt_*_f32` (most conversions)          | 32                          | |
| `v_cvt_pkrtz_f16_f32`                     | 32 → 64 lanes packed        | Useful for f32→f16 stores. |

**Peak FP32 FMA on RX 6700 XT** = 32 lanes × 4 SIMDs × 40 CUs × 2 FLOPs × 2.58 GHz
≈ **26.4 TFLOPS** (matches AMD’s spec).

**Peak int8 dot4** = 4× FP32 = ~106 TOPS, achievable only with `v_dot4_i32_i8`
fed from packed-int8 VGPRs.

### 8.3 dot4 / dot8 hardware

On gfx1030 the LLVM target includes `dot1-insts`, `dot2-insts`, etc. The relevant
ones:

* `v_dot2_f32_f16 d, a, b, c` — 2-way f16 FMA into f32 accumulator.
* `v_dot4_i32_i8 d, a, b, c` — 4-way int8 dot product, i32 accumulator.
* `v_dot4_u32_u8` — unsigned variant.
* `v_dot8_i32_i4` — 8-way int4 dot.
* `v_dot2c_f32_f16_dpp` and `v_dot4c_i32_i8_dpp` — DPP variants for in-instruction
  shuffle [4].

These are not "matrix cores" (those are CDNA / RDNA 3 WMMA only — see §12). They
are vector-lane SIMD dot instructions that pack work into a single VALU cycle.
They are still the right path for **int8 inference / int8 GEMM** on this GPU.

### 8.4 Packed half / bfloat

* RDNA 2 supports the `v_pk_*` family for packed f16: `v_pk_fma_f16`,
  `v_pk_mul_f16`, `v_pk_add_f16`, `v_pk_max_f16`, plus integer `v_pk_*_u16`/`_i16`.
* **bf16** has weaker support: conversions exist (`v_cvt_*`), but the packed FMA
  for bf16 (`v_pk_fma_bf16`) is RDNA 3+.
* For training on RDNA 2, the realistic dtype set is **f32 master, f16 packed
  compute, fall back to f32 for bf16 ops**.

---

## 9. Memory Subsystem Performance

### 9.1 Bandwidth

* **Peak DRAM:** 384 GB/s (192-bit GDDR6 @ 16 Gbps) on RX 6700 XT.
* **Infinity Cache hit:** approximately **1.5–1.8 TB/s** effective on Navi 22 — a
  4× to 5× multiplier over DRAM for working sets ≤ 96 MB.
* **L2:** ~3 TB/s aggregate.
* **L1 (per SA):** ~6 TB/s aggregate.
* **L0 (per CU):** 64 B/cycle/CU; with 40 CUs at 2.58 GHz that's ~6.6 TB/s aggregate.

### 9.2 Coalescing

A wave32 vector load coalesces if all 32 lanes hit the **same 128-byte L0 line**.
Stride-1 dword load by 32 contiguous lanes = exactly one line, one transaction.
Stride-2 dword load = two lines, two transactions; this halves effective
bandwidth.

> Because the L0 line is 128 B (not 64 B as on GCN), aligning your gradient and
> activation tensors to **128 B** is more important than on GCN.

### 9.3 Vectorized loads

| Instruction                  | Bytes/lane | Use case                              |
| ---------------------------- | ---------- | ------------------------------------- |
| `global_load_dword`          | 4          | Default per-element load              |
| `global_load_dwordx2`        | 8          | f64 or 2×f32                         |
| `global_load_dwordx3`        | 12         | float3 / RGB                          |
| `global_load_dwordx4`        | 16         | **Tile loads, weight loads**          |
| `global_load_short_d16`      | 2          | f16 with VGPR packing                 |
| `global_load_ubyte_d16`      | 1          | int8 weights with packing             |

A single `global_load_dwordx4` from 32 lanes is **512 B = 4 cache lines**. This
maximizes the coalescer and the line-fill engines. For matmul tile loaders, always
use the `_x4` form.

### 9.4 Scalar memory (SMEM)

`s_load_dword` reads through the **L0 scalar cache** (16 KB per CU [5], shared by
all waves on the CU). It is the right path for:

* Kernel arguments (compiler does this automatically).
* Read-only constants that are *uniform across the wave* (a bias broadcast,
  layer-norm gamma if the layout is shared).
* Small lookup tables (≤ 16 KB, hit rate is ~90%+ in practice [11]).

Do **not** use SMEM for per-lane data; it is one-load-per-wave, not 32 loads.

---

## 10. Scheduling and Occupancy Tuning

### 10.1 Aim for 8 waves/SIMD on training kernels

Training kernels are usually memory-bound or atomic-bound, not ALU-bound. The
scheduler needs many independent waves to hide the 200–500 cycle DRAM latencies.
**8 waves/SIMD = 32 waves/WGP = 64 waves on a 2-WGP launch** — this is the practical
sweet spot when register pressure ≤ 128 VGPRs/wave.

### 10.2 When low-occupancy is faster

Sometimes fewer, fatter waves win:

* Heavy register reuse in a matmul tile (e.g., 16×16 tile per wave kept entirely
  in VGPRs) — cutting tile size to fit 8 waves/SIMD often *reduces* arithmetic
  intensity and net throughput.
* Long FMA chains where ILP within the wave already covers VALU latency.
* Kernels with very large LDS tiles (e.g., a 32 KB LDS tile lets only 4
  workgroups per WGP).

The rule of thumb: if the kernel hits ≥80% of memory-bandwidth or ≥80% of
peak FLOPS at 4 waves/SIMD, do not push for 8.

### 10.3 Reading register pressure from `.hsaco`

```bash
# Disassemble to see the kernel descriptor
llvm-objdump -d --mcpu=gfx1030 my_kernel.hsaco | head -80

# At compile time
hipcc -O3 -mcpu=gfx1030 \
      -Rpass-analysis=kernel-resource-usage \
      -save-temps \
      kernel.hip -c -o kernel.o
```

The compile-time pass prints per-kernel:

```
remark: <kernel_name>: VGPRs: 88, SGPRs: 48,
        ScratchSize: 0, AGPRs: 0,
        Occupancy: 8 waves per SIMD
```

If `Occupancy < target`, the dump tells you which resource is the bottleneck (read
the VGPR / SGPR / LDS lines).

---

## 11. Wavefront Execution Model

* **EXEC** (64 bits, but only low 32 used in wave32) — per-lane execute mask. A
  divergent branch saves EXEC, narrows it, runs one side, restores, narrows again.
* **VCC** (vector condition code, 64 bits, low 32 used in wave32) — destination
  for compare instructions; consumed by `s_cbranch_vcc[n]z` and as the carry-in
  for adds.
* **SCC** (scalar condition code, 1 bit) — set by SALU compares; consumed by
  `s_cbranch_sc[c]0/1`.

### 11.1 Re-convergence on RDNA

Unlike pre-Volta NVIDIA, AMD does not promise per-thread program counters: lanes
re-converge implicitly when the EXEC stack is restored at the end of an `if/else`.
This means:

* You can rely on lockstep at the end of any `if` block.
* `__syncwarp()` exists in HIP but is a no-op on AMD wave32 — the wave is already
  in lockstep (unless mid-divergence).
* Be careful with `__syncthreads()` inside a divergent branch: all lanes of a
  workgroup must reach the same `s_barrier`. If a divergent branch can skip the
  barrier, you deadlock.

---

## 12. RDNA 2 vs RDNA 3 vs CDNA — Forward-Compat Planning

| Feature                              | RDNA 2 (gfx1030) | RDNA 3 (gfx1100) | CDNA 2 (gfx90a, MI250) | CDNA 3 (gfx940/942, MI300) |
| ------------------------------------ | ---------------- | ---------------- | ---------------------- | -------------------------- |
| Wave size native                     | 32               | 32 (graphics) / 64 (some compute) | 64                | 64                          |
| `global_atomic_add_f32`              | ❌ (CAS loop)    | ✅ native        | ✅ native              | ✅ native                   |
| `global_atomic_pk_add_f16/bf16`      | ❌               | ✅               | ✅                     | ✅                          |
| `ds_add_f32`                         | ✅               | ✅               | ✅                     | ✅                          |
| Matrix Cores (MFMA)                  | ❌               | ❌               | ✅ MFMA                | ✅ MFMA bf16/f8/i8          |
| WMMA (vector matmul)                 | ❌               | ✅ `v_wmma_*`    | ❌                     | ❌                          |
| VOPD dual-issue                      | ❌               | ✅ V_DUAL_*      | ❌                     | ❌                          |
| Packed bf16 FMA                      | ❌               | ✅               | ✅                     | ✅                          |
| Dynamic register allocation          | ❌               | ❌               | ❌                     | partial / RDNA 4 has it    |

**Portability rules:**

* Code that uses `atomicAdd(float*, float)` is portable but pays the CAS-loop tax
  on RDNA 2 only. Reduction-tree pattern (§6.3) is the portable optimization.
* Code that uses `__half2 atomicAdd` (`atomic_pk_add_f16`) compiles **only on
  gfx11 / CDNA**. Guard with `#if defined(__gfx1100__) || ...`.
* Code that uses `v_wmma_*` (HIP `__builtin_amdgcn_wmma_*`) is RDNA 3 only;
  matrix-core code is CDNA only. **Neither runs on gfx1030**.
* `v_dot4_i32_i8` is portable RDNA 2 → RDNA 4 (and CDNA via emulation).

For an MI300 deployment of the same Rust/HIP codebase, the changes would be:

1. Replace the CAS-loop f32 atomic with native `global_atomic_add_f32`.
2. Optionally add an MFMA path for matmul (different intrinsics, BF16 input).
3. Switch wave size mode to wave64 (CDNA default).

---

## 13. Performance Counters and Profiling

### 13.1 Useful counters via `rocprof`

| Counter                       | What it measures                                  |
| ----------------------------- | ------------------------------------------------- |
| `SQ_WAVES`                    | Wavefronts dispatched. Sanity-check workload size.|
| `SQ_INSTS_VALU`               | VALU instructions issued. Roofline model FLOPs.   |
| `SQ_INSTS_VMEM`               | Vector memory instructions.                       |
| `SQ_INSTS_SMEM`               | Scalar memory instructions.                       |
| `SQ_INSTS_LDS`                | LDS (`ds_*`) instructions.                        |
| `SQ_INSTS_SALU`               | Scalar ALU instructions.                          |
| `GRBM_GUI_ACTIVE`             | Total GPU-active cycles. Denominator for "% busy". |
| `SQ_BUSY_CYCLES`              | SQ busy cycles.                                   |
| `SQ_WAIT_INST_LDS`            | Cycles waited on `lgkmcnt` (LDS/SMEM) — too high = LDS-bound. |
| `TCC_HIT[0-15]_sum`           | L2 hits per channel.                              |
| `TCC_MISS[0-15]_sum`          | L2 misses → DRAM/IC reads.                        |
| `TCP_TCC_READ_REQ_sum`        | Read requests from L0 to L2. Bandwidth proxy.     |
| `GL2C_HIT_sum` / `GL2C_MISS_sum` | L2 hit/miss on RDNA naming.                    |
| `GCEA_*` / `GCR_*`            | Memory client / cache controllers.                |

### 13.2 Occupancy in rocprof v2

Derived metrics (`pmc:` group):

* `MeanOccupancyPerCU` = `SQ_WAVES / GRBM_GUI_ACTIVE / 4 SIMDs / 16 slots`
* `VALUUtilization` = `SQ_INSTS_VALU * 4 / SQ_BUSY_CYCLES` (rough; 4 cycles per
  wave64, 1 cycle per wave32)
* `LDSBankConflict` = approximated from `SQ_WAIT_INST_LDS` divided by
  `SQ_INSTS_LDS`; > ~1 cycle/op suggests conflicts.

### 13.3 Quick recipe

```bash
rocprof --hip-trace --hsa-trace -i counters.txt --timestamp on ./train

# counters.txt:
pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_LDS SQ_INSTS_VMEM
pmc: GL2C_HIT_sum GL2C_MISS_sum GRBM_GUI_ACTIVE SQ_BUSY_CYCLES
```

---

## 14. Compiler Flag Reference for hipcc on gfx1030

```bash
hipcc -O3 \
      --offload-arch=gfx1030 \
      -mwavefrontsize32 \      # default for gfx10+, but be explicit
      -mno-cumode \            # WGP mode (RDNA default & better for compute)
      -ffast-math \            # allow FMA contraction, no-NaN, no-Inf
      -fno-signed-zeros \
      -fhip-new-launch-api \
      -Rpass-analysis=kernel-resource-usage \
      -save-temps \
      kernel.hip -c -o kernel.o
```

Notes:

* `-mwavefrontsize32` — default for `gfx10+`; using `-mwavefrontsize64` is rarely
  beneficial on this part [3].
* `-mcumode` / `-mno-cumode` — WGP mode (`-mno-cumode`) is the RDNA default and
  the correct choice for HIP compute. CU mode splits LDS into two halves [3, §10].
* `-O3 -ffast-math` — necessary for the compiler to fuse `v_mul + v_add` into
  `v_fma`. Without it, the compiler keeps strict IEEE-754 ordering and you lose
  half the FLOPS.
* `-Rpass-analysis=kernel-resource-usage` — prints VGPR/SGPR/LDS/occupancy per
  kernel.
* `-Rpass-missed=loop-vectorize` — diagnose missed vectorization.
* `--offload-device-only -S` — emit raw GFX10 assembly for inspection.

For Rust + HIP via `hip-rs` or a custom build script, set these in the
`HIPCC_COMPILE_FLAGS` env var or in the `hip_runtime_sys` build script.

---

## 15. Authoritative References

* AMD, *“RDNA 2” Instruction Set Architecture Reference Guide* (Nov 2020).
  <https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf>
* AMD, *RDNA 1.0 Instruction Set Architecture Reference Guide*.
  <https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna-shader-instruction-set-architecture.pdf>
* AMD, *RDNA Architecture Whitepaper* (2019).
  <https://gpuopen.com/download/RDNA_Architecture_public.pdf>
* AMD, *“RDNA 3” ISA Reference Guide* (Feb 2023).
  <https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf>
* LLVM, *AMDGPU Backend User Guide*. <https://llvm.org/docs/AMDGPUUsage.html>
* LLVM, *Syntax of GFX10 RDNA2 Instructions*. <https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX1030.html>
* AMD ROCm, *GPU hardware specifications*. <https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html>
* AMD ROCm, *Data types and precision support*. <https://rocm.docs.amd.com/en/latest/reference/precision-support.html>
* AMD ROCm, *HIP Hardware Implementation*. <https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html>
* AMD GPUOpen, *Occupancy explained*. <https://gpuopen.com/learn/occupancy-explained/>
* AMD GPUOpen, *RDNA Performance Guide*. <https://gpuopen.com/learn/rdna-performance-guide/>
* AMD GPUOpen, *AMD GCN Cross-Lane Operations*. <https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/>
* AMD GPUOpen, *Optimizing for the Radeon RDNA Architecture* (slides).
  <https://gpuopen.com/wp-content/uploads/slides/GPUOpen_Let%E2%80%99sBuild2020_Optimizing%20for%20the%20Radeon%20RDNA%20Architecture.pdf>
* AMD ROCm Blogs, *Avoiding LDS Bank Conflicts*.
  <https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html>
* Chips and Cheese, *AMD’s RDNA 2: Shooting For the Top* (Chester Lam).
  <https://chipsandcheese.com/p/amds-rdna-2-shooting-for-the-top>
* Chips and Cheese, *Microbenchmarking AMD’s RDNA 3 Graphics Architecture*.
  <https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture>
* Interplay of Light, *Optimizing for the RDNA Architecture: presentation notes*.
  <https://interplayoflight.wordpress.com/2020/05/23/optimizing-for-the-rdna-architecture-presentation-notes/>
* AMD, *Radeon GPU Analyzer (RGA)* — instruction-level static analysis.
  <https://gpuopen.com/rga/>
* AMD, *Radeon GPU Profiler (RGP)* — runtime profiling, occupancy view.
  <https://gpuopen.com/rgp/>
* AMD, *rocprof user guide*.
  <https://rocm.docs.amd.com/projects/rocprofiler/en/latest/>

---

## Appendix A — Backprop Kernel Checklist (gfx1030)

1. **Compile flags**: `-O3 -mcpu=gfx1030 -mwavefrontsize32 -mno-cumode -ffast-math
   -Rpass-analysis=kernel-resource-usage`.
2. **Tile size**: pick so VGPRs ≤ 128 ⇒ 8 waves/SIMD; LDS ≤ 32 KB so 4 workgroups
   per WGP fit in 128 KB.
3. **Loads**: use `global_load_dwordx4`-equivalent (`float4` or `int4` in HIP) for
   every tile load; align tensors to 128 B.
4. **Wave reduction**: 5× `__shfl_xor` for FP32 sum/max — never use LDS for
   wave-internal reductions.
5. **Workgroup reduction**: lane 0 of each wave does **`ds_add_f32`** (native);
   then `__syncthreads()`.
6. **Global gradient write**: **one** `atomicAdd(&g[bin], v)` per workgroup per bin.
   Accept the CAS-loop cost there because it is paid once.
7. **Avoid**: per-lane `atomicAdd` to global f32 (CAS-loop storm), `__half2`
   atomic adds (RDNA 3+), wave64 mode, CU mode, branching where you could
   predicate.
8. **Validate**: `rocprof` for `SQ_WAVES`, `GL2C_MISS_sum`,
   `SQ_WAIT_INST_LDS` to confirm the kernel is bandwidth-bound or
   atomic-bound and not LDS-conflict bound.

## Appendix B — Quick Cheatsheet

```
SIMD       : 32 lanes, wave32 = 1 issue cycle
SIMDs/CU   : 4
CUs/WGP    : 2  (LDS shared)
WGPs       : 20 on RX 6700 XT
VGPRs      : 256/lane max, 1024-VGPR pool / SIMD, granularity 24 (wave32)
SGPRs      : 106/wave max, 32 KB pool / CU
LDS        : 128 KB / WGP, 32 banks × 4 B
L0 vec     : 16 KB / CU, 128 B lines
L0 scal    : 16 KB / CU
L1         : 128 KB / SA
L2         : 4 MB / chip
IC         : 96 MB on Navi 22 (128 MB on Navi 21)
DRAM BW    : 384 GB/s @ 16 Gbps GDDR6, 192-bit
Peak FP32  : ~26.4 TFLOPS (RX 6700 XT @ ~2.58 GHz boost)
Peak int8  : ~106 TOPS via v_dot4_i32_i8
VALU lat   : 5 cycles
LDS lat    : ~20 cycles
DRAM lat   : ~400-600 cycles
Atomic add f32 — global   : ❌ (CAS loop)
Atomic add f32 — LDS      : ✅ ds_add_f32 native
```
