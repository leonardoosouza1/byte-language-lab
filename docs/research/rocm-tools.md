# ROCm Developer Tooling — Practical Reference for RDNA2 (gfx1030/1031)

Audience: someone writing custom HIP kernels and HSA AQL dispatch from Rust on RDNA2 (RX 6000 series, Navi 21/22). This is a working reference: every section gives commands, expected output, and decision rules.

Date of this snapshot: 2026-05. ROCm 7.x is current. Several legacy tools are now deprecated and renamed; the migration map is given inline.

---

## 0. Tool landscape in ROCm 7 (the migration map)

ROCm 7 has rebranded most of the profiling stack onto a single SDK (`rocprofiler-sdk`). Names you may still see in old blog posts vs current names:

| Legacy name      | ROCm 7 name            | Status                                |
|------------------|------------------------|---------------------------------------|
| `rocprof` (v1)   | `rocprofv3`            | Deprecated, EOS planned ~2026 Q2      |
| `rocprofv2`      | `rocprofv3`            | Deprecated, never left beta           |
| `omniperf`       | `rocprofiler-compute` (`rocprof-compute`) | Renamed in ROCm 6.3, active |
| `omnitrace`      | `rocprofiler-systems` (`rocprof-sys-*`)  | Renamed, active            |
| `roc-obj-ls`     | `llvm-objdump --offloading` | Removed in ROCm 7.0              |
| `roc-obj-extract`| `llvm-objdump --offloading` | Removed in ROCm 7.0              |

**Decision rule for a new project on RDNA2 in 2026:** start with `rocprofv3` for counters/traces, `rocprof-compute` for per-kernel roofline analysis if you can get it working on consumer cards (it is officially Instinct-only), `rocgdb` for correctness, and offline analysis via `llvm-objdump`/`rga` for ISA. Avoid investing in `rocprof` (v1) or `rocprofv2` even though they still work on ROCm 7.x.

---

## 1. rocprof v1 vs v2 vs v3 vs ATT

### 1.1 What each is

- **rocprof (v1)** — Original CLI wrapper around the legacy `rocprofiler` library. Kernel-level metrics (PMC counters), HSA/HIP API tracing, basic CSV/JSON output. Simple, well-known, but the input grammar is awkward and parallel multi-counter sets force you to re-run the kernel.
- **rocprofv2** — Refactored CLI on top of the same library. Slightly cleaner, plugin system (`-d <plugin>`), still the same counter back-end. Officially never left beta and now deprecated.
- **rocprofv3** — Modern wrapper on top of `rocprofiler-sdk` (the new C/C++ profiling library that replaces `rocprofiler` + `roctracer`). Supports text, YAML, and JSON input files; output as CSV, JSON, `pftrace` (Perfetto), `otf2`, and the new `rocpd` SQLite database. Use this one.
- **rocprof ATT (Advanced Thread Trace / SQTT)** — Per-wave instruction-level trace. Captures the actual instruction stream of selected wavefronts on selected SIMDs/CUs, with stall reasons and timestamps. On RDNA2 (gfx10) "Latency = Stall time + Execute time". The trace is binary and is decoded by `rocprof-trace-decoder` (now open-source) and visualized in `rocprof-compute-viewer`. This is the AMD analogue of NVIDIA SASS-level Nsight Compute traces.

### 1.2 Typical command lines

```sh
# v1 (legacy) — basic stats and HSA/HIP trace
rocprof --stats --hsa-trace --hip-trace ./prog
# produces: results.csv, results.stats.csv, results.hsa_trace.txt, results.json

# v1 — counter file
cat > counters.txt <<EOF
pmc : SQ_WAVES SQ_INSTS_VALU SQ_INSTS_LDS GRBM_GUI_ACTIVE TCC_HIT_sum TCC_MISS_sum
pmc : SQ_INSTS_FLAT SQ_INSTS_SMEM TA_BUSY_avr
range: 0 : 50
gpu: 0
kernel: my_kernel reduce_kernel
EOF
rocprof -i counters.txt --output-format csv -o counters.csv ./prog

# v3 — modern equivalent
rocprofv3 --hip-trace --hsa-trace --kernel-trace --stats \
          --output-format csv -- ./prog

rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE \
          --kernel-trace --output-format csv -- ./prog

# v3 — input file (txt is just a counter list, YAML/JSON can carry the full config)
echo "pmc: SQ_WAVES SQ_INSTS_VALU L2CacheHit" > counters.txt
rocprofv3 -i counters.txt -- ./prog

# v3 — ATT (advanced thread trace)
rocprofv3 --att --att-target-cu 0 --att-simd-select 0xF \
          --att-buffer-size 0x60000000 --kernel-iteration-range 0 \
          --output-format csv -- ./prog
```

### 1.3 Counter file format

For **rocprof v1** the file is a multi-line grammar:

```
pmc : <counter> [<counter> ...]
pmc : <counter> [<counter> ...]   # second pass — kernel will be re-run
range: <start> : <end>             # which dispatches to capture
gpu: 0                              # which device
kernel: <substring filter>
```

Each `pmc:` line is one *pass* over the application; counters are limited per pass by hardware (you get 4–8 PMC slots per shader engine for SQ counters, and similar limits for TCC/TA blocks). If you list more counters than fit, rocprof runs the program multiple times.

For **rocprofv3** the input can be:
- plain text: just `pmc: <counter list>` lines (same multi-pass semantics)
- YAML/JSON: full config with kernel filters, kernel iteration ranges, output format, ATT options
- Or pass counters directly via `--pmc <list>` on the command line.

### 1.4 Expected output snippet (v3 stats)

```
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"reduce_kernel",128,4_812_345,37_596,42.7
"matmul_kernel",16,3_201_900,200_118,28.4
"hipMemcpyAsync",256,1_021_004,3_988,9.0
```

A `_pmc.csv` companion will list one row per kernel invocation per counter set.

---

## 2. Useful counters for RDNA2 training kernels

### 2.1 Counter cheat sheet (gfx10 names)

| Counter                 | Block | What it tells you                                      |
|-------------------------|-------|--------------------------------------------------------|
| `SQ_WAVES`              | SQ    | Total waves dispatched. Use as a denominator.          |
| `SQ_INSTS_VALU`         | SQ    | VALU instructions issued (vector math).                |
| `SQ_INSTS_SALU`         | SQ    | SALU instructions issued (scalar/uniform).             |
| `SQ_INSTS_LDS`          | SQ    | LDS (groupshared) instructions.                        |
| `SQ_INSTS_GDS`          | SQ    | GDS instructions (rare on RDNA2 compute).              |
| `SQ_INSTS_FLAT`         | SQ    | FLAT memory ops (generic addr — slow path, want low).  |
| `SQ_INSTS_SMEM`         | SQ    | Scalar memory loads (constant/uniform).                |
| `SQ_INSTS_VMEM`         | SQ    | Vector memory ops (the bulk of HBM traffic).           |
| `SQ_WAIT_INST_LDS`      | SQ    | Cycles waited on LDS — bank conflicts or pressure.     |
| `SQ_ACTIVE_INST_VALU`   | SQ    | Cycles with VALU active. Divide by `GRBM_GUI_ACTIVE` for VALU utilization. |
| `GRBM_GUI_ACTIVE`       | GRBM  | Cycles GPU was processing graphics/compute. Denom.     |
| `GRBM_COUNT`            | GRBM  | Total clock cycles, busy or idle.                      |
| `GPUBusy`               | derived | % of `GRBM_GUI_ACTIVE / GRBM_COUNT`.                |
| `TCC_HIT_sum`           | L2    | L2 cache hits (sum across L2 slices).                  |
| `TCC_MISS_sum`          | L2    | L2 cache misses.                                       |
| `L2CacheHit`            | derived | `TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)`.       |
| `TCP_TOTAL_CACHE_ACCESSES_sum` | L1 (TCP) | L1 vector cache accesses.                  |
| `FetchSize`             | derived | Bytes pulled from HBM (memory traffic in).           |
| `WriteSize`             | derived | Bytes written to HBM.                                |
| `MemUnitBusy`           | derived | % of cycles the memory unit was busy.                |
| `MemUnitStalled`        | derived | % of cycles the memory unit was stalled — high here means HBM-bound. |
| `WriteUnitStalled`      | derived | Stalled on store path.                               |
| `VALUUtilization`       | derived | `SQ_ACTIVE_INST_VALU / SQ_INSTS_VALU` (cycles with active threads vs issued). |
| `VALUBusy`              | derived | % of cycles VALU was busy.                           |
| `LDSBankConflict`       | derived | LDS bank conflict cycles.                            |

For RDNA2 specifically, `SQ_WAVES` is reported per-SIMD; multiply or sum across SIMDs for a chip-level count. The full counter list is exposed by `rocprofv3 --list-avail` and the YAML in `/opt/rocm/share/rocprofiler-sdk/counters/derived_counters.xml` (or `.yaml` in newer builds).

### 2.2 Occupancy on gfx1030

RDNA2 numbers to memorize:

- 40 CUs (Navi 21 full chip; gfx1030 = RX 6800/6800 XT/6900 XT, gfx1031 = RX 6700 XT with 40 CUs)
- 2 SIMDs per CU on RDNA (vs 4 on GCN/CDNA) — so 80 SIMDs per chip
- 1024 VGPRs × 32 lanes per SIMD in wave32 mode (so 32 KiB/SIMD)
- Max 16 wavefronts per SIMD = 32 waves per CU
- 64 KiB LDS per workgroup (shared across 2 SIMDs of the same CU)
- 40 CUs × 32 waves = 1280 waves maximum in flight

**Occupancy formula (wave32):**

```
max_waves_per_simd = min(
    16,
    floor(1024 / vgpr_per_wave),
    floor(800  / sgpr_per_wave),     # 800 SGPRs per SIMD on gfx10
    floor(workgroup_lds_budget / lds_per_wg)
)
```

A kernel using 32 VGPRs gets the full 16 waves/SIMD. At 64 VGPRs you drop to 16 (still capped), at 80 you drop to 12, at 128 you drop to 8 (50%). For wave64 the same VGPR file is used but each wave consumes 2× registers, so the cap is 8 waves/SIMD.

**Decision rules from counters:**

| Symptom                               | Counter signature                         | Likely cause                          |
|---------------------------------------|-------------------------------------------|---------------------------------------|
| Kernel is slow but VALU is idle       | `VALUBusy` < 30 %, `MemUnitStalled` > 40 %| HBM-bound — fetch-bound               |
| VALU busy but VALU utilization low    | `VALUBusy` > 80 %, `VALUUtilization` < 60 %| Divergence — many lanes masked        |
| Many waves but memory-bound           | `SQ_WAVES`/`MAX` > 0.8, `L2CacheHit` < 30 %| Cache-thrashing access pattern        |
| LDS-heavy but slow                    | `LDSBankConflict` > 20 %                  | Bank conflicts in LDS layout          |
| Lots of FLAT instructions             | `SQ_INSTS_FLAT` ≈ `SQ_INSTS_VMEM`         | Generic-addressed pointers; convert to typed buffer / `__global__` |
| VGPR spills                           | Compiler `-Rpass-analysis` shows `Spills`  | Register pressure → manual reduction  |

A healthy training kernel target on RDNA2: `VALUBusy` ≥ 70 %, `VALUUtilization` ≥ 80 %, `L2CacheHit` ≥ 50 %, `SQ_INSTS_FLAT` < 5 % of `SQ_INSTS_VMEM`, occupancy ≥ 8 waves/SIMD.

---

## 3. rocprof-compute (Omniperf) and rocprof-sys (Omnitrace)

### 3.1 What each solves

- **rocprof-compute** = per-kernel performance counter analysis with roofline. It runs `rocprofv3` under the hood with a curated counter set (often hundreds of counters across multiple passes), then post-processes into a hierarchical "speed-of-light" report: System SoL → block SoL (CU, L1, L2, HBM) → memory chart → roofline.
- **rocprof-sys** = system-level / whole-program trace. Combines CPU sampling, MPI, HIP API, kernel events, NIC, and rocm-smi sampling into a single timeline (Perfetto / OTF2). Use it to find *where* time goes across the application, not inside one kernel.

### 3.2 Setup

```sh
# Install (Ubuntu/Debian on ROCm 7.x)
sudo apt install rocprofiler-compute rocprofiler-systems

# rocprof-sys uses an LD_PRELOAD'd instrumentation library + dynamic instrumentation
# of the binary. Two flows:

# (a) Runtime-only sampling: no binary instrumentation, just preload
rocprof-sys-run -- ./prog

# (b) Binary instrumentation: rewrites or wraps the binary
rocprof-sys-instrument -o prog.inst -- ./prog
rocprof-sys-run -- ./prog.inst
```

`rocprof-compute` workflow has three phases:

```sh
# 1. Profile (collects all counters across multiple passes — slow)
rocprof-compute profile -n my_run -- ./prog

# 2. Analyze — produces text or interactive web output
rocprof-compute analyze -p workloads/my_run/MI200/

# 3. Optional: spin up a local web UI on :8050
rocprof-compute analyze -p workloads/my_run/MI200/ --gui
```

### 3.3 Flame graphs and roofline plots

`rocprof-sys` writes a Perfetto trace (`.proto`) you open at https://ui.perfetto.dev or the local `rocprof-compute-viewer`. Flame graphs come from the CPU-side sampling component.

`rocprof-compute` produces a roofline PDF/PNG combining peak FP32/FP16 throughput vs HBM bandwidth, with each kernel plotted by arithmetic intensity. Rooflines on the Instinct path use derived counters; on RDNA2 consumer cards some counters are missing or differ, so the produced roofline can be incomplete.

### 3.4 Status on RDNA2 in ROCm 7.x

Both tools target Instinct (CDNA) primarily. On RDNA2 (gfx1030/1031):

- `rocprof-sys` works fine — it is just a tracer on top of `rocprofiler-sdk`.
- `rocprof-compute` runs but several panels (notably the roofline and L2 channel charts) are gated to MI-series devices. You will get the "Speed-of-Light" CU/VALU panels and HBM/L2 hit-rate panels, but the architecture-specific tuning advisors are disabled. Expect "panel skipped: arch not supported" messages for parts of the report.

**Decision rule:** for RDNA2, prefer `rocprofv3` + a hand-maintained counter list and read the CSVs. Use `rocprof-compute` only if you have an MI-series box for the same kernel.

---

## 4. Radeon GPU Profiler (RGP)

RGP is a DirectX 12 / Vulkan / OpenCL / HIP profiler from the GPUOpen team. It is the only tool that gives a *driver-level* visualization of dispatches with per-CU occupancy, wavefront timeline, and barrier analysis on consumer cards.

- **Linux capture path:** RGP takes its capture from the `amdgpu`/`amdvlk` or `RADV` Vulkan driver (driver versions ≥ 25.10 work; older Mesa builds do not expose the SQTT interface). For HIP applications on Linux, the capture is harder — typically you wrap with the AMD GPU developer tools driver and use Radeon Developer Panel to trigger the capture. This is fiddly on RDNA2 Linux compared to Windows.
- **Pure compute capture:** since RGP 2.4, console-based pure-compute apps (no swapchain present) can be profiled on Vulkan. For HIP this is still indirect; AQL-only Rust submitters generally need to be wrapped in a small Vulkan compute app or profiled via SQTT decoded by `rocprofv3 --att` instead.
- **What you get:** event timeline, per-shader/per-dispatch occupancy bar, instruction-level latency histogram, wavefront occupancy chart per CU, barrier analysis, pipeline state.
- **Limitations on RDNA2 Linux:** capture support is partial; Vulkan path is well-tested, the HIP path requires the proprietary stack. Many users on Linux RX 6000 fall back to `rocprofv3 --att` plus `rocprof-compute-viewer` rather than RGP.

**Decision rule:** if you ship a Vulkan compute fallback, RGP is the best Linux option for shader-level profiling; if you only have HIP/HSA, use `rocprofv3 --att`.

---

## 5. Kernel disassembly and inspection

### 5.1 Compile to a standalone code object

```sh
# Produce a .hsaco for one architecture
hipcc --offload-arch=gfx1030 -O3 --genco -o k.hsaco k.hip

# Or via clang directly
clang++ -x hip --offload-arch=gfx1030 -O3 --cuda-device-only \
        -nogpulib -emit-llvm -c -o k.bc k.hip
```

### 5.2 Disassemble with llvm-objdump

```sh
llvm-objdump -d --mcpu=gfx1030 k.hsaco | head -80
```

Expected snippet for an RDNA2 build:

```
Disassembly of section .text:

<reduce_kernel>:
        s_load_dwordx2 s[0:1], s[4:5], 0x0           // load arg buffer ptr
        s_waitcnt lgkmcnt(0)
        v_lshlrev_b32_e32 v1, 2, v0
        global_load_dword v2, v1, s[0:1]             // strided global load
        s_waitcnt vmcnt(0)
        v_add_f32_e32 v2, v2, v3
        ds_write_b32 v0, v2                          // LDS write
        s_waitcnt lgkmcnt(0)
        s_barrier
        ...
        s_endpgm
```

Useful flags:
- `-d` disassemble executable sections
- `--mcpu=gfx1030` so you get the right RDNA2 mnemonics (defaults can be wrong)
- `--symbolize-operands` annotates branch targets
- `--offloading` extracts the device portion of a host+device fat binary
- `--source` interleaves with source if `-g` was used
- `--print-imm-hex` for hex literals

### 5.3 Inspect the offload bundle

In ROCm 7, `roc-obj-ls` and `roc-obj-extract` were removed. The replacement is:

```sh
# List the device images embedded in a fat binary
llvm-objdump --offloading prog
# Output:
# OFFLOADING IMAGE [0]:
# kind            = hipv4
# arch            = gfx1030
# triple          = amdgcn-amd-amdhsa
# producer        = clang
# ...

# Extract them to separate files (the tool writes .bundle / .img files alongside)
clang-offload-bundler --type=o --input=prog --unbundle \
                     --targets=hipv4-amdgcn-amd-amdhsa--gfx1030 \
                     --output=prog.gfx1030.hsaco
```

### 5.4 Read the AMDGPU metadata note

```sh
readelf -n k.hsaco
```

Expected output:

```
Displaying notes found in: .note
  Owner                Data size    Description
  AMDGPU               0x00000abc   NT_AMDGPU_METADATA (msgpack)
  AMDGPU               0x00000018   NT_AMD_HSA_ISA_NAME (amdgcn-amd-amdhsa--gfx1030)
```

The `NT_AMDGPU_METADATA` note is a MessagePack-encoded map. Pretty-print it with:

```sh
llvm-readelf --notes k.hsaco
# or
llvm-objdump --offloading --arch-name=amdgcn k.hsaco
```

Key fields you will see for each kernel:
- `.kernel_name`
- `.symbol`
- `.kernarg_segment_size`, `.kernarg_segment_align`
- `.group_segment_fixed_size` (LDS)
- `.private_segment_fixed_size` (scratch)
- `.sgpr_count`, `.vgpr_count`, `.agpr_count` (always 0 on RDNA2)
- `.sgpr_spill_count`, `.vgpr_spill_count`
- `.max_flat_workgroup_size`
- `.wavefront_size` (32 or 64)
- `.args`: full argument list with offsets, types, address spaces

The Rust HSA AQL submitter must read this msgpack to fill the AQL packet correctly (kernarg layout, group/private segment sizes, the kernel object handle). If you're rolling your own loader, this is the canonical source of truth.

### 5.5 Radeon GPU Analyzer (rga) CLI

`rga` is an *offline* compiler+analyzer. You feed it source (HIP/OpenCL/GLSL/HLSL/SPIR-V); it emits ISA, performance statistics, live-VGPR analysis, and occupancy estimates *without running on a GPU*. Particularly valuable on a CI box without an AMD card.

```sh
# List supported GPUs (gfx1030/1031 are present)
rga --list-asics

# Compile an OpenCL kernel for gfx1030 and dump ISA + statistics
rga -s opencl -c gfx1030 \
    --isa my.isa.txt \
    --livereg my.livereg.txt \
    --analysis my.stats.csv \
    kernel.cl

# Compile a Vulkan compute SPIR-V for gfx1030
rga -s vulkan -c gfx1030 --isa my.isa.txt --analysis my.stats.csv \
    --pipeline-stage cs comp.spv
```

The `analysis` CSV has rows like:

```
KernelName,ScratchMemory,ThreadsPerWorkgroup,WavefrontSize,UsedVGPRs,UsedSGPRs,LDSByteSize,Occupancy
my_kernel,0,256,32,40,32,16384,12 waves/SIMD (75%)
```

The `livereg` output is a per-line count of live VGPRs through the kernel — invaluable for finding the hot spot driving register pressure.

**Inferring occupancy from register count (RDNA2 wave32):**

| VGPRs | Max waves/SIMD | Occupancy of theoretical 16 |
|-------|----------------|-----------------------------|
| ≤ 64  | 16             | 100 %                       |
| 65–80 | 12             | 75 %                        |
| 81–96 | 10             | 62 %                        |
| 97–128| 8              | 50 %                        |
| 129–168| 6             | 37 %                        |
| 169–256| 4             | 25 %                        |

The same drop applies if SGPRs > 64 (granularity differs) or LDS-per-workgroup is very high.

---

## 6. Compiler resource reports

### 6.1 -Rpass-analysis=kernel-resource-usage

This is the cheapest, fastest way to know what your kernel costs. It runs at compile time and emits a remark per kernel.

```sh
hipcc --offload-arch=gfx1030 -O3 -c k.hip \
      -Rpass-analysis=kernel-resource-usage 2>&1 | tee k.report.txt
```

Expected output:

```
remark: k.hip:42:0: Function Name: _Z13reduce_kernelPfS_i
remark: k.hip:42:0:     SGPRs: 40
remark: k.hip:42:0:     VGPRs: 56
remark: k.hip:42:0:     AGPRs: 0
remark: k.hip:42:0:     ScratchSize [bytes/lane]: 0
remark: k.hip:42:0:     Occupancy [waves/SIMD]: 16
remark: k.hip:42:0:     SGPRs Spill: 0
remark: k.hip:42:0:     VGPRs Spill: 0
remark: k.hip:42:0:     LDS Size [bytes/block]: 8192
```

Read VGPR spills and Scratch first — non-zero numbers there mean you blew the register file and the compiler is spilling to private memory, which on RDNA2 is HBM-backed scratch and an order of magnitude slower than registers.

### 6.2 --save-temps

```sh
hipcc --offload-arch=gfx1030 -O3 --save-temps -c k.hip
ls k-hip-*
# k-hip-amdgcn-amd-amdhsa-gfx1030.bc       <- post-frontend, before opt
# k-hip-amdgcn-amd-amdhsa-gfx1030.ll       <- LLVM IR, human-readable
# k-hip-amdgcn-amd-amdhsa-gfx1030.s        <- AMDGPU assembly
# k-hip-amdgcn-amd-amdhsa-gfx1030.o        <- ELF
# k-hip-amdgcn-amd-amdhsa-gfx1030.hsaco
```

Caveat: with `--save-temps` the compiler implicitly passes `-disable-llvm-passes` so the `.ll` you see is unoptimized. To inspect the optimized IR add `-mllvm -print-after-all` or compile with `-O3 -emit-llvm -S` *without* `--save-temps`.

### 6.3 Reading the YAML metadata embedded in .hsaco

Same as §5.4. You can also produce a YAML dump directly with:

```sh
llvm-readobj --elf-output-style=GNU --notes k.hsaco
# or full JSON via llvm-objcopy --dump-section .note=- k.hsaco
```

---

## 7. rocgdb — debugging GPU kernels

`rocgdb` is a fork of GDB with AMDGPU support added. It can stop the host process *and* GPU waves, set breakpoints in device code, inspect VGPRs/SGPRs of a stopped wave, and step instruction-by-instruction.

### 7.1 Compile for debugging

```sh
hipcc -ggdb -O0 -fno-omit-frame-pointer --offload-arch=gfx1030 prog.hip -o prog
# or for partial optimization but still inspectable:
hipcc -ggdb -O2 --offload-arch=gfx1030 prog.hip -o prog
```

The `-g` flag automatically enables `-mllvm -amdgpu-spill-cfi-saved-regs` for correct DWARF unwinding; do not strip it.

### 7.2 Driving rocgdb

```sh
rocgdb ./prog
(rocgdb) break my_kernel
(rocgdb) run
# (host stops on entry to my_kernel; GPU waves are now schedulable)
(rocgdb) info threads
#  Id   Target Id            Frame
#  1    Thread 0x... (LWP …)  main () at host.cpp:42
#  2    AMDGPU Wave 1:1:1:0/0 my_kernel (...) at k.hip:13
#  3    AMDGPU Wave 1:1:1:0/1 my_kernel (...) at k.hip:13
(rocgdb) thread 2
(rocgdb) info args
(rocgdb) info locals
(rocgdb) print v[0:31]@4    # print 4 VGPR lanes
(rocgdb) info registers     # dumps SGPR/VGPR state of the current wave
(rocgdb) x/8wx $v0          # treat v0 as 8 dwords
(rocgdb) next               # source-level step
(rocgdb) stepi              # instruction step
```

### 7.3 Limitations (RDNA2)

- **No SR-IOV** debugging on gfx1030/1031/1032.
- **`s_clause` blocks** — on gfx10/11/12 do not place a breakpoint inside a clause; rocgdb will emit "cannot insert breakpoint inside s_clause group". Place it before or after the clause.
- **Optimized code skips** — at `-O3` many lines are folded; expect "value optimized out" for half the locals.
- **Single-stepping** is per-wave; the other 31 lanes of the wave step together. You can mask via `info threads`/`thread <id>`.
- **No watchpoints on global memory** from the GPU side; watchpoints work on host memory only.

For a focused VGPR dumper that doesn't require interactive driving, the `vosen/amdgpu_debug` helper crate provides a `print_vector_register` function callable from a stopped breakpoint.

---

## 8. Memory tools — sanitizers

ROCm ships an AddressSanitizer build of the runtime (a.k.a. "rocsan" / "hipsanitizer", though the canonical name is just AMDGPU AddressSanitizer). It detects out-of-bounds and use-after-free in *device* memory, mirroring the CPU `-fsanitize=address` model.

```sh
# Compile with both host and device sanitizers
hipcc -fsanitize=address -shared-libsan -g \
      --offload-arch=gfx1030:xnack+ \
      prog.hip -o prog

# Run with the sanitized runtime preloaded
HSA_XNACK=1 ./prog
```

Notes:
- `xnack+` is required because ASan needs to intercept page faults on shadow memory; this halves available occupancy on Instinct and roughly the same on RDNA2.
- On consumer RDNA2, xnack support is limited; many distributions disable it. If `HSA_XNACK=1` fails with "page-migration not supported", you can still build the host side with ASan but device-side detection is best-effort.
- Output looks like LLVM ASan reports with an extra "GPU buffer" line indicating the offending wave/lane.

For race detection (LDS data races, missing barriers), there is no first-party ROCm tool comparable to `racecheck`. The closest is the HPE/Cray `sanitizers4hpc` wrapper (only on Cray installs) and manual `__threadfence_block()` audits. On RDNA2 you generally rely on rocgdb + targeted unit tests.

---

## 9. HIP-level debugging hooks

These are the runtime knobs you set without recompiling.

| Variable                 | Effect                                                                 |
|--------------------------|------------------------------------------------------------------------|
| `HIP_LAUNCH_BLOCKING=1`  | Every `hipLaunchKernelGGL` becomes synchronous; great for isolating which kernel asserts/segfaults. |
| `HIP_VISIBLE_DEVICES=0`  | Restrict process to a subset of devices.                              |
| `AMD_SERIALIZE_KERNEL=3` | 0=off, 1=before, 2=after, 3=both. Forces `hipDeviceSynchronize` around launches. |
| `AMD_SERIALIZE_COPY=3`   | Same for `hipMemcpy*`.                                                |
| `AMD_LOG_LEVEL=4`        | 0–5 verbosity. 4 = debug, 5 = extra debug.                            |
| `AMD_LOG_MASK=0xFFFF...` | Bit mask of categories (HSA, COPY, KERN, MEM, …). `0xFFFFFFFF` = all. |
| `HSA_TOOLS_LIB=...`      | Comma-separated list of `.so`s loaded into the HSA runtime — this is the official hook for *your own* AQL/queue interceptor to be picked up before `hsa_init`. |
| `HSA_ENABLE_INTERRUPT=0` | Use polling instead of interrupts for completion signals; sometimes lower latency for tight benchmarks. |
| `HSA_XNACK=1`            | Enable on-demand page faulting for managed/shared VM.                 |
| `HIP_TRACE_API=1`        | Print every HIP API call — analogous to `cuda-gdb` API tracing.       |

`AMD_LOG_LEVEL=4` produces lines like:

```
:3:rocdevice.cpp           :2755: 7321 us: 12345: [tid:0x7f...] hipLaunchKernel ( ... )
:3:hip_module.cpp          :0612: 7325 us: 12345: [tid:0x7f...] kernel "_Z13reduce..." gridDim={64,1,1} blockDim={256,1,1}
:4:command.cpp             :0379: 7329 us: 12345: [tid:0x7f...] command 0x... enqueued, signal 0x...
:4:rocvirtual.cpp          :0729: 7334 us: 12345: [tid:0x7f...] HSA AQL kernel dispatch packet ...
```

`HSA_TOOLS_LIB` is the cleanest way to insert telemetry from a Rust submitter: build a small `cdylib` that exports `OnLoad`/`OnUnload`/`hsa_runtime_load_tool` and intercepts queue creation. The runtime calls each library in order before `hsa_init` returns.

---

## 10. Validation strategy for a Rust HIP+HSA project

Your stack uses both HIP-launched kernels (for ergonomics in tests) and direct HSA AQL dispatch (for production). The validation ladder:

1. **CPU reference vs HIP launch (already in `microcolumn_gpu` tests).** Use a tight tolerance for elementwise ops (`< 1e-6` rel for f32 multiplies) and a looser one for reductions (`< 1e-4` rel for sums of ~10⁴ elements, since reduction order changes summation).
2. **HIP launch vs HSA AQL launch — bit-exact.** If both invoke the same compiled code object with the same kernarg layout, output must match byte-for-byte. Any difference points to:
   - kernarg layout bug (offsets, alignment) — diff against the msgpack metadata
   - workgroup/grid mismatch — verify against `.max_flat_workgroup_size`
   - missing acquire/release fences in the AQL packet header
   - signal not yet completed when reading the result
3. **Stride-varying f32 reduce.** Write the reduction with stride-1, stride-2, stride-256 input layouts; expected delta `~ N · ε_f32` ≈ `N · 1.2e-7`. Anything bigger means you have a real correctness bug masked as "floating-point noise".
4. **Determinism under `HIP_LAUNCH_BLOCKING=1`.** If results change when launches are serialized, you have a missing dependency edge between streams.
5. **Sanitized build pass** for OOB and uninitialized reads.

---

## 11. Performance experimentation workflow

A four-step loop that fits on one screen:

1. **Correctness baseline (HIP).**
   ```sh
   hipcc -O3 --offload-arch=gfx1030 -Rpass-analysis=kernel-resource-usage prog.hip -o prog
   rocprofv3 --hip-trace --kernel-trace --stats --output-format csv -- ./prog
   ```
   Read `Occupancy [waves/SIMD]` and total kernel time. This is your reference line.

2. **Switch to HSA AQL, validate.**
   Same kernel, your custom dispatcher. Keep the HIP path under a test feature flag and run both in CI; assert byte-equal output. Re-run `rocprofv3 --kernel-trace` and compare durations — they should be within ±2 %; large deltas mean your AQL packets have wrong dependency / fence settings.

3. **Profile launch overhead.** A `hipBench`-style microbenchmark that submits 100 000 trivial kernels (just `*out = 0;`) and measures end-to-end time. On RDNA2 expect ~3–6 µs/launch via HIP; AQL submission with reused signal pools should beat that to ~1.5–2 µs/launch.

4. **Optimize hot kernel.**
   ```sh
   rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM \
             SQ_ACTIVE_INST_VALU TCC_HIT_sum TCC_MISS_sum \
             GRBM_GUI_ACTIVE GRBM_COUNT \
             --kernel-trace -- ./prog
   ```
   Compute `VALUUtilization`, `L2CacheHit`, `MemUnitStalled`. Use the table in §2.2 to pick the next experiment (tiling? coalescing? wave32→wave64?). Rinse, repeat.

---

## 12. Bookmark-this command list

```sh
# ---- Compile and inspect a kernel
hipcc --offload-arch=gfx1030 -O3 --genco -o k.hsaco k.hip
hipcc -O3 -Rpass-analysis=kernel-resource-usage --offload-arch=gfx1030 -c k.hip
hipcc --offload-arch=gfx1030 -O3 --save-temps -c k.hip

# ---- Disassemble and inspect metadata
llvm-objdump -d --mcpu=gfx1030 --symbolize-operands k.hsaco | head -80
llvm-objdump --offloading prog                        # list embedded device images
llvm-readelf --notes k.hsaco                          # AMDGPU msgpack metadata
clang-offload-bundler --type=o --input=prog --unbundle \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx1030 --output=prog.gfx1030.hsaco

# ---- Static analysis (no GPU needed)
rga --list-asics
rga -s opencl -c gfx1030 --isa isa.txt --livereg lr.txt --analysis stats.csv kernel.cl

# ---- Profile (modern path)
rocprofv3 --hip-trace --hsa-trace --kernel-trace --stats \
          --output-format csv -- ./prog
rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM TCC_HIT_sum TCC_MISS_sum \
          --kernel-trace --output-format csv -- ./prog
rocprofv3 -i counters.txt -- ./prog
rocprofv3 --att --att-target-cu 0 --att-simd-select 0xF \
          --att-buffer-size 0x60000000 --kernel-iteration-range 0..0 -- ./prog

# ---- Profile (legacy v1, still works in ROCm 7)
rocprof --stats --hsa-trace --hip-trace ./prog
rocprof -i counters.txt --output-format csv -o counters.csv ./prog

# ---- System-level trace
rocprof-sys-run -- ./prog                             # sample-based
rocprof-sys-instrument -o prog.inst -- ./prog && rocprof-sys-run -- ./prog.inst

# ---- Per-kernel speed-of-light + roofline (limited on RDNA2)
rocprof-compute profile -n my_run -- ./prog
rocprof-compute analyze -p workloads/my_run/<arch>/ --gui

# ---- Debug
HIP_LAUNCH_BLOCKING=1 AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3 ./prog
AMD_LOG_LEVEL=4 ./prog 2> hiplog.txt
rocgdb ./prog
HSA_TOOLS_LIB=./libmytool.so ./prog                   # custom AQL interceptor

# ---- Sanitizer
hipcc -fsanitize=address -shared-libsan -g --offload-arch=gfx1030:xnack+ prog.hip -o prog
HSA_XNACK=1 ./prog
```

---

## 13. References

- ROCprofiler-SDK (rocprofv3) docs — https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/
- rocprofv3 how-to — https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- Thread trace (ATT) — https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html
- rocprof-trace-decoder (open source) — https://github.com/ROCm/rocprof-trace-decoder
- rocprof-compute-viewer — https://github.com/ROCm/rocprof-compute-viewer
- rocprofiler-compute (formerly Omniperf) — https://github.com/ROCm/rocprofiler-compute
- rocprofiler-systems (formerly Omnitrace) — https://github.com/ROCm/rocprofiler-systems
- ROCm profiling tools overview — https://rocm.blogs.amd.com/software-tools-optimization/profilers/README.html
- Radeon GPU Profiler — https://gpuopen.com/rgp/ and https://radeon-gpuprofiler.readthedocs.io/
- Radeon GPU Analyzer (CLI) — https://github.com/GPUOpen-Tools/radeon_gpu_analyzer and https://radeon-gpu-analyzer-rga.readthedocs.io/
- ROCgdb manual — https://rocm.docs.amd.com/projects/ROCgdb/en/latest/
- HIP debugging guide — https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/debugging.html
- LLVM AMDGPU backend (ISA, code-object format, msgpack metadata) — https://llvm.org/docs/AMDGPUUsage.html
- AMD register-pressure lab note — https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-register-pressure-readme/
- "Occupancy explained" (RDNA wave32/64 numbers) — https://gpuopen.com/learn/occupancy-explained/
- HSA Runtime API — https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/api-reference/api.html
- ROCm 7.x release notes (deprecation timeline) — https://rocm.docs.amd.com/en/latest/about/release-notes.html
