# AMDGPU Code Object Internals: A Deep Reference

A self-contained technical reference covering the byte-level layout of AMD GPU
code objects (`.hsaco`), kernel descriptors, the hidden ABI v4/v5, clang offload
bundles, and the loader interfaces. Targeted at people writing their own runtime
on top of the HSA layer — i.e. people who do not want to ship the HIP runtime
and instead drive the device through `libhsa-runtime64.so` directly.

Everything below is verified against `/opt/rocm-7.2.1/include/hsa/` headers and
the LLVM `AMDHSAKernelDescriptor.h` / `AMDGPUUsage.rst` sources. Where a value
depends on the GPU family, RDNA2 (gfx1030/gfx1031/gfx1032 — the user's RX 6750
XT) is the reference target.

---

## 1. What a `.hsaco` actually is

A `.hsaco` file is a **standard ELF64 little-endian object** with AMDGPU-specific
header fields, sections, notes and relocation types. It is *not* a custom
container. Anything that can read ELF64 (`readelf`, `objdump`, the `goblin` or
`object` Rust crates) can walk it.

### 1.1 ELF header invariants

```
e_ident[EI_CLASS]    = ELFCLASS64        (= 2)
e_ident[EI_DATA]     = ELFDATA2LSB       (= 1, little-endian)
e_ident[EI_OSABI]    = ELFOSABI_AMDGPU_HSA = 64
e_ident[EI_ABIVERSION] = 0..4   (V2..V6 — see §2)
e_type               = ET_DYN            (= 3, shared object)
e_machine            = EM_AMDGPU         (= 224 / 0xE0)
e_flags              = (XNACK/SRAMECC bits) | EF_AMDGPU_MACH_AMDGCN_GFXxxxx
```

The `e_flags` low byte is the GPU family selector. From
`/opt/rocm-7.2.1/include/hsa/amd_hsa_elf.h`:

| GFX               | `EF_AMDGPU_MACH_*` |
|-------------------|--------------------|
| gfx906 (Vega 7nm) | 0x02f              |
| gfx908 (MI100)    | 0x030              |
| gfx90a (MI250)    | 0x03f              |
| gfx942 (MI300)    | 0x04c              |
| gfx1030 (Navi 21) | 0x036              |
| gfx1032 (Navi 23) | 0x038              |
| gfx1100 (Navi 31) | 0x041              |
| gfx1200 (Navi 4x) | 0x048              |

For RDNA2 (gfx1030/32) the V5 ELF header begins with this exact magic
sequence:

```
00:  7F 45 4C 46 02 01 01 40   .ELF...@      ; magic, ELF64, LSB, OSABI=AMDGPU_HSA
08:  03 00 00 00 00 00 00 00                 ; ABI version 3 = V5
10:  03 00 E0 00 01 00 00 00   ........      ; e_type=ET_DYN, e_machine=0xE0
18:  ...                                     ; e_entry, e_phoff, e_shoff
                                             ; e_flags = 0x036 (gfx1030)
                                             ;        | 0x300 (XNACK_ON_V4) etc.
```

### 1.2 Sections you will see

| Section             | Purpose                                                     |
|---------------------|-------------------------------------------------------------|
| `.text`             | The actual GPU instructions (S_*, V_*, FLAT_*, BUFFER_*…)   |
| `.rodata`           | Constant pools, string literals                             |
| `.data` / `.bss`    | Mutable globals, zero-init globals                          |
| `.dynsym` / `.dynstr` | Dynamic symbol table — kernels exported here              |
| `.rela.text`        | Relocations applied to `.text` (R_AMDGPU_REL32, ABS64…)     |
| `.rela.dyn`         | Relocations on global data (R_AMDGPU_RELATIVE64, ABS64)     |
| `.note`             | The metadata section — see below                            |
| `.AMDGPU.config`    | Sometimes present, holds early-style register assignments   |

The kernel descriptor (the 64-byte structure described in §3) lives **inside
`.rodata`**, at the address pointed to by the `<kernel_name>.kd` dynamic
symbol. The actual machine code starts at `kd_addr + kernel_code_entry_byte_offset`
which by convention is **256 bytes after the KD** (KD is 64 B, then 192 B of
padding to the 256 B `AMD_ISA_ALIGN_BYTES`).

### 1.3 The `.note` section

There is exactly one note of type `NT_AMDGPU_METADATA = 32` (named `"AMDGPU"`)
that contains a **msgpack** blob. The msgpack root is a map with a single key
`"amdhsa.kernels"` (V3+) holding an array of kernel descriptors. The historical
note types from `amd_hsa_elf.h`:

```c
#define NT_AMD_HSA_CODE_OBJECT_VERSION 1   // V2 only
#define NT_AMD_HSA_HSAIL               2
#define NT_AMD_HSA_ISA_VERSION         3   // V2: human-readable text
#define NT_AMD_HSA_PRODUCER            4
#define NT_AMD_HSA_PRODUCER_OPTIONS    5
#define NT_AMD_HSA_EXTENSION           6
#define NT_AMD_HSA_ISA_NAME            11  // V3: text
// V3+:
#define NT_AMDGPU_METADATA             32  // msgpack
```

`readelf -n kernel.hsaco` decodes V2 notes (text/YAML) but emits raw hex for
V3+ msgpack. The proper readers are `llvm-readobj --notes --elf-output-style=GNU`
which auto-decodes msgpack into YAML, and `roc-obj-ls -v`.

A typical V5 metadata YAML for a single kernel looks like:

```yaml
amdhsa.kernels:
  - .name:          'matmul_bf16'
    .symbol:        'matmul_bf16.kd'
    .kernarg_segment_size: 312          # user_args_size + hidden_args_size, 8B-aligned
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .sgpr_count:    24
    .vgpr_count:    32
    .max_flat_workgroup_size: 256
    .args:
      - .name: A   .size: 8 .offset:   0 .value_kind: global_buffer .address_space: global
      - .name: B   .size: 8 .offset:   8 .value_kind: global_buffer
      - .name: C   .size: 8 .offset:  16 .value_kind: global_buffer
      - .name: M   .size: 4 .offset:  24 .value_kind: by_value
      # ... padding to 8-byte align, then hidden args:
      - .size: 4 .offset: 256 .value_kind: hidden_block_count_x
      - .size: 4 .offset: 260 .value_kind: hidden_block_count_y
      - .size: 4 .offset: 264 .value_kind: hidden_block_count_z
      - .size: 2 .offset: 268 .value_kind: hidden_group_size_x
      # ... etc
amdhsa.version: [1, 2]                 # V5 = (1,2). V4 = (1,1). V3 = (1,0)
amdhsa.target:  'amdgcn-amd-amdhsa--gfx1030'
```

**The `.args` table is the contract between host and device.** The runtime
reads `.offset` and `.size` and writes user values exactly there. The hidden
args also appear in this table — their `.offset` is past the user args.

### 1.4 Inspection tools

```bash
roc-obj-ls -v kernel.hsaco              # lists kernels + sgpr/vgpr/lds usage
llvm-objdump -d --mcpu=gfx1030 kernel.hsaco   # disassembles .text
llvm-readobj --elf-output-style=GNU --notes kernel.hsaco   # decodes msgpack
readelf -h kernel.hsaco                 # sanity-check the header
```

`llvm-objdump` requires the `--mcpu` to match the `e_flags` GFX field — if you
pass `gfx900` to a gfx1030 file you will get garbage instructions or `<unknown>`.

---

## 2. Code object versions

The version is encoded in **`e_ident[EI_ABIVERSION]`** (the 9th byte of the
ELF header) AND in the msgpack `amdhsa.version` array in the note. They must
agree.

| ABI byte | Version  | Notes (`amdhsa.version`) | Status                     |
|----------|----------|--------------------------|----------------------------|
| —        | V1       | (legacy HSAIL/Finalizer) | dead, removed              |
| 0        | V2       | YAML in `.note`          | dead, dropped in LLVM 14   |
| 1        | V3       | msgpack `[1, 0]`         | LLVM 9..13 default         |
| 2        | V4       | msgpack `[1, 1]`         | LLVM 14..15 default        |
| 3        | V5       | msgpack `[1, 2]`         | LLVM 16+ / **ROCm 6.x–7.x default** |
| 4        | V6       | msgpack `[1, 3]`         | LLVM 21+; "generic targets" support |

What each version added:

- **V3**: msgpack metadata replaces YAML; `kernel_descriptor_t` becomes the
  authoritative 64-byte struct (instead of the 256-byte legacy
  `amd_kernel_code_t`). `<kernel>.kd` symbol convention.
- **V4**: target ID encoding moves from `e_flags` bits into a string in
  the metadata; XNACK/SRAMECC become `_V4` two-bit selectors with explicit
  `ANY/OFF/ON/UNSUPPORTED`. Hidden args reorganized — old hidden-args block
  was 88 bytes, V4 keeps the same layout but adds `hidden_heap_v1`.
- **V5**: **complete redesign of hidden args** (see §5). Adds
  `hidden_block_count_*` (replaces "global size / workgroup size" loads with
  a precomputed grid-block count), `hidden_group_size_*`, `hidden_remainder_*`,
  `hidden_dynamic_lds_size`. Drops `hidden_global_offset_*` (OpenCL-specific)
  by default. Aligns kernarg buffer to 256 B. Total size shrinks to 56 B.
- **V6**: "generic" target IDs (e.g. `gfx10-3-generic` instead of a specific
  gfx1030/31/32/35); top byte of `e_flags` carries a `EF_AMDGPU_GENERIC_VERSION`.

**Default of `hipcc` on ROCm 7.2.x:** V5. Override via:

```bash
hipcc -mcode-object-version=4 -O3 --offload-arch=gfx1030 --genco kernel.hip -o k.hsaco
hipcc -mcode-object-version=none ...   # use minimum required for the target
```

To inspect: `readelf -h kernel.hsaco | grep ABI` shows
`OS/ABI: <unknown: 40>` (40 hex = 64 dec = `ELFOSABI_AMDGPU_HSA`) and
`ABI Version: 3` for V5.

---

## 3. The kernel descriptor (64 bytes)

When you call `hsa_executable_get_symbol_by_name(exe, "foo.kd", agent, &sym)`
and then `hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
&kernel_object)`, the `u64` you get back is the **device VA of a 64-byte
kernel_descriptor_t**. The HSA AQL kernel-dispatch packet's `kernel_object`
field is exactly this pointer.

From LLVM `llvm/include/llvm/Support/AMDHSAKernelDescriptor.h` (current as of
LLVM 21):

```c
struct kernel_descriptor_t {
  uint32_t group_segment_fixed_size;     // [ 0..3 ]  bytes of LDS
  uint32_t private_segment_fixed_size;   // [ 4..7 ]  scratch per workitem
  uint32_t kernarg_size;                 // [ 8..11]  total kernarg bytes (V5)
  uint8_t  reserved0[4];                 // [12..15]
  int64_t  kernel_code_entry_byte_offset;// [16..23]  +offset from KD to .text
  uint8_t  reserved1[20];                // [24..43]
  uint32_t compute_pgm_rsrc3;            // [44..47]  GFX10+ / GFX90A
  uint32_t compute_pgm_rsrc1;            // [48..51]
  uint32_t compute_pgm_rsrc2;            // [52..55]
  uint16_t kernel_code_properties;       // [56..57]
  uint16_t kernarg_preload;              // [58..59]  GFX940+ kernarg preload
  uint8_t  reserved3[4];                 // [60..63]
};
// total = 64 bytes
```

Note `kernarg_size` is **only present in V5** — V3/V4 had reserved bytes
there. The runtime reads it in V5 to allocate a kernarg buffer of the
correct size; in V4 the runtime reads `.kernarg_segment_size` from the note
metadata instead.

### 3.1 `compute_pgm_rsrc1` (offset 48, 32 bits)

This word is loaded into a hardware MMIO register at dispatch time and
controls the wave's resource budget.

| Bits  | Field                              | Notes                          |
|-------|------------------------------------|--------------------------------|
| 0–5   | GRANULATED_WORKITEM_VGPR_COUNT     | (vgpr_count/4)-1 GFX6-9; (vgpr_count/8)-1 GFX10+ wave32; (vgpr_count/4)-1 wave64 |
| 6–9   | GRANULATED_WAVEFRONT_SGPR_COUNT    | (sgpr_count/8)-1 (GFX6-9). GFX10+ ignored — SGPRs are flat |
| 10–11 | PRIORITY                           | 0–3                            |
| 12–13 | FLOAT_ROUND_MODE_32                | 0 = nearest_even               |
| 14–15 | FLOAT_ROUND_MODE_16_64             |                                |
| 16–17 | FLOAT_DENORM_MODE_32               | 3 = no flush (IEEE)            |
| 18–19 | FLOAT_DENORM_MODE_16_64            | 3 = no flush                   |
| 20    | PRIV                               | privileged execution           |
| 21    | ENABLE_DX10_CLAMP (GFX6-9)         | unused on GFX10+               |
| 22    | DEBUG_MODE                         |                                |
| 23    | ENABLE_IEEE_MODE (GFX6-9)          | unused on GFX10+               |
| 24    | BULKY                              |                                |
| 25    | CDBG_USER                          | conditional debug              |
| 26    | FP16_OVFL (GFX9+)                  |                                |
| 29    | WGP_MODE (GFX10+)                  | 0 = CU mode, 1 = WGP mode (RDNA pairs CUs) |
| 30    | MEM_ORDERED (GFX10+)               | 1 = waitcnt-ordered memops     |
| 31    | FWD_PROGRESS (GFX10+)              | 1 = guarantees forward progress |

For RDNA2 (gfx1030/32) you almost always want `WGP_MODE=0, MEM_ORDERED=1,
FWD_PROGRESS=1`. The wave32 path sets `kernel_code_properties.ENABLE_WAVEFRONT_SIZE32=1`.

### 3.2 `compute_pgm_rsrc2` (offset 52, 32 bits)

| Bits  | Field                               | Notes                                |
|-------|-------------------------------------|--------------------------------------|
| 0     | ENABLE_PRIVATE_SEGMENT              | enable scratch                       |
| 1–5   | USER_SGPR_COUNT                     | how many SGPRs the runtime preloads (0..16) |
| 6     | ENABLE_TRAP_HANDLER                 |                                      |
| 7     | ENABLE_SGPR_WORKGROUP_ID_X          | system SGPR for wg.x                 |
| 8     | ENABLE_SGPR_WORKGROUP_ID_Y          |                                      |
| 9     | ENABLE_SGPR_WORKGROUP_ID_Z          |                                      |
| 10    | ENABLE_SGPR_WORKGROUP_INFO          |                                      |
| 11–12 | ENABLE_VGPR_WORKITEM_ID             | 0=X, 1=X+Y, 2=X+Y+Z                  |
| 13    | EXCEPTION_ADDRESS_WATCH             |                                      |
| 14    | EXCEPTION_MEMORY_VIOLATION          |                                      |
| 15–23 | GRANULATED_LDS_SIZE                 | rounded units of 128 dwords          |
| 24–30 | IEEE-754 / int exception enables    | inv_op, fp_denorm, div0, ovf, unf, inex, int_div0 |

### 3.3 `compute_pgm_rsrc3` (offset 44, 32 bits)

GPU-family-specific. **Crucially: this word is only consumed on GFX10+ and
GFX90A+. On GFX9 it must be zero.**

| Family       | Field                                | Bits  |
|--------------|--------------------------------------|-------|
| GFX90A       | ACCUM_OFFSET                         | 0–5   |
| GFX90A       | TG_SPLIT                             | 16    |
| GFX10/11     | SHARED_VGPR_COUNT (wave32 only)      | 0–3   |
| GFX11        | INST_PREF_SIZE                       | 4–9   |
| GFX12+       | INST_PREF_SIZE                       | 4–11  |
| GFX12+       | GLG_EN (global LDS group)            | 13    |
| GFX125+      | NAMED_BAR_CNT                        | 14–16 |
| GFX125+      | TCP_SPLIT                            | 18–20 |

For RDNA2 (gfx1030/32), `compute_pgm_rsrc3 = 0` is correct.

### 3.4 `kernel_code_properties` (offset 56, 16 bits)

These bits tell the runtime which "user SGPRs" the kernel expects to be
preloaded before entry. The runtime fills the corresponding SGPRs from the
AQL packet.

| Bit | Field                                       |
|-----|---------------------------------------------|
| 0   | ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER          |
| 1   | ENABLE_SGPR_DISPATCH_PTR                    |
| 2   | ENABLE_SGPR_QUEUE_PTR                       |
| 3   | ENABLE_SGPR_KERNARG_SEGMENT_PTR             |
| 4   | ENABLE_SGPR_DISPATCH_ID                     |
| 5   | ENABLE_SGPR_FLAT_SCRATCH_INIT               |
| 6   | ENABLE_SGPR_PRIVATE_SEGMENT_SIZE            |
| 10  | ENABLE_WAVEFRONT_SIZE32 (GFX10+)            |
| 11  | USES_DYNAMIC_STACK                          |

The sum of bits 0..6 set, multiplied by their natural SGPR width (most are
2 SGPRs = 64-bit pointer), must equal `USER_SGPR_COUNT` in
`compute_pgm_rsrc2` bits 1–5. **Mismatch is the single most common reason a
kernel page-faults at entry.**

### 3.5 Reference headers

In `/opt/rocm-7.2.1/include/hsa/`:
- `amd_hsa_kernel_code.h` — the legacy 256-byte `amd_kernel_code_t` (used by
  V2 and as historical reference for what each bit means).
- `amd_hsa_elf.h` — the ELF constants quoted above (`EM_AMDGPU=224`,
  `ELFOSABI_AMDGPU_HSA=64`, `ELFABIVERSION_AMDGPU_HSA_V5=3`, all the
  `EF_AMDGPU_MACH_*`).
- `amd_hsa_common.h` — the bit-field macros.

In LLVM:
- `llvm/include/llvm/Support/AMDHSAKernelDescriptor.h` — the canonical V3+
  64-byte `kernel_descriptor_t`.
- `llvm/lib/Target/AMDGPU/AMDKernelCodeT.h` — the bit-field enums for rsrc1/2/3.

---

## 4. The clang offload bundle wrapper

`hipcc --genco kernel.hip -o kernel.hsaco` does **not** emit a raw ELF.
It emits a **clang offload bundle** that contains one or more embedded ELFs
plus an empty "host" placeholder. Even if you only target one GPU, the bundle
wrapper is always there. The HSA runtime cannot load a bundle — you must
unwrap it first.

### 4.1 Bundle on-disk layout

```
+-----------------------------------------+
| "__CLANG_OFFLOAD_BUNDLE__"  (24 bytes)  |   magic, no null terminator
+-----------------------------------------+
| u64  num_entries                        |   little-endian
+-----------------------------------------+
| Entry 0:                                |
|   u64  code_object_offset               |   absolute file offset
|   u64  code_object_size                 |   in bytes
|   u64  bundle_entry_id_length           |
|   char bundle_entry_id[id_length]       |   no null terminator
+-----------------------------------------+
| Entry 1:                                |
|   ... (same shape)                      |
+-----------------------------------------+
| ...                                     |
+-----------------------------------------+
| code_object_0  (raw ELF bytes)          |   at offset entry0.code_object_offset
| code_object_1                           |
| ...                                     |
+-----------------------------------------+
```

There is **no padding between the entry list and the embedded objects** by
default. Newer clang accepts `--bundle-align=N` to force ELFs onto N-byte
boundaries (so they can be `mmap`-ed), but the default produces tight
packing. Your reader must trust the offsets.

### 4.2 Bundle entry IDs

The format is `<offload-kind>-<target-triple>[-<target-id>]`. For HIP/AMDGPU:

```
host-x86_64-unknown-linux-gnu                     ; placeholder, often size 0
hip-amdgcn-amd-amdhsa--gfx906                     ; pre-V4
hipv4-amdgcn-amd-amdhsa--gfx1030                  ; V4 ABI
hipv4-amdgcn-amd-amdhsa--gfx1030:xnack-           ; V4 with target-feature
hipv4-amdgcn-amd-amdhsa--gfx10-3-generic          ; V6 generic
openmp-amdgcn-amd-amdhsa--gfx906                  ; OpenMP offload
```

Note the **double dash** `--` between `amdhsa` and the GFX name — that is
because the triple is `<arch>-<vendor>-<sys>-<env>` with `env` empty.

### 4.3 Extraction in Rust (verbatim from `hsa_dispatch.rs:428`)

```rust
/// Parse Clang Offload Bundle to extract the amdgcn ELF for our GPU target.
/// Bundle layout:
///   "__CLANG_OFFLOAD_BUNDLE__" (24 bytes)
///   u64 num_entries
///   per entry: u64 offset, u64 size, u64 id_len, [u8; id_len] id_string
fn extract_amdgcn_elf(bundle: &[u8]) -> Option<&[u8]> {
    const MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";
    if bundle.len() < MAGIC.len() + 8 || &bundle[..MAGIC.len()] != MAGIC {
        // Not a bundle — assume raw ELF.
        return Some(bundle);
    }
    let mut p = MAGIC.len();
    let num = u64::from_le_bytes(bundle[p..p + 8].try_into().ok()?) as usize;
    p += 8;
    for _ in 0..num {
        if p + 24 > bundle.len() { return None; }
        let off    = u64::from_le_bytes(bundle[p..p + 8].try_into().ok()?) as usize;
        let sz     = u64::from_le_bytes(bundle[p + 8..p + 16].try_into().ok()?) as usize;
        let id_len = u64::from_le_bytes(bundle[p + 16..p + 24].try_into().ok()?) as usize;
        p += 24;
        if p + id_len > bundle.len() { return None; }
        let id_str = std::str::from_utf8(&bundle[p..p + id_len]).ok()?;
        p += id_len;
        // Skip "host-" entry. Pick first amdgcn entry.
        if id_str.contains("amdgcn") || id_str.contains("gfx") {
            return Some(&bundle[off..off + sz]);
        }
    }
    None
}
```

What it does, line by line:

1. Sniff the 24-byte magic. If absent, the input is already a raw ELF (this
   happens when you build with `clang -target amdgcn-amd-amdhsa` directly,
   bypassing hipcc). Return as-is.
2. Read `num_entries` (LE u64).
3. For each entry: read `offset`, `size`, `id_len` (each LE u64), then the
   id string of length `id_len`.
4. The first entry is almost always `host-x86_64-unknown-linux-gnu` with
   `size=0`. We skip anything that doesn't look like an amdgcn target.
5. Return the slice `[offset..offset+size]` of the embedded ELF — that slice
   is what we hand to `hsa_code_object_reader_create_from_memory`.

Equivalent CLI: `clang-offload-bundler --type=o --unbundle
--input=kernel.hsaco --output=kernel.gfx1030.elf --targets=hipv4-amdgcn-amd-amdhsa--gfx1030`,
or the simpler `roc-obj-extract kernel.hsaco` from rocm-tools.

---

## 5. Hidden kernel arguments (V5 ABI)

When clang lowers a kernel for V5, it appends a block of **hidden args** at
the end of the user kernarg buffer. These supply the values that
`__builtin_amdgcn_workgroup_id_x()`, `blockDim.x`, `gridDim.x` etc. resolve
to. The kernel reads them via `kernarg_segment_ptr + .offset` exactly like
user args.

**The runtime is responsible for filling them.** If you forget, the kernel
reads garbage — typically zeroes if the kernarg buffer was zero-initialized,
which yields `gridDim = (0,0,0)` and the kernel computes wrong outputs but
does not fault. Silent corruption.

### 5.1 V5 layout (relative to start of hidden region)

| Offset | Size | Field                          | Filled by runtime with…                       |
|-------:|-----:|--------------------------------|-----------------------------------------------|
|      0 |    4 | hidden_block_count_x           | `(grid_size_x + workgroup_size_x - 1) / wg.x` |
|      4 |    4 | hidden_block_count_y           |                                               |
|      8 |    4 | hidden_block_count_z           |                                               |
|     12 |    2 | hidden_group_size_x            | workgroup size X (u16)                        |
|     14 |    2 | hidden_group_size_y            |                                               |
|     16 |    2 | hidden_group_size_z            |                                               |
|     18 |    2 | hidden_remainder_x             | `grid_size_x % workgroup_size_x` (u16)        |
|     20 |    2 | hidden_remainder_y             |                                               |
|     22 |    2 | hidden_remainder_z             |                                               |
|     24 |    8 | (padding / reserved)           |                                               |
|     32 |    2 | hidden_grid_dims               | 1, 2, or 3                                    |
|     34 |    6 | (padding)                      |                                               |
|     40 |    8 | hidden_private_base            | scratch base ptr (often 0 / set by HSA)       |
|     48 |    8 | hidden_shared_base             | LDS base (often 0)                            |
|     56 |    8 | hidden_queue_ptr               | `hsa_queue_t*`                                |
|     64 |    8 | hidden_default_queue           | 0 unless device-side enqueue                  |
|     72 |    8 | hidden_completion_action       | 0                                             |
|     80 |    8 | hidden_multigrid_sync_arg      | 0 (non-cooperative)                           |
|     88 |    4 | hidden_dynamic_lds_size        | dynamic shared mem bytes (V5 only)            |
|     92 |    4 | (padding)                      |                                               |
|     96 |    8 | hidden_heap_v1                 | device heap ptr (or 0)                        |
|    104 |    8 | hidden_hostcall_buffer         | hostcall buffer ptr (or 0)                    |
|    112 |    8 | hidden_printf_buffer (opt.)    | only if printf was used                       |

The total hidden block is **120 bytes** when printf is enabled, **112 bytes**
otherwise. Older docs cite "56 bytes" for V5 and "88 bytes" for V4 — those
counts are what is **mandatory**; the rest of the slots are only emitted by
the compiler when the kernel uses them.

**Important:** the compiler only emits the *prefix* of this table that the
kernel actually reads. The msgpack `.args` array tells you which slots are
present and at what offset. Do not assume a fixed total — always scan the
metadata and write to the offsets it specifies.

### 5.2 V4 layout (for comparison, 88 bytes mandatory)

V4 uses the OpenCL-ish layout:

| Offset | Size | Field                          |
|-------:|-----:|--------------------------------|
|      0 |    8 | hidden_global_offset_x         |
|      8 |    8 | hidden_global_offset_y         |
|     16 |    8 | hidden_global_offset_z         |
|     24 |    2 | hidden_printf_buffer (or grid_dims) |
|     26 |    6 | padding                        |
|     32 |    8 | hidden_default_queue           |
|     40 |    8 | hidden_completion_action       |
|     48 |    8 | hidden_multigrid_sync_arg      |
|     56 |    8 | hidden_hostcall_buffer         |
|     64 |    8 | hidden_heap_v1                 |
|     72 |    8 | hidden_queue_ptr               |
|     80 |    8 | (reserved)                     |

V4 made the kernel compute `block_count` itself by dividing
`grid_size / wg_size` on the device; V5 precomputes it on the host as a u32
to save VALU instructions.

### 5.3 Failure modes if hidden args are wrong

- **All zero** → kernel sees `gridDim=(0,0,0)`, `blockDim=(0,0,0)`. Loops
  bounded by these never execute. Output buffer stays untouched (or is
  whatever it was initialized to). No fault, no diagnostic.
- **`hidden_queue_ptr = 0`** → only matters if the kernel calls
  `__ockl_*` runtime helpers (printf, malloc). Page fault when they're hit.
- **`hidden_hostcall_buffer = 0`** with printf → page fault on first
  printf call.
- **Wrong `hidden_grid_dims`** → some atomics that key off dim count
  produce wrong wave layouts. Subtle.

---

## 6. The kernarg buffer as a whole

The runtime allocates a single contiguous buffer per dispatch (or pool of
buffers for ring-buffer dispatch) of size:

```
kernarg_total = align_up(user_args_layout_end + hidden_args_size, 8)
```

`kernarg_size` from the descriptor (V5) or `.kernarg_segment_size` from
metadata (V4) gives this exact number — use it, don't recompute.

### 6.1 Alignment rules

- **Buffer base address:** must be aligned to 256 bytes for V5
  (`AMD_ISA_ALIGN_BYTES`). HSA's `hsa_memory_allocate` from a kernarg pool
  guarantees this.
- **Inside the buffer:** each arg uses **natural alignment** (1, 2, 4, 8, 16
  bytes for the arg's type). Pointers and `u64` are 8-aligned. `float4` is
  16-aligned. The compiler inserts padding automatically and the metadata
  `.offset` field reflects the final layout — you never compute offsets, you
  read them from the YAML.
- **Hidden args region:** starts at the first 8-byte boundary past the user
  args. The metadata's first hidden arg `.offset` is that boundary.

### 6.2 Practical sizing helper

```rust
fn kernarg_total_size(meta: &KernelMeta) -> usize {
    // Find max(.offset + .size) across all args (user + hidden), round to 8.
    let end = meta.args.iter()
        .map(|a| a.offset + a.size)
        .max().unwrap_or(0);
    (end + 7) & !7
}
```

But again: prefer reading `kernarg_size` directly from
`kernel_descriptor_t.kernarg_size` (V5).

---

## 7. Inspection commands cheat sheet

```bash
# Header + ABI version
readelf -h kernel.hsaco | head -20

# Sections
readelf -SW kernel.hsaco

# Notes (decoded msgpack as YAML on modern llvm)
llvm-readobj --elf-output-style=GNU --notes kernel.hsaco
# or:
roc-obj-ls -v kernel.hsaco

# Symbols (find the .kd entry)
llvm-readobj --dyn-symbols kernel.hsaco | grep '\.kd'

# Disassemble
llvm-objdump -d --mcpu=gfx1030 kernel.hsaco | less

# Strip the offload bundle wrapper
clang-offload-bundler --type=o --unbundle \
    --input=kernel.hsaco \
    --output=kernel.elf \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx1030

# Pretty-print kernel metadata as YAML
roc-obj-extract -o /tmp/extracted kernel.hsaco
```

To dump the kernel descriptor bytes themselves once you have the inner ELF:

```bash
# Find KD address
KD_ADDR=$(llvm-readobj --dyn-symbols kernel.elf | grep 'foo\.kd' \
          | awk '/Value/{print $2}')
# Find .rodata offset
SECT=$(readelf -SW kernel.elf | awk '$2==".rodata"{print $5,$7}')
# Hexdump the 64 bytes
dd if=kernel.elf bs=1 skip=$((rodata_offset + KD_ADDR - rodata_addr)) count=64 \
   | xxd
```

---

## 8. Loading patterns (HSA layer)

The modern path:

```c
hsa_code_object_reader_t reader;
hsa_code_object_reader_create_from_memory(elf_ptr, elf_size, &reader);

hsa_executable_t exe;
hsa_executable_create_alt(HSA_PROFILE_FULL,
                          HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                          NULL, &exe);

hsa_loaded_code_object_t loaded;
hsa_executable_load_agent_code_object(exe, agent, reader, NULL, &loaded);

hsa_executable_freeze(exe, NULL);

hsa_executable_symbol_t sym;
hsa_executable_get_symbol_by_name(exe, "matmul_bf16.kd", &agent, &sym);

uint64_t kernel_object;
hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                               &kernel_object);

uint32_t kernarg_size;
hsa_executable_symbol_get_info(sym,
    HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg_size);
```

Notes:

- The reader takes a **raw ELF** — strip the offload bundle first (§4).
- `hsa_executable_load_code_object` (without `_agent_`) is deprecated since
  HSA 1.2; the agent variant supports per-agent loading and is required for
  multi-GPU.
- `hsa_executable_freeze` validates relocations and finalizes addresses.
  After this, kernel object pointers are stable until `hsa_executable_destroy`.
- The reader does **not** copy the ELF — `elf_ptr` must stay alive until
  freeze completes, then can be freed.

---

## 9. Symbol resolution and name mangling

Clang emits two symbols per kernel:

1. `<name>` — type `STT_FUNC`, points at the `.text` entry. Useful for
   profiling/debugging only.
2. `<name>.kd` — type `STT_OBJECT`, lives in `.rodata`, points at the
   64-byte kernel descriptor.

**You always look up `<name>.kd`** — that pointer is what goes into the AQL
packet's `kernel_object` field.

For C++ HIP kernels (e.g. `__global__ void foo(int*)`), the C++ name
mangling applies first: clang produces `_Z3fooPi.kd`. Use the unmangled
demangled name for the user-facing API and let the build system extract the
mangled `.kd` symbol via `nm -D` or by parsing the metadata `.symbol` field
(which always carries the final mangled name).

For C-linkage kernels declared `extern "C"`, the symbol is plain `<name>.kd`
with no mangling. This is the easiest path when writing your own kernels in
HIP-C and loading them by name.

---

## 10. Debugging tips

### 10.1 Validate the descriptor

```bash
# Inspect the 64 bytes at the .kd symbol
python3 -c '
import struct, sys
data = open(sys.argv[1], "rb").read()
# elf parsing snipped — find rodata offset of .kd
kd = data[rodata_off + kd_va_offset : rodata_off + kd_va_offset + 64]
fields = struct.unpack("<IIII4xqQ20xIIIHHI", kd)  # adjust
print("group_segment_fixed:", fields[0])
print("private_segment_fixed:", fields[1])
print("kernarg_size:", fields[2])
print("kernel_code_entry_offset:", fields[5])
print("compute_pgm_rsrc1:", hex(fields[7]))
print("compute_pgm_rsrc2:", hex(fields[8]))
print("kernel_code_properties:", hex(fields[9]))
' kernel.elf
```

Sanity checks:
- `kernarg_size` matches what your runtime computes.
- `kernel_code_entry_byte_offset` is 256 (typical) — KD + 192 B of pad.
- `compute_pgm_rsrc1.GRANULATED_WORKITEM_VGPR_COUNT` is consistent with the
  `.vgpr_count` in the metadata: for wave32 RDNA2, formula is
  `vgpr_count = (granulated + 1) * 8`.
- `kernel_code_properties.ENABLE_SGPR_KERNARG_SEGMENT_PTR = 1` if your
  kernel takes any args.
- `ENABLE_WAVEFRONT_SIZE32 = 1` for RDNA1+ — RDNA wants wave32 by default.

### 10.2 Common GPU faults

- **Page fault on kernarg buffer**: the AQL packet's `kernarg_address` is
  null, unaligned (must be 8-byte minimum), or in a memory pool the agent
  cannot read. Allocate from the kernarg pool returned by
  `hsa_amd_memory_pool_get_info(... GLOBAL_FLAG_KERNARG ...)`.
- **`SQ_WAVE_INST_DOWN` / illegal instruction**: the GPU received an opcode
  it doesn't understand. Causes: (a) wrong `--mcpu` at compile time
  (compiled gfx1100 instructions running on gfx1030); (b) `compute_pgm_rsrc1`
  has `WGP_MODE` set to a value the chip doesn't support; (c) ELF bytes
  loaded from an offload bundle without unwrapping (the runtime reads the
  bundle magic as instructions).
- **`MEMORY_VIOLATION`**: the kernel is reading past a buffer. Most often
  caused by wrong `hidden_block_count_*` (kernel computed wrong loop bound)
  or wrong stride in user args.
- **Hang / no completion**: `hidden_queue_ptr` is null and the kernel called
  a runtime helper, or the kernarg buffer was overwritten by the next
  dispatch before this one drained. Use one kernarg slot per outstanding
  dispatch.

### 10.3 rocgdb basics

```bash
ROCM_DEBUG_ENABLE=1 rocgdb --args ./your_app
(rocgdb) break matmul_bf16
(rocgdb) info threads      # lists waves
(rocgdb) thread <wave>
(rocgdb) info registers
(rocgdb) x/64xb $kd        # dump the kernel descriptor
```

The `AMD_LOG_LEVEL=4` env var also produces verbose output from
libhsa-runtime that often shows you exactly which descriptor field the
loader rejected.

---

## 11. References

Primary specs:
- LLVM AMDGPU User Guide (definitive ABI):
  [llvm/docs/AMDGPUUsage.rst](https://github.com/llvm/llvm-project/blob/main/llvm/docs/AMDGPUUsage.rst)
  rendered at [llvm.org/docs/AMDGPUUsage.html](https://llvm.org/docs/AMDGPUUsage.html).
- Kernel descriptor struct + bitfield enums:
  [llvm/include/llvm/Support/AMDHSAKernelDescriptor.h](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/AMDHSAKernelDescriptor.h).
- Legacy 256-byte AMD kernel-code header:
  `llvm/lib/Target/AMDGPU/AMDKernelCodeT.h` and
  `/opt/rocm-7.2.1/include/hsa/amd_hsa_kernel_code.h`.
- ELF constants for AMDGPU:
  `/opt/rocm-7.2.1/include/hsa/amd_hsa_elf.h` (mirrors `llvm/BinaryFormat/ELF.h`).

Tooling and bundling:
- Clang offload bundler:
  [clang.llvm.org/docs/ClangOffloadBundler.html](https://clang.llvm.org/docs/ClangOffloadBundler.html).
- ROCR-Runtime headers (HSA implementation): `/opt/rocm-7.2.1/include/hsa/`,
  notably `hsa.h`, `hsa_ext_amd.h`, `amd_hsa_kernel_code.h`,
  `amd_hsa_elf.h`, `hsa_ven_amd_loader.h`.
- HIP runtime: `/opt/rocm-7.2.1/include/hip/`, but for direct HSA dispatch
  you typically don't need it.

Patches and design discussions worth reading:
- [D120265 — Use the implicit kernargs for code object version 5](https://reviews.llvm.org/D120265)
- [D158150 — Add hidden_dynamic_lds_size to v5](https://reviews.llvm.org/D158150)
- [D92434 — Code object V4 ABI documentation](https://reviews.llvm.org/D92434)

LLVM commit announcements that document V5 changes:
- [llvm-commits 2022-01-31 — V5 implicit kernel args](https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20220131/1004054.html)

GFX1030/RDNA2 ISA syntax:
- [Syntax of GFX10 RDNA2 Instructions (LLVM)](https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX1030.html).

In-tree code reference:
- The user's own extractor:
  `/home/leonardo/projects/LLM/llm-lab/hip-ffi/src/hsa_dispatch.rs`,
  function `extract_amdgcn_elf` at lines 428–457 — implements §4.3 above.
