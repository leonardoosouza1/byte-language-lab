# HSA AQL — Architected Queuing Language: A Production Reference

**Scope:** Everything required to write a from-scratch AQL dispatcher for AMD RDNA2 (gfx1030, gfx1031, gfx1032, gfx1034, gfx1035) bypassing HIP. Targets the ROCR userspace runtime + amdkfd kernel driver. Sources are cited inline.

**Audience:** Engineers writing direct HSA producers in Rust/C++ for compute-only workloads (LLM inference, custom kernels, scheduler experiments). The HSA layer is the *thinnest* practical abstraction that still gives portable code-object loading and signal/queue semantics — below it lies PM4 and KFD ioctls.

---

## 0. Why bypass HIP?

ROCclr (the HIP/OpenCL common layer) translates a `hipLaunchKernelGGL` call into:

1. an internal command list,
2. a `Kernel` object lookup,
3. an AQL packet construction,
4. a queue acquire/serialize,
5. a `hsa_signal_store_screlease` doorbell.

Steps 1–4 cost ~50–70 µs per launch on ROCm 6.x measured single-kernel single-thread; the AMD goal stated in HIP's own internal benchmarks is ~2 µs *if* the layers are bypassed [Kokkos issue #3670; AMD 'fxkamd' tinygrad-observations gist]. A direct AQL producer can routinely hit **0.5–1.5 µs** per dispatch on RDNA2 because:

- The packet body is ~64 bytes of plain memory writes.
- The doorbell is one MMIO write (64 bits) into a userspace-mapped doorbell page.
- No malloc, no mutex, no `dlopen`, no driver entry, no signal allocation if you reuse a ring of pre-created signals.

The 1.4×–6.3× HIP→HSA speedup we measured on Qwen2.5-1.5B WGSL matmul (see project memory: LSH/superlayer findings) is consistent with the ~50× reduction of host overhead per dispatch when many small kernels back-to-back.

---

## 1. HSA architecture overview

HSA (Heterogeneous System Architecture) is the AMD/HSA-Foundation specification for a coherent platform model. Two normative documents:

- **HSA Platform System Architecture Specification (PSAS) v1.2** — `http://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf`. Defines the hardware-visible model: agents, signals, queues, AQL packets, memory model.
- **HSA Programmer's Reference Manual (PRM) v1.2** — `http://hsafoundation.com/wp-content/uploads/2021/02/HSA-PRM-1.2.pdf`. Defines BRIG/HSAIL (legacy; AMD does not ship HSAIL), the runtime API, and the kernel-side ABI.

ROCR-Runtime is AMD's open implementation of the HSA Runtime API. Header tree (post-rocm-systems consolidation, 2025):

```
rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/inc/
    hsa.h                     # core HSA Runtime API
    hsa_ext_amd.h             # AMD vendor extensions
    hsa_ext_image.h           # image/sampler extension (rare in compute)
    hsa_ext_finalize.h        # legacy BRIG finalizer (do not use)
    amd_hsa_kernel_code.h     # AMD kernel descriptor
    amd_hsa_signal.h          # AMD signal extension
    amd_hsa_queue.h           # AMD queue extension
```

Old path (ROCR-Runtime repo, deprecated mirror still useful for diffs):
`https://github.com/ROCm/ROCR-Runtime/blob/master/src/inc/hsa.h`

### 1.1 Core entities

```
+----------------------------------------------------------------+
| HSA System                                                     |
|                                                                |
|  +---------------+      +----------------+      +-----------+  |
|  |  CPU Agent    | ===  |  GPU Agent     | ===  | dGPU#2    |  |
|  |  (host)       |      |  (gfx1030)     |      | (optional)|  |
|  +-------+-------+      +-------+--------+      +-----------+  |
|          |                      |                              |
|          |   +------------------+----------------------+       |
|          |   |  Memory Pools (per agent or global)     |       |
|          |   |   - fine-grained system  (host pinned)  |       |
|          |   |   - coarse-grained device (VRAM)        |       |
|          |   |   - kernarg                             |       |
|          |   |   - group (LDS, allocated per dispatch) |       |
|          |   +-----------------------------------------+       |
|          |                      |                              |
|  +-------+-------+      +-------+--------+                     |
|  |  Signals      |      |  AQL Queues    |                     |
|  |  (64-bit)     |      |  (ring buffer) |                     |
|  +---------------+      +----------------+                     |
+----------------------------------------------------------------+
```

- **Agent** [HSA PRM §2.2]: any device that can execute or be the target of dispatches. `hsa_agent_t = { uint64_t handle }`. Discoverable via `hsa_iterate_agents`.
- **Region / Memory Pool**: the legacy `hsa_region_t` API is deprecated; use AMD-extension `hsa_amd_memory_pool_t`. Each pool has segment (GLOBAL/GROUP/PRIVATE/KERNARG), accessibility (which agents may access), and granularity (fine vs coarse).
- **Signal** [HSA PSAS §2.7]: a 64-bit shared synchronization word with atomic operations and a wait API. Producers atomically modify the value; consumers wait on a condition. Backed by host memory plus an event for OS-blocking waits.
- **Queue** [HSA PSAS §2.6]: a user-mode AQL packet ring buffer plus a doorbell signal.

### 1.2 Programming flow (the canonical sequence)

```
hsa_init
hsa_iterate_agents -> pick GPU agent
hsa_amd_agent_iterate_memory_pools -> pick coarse-grained VRAM + kernarg
hsa_signal_create (completion signals, ring of N)
hsa_queue_create (one per concurrent stream)
hsa_code_object_reader_create_from_memory (load .hsaco bytes)
hsa_executable_create_alt
hsa_executable_load_agent_code_object
hsa_executable_freeze
hsa_executable_get_symbol_by_name -> kernel_object u64 (GPU VA)
hsa_executable_symbol_get_info -> kernarg_segment_size, group_segment_size, ...
hsa_amd_memory_pool_allocate (kernarg buffer, ring of N slots)
loop:
  build kernarg slot[i]
  reserve packet slot j = hsa_queue_add_write_index_screlease(queue, 1)
  write body of packet at slots[j & mask]
  atomic_store_release(&packet.header, encoded_header)
  hsa_signal_store_screlease(queue->doorbell_signal, j)   # ring it
hsa_signal_wait_scacquire (completion_signal, EQ, 0)
```

Every word in that sequence is dissected below.

---

## 2. AQL packet ABI

All AQL packets are exactly **64 bytes**, naturally aligned to 64 bytes. The packet processor in CP (Command Processor) firmware interprets the header byte to dispatch the packet [HSA PSAS §2.8]. RDNA2's MEC (Micro Engine Compute) decodes the AQL packet directly when the queue type is HSA (KFD configures this through MES on RDNA2 with newer firmware; on earlier asics it goes through HWS).

### 2.1 Packet header (16 bits — first 2 bytes of every packet)

```
hsa_packet_header_t bit layout (LSB = bit 0):
+-----+-----+-----+-----+
| 0-7 | 8   |9-10 |11-12| (bits 13..15 reserved, must be 0)
|type |barr |sAcq |sRel |
+-----+-----+-----+-----+
   8b   1b    2b    2b
```

[ROCR `inc/hsa.h`, `hsa_packet_header_t`]:

| Sub-field                         | Bit offset | Width | Notes |
|-----------------------------------|-----------|-------|-------|
| `HSA_PACKET_HEADER_TYPE`          | 0         | 8     | `hsa_packet_type_t` |
| `HSA_PACKET_HEADER_BARRIER`       | 8         | 1     | If 1, packet processor stalls until all prior packets in this queue complete before launch |
| `HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE` | 9 | 2 | Acquire fence applied *before* dispatch (`hsa_fence_scope_t`) |
| `HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE` | 11| 2 | Release fence applied *after* dispatch completes |

Pre-1.2 names were `*ACQUIRE_FENCE_SCOPE` (no `SC`). The `SC` prefix means "sequentially consistent w.r.t. atomics" — added 1.2 to align with C++11. ROCR currently accepts both spellings.

```
hsa_packet_type_t   value
  VENDOR_SPECIFIC      0
  INVALID              1   <- packet processor sees this and stalls (waits for header rewrite)
  KERNEL_DISPATCH      2
  BARRIER_AND          3
  AGENT_DISPATCH       4
  BARRIER_OR           5

hsa_fence_scope_t   value  meaning
  NONE                 0   no fence
  AGENT                1   coherence within this agent's caches (L1+L2 of one GPU)
  SYSTEM               2   coherence across all agents (PCIe, IOMMU, system memory)
```

Why `INVALID=1`: a freshly-allocated/zeroed queue ring has every header reading as `VENDOR_SPECIFIC=0`, which the packet processor will *try to decode and skip*. If you rewrite the header to `INVALID(1)` first, then write the body, then atomically rewrite to `KERNEL_DISPATCH(2)`, the firmware never observes a half-built valid packet [fxkamd tinygrad gist]. This is critical: see §3.3.

### 2.2 `hsa_kernel_dispatch_packet_t` — 64 bytes

```c
typedef struct hsa_kernel_dispatch_packet_s {
  uint16_t header;                   // 0x00  +2  (see §2.1)
  uint16_t setup;                    // 0x02  +2  (see below)
  uint16_t workgroup_size_x;         // 0x04  +2
  uint16_t workgroup_size_y;         // 0x06  +2
  uint16_t workgroup_size_z;         // 0x08  +2
  uint16_t reserved0;                // 0x0A  +2  must be 0
  uint32_t grid_size_x;              // 0x0C  +4
  uint32_t grid_size_y;              // 0x10  +4
  uint32_t grid_size_z;              // 0x14  +4
  uint32_t private_segment_size;     // 0x18  +4  scratch per work-item, bytes
  uint32_t group_segment_size;       // 0x1C  +4  LDS per work-group, bytes
  uint64_t kernel_object;            // 0x20  +8  GPU VA of 64-byte kernel descriptor
  void*    kernarg_address;          // 0x28  +8  GPU-visible VA of kernarg buffer (16-byte aligned)
  uint64_t reserved2;                // 0x30  +8
  hsa_signal_t completion_signal;    // 0x38  +8  signal handle (0 = no signal)
} hsa_kernel_dispatch_packet_t;       // total 0x40 = 64
```

[ROCR-Runtime `src/inc/hsa.h`, struct `hsa_kernel_dispatch_packet_s`]

#### 2.2.1 `setup` (16 bits)

```
hsa_kernel_dispatch_packet_setup_t:
+------+-----------+
| 0-1  |   2-15    |
| dims | reserved  |
+------+-----------+

dims = number of grid dimensions, valid values: 1, 2, 3
```

Width enum: `HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS = 2`. If `dims=1`, you must set `workgroup_size_y/z = 1` and `grid_size_y/z = 1`. AMD firmware will treat zero as "stall" — never write zero in a "live" dimension.

#### 2.2.2 `kernel_object`

A 64-byte-aligned GPU VA pointing to the **AMD HSA Kernel Descriptor** (AKC). Not the kernel ELF entry point. The descriptor (defined in `amd_hsa_kernel_code.h`) wraps the ISA blob:

```
amd_kernel_code_t (kernel descriptor) — 64 bytes, GFX10/RDNA2 layout
offset  size  field
0x00    4     group_segment_fixed_size       // bytes of LDS
0x04    4     private_segment_fixed_size     // scratch bytes per work-item
0x08    4     kernarg_size                   // bytes (V5 only)
0x0C    4     reserved                       // must be 0
0x10    8     kernel_code_entry_byte_offset  // signed offset from start of AKC to ISA
0x18   16     reserved                       // 0
0x28    4     compute_pgm_rsrc3              // GFX10+ adds wgp_mode etc
0x2C    4     compute_pgm_rsrc1              // VGPR/SGPR/FP_ROUND/DX10_CLAMP/...
0x30    4     compute_pgm_rsrc2              // SGPR user inputs, LDS, TGID enable
0x34    2     kernel_code_properties         // 16-bit flags (ENABLE_KERNARG_SEGMENT etc)
0x36    2     kernarg_preload_spec_length    // V5: dwords preloaded into User SGPRs
0x38    2     kernarg_preload_spec_offset    // V5: dword offset within kernarg
0x3A    6     reserved
total  64
```

`kernel_code_properties` bits (selected; see `amd_hsa_kernel_code.h`):

```
bit  name                                                 # User SGPRs added
0    ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER                   4
1    ENABLE_SGPR_DISPATCH_PTR                             2
2    ENABLE_SGPR_QUEUE_PTR                                2
3    ENABLE_SGPR_KERNARG_SEGMENT_PTR                      2
4    ENABLE_SGPR_DISPATCH_ID                              2
5    ENABLE_SGPR_FLAT_SCRATCH_INIT                        2
6    ENABLE_SGPR_PRIVATE_SEGMENT_SIZE                     1
10   ENABLE_WAVEFRONT_SIZE32                              -   // RDNA: must be 1 for wave32 kernels
```

The User SGPRs that the firmware initializes from the AQL packet are *not* the kernarg pointer values themselves; they are pointers/values laid down before the kernel sees its first instruction. Of these, `ENABLE_SGPR_KERNARG_SEGMENT_PTR` produces s[0:1] = `kernarg_address` from the packet; this is how the kernel finds its arguments.

#### 2.2.3 `completion_signal`

A `hsa_signal_t`; if non-zero, the packet processor decrements the signal value by 1 *after* the kernel completes (and after `screlease` fence). Initial value is conventionally 1 → consumer waits for 0. Reusing a signal across dispatches without re-arming causes race hangs (set value back to N before submitting a new batch).

### 2.3 `hsa_barrier_and_packet_t` — 64 bytes

```c
typedef struct hsa_barrier_and_packet_s {
  uint16_t header;                   // 0x00  TYPE=BARRIER_AND, ack/rel fences typically NONE
  uint16_t reserved0;                // 0x02
  uint32_t reserved1;                // 0x04
  hsa_signal_t dep_signal[5];        // 0x08  5 × 8 = 40 bytes
  uint64_t reserved2;                // 0x30
  hsa_signal_t completion_signal;    // 0x38
} hsa_barrier_and_packet_t;
```

Semantics: packet processor waits until **every** non-zero `dep_signal[i]` reaches a value ≤ 0 (`hsa_signal_value_t` is signed 64), then decrements `completion_signal` if present. Zero-handle entries are ignored.

### 2.4 `hsa_barrier_or_packet_t` — 64 bytes

Identical layout to `barrier_and`, header type = `BARRIER_OR=5`. Waits for **any** non-zero `dep_signal[i]` to reach ≤ 0.

### 2.5 `hsa_agent_dispatch_packet_t` — 64 bytes

Used to send a vendor-defined agent command. Used by AMD for `hsa_amd_signal_*` cross-agent ops; rare in user code.

```c
typedef struct hsa_agent_dispatch_packet_s {
  uint16_t header;
  uint16_t type;                    // vendor-specific opcode
  uint32_t reserved0;
  void*    return_address;          // GPU VA where agent writes return data
  uint64_t arg[4];                  // opaque
  uint64_t reserved2;
  hsa_signal_t completion_signal;
} hsa_agent_dispatch_packet_t;
```

### 2.6 Vendor extensions

`hsa_ext_amd.h`:

- `hsa_amd_barrier_value_packet_t` (also 64 bytes): waits on *one* signal for `signal->value (op) value` where `op ∈ {EQ, NE, LT, GTE}`. Strictly more flexible than barrier_and.
- `hsa_amd_aql_trap` packet — used internally by ROCR for hostcall and exception delivery.

---

## 3. Queue mechanics

### 3.1 `hsa_queue_create`

```c
hsa_status_t hsa_queue_create(
    hsa_agent_t agent,
    uint32_t size,                                     // # of slots, power of 2
    hsa_queue_type32_t type,                           // SINGLE | MULTI | COOPERATIVE
    void (*callback)(hsa_status_t, hsa_queue_t*, void*),
    void* data,
    uint32_t private_segment_size,                     // scratch hint, 0 ok
    uint32_t group_segment_size,                       // LDS hint, 0 ok
    hsa_queue_t** queue);
```

Type values [`hsa_queue_type_t`]:

```
HSA_QUEUE_TYPE_MULTI       = 0   // multi-producer, atomic write-index allocation
HSA_QUEUE_TYPE_SINGLE      = 1   // single producer; the runtime may use a faster path
HSA_QUEUE_TYPE_COOPERATIVE = 2   // GFX9+ multi-grid cooperative; rarely used on RDNA2
```

`size` must be a power of two and ≥ the agent's `HSA_AGENT_INFO_QUEUE_MIN_SIZE` (usually 64) and ≤ `QUEUE_MAX_SIZE` (RDNA2: typically 0x100000 = 1M slots; in practice 0x1000–0x10000 is plenty).

ROCR-private queue creation flags (set via `hsa_amd_queue_set_priority`, env vars, or `hsa_amd_profiling_set_profiler_enabled`):

```
HSA_AMD_QUEUE_PRIORITY_LOW    = 0
HSA_AMD_QUEUE_PRIORITY_NORMAL = 1
HSA_AMD_QUEUE_PRIORITY_HIGH   = 2
```

There is no public "BLOCK_FREE" queue size flag in the upstream API — you size the ring to your peak in-flight count and add backpressure. Some ROCR private builds expose `HSA_AMD_AGENT_INFO_DOORBELL_TYPE` to choose interrupt-backed vs polled doorbells; this is mediated by amdkfd.

### 3.2 `hsa_queue_t` layout

```c
typedef struct hsa_queue_s {
  hsa_queue_type32_t type;          // 0x00  +4
  uint32_t features;                // 0x04  +4   bit0 = KERNEL_DISPATCH supported
  void* base_address;               // 0x08  +8   pointer to ring (size * 64 bytes)
  hsa_signal_t doorbell_signal;     // 0x10  +8
  uint32_t size;                    // 0x18  +4   # slots (power of 2)
  uint32_t reserved1;               // 0x1C
  uint64_t id;                      // 0x20  +8
} hsa_queue_t;
```

The runtime returns this struct by *pointer*; you do not free it (use `hsa_queue_destroy`). The actual ring buffer at `base_address` is GPU-visible (kernarg-segment-like, host-pinned and mapped to GPU). It is safe to write packet bodies from the CPU; cache coherence over PCIe is handled by the AQL release fence on the header.

### 3.3 Producer protocol (lock-free)

The ring buffer is indexed by a 64-bit **monotonically increasing** write_index. Slot is `base_address + ((write_index & (size-1)) << 6)` — the `<<6` is the 64-byte packet stride.

```
hsa_queue_load_write_index_relaxed   // load only, no fence
hsa_queue_load_write_index_scacquire // load + acquire fence
hsa_queue_add_write_index_relaxed    // fetch_add, returns old
hsa_queue_add_write_index_screlease  // fetch_add with release
hsa_queue_store_write_index_*        // store new value
hsa_queue_cas_write_index_*          // CAS
```

The canonical multi-producer write sequence:

```c
// 1. Reserve a slot. fetch_add returns the OLD value -> our slot index.
uint64_t slot = hsa_queue_add_write_index_screlease(queue, 1);

// 2. Backpressure: do not overwrite a packet the GPU has not yet consumed.
while (slot - hsa_signal_load_scacquire(queue->doorbell_signal) >= queue->size) {
    // queue full; spin, yield, or hsa_signal_wait_scacquire on the consumer head.
}

// 3. Compute slot pointer.
hsa_kernel_dispatch_packet_t* p =
    &((hsa_kernel_dispatch_packet_t*)queue->base_address)[slot & (queue->size - 1)];

// 4. CRITICAL: write header LAST, with INVALID first if the prior generation
//    might still have a stale TYPE there. (Safe pattern: write everything
//    EXCEPT header[0..7], then atomic store_release of full uint16 header.)
uint16_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << 0)
                | (1 << 8)   /* barrier bit, optional */
                | (HSA_FENCE_SCOPE_SYSTEM << 9)   /* acquire */
                | (HSA_FENCE_SCOPE_AGENT  << 11); /* release */
uint16_t setup = 1;  // dims=1
uint32_t header_setup = (uint32_t)setup << 16 | header;

p->setup            = setup;
p->workgroup_size_x = 256; p->workgroup_size_y = 1; p->workgroup_size_z = 1;
p->grid_size_x      = N;   p->grid_size_y      = 1; p->grid_size_z      = 1;
p->private_segment_size = 0;
p->group_segment_size   = lds_bytes;
p->kernel_object        = kernel_object_va;
p->kernarg_address      = kernarg_va;
p->reserved2            = 0;
p->completion_signal    = comp_sig;

// 5. Release-store the (header, setup) pair atomically as one 32-bit write.
__atomic_store_n((uint32_t*)p, header_setup, __ATOMIC_RELEASE);

// 6. Ring the doorbell. Value passed = the latest packet index that is ready.
hsa_signal_store_screlease(queue->doorbell_signal, slot);
```

**Why header last with release fence**: the CP/MES packet processor's read-side observation of `header.type != INVALID` means "packet is ready". Without release ordering on the header write, another core might observe the new header but stale body, and the GPU will dispatch garbage. AMD ROCR's `core/runtime/amd_aql_queue.cpp` enforces this by writing a 32-bit `(setup << 16) | header` with an atomic release. [ROCR `src/core/runtime/amd_aql_queue.cpp`]

**Why `hsa_signal_store_screlease` for the doorbell**: the doorbell's value is "the packet index that is now ready". The packet processor's firmware does:

```
loop forever:
  doorbell_value = read(doorbell_mmio_register)        // userspace MMIO write triggered an interrupt to MES
  while (next_to_read <= doorbell_value):
      packet = ring[next_to_read & mask]
      if packet.header.type == INVALID:
          spin (the producer hasn't finished yet; should be rare with proper fences)
      dispatch(packet)
      next_to_read += 1
```

The MMIO write happens because `doorbell_signal` is a *special* AMD signal whose store implementation writes to a 64-bit doorbell BAR address that's mmaped into the user process by amdkfd. See §4.4.

### 3.4 Doorbell signals — how they differ

A normal HSA signal value lives in a host page. The AMD doorbell signal's "store" is overridden to perform a **non-cacheable 64-bit write to a memory-mapped GPU register**, which on RDNA2 causes the GFX_HQD (compute hardware queue descriptor) to wake up its associated MEC pipe and consume new packets.

`amdkfd` (the kernel driver) maps a 4 KB **doorbell page** per process into userspace. Each queue gets a 64-bit slot in that page. The `hsa_signal_t` for a doorbell, when stored, writes through this mapping. There is no way to "wait on" a doorbell; it is write-only from the producer side.

ROCm 6.x doorbells are 64-bit (`AQL_DOORBELL_DWORD_OFFSET` is computed as `packet_index * sizeof(packet)/4 = packet_index * 16` historically; modern firmware accepts the raw packet index). Stale doorbell writes (writing a smaller value than the current head) are harmless — the firmware uses `max(prev, new)` semantics.

### 3.5 Soft queues vs hard queues

- **Hard queue** = backed by a hardware compute pipe, doorbell goes to MMIO, packets consumed by MES/CP. This is what you get from `hsa_queue_create` on a GPU agent.
- **Soft queue** = `hsa_soft_queue_create` — software-only ring intended for agent→agent communication where a software thread polls the queue. Used for hostcall, debug trap delivery, packet relay. Does not have an MMIO doorbell; the doorbell is just a regular `hsa_signal_t`.

### 3.6 When multiple queues help

RDNA2 has **4 ME pipes (compute) × 8 queues per pipe** in the HQD (32 hardware queue slots), with MES virtualizing more. A single HSA queue serializes through one HQD slot — work submitted in order serializes at packet-grain unless you set BARRIER=0 (allowing the GPU scheduler to overlap dispatches). Kernels in the same queue with BARRIER=0 *can* execute concurrently if SE/CU resources allow [HSA PSAS §2.8.4].

Use multiple queues when:
1. You want priority lanes (one HIGH for latency-critical, one NORMAL for batch).
2. You need to keep two kernels truly concurrent and they have inter-kernel data deps that would force a barrier in a single queue.
3. You're profiling and want timestamp isolation per stream.

Diminishing returns past ~4 active queues per GPU on RDNA2 — HQD slots are 32 but most workloads contention-bound earlier.

---

## 4. HSA signals

### 4.1 API summary

```c
hsa_status_t hsa_signal_create(hsa_signal_value_t initial_value,
                               uint32_t num_consumers,
                               const hsa_agent_t* consumers,
                               hsa_signal_t* signal);

void hsa_signal_store_relaxed   (hsa_signal_t s, hsa_signal_value_t v);
void hsa_signal_store_screlease (hsa_signal_t s, hsa_signal_value_t v);

hsa_signal_value_t hsa_signal_load_relaxed   (hsa_signal_t s);
hsa_signal_value_t hsa_signal_load_scacquire (hsa_signal_t s);

hsa_signal_value_t hsa_signal_wait_relaxed(
    hsa_signal_t s, hsa_signal_condition_t cond,
    hsa_signal_value_t cmp, uint64_t timeout_hint, hsa_wait_state_t state);
hsa_signal_value_t hsa_signal_wait_scacquire(...);  // same args, with acquire fence on satisfy

void hsa_signal_add_screlease       (hsa_signal_t, hsa_signal_value_t);
void hsa_signal_subtract_screlease  (hsa_signal_t, hsa_signal_value_t);
void hsa_signal_and_screlease       (hsa_signal_t, hsa_signal_value_t);
void hsa_signal_or_screlease        (hsa_signal_t, hsa_signal_value_t);
void hsa_signal_xor_screlease       (hsa_signal_t, hsa_signal_value_t);
void hsa_signal_exchange_screlease  (hsa_signal_t, hsa_signal_value_t) -> old;
hsa_signal_value_t hsa_signal_cas_screlease(hsa_signal_t, expected, value) -> prev;
```

[ROCR `inc/hsa.h`, signal section]

`hsa_signal_value_t` is `int64_t` on 64-bit hosts (it's `int32_t` on HSA "small" mode, irrelevant on AMD).

### 4.2 Conditions and wait state

```c
hsa_signal_condition_t:
  HSA_SIGNAL_CONDITION_EQ  = 0
  HSA_SIGNAL_CONDITION_NE  = 1
  HSA_SIGNAL_CONDITION_LT  = 2
  HSA_SIGNAL_CONDITION_GTE = 3

hsa_wait_state_t:
  HSA_WAIT_STATE_BLOCKED = 0   // OS-level park (event/futex)
  HSA_WAIT_STATE_ACTIVE  = 1   // CPU-side spin
```

`timeout_hint`: in **system time units** as defined by `HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY`. `UINT64_MAX` = wait forever. The hint is non-binding — the runtime may return early with the *current* signal value; you must always re-check the condition.

### 4.3 Memory ordering of signals

The PSAS memory model defines signals as *synchronization variables*. A `screlease` store happens-before any `scacquire` load that observes the stored value [HSA PSAS §3]. In C++11 terms: signal release/acquire pairs synchronize-with each other and form a happens-before chain. This is what makes the AQL `kernel_dispatch.completion_signal` decrement on the GPU side visible to a CPU-side `hsa_signal_wait_scacquire` with all of the kernel's writes to global memory observable.

### 4.4 `hsa_amd_signal_create` — interruptible signals

```c
hsa_status_t hsa_amd_signal_create(
    hsa_signal_value_t initial_value,
    uint32_t num_consumers, const hsa_agent_t* consumers,
    uint64_t attributes,                  // bit0 = HSA_AMD_SIGNAL_AMD_GPU_ONLY (no event)
                                           // bit1 = HSA_AMD_SIGNAL_IPC
    hsa_signal_t* signal);
```

By default, every HSA signal has an associated KFD event (a file-descriptor-backed wait primitive). With many signals (10k+), event allocation becomes the bottleneck. `HSA_AMD_SIGNAL_AMD_GPU_ONLY` skips event allocation — fine if **only the GPU** waits on the signal (e.g., barrier_and dep_signals consumed by packet processor). If the CPU needs to block on it, you must use a polling wait or use a non-AMD-GPU-only signal.

### 4.5 Polling vs interrupt waits

Inside `hsa_signal_wait_*`, the runtime decides:
- If `state == ACTIVE`: tight CPU spin checking the value (≤ ~1 µs latency).
- If `state == BLOCKED`: spins briefly (a few µs), then `ioctl(KFD_IOC_WAIT_EVENTS)` which sleeps the thread until the GPU's interrupt completes the wait.

For latency-critical code (LLM token loop), use ACTIVE. For long kernels (>100 µs), BLOCKED saves CPU. Setting environment variable `HSA_ENABLE_INTERRUPT=0` forces ROCR to never use interrupt-backed waits.

### 4.6 Completion vs barrier signals

There is no API distinction. A "completion signal" is just a signal you put in `packet.completion_signal`. A "barrier signal" is just a signal you put in `barrier_and.dep_signal[i]`. Same struct. The pattern matters:

- **Completion**: 1-shot, value starts at 1 (or N for batch), packet processor `subtract(1)`, consumer waits for 0.
- **Barrier dependency**: producer side decrements; barrier_and waits for all → 0.
- **Counting batch**: pre-set value = N, submit N dispatches all sharing that signal as their completion → wait for 0 once. Cheaper than per-kernel signals.

---

## 5. Memory pools and allocation

### 5.1 Discovery

```c
hsa_amd_agent_iterate_memory_pools(agent, cb, data);

// Per pool, query:
HSA_AMD_MEMORY_POOL_INFO_SEGMENT       // GLOBAL | KERNARG | GROUP | PRIVATE
HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS  // KERNARG_INIT | FINE_GRAINED | COARSE_GRAINED
HSA_AMD_MEMORY_POOL_INFO_SIZE
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED
HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE
```

### 5.2 Coarse-grained vs fine-grained device memory

| Property                         | Fine-grained          | Coarse-grained          |
|----------------------------------|-----------------------|--------------------------|
| Cache coherence with CPU         | Yes (via PCIe atomics)| No (GPU L2 holds dirty)  |
| Where it lives                   | Host (system RAM)     | VRAM (or system if APU)  |
| Throughput                       | PCIe-bound            | HBM/GDDR-bound           |
| Use for                          | sync ringbuffers, kernarg, hostcall | bulk tensors |
| Requires fence on access change  | No                    | Yes — release fence on dispatch |

The kernarg buffer is allocated from a **fine-grained host pool** so the CPU can write args and the GPU read them coherently without explicit flush. The doorbell page is similarly a coherent BAR mapping.

### 5.3 Allocation

```c
hsa_amd_memory_pool_allocate(pool, size, flags=0, &ptr);
hsa_amd_agents_allow_access(num_agents, agents, NULL, ptr); // grant GPU access
hsa_amd_memory_pool_free(ptr);
```

For coarse VRAM, `allow_access` with the GPU agent makes the buffer GPU-visible. For multi-GPU, you can pass multiple agents to make it peer-accessible.

### 5.4 Host-pinned + GPU-mapped (the kernarg ring pattern)

Allocate a single **kernarg ring** of `N × kernarg_segment_size` bytes from the kernarg fine-grained pool (or fine-grained system pool with `KERNARG_INIT`). For each in-flight dispatch, write into slot `i mod N`, set `packet.kernarg_address = ring + i*size`. After the corresponding `completion_signal` decrements, the slot is reusable.

```
+----------------+----------------+----------------+----+----------------+
|  kernarg[0]    |  kernarg[1]    |  kernarg[2]    | .. |  kernarg[N-1]  |
+----------------+----------------+----------------+----+----------------+
^
|  16-byte aligned, fine-grained, GPU-visible
```

Match `N` to the queue depth + small headroom.

### 5.5 `hsa_amd_memory_lock` for zero-copy

```c
hsa_amd_memory_lock(host_ptr, size, agents, num_agents, &gpu_visible_ptr);
```

Pins existing `malloc`'d host memory and maps it into the GPU's address space. The returned `gpu_visible_ptr` is what the kernel uses; the original host pointer remains valid for CPU access. Costs an `mlock` + IOMMU map. Faster than `hipMemcpyHtoDAsync` for one-shot read by the GPU when the data is already host-resident, slower than coarse VRAM if the GPU re-reads many times.

---

## 6. Code object loading

### 6.1 Modern API (use this)

```c
hsa_code_object_reader_t reader;
hsa_code_object_reader_create_from_memory(blob, blob_size, &reader);
// (or _create_from_file with a fd)

hsa_executable_t exe;
hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                          /*options=*/NULL, &exe);

hsa_executable_load_agent_code_object(exe, agent, reader, /*options=*/NULL,
                                      /*loaded_code_object=*/NULL);

hsa_executable_freeze(exe, /*options=*/NULL);

hsa_executable_symbol_t sym;
hsa_executable_get_symbol_by_name(exe, "my_kernel.kd", &agent, &sym);
// note: ".kd" suffix per the AMDGPU mangling

uint64_t kernel_object;
hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);

uint32_t kernarg_size, group_size, private_size, kernarg_align;
hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,      &kernarg_size);
hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT, &kernarg_align);
hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,        &group_size);
hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,      &private_size);
```

### 6.2 Legacy (avoid)

`hsa_executable_create` (no `_alt`) plus `hsa_code_object_t` (no `_reader`) — these are the pre-1.1 path; AMD still supports them but they go through additional copies.

### 6.3 Symbol info queries

| Query                                                          | Type      | Use |
|----------------------------------------------------------------|-----------|-----|
| `KERNEL_OBJECT`                                                | uint64    | `packet.kernel_object` |
| `KERNEL_KERNARG_SEGMENT_SIZE`                                  | uint32    | bytes to allocate per dispatch |
| `KERNEL_KERNARG_SEGMENT_ALIGNMENT`                             | uint32    | usually 16 |
| `KERNEL_GROUP_SEGMENT_SIZE`                                    | uint32    | static LDS bytes |
| `KERNEL_PRIVATE_SEGMENT_SIZE`                                  | uint32    | scratch per work-item |
| `KERNEL_DYNAMIC_CALLSTACK`                                     | bool      | needs scratch even if size=0 |

---

## 7. Kernel argument ABI (AMDGPU code object V5)

The kernarg buffer concatenates **explicit user arguments first, then the hidden ABI block**. Total size = `KERNEL_KERNARG_SEGMENT_SIZE` returned above. V5 hidden args sit at fixed offsets *relative to the start of the hidden block*, but the compiler tells you the absolute kernarg offset of each via the AMD metadata (NT_AMDGPU_METADATA ELF note, MessagePack format) — you read this once and cache it.

For most LLVM-built code object V5 kernels, the hidden block layout (offsets are *within hidden block*; the compiler typically places the block at offset 0 if there are no user args, or after user args rounded up to 8) is:

| Hidden arg               | Offset | Size (bytes) | Meaning |
|--------------------------|-------:|-------------:|---------|
| `hidden_block_count_x`   | 0      | 4            | grid_size_x / wg_size_x (ceiling) |
| `hidden_block_count_y`   | 4      | 4            | grid_size_y / wg_size_y |
| `hidden_block_count_z`   | 8      | 4            | grid_size_z / wg_size_z |
| `hidden_group_size_x`    | 12     | 2            | wg_size_x as actually launched |
| `hidden_group_size_y`    | 14     | 2            |  |
| `hidden_group_size_z`    | 16     | 2            |  |
| `hidden_remainder_x`     | 18     | 2            | grid_x % wg_x |
| `hidden_remainder_y`     | 20     | 2            |  |
| `hidden_remainder_z`     | 22     | 2            |  |
| (reserved / padding)     | 24     | 16           |  |
| `hidden_global_offset_x` | 40     | 8            | usually 0 |
| `hidden_global_offset_y` | 48     | 8            |  |
| `hidden_global_offset_z` | 56     | 8            |  |
| `hidden_grid_dims`       | 64     | 2            | 1, 2, or 3 |
| (padding)                | 66     | 6            |  |
| `hidden_printf_buffer`   | 72     | 8            | VA of printf buffer or 0 |
| `hidden_hostcall_buffer` | 80     | 8            | VA of hostcall buffer or 0 |
| `hidden_multigrid_sync`  | 88     | 8            |  |
| `hidden_heap_v1`         | 96     | 8            | malloc/free heap base |
| `hidden_default_queue`   | 104    | 8            |  |
| `hidden_completion_action`| 112   | 8            |  |
| `hidden_dynamic_lds_size`| 120    | 4            | dynamic LDS bytes (matches packet.group_segment_size beyond static) |
| (reserved)               | 124    | …            |  |
| `hidden_private_base`    | (GFX8 only) | 4       | GFX8 legacy |
| `hidden_shared_base`     | (GFX8 only) | 4       | GFX8 legacy |
| `hidden_queue_ptr`       | (deprecated) | 8      | only if MD requests it |

Total hidden block on V5 modern GFX10/RDNA2: **256 bytes** padded.

[Source: `llvm/test/CodeGen/AMDGPU/hsa-metadata-hidden-args-v5.ll`, plus AMDGPU LLVM backend `lib/Target/AMDGPU/AMDGPUHSAMetadataStreamer.cpp`.]

**Practical rule**: do not hard-code offsets. Read the kernel's ELF metadata once at load time and cache an `(offset, size)` table per hidden arg. If you skip this and rely on guessed offsets, a compiler version bump silently corrupts your launches. If you absolutely must avoid metadata parsing, the offsets above hold for clang ≥ 14 in code object V5 mode for RDNA2.

### 7.1 Kernarg buffer construction

```
| user_arg_0 | user_arg_1 | ... | user_arg_N | pad | hidden_block_count_x | ... |
^                                                  ^
|                                                  |
kernarg_address                                    kernarg_address + hidden_offset
```

User args are packed per their natural alignment from the kernel's source (HIP/OpenCL semantics): each pointer 8-byte aligned, each scalar at its natural alignment, total padded to 16. The compiler's metadata gives byte offset and size for *every* arg, hidden or not — read it once.

### 7.2 Worked example — single buffer kernel

```hip
__global__ void scale(float* x, float a, uint32_t n) { ... }
```

V5 layout (typical):
```
offset  size  field
0       8     x          (pointer)
8       4     a          (float)
12      4     n          (uint32)
16     32     pad
48      4     hidden_block_count_x
52      4     hidden_block_count_y
56      4     hidden_block_count_z
60      2     hidden_group_size_x
... (rest of hidden block)
total: 16 user + 240 hidden = 256, 16-byte aligned
```

`KERNEL_KERNARG_SEGMENT_SIZE` returns 256. You allocate 256 bytes, fill in slots 0/8/12 from your user args, and slots 48..255 from the launch geometry.

---

## 8. Async batching pattern

The performance unlock: **submit N dispatches, then one barrier_and, then wait once**.

```
Queue:
  [DISPATCH 0, comp_sig=NULL, BARRIER=0]
  [DISPATCH 1, comp_sig=NULL, BARRIER=0]
  ...
  [DISPATCH N-1, comp_sig=NULL, BARRIER=0]
  [BARRIER_AND, dep=NULL, completion_signal=batch_sig (init=1)]
host: hsa_signal_wait_scacquire(batch_sig, EQ, 0)
```

Why this is fast:
- Only **one** signal is allocated per batch (vs. N).
- The CPU only does **one** wait syscall.
- The GPU can pipeline within the queue freely because every dispatch has BARRIER=0 — the firmware can issue successor packets while predecessor wave is still draining (limited by resource availability).
- The terminating barrier_and forces the comp signal to fire only after all prior packets retire.

If kernels have data dependencies, set `BARRIER=1` on the dependent packet (cheaper than chained signals) or use chained completion_signal → barrier_and dep_signal patterns:

```
[DISPATCH A, comp_sig=sigA]
[BARRIER_AND, dep=[sigA], comp_sig=sigB]   <- equivalent to "wait for A"
[DISPATCH C, kernarg uses A's output]
```

GPU hardware reordering: within a queue with BARRIER=0, the MES scheduler may issue work back-to-back, but observable order at memory is governed by the release fence on the producer header and acquire fence on the consumer header. With both `acquire=SYSTEM, release=SYSTEM` between dependent kernels, you get C++-equivalent ordering [HSA PSAS §3.4]. With both = NONE, the GPU is free to reorder visible writes — only do this for fully independent kernels.

### 8.1 The 1.4×–6.3× HIP→HSA story

Measured on Qwen2.5-1.5B WGSL matmul kernels (project-internal, see `project_lsh_finding.md`):

- HIP path (single dispatch): ~12 µs end-to-end host time per launch.
- HSA AQL direct (single dispatch): ~8.5 µs (1.4×).
- HSA AQL batched (32 dispatches + 1 barrier): ~1.9 µs/launch amortized (6.3×).

The batched gain is mostly removed CPU work + removed signal handling; the 1.4× single-launch gain is removed C++ object allocation in ROCclr.

---

## 9. Multi-queue strategies on RDNA2

RDNA2 compute pipe topology (gfx1030, 40 CUs, 4 SE × 10 WGP × 2 CU):

```
MEC0 (Micro-Engine Compute 0)
  Pipe 0: 8 HQD slots
  Pipe 1: 8 HQD slots
  Pipe 2: 8 HQD slots
  Pipe 3: 8 HQD slots
MES (Micro-Engine Scheduler)  -- virtualizes more than 32 software queues
GFX  (Graphics ME)            -- separate from compute, used for Vulkan graphics
```

When you create `hsa_queue_t`s on the GPU agent, KFD assigns each one to an MEC pipe (round-robin or by priority). With **2 queues**, kernel A on queue Q0 can run concurrently with kernel B on queue Q1 if CU resources are available (each kernel allocates VGPRs/SGPRs/LDS from a shared pool).

Heuristic for an LLM/transformer workload:
- 1 queue: matmuls, sequential.
- 2 queues: matmul-stream + KV-cache-DMA stream (overlap copy with compute).
- 4+ queues: rarely helps; the CU pool becomes the bottleneck.

Priority: `hsa_amd_queue_set_priority(queue, HSA_AMD_QUEUE_PRIORITY_HIGH)` raises MES scheduling weight. Useful for "interactive token" + "background prefill".

---

## 10. Practical gotchas

1. **Queue overflow** (write_index outpacing GPU consumption). The ring is finite. Always backpressure: `while (slot - read_index >= queue->size) yield;`. The read index is *not* directly exposed; on AMD ROCR you can derive it from `queue->doorbell_signal` value (latest packet GPU has been told about) plus inspecting completion signals. Pragmatic approach: allocate the queue large enough (4096 slots = 256 KB) and cap inflight by counting completions.

2. **Stale doorbell writes**. Writing an older index than the firmware has already seen is silently ignored. Writing a *much larger* index (skipping slots) tells the firmware to consume slots whose headers are still INVALID — it will spin. Solution: only doorbell after writing the header.

3. **Header-vs-body race**. If you do not use a release-store on the header (or equivalent), the packet processor can read your packet mid-construction. Always atomic-store the header *last*. Some implementations mandate that the header byte be `INVALID` on first init; ROCR clears the ring to all-zeros, which makes header.type = `VENDOR_SPECIFIC=0`. Vendor-specific packets are skipped by AMD's MEC firmware, so this is safe — but writing your own zero-init code, prefer to fill the ring with `INVALID(1)` first to be portable.

4. **Signal value race**. Reusing a completion signal across batches without resetting its value to N first → wait may immediately return on stale 0. Always `hsa_signal_store_screlease(sig, N)` before submitting batch N.

5. **Page fault / OOM in kernel**. The GPU raises a *retry* page fault (XNACK) if the kernel touches an unmapped page; without HMM enabled, this becomes a **memory access fault** that hangs the queue. KFD eventually times out and kills the queue; from your host process you'll see `HSA_STATUS_ERROR_EXCEPTION` returned on subsequent operations. Recovery: destroy the queue, recreate. Set `HSA_ENABLE_DEBUG=1` to surface kernel asserts.

6. **`hsa_status_t` ambiguities**: `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` (returned on AQL submission failure) is overloaded — could be unaligned kernarg, bad kernel_object VA, or a mis-encoded header. `HSA_STATUS_ERROR_OUT_OF_RESOURCES` from `hsa_signal_create` typically means KFD events exhausted, not memory. `HSA_STATUS_ERROR_FATAL` after a kernel hang means the queue is unrecoverable.

7. **`completion_signal = {0}` is not "no signal"**. The spec says `0` handle means "no signal", and ROCR honors that — but a signal with value 0 and a non-zero handle is a *different* situation (consumer immediately sees 0 and proceeds). Be explicit: zero the field if you don't want a signal.

8. **`BARRIER` bit and barrier_and packet are different things**. The header BARRIER bit on a kernel_dispatch means "stall this packet's launch until all prior packets in this queue retire". The BARRIER_AND packet type means "wait on these explicit signals before letting later packets in this queue launch". They compose.

9. **`HSA_PACKET_TYPE_INVALID` is meaningful**. Do not "leave header alone" between submissions. Either zero the slot before reuse or write an explicit `INVALID` to defeat the consumer.

10. **Doorbell page lifetime**. KFD allocates one doorbell page per process. If you fork without exec, the child cannot use the queues — the doorbell page is process-private. Multi-process GPU sharing requires per-process queues.

---

## 11. Comparison: AQL vs HIP vs CUDA Driver vs OpenCL vs Vulkan

| Mechanism                        | Kernel launch CPU time (host-only, single dispatch)    | Setup overhead | Driver path | Programmability |
|----------------------------------|-------------------------------------------|----------------|-------------|-----------------|
| HSA AQL direct                   | **0.5–1.5 µs**                            | High (write your own dispatcher) | userspace ring + 1 MMIO | Hard |
| HIP (ROCm 6.x)                   | **8–12 µs** typical, **~70 µs** worst-case [Kokkos #3670] | Low (familiar) | ROCclr → AQL | Easy |
| CUDA Driver API (`cuLaunchKernel`) | **~3 µs** on H100 hosts (NVIDIA SO numbers) | Low | CUDA driver → SM | Easy |
| OpenCL (`clEnqueueNDRangeKernel`) | **20–40 µs** on AMD ROCm OpenCL | Low | OpenCL → ROCclr → AQL | Easy |
| Vulkan compute                   | **~5–15 µs** depending on impl | High (descriptor sets) | Vulkan driver → graphics ring | Hard |
| Direct PM4 (below AQL)           | **~0.3 µs**                               | Extreme (build PM4 yourself) | KFD ioctls | Brutal |

Numbers are order-of-magnitude indicators on a typical desktop CPU + RDNA2 dGPU; vary heavily with CPU model, NUMA, kernel size, and PCIe gen. The *relative* ordering is stable: PM4 < AQL < HIP/CUDA < OpenCL.

The 70 µs HIP figure is from the Kokkos investigation and represents an unoptimized HIP path; modern ROCm 6.x is closer to 8–12 µs amortized when the runtime caches modules and kernels.

---

## 12. Authoritative references

### Specifications

- HSA Platform System Architecture Specification v1.2 — `http://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf`
- HSA PRM v1.2 — `http://hsafoundation.com/wp-content/uploads/2021/02/HSA-PRM-1.2.pdf`
- HSA PSAS v1.0 (older but cited in old code) — `http://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.01.pdf`

### ROCR-Runtime (HSA implementation)

- Repository: `https://github.com/ROCm/rocm-systems` (current; `ROCR-Runtime` repo deprecated, mirror at `https://github.com/RadeonOpenCompute/ROCR-Runtime`)
- `runtime/hsa-runtime/inc/hsa.h` — core API, struct layouts, enum bit positions
- `runtime/hsa-runtime/inc/hsa_ext_amd.h` — AMD vendor extensions (memory pools, signal create, queue priority)
- `runtime/hsa-runtime/inc/amd_hsa_kernel_code.h` — `amd_kernel_code_t` (the descriptor)
- `runtime/hsa-runtime/inc/amd_hsa_signal.h` — `amd_signal_t` internal layout (signal kind, event, value addr)
- `runtime/hsa-runtime/inc/amd_hsa_queue.h` — `amd_queue_t` extended fields
- `runtime/hsa-runtime/src/core/runtime/amd_aql_queue.cpp` — reference producer logic, doorbell ring
- API doc: `https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/api-reference/api.html`

### AMDGPU code object & ABI

- LLVM AMDGPU User Guide (current): `https://llvm.org/docs/AMDGPUUsage.html`
- ROCm-hosted mirror: `https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html`
- Hidden args V5 reference test: `https://llvm.googlesource.com/llvm-project/llvm/+/refs/heads/main/test/CodeGen/AMDGPU/hsa-metadata-hidden-args-v5.ll`
- Backend metadata streamer source (for hidden arg names/layout truth): `llvm/lib/Target/AMDGPU/AMDGPUHSAMetadataStreamer.cpp`

### KFD (kernel driver, doorbell, events)

- amdkfd source: `drivers/gpu/drm/amd/amdkfd` in Linux kernel tree
- Felix Kuehling's HSA/KFD observations: `https://gist.github.com/fxkamd/ffd02d66a2863e444ec208ea4f3adc48`
- DeepWiki summary: `https://deepwiki.com/ROCm/rocm-systems/4-rocr-runtime-(hsa-runtime)`

### Cited benchmarks

- "Exploring AMD GPU Scheduling Details by Experimenting With Worst Practices" — `https://dl.acm.org/doi/fullHtml/10.1145/3453417.3453432`
- Kokkos issue #3670 "HIP Launch Latency Investigation" — `https://github.com/kokkos/kokkos/issues/3670`

### RDNA2 ISA / hardware

- AMD RDNA2 ISA Reference Guide — `https://www.amd.com/system/files/TechDocs/rdna2-shader-instruction-set-architecture.pdf`
- gem5 AQL doxygen (independent reference impl): `http://doxygen.gem5.org/release/current/group__aql.html`

---

## Appendix A — Minimal end-to-end Rust producer skeleton

```rust
// Pseudo-Rust; assumes hsa-sys bindings (see crates: hsa, hsakmt-sys).
// Production code should batch dispatches; this is the single-shot path.

unsafe fn launch(
    queue: *mut hsa_queue_t,
    kernel_object: u64,
    kernarg: *mut u8,
    kernarg_size: u32,
    grid: (u32, u32, u32),
    wg: (u16, u16, u16),
    lds_bytes: u32,
    completion: hsa_signal_t,
) {
    let slot = hsa_queue_add_write_index_screlease(queue, 1);

    // Backpressure (omitted: poll until consumer caught up).
    let size = (*queue).size as u64;
    let base = (*queue).base_address as *mut hsa_kernel_dispatch_packet_t;
    let p = base.add((slot & (size - 1)) as usize);

    (*p).workgroup_size_x = wg.0;
    (*p).workgroup_size_y = wg.1;
    (*p).workgroup_size_z = wg.2;
    (*p).reserved0        = 0;
    (*p).grid_size_x      = grid.0;
    (*p).grid_size_y      = grid.1;
    (*p).grid_size_z      = grid.2;
    (*p).private_segment_size = 0;
    (*p).group_segment_size   = lds_bytes;
    (*p).kernel_object        = kernel_object;
    (*p).kernarg_address      = kernarg as _;
    (*p).reserved2            = 0;
    (*p).completion_signal    = completion;

    // Compose 32-bit (setup << 16) | header.
    let header: u16 =
          (HSA_PACKET_TYPE_KERNEL_DISPATCH as u16) << 0
        | (1u16 << 8)                                       // BARRIER (safe default)
        | (HSA_FENCE_SCOPE_SYSTEM as u16) << 9              // acquire
        | (HSA_FENCE_SCOPE_SYSTEM as u16) << 11;            // release
    let setup: u16 = if grid.2 > 1 { 3 } else if grid.1 > 1 { 2 } else { 1 };
    let header_setup: u32 = ((setup as u32) << 16) | (header as u32);

    let header_ptr = p as *mut u32;
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    (header_ptr).write_volatile(header_setup);  // release store of header

    // Doorbell. ROCR's signal store dispatches to the doorbell MMIO write path.
    hsa_signal_store_screlease((*queue).doorbell_signal, slot as i64);
}
```

---

## Appendix B — Header word quick reference

```
header (16 bits, little-endian byte 0..1 of the packet):

bits  | value | meaning
------+-------+--------------------------------------
 0..7 | 0     | VENDOR_SPECIFIC
 0..7 | 1     | INVALID (consumer waits)
 0..7 | 2     | KERNEL_DISPATCH
 0..7 | 3     | BARRIER_AND
 0..7 | 4     | AGENT_DISPATCH
 0..7 | 5     | BARRIER_OR
 8    | 0/1   | BARRIER bit (in-queue ordering)
 9..10| 0/1/2 | acquire fence scope (NONE/AGENT/SYSTEM)
11..12| 0/1/2 | release fence scope (NONE/AGENT/SYSTEM)
13..15| 0     | reserved

Useful constants for a "safe default" kernel dispatch:
  KERNEL_DISPATCH | BARRIER=1 | acq=SYSTEM | rel=SYSTEM
  = 2 | (1 << 8) | (2 << 9) | (2 << 11)
  = 0x1502  (decimal 5378)

For a "fully relaxed" overlap-friendly batch element:
  KERNEL_DISPATCH | BARRIER=0 | acq=NONE | rel=NONE
  = 0x0002
```

---

**End of reference.** Anything missing is either in the cited PSAS/PRM PDFs or in the live ROCR header. When the AMDGPU backend bumps to V6 (anticipated post-ROCm 7), re-validate §7 hidden-arg offsets against `AMDGPUHSAMetadataStreamer.cpp` HEAD.
