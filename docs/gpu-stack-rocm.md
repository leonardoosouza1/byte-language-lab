# GPU stack — migração wgpu → HSA AQL "bit-polishing"

**Status:** decisão arquitetural tomada 2026-05-07. wgpu/WGSL **e** HIP module API fora do hot path. Dispatch via **HSA AQL packets escritos direto no ring buffer da hardware queue**, com doorbell signal. HIP usado **apenas** offline (build-time `hipcc` para gerar `.hsaco`).

**Hardware alvo:** AMD RX 6750 XT, RDNA2, gfx1031, 40 CU, 96 MB Infinity Cache, wave32 nativo.

**ROCm em uso no host:** 7.2.1 (`/opt/rocm-7.2.1`).

---

## Por que sair do wgpu/Vulkan — e por que HIP módulo também não basta

Latência medida no `train_step` SMG v3 hoje (Vulkan/RADV/Linux):

| Submit | Custo medido |
|---|---|
| forward + readback logits | ~50 ms |
| backward + sumsq + readback norm | ~50 ms |
| 7×SGD + 7×zero | ~30 ms |
| **Total/step** | **~570 ms** (com overhead host) |

Driver Vulkan + RADV no Linux tem 20–50 ms de overhead por submit/readback. wgpu adiciona mais uma camada por cima (validação, command encoder, bind groups transientes). Não dá pra otimizar embaixo desse piso sem trocar de stack.

**Comparativo medido no llm-lab** (mesmo HW, RDNA2):

| Path | Latência por launch |
|---|---|
| wgpu/Vulkan + RADV (atual) | ~50 ms |
| HIP `hipModuleLaunchKernel` sync | **16.33 µs** |
| HSA AQL sync packet | **11.96 µs** (1.4× HIP) |
| **HSA AQL async + barrier no fim do batch** | **2.61 µs** (6.3× HIP) ← target |

**Por que HIP módulo também não foi suficiente:** HIP força `hipStreamSynchronize` ou graph capture por sequência. Graph capture tem overhead próprio para sequências dinâmicas. AQL queue é hardware queue — você empilha N pacotes async, GPU consome via DMA do ring, você bota um `barrier_and` packet no fim e espera **um signal só**. Zero supervisão CPU enquanto o batch roda.

Esse é o "escovar bit": fora HSA AQL só sobrariam ioctls KFD não-documentados (`/dev/kfd`) — que é o que ROCR usa internamente. Não vale.

---

## Stack escolhido

```
.hip (HIP C++)  ──hipcc──▶  .hsaco (ELF gfx1030)  ──include_bytes!──▶  binário Rust
                                                                              │
                                                                              ▼
                                                              hsa_executable_create_alt
                                                              hsa_executable_load_agent_code_object
                                                              hsa_executable_get_symbol_by_name
                                                              ↓
                                                              kernel_object: u64 (GPU virt addr)
                                                              ↓
                                                              [a cada launch]
                                                              ring slot = base + (idx & mask)
                                                              memcpy AQL packet body (56 B)
                                                              atomic store header (Release)   ← GPU vê
                                                              atomic store doorbell           ← GPU acorda
```

- **Compile-time:** `build.rs` invoca `hipcc --offload-arch=gfx1030 -O3 -mno-wavefrontsize64 -ffast-math --genco -o $OUT_DIR/<kernel>.hsaco`. (Em RDNA2, `gfx1030` e `gfx1031` compartilham ISA — usamos `gfx1030` por compatibilidade com a base do llm-lab.)
- **Runtime, init (1×):**
  - `hsa_init()` + `hsa_iterate_agents` → pega o GPU agent + memory pools
  - `hsa_queue_create(64 slots, MULTI)` → ring buffer de pacotes AQL (cada slot = 64 bytes), mapeado em userspace tanto CPU quanto GPU
  - `hsa_executable_create_alt` + `load_agent_code_object(<bytes>)` → carrega `.hsaco` na GPU
  - `hsa_executable_get_symbol_by_name` por kernel → cacheia `kernel_object: u64` (endereço GPU do kernel descriptor)
  - Kernarg ring buffer pré-alocado (host-mapped, GPU-readable)
- **Runtime, dispatch (cada launch ~2.6 µs amortizado):**
  - `dispatch_async("smg_forward", grid, block, args)`:
    1. Reserva slot no kernarg ring, escreve args + hidden ABI fields (`block_count_x/y/z`, `group_size_x/y/z`, etc.)
    2. Reserva slot no AQL ring buffer (`base + (write_index & mask)`)
    3. Escreve corpo do pacote (kernel_object, dims, kernarg_addr, completion_signal)
    4. **Store atômico do header com `Ordering::Release`** ← GPU vê pacote válido
    5. Store atômico no `doorbell_signal` ← firmware é acordado, lê do ring
  - Sequência de dispatches: enfileira N async, depois `submit_barrier_and_wait` (1 packet de barrier_and + spin/wait num signal). **1 sync por sequência**, não por kernel.

### AQL packet body (referência)

```
struct hsa_kernel_dispatch_packet_t {  // 64 bytes
    u16  header;                       // type=KERNEL_DISPATCH | barriers | scope
    u16  setup;                        // dim count
    u16  workgroup_size_x/y/z;
    u16  reserved;
    u32  grid_size_x/y/z;
    u32  private_segment_size;
    u32  group_segment_size;           // LDS bytes
    u64  kernel_object;                // GPU virt addr (do .hsaco)
    u64  kernarg_address;              // GPU virt addr do kernarg slot
    u64  reserved2;
    hsa_signal_t completion_signal;    // 0 = no signal
}
```

### Reuso direto: `llm-lab/hip-ffi`

**Pattern já existe e funciona** em `/home/leonardo/projects/LLM/llm-lab/hip-ffi/`:

| Arquivo | Função | Linhas |
|---|---|---|
| `build.rs` | bindgen wrapper.h + invoca hipcc por kernel | 368 |
| `src/raw.rs` | tipos FFI gerados (`hipFunction_t`, `hipStream_t`, …) | (gerado) |
| `src/safe.rs` | wrappers seguros (`hip_module_load_data`, `hip_launch_kernel`) | 323 |
| `src/kernels.rs` | `GpuKernels` struct, carrega módulo + cacheia handles | 5 064 |
| `src/blas.rs` | hipBLAS wrapper opcional | 225 |
| `src/hsa_dispatch.rs` | AQL packet dispatch raw (path "ainda mais rápido") | 727 |

**Decisão:** adicionar `hip-ffi` como path-dep no Cargo.toml deste projeto. Não copiar nem reimplementar bindgen.

```toml
# byte-language-lab/Cargo.toml (novo)
[dependencies]
hip-ffi = { path = "../llm-lab/hip-ffi" }
```

Nossos kernels SMG ficam em `src/gpu/kernels/*.hip`, compilados pelo nosso próprio `build.rs` que copia o pattern do hip-ffi.

---

## RDNA2 (gfx1031) — o que importa para os kernels

### Wavefront size

- **wave32 nativo.** Use `-mno-wavefrontsize64` (default em LLVM ≥ 12 para gfx10+, mas explicitar não machuca).
- Wave64 emula via two-pass; só vale a pena para GEMV bandwidth-bound (vide `gemv_w64.hip` em llm-lab).
- Para nosso SMG (1 workgroup/sample, 128 threads = 4 waves), wave32 é o caminho.

### LDS

- **128 KB por WGP** (= 2 CUs). 1 workgroup vê **até 64 KB safely**.
- 40 CUs → 20 WGPs. Ocupação típica: 2 workgroups por WGP simultâneos com 32 KB LDS cada.
- Nosso forward usa 32 KB (state_curr 16 KB + lat_proj 16 KB). Backward usa 36 KB. **Cabe.**

### Registradores

- 256 VGPRs/lane, 106 SGPRs por wave. RDNA2 NÃO é register-starved como GCN.
- Aim: 8 waves/SIMD (= ocupância máxima). Compilador HIP geralmente acerta com `-O3`.

### Atomic float — **TRAP**

> **`global_atomic_add_f32` NÃO existe em gfx1031.**

- Disponível em gfx90a (CDNA2) e gfx11+ (RDNA3). RDNA2 não tem.
- Para acumular f32 em VRAM: **CAS loop sobre `atomicCAS` na bit-pattern u32** (mesmo approach que já usamos em WGSL).
- Para acumular f32 em **LDS**: `ds_add_f32` é nativo em RDNA2 — usar para reduções intra-workgroup.
- `flat/global_atomic_fmin/fmax` (f32) e `flat_atomic_fcmpswap` existem; `atomicCAS` em u32 também é nativo.

**Implicação prática:** o backward continua com a mesma estrutura de CAS-loop por gradiente que já temos no WGSL. Não muda o algoritmo, só a sintaxe (HIP C++ em vez de WGSL).

### dp4a / DOT instructions

- `v_dot4_i32_i8` e `v_dot2_f32_f16` funcionam em wave32 — confirmado em ONDA/llama.cpp-turboquant.
- Não precisamos por ora (SMG é f32 puro), mas fica anotado para quantização futura.

### Infinity Cache

- 96 MB transparente. Tune para L2 (4 MB). Reduções stage-via-LDS continuam ganhando.

---

## Lançamento de kernel — anatomia

```rust
// Pseudo-API (espelha hip-ffi::safe)
unsafe fn launch_forward(
    kernels: &SmgKernels,
    grid: u32,            // = batch_size
    block: u32,           // = 128
    shared_bytes: u32,    // = 32_768
    stream: hipStream_t,
    args: &mut [*mut c_void; N],   // ponteiros para cada kernel arg
) -> Result<(), HipError> {
    hip_launch_kernel(
        kernels.fn_forward,
        (grid, 1, 1),
        (block, 1, 1),
        shared_bytes,
        stream,
        args,
    )
}
```

**Chave:** **um único stream**, todos os kernels do `train_step` enfileirados sem sync intermediário. Sync **só** no final (`hipStreamSynchronize`) ou quando precisar do loss/norm na CPU.

### Eliminação de readbacks

- **Loss na GPU:** kernel pequeno reduz `sum_loss` para 1 f32 em VRAM. Readback **a cada N batches** (configurável), não por step.
- **Norm para clipping:** já é 1 f32. Lê 1× por step se clipping ativo. Em ROCm o `hipMemcpyAsync` + 1 sync custa ~10 µs (vs ~50 ms hoje).
- **Forward/backward chain:** logits ficam em VRAM, backward lê direto do mesmo buffer. Zero CPU↔GPU no caminho.

---

## Plano de portabilidade (alto nível)

| Etapa | Hoje (wgpu) | Depois (HSA AQL) |
|---|---|---|
| Init | `wgpu::Instance::new` + `request_adapter` + `request_device` | `hsa_init()` + `hsa_iterate_agents` + `hsa_queue_create` (via `HsaDispatcher::new` do hip-ffi) |
| Buffer alloc | `device.create_buffer(&desc)` | `hipMalloc(&ptr, bytes)` (HIP runtime usado só pra alocação; o dispatch é HSA) |
| Upload | `queue.write_buffer` | `hipMemcpyAsync(dst, src, bytes, HostToDevice, stream)` |
| Compile shader | `device.create_shader_module(WGSL)` | build-time: `hipcc → .hsaco`; runtime: `include_bytes!` + `hsa_executable_load_agent_code_object` |
| Bind group | `create_bind_group([buf, …])` | nenhum — args são ponteiros escritos no kernarg ring buffer |
| Dispatch | `pass.set_pipeline + set_bind_group + dispatch_workgroups` (CPU↔GPU sync) | `dispatch_async(name, grid, block, args)` → escreve AQL packet + doorbell (sem sync) |
| Sync de batch | `queue.submit` + readback síncrono | `submit_barrier_and_wait()` (1 sync no fim de N kernels async) |
| Readback | `map_async` + `device.poll(Wait)` | `hipMemcpyAsync(D2H)` + signal wait |
| f32 atomic add (VRAM) | CAS loop em WGSL | CAS loop em HIP C++ (mesma lógica — RDNA2 não tem `global_atomic_add_f32`) |
| f32 atomic add (LDS) | CAS loop em WGSL | `atomicAdd(&shared_buf[i], val)` → compila pra `ds_add_f32` (nativo RDNA2) |
| LDS | `var<workgroup>` | `__shared__` |
| Workgroup barrier | `workgroupBarrier()` | `__syncthreads()` |
| Lane id (wave32) | `local_invocation_id` | `threadIdx.x`; wave-level via `__lane_id()` |

**Kernels a portar:** 4 arquivos WGSL → 4 arquivos `.hip`:

1. `microcolumn_v3_forward.wgsl` → `kernels/smg_forward.hip`
2. `microcolumn_v3_backward.wgsl` → `kernels/smg_backward.hip`
3. `microcolumn_v3_sgd.wgsl` → `kernels/smg_sgd.hip` (sgd_main + zero_grads)
4. `microcolumn_v3_norm.wgsl` → `kernels/smg_norm.hip` (sumsq_main + zero_acc)

Estrutura de cada kernel não muda — workgroup_size, LDS layout, atomic CAS loop. É tradução sintática.

---

## Bookmarks (consulta futura)

### AMD oficial
- **HIP runtime API (Module group):** https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___module.html
- **HIP programming guide:** https://rocm.docs.amd.com/projects/HIP/en/latest/
- **GPU arch specs (LDS/CU/wavefront por gen):** https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
- **RDNA2 ISA PDF:** https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf

### LLVM AMDGPU
- **AMDGPU backend overview:** https://llvm.org/docs/AMDGPUUsage.html
- **gfx1030/1031 ISA syntax:** https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX1030.html

### Rust ecosystem
- **cubecl-hip-sys:** https://github.com/tracel-ai/cubecl-hip-sys (alternativa se quiséssemos crate publicado em vez de path-dep)
- **rustc amdgcn target (Tier 3, nightly):** https://doc.rust-lang.org/rustc/platform-support/amdgcn-amd-amdhsa.html
- **Tracking issue Rust on AMDGPU:** https://github.com/rust-lang/rust/issues/51575
- **hipBench (medir overhead):** https://github.com/ROCm/hipBench

### Internos do workspace
- **hip-ffi (crate base, expõe HSA dispatcher):** `~/projects/LLM/llm-lab/hip-ffi/`
- **HSA AQL dispatcher (target deste projeto):** `~/projects/LLM/llm-lab/hip-ffi/src/hsa_dispatch.rs` (727 linhas)
  - `HsaContext::init()` linhas 125–167 — agent + queue + kernarg ring
  - `HsaContext::dispatch_sync()` linhas 357–425
  - `HsaContext::dispatch_async()` linhas 255–310 — hot path
  - `extract_amdgcn_elf()` linhas 428–457 — descasca offload bundle do `.hsaco`
  - `HsaDispatcher` linhas 473–718 — API de alto nível
- **Smoke/bench HIP vs HSA:** `~/projects/LLM/llm-lab/hip-ffi/examples/hsa_smoke.rs` (referência de medição)
- **Exemplo de kernels f32/quantizados RDNA2:** `~/projects/LLM/llm-lab/hip-ffi/kernels/{llm_kernels,gemv_w64,mmvq_llama}.hip`
- **rocm-backend (uso de hip-ffi com `IARA_HSA=1`/`IARA_HSA_FAST=1`):** `~/projects/LLM/llm-lab/rocm-backend/`
- **Memo do projeto sobre RDNA2:** `~/.claude/projects/-home-leonardo-projects-LLM/memory/project_rdna2_wave32_discovery.md`

---

## Não fazer (por enquanto)

- **rustc amdgcn target.** Tier 3, nightly, sem `compiler_builtins` ainda. Útil só para kernel trivial. Não vale o yak-shaving.
- **Raw HSA AQL.** Ganho marginal vs HIP (~5 µs vs ~10 µs por launch). Só justifica depois de confirmar que HIP é o gargalo. `hsa_dispatch.rs` do hip-ffi já está pronto se precisar.
- **Inline AMDGCN asm a partir do Rust.** Não há syntax estável para registradores vetoriais via Rust. Se precisar, escrever em HIP C++ com `asm volatile`.
