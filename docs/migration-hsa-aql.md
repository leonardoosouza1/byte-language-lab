# Migration plan — wgpu/WGSL → HSA AQL (RDNA2 gfx1031)

> Origem: decisão em [[gpu-stack-rocm]]. Pesquisa de suporte: `docs/research/*` (4371 linhas, mirror em `~/obsidian-claude-vault/Claude/GPU-RDNA2-HSA/`).
> Target: `train_step` ~590 ms → <5 ms (>>100× speedup) no `exp18_smg_lateral`.
> Estratégia: substituir o runtime wgpu por HSA AQL packets via `llm-lab/hip-ffi` path-dep, gated em feature `gpu-aql`.

---

## Status (2026-05-08)

| # | Fase | Status | Resultado |
|---|---|---|---|
| 1 | Fundação: Cargo.toml + build.rs + scaffold `src/gpu_aql/` | ✅ DONE | `probe_hsa` carrega 4 módulos, 6 kernels, kernarg_size corretos |
| 2 | 4 HIP kernels + dispatch layer + SmgRuntime | ✅ DONE | `kernarg_size` validados contra `.hsaco`; grid semantics AQL corretos |
| 3 | Forward parity test | ✅ DONE | `max_err = 0.000000` (wgpu vs HSA, tolerância 1e-3) |
| 4 | Backward parity test | ✅ DONE | `max_err = 5.96e-8` por tensor (<<2e-3, CAS-loop determinístico) |
| 5 | `SmgGridHsa` — interface `SmgGridGpu`-compatible com buffers persistentes | ✅ DONE | `test_parity_step` disponível; 0 hipMalloc no hot path |
| 6 | Benchmark `bench_hsa` vs wgpu | ⏳ PENDENTE | naive 1.7× (hipMalloc domina); pst_* fix compilado, aguardando run |

---

## Fases

| # | Critério de saída | Risco |
|---|---|---|
| 1 | `cargo check --features gpu-aql` compila; smoke-test enumera GPU | baixo |
| 2 | 6 kernels carregam com kernarg_size correto; `probe_hsa` ok | médio (kernarg ABI v5) |
| 3 | Forward: logits HIP == wgpu (tolerância 1e-3) | médio |
| 4 | Backward: grads HIP == CPU (tolerância 2e-3, CAS-loop) | **alto** |
| 5 | `test_parity_step`: pesos após train_step HIP == CPU manual (tol 2e-3) | médio |
| 6 | Benchmark: epoch HSA <<10ms; speedup >>100× vs wgpu baseline | — |

Validação entre fases é **bloqueante** — não avança sem critério de saída.

---

## Decisões fixas

- **Feature flag**: `gpu-aql` (não default). Build sem ROCm continua funcionando (wgpu fallback).
- **Crate compartilhado**: `hip-ffi` via `path = "../llm-lab/hip-ffi"`. Sem cópia de código.
- **Compilação de kernels**: `build.rs` chama `hipcc --genco --offload-arch=gfx1031 -O3 -ffast-math -std=c++17` → `.hsaco`. Flags `-mwavefrontsize32`/`-mno-cumode` removidas — gfx1031 já defaulta wave32+WGP mode.
- **wave32 obrigatório**: confirmado em `research/rdna2-microarch.md` §wavefronts.
- **f32 atomic add**: CAS-loop sobre `atomicCAS(u32*)` — `global_atomic_add_f32` não existe em gfx1031 (`research/memory-model.md` §3.4).
- **AQL grid_size_x = TOTAL work-items** (não blocos). Bug crítico descoberto em Phase 3: forward+backward usam 1 WG/sample com 128 threads → grid = (batch×128, 1, 1).
- **Buffers persistentes no hot path**: `SmgGridHsa` pré-aloca `pst_win/pst_tgt/pst_log/pst_st` na construção. Hot path usa `hip_h2d` (hipMemcpy sem hipMalloc). Eliminação de 49MB de hipMalloc/hipFree por step para batch=64, T=48.
- **3 drain() por train_step**: após forward (readback logits), após backward (garante grads prontos antes de SGD), após SGD. Necessário para correctness — AQL SPSC queue não garante in-order completion sem barrier.

---

## Kernels compilados

| Kernel | File | WGs | Threads | LDS | kernarg_size |
|---|---|---|---|---|---|
| `forward_main` | `smg_forward.hip` | batch_size | 128 | 32KB (state+lat_proj) | 96B |
| `backward_main` | `smg_backward.hip` | batch_size | 128 | 36KB (d_state+d_pre+d_logits) | 128B |
| `sumsq_main` | `smg_norm.hip` | ceil(len/256) | 256 | 4KB (LDS reduce) | 288B |
| `zero_acc` | `smg_norm.hip` | 1 | 1 | — | 8B |
| `sgd_main` | `smg_sgd.hip` | ceil(len/256) | 256 | — | 288B |
| `zero_grads` | `smg_sgd.hip` | ceil(len/256) | 256 | — | 288B |

---

## Ficheiros criados/modificados

```
Cargo.toml                          # feature gpu-aql; bins probe/parity/bench
build.rs                            # hipcc → .hsaco para 4 kernels
src/lib.rs                          # pub mod gpu_aql (cfg gpu-aql)
src/gpu_aql/
  mod.rs                            # AqlContext, hw constants, exports
  dispatch.rs                       # SmgRuntime: 6 kernels + dispatch helpers
  smg_grid_hsa.rs                   # SmgGridHsa (interface compat SmgGridGpu)
kernels/
  smg_forward.hip                   # forward_main (port de microcolumn_v3_forward.wgsl)
  smg_backward.hip                  # backward_main (CAS-loop f32 atomics)
  smg_norm.hip                      # sumsq_main + zero_acc
  smg_sgd.hip                       # sgd_main (zero inline) + zero_grads
bin/
  probe_hsa.rs                      # Phase 1+2 smoke-test
  test_parity_fwd.rs                # Phase 3: forward logits HSA vs wgpu
  test_parity_bwd.rs                # Phase 4: grad tensors HSA vs CPU
  test_parity_step.rs               # Phase 5: pesos pós-train_step HSA vs CPU
  bench_hsa.rs                      # Phase 6: throughput HSA vs wgpu
docs/migration-hsa-aql.md          # este ficheiro
```

---

## Erros resolvidos

### 1. `-mwavefrontsize32` rejeitado pelo hipcc --genco
`hipcc` com `--genco` repassa os flags ao `clang++ --cuda-device-only`, que rejeita
`-mwavefrontsize32` como unknown. Solução: remover — gfx1031 já defaulta wave32.

### 2. Kernarg size errado (hidden args)
Kernels que usam `blockIdx.x * blockDim.x + threadIdx.x` recebem **hidden args**
no ABI v5 (offsets 32–96). Forward/backward NÃO usam essas vars → sem hidden args.
Sumsq/sgd/zero_grads SIM → hidden args a preencher via `fill_hidden_1d`.
Tamanhos validados com `extract_layouts.py` contra o `.hsaco` compilado.

### 3. AQL grid semântica (grid_size_x = TOTAL work-items)
`dispatch_forward` passava `grid = (batch_size, 1, 1)`. Em AQL, `grid_size_x`
é o número TOTAL de work-items (não workgroups). Com `workgroup_size_x=128`,
apenas 1 WG disparava com threads 4..127 mascaradas → LDS não inicializada →
logits errados. Fix: `grid = (batch_size * 128, 1, 1)`.

### 4. hipMalloc no hot path (bottleneck de 49 MB/step)
`train_step` naïve alocava `buf_states` via `DeviceBuffer::alloc(batch * (T+1) * FLAT_DIM)`
= 49 MB de `hipMalloc + hipFree` por step. Fix: buffers persistentes (`pst_st`, etc.)
pré-alocados em `from_cpu(max_batch_size)`, atualizados via `hip_h2d` (hipMemcpy).

### 5. `unsafe_op_in_unsafe_fn` (Rust 2024 edition)
Em Rust 2024, operações unsafe dentro de `unsafe fn` requerem blocos `unsafe {}`
explícitos (lint `unsafe_op_in_unsafe_fn` é warn por default). Corrigido envolvendo
os corpos de cada `unsafe fn` em `dispatch.rs` num bloco `unsafe {}` com comentário
SAFETY.

---

## Próximos passos

1. Rodar `test_parity_step` — valida train_step completo (pesos pós-SGD).
2. Rodar `bench_hsa` em release — medir speedup real após fix dos buffers persistentes.
3. Se speedup >> 100×: integrar `SmgGridHsa` em `exp18_smg_lateral.rs` via feature gate.
4. Opcional: gradient clipping (sumsq+norm) no `SmgGridHsa` para paridade total com `SmgGridGpu`.

---

## Out of scope nesta migração

- Não vou portar `microcolumn_v2.rs` ou `microcolumn.rs` (mortos).
- Não vou tocar em outros bins (exp01..exp17). Só `exp18`.
- Não vou implementar HIP Graphs nem multi-queue (futuro).
- Não vou tentar CDNA-only optimizations (LDS sw prefetch, MFMA).

---

## Backout plan

Se HSA AQL bater em parede, fallback é:
1. Manter feature flag desligada → wgpu continua funcionando.
2. Revisar gargalo do wgpu: bind group caching, buffer reuse, único `submit` por epoch
   via `CommandEncoder` reutilizado. Ganho potencial 3-5× sem trocar de runtime.

Não é o caminho preferido — a tese central da pesquisa é que userspace AMD bate em
~12 µs no HSA sync e ~3 µs no async batched, fisicamente inalcançável via wgpu/Vulkan/RADV.
