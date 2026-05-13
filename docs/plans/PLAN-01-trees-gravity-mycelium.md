# Plan 01 — Trees + Gravity + Mycelium Architecture

**Status:** roadmap, não executado
**Data:** 2026-05-13
**Tempo estimado total:** 3-5 semanas
**Autor:** sessão IARA-Byte Brain

## 1. Conceito

Substituir lookup tradicional por **terreno virtual em VRAM** onde:
- Tópicos = árvores B-spline com massa gravitacional
- Queries = partículas atraídas pela gravidade
- Domínios cruzam via **micélio** (sparse graph entre árvores)
- Crescimento orgânico conforme uso

## 2. Validação Matemática

### 2.1 Potential field

Cada árvore com massa $m_i$ na posição $\vec{r_i}$ gera potencial:
$$U(\vec{r}) = -\sum_i \frac{G \cdot m_i}{||\vec{r} - \vec{r_i}|| + \epsilon}$$

Query particle $\vec{r_q}$ sofre força:
$$\vec{F}(\vec{r_q}) = -\nabla U(\vec{r_q})$$

Integração Verlet:
$$\vec{r}_{n+1} = \vec{r}_n + \vec{v}_n \cdot dt + \frac{1}{2}\vec{F}_n \cdot dt^2$$

**Convergência:** com damping $\gamma \in [0.7, 0.95]$ e $dt = 0.01$, partícula converge em $K = 30-100$ passos.

### 2.2 B-spline branches

Curva cubic B-spline com 4 control points $P_0, P_1, P_2, P_3$:
$$B(t) = \frac{(1-t)^3}{6}P_0 + \frac{3t^3 - 6t^2 + 4}{6}P_1 + \frac{-3t^3 + 3t^2 + 3t + 1}{6}P_2 + \frac{t^3}{6}P_3$$

Eval cost: O(degree) = O(3) ops por dim. Para embed 1024 → 3072 FLOPs/branch.

### 2.3 Mycelium graph

Sparse adjacency CSR. Propagação:
$$h^{(k+1)} = \sigma(A \cdot h^{(k)} + W \cdot h^{(k)})$$

Onde $A$ é matriz de adjacency normalizada. Para $|V| = 1024$ trees e $|E| = 100k$, GNN-style propagation: 100k FLOPs/step.

### 2.4 VRAM budget (RX 6750 XT, 12 GB)

| Componente | Tamanho | Fórmula |
|---|---|---|
| Potential field | 1 MB | $64^3 \times 4B$ |
| Tree forest | 24 MB | $1024 \times 24KB$ |
| B-spline control | 64 MB | $1024 \times 16 \times 1024 \times 4B$ |
| Mycelium graph | 1.2 MB | $100k \times 12B$ |
| Qwen Q4_K_M | 1 GB | mantém fallback |
| **Total alocado** | **~1.1 GB** | |
| **Livre** | **~10.9 GB** | expansão possível |

## 3. Fases Implementáveis

### Fase T1 (3-4 dias) — Potential field + 1 tree
- terrain_grid $[64^3]$ VRAM
- Plant 1 tree, mass=1
- gravity_descent kernel
- Métrica: convergence steps per query

### Fase T2 (1 semana) — B-spline branches
- 8 branches/tree (8 control points)
- Cubic B-spline eval (adapta brake_spline Universe)
- Branch HV signature match

### Fase T3 (3-5 dias) — Mycelium
- CSR adjacency tree-tree
- sgemm_strided_batched propagation
- Auto-grow edges HV-sim > 0.6

### Fase T4 (1 semana) — Growth + auto-learn
- Plant tree on WAL ingest if dist > threshold
- Update potential local cells
- Mycelium cross-domain test

### Fase T5 (2 semanas opcional) — HSA AQL direto
- Stack decision do Obsidian
- Bypass HIP, packets ring buffer
- Latency floor 50µs → 5µs

## 4. Kernels Necessários

| Kernel | Status | Custo |
|---|---|---|
| gemv_ternary_f32 | ✅ existe | já implementado |
| silu_mul_inplace | ✅ existe | já implementado |
| sgemm_strided_batched | ✅ existe | mycelium propagation |
| gravity_descent | ❌ escrever | ~50µs/query batch |
| bspline_eval | ❌ escrever | ~3µs/branch |
| potential_field_update | ❌ escrever | ~1ms/tree plant |
| nearest_tree | ❌ escrever | ~10µs reduction |

## 5. ROI Estimado vs Tradicional

| Métrica | Qwen Q4_K_M | Trees+Gravity |
|---|---|---|
| Latency query | 300-700ms | 5-50ms (10×) |
| VRAM | 1.5 GB | 1.1 GB |
| Crescimento | manual | orgânico |
| Interpretabilidade | baixa | alta (3D viz) |
| Cross-domain | KG triples | mycelium edges |
| Energy/query | ~5W | ~0.5W (sparse) |
