# Plan 03 — Auto-Specialist MoE (3 fases evolutivas)

**Status:** Fase 1 em implementação
**Data:** 2026-05-13
**Inspiração:** SACI swarm + FEP (Friston) + Mixtral MoE

## 1. Objetivo

Substituir lookup monolítico por **Mixture-of-Experts auto-organizada**:
- Brain detecta domínios em uso
- Spawna specialists (eventually)
- Cada specialist learns via predictive coding (chaos→regra)
- Coordinator agrega via routing soft-gating

## 2. Por que NÃO auto-spawn full direto

Riscos identificados:
1. Domain detection HV ruidosa
2. Routing chicken-and-egg
3. Sem ground truth por specialist (só Qwen como global teacher)
4. Kuramoto sync entre experts instável
5. Auto-spawn dinâmico = state-of-art research, frágil

Decisão: **Sandwich evolutivo — 3 fases**, do simples ao radical.

---

## FASE 1 — Static MoE com Backbone Compartilhado (PROVED)

**Tempo:** 1-2 semanas | **Risco:** BAIXO

### Arquitetura

```
Query (bytes) → HV encode (1024-bit)
                    │
                    ▼
            ┌───────────────────┐
            │ Shared Backbone   │
            │ CfC 1024-dim      │
            │ (~32 MB)          │
            └─────────┬─────────┘
                      │
                      ▼
            ┌───────────────────┐
            │ Router Gating     │
            │ softmax top-2     │
            │ over 10 experts   │
            └─────────┬─────────┘
                      │
            ┌─────────▼────────┐
            ▼                  ▼
    ┌──────────────┐    ┌──────────────┐
    │ Expert head  │    │ Expert head  │
    │ Tech (~5MB)  │    │ Sciences     │
    │ specialized  │    │ specialized  │
    └──────┬───────┘    └──────┬───────┘
           │                   │
           └─────────┬─────────┘
                     ▼
              Weighted output
                     ▼
                Response
```

### Math

**Backbone forward:**
$$h = \text{CfC}(x) \in \mathbb{R}^{1024}$$

**Router gating:**
$$g = \text{softmax}(W_g \cdot h) \in \mathbb{R}^{N_{experts}}$$
$$\text{top-2 mask:} \quad m_i = \mathbb{1}[g_i \in \text{top-2}(g)]$$

**Expert output:**
$$y_i = W_i \cdot h$$

**Combined:**
$$y = \sum_i m_i \cdot g_i \cdot y_i / \sum_i m_i \cdot g_i$$

### Validação Math

**VRAM:**
- Backbone CfC 1024² = 4MB × 4 ops (input+output+hidden+gate) = 16MB
- 10 experts × 5MB heads = 50MB
- Router 1024×10 = 40KB
- **Total: ~70 MB**, dentro do orçamento

**Compute per query:**
- Backbone: 1024² = 1M FLOPs
- Router: 1024×10 = 10k FLOPs
- 2 active experts: 2 × 5M FLOPs = 10M FLOPs
- **Total: ~11M FLOPs/query** vs Qwen ~5G FLOPs (450× menor)

### 10 Domínios já validados (hard test 100%)

```
1. easy_recall      (Tecnologia)
2. medium_composition
3. hard_multihop
4. generative_creative
5. reasoning_logic
6. lateral_tricky
7. oov_slang
8. memory_self
9. (História BR + Mundial)
10. (Ciências + Saúde + Filosofia + Artes + Economia + Geografia + Matemática)
```

### Implementação

```rust
struct ExpertHead {
    name: String,
    domain_centroid: Hv,         // 1024-bit signature
    weights: Vec<f32>,           // [1024×output_dim]
    dialog_indices: Vec<usize>,  // pointer to brain.dialog_pairs for this domain
    n_hits: u64,                 // usage counter (pruning metric)
}

struct MoE {
    backbone: CfC,
    experts: Vec<ExpertHead>,
    router_weights: Vec<f32>,    // [hidden × N_experts]
}
```

Estado mínimo viável: cada expert é só um **centroid HV + dialog subset**. Backbone é o existing brain. Router = cosine similarity HV → top-2 experts.

### Validation Plan Fase 1

```bash
1. Implementar ExpertHead struct
2. Compute centroid HV per domain (média dos dialog HVs do domain)
3. Add MoE router em respond_with_confidence
4. Benchmark vs current brain
```

Métricas alvo:
- Brain-hit mantém ≥ 95% (no regression)
- Latency reduz (menos dialogs scan per query)
- Domain accuracy aumenta (right expert chooses)

---

## FASE 2 — Domain Detection Passivo (DETECT-only)

**Tempo:** 3-5 dias | **Risco:** BAIXO (passivo, não modifica nada)

### Conceito

Brain monitora WAL ingests + dialog HVs. Online clustering detecta NOVOS clusters
emergindo. **LOGA** o evento. Não spawna automaticamente. User decide.

### Math

**Online k-means / Birch:**
$$\mu_k^{t+1} = \mu_k^t + \eta_t (x_t - \mu_k^t) \cdot \mathbb{1}[k = \arg\min_j d(x_t, \mu_j^t)]$$

Com $\eta_t = 1/t$ (Robbins-Monro), centroids convergem.

**Cluster birth criterion:**
- Distance to nearest centroid > $d_{birth}$ (e.g., 0.5 hamming normalized)
- Local density > $\rho_{min}$:
$$\rho(x) = \frac{1}{N} \sum_{y \in pool} K_h(x - y)$$
- Kernel Gaussian: $K_h(d) = \exp(-d^2 / 2h^2)$
- Trigger: $\rho > 0.05 \cdot \rho_{max}$ in neighborhood

### Endpoint novo

```
GET /domains
  → JSON com lista clusters detected
    [{centroid_hv, n_dialogs, top_words, suggestion}, ...]

POST /domains/approve
  body: {centroid_id, name}
  → Cria expert head para esse centroid
```

User reviews `/domains` periodicamente, aprova os bons.

---

## FASE 3 — Auto-spawn Condicional (RESEARCH)

**Tempo:** 2-3 semanas | **Risco:** ALTO

### Spawn rules

```
SPAWN se:
  - density > ρ_high (threshold conservador)
  - dist_nearest > d_high
  - n_dialogs_in_cluster > N_min (e.g., 50)
  - 24h passaram desde último spawn (cooldown)
  - VRAM available > expert_size
```

### Predictive Coding training (FEP)

Cada specialist treina via:
$$L_{FEP} = -\log p(b_{t+1} | h_t) + \beta \cdot D_{KL}(q(h) || p(h))$$

Specialist observe sequência bytes de seu domínio. Forçado a prever próximo byte.
Erro de predição = surprise → atualiza weights.

**Top-down (prior):**
$$p(h_t | h_{t-1}) = \text{CfC}_{slow}(h_{t-1})$$

**Bottom-up (likelihood):**
$$p(b_t | h_t) = \text{softmax}(W_{out} h_t)$$

**Posterior approx:**
$$q(h_t | b_{1:t}) \propto p(b_t | h_t) p(h_t | h_{t-1})$$

### Pruning lifecycle

Specialist morre se:
- accuracy < floor (e.g., 0.3) por 24h
- usage_freq < min (e.g., 1 query/hour)
- centroid drift > max (merged into another)

Free VRAM. Update routing.

### Kuramoto coordinator (cross-domain coherence)

Specialists $i$ têm fase $\theta_i$. Coupling matrix $K_{ij}$ = mycelium edges.

$$\dot\theta_i = \omega_i + \sum_j K_{ij} \sin(\theta_j - \theta_i)$$

Order parameter:
$$r = \frac{1}{N} \left| \sum_j e^{i\theta_j} \right|$$

Quando $K > K_c = 2/(\pi g(0))$, sincronização emerge.

**Partial sync** desejável: domínios não-relacionados decoupled, relacionados lock.

### Math validation

**Capacity:**
- VRAM 500 MB para experts
- Each expert ~5 MB → max 100 experts
- Sustainable: 50 experts ativos (com pruning)

**Routing cost:**
- Soft gating: O(N) per query = 100 × 10k FLOPs = 1M FLOPs
- + top-2 expert forward: 10M FLOPs
- Total: ~11M FLOPs (linear with N)

**Forgetting bound (EWC):**
$$L = L_{new} + \lambda \sum_i F_i (\theta_i - \theta_i^*)^2$$

Fisher info $F_i$ protege parameters importantes. Empirical: $\lambda \approx 100$.

---

## ROADMAP UNIFICADO

```
Sprint A (semana 1-2): FASE 1 estático
  ✓ ExpertHead struct
  ✓ 10 expert centroids
  ✓ Router gating
  ✓ Benchmark vs current

Sprint B (semana 2-3): FASE 2 passive detection
  ✓ Online k-means daemon
  ✓ /domains endpoint
  ✓ Manual approval workflow

Sprint C (semana 4-6): FASE 3 auto-spawn cautious
  ✓ Spawn rules + cooldown
  ✓ FEP training loop
  ✓ Pruning lifecycle
  ✓ Kuramoto coupler

Sprint D (semana 7-8): integration + tuning
  ✓ Multi-teacher ensemble
  ✓ EWC anti-forgetting
  ✓ Cross-domain mycelium edges
  ✓ Final benchmark
```

## Math Reality Check Final

| Component | Hopeful | Realistic | Pessimistic |
|---|---|---|---|
| Fase 1 brain hit | 100% | 95% | 90% |
| Fase 1 latency | 50ms | 100ms | 150ms |
| Fase 2 false-positives | 5% | 15% | 30% |
| Fase 3 specialist survival rate | 80% | 50% | 30% |
| Total VRAM (10 experts) | 70MB | 100MB | 150MB |

Fase 1 é **win garantido**. Fase 2 + 3 são experimentos sem regressão se feito direito.
