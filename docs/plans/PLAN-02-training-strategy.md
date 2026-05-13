# Plan 02 — Training Strategy + Distillation + VRAM Expansion

**Status:** debate
**Data:** 2026-05-13

## 1. CONTEXTO ATUAL

```
VRAM RX 6750 XT 12 GB
─────────────────────────────────
Atual em uso:    ~1.5 GB (Qwen Q4_K_M)
LIVRE:           ~10.5 GB ← oportunidade
Pico aceitável:  ~10 GB (deixa 2 GB headroom)
```

**Brain hoje** é estatístico (bigram+trigram+KG+HDC) — **não tem pesos treináveis** no sentido neural clássico. Só CfC tiny (16-dim hidden) que existe mas não treina.

## 2. PERGUNTA: COMO TREINAR?

### 2.1 O que existe pra treinar

| Componente | Trainable? | Método sugerido |
|---|---|---|
| char_cfc / word_cfc / phrase_cfc | sim, ~hundreds params | online STDP ou backprop offline |
| word_bigram counts | counter, não gradient | reinforcement por similarity |
| byte_attention_qwen distilled | weights | re-distill periódico |
| dialog HDC index | hash, não gradient | reweight via use freq |
| **NOVO: tree forest weights** | sim, B-spline control pts | offline GD ou online evolutionary |
| **NOVO: mycelium edges** | sparse weights | Hebbian + decay |

### 2.2 Objetivos de loss matemáticos

**Loss A — distillation byte-level:**
$$L_{distill} = \sum_b KL(p^{student}_b || p^{teacher}_b)$$

Teacher = Qwen forward; student = brain forward. Sample bytes em paralelo.

**Loss B — retrieval consistency:**
$$L_{retrieval} = \sum_{q,a^*} -\log P_{brain}(a^* | q)$$

Para cada par (query, ground-truth answer) no WAL, maximizar prob brain retornar $a^*$.

**Loss C — sparse activation (Kuramoto-aligned):**
$$L_{sparse} = ||\text{active\_mask}||_0 - K_{target}$$

Penaliza mais que K neurônios ativos por step. Força sparse routing.

**Loss D — energy minimization (gravity field):**
$$L_{energy} = \sum_{tree} \frac{1}{1 + ||\nabla U(r_{tree})||^2}$$

Trees em mínimos locais do potential ↔ stable knowledge centers.

### 2.3 Combinação total

$$L_{total} = \alpha L_{distill} + \beta L_{retrieval} + \gamma L_{sparse} + \delta L_{energy}$$

Pesos sugeridos: $\alpha = 0.5, \beta = 0.3, \gamma = 0.1, \delta = 0.1$.

### 2.4 Quanto custa treinar?

**FLOPS estimate por batch de 32 queries:**
- Brain forward: 1M FLOPs/query × 32 = 32M FLOPs
- Qwen forward (teacher, no_grad): 5G FLOPs/query × 32 = 160G FLOPs
- KL divergence: 256 × 32 = 8k FLOPs
- Backward brain only: ~3M FLOPs (small)
- Total: ~160G FLOPs/batch

RX 6750 XT: 13.21 TFLOPS FP32. Batch teórico: 0.012s. Real (overhead): ~50ms.

**Throughput:** 20 batches/s × 32 queries = 640 queries/s training rate.

**Para 1M queries:** ~26 min de training continuous. Viável overnight.

## 3. REAPROVEITAR INTELIGÊNCIA DE OUTROS MODELOS

### 3.1 Modelos disponíveis localmente

```
~/projects/LLM/llm-lab/models/
├── qwen2.5-0.5b.gguf       ← draft model
├── qwen2.5-1.5b.gguf
├── qwen2.5-1.5b-q4km.gguf  ← atual teacher
├── qwen2.5-1.5b.dna        ← DNA-SVD
├── qwen2.5-1.5b-q4.dna
└── qwen3-4b                ← 4B params, qwen3 thinking model
```

### 3.2 Estratégias de reuse — matematicamente

**A) Distillation hard (token-level)**
- Teacher gera N tokens com greedy
- Student treina pra mimicar
- Loss: cross-entropy
- $L = -\sum_t \log P_{student}(y^{teacher}_t | x)$
- Limitação: greedy perde diversidade

**B) Distillation soft (logits)**
- Teacher retorna distribuição completa $p^{teacher}$
- Student aprende distribuição
- Loss: KL divergence com temperature
- $L = T^2 \cdot KL(p_T^{student} || p_T^{teacher})$
- $p_T = \text{softmax}(z/T)$, T=2-5 típico

**C) Feature distillation (intermediate)**
- Aligna hidden states em layers intermediárias
- $L_{feat} = ||h^{student}_l - W \cdot h^{teacher}_l||^2$
- W aprende projection se dims diferem
- Captura "raciocínio" do teacher

**D) Attention transfer**
- Match attention maps
- $L_{attn} = ||A^{student} - A^{teacher}||_F^2$
- Captura "onde o teacher olha"
- Já fizemos parcial (qwen_byte_attention distilado)

**E) Self-distillation multi-modelo**
- Vários teachers (Qwen 1.5b + Qwen3-4b + outros)
- Ensemble logits: $p^{teacher} = \frac{1}{K}\sum_k p_k$
- Student aprende consenso → mais robusto
- Custo: K× forward teachers

### 3.3 Quanto cada teacher contribui?

**Qwen 1.5b q4km**: rápido (0.4s/call), boa qualidade PT-BR
**Qwen3-4b**: lento (4s/call), thinking model com <think> blocks, raciocínio melhor
**Qwen2.5-0.5b**: muito rápido, qualidade limitada

**Ensemble pesado:**
$$p^{ensemble} = 0.4 \cdot p_{1.5b} + 0.5 \cdot p_{4b} + 0.1 \cdot p_{0.5b}$$

Pesos baseados em accuracy benchmark.

### 3.4 External — modelos open-source extras

Pode-se baixar via HuggingFace:
- **Llama 3.1 8B** (3.5 GB Q4) — strong general
- **Mistral 7B Q4** (3.8 GB) — code-strong
- **Phi-3 Mini Q4** (~2 GB) — efficient
- **Gemma 2 2B Q4** (~1 GB) — leve

Com 10 GB livres, dá pra ter 2-3 desses simultâneos. Diversidade aumenta robustness.

## 4. EXPANSÃO DE PARÂMETROS USANDO VRAM SOBRANDO

### 4.1 Onde alocar os 10 GB livres?

**Proposta de orçamento:**

| Componente | VRAM | Propósito |
|---|---|---|
| Qwen 1.5b Q4 (atual) | 1.0 GB | Fast teacher PT-BR |
| Qwen3-4b (thinking) | 3.5 GB | Reasoning teacher |
| Llama 3.1 8B Q4 | 4.5 GB | English diverse teacher |
| Brain weights expandido | 0.5 GB | trees + mycelium + cfc-large |
| KV caches (3 modelos) | 0.2 GB | concurrent inference |
| Headroom | 2.0 GB | spike protection |
| **Total** | **11.7 GB** | dentro do limite |

### 4.2 CfC expandido (substantial trainable)

Atual: 16-dim CfC. Pode crescer:

$$\text{CfC}_{large}: hidden\_dim = 1024, \text{layers} = 8$$

Params: $1024^2 \times 8 \times 4B = 32$ MB. Cabe fácil.

Throughput: ~1M params trainable. Útil pra "compress" knowledge em weights vs hashmap.

### 4.3 Mixture of Experts (MoE) caseiro

Vários brain "specialists" — cada um focado em domínio:

```
Brain Tech       (1024-dim CfC)
Brain História   (1024-dim CfC)
Brain Ciências   (1024-dim CfC)
...
Router (sparse) → seleciona top-2 specialists por query
```

Math: 10 specialists × 4 MB = 40 MB. Router gating: 10×1024 = 40 KB.

**Vantagem:** specialists podem ser trainados separadamente, parallel.

### 4.4 Hyperparam grid suggestion

| Hyperparam | Range | Best guess |
|---|---|---|
| Tree count | 256–8192 | 1024 |
| B-spline control pts/branch | 4–16 | 8 |
| Mycelium edges | 10k–1M | 100k |
| CfC hidden | 16–4096 | 1024 |
| Batch size training | 8–256 | 32 |
| Learning rate | 1e-5 – 1e-2 | 3e-4 (AdamW) |
| KL temperature T | 1–10 | 3 |
| Distill loss weight α | 0.1–0.9 | 0.5 |

## 5. SUGESTÕES ADICIONAIS

### 5.1 Curriculum Learning

Ordenar dialogos por dificuldade durante training:
- Easy: recall direto (fato simples)
- Medium: composição
- Hard: multi-hop reasoning

$$P_{epoch}(t) = \text{linspace}(easy, hard, \text{epochs})$$

Acelera convergência ~2×.

### 5.2 Reinforcement Learning via brain hit rate

Cada query bem-respondida (brain-hi com user satisfaction) → recompensa.
$$R(q, a) = \mathbb{1}[\text{brain-hi}] \cdot (1 - \text{user\_correction})$$

Update tree masses + mycelium weights via policy gradient.

### 5.3 Contrastive learning

Dada query $q$ e correct answer $a^+$, negativos $a^-$:
$$L_{contrast} = -\log \frac{\exp(\text{sim}(q, a^+)/\tau)}{\sum_a \exp(\text{sim}(q, a)/\tau)}$$

Aprende representações que separam respostas certas das erradas. Boost HDC retrieval.

### 5.4 Knowledge graph completion (KGC)

Atual: 5k triplas. Faltam muitas.

TransE-style:
$$\vec{e}_{head} + \vec{r}_{relation} \approx \vec{e}_{tail}$$

Treinar embeddings + relations. Inferir triples faltantes. Cresce KG sem novo input.

### 5.5 Multi-task fine-tune

Adicionar tarefas auxiliares:
- Masked LM (predict byte from context)
- Next sentence prediction
- Sentiment classification (PT-BR)

Multi-task improves transfer learning.

### 5.6 Continual learning sem esquecer (EWC)

Elastic Weight Consolidation evita catastrophic forgetting:
$$L = L_{task} + \lambda \sum_i F_i (\theta_i - \theta_i^{old})^2$$

$F_i$ = Fisher info dos params importantes. Permite aprender novo sem destruir velho.

### 5.7 Aceleração HSA AQL bypass HIP

Stack decision do Obsidian: dispatch direto via AQL packets → ring buffer → doorbell signal. Latency floor cai 50µs → 5µs por kernel launch.

Para batch training 20 batches/s → 1000 batches/s teórico se hot path mais magro.

### 5.8 Mixed precision training (BF16 weights, F32 grads)

Hoje: F32 puro. Migrar pra BF16 weights + F32 master + loss scaling.
- VRAM weights: 2× menor
- Throughput RDNA2: emulated BF16 mas v_dot2 packed dá ganho
- Treinamento: estável com loss scaling 2^10–2^15

## 6. VALIDAÇÃO MATEMÁTICA — SANITY CHECKS

### 6.1 Convergence guarantee gravity

Lyapunov function: $V(\vec{r}, \vec{v}) = \frac{1}{2}||\vec{v}||^2 + U(\vec{r})$

Com damping $\gamma > 0$:
$$\dot V = \vec{v} \cdot \dot{\vec{v}} + \nabla U \cdot \vec{v} = -\gamma ||\vec{v}||^2 \leq 0$$

Garante convergência pra mínimo local. **Validado.**

### 6.2 Capacity hypervectors HDC

Hopfield capacity: $C \approx 0.14 \cdot n$ para $n$-dim binary.
- 1024-bit HV → $C \approx 143$ patterns sem interferência
- 12k dialogs no pool → saturado, mas Hamming search é robust até $4-5C$

**Conclusion:** atual HDC sub-óptimal. Subir pra 4096-bit HV → $C = 573$ → comfortable for 12k.

### 6.3 Mycelium scale

Erdős–Rényi: random graph $G(n, p)$ é conectado quando $p > \frac{\ln n}{n}$.
- $n = 1024$ trees → $p_{crit} = 0.0068$
- 100k edges sobre $\binom{1024}{2} = 523k$ possíveis → $p \approx 0.19$ >> $p_{crit}$
- **Garantido conexo.** Diâmetro ~$\frac{\ln n}{\ln(np)} \approx 3$ hops.

### 6.4 Tree forest capacity

Cada tree representa um topic-cluster. Com 1024 trees:
- Cobertura cross-domain razoável (português ~50k lemas em uso)
- Cada tree concentra ~50 lemas similares
- B-spline branches refinam (4-16 sub-conceitos por tree)

Comparação: Qwen embedding table 151k vocab × 1536 → 232M params.
**Trees:** 1024 × 1024 (HV) × 4B = 4 MB. **58000× menor**.

Trade-off: representação mais grossa mas semanticamente coerente via gravity manifolds.

## 7. ROADMAP UNIFICADO

```
Sprint 1 (semana 1-2)
  • PLAN-01 Fase T1 — potential field + 1 tree (gravity_descent kernel)
  • Distillation pipeline básica (Qwen → CfC large)
  • CfC expandido 16 → 1024 dim

Sprint 2 (semana 3-4)
  • PLAN-01 Fase T2 — B-spline branches
  • Adicionar Qwen3-4b como second teacher
  • Ensemble distillation
  • EWC para evitar forgetting

Sprint 3 (semana 5-6)
  • PLAN-01 Fase T3+T4 — mycelium + growth
  • RL via brain-hit reward
  • TransE KGC para completar triplas

Sprint 4 (semana 7-8)
  • Llama 3.1 8B Q4 como teacher diversidade
  • MoE specialists (10 brains por domínio)
  • Contrastive learning HDC boost

Sprint 5 (semana 9-10)
  • PLAN-01 Fase T5 — HSA AQL bypass
  • Mixed precision BF16 training
  • Curriculum learning
  • Benchmark final vs Qwen
```

## 8. MÉTRICAS DE SUCESSO

```
Target após 10 semanas:

Latency query brain      <50ms     (vs hoje 200ms)
VRAM total               <8 GB
Brain coverage hard test  100%
Qwen fallback rate       <5%       (vs hoje ~30% em queries novel)
Throughput training      ~600 q/s
Knowledge graph triples  >50k     (vs hoje 5k)
Trees forest            1024-2048
Mycelium edges          100k-1M
```

## 9. RISCOS

| Risco | Probabilidade | Mitigação |
|---|---|---|
| Distillation degrada quality | média | KL temperature tuning, multi-teacher |
| Mycelium scales O(n²) | baixa | sparse CSR garante O(E) |
| GPU OOM com multi-model | média | unload modelos não-ativos |
| Forgetting catastrófico | alta | EWC + replay buffer |
| HSA AQL stack instável | média | fallback para HIP módulo |

## 10. DECISÃO

Próximo passo concreto: **Sprint 1** — começar com gravity_descent kernel + CfC expansion. Ambos isolados, não tocam pipeline produtivo.
