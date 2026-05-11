# SMG — Stacked Microcolumn Grid

**Data:** 2026-04-30
**Status:** spec única, comitada. Sem alternativas no corpo do texto.
**Tese:** inteligência fala bytes estruturados. O modelo aprende **o bit**, não o conceito. Semântica emerge do padrão.

---

## 1. Princípios não-negociáveis

1. **Tudo é byte 8-bit.** Pesos, ativações, mensagens entre camadas. Float só existe na hora de treinar (e some na inferência).
2. **FSM byte é a unidade computacional.** 1 neurônio = 1 byte = `[família:3][nível:3][escala:1][spike:1]`. Já implementado em `src/fsm_byte`.
3. **Microcoluna (8 bytes) é a unidade arquitetural.** 8 FSMs cooperando — primitiva da hierarquia.
4. **Bus 32×32 é a coluna vertebral lateral.** Qualquer ponto da hierarquia alcança qualquer outro em **≤ 3 hops**.
5. **Treino é destilação de LLM existente.** Não treinamos do zero. Pegamos `iara-engine::DeepTrace`, codificamos em Path C, e supervisionamos next-byte. Resolve o problema de surrogate gradient.
6. **Geração é autoregressiva no Nível 0.** Saída tagged Path C decide por si: texto, número, comando, imagem.

---

## 2. Camadas (de baixo para cima)

### 2.1. FSM byte (já existe)

Layout:
```
bit 7 6 5 | 4 3 2 | 1 | 0
    família  nível  esc spk
```

8 famílias: `INTEGRADOR, DIFERENCIADOR, RESSONADOR, INIBIDOR, AMPLIFICADOR, COMPARADOR, MEMORIZADOR, ALEATORIZADOR`.

Transição: **lookup table 256→256 por família** (8 LUTs × 256 entradas = 2 KiB total). Inferência é zero-float, zero-multiplicação. Mapeia direto a `dp4a` em RDNA2.

### 2.2. Microcoluna (8 FSMs = 64 bits)

Unidade básica. 8 FSMs em paralelo recebem mesmo input, cada uma com família diferente. Output da microcoluna = 8 bytes (estado coletivo).

**Por que 8:** alinhamento com `u64`, lane wave32 RDNA2 processa 4 microcolunas simultaneamente via dp4a.

### 2.3. SMG — 6 níveis hierárquicos

| Nível | Grid       | Microcolunas | Bytes de estado |
|------:|:-----------|-------------:|----------------:|
| 0     | 256 × 256  | 65 536       | 524 288         |
| 1     | 128 × 128  | 16 384       | 131 072         |
| 2     | 64 × 64    | 4 096        | 32 768          |
| 3     | 32 × 32    | 1 024        | 8 192           |
| 4     | 16 × 16    | 256          | 2 048           |
| 5     | 8 × 8      | 64           | 512             |

**Total:** ~700 KiB de estado vivo. Pesos comprimidos em LUTs + tabelas de transição entre níveis (~5 MiB).

**Pool entre níveis:** 2×2 → 1 microcoluna. Operação = XOR + shift dos 4 bytes de família + OR dos spikes. Zero float.

### 2.4. Bus 32×32 (1024 microcolunas, 8 KiB)

Plano lateral compartilhado. **Cada nível tem leitura/escrita ao bus.** 

- Nível 0..5 fazem **`gather`** do bus (lê 8 microcolunas vizinhas no bus = 64 bytes)
- Nível 0..5 fazem **`scatter`** ao bus (escreve estado agregado da própria região)
- Roteamento: hash espacial — posição `(x,y)` no nível N projeta a `(x mod 32, y mod 32)` no bus

**Caminho máximo:** Nível 0 → Bus → Nível 0 = **3 hops** (scatter, route, gather). Igual tálamo coordenando córtex.

**Por que 8 KiB de bus para 700 KiB de estado:** o bus carrega **abstração agregada**, não cópia. Análogo a thalamic relay — comprime, distribui, não armazena tudo.

### 2.5. Path C tagged I/O (já existe)

Toda entrada e saída do SMG é stream Path C (`src/format/path_c.rs`). Domínios: `Number, Hex, Text, Command, State, Boolean, Percent`. Adicionaremos `Image` como nono domínio (header `0x90`, payload = bytes raw 8-bit).

**Multi-modal nativo:** texto, número, imagem coexistem no mesmo stream. SMG não distingue — é tudo byte tagged.

---

## 3. Como gera texto (caminho concreto)

### 3.1. Inferência

```
prompt "qual é a capital da frança?"
  ↓ [encode Path C: header Text + payload UTF-8]
bytes: 0x2A 'q' 'u' 'a' 'l' ... 0xFE
  ↓ [injeta na região esquerda do Nível 0 (256×256)]
Nível 0 ativa FSMs, scatter ao bus
  ↓ [pool 2×2 sobe para Nível 1..5]
Nível 5 (8×8) faz gather do bus, scatter de volta
  ↓ [Nível 4..0 fazem gather do bus durante descida]
Nível 0 região direita produz próximo byte (top-1 de 256)
  ↓ [autoregressivo: alimenta byte gerado de volta como input]
output: 0x25 'p' 'a' 'r' 'i' 's' 0xFE
  ↓ [decode Path C → "paris"]
```

Latência alvo (RDNA2 7700S, wave32): **~5 ms/byte** (dominado por LUT lookups e bus routing, sem matmul float).

### 3.2. Como armazena valor

Não armazena. **Computa.**

Pesos do SMG = tabelas de transição + LUTs FSM. O fato "capital da França é Paris" não vive numa célula; vive como **caminho determinístico**: dado prompt-bytes, a hierarquia + bus produz `paris`-bytes via FSM transitions.

A "equação" é o conjunto de (LUT_familia, transição_pool, hash_bus). Memória = função, não tabela.

### 3.3. Como categoriza

Path C header já categoriza no I/O. Internamente, **Nível 5 (8×8) é o ponto de menor entropia**: 64 microcolunas × 8 bytes = 512 bytes representam o "tópico" da inferência atual. SMG aprende a usar Nível 5 como key dispatcher (qual família/nível domina = qual domínio responder).

---

## 4. Treino — destilação de LLM existente

### 4.1. Por que destilação resolve o problema de gradiente

Backprop não passa por estado discreto FSM. Surrogate gradient é frágil. **Solução:** treinamos uma versão *contínua* do SMG (onde cada FSM é uma rede pequena que aproxima a LUT), supervisionamos contra bytes da LLM teacher, e **discretizamos só ao serializar**. Igual BitNet/QAT.

### 4.2. Pipeline

1. **Teacher:** `iara-engine` rodando Qwen2.5 ou Llama 3 (já temos infraestrutura).
2. **Trace:** para cada token gerado, capturamos `DeepTrace` (per-layer hidden_norm, attn_entropy, mlp_activation_rate, top_tokens, residual_delta, phase_coherence). Já implementado em `iara-engine/src/inference/tracer.rs`.
3. **Encode:** trace → Path C stream.
   - Cada layer vira bloco `Number` (norms, entropies) + `Percent` (activation rates) + `Text` (top tokens reduzidos a bytes ASCII).
   - Sequência completa de N layers + output = ~2 KiB de bytes Path C por token gerado.
4. **Dataset:** rodamos a teacher em corpus diverso (BR-CC, Wikipedia-PT, código). Cada inferência produz (prompt-bytes → trace-bytes → output-byte).
5. **Loss:** SMG vê prompt-bytes na entrada, deve prever os bytes do trace + output. Next-byte cross-entropy. **Zero surrogate gradient** porque o target é discreto e supervisionado.
6. **Discretização final:** após convergência, FSMs viram LUTs (round-to-nearest), pool transitions viram XOR+shift fixos. Inferência é 100% inteira.

### 4.3. Por que isso é defensável

- ✅ **Teacher signal existe** — DeepTrace já captura caminho completo.
- ✅ **Supervisão é next-byte-prediction** — problema resolvido (exp15 mostrou MLP minúsculo memoriza em 100 épocas).
- ✅ **Sem trilhão de tokens** — destilamos o que a teacher sabe, não re-aprendemos do zero. 1B tokens de teacher é suficiente para começar.
- ✅ **"Aprende o bit"** — SMG nunca vê texto raw como string, só como Path C bytes. Semântica vira propriedade emergente do padrão.

### 4.4. Etapas verificáveis

| Marco | Critério |
|------:|:---------|
| M1 | exp16: SMG-tiny (Nível 5 sozinho) reproduz output_byte da teacher para 100 prompts. Recall ≥ 90%. |
| M2 | exp17: SMG completa hierarquia + bus, reproduz 1000 prompts. Recall ≥ 70%. |
| M3 | exp18: SMG generaliza — prompts não vistos no treino, agreement com teacher ≥ 50%. |
| M4 | exp19: imagem. Path C `Image` domain, destilação de modelo de visão (CLIP/SDXL VAE). |
| M5 | exp20: model card 500M params. Benchmark vs Qwen 1.5B em texto+aritmética+imagem. |

---

## 5. Riscos conhecidos (honestidade)

1. **Bus 32×32 pode ser bottleneck.** 8 KiB compartilhado para 700 KiB de estado. Mitigação: medimos em M2; se for, expandimos para 64×64 (32 KiB). Não muda o design, muda constante.
2. **Discretização pós-treino degrada.** Pode perder 5-15% recall ao virar LUT. Padrão em QAT. Mitigação: fine-tune curto pós-discretização.
3. **DeepTrace é caro de coletar.** Cada token = vários downloads GPU→CPU. Mitigação: rodamos em batch overnight, dataset fica em disco.
4. **"Image domain" precisa ser definido com cuidado.** Path C atual suporta blocos longos via `length=0 + EOB`. Imagens de 192 KiB cabem mas serão lentas. Mitigação: tile 32×32 patches independentes, paralelizam no Nível 0.
5. **500M params é meta, não garantia.** O modelo real será o que couber em ~700 KiB de estado + ~5 MiB de tabelas. Os "500M" aparecem ao contar o espaço de programas representáveis pelas combinações de FSM × pool × bus, não pesos float.

---

## 6. Próximo experimento (exp16)

**Objetivo:** validar que SMG-tiny aprende a reproduzir `output_byte` de uma teacher rodando em iara-engine.

**Setup:**
- Teacher: Qwen2.5-0.5B em iara-engine.
- 100 prompts curtos (perguntas factuais simples).
- Coletar DeepTrace + token escolhido para cada um.
- SMG-tiny: só Nível 5 (8×8 microcolunas) + bus 8×8 (não 32×32, prova-de-conceito).
- Treino contínuo (FSMs são MLPs minúsculos), discretização só na avaliação final.
- Métrica: % de prompts em que SMG-tiny escolhe mesmo byte que teacher.

**Critério de sucesso:** ≥ 90% agreement com teacher discretizada. Se < 50%, design tem falha conceitual e voltamos à prancheta.

**Critério de falha clara:** se SMG-tiny converge em loss mas escolhe bytes errados após discretização > 30% de drop, a hipótese "FSM LUT é suficiente" cai e precisamos repensar 2.1.

---

## 7. O que NÃO está neste spec (a fazer depois)

- Detalhe da família AMPLIFICADOR vs INTEGRADOR (regras concretas de transição) — definimos no exp16.
- Esquema exato de hash espacial do bus — três candidatos a benchmark.
- Format binário do checkpoint SMG — definir após M2.
- Inferência distribuída multi-GPU — só importa em M5+.

Estes são detalhes de engenharia, não decisões arquiteturais. O spec acima fixa a arquitetura.
