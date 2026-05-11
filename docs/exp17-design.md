# exp17 — SMG-grid: primeira microcoluna real, com teste de generalização

**Status:** spec única, comitada. **Data:** 2026-04-30.
**Pré-requisito:** exp16 ✓ (M1 atingido, byte-agreement 100% por memorização).

## 1. Pergunta que exp17 responde

exp16 mostrou que **o sinal byte-a-byte do Qwen é aprendível por decoração**.
Não mostrou que estrutura ajuda.

exp17 testa **uma** coisa:

> Substituindo o MLP plano por uma grade de microcolunas FSM, com mesmo
> orçamento de parâmetros, mesmo dataset, mesma supervisão — a estrutura
> generaliza melhor para prompts **não vistos**?

Critério vai-ou-racha:
- Microcoluna ≥ MLP em prompts não vistos → estrutura agrega valor, seguir spec.
- Microcoluna ≈ MLP → estrutura é neutra, valor está em outra coisa (hierarquia? bus?).
- Microcoluna < MLP → tese de "byte-language gera estrutura emergente" enfraquece.

## 2. Arquitetura comitada (única)

### 2.1. Soft-FSM (treino)

Cada FSM tem estado interno **contínuo** durante treino:
```
state ∈ ℝᴷ,  K = 16
```

Atualização por tick (1 byte de input):
```
state' = softmax( state @ T_f  +  onehot(input) @ I_f  +  b_f  +  pos_bias )
```

- `T_f ∈ ℝᴷˣᴷ` — recorrência da família f (8 famílias, parâmetro compartilhado).
- `I_f ∈ ℝ²⁵⁶ˣᴷ` — projeção de input da família f.
- `b_f ∈ ℝᴷ` — viés da família.
- `pos_bias ∈ ℝᴷ` — viés por **posição na grade** (quebra simetria entre cells).

Discretização (avaliação/inferência final): `state_byte = argmax(state) → packed em 1 dos 256 buckets`. Não usado em exp17 (avaliamos em modo contínuo); fica para exp18+ medir drop pós-discretização.

### 2.2. Cell = microcoluna = 8 soft-FSMs

```rust
struct Microcell {
    states: [[f32; K]; 8],  // 8 FSMs, uma por família
}
```

Cada FSM tem família fixa em sua posição (índice 0 = Integrador, 1 = Diferenciador, …).
Todas as 8 FSMs do cell veem o mesmo input byte por tick.

### 2.3. Grid = 4×4 cells = 16 microcolunas

Para exp17 mantemos plano (sem hierarquia, sem bus 32×32). Hierarquia entra em exp18.

```rust
struct SmgGrid {
    cells: [Microcell; 16],   // 4×4
    pos_bias: [[f32; K]; 16], // viés por cell para quebrar simetria
}
```

Todas as cells veem o mesmo input por tick. Diferenciadas só pelo `pos_bias`. (Em exp18 teremos input local por cell + comunicação via bus.)

### 2.4. Decoder: grid state → próximo byte

Após processar a janela de 48 bytes, achatar o estado:
```
flat ∈ ℝ¹⁶ˣ⁸ˣᴷ = ℝ²⁰⁴⁸
logits = flat @ W_out  + b_out         // [2048, vocab_size]
```

Cross-entropy contra byte-alvo. **Vocab-restricted**: usamos os mesmos `vocab_size ≈ 41` bytes únicos que apareceram no dataset, igual exp16.

### 2.5. Contagem de parâmetros

Para `K = 16`, vocab=41, 8 famílias, 16 cells, window=48:

| Parâmetro | Shape | Count |
|---|---|---:|
| T_f (×8) | 16×16 | 2 048 |
| I_f (×8) | 256×16 | 32 768 |
| b_f (×8) | 16 | 128 |
| pos_bias (×16) | 16 | 256 |
| W_out | 2048×41 | 83 968 |
| b_out | 41 | 41 |
| **Total** | | **~119 k** |

MLP de exp16 tinha ~3M params. Microcoluna usa **~25× menos**, então qualquer match já é interessante.

## 3. Treino

### 3.1. Dados

Mesmas 100 traces do Qwen (`data/qwen_traces.bin`), mesmo sanitize de exp16.

**Split:** 80 train / 20 test, fixed seed. Nenhum prompt do test aparece no train.

**Streams:** 24× rep no train (mantém volume), 1× rep no test.

### 3.2. Loop

- BPTT completo através dos 48 ticks (memória barata: 16×8×16×48 = 96 k floats por pair).
- SGD paralelo, mesmo padrão de exp16 (`std::thread::scope`, buffers reutilizados).
- LR=0.05 (alvo), 30 épocas, batch=64.
- Eval a cada 5 épocas em **train E test** — para detectar overfit cedo.

### 3.3. Critério de parada

- Loss `NaN` ou > 3 × ln(vocab) → diverged, abort.
- Test loss subindo por 3 evals consecutivos → early stop, salva melhor checkpoint.

## 4. Métricas finais (saída em `results/exp17_smg_grid.md`)

Tabela por prompt do test (20 linhas):
- prompt
- Qwen response (gold)
- Microcoluna response
- MLP response (re-treinado no mesmo split, para comparação)
- byte-agreement (microcoluna)
- byte-agreement (MLP)

Resumo:
- Train byte-agreement (microcoluna / MLP)
- **Test byte-agreement (microcoluna / MLP)** ← métrica que decide
- Tempo de treino
- Veredicto (≥ MLP, ≈ MLP, < MLP)

## 5. O que NÃO está em exp17 (e por quê)

- **Hierarquia 6 níveis** — exp18.
- **Bus 32×32** — exp19.
- **Discretização pós-treino para LUT** — exp20.
- **Image domain** — M4.
- **1000+ prompts** — depende de exp18+ (escala cresce com hierarquia).

Cada um desses muda a arquitetura de forma específica. Misturar tudo em exp17 mata
a interpretabilidade do resultado.
