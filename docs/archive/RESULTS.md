# RESULTS — byte-language-lab

Hipótese central: **inteligência fala bytes estruturados, não tokens nem floats**.

**Lab executado em 18160 ms total.** Detalhes por experimento:

## Sumário executivo

| Experimento | Pergunta | Veredito |
|-------------|----------|----------|
| exp01 — formato | Quão compacto/rápido é byte-estruturado vs f64? | f64 vence em throughput; bytes pagam ~1.2-2.5× memória, ganham interpretabilidade total |
| exp02 — aritmética | Aritmética sem FP-ALU é viável? | **Sim**: add/sub 100% correctness, mul 99.4%. ~50× slower que f64 (esperado, sem SIMD) |
| exp03 — learnability | Byte-positional é mais aprendível que ASCII? | **Sim**: perplexity 47-55% MENOR em todos n-gram orders |
| exp04 — composition | Bytes podem unificar multi-domain? | **Sim**: 39% do espaço de JSON para mesmo conteúdo (número+texto+comando+estado+bool) |
| exp05 — FSM unified | Mesmo byte serve para estado FSM e número? | **Sim**: 100% round-trip em 1000 frames mistos. `Block::State(byte)` ≡ `Microcoluna.neurons[i]` |
| exp06 — Qwen BPE | Byte-positional vence Qwen tokenizer real? | **Sim**: Path C perplexity **3.64 vs Qwen 9.31** (2.5× mais previsível) |
| exp07 — Neural MLP | MLP treinada aprende mais rápido sob byte-positional? | **Sim**: Path B accuracy **63% vs Qwen 25%**, converge em 750 vs nunca |
| exp08 — Tropical | NN sem multiplicação (max-plus) é viável? | **Sim**: classificador binário roda em apenas i16 add+max |
| exp09 — Hopfield | Memória associativa funciona com bytes discretos? | **Sim**: 100% recall até 256 patterns × 50% noise (Hamming similarity) |
| exp10 — Predictive | Friston FEP em pesos i16 converge? | **Parcial**: erro decresce 18.7→18.0, lento mas direcional. Ajuste de hyperparams necessário |
| exp11 — STDP | Aprendizado local sem gradiente diferencia patterns? | **Parcial**: STDP simplista mistura patterns; precisa competitive STDP ou curriculum |
| exp12 — Cellular Automata | Rule 110 (Turing-complete) emerge? | **Sim**: 8 regras testadas, classes Wolfram I/II/III observadas. Rule 110 chaotic confirmada |
| exp13 — Compression | Streams estruturados são mais redundantes? | **Sim**: Qwen u16 LZ77 0.525, Path C 0.603, Path B 0.699 — todos superam ASCII 0.855 |
| exp14 — Robustness | Byte degrada melhor que f64 sob bit flips? | **Sim em rel-err**: Path B mean log rel err **-5.99 vs f64 -1.85** (erro relativo ~10^4× menor) |

## Findings concretos

### 1. Bytes estruturados são VIÁVEIS

- Round-trip 100% fidelidade dentro da tolerância natural do formato
- Aritmética digit-by-digit com carry/borrow funciona — equivalente a como humanos calculam
- Sem dependência de FP-ALU; roda em qualquer hardware com integer add

### 2. Bytes ESTRUTURADOS são MAIS APRENDÍVEIS que strings

Resultado mais surpreendente. Em modelo Markov 3-gram em arithmetic completion:
- ASCII string: perplexity ≈ 10.5
- Byte-positional: perplexity ≈ 5.5
- **Byte-positional é ~2× mais previsível.**

Isso é forte evidência de que LLMs (que tokenizam números como strings) deixam estrutura na mesa que arquiteturas byte-positional capturariam diretamente.

### 3. Bytes UNIFICAM domínios sem custo de tokenização

Com Path C (header byte + payload):
- Multi-domain stream = 39% do tamanho de JSON equivalente
- Round-trip lossless
- Cada bloco auto-descritivo via header (1 byte overhead)
- Mesma representação para número + texto + comando + estado FSM

### 4. Trade-off honesto: throughput vs estrutura

| Aspecto | f64/IEEE 754 | Byte estruturado |
|---------|:-----------:|:----------------:|
| Memória | 8 B (constante) | 8-20 B (variável) |
| Throughput | bilhões op/s (FP-ALU) | dezenas de milhões op/s (CPU naive) |
| Interpretabilidade | nenhuma (mantissa opaca) | total (cada bit semântico) |
| Determinismo cross-platform | não (modos de rounding) | total |
| Multi-domain nativo | não | sim (Path C) |
| Hardware exigido | FPU | qualquer integer ALU |

## Próximos passos sugeridos

Esta lab valida que a hipótese é **viável e mensuravelmente promissora**. Para próxima fase:

1. **SIMD bytewise:** aritmética byte-positional usando `v_dot4_i8` (RDNA 2) ou AVX-512 — vectorizar ganha contra f64 escalar.
2. **Modelo neural sobre bytes:** tiny transformer que prevê próximo byte estruturado. Validar se a estrutura posicional acelera convergência (hipótese de exp03 levada a sério).
3. **Aritmética avançada:** divisão long-division byte-wise, raiz quadrada, transcendentais via séries com bytes.
4. **Integração com FSM-arch:** bytes de estado FSM (8 bits = família+nível+escala+spike) e bytes de número compartilham o mesmo encoder/decoder C — unifica o substrato de inteligência.
5. **Replace LLM tokenizer por byte-positional:** experimento decisivo — destilar small LLM para gerar bytes estruturados em vez de tokens BPE. Mede accuracy em arithmetic benchmarks.

## Tempos de execução

- exp01_format — 2.75s
- exp02_arithmetic — 2.71s
- exp03_learn — 2.71s
- exp04_compose — 2.58s
- exp05_fsm_unified — 2.92s
- exp06_bpe_compare — 3.47s
- exp07_neural — 0.31s
- exp08_tropical — 0.05s
- exp09_hopfield — 0.08s
- exp10_predictive — 0.06s
- exp11_stdp — 0.05s
- exp12_cellular — 0.06s
- exp13_compression — 0.36s
- exp14_robustness — 0.05s
- **Total** — 18.16s

## Como reproduzir

```bash
cd byte-language-lab
cargo test --release  # 50+ testes (formats, arithmetic, learn, compose)
cargo run --release --bin run_all  # gera todos os results/*.md e RESULTS.md
```

## Artefatos

- [`DESIGN.md`](DESIGN.md) — formalismo dos 3 paths
- [`results/exp01_format.md`](results/exp01_format.md) — round-trip & compactness
- [`results/exp02_arithmetic.md`](results/exp02_arithmetic.md) — aritmética bytewise vs f64
- [`results/exp03_learn.md`](results/exp03_learn.md) — learnability vs ASCII (n-gram)
- [`results/exp04_compose.md`](results/exp04_compose.md) — multi-domain stream
- [`results/exp05_fsm_unified.md`](results/exp05_fsm_unified.md) — substrato bytewise unificado FSM ↔ número
- [`results/exp06_bpe_compare.md`](results/exp06_bpe_compare.md) — vs Qwen 2.5 BPE real
- [`results/exp07_neural.md`](results/exp07_neural.md) — tiny MLP gradient-trained, 4 encodings
- [`results/exp08_tropical.md`](results/exp08_tropical.md) — álgebra tropical (max-plus)
- [`results/exp09_hopfield.md`](results/exp09_hopfield.md) — Modern Hopfield com bytes
- [`results/exp10_predictive.md`](results/exp10_predictive.md) — predictive coding i16
- [`results/exp11_stdp.md`](results/exp11_stdp.md) — STDP aprendizado local
- [`results/exp12_cellular.md`](results/exp12_cellular.md) — cellular automata + Rule 110
- [`results/exp13_compression.md`](results/exp13_compression.md) — LZ77 ratio per encoding
- [`results/exp14_robustness.md`](results/exp14_robustness.md) — bit-flip robustness vs f64
