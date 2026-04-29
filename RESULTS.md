# RESULTS — byte-language-lab

Hipótese central: **inteligência fala bytes estruturados, não tokens nem floats**.

**Lab executado em 2662 ms total.** Detalhes por experimento:

## Sumário executivo

| Experimento | Pergunta | Veredito |
|-------------|----------|----------|
| exp01 — formato | Quão compacto/rápido é byte-estruturado vs f64? | f64 vence em throughput; bytes pagam ~1.2-2.5× memória, ganham interpretabilidade total |
| exp02 — aritmética | Aritmética sem FP-ALU é viável? | **Sim**: add/sub 100% correctness, mul 99.4%. ~50× slower que f64 (esperado, sem SIMD) |
| exp03 — learnability | Byte-positional é mais aprendível que ASCII? | **Sim**: perplexity 47-55% MENOR em todos n-gram orders |
| exp04 — composition | Bytes podem unificar multi-domain? | **Sim**: 39% do espaço de JSON para mesmo conteúdo (número+texto+comando+estado+bool) |

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

- exp01_format — 2.50s
- exp02_arithmetic — 0.06s
- exp03_learn — 0.06s
- exp04_compose — 0.04s
- **Total** — 2.66s

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
- [`results/exp03_learn.md`](results/exp03_learn.md) — learnability comparison
- [`results/exp04_compose.md`](results/exp04_compose.md) — multi-domain stream
