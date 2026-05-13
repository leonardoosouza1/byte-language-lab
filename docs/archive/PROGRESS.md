# PROGRESS — Final (14 frentes empíricas)

✅ **14 experimentos rodados.** Multi-front exploration da hipótese Pós-LLM.

## Frentes testadas

### Bloco I — Formato e operações fundamentais
- **exp01** — Formato byte: round-trip + compactness vs f64 ✅
- **exp02** — Aritmética bytewise sem FP-ALU ✅ (+/- 100%, mul 99.4%)
- **exp03** — Learnability byte-positional vs ASCII (n-gram) ✅ (47-55% perplexity menor)
- **exp04** — Multi-domain stream (Path C) vs JSON ✅ (39% do tamanho)

### Bloco II — Conexões com LLMs e FSM
- **exp05** — Substrato bytewise unificado FSM ↔ número ✅ (100% round-trip)
- **exp06** — Comparação contra Qwen 2.5-1.5B BPE real ✅ (Path C 2.5× mais previsível)
- **exp07** — Tiny MLP gradient-trained: convergência por encoding ✅ (Path B 63% vs Qwen 25%)

### Bloco III — Paradigmas alternativos
- **exp08** — Álgebra tropical (max-plus, sem multiplicação) ✅ (classificador i16 puro funciona)
- **exp09** — Modern Hopfield com bytes discretos ✅ (recall robusto a ruído)
- **exp10** — Predictive coding (Friston FEP) em pesos i16 ⚠️ (parcial, erro decresce mas lento)
- **exp11** — STDP discreto (aprendizado local) ⚠️ (parcial, mistura patterns simultâneos)

### Bloco IV — Validações estruturais
- **exp12** — Cellular automata (Rule 110 et al.) — substrato de compute ✅
- **exp13** — Compression-based learnability (LZ77 manual) ✅ (estruturados comprimem melhor)
- **exp14** — Robustez a bit flips: byte vs f64 ✅ (Path B 10^4× menos disruptive)

## Findings consolidados

| Tese | Status | Evidência principal |
|------|:------:|---------------------|
| Bytes estruturados são VIÁVEIS | ✅ | Round-trip 100% em 4 formatos |
| Aritmética sem FP-ALU funciona | ✅ | +/- 100%, mul 99.4% |
| Byte-positional é mais APRENDÍVEL | ✅ | exp03/06/07: perplexity menor + MLP accuracy 2-3× maior |
| Substrato unificado existe | ✅ | exp05: byte FSM = byte Path C State |
| NN sem multiplicação é possível | ✅ | exp08: tropical funciona |
| Memória associativa over bytes | ✅ | exp09: Hopfield robusto |
| Computation universal em bytes | ✅ | exp12: Rule 110 Turing-complete |
| Mais redundante = mais aprendível | ✅ | exp13: compression ratio confirma |
| Robusto a corrupção | ✅ | exp14: rel err 10^4× menor que f64 |
| Aprendizado local sem backprop | ⚠️ | exp10/11: parcial, hyperparams importam |

## Métricas finais

- **Linhas de código Rust:** ~5200 (lib + 14 bins + tests)
- **Testes passando:** 58
- **Tempo total dos 14 experimentos:** ~18s
- **Build (release, fresh):** ~5s
- **Dependências externas:** zero
- **Linhas unsafe:** zero

## Próxima fronteira

Os experimentos parciais (exp10/11) sugerem que aprendizado biológico-plausível
(predictive coding, STDP) precisa de cuidado adicional. Possíveis próximos passos:

1. Competitive STDP com winner-take-all entre patterns
2. Predictive coding com mais camadas e schedule de learning rate
3. Hardware mapping: implementar exp08 (tropical) em HIP kernel para RDNA 2
4. Integração concreta: substituir 1 SpecialistBrain do Universe por byte-FSM
