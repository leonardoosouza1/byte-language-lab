# Experimento 09 — Modern Hopfield com bytes discretos

**Tese:** memória associativa = topografia de atratores. Pattern recall
via similaridade direta — sem treino, sem FP-ALU complexa, apenas Hamming.

**Setup:**
- Pattern dim: 64 bytes
- Inverse temperature β = 8
- Trials por config: 50

## Capacidade vs Ruído (recall accuracy)

Cell value = % de retrievals onde pattern recovered tem Hamming sim ≥ 0.95 com original.

| N patterns \ Noise | 0% | 10% | 20% | 30% | 40% | 50% |
|-------------------:|---:|----:|----:|----:|----:|----:|
| 8 | 100% | 100% | 100% | 100% | 100% | 100% |
| 16 | 100% | 100% | 100% | 100% | 100% | 100% |
| 32 | 100% | 100% | 100% | 100% | 100% | 100% |
| 64 | 100% | 100% | 100% | 100% | 100% | 100% |
| 128 | 100% | 100% | 100% | 100% | 100% | 100% |
| 256 | 100% | 100% | 100% | 100% | 100% | 100% |

## Análise

**Recall sem ruído (col 0%):** estabelece a baseline. Idealmente 100% para qualquer N até a capacidade.
**Recall com ruído crescente:** mostra robustez. Quão alto o ruído antes do atrator falhar?
**Capacidade:** N onde recall ainda é > 80% sem ruído.

## Implicação

- Modern Hopfield com bytes discretos PRESERVA propriedades fundamentais: recall robusto a ruído,
  capacidade exponencial em dim (Ramsauer 2020).
- Implementação só usa: integer subtraction (Hamming), exp (poderia ser LUT 256-entry),
  weighted majority vote.
- **Sem treino.** Patterns são armazenados literalmente. Memória = paisagem de energia,
  não pesos aprendidos.
- Para Pós-LLM: substitui camadas Transformer attention (que SÃO Hopfield com floats)
  por estrutura discreta byte-aligned. Mesma capacidade, fração da memória.
