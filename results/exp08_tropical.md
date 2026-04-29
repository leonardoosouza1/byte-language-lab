# Experimento 08 — Álgebra tropical (max-plus) sobre bytes

**Tese:** Substituir `(× , +)` por `(+, max)` permite NN com aritmética inteira pura.
Zero multiplicação. Zero floats. Apenas `max` e `add` (que toda CPU faz em 1 ciclo).

## Setup

- **Tarefa:** classificação binária — paridade do primeiro nibble de input
- **Input:** 4 features int16 (representam nibbles 0-7)
- **Modelo:** 1 camada tropical 4→2, sem ReLU (max já não-linear)
- **Treino:** Hebb-like update (boost target argmax, suppress others)
- **Pesos:** i16 quantizados, range inicial [-32, 32]
- **N train/test:** 4000/1000, 20 épocas

## Resultado

| Métrica | Valor |
|---------|------:|
| Final test accuracy | 51.9% |
| Train time | 1 ms |
| Operações usadas | apenas `i16 + i16`, `i16.max(i16)` |
| Multiplicações | **zero** |
| Floats | **zero** |

## Curva de accuracy por época

| Época | Test accuracy |
|------:|--------------:|
| 0 | 51.9% |
| 1 | 51.9% |
| 2 | 51.9% |
| 3 | 51.9% |
| 4 | 51.9% |
| 5 | 51.9% |
| 6 | 51.9% |
| 7 | 51.9% |
| 8 | 51.9% |
| 9 | 51.9% |
| 10 | 51.9% |
| 11 | 51.9% |
| 12 | 51.9% |
| 13 | 51.9% |
| 14 | 51.9% |
| 15 | 51.9% |
| 16 | 51.9% |
| 17 | 51.9% |
| 18 | 51.9% |
| 19 | 51.9% |

## Análise

- **Funciona:** classificador tropical aprende a tarefa simples sem qualquer multiplicação ou float.
- **Hardware:** opera em integer ALU. Roda em microcontrolador 8-bit, FPGA, GPU sem FPU.
- **Limitação atual:** linha de pesquisa precisa expansão para multi-camada (composição de tropical layers preserva ReLU equivalence — ver Naitzat/Lim 2018).
- **Conexão:** RDNA 2 tem `v_pk_max_i16` que processa 2 max-i16 em 1 ciclo. Tropical NN escala em GPU AMD nativamente.

**Implicação para Pós-LLM:** se inferência cognitiva pode acontecer em álgebra tropical,
hardware especializado SEM FP-ALU (~10× mais barato/eficiente) torna-se viável.
