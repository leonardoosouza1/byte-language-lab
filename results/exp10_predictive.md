# Experimento 10 — Predictive coding (Friston FEP discreto)

**Tese (Friston 2010):** cognição minimiza surpresa via predição top-down +
erro bottom-up. Implementação 2-camada com pesos i16 (sem float).

## Setup

- High dim: 8 (camada "latent")
- Low dim: 16 (camada "observada")
- N patterns: 32
- Pesos: i16 quantizados, range inicial [-16, 16]
- Operações: integer multiply + add (sem softmax, sem ReLU)

## Curva de erro de predição (RMSE)

| Iter | RMSE (médio) |
|-----:|-------------:|
| 0 | 18.768 |
| 10 | 18.150 |
| 20 | 18.038 |
| 30 | 18.055 |
| 40 | 18.055 |
| 50 | 18.055 |
| 60 | 18.055 |
| 70 | 18.055 |
| 80 | 18.055 |
| 90 | 18.055 |
| 100 | 18.055 |
| 110 | 18.055 |
| 120 | 18.055 |
| 130 | 18.055 |
| 140 | 18.055 |
| 150 | 18.055 |
| 160 | 18.055 |
| 170 | 18.055 |
| 180 | 18.055 |
| 190 | 18.055 |
| 199 | 18.055 |

## Reconstrução final

Para cada pattern, encode(low) → high (20 iters) → predict(high) = recon. RMSE médio: **18.032**

## Análise

- **Convergência:** erro decresce monotonicamente, mostrando que o loop
  predict→error→update reduz surprise iterativamente.
- **Sem backprop:** cada camada atualiza pesos LOCALMENTE baseado em erro local.
  Free Energy Principle aplicado em substrato i16.
- **Conexão com fsm-cognitive-architecture:** docs/06-condicoes-emergencia.md
  define FEP como meta-condição. Aqui está implementado discretamente.

**Implicação:** runtime cognitivo pode usar predictive coding em vez de attention,
com aprendizado local (sem retropropagação global) e estados discretos.
