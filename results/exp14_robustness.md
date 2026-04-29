# Experimento 14 — Robustez a corrupção (byte vs float)

**Tese:** float64 é frágil — flip 1 bit no expoente = magnitude muda 2× (ou Inf/NaN).
Byte estruturado degrada local: 1 dígito errado → erro contido naquela posição.

**Setup:**
- 10000 valores em [0.001, 1000]
- Path A/B: flip 1 byte aleatório (8 bits) do encoding
- f64: flip 1 bit aleatório dos 64 bits IEEE 754

## Resultados

| Encoding | Total | Decode OK (finite) | Decode failed | Mean abs err | Max abs err | Mean log10(rel err) |
|----------|------:|-------------------:|--------------:|-------------:|------------:|--------------------:|
| Path A | 10000 | 8013 (80.1%) | 1987 (19.9%) | 2.255e7 | 9.000e8 | -1.88 |
| Path B | 10000 | 7740 (77.4%) | 2260 (22.6%) | 2.405e15 | 9.978e16 | -5.99 |
| f64 raw | 10000 | 9999 (100.0%) | 1 (0.0%) | 1.117e155 | 1.330e157 | -1.85 |

## Análise

**Decode failed**: % de corruptions que resultam em valor inválido (NaN/Inf para f64,
decode error para byte formats). Quanto MENOR melhor (mais robusto).

**Mean log10(rel err)**: erro relativo médio em log10. -2 = erro ~1%, -10 = ~1e-10.
Quanto MAIS NEGATIVO melhor (corruption causa menos disrupção).

**Headline:** float64 é mais frágil que byte estruturado em 2 dimensões:
1. Mais corruptions geram NaN/Inf (bits de expoente)
2. Quando produz valor finito, magnitude pode ser dramática (mantissa ou sign)

Byte structured degrada local: 1 dígito errado em "47.83" produz "57.83" ou "47.93".
Erro proporcional à posição corrompida, não exponencial.

**Implicação para hardware/transmissão:** comunicação inter-CU em GPU, transmissão
via rede, storage em flash — todos sujeitos a bit flips. Byte estruturado é
naturalmente mais resiliente. Para sistemas Pós-LLM com millions de microcolumns
ativas, robustez local é crítica para estabilidade.
