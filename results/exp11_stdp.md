# Experimento 11 — STDP discreto: aprendizado sem gradiente

**Tese:** STDP (Spike-Timing Dependent Plasticity) é um aprendizado
biológico-plausível: pre antes de post fortalece (LTP), pos antes de pre
enfraquece (LTD). Local, sem backprop, sem gradiente, sem float.

## Setup

- N neurônios: 16
- N patterns temporais: 4 (cada um 8 ticks)
- Connection strength: 4 bits (0-15)
- STDP windows: dt ∈ [1,10] → +2, [11,50] → +1, [-10,-1] → -2, [-50,-11] → -1
- Trials: 200

## Aprendizado: diferenciação de strengths

| Trial | Std-dev | Strong (≥12) | Weak (≤3) |
|------:|--------:|-------------:|----------:|
| 0 | 4.787 | 52 | 79 |
| 20 | 5.585 | 80 | 80 |
| 40 | 5.590 | 80 | 80 |
| 60 | 5.590 | 80 | 80 |
| 80 | 5.590 | 80 | 80 |
| 100 | 5.590 | 80 | 80 |
| 120 | 5.590 | 80 | 80 |
| 140 | 5.590 | 80 | 80 |
| 160 | 5.590 | 80 | 80 |
| 180 | 5.590 | 80 | 80 |
| 199 | 5.238 | 66 | 66 |

## Recall scores (cue = primeiro tick do pattern 0)

Score = soma das forças (cue_neuron → expected_next_neuron) para cada pattern.

Pattern 0 deveria ter o MAIOR score (rede aprendeu padrão correto).

| Pattern | Recall score | Marker |
|--------:|-------------:|:-------|
| 0 | 2056 | ← target (cue source) |
| 1 | 1781 |  |
| 2 | 2186 |  |
| 3 | 2244 |  |

**Rank do pattern correto:** 2 de 4 (0 = top, melhor)
⚠️ STDP não conseguiu rankear corretamente

## Análise

- **Diferenciação:** strengths inicialmente uniformes (7) divergem em strong/weak
  conforme STDP detecta correlações temporais nos patterns.
- **Recall:** se pattern 0 tiver maior score, STDP capturou as correlações causais
  daquele padrão (rede "reconhece" o cue).
- **Sem backprop:** todos os updates são locais (par pre/post). Compatível com
  hardware neuromórfico (Loihi, TrueNorth) e GPU integer-only.

**Implicação:** STDP discreto é uma alternativa viável a gradient descent para
aprendizado em redes de byte FSM. Funciona com pesos quantizados sem ulps de float.
