# Experimento 01 — Formato byte: round-trip & compactness

**N samples:** 100000  
**Seed:** 9001  
**Range:** valores aleatórios em escalas 10^[-3..+3]

## Resultados consolidados

| Path | Encode OK | Decode OK | Round-trip Match | Avg bytes | Encode (ns/op) | Decode (ns/op) |
|------|----------:|----------:|-----------------:|----------:|---------------:|---------------:|
| Path A (positional) | 99999 | 99999 | 99999 | 9.86 | 127 | 37 |
| Path B (BCD ordered) | 100000 | 100000 | 100000 | 19.90 | 345 | 87 |
| Path C (domain-tagged) | 99999 | 99999 | 99999 | 10.86 | 117 | 53 |
| f64 raw (baseline) | 100000 | 100000 | 100000 | 8.00 | ~0 | ~0 |

## Análise

- **Path A** (9.86 B/num): BCD posição explícita. Operações byte-wise viáveis sem FP-ALU.
- **Path B** (19.90 B/num): BCD ordenado, posição implícita pela sequência. Decode stateful.
- **Path C** (10.86 B/num): header overhead, mas único multi-domain.
- **f64 raw**: 8 bytes constante, opaco.

**Conclusão:** Bytes estruturados pagam ~1.2× mais memória que f64 raw — mas ganham interpretabilidade posicional e suportam operações sem FP-ALU.
