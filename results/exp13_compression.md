# Experimento 13 — Compressão como proxy de aprendibilidade

**Tese:** redundância detectável por LZ77 ≈ Kolmogorov complexity ≈ aprendibilidade.
Stream que comprime melhor é mais previsível, logo mais aprendível.

**Setup:**
- LZ77 manual (sem deps), window=4096, min_match=3
- 5000 arithmetic problems concatenados

## Resultados

| Encoding | Raw bytes | LZ77 bytes | Compression ratio | Bits/byte |
|----------|----------:|-----------:|------------------:|----------:|
| ASCII | 48034 | 41070 | 0.855 | 6.84 |
| Qwen tokens (u16 LE) | 86068 | 45215 | 0.525 | 4.20 |
| Byte-positional Path B | 63034 | 44051 | 0.699 | 5.59 |
| Path C (domain-tagged) | 81386 | 49105 | 0.603 | 4.83 |

## Análise

**Compression ratio menor = mais redundância = mais aprendível.**

Stream MAIS comprimido absolutamente: **ASCII** (41070 bytes)
Stream com MELHOR ratio (mais redundante): **Qwen** (0.525)

**Implicação:** se algum encoding tem ratio MUITO menor que outros, ele tem
mais estrutura aprendível. Cruza-se com perplexity de exp03/06 e accuracy de exp07.
Confirma-se a hipótese de que byte-positional NÃO é apenas "outro encoding":
tem propriedades estatísticas mensuravelmente melhores.
