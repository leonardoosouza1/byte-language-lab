# Experimento 03 — Learnability: byte-positional vs ASCII-string

**N problemas:** 5000 (operações + - × com inteiros 0-99)  
**Seed:** 9001  
**Train/test split:** 80% / 20%

## Resultados

| Métrica | ASCII | Byte-positional | Razão (BP/ASCII) |
|---------|------:|----------------:|-----------------:|
| Total bytes | 48034 | 63034 | 1.31× |
| Bytes/problema (médio) | 9.61 | 12.61 | 1.31× |
| Vocab size (símbolos distintos) | 15 | 16 | — |
| Shannon entropy (bits/byte) | 3.839 | 3.717 | 0.97× |
| Perplexity 2-gram | 12.11 | 6.37 | 0.53× |
| Perplexity 3-gram | 10.53 | 5.55 | 0.53× |
| Perplexity 4-gram | 18.13 | 8.14 | 0.45× |

## Interpretação

- **Perplexity menor = mais previsível para um modelo n-gram.**
- **Shannon entropy menor = stream mais redundante (mais aprendível).**

**Veredito empírico:**
- Compactness: ASCII vence (63034 bytes vs 48034 bytes)
- Perplexity 3-gram: **byte-positional vence** (5.55 vs 10.53)
- Perplexity 4-gram: **byte-positional vence** (8.14 vs 18.13)

## Observações importantes

- Esta tarefa (arithmetic completion com inteiros 0-99) tem vocab restrito; ASCII tem ~14 chars distintos, byte-positional ~16. Diferenças sutis em perplexity refletem ordem das sequências (mais que vocab).
- O ganho REAL de byte-positional aparece em tarefas onde POSIÇÃO do dígito importa para o cálculo (ex: aritmética com fracionais). N-gram não captura isso plenamente.
- Para validação completa da hipótese 'inteligência fala bits', precisamos modelo neural que possa explorar a estrutura posicional — fora do escopo desta lab inicial.
