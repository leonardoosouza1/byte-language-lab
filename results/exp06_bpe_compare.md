# Experimento 06 — Qwen BPE real vs byte-positional

**Modelo de referência:** Qwen 2.5-1.5B  
**Tokenizer:** `~/projects/LLM/llm-lab/models/qwen2.5-1.5b/vocab.json` (somente lido, não copiado)  
**N problemas:** 5000 (operações + - × com inteiros 0-99)  
**Seed:** 9001

## Descoberta sobre Qwen tokenizer

Qwen 2.5 (e a maioria dos LLMs modernos) tokeniza dígitos **um por um**.
O vocabulário base não contém tokens multi-dígito como `"47"` ou `"100"`:

| Char | Token ID Qwen |
|------|--------------:|
| `0` | 15 |
| `1` | 16 |
| `2` | 17 |
| `3` | 18 |
| `4` | 19 |
| `5` | 20 |
| `6` | 21 |
| `7` | 22 |
| `8` | 23 |
| `9` | 24 |
| `+` | 10 |
| `-` | 12 |
| `*` | 9 |
| `=` | 28 |

→ Qwen lê `47` como `[19, 22]` (dois tokens), forçando o modelo a aprender
estatisticamente que esses dois tokens consecutivos formam o número 47.
**Byte-positional** carrega essa info estruturalmente nos próprios bits.

## Comparação consolidada

| Encoding | Total bytes/tokens | Por sample | Vocab observado | Entropy (bits/sym) | Perp 2-gram | Perp 3-gram | Perp 4-gram |
|----------|-------------------:|----------:|----------------:|-------------------:|------------:|------------:|------------:|
| ASCII string | 48034 | 9.6 | 15 | 3.839 | 12.11 | 10.53 | 18.13 |
| Qwen BPE tokens | 43034 | 8.6 | 14 | 3.747 | 11.63 | 9.31 | 8.98 |
| Byte-positional (Path B) | 63034 | 12.6 | 16 | 3.717 | 6.37 | 5.55 | 8.14 |
| Path C (domain-tagged) | 81386 | 16.3 | 49 | 4.470 | 3.76 | 3.64 | 4.20 |

## Análise

**Headline finding:**
- Qwen BPE perplexity 3-gram: **9.31**
- Byte-positional perplexity 3-gram: **5.55** (60% do Qwen)
- Path C perplexity 3-gram: **3.64** (39% do Qwen)

Qwen e ASCII têm essencialmente a mesma estrutura (1 token = 1 char). A vantagem da
byte-positional vem da CONFIDENCE byte (Path B) ou DOMAIN HEADERS (Path C) que
criam padrões mais previsíveis em sequências curtas.

**Implicação para Pós-LLM:** se uma arquitetura abandona BPE per-char e usa byte
estruturado, ela vê padrões aritméticos com perplexity ~50% menor — em teoria,
convergiria mais rápido em treino e teria menos erros de carry/posição em geração.

## Como reproduzir

```bash
cd byte-language-lab
cargo run --release --bin exp06_bpe_compare
```

**Não copia modelos.** Só lê `vocab.json` (~2.7MB) por path absoluto. Cargo crate
fica auto-contido (~2KB de tabela inferida sob demanda).
