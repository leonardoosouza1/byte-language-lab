# IARA Byte Chat — Manual de Uso

Sistema de chat PT-BR baseado em **byte-level retrieval** com hyperdimensional computing. Sem rede neural em runtime. Microsegundos por query. ~750KB total.

---

## Quick Start

```bash
cd ~/projects/LLM/byte-language-lab

# Build os binários principais
cargo build --release --bin iara_chat_rerank --bin iara_intelligent_chat

# Verifica que o engine Qwen está rodando (necessário pra primeira run gerar byte HVs)
curl -s http://localhost:3000/health | jq

# Roda chat principal (recomendado)
./target/release/iara_chat_rerank
```

---

## Variantes disponíveis

| Binário | Descrição | Quando usar |
|---|---|---|
| **`iara_chat_rerank`** ★ | HDC top-K + Levenshtein re-rank | **Default**: robusto a typos, <400μs |
| `iara_intelligent_chat` | HDC + Qwen fallback (sim<0.85) | Quando precisa generativo |
| `iara_hdc_chat_v2` | HDC puro v2 (bag+bigram+seq) | Pure bit ops, sem Qwen runtime |
| `iara_hdc_chat` | HDC v1 (XOR+pos) | Mais simples, baseline |
| `iara_byte_chat` | Bigram + char-match (sem HDC) | Comparação CPU pura |

## Comandos no chat

- Texto livre: query normal
- `/quit` ou `/exit`: sair
- `/show N`: mostrar top-N candidatos (apenas v2 e variantes que suportam)
- `/reset`: limpar contexto multi-turn (apenas intelligent)
- `/stats`: stats da sessão

## Suítes de avaliação

```bash
# Compara variantes em 40 prompts
./target/release/iara_chat_eval

# Testa robustez a typos: Qwen-HV vs Phonetic-HV
./target/release/iara_typo_eval

# Análise bit dos caracteres ASCII (valida tese)
./target/release/byte_harmony

# Vector arithmetic / vizinhança semântica
./target/release/iara_hdc_analogy
```

## Re-build dos byte HVs

Apenas necessário se mudar `build_phonetic_hvs.rs` ou perder o cache:

```bash
# HV Qwen vanilla (1024-bit sign de Qwen embeddings)
./target/release/iara_hdc_chat   # auto-build na primeira execução, salva em data/byte_hvs_qwen.bin

# HV fonética PT-BR (256 fonética + 768 Qwen)
./target/release/build_phonetic_hvs
```

## Aumentar corpus

```bash
# One-shot rápido: 200 pares from Qwen
python3 data/qwen_augment.py

# Sustained background: até 5700 pares (1.5h)
nohup python3 data/qwen_augment_sustained.py > /tmp/qwen_sustained.log 2>&1 &
```

O chat reaprende automaticamente: cada Qwen fallback no `iara_intelligent_chat`
salva em `data/online_learning.txt`. Próxima execução tem cobertura maior.

## Tese bit-level

Verificação dos bits em letras ASCII:
```
bit 7 (MSB):    sempre 0 ASCII alpha     ✓
bit 6 (struct): sempre 1 ASCII alpha     ✓
bit 5 (case):   diferencia A/a, B/b...    ✓ 50.98% pares alpha
bit 4 (group):  vogal/cons grupo          ✓
bits 3-0:        posição no alfabeto      ✓
```

Hamming(O, I) = **2** (em 8 bits) — apenas bits 1,2 diferem. "OI" é geometricamente próximo no espaço byte.

## Performance benchmark

Corpus: 5770 pares, hardware: AMD RX 6750 XT (Qwen na GPU) + Ryzen CPU (HDC)

| Sistema | Storage | Setup | Query lat | Qualidade |
|---|---|---|---|---|
| Qwen GPU direto | 1.51 GB VRAM | ~5s reload | ~15ms (1 tok) | ★★★★★ |
| **HDC + rerank** | **~750 KB** | 335ms | **<400μs** | ★★★★ in corpus |
| HDC v2 puro | ~700 KB | 295ms | 206μs | ★★★ in corpus |
| byte_chat | ~500 KB | 7ms | 1410μs | ★★ |
| SMG distill | 4.5 MB | 19s train | ~5ms | ★ (incoerente) |

## Limitações conhecidas

- Out-of-corpus queries: HDC tende a matchar algo próximo ortograficamente, não semanticamente
- Multi-turn: contexto via bundle, decai rápido (5 turns)
- Strings curtas (1-3 chars): colapso XOR no `seq` signature; bag still works
- Typos com 3+ chars trocados: ~50% miss rate

Mitigação: usar `iara_intelligent_chat` que dispara Qwen quando sim < 0.85.

## Stack técnica

- **Rust 1.85** (edition 2024)
- **wgpu** (apenas se compilar com `--features gpu-aql`)
- **HSA AQL** infra (kernel HIP existente, não usado em HDC chat)
- **serde_json** (parsing vocab Qwen)
- **CPU only** runtime para HDC chat (sem GPU dependency)
- **Qwen via /api/generate** (apenas para fallback ou augment)

Sem GPU em produção, sistema roda em qualquer máquina com Rust.
