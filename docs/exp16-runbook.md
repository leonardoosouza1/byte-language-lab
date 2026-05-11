# exp16 — Runbook (SMG-tiny distillation from Qwen 2.5 1.5B)

Implementa o marco **M1** do `smg-design.md`. Valida que destilação de Qwen
(prompt → response em bytes) é aprendível por modelo minúsculo.

## Pipeline

```
Qwen 2.5 1.5B (teacher)            byte-language-lab (student)
─────────────────────────          ────────────────────────────
iara-engine /api/generate          exp16_smg_distill
        │                                    ▲
        ▼                                    │
dump_qwen_traces                  read_traces (formato QTRC v1)
        │                                    ▲
        ▼                                    │
data/qwen_traces.bin  ──────────────────────┘
```

## Pré-requisitos

- ROCm + AMD GPU (RX 6750 XT testado), OU OpenCL (Intel iGPU) como fallback.
- Pesos Qwen 2.5 1.5B em `llm-lab/models/qwen2.5-1.5b/` (já presente).
- iara-engine compilado (release, com features `rocm` ou `opencl`).

## Passo 1 — Subir iara-engine

Em um terminal dedicado:

```bash
cd ~/projects/LLM/llm-lab
./scripts/run_iara.sh release       # ROCm (RX 6750 XT)
# ou: ./scripts/run_igpu.sh         # OpenCL (Intel iGPU)
```

Em outro terminal, carregue o modelo:

```bash
curl -X POST http://localhost:3000/api/load-model \
     -H 'content-type: application/json' \
     -d '{"name":"qwen2.5-1.5b"}'
```

Aguarde até `/health` reportar modelo carregado (logs do engine confirmam).

## Passo 2 — Dump de traces (1 vez, ~5-10 min)

```bash
cd ~/projects/LLM/llm-lab/iara-byte-bench
cargo run --release --bin dump_qwen_traces -- \
  --prompts ../../byte-language-lab/data/exp16_prompts.txt \
  --out     ../../byte-language-lab/data/qwen_traces.bin
```

Saída: 100 linhas `[i/100] prompt → response_preview`. Ao final:
`gravado: ../../byte-language-lab/data/qwen_traces.bin` (~30-100 KB).

## Passo 3 — Treinar SMG-tiny

```bash
cd ~/projects/LLM/byte-language-lab
cargo run --release --bin exp16_smg_distill
```

Saída esperada (~1-3 min):

- vocab_size: ~30-60 bytes únicos
- training pairs: ~30k-80k
- loss curve até < 0.05
- Veredicto vs critério M1 (≥ 90% byte-agreement)

Resultados em `results/exp16_smg_distill.md`.

## Critérios (vindos de `smg-design.md` §6)

| Resultado | Significado |
|-----------|-------------|
| **agreement ≥ 90%** | M1 atingido. Destilação valida. Avança para exp17 (estrutura microcoluna real). |
| **70-90%** | Sinal aprendível mas com erro. Antes de exp17, tentar window maior, vocab refinado, mais reps. |
| **50-70%** | Sanitização agressiva ou capacidade insuficiente. Revisar antes de avançar. |
| **< 50%** | Falha conceitual. Hipótese "destilação resolve treino" precisa ser repensada. |

## Próximos passos pós-exp16

Independente do resultado:

1. Se ✓: começar `exp17_smg_microcoluna` — substitui MLP por grade 8×8 microcolunas
   FSM com transição contínua durante treino, discretizada na avaliação.
2. Se ~: investigar com `exp16b` (variantes de window/hidden/reps).
3. Se ✗: voltar ao spec §5 (Riscos conhecidos) e reformular.

## Troubleshooting

- **`Não consegui ler data/qwen_traces.bin`** — pulou Passo 2.
- **Connection refused na porta 3000** — iara-engine não está rodando ou não carregou modelo.
- **Loss não converge** — provavelmente `WINDOW` muito pequeno para a maior resposta. Aumentar para 32.
- **Vocab > 80 bytes** — Qwen está retornando UTF-8 rico. Reforçar `sanitize()` ou aceitar overhead.
