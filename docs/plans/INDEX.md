# Plans Index — IARA-Byte Brain

Roadmap docs, não executados. Cada plan é independente, math-validado.

## Lista

| # | Arquivo | Tópico | Tempo | Status |
|---|---|---|---|---|
| 01 | PLAN-01-trees-gravity-mycelium.md | B-spline forest + gravity field + mycelium graph em VRAM | 3-5 sem | roadmap |
| 02 | PLAN-02-training-strategy.md | Training math + distillation + VRAM expansion + 8 ideas extras | 10 sem | roadmap |

## Como executar

Cada plan tem fases isoladas validáveis. Padrão recomendado:
1. Sprint 1 — começar com kernel isolado (gravity_descent)
2. Validar com microbenchmark
3. Iterar antes de escalar

## Referências cruzadas

- Pipeline atual: brain_v5.bin + WAL + Qwen Q4_K_M (operando, 100% hard test)
- Obsidian GPU notes: `~/obsidian-claude-vault/Claude/GPU-RDNA2-HSA/`
- Existing kernels: `llm-lab/compute-core/src/backend.rs`
- Gravity já em uso: `llm-lab/iara-engine/src/lnn/gravity.rs`
- DNA-V4 format: `llm-lab/compute-core/src/dna_format.rs`
- Universe (B-spline brake/throttle, dual-genome): `~/projects/LLM/Universe/`
