# byte-language-lab — IARA-Byte Brain v0.5

> **Tese:** inteligência fala bytes estruturados, não tokens nem floats.
> Streams de bytes com positional coding como substrate fundamental.

## O que é

Brain híbrido byte-level que cresce continuamente via auto-learn loop usando Qwen como teacher.

- **256 bytes** vocab fundamental (vs 151k tokens BPE de LLMs).
- **Bit 5 = case** validado empiricamente (KL=0.003).
- **Knowledge unificado** em arquivo binário (~18 MB) com 12k+ dialogs, 4.9k+ KG triples.
- **MoE static** com 10 experts auto-routed por keyword + HV centroid.
- **Tool routing**: wiki, search, url, pdf, arxiv, github via subprocess Python.
- **100% cross-domain coverage** em hard test (8 categorias, 64 questions).

## Quick start

```bash
# 1. Build
cargo build --release --bin iara_brain_full

# 2. Carregar Qwen no iara-engine (separadamente)
cd ~/projects/LLM/llm-lab && ./scripts/run_iara.sh release
curl -X POST http://127.0.0.1:3000/api/load-model \
  -H 'Content-Type: application/json' \
  -d '{"name":"qwen2.5-1.5b-q4km"}'

# 3. Start brain (always-on supervisor)
./scripts/brain_forever.sh serve

# 4. Conversa
curl -s -X POST http://127.0.0.1:3012/chat \
  -H 'Content-Type: application/json; charset=utf-8' \
  -d '{"prompt":"qual a capital do Brasil"}'
```

## Endpoints

| Endpoint | Método | Descrição |
|---|---|---|
| `/chat` | POST | Chat principal — tier routing (KG→compose→reasoning→generative→tools→MoE→HDC→BoW→phrase→Markov→Qwen) |
| `/iara_gen` | POST | Generative byte-native (experimental) |
| `/stats` | GET | bytes, words, phrases, dialogs, triples counters |
| `/persona` | GET | User stylometric profile |
| `/graph` | GET | Top byte edges + total active |
| `/moe` | GET | MoE experts state |
| `/domains` | GET | Domain detector emerging clusters |
| `/autolearn` | GET | Auto-learn daemon state |
| `/curiosity` | GET | Curiosity engine state |
| `/curiosity/start` | POST | Start N workers |
| `/curiosity/stop` | POST | Stop workers |
| `/checkpoint` | GET/POST | Brain checkpoint status / manual save |
| `/learn` | POST | Manual ingest (legacy TXT path) |

## Arquitetura

Veja [docs/](docs/) + [docs/plans/](docs/plans/) para deep dives:

- `PLAN-01-trees-gravity-mycelium.md` — Trees + Gravity + Mycelium VRAM architecture
- `PLAN-02-training-strategy.md` — Training math + multi-teacher distillation
- `PLAN-03-auto-specialist-moe.md` — Auto-specialist MoE 3 fases evolutivas

## Estrutura do projeto

```
byte-language-lab/
├── bin/
│   ├── iara_brain_v5.rs    ← brain principal (compila como iara_brain_full)
│   ├── exp01..exp18_*.rs   ← experiments arquivados (research)
│   └── chat_hsa.rs, ...    ← HDC/HSA bins legacy
├── src/
│   ├── lib.rs              ← blab crate (shared)
│   ├── arithmetic/, compose/, format/, fsm_byte/
│   ├── gpu/, gpu_aql/      ← experimental kernels
│   └── learn/
├── data/
│   ├── brain_v5.bin        ← brain state (~18MB)
│   ├── auto_learn.wal      ← WAL pendente (drenado por daemon)
│   ├── byte_*.bin          ← Qwen distilled artifacts
│   ├── lists/              ← perguntas-{500,1500,5k,10k}.txt
│   └── legacy_weights/     ← old .blab files
├── scripts/
│   ├── brain_forever.sh    ← always-on supervisor
│   ├── tools.py            ← wiki/search/url/pdf/arxiv/github
│   ├── battery_parallel.py ← bench parallel queries
│   ├── hard_test.py        ← cross-domain validation
│   ├── iq_test.py          ← intelligence probe
│   ├── gap_fill_loop.py    ← targeted domain learning
│   └── self_ingest.py      ← identity Q&A ingest
├── docs/
│   ├── plans/              ← roadmap planning
│   └── archive/            ← legacy DESIGN, PROGRESS, USAGE
├── kernels/                ← GPU kernels (research)
├── tests/                  ← integration tests
└── benches/                ← criterion benchmarks
```

## Versão atual: v0.5.0

Veja [CHANGELOG.md](CHANGELOG.md) para histórico completo.

Highlights v0.5:
- MoE static + domain detection passivo
- Tool router (wiki/url/pdf/arxiv/github)
- HDC dialog index 1024-bit BoW
- Compose handler expandido (10+ patterns)
- Reasoning engine + generative templates
- Web search Wikipedia
- 100% hard test coverage
- 3.4× throughput speedup via Q4_K_M

## Hardware

- AMD RX 6750 XT 12GB GDDR6 (GFX1032, RDNA2, wave32)
- ROCm 7.2.1 com `HSA_OVERRIDE_GFX_VERSION=10.3.0`
- Intel Iris Plus Gen 11 (notebook fallback)

## Roadmap

Veja [docs/plans/INDEX.md](docs/plans/INDEX.md).

## License

MIT.
