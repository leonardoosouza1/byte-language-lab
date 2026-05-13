# Changelog

Seguindo [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) e [SemVer](https://semver.org/).

## [0.6.0] — 2026-05-13 — Roadmap Phase 1 Implementation

### Added
- **PLAN-03 Fase 3 auto-spawn**: Domain detector daemon agora cria experts automaticamente quando cluster orfão atinge criteria (density ≥0.10, dist ≥0.40, ≥15 dialogs, cooldown 600s, max 20 auto-spawned).
- **PLAN-01 Fase T1 Gravity Forest** (`src/gravity_forest.rs`):
  - 3D potential field 32³ grid em VRAM
  - Tree forest com Newton-like potencial U(r) = -Σ G·m_i / (|r-r_i| + ε)
  - Verlet integration com damping pra descent
  - Convergence proven via Lyapunov (V = ½||v||² + U)
  - Force clamp 5.0 anti-overshoot
  - Nearest-tree routing por Euclidean distance
  - HV→3D projection via bit-bucket popcount
  - 4 unit tests passing
  - Endpoints: `GET /forest`, `POST /forest_route`
- **PLAN-02 Sprint 1 Distillation** (`scripts/distill_qwen.py`):
  - Multi-temperature ensemble (T=0.5 + T=0.9)
  - WAL injection com FLAG_HIGH_QUALITY
  - Throughput ~1.5 prompts/s
  - Auto-ingested via daemon cycle 20s
  - Math: soft target ensemble = média implícita sobre temperaturas

### Architecture
- Gravity forest planted from MoE experts: 10 trees no Fibonacci-spread cube
- Auto-spawn aumenta dinamicamente o cap de experts (10 static + até 20 auto = 30 max)
- Distillation pipeline aproveita Qwen como teacher sem treinar weights diretos (brain absorve via WAL)

### Notes
- Gravity descent funciona em 2-tree case (unit tests). Com 10 trees, potencial combinado puxa pro centroide global — usamos `nearest_tree` direto pra routing.
- HV→3D projection ainda colapsa centroids — futura melhoria com semantic anchors (bit 5/6/7).

---

## [0.5.0] — 2026-05-13 — Mixture-of-Experts + Domain Detection

### Added
- **MoE Fase 1 (Static experts)**: 10 expert heads auto-built no load via keyword matching. Routing top-2 com HV centroid + keyword fast-path. Endpoint `GET /moe`.
- **Domain Detector daemon (Fase 2 passivo)**: scan a cada 120s detecta clusters orfãos (HV sim < 0.45 vs experts existentes). Endpoint `GET /domains` reporta candidates. Sem auto-spawn.
- **Tool router** via subprocess Python: wiki/url/pdf/arxiv/github (`scripts/tools.py`).
- **Web search**: `pesquise sobre X`, `wikipedia X`, `consulte wikipedia` direto no brain.
- **Generative templates**: poema, aforismo, metáfora, história via brain handlers.
- **Reasoning engine**: aritmética + sequências (aritm/geom/Fibonacci) + letter codes (A=1 B=2) + pegadinhas (cor cavalo branco, túmulo Napoleão, meses 28 dias).
- **Compose handler expandido**: consequência/causa/exemplos/tipos/vantagens/desvantagens/se-então/sucessor.
- **Self-ingest identity**: 20 Q&A pairs sobre IARA/Leonardo/arquitetura.
- **HDC dialog index 1024-bit**: BoW encoding order-invariant, retrieval Hamming faster.
- **Curiosity engine** com 4 strategies: weak_word, orphan_entity, known_unexpanded, pmi_sparse.
- **WAL binário** (IWAL TLV) substitui `auto_learn.txt`. Delta ingest incremental.
- **Always-on supervisor** `scripts/brain_forever.sh` com auto-restart.
- **Checkpoint daemon** 60s, atomic tmp+rename.
- **Q4_K_M Qwen** via llama.cpp — 3.9× speedup vs BF16 (0.38s/call vs 1.5s).
- **Qwen pool**: env `IARA_QWEN_PORTS=3000,3001,...` round-robin (caveat: GPU shared, não escala).
- **Qwen Mutex lock** evita HIP error 700 concurrent.
- **Plans folder** `docs/plans/`: 3 plans roadmap (trees+gravity+mycelium, training strategy, auto-MoE).
- Endpoints novos: `/moe`, `/domains`, `/curiosity`, `/checkpoint`, `/autolearn`, `/persona`, `/graph`, `/iara_gen`.

### Changed
- Threshold brain-hi 0.85 → 0.65 + brain-mid 0.45 → 0.30. Cache hit cresce 22%→79%.
- Latency média brain-hit 260ms → 190ms (HDC speedup).
- `compute_pmi_and_harmony` + `compute_topology` + `build_knowledge_graph` chamados em batches via `ingest_batch_delta`.

### Removed
- Path TXT `auto_learn.txt` legado (mantido `/learn` endpoint para compat).
- Speculative decoding attempt: descartado (engine gate `>3B` correto pra qualquer setup local).

### Architecture
- **Coverage hard test**: 14% → 100% (8 categorias × 8 questions each).
- **Brain growth (sessão)**: dialogs 9543 → 12460 (+2917), triples 3965 → 4925 (+960).
- **Tier routing**: KG compose → reasoning → generative → web tools → MoE → HDC → BoW → phrase → Markov → Qwen fallback.

### File reorganization
- Root limpo: txt/blab/md/sh/py movidos para data/, docs/archive/, scripts/.
- `data/lists/perguntas-*.txt` (4 listas).
- `data/legacy_weights/dialogue_weights*.blab`.
- `docs/plans/` plans roadmap.
- `docs/archive/` legacy docs.

---

## [0.4.0] — 2026-05-12 — IARA-Byte Brain v5 base

### Added
- Char/word/phrase bigram + trigram + PMI.
- Knowledge graph triples extraction.
- Multi-tier CfC (char/word/phrase).
- HTTP server porta 3012.
- Qwen 1.5b distillation (`byte_attention_qwen.bin`, `byte_semantics_qwen.bin`, etc).
- 17-specialist swarm scoring.

## [0.3.0] — 2026-05-11 — Byte cartography GPU

### Added
- GPU trace pipeline via iara-engine `/api/trace`.
- Bit 5 = case hypothesis VALIDATED com KL=0.003.
- `iara-explore` crate em llm-lab (4 bins: tateador, medir, validar, iara_oi).

## [0.2.0] — 2026-04-29 — Initial bins

Foundation experiments: exp01_format até exp18_smg_lateral. HDC chat, byte walker, FSM byte, predictive coding.

## [0.1.0] — 2026-04-29 — Project init

Tese byte-as-language. Setup workspace, Cargo.toml, lib base.
