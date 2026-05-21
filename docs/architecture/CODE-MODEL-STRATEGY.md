# Code Model Strategy — a cheap, local, specialized code generator

The bet: not beating Claude on raw capability (impossible on one consumer
GPU), but a **fast, local, free, specialized** code model that is good
enough on the common 90% of web-dev tasks — instant, on-device, private.

## How a code LLM (Claude-class) is actually trained

```
1  PRETRAIN          next-token over trillions of tokens (web + code).
                     learns syntax/semantics implicitly. thousands of GPU,
                     months. THE expensive stage.
2  CODE DATA MIX     GitHub (permissive) + docs + notebooks; dedup, quality
                     + license filter; code up-weighted.
3  MID-TRAIN / FIM   code-focused phase. repo-level (cross-file) context.
                     Fill-In-the-Middle objective (predict middle from
                     prefix+suffix) — this is what powers completion/edit.
4  SFT               (instruction -> response) pairs; for code
                     (task -> solution). teaches following requests.
5  EXECUTION-RL       generate N -> RUN / pass tests / compile -> reward the
                     one that passes (GRPO/DPO/RLHF). where "code that works"
                     is reinforced. the secret to correctness.
6  AGENTIC           tool use (read/edit/bash), multi-turn, planning;
                     trajectory data of agents solving tasks.
```

Code-specific stages: **FIM (3)**, **execution-RL (5)**, **agentic (6)**.

## Our cheap mirror (each stage, what's feasible here)

```
CLAUDE STAGE           OUR CHEAP EQUIVALENT
─────────────────────────────────────────────────────────────────
1 pretrain (trillions) → DISTILL from a teacher (Qwen3 now; Gemma /
  — infeasible           DeepSeek later). student inherits the base,
                         skips the expensive stage. ONLY viable path.
2 code data mix        → HAVE IT. 35MB local harvest + the-stack-smol.
                         dedup + Prettier-normalize (lower entropy → small
                         model learns faster).
3 FIM / repo context   → CHEAP. FIM is just masking the middle of the same
                         data. big win for edit/complete; low cost.
4 SFT (intent->code)   → HAVE IT. distill_code pairs ARE SFT data; today
                         they feed the cache, can also train a model.
5 execution-RL         → CHEAP + the differentiator. we have node/parser/
  ← the secret           lint. generate N -> run -> reward what passes.
                         reward is FREE and OBJECTIVE. how a small model
                         gets good without scale: it iterates against the
                         interpreter (ground truth), not against data scale.
6 agentic              → embryo exists: code_agent decomposes + stitches.
```

Central idea: **swap the expensive pretrain for distillation; copy the
cheap execution-RL.** The "generate → run → learn from failure" loop is how
a small specialized model can become genuinely good at code without a GPU
cluster — the reward signal doesn't need scale.

## The moat Claude can't have

```
SHARED BRAIN (HDC-mergeable)  LLM weights don't merge across users; our
  KG + dialog_pairs DO. each user contributes → everyone's brain grows.
  network effect on a closed-weight monolith is impossible.
LOCAL + FREE + INSTANT  cache serves the common 90% in µs at zero cost;
  only novel escapes to the generator. private, offline, laptop-class.
```

## Hard constraint discovered (2026-05-20)

The SMG generator is **LDS-bound**, not just kernel-hardcoded:

```
state lives in LDS (RDNA2 = 64 KB/CU). K=32 uses ~32 KB. K=64 ~ fits at
the limit. K=128 overflows. WGSL (wgpu) AND HSACO both hardcode
K=32/N_CELLS=16/FLAT_DIM=4096 (compile-time const / include_str!).

→ scaling SMG by K caps at ~K=64 (~4.5M params) — still too small for
  real code. Meaningful scale needs MORE CELLS or a different architecture
  (byte-transformer with state in VRAM, not LDS).
```

So the student architecture decision: **byte-transformer (state in VRAM,
arbitrary size) is the right target, not a bigger SMG.** It needs new
kernels (tiled GEMM + attention + layernorm), est. 3-5 weeks of HSA/wgpu
kernel work — the real prerequisite for any sizeable generator.

## Roadmap (ordered by dependency)

```
P0  DATA PIPELINE (now, architecture-agnostic) ──────────── THIS COMMIT
    distill_teacher.py: large (intent->code) SFT set + FIM examples +
    structural annotations, organized by tech. multi-teacher ready.
    feeds the cache today AND any future trained student.

P1  STUDENT KERNELS  byte-transformer kernels (GEMM/attention/norm) on
    wgpu first (runtime-compiled, arbitrary size, lower risk than HSACO),
    HSA later for speed. ~weeks. unblocks training anything real.

P2  DISTILL BASE     sequence-level KD: train the student on teacher
    completions. then logit/on-policy KD to close the gap.

P3  FIM              add fill-in-the-middle masking to the training data
    + objective. edit/complete capability.

P4  EXECUTION-RL     generate N -> run via node/parser/lint -> reward pass.
    GRPO-style. the correctness engine. cheap, local, objective.

P5  AGENTIC + RAG    code_agent decompose/stitch + KG retrieval as context
    (sketch-and-fill, grammar-constrained decoding).

P6  FEDERATED        merge brains across machines (WAL union; HDC/KG are
    set-mergeable). the network-effect moat.
```

## Teachers (multi-teacher distillation)

```
now:    Qwen3-4B-Q4 (local, running). general + decent code.
next:   a code specialist (Qwen2.5-Coder, DeepSeek-Coder, CodeGemma) —
        better code targets. distill_teacher.py is teacher-agnostic
        (just an endpoint + model name), so adding a teacher = config.
ensemble: distill from multiple teachers, keep what passes execution-RL.
```

## Honest cost

```
from-scratch good model = millions in compute. NOT here.
distill from teacher on 1 GPU = viable (days, low cost). THE path.
prerequisite for any sizeable student = the kernel work (P1).
without it: stuck at the toy SMG (1M, LDS-bound).
```
