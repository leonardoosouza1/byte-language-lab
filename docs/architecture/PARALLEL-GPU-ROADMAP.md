# Parallel GPU — Roadmap for a Shared IARA Brain

Plan for true parallel execution on one GPU (and beyond), as the foundation
for a shared brain that multiple workers — eventually multiple machines —
train and serve concurrently.

## Goals

- **Concurrent HSA streams** on a single RX 6750 XT (RDNA2, gfx1032).
- **Shared model, private gradients** — many workers contribute to one brain.
- **Saci-style saturation** — sustained 180–200 W draw under load.
- **Foundation for distributed** — same primitives later span multiple GPUs
  and machines without architectural rewrite.
- **Correctness-first** — every step has numerical parity tests against
  the single-stream reference.

## Non-goals (defer)

- Multi-machine networking (Phase 5 stub only).
- Kernel rewrite for bigger K / grid (separate roadmap).
- Pipeline parallelism across layers (SMG is too shallow to benefit).

## Why now

Current SMG HSA pipeline saturates GPU **use% to 99 but power to 116 W** —
classic memory-bound profile (state buffer 4 KB × batch × window is the hot
data). Pure batch scaling can't break that. Multi-stream + async overlap
gives more independent compute pressure per memory cycle, lifting power.
It also unlocks the multi-process / multi-machine roadmap.

---

## Architecture — HSA Multi-Queue Worker (HMQW)

```
   ┌─────────────────────── orchestrator ───────────────────────┐
   │  data shards · curriculum · LR · validation · brain commit │
   └──┬──────────┬──────────┬──────────┬───────────────────────┘
      │          │          │          │
   ┌──┴──┐   ┌──┴──┐    ┌──┴──┐    ┌──┴──┐
   │ W0  │   │ W1  │    │ W2  │    │ W3  │     ← N workers, each owns:
   │ HSA │   │ HSA │    │ HSA │    │ HSA │       • 1 HSA queue + signal pool
   │  Q  │   │  Q  │    │  Q  │    │  Q  │       • private state/grad buffers
   └──┬──┘   └──┬──┘    └──┬──┘    └──┬──┘       • private kernarg ring slice
      │         │          │          │
      └─────────┴──┬───────┴──────────┘
                   │ all-reduce sum of grads (atomic on shared g_*)
                   ▼
         ┌──────────────────────────┐
         │  shared model weights     │   ← single source of truth on device
         │  (mmap or DeviceBuffer    │     (read-many during forward,
         │   under Arc/RwLock)       │      written by SGD step only)
         └──────────────────────────┘
```

### Component contracts

| Component | Owns | Reads | Writes |
|-----------|------|-------|--------|
| Worker | queue, signals, kernarg slice, private fwd/bwd buffers | shared weights | private grads, then atomic-add to shared grads |
| Orchestrator | data shards, schedule, val results | val loss | shared weights via SGD step at sync points |
| Shared weights | device buffer | all workers (forward), SGD step (write) | SGD step only |
| Shared grads | device buffer initialized to 0 | SGD step (reduce + clear) | atomic-add by workers |

---

## Phases

### Phase 1 — Stream foundations (no model change)

Goal: prove two HSA dispatchers can co-exist in one process.

1. Refactor `HsaDispatcher` → per-instance queue, signals, kernarg pool.
   Today it's effectively a singleton wrapper (`SmgRuntime`).
2. Add `HsaCtx` (shared) holding agent, hsaco modules, kernel handles.
3. Spin 2 dispatchers, run two independent kernels concurrently on a
   throwaway tensor, observe via `rocm-smi --showuse --showpower`.
4. Parity test: 2 streams running the SMG forward on disjoint data must
   match 1 stream running them sequentially (bitwise within ffast-math tol).

Exit: 2 concurrent dispatches on the same agent, no HIP 700, parity green.

### Phase 2 — Per-worker buffers + shared weights

Goal: each worker has its own forward/backward state; weights are shared.

1. Split `SmgGridHsa` into `SharedSmg { weights, hsa_ctx }` and
   `SmgWorker { worker_id, shared: Arc<SharedSmg>, private_buffers }`.
2. Forward: worker reads shared weights, writes private `pst_st` / `pst_log`.
3. Backward: worker writes private `pst_dpre` / `pst_dlog` / private grad
   staging buffers.
4. Reduce: worker `atomicAdd` private grads → shared grad buffer (kernel
   already does atomic adds for grad_acc — adapt to inter-worker target).
5. SGD step: orchestrator only — single writer, after barrier.

Exit: 1-worker run matches the legacy SmgGridHsa exactly; 4-worker run
converges to the same val loss within noise.

### Phase 3 — Async pipeline (the saturation win)

Goal: overlap host→device, forward, backward, SGD; lift power past 150 W.

1. Triple-buffer per worker: while step N runs forward, step N-1 backward
   is finishing and N+1 H2D is staging.
2. Use HSA dependency signals (kernel barrier packets) — every dispatch
   posts a completion signal the next one depends on.
3. Drop the `drain()` between forward and backward where the data flow
   permits it; keep one drain at the end of a micro-batch only.
4. Power profiling: `rocm-smi --showpower` sampled at 1 Hz during a run.
   Target: ≥150 W sustained at 4 workers, ≥180 W at 8.

Exit: end-to-end loss curve unchanged vs Phase 2, wall-clock per epoch
≥2× faster at 4 workers, power ≥150 W sustained.

### Phase 4 — Multi-process / shared brain (single host)

Goal: workers are processes, not threads. Foundation for "the brain is
hosted somewhere, workers connect and contribute."

1. Move shared weights to `mmap` on `/dev/hugepages` or anonymous shm.
2. Each worker is its own process; opens the shm; constructs HsaCtx;
   spins its queue.
3. Gradient sync via lock-free SPSC queue per worker → orchestrator
   process performs the reduce + SGD step.
4. Risk: prior session hit HIP 700 with two iara-engine processes on
   the same GPU. Diagnosis was concurrent dispatches racing on shared
   GPU buffers. The HMQW design avoids that — each process has its own
   queue and private buffers — but **add a smoke test as gate**: 2
   trivial dispatches, 2 processes, 10k iterations, no HIP 700.

Exit: 4 worker processes train one shared brain to the same loss curve
as Phase 3 in-process, no GPU crash.

### Phase 5 — Multi-machine (stub)

Goal: same orchestrator API, but workers can live on another box.

1. Replace IPC SPSC with a thin TCP gradient-exchange protocol — same
   reduce semantics.
2. Use the existing `data/auto_learn.wal` pattern as the brain's
   append-only log; merge across machines is just WAL union + dedup
   (HDC + KG are set-mergeable — see [federated brain
   memory](../../../../../.claude/projects/-home-leonardo-projects-LLM/memory/MEMORY.md)).
3. Out of scope right now — record the protocol surface so Phase 4
   primitives don't paint us into a corner.

---

## Validation plan

Every phase ships with:

- **Numerical parity** — `cargo test` comparing N-worker run vs 1-worker run
  on a tiny fixed corpus; loss curves identical within rounding tolerance.
- **Power/util trace** — `rocm-smi` sampled, plotted; each phase must show
  monotonic improvement or be rolled back.
- **Held-out score** — `scripts/heldout_eval.py` runs against the resulting
  brain; the rollback gate of `train_v2.sh` applies here too.
- **HIP 700 watchdog** — run for 10k iterations under stress before declaring
  a phase done (the engine multi-process bug was caught only under load).

## File layout (planned)

```
src/gpu_aql/
  ctx.rs           ← HsaCtx (agent, modules, kernel handles) — shared, immutable
  worker.rs        ← SmgWorker (queue, signals, private buffers) — per-worker
  shared.rs        ← SharedSmg (weight DeviceBuffers, Arc) — single source of truth
  reduce.rs        ← grad atomic-add + SGD step orchestration
  dispatch.rs      ← (existing) per-kernel ParamsXxx structs, dispatch helpers
bin/
  train_parallel.rs   ← N-worker orchestrator binary (Phase 2+)
  bench_parallel.rs   ← latency + power benchmark
docs/architecture/
  PARALLEL-GPU-ROADMAP.md   ← this file
```

## Open questions

1. **Atomic add on device-side grad buffers** — current `grad_acc_*` kernels
   already atomically reduce per-thread contributions; verify the same path
   works for inter-worker (just larger reduce tree).
2. **Signal pool size** — HSA signals are a finite resource; need a per-queue
   pool with reuse, not allocate-per-dispatch.
3. **LR scaling for N workers** — effective batch grows N×; linear LR scaling
   is the safe default but warmup may need extension.
4. **Determinism** — at N workers, grad reduce order is nondeterministic. For
   parity tests, force a sync barrier per micro-batch.

## What this unlocks

- **Today** — same single GPU, same kernels, but ≥150 W sustained and faster
  wall-clock per epoch.
- **Tomorrow** — multiple processes contribute to one brain on one machine:
  the orchestrator validates each contribution before commit (ties cleanly
  into `train_v2.sh`'s held-out gate + rollback).
- **Future** — same primitives extend to a federated brain across machines.
  Phase 5 plans for that without committing engineering now.
