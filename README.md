# byte-language-lab

**A Rust lab for byte-level language models: a transformer and a liquid/SSM core written from scratch on
wgpu (WGSL) and HIP, a JavaScript deminifier that fits in 4 MB, and a byte/HDC "brain" that uses Qwen as a
teacher.**

The thesis is that 256 byte values are enough vocabulary, and that the bits of a byte carry structure a model
can use (for example, bit 5 is ASCII case). Everything here is Rust with minimal dependencies; GPU kernels are
our own WGSL shaders (through `wgpu`) or HIP kernels dispatched through HSA AQL packets. Developed and measured
on an AMD Radeon RX 6750 XT.

Every number below comes from a logged run in this repository's history or result files.

---

## Highlights

| What | Result | Baseline / control |
|---|---|---|
| Deminify `.min.js` with a byte model + constrained decoding | 10/10 real minified files from `node_modules` come out as valid JS (`node --check`), content 100% preserved | similarity to the original 0.926 vs 0.863 for copying the input unchanged |
| Liquid/SSM core (minGRU-style diagonal recurrence) vs our byte transformer | 1.938 bits/byte (d256, 6 layers, 1.25M params, 2.56 epochs) | transformer 3.71 bits/byte on the same 18 MB corpus (not compute-matched, see notes) |
| Device-resident tensors in the wgpu stack | forward 8.0× faster, parity 1.6e-4; training 1500 steps 28 s → 15.7 s | per-op host readback |
| On-device gradient clipping + block-diagonal batching | 1500 steps 15.7 s → 6.71 s; batch 8 at 53k tokens/s (5.3×) | |
| Batched generation without a KV cache | 362 → 2,399 bytes/s at batch 8 (6.6×) | |
| HDC brain on a fixed held-out set of 50 factual questions | 90% answered without calling the teacher, after enabling the gravity forest (was 66%) | keyword-scored, small set |

---

## What is in here

### 1. From-scratch GPU training stack (`src/gpu/`)

- `transformer.rs`: byte-level transformer with forward, backward, AdamW, gradient clipping, multi-token
  prediction heads (t+2) and speculative generation. Attention is gradient-checked (`attention_gradient_check`);
  Q/K gradients are proven end to end by `transformer_beats_unigram_floor` (loss 5.50 → 0.001 on a memorization
  task).
- `ssm.rs`: liquid/SSM core, `z_t = σ(x·Wz)`, `h_t = z_t·h_{t−1} + (1−z_t)·u_t`. The recurrence is diagonal, so it
  parallelizes per channel without an associative scan. Gradient-checked against CPU (forward 2e-4, backward
  ~1e-5).
- `device.rs`, `gemm.rs`, `nn_ops.rs`: device-resident tensors, cached pipelines, fused ops.
- `src/gpu_aql/` + `kernels/*.hip` (feature `gpu-aql`): HIP kernels for the SMG model, dispatched through HSA AQL
  packets without Vulkan.

A bug worth recording: the causal softmax leaked the future into its denominator. It showed up as a
deterministic NaN once Adam grew the weights, and is now covered by `causal_softmax_masks_future`.

### 2. Deminify (`src/bin/*deminify*`, `web/deminify.html`, `DEMINIFY.md`)

Free code generation from a 2M-parameter byte model compiled only 5% of the time, because it learned format
but invented content (`getEncoder` → `okeneqIdend`). Deminifying is a constrained translation instead: at each
step the decoder may emit only whitespace or the next input byte, so content is preserved by construction and
the model only decides where whitespace goes. Extra constraints found on real files: never split a
multi-character operator (`++`, `>>>`, `&&`), never insert a space inside an identifier, copy strings and
comments byte-exact. The training data is free: minify any readable JS.

### 3. HDC / byte brain (`bin/iara_brain_v5.rs`, built as `iara_brain_full`)

An always-on HTTP service that answers from hyperdimensional (HDC) byte encodings of cached dialogs and a
knowledge graph, routes by confidence, and falls back to a Qwen model served by the sibling inference engine.
Answers it learns from the teacher are appended to a write-ahead log and checkpointed. A guard drops teacher
answers with leaked reasoning or broken escapes before they are stored (these had poisoned the cache).

### 4. Experiments (`bin/exp01…exp18`, `results/`)

Numbered experiments on byte encodings, arithmetic, FSMs, BPE comparison, tropical/Hopfield/STDP/cellular
variants, robustness, and the SMG (Sparse Mixture of Graphs) model distilled from Qwen2.5-1.5B. Each has a
result file in `results/`.

---

## Results

| Experiment | Result | Notes |
|---|---|---|
| Deminify on real files | 10/10 valid JS, content identical ignoring whitespace | details in `DEMINIFY.md` |
| Deminify similarity to the original formatting | 0.918 → 0.926 over 4 training rounds; held-out 3.56 → 1.08 bits/byte | copy baseline 0.863; the remaining gap is style (blank lines, brace placement) |
| SSM vs transformer, same corpus | held-out 2.92 vs 3.71 bits/byte with 0.23M params (3.6× fewer); scaled SSM 1.938 | SSM throughput 106–113k tokens/s at seq 256, about 2× the batched transformer |
| Scratchpad VM (`src/bin/byte_vm.rs`) for text reversal | 0 → 100% | an explicit scratchpad removes the window limit |
| HDC brain vs Qwen3-4B on 20 factual questions | 95% total (50% without calling the teacher) vs Qwen3-4B 100% | keyword-scored, 20 questions |

The SSM comparison is not compute-matched. An earlier, smaller corpus had 56% duplicated lines that inflated
held-out numbers by about 0.5 bit; the SSM runs above used a different 18 MB corpus. A later controlled test in a sibling project, with
parameters and BPTT matched, did not find an advantage for liquid cells, so treat the SSM gap as specific to
this setup.

---

## What didn't work (and why)

- **SMG distilled from Qwen generalizes poorly.** The SMG-tiny MLP reproduces all 100 training prompts byte-exact
  (`results/exp16_smg_distill.md`), but on 20 unseen prompts both the MLP and the SMG grid get 0/20 exact (byte
  agreement 38.5% vs 33.9%, `results/exp17b_smg_grid_v2.md`). The grid structure is neutral; it memorizes.
- **Associative recall by attention at 2M parameters.** 0/100 in 3 configurations (no mask, ALiBi 0.5, ALiBi 0);
  the model stays at the random floor. Retrieval has to be an explicit operation, not learned attention at
  this size.
- **Fixes on the Q/K side for the training plateau.** A 16×/64× Q/K learning-rate multiplier, QK-norm and ALiBi
  had no effect (QK-norm gave a bit-identical trajectory). The plateau was the optimizer: gradient clipping plus
  a higher learning rate broke it.
- **Free code generation at 2M parameters.** 5% of samples compile. This is what led to the deminify reframe.
- **Deminify does not recover mangled names.** Constrained decoding only places whitespace; turning `a, b, c`
  back into real names needs semantics the model does not have.

---

## Hardware and stack

- AMD Radeon RX 6750 XT, 12 GB GDDR6, Navi22 `gfx1031`, run as `gfx1030` via `HSA_OVERRIDE_GFX_VERSION=10.3.0`
  (RDNA2, wave32).
- ROCm 7.2.1 for the `gpu-aql` feature; the default build uses `wgpu` and does not need ROCm.
- Rust 2024 edition, `rust-version = 1.85`. Dependencies: `wgpu`, `bytemuck`, `pollster`, `serde_json`, `rayon`,
  `num_cpus`.

---

## Build and run

The manifest has path dependencies on sibling checkouts (`../llm-lab/hip-ffi` for the `gpu-aql` feature).
Clone them next to this repository, or remove those entries, before building.

```bash
cargo build --release --bin deminify_file --bin serve_deminify

# deminify one file
./target/release/deminify_file <model.txm> input.min.js [ws_bias]

# HTTP service + web page (web/deminify.html); default port 3032
./target/release/serve_deminify <model.txm> 3032 [ws_bias]

# train a deminify model (dataset from scripts/build_deminify_dataset.py)
./target/release/train_deminify_gpu <corpus> <budget> <d> <L> <seq> <lr> <ff> <save> [load]
```

HIP/HSA binaries need the feature flag and ROCm:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0
cargo build --release --features gpu-aql --bin train_smg_hsa
```

The HDC brain (`iara_brain_full`) expects a Qwen model served by the sibling inference engine on port 3000 and
listens on port 3012:

```bash
cargo build --release --bin iara_brain_full
./scripts/brain_forever.sh serve
curl -s -X POST http://127.0.0.1:3012/chat \
  -H 'Content-Type: application/json; charset=utf-8' \
  -d '{"prompt":"qual a capital do Brasil"}'
```

Main endpoints: `POST /chat`, `GET /stats`, `GET /moe`, `GET /domains`, `GET|POST /checkpoint`, `POST /learn`.

---

## Repository layout

```
src/gpu/          wgpu transformer, SSM, device tensors, GEMM, NN ops, WGSL shaders
src/gpu_aql/      HSA AQL dispatch for the HIP kernels (feature gpu-aql)
src/bin/          deminify, SSM/transformer training, serving, evaluation tools
bin/              brain service (iara_brain_v5.rs), numbered experiments exp01..exp18, HDC chat tools
kernels/          HIP kernels for the SMG model
results/          one result file per numbered experiment
docs/             SMG design, ROCm/HSA notes, plans (docs/plans/INDEX.md)
scripts/          dataset builders, distillation, evaluation and supervisor scripts
web/              deminify and playground pages
```

Version history: [`CHANGELOG.md`](CHANGELOG.md).

## Status

Research code. Deminify (whitespace level) works on real files; renaming mangled identifiers is not solved.
The brain service runs but depends on the external teacher for about half of new questions. Most longer design
notes are in Portuguese.

License: TBD.
