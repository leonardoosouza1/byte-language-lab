#!/usr/bin/env python3
"""DEAD NEURONS — identify neurônios mortos no Qwen sem rodar forward.
Analise direto das matrizes:
  - MLP intermediate: 8960 neurônios → quantos têm output ~0?
  - Q/K/V heads: 12 heads → algum head 'inativo'?
  - Per-layer redundância
Objetivo: descobrir Qwen Lite só com neurônios vivos.
"""
import struct, json, numpy as np, os, sys

MODEL = '/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/model.safetensors'

with open(MODEL, 'rb') as f:
    hsize = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(hsize))
    DOFF = 8 + hsize

def load_t(name):
    info = header[name]; off = info['data_offsets']; shape = info['shape']
    with open(MODEL, 'rb') as f:
        f.seek(DOFF + off[0]); raw = f.read(off[1] - off[0])
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape).copy()

print("═══════════════════════════════════════════════════════════")
print("  Dead Neuron Analysis — Qwen 28 layers")
print("═══════════════════════════════════════════════════════════\n")

# Per MLP layer: detect dead intermediate neurons
# gate_proj: [INTER=8960, HIDDEN=1536] - cada linha é um neurônio intermediate
# up_proj:   [INTER=8960, HIDDEN=1536] - mesma estrutura
# down_proj: [HIDDEN, INTER=8960] - cada COLUNA é peso do neurônio

# Neurônio "morto" se:
#   gate row norm ~ 0  → nunca ativa
#   up row norm ~ 0    → nada amplifica
#   down col norm ~ 0  → output ignorado

stats_per_layer = []
print("▸ Per-layer MLP neuron health (8960 neurônios cada)")
print("  Layer | dead | weak | normal | strong | ratio_dead")
print("  ──────┼──────┼──────┼────────┼────────┼───────────")

all_layers_dead_mask = []  # per layer: boolean array [8960]
for L in range(28):
    gate = load_t(f'model.layers.{L}.mlp.gate_proj.weight')  # [8960, 1536]
    up = load_t(f'model.layers.{L}.mlp.up_proj.weight')
    down = load_t(f'model.layers.{L}.mlp.down_proj.weight')  # [1536, 8960]

    # Norm per intermediate neuron
    gate_norm = np.linalg.norm(gate, axis=1)  # [8960]
    up_norm = np.linalg.norm(up, axis=1)
    down_norm = np.linalg.norm(down, axis=0)  # column norm

    # Combined: neurônio precisa estar vivo em todos 3
    combined = gate_norm * up_norm * down_norm  # [8960]

    # Histogram
    pmax = combined.max() + 1e-9
    norm_combined = combined / pmax
    dead = (norm_combined < 0.001).sum()
    weak = ((norm_combined >= 0.001) & (norm_combined < 0.01)).sum()
    normal = ((norm_combined >= 0.01) & (norm_combined < 0.5)).sum()
    strong = (norm_combined >= 0.5).sum()

    stats_per_layer.append((L, dead, weak, normal, strong, combined))
    all_layers_dead_mask.append(norm_combined < 0.001)

    ratio = dead / 8960 * 100
    bar = '█' * int(ratio / 5)
    print(f"  {L:>5} | {dead:>4} | {weak:>4} | {normal:>6} | {strong:>6} | {ratio:>5.1f}% {bar}")

# Agg
total_dead = sum(s[1] for s in stats_per_layer)
total_neurons = 28 * 8960
print(f"\n  Total dead neurons across 28 layers: {total_dead}/{total_neurons} ({total_dead/total_neurons*100:.1f}%)")

# Layers com mais dead neurons
sorted_by_dead = sorted(stats_per_layer, key=lambda x: -x[1])
print(f"\n  Top 5 layers com MAIS dead neurons:")
for L, d, w, n, s, _ in sorted_by_dead[:5]:
    print(f"    L{L}: {d} dead ({d/8960*100:.1f}%)")

print(f"\n  Top 5 layers com MENOS dead neurons (mais densas):")
for L, d, w, n, s, _ in sorted_by_dead[-5:]:
    print(f"    L{L}: {d} dead ({d/8960*100:.1f}%)")

# Save masks
all_masks = np.array(all_layers_dead_mask, dtype=bool)  # [28, 8960]
np.save('data/qwen_dead_mask.npy', all_masks)
print(f"\n  ✓ Saved data/qwen_dead_mask.npy")

# Theoretical compression
alive_neurons = (~all_masks).sum(axis=1)  # per layer
print(f"\n  Alive neurons per layer: min={alive_neurons.min()}, max={alive_neurons.max()}")
print(f"  Mean alive per layer: {alive_neurons.mean():.0f}")

# Reduced MLP cost
orig_mlp_params = 28 * 8960 * 1536 * 3  # gate + up + down per layer
reduced_params = sum(int(a) for a in alive_neurons) * 1536 * 3
print(f"\n  Original MLP params: {orig_mlp_params/1e6:.0f}M")
print(f"  Reduced MLP params:  {reduced_params/1e6:.0f}M")
print(f"  Compression:         {(1 - reduced_params/orig_mlp_params)*100:.1f}%")

# ─────────────────────────────────────────────────────────────────
# HEAD-LEVEL analysis
# ─────────────────────────────────────────────────────────────────
print("\n▸ Attention heads — dead head detection")
print("  Layer | head | norm_q | norm_k | norm_v | combined | dead?")
print("  ──────┼──────┼────────┼────────┼────────┼──────────┼──────")

dead_heads = []
for L in [0, 7, 14, 21, 27]:
    q = load_t(f'model.layers.{L}.self_attn.q_proj.weight')  # [1536, 1536]
    k = load_t(f'model.layers.{L}.self_attn.k_proj.weight')  # [256, 1536] — GQA
    v = load_t(f'model.layers.{L}.self_attn.v_proj.weight')  # [256, 1536]

    # Q has 12 heads × 128. Per head row block
    for h in range(12):
        q_head = q[h*128:(h+1)*128, :]
        nq = np.linalg.norm(q_head)
        # K/V have only 2 kv_heads — group by 6 q_heads per kv_head
        kv_idx = h // 6
        k_head = k[kv_idx*128:(kv_idx+1)*128, :]
        v_head = v[kv_idx*128:(kv_idx+1)*128, :]
        nk = np.linalg.norm(k_head); nv = np.linalg.norm(v_head)
        combined = nq * nk * nv
        is_dead = nq < 1.0 or combined < 100
        if is_dead:
            dead_heads.append((L, h))
        print(f"  {L:>5} | {h:>4} | {nq:6.2f} | {nk:6.2f} | {nv:6.2f} | {combined:8.1f} | {'⚠️' if is_dead else ''}")

print(f"\n  Total dead heads detectadas (sample 5 layers): {len(dead_heads)}")

# ─────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Dead mask heatmap (layer × neuron)
ax1.imshow(all_masks, cmap='Reds', aspect='auto', interpolation='nearest')
ax1.set_xlabel('Neuron (0-8960)')
ax1.set_ylabel('Layer')
ax1.set_title('Dead neurons per layer (red = dead)')

# Alive neurons per layer
ax2.bar(range(28), alive_neurons, color='steelblue')
ax2.axhline(8960, color='gray', linestyle='--', label='Total (8960)')
ax2.set_xlabel('Layer'); ax2.set_ylabel('Alive neurons')
ax2.set_title('Alive intermediate neurons per layer')
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/viz/qwen_dead_neurons.png', dpi=110)
plt.close()
print(f"\n  ✓ Plot: data/viz/qwen_dead_neurons.png")

print("\n" + "═" * 60)
print("  PROPOSTAS DE QWEN LITE")
print("═" * 60)
print(f"""
  Original Qwen 1.5B = 3.09 GB

  IDEIA 1 — Prune dead neurons (≥0.1% dead avg):
    • Reduce MLP by {total_dead/total_neurons*100:.1f}% ({total_dead} neurônios)
    • New size: {3.09 * (1 - 0.6 * total_dead/total_neurons):.2f} GB (60% size from MLP)

  IDEIA 2 — Keep only TOP-50% intermediate neurons (mais ativos):
    • 4480 neurônios por layer (de 8960)
    • Size: {3.09 * (1 - 0.6 * 0.5):.2f} GB
    • Likely lossy mas compacto

  IDEIA 3 — Variable density per layer:
    • Layers iniciais com poucos dead → manter cheias
    • Layers tardias → comprimir se densas
    • Adaptive compression per layer
""")
