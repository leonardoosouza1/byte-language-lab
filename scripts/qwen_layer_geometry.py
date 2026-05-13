#!/usr/bin/env python3
"""LAYER GEOMETRY — descobrir grafo das 28 layers do Qwen.
Hipótese user: ao invés de cascata 28 sequencial, descobrir caminhos alternativos.
Análise:
  1. Forward prompt → captura hidden state ANTES e DEPOIS de cada layer
  2. delta_L = ||h_L - h_{L-1}|| → quanto layer L muda
  3. similarity matrix [28, 28] de layer outputs
  4. Identificar layers redundantes / clusters
  5. Propor rotas alternativas (skip layers, reorder)
"""
import struct, json, numpy as np, os, sys

MODEL = '/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/model.safetensors'
TOKENIZER = '/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/tokenizer.json'

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

HIDDEN = 1536; N_HEAD = 12; N_KV_HEAD = 2; HEAD_DIM = 128
ROPE_THETA = 1_000_000.0; EPS = 1e-6

def rmsnorm(x, w):
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + EPS)
    return (x / rms) * w

def rope(x, pos):
    half = HEAD_DIM // 2
    freqs = 1.0 / (ROPE_THETA ** (np.arange(0, half, dtype=np.float32) * 2.0 / HEAD_DIM))
    t = np.arange(pos, pos + x.shape[0], dtype=np.float32)
    angles = np.outer(t, freqs)
    cos = np.cos(angles)[:, None, :]; sin = np.sin(angles)[:, None, :]
    return np.concatenate([x[..., :half] * cos - x[..., half:] * sin,
                            x[..., :half] * sin + x[..., half:] * cos], axis=-1)

def silu(x):
    return x / (1.0 + np.exp(-x))

print("▸ Loading embed + final_norm")
embed = load_t('model.embed_tokens.weight')
final_norm = load_t('model.norm.weight')

def forward_layer(h, ln, layer):
    """Single transformer layer forward."""
    seq = h.shape[0]
    h_in = rmsnorm(h, ln['in_ln'])
    q = h_in @ ln['q'].T + ln['q_b']
    k = h_in @ ln['k'].T + ln['k_b']
    v = h_in @ ln['v'].T + ln['v_b']
    q = q.reshape(seq, N_HEAD, HEAD_DIM); k = k.reshape(seq, N_KV_HEAD, HEAD_DIM); v = v.reshape(seq, N_KV_HEAD, HEAD_DIM)
    q = rope(q, 0); k = rope(k, 0)
    rf = N_HEAD // N_KV_HEAD
    k = np.repeat(k, rf, axis=1); v = np.repeat(v, rf, axis=1)
    q_t = q.transpose(1, 0, 2); k_t = k.transpose(1, 0, 2); v_t = v.transpose(1, 0, 2)
    scores = q_t @ k_t.transpose(0, 2, 1) / np.sqrt(HEAD_DIM)
    mask = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)
    scores = scores + mask[None]
    scores = scores - scores.max(axis=-1, keepdims=True)
    scores = np.exp(scores); scores = scores / scores.sum(axis=-1, keepdims=True)
    attn_out = (scores @ v_t).transpose(1, 0, 2).reshape(seq, HIDDEN)
    h = h + attn_out @ ln['o'].T
    h_mlp_in = rmsnorm(h, ln['post_ln'])
    h = h + (silu(h_mlp_in @ ln['gate'].T) * (h_mlp_in @ ln['up'].T)) @ ln['down'].T
    return h

def load_layer(L):
    return {
        'in_ln': load_t(f'model.layers.{L}.input_layernorm.weight'),
        'post_ln': load_t(f'model.layers.{L}.post_attention_layernorm.weight'),
        'q': load_t(f'model.layers.{L}.self_attn.q_proj.weight'),
        'k': load_t(f'model.layers.{L}.self_attn.k_proj.weight'),
        'v': load_t(f'model.layers.{L}.self_attn.v_proj.weight'),
        'o': load_t(f'model.layers.{L}.self_attn.o_proj.weight'),
        'q_b': load_t(f'model.layers.{L}.self_attn.q_proj.bias'),
        'k_b': load_t(f'model.layers.{L}.self_attn.k_proj.bias'),
        'v_b': load_t(f'model.layers.{L}.self_attn.v_proj.bias'),
        'gate': load_t(f'model.layers.{L}.mlp.gate_proj.weight'),
        'up': load_t(f'model.layers.{L}.mlp.up_proj.weight'),
        'down': load_t(f'model.layers.{L}.mlp.down_proj.weight'),
    }

# Use simple token sequences
test_prompts_tokens = [
    [42719, 1953, 11, 220],     # 'Olá, '
    [3838, 374, 279, 6722],     # 'What is the capital'
    [11843, 24221, 105, 100],   # random
]

print("\n▸ Loading all 28 layers (~18 GB)... this takes time")
layers = []
for L in range(28):
    sys.stdout.write(f"\r  Layer {L:2d}/28"); sys.stdout.flush()
    layers.append(load_layer(L))
print("\n  ✓ All 28 layers loaded")

# ─────────────────────────────────────────────────────────────────
# Forward + capture hidden state after each layer
# ─────────────────────────────────────────────────────────────────
print("\n▸ Forward + capture hidden states")
all_states = []  # [prompt][layer] = hidden state of last token

for pi, tokens in enumerate(test_prompts_tokens):
    print(f"  Prompt {pi}: tokens {tokens}")
    h = embed[tokens].copy()
    layer_states = [h[-1].copy()]  # initial = embedded
    for L, layer in enumerate(layers):
        h = forward_layer(h, layer, layer)  # process layer
        layer_states.append(h[-1].copy())
    all_states.append(layer_states)

# all_states[prompt][0..28] = hidden state after layer 0..27 (index 0 = pre)
all_states_np = np.array(all_states)  # [n_prompts, 29, 1536]
print(f"  Captured: {all_states_np.shape}")

# ─────────────────────────────────────────────────────────────────
# 1. Delta per layer = quanto cada layer MUDA representação
# ─────────────────────────────────────────────────────────────────
print("\n▸ Delta per layer (||h_L - h_{L-1}||)")
deltas = []
for L in range(28):
    # delta médio entre todos prompts
    d = np.mean([np.linalg.norm(all_states_np[p, L+1] - all_states_np[p, L]) for p in range(len(test_prompts_tokens))])
    deltas.append(d)

print("  Layer | delta | grau de mudança")
for L, d in enumerate(deltas):
    bar = '█' * int(d / max(deltas) * 40)
    print(f"  {L:>5} | {d:7.2f} | {bar}")

print(f"\n  Top 5 LAYERS QUE MAIS MUDAM: {sorted(range(28), key=lambda L: -deltas[L])[:5]}")
print(f"  Top 5 LAYERS QUE MENOS MUDAM (candidatas a skip): {sorted(range(28), key=lambda L: deltas[L])[:5]}")

# ─────────────────────────────────────────────────────────────────
# 2. Layer-layer similarity matrix (cossim de delta vectors)
# ─────────────────────────────────────────────────────────────────
print("\n▸ Layer-layer similarity matrix")
# Compute delta vector per layer per prompt, then average direction
delta_vecs = []
for L in range(28):
    avg_delta = np.mean([all_states_np[p, L+1] - all_states_np[p, L] for p in range(len(test_prompts_tokens))], axis=0)
    delta_vecs.append(avg_delta / (np.linalg.norm(avg_delta) + 1e-9))
delta_vecs = np.array(delta_vecs)  # [28, 1536]

sim_matrix = delta_vecs @ delta_vecs.T  # [28, 28]

# Identify highly similar layer pairs (redundancy)
print("\n  Pares de layers com cossim > 0.7 (REDUNDANTES):")
redundant_pairs = []
for i in range(28):
    for j in range(i+1, 28):
        if sim_matrix[i, j] > 0.7:
            redundant_pairs.append((i, j, sim_matrix[i, j]))
            print(f"    L{i:2d} ↔ L{j:2d}: cossim = {sim_matrix[i, j]:+.3f}")

# ─────────────────────────────────────────────────────────────────
# 3. Cluster layers by behavior
# ─────────────────────────────────────────────────────────────────
print("\n▸ Hierarchical clustering layers")
from scipy.cluster.hierarchy import linkage, fcluster
# Use 1 - cossim as distance
dist_matrix = 1 - sim_matrix
np.fill_diagonal(dist_matrix, 0)
# Convert to condensed form
from scipy.spatial.distance import squareform
condensed = squareform(dist_matrix, checks=False)
Z = linkage(condensed, method='average')

# Try different cut levels: 4, 7, 14 clusters
for k in [4, 7, 14]:
    clusters = fcluster(Z, t=k, criterion='maxclust')
    print(f"\n  {k} clusters:")
    from collections import defaultdict
    groups = defaultdict(list)
    for L, c in enumerate(clusters):
        groups[c].append(L)
    for c, members in sorted(groups.items()):
        print(f"    Cluster {c}: layers {members}")

# ─────────────────────────────────────────────────────────────────
# 4. Save sim matrix + propose routes
# ─────────────────────────────────────────────────────────────────
np.save('data/qwen_layer_sim.npy', sim_matrix)
np.save('data/qwen_layer_deltas.npy', np.array(deltas))
print(f"\n  ✓ Saved data/qwen_layer_sim.npy + qwen_layer_deltas.npy")

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
im1 = ax1.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax1.set_title('Qwen layer-layer cosine similarity (delta vectors)')
ax1.set_xlabel('Layer'); ax1.set_ylabel('Layer')
plt.colorbar(im1, ax=ax1)

ax2.bar(range(28), deltas, color='steelblue')
ax2.set_xlabel('Layer'); ax2.set_ylabel('||h_L - h_{L-1}||')
ax2.set_title('Delta per layer (impact of each layer)')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/viz/qwen_layer_geometry.png', dpi=110)
plt.close()
print(f"  ✓ Plot saved data/viz/qwen_layer_geometry.png")

# ─────────────────────────────────────────────────────────────────
# 5. PROPOSTAS DE ROTAS ALTERNATIVAS
# ─────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  PROPOSTAS DE ROTAS")
print("═" * 60)

# Route 1: Top-K layers by delta (keep most impactful)
top_layers = sorted(range(28), key=lambda L: -deltas[L])
print(f"\n  ROUTE-A: top-14 layers por delta (skip metade):")
chosen = sorted(top_layers[:14])
print(f"    Layers to keep: {chosen}")
print(f"    Compute reduction: {(28-14)/28*100:.0f}%")

# Route 2: Skip redundant pairs (keep one of each)
keep = set(range(28))
for i, j, s in redundant_pairs:
    # remove one with LOWER delta (less impactful)
    if deltas[i] < deltas[j]:
        keep.discard(i)
    else:
        keep.discard(j)
print(f"\n  ROUTE-B: drop redundant (cossim>0.7):")
print(f"    Layers to keep: {sorted(keep)}")
print(f"    Removed: {sorted(set(range(28)) - keep)}")

# Route 3: cluster-based (1 representative per cluster)
clusters_7 = fcluster(Z, t=7, criterion='maxclust')
groups_7 = defaultdict(list)
for L, c in enumerate(clusters_7):
    groups_7[c].append(L)
reps = []
for c, members in groups_7.items():
    # Pick member with highest delta
    rep = max(members, key=lambda L: deltas[L])
    reps.append(rep)
print(f"\n  ROUTE-C: 1 representative per cluster (7 layers):")
print(f"    Layers to use: {sorted(reps)}")
print(f"    Compute reduction: {(28-7)/28*100:.0f}%")
