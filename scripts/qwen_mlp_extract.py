#!/usr/bin/env python3
"""Extract MLP gate patterns + cross-layer drift.
- gate_proj captures SiLU activation patterns
- Multi-layer attention (early+mid+late)
- Per-layer byte representation drift
"""
import struct, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL = '/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/model.safetensors'

with open(MODEL, 'rb') as f:
    hsize = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(hsize))
    DOFF = 8 + hsize

def load_t(name):
    info = header[name]
    off = info['data_offsets']
    shape = info['shape']
    with open(MODEL, 'rb') as f:
        f.seek(DOFF + off[0])
        raw = f.read(off[1] - off[0])
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)

# Load input embedding
print("▸ Loading input byte embedding")
emb = load_t('model.embed_tokens.weight')

with open('/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/tokenizer.json') as f:
    tok = json.load(f)
id_to_str = {v: k for k, v in tok['model']['vocab'].items()}

byte_input = np.zeros((256, 1536), dtype=np.float32)
for tok_id in range(emb.shape[0]):
    s = id_to_str.get(tok_id, '')
    if not s or (s.startswith('<') and s.endswith('>')): continue
    try:
        b = s.encode('utf-8')[0]
        byte_input[b] += emb[tok_id]
    except: pass

# ─────────────────────────────────────────────────────────────────
# MULTI-LAYER ATTENTION extraction (early/mid/late)
# ─────────────────────────────────────────────────────────────────
print("\n▸ Multi-layer attention (early L=0, mid L=14, late L=27)")
def attn_for_layer(L):
    q = load_t(f'model.layers.{L}.self_attn.q_proj.weight')
    k = load_t(f'model.layers.{L}.self_attn.k_proj.weight')
    bq = byte_input @ q.T  # [256, 1536]
    bk = byte_input @ k.T  # [256, 256]
    # Compute Q·K^T using first 256 dims of bq
    a = bq[:, :256] @ bk.T  # [256, 256]
    a -= a.max(axis=1, keepdims=True)
    a = np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)
    return a.astype(np.float32)

attn_layers = {}
for L in [0, 7, 14, 21, 27]:
    print(f"  Layer {L}...")
    attn_layers[L] = attn_for_layer(L)

# Save concat as multi-layer attention (5 maps stacked = 5 × 256 × 256)
out_multi = 'data/byte_attention_qwen_multi.bin'
with open(out_multi, 'wb') as f:
    f.write(struct.pack('<III', 5, 256, 256))  # n_layers, h, w
    for L in [0, 7, 14, 21, 27]:
        f.write(attn_layers[L].tobytes())
print(f"  ✓ Saved multi-layer attention {out_multi} ({os.path.getsize(out_multi)/1024:.0f} KB)")

# Plot 5 attention maps side by side
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
from matplotlib.colors import LogNorm
for ax, L in zip(axes, [0, 7, 14, 21, 27]):
    a = attn_layers[L]
    im = ax.imshow(a, cmap='hot', aspect='auto', norm=LogNorm(vmin=1e-5, vmax=a.max()))
    ax.set_title(f'Layer {L}')
fig.suptitle('Qwen attention map evolution across depth')
fig.tight_layout()
fig.savefig('data/viz/attention_multilayer.png', dpi=100)
plt.close()
print("  ✓ Saved data/viz/attention_multilayer.png")

# ─────────────────────────────────────────────────────────────────
# MLP gate_proj patterns
# ─────────────────────────────────────────────────────────────────
print("\n▸ MLP gate_proj patterns (SiLU activation signatures)")
gate_signatures = {}
for L in [0, 7, 14, 21, 27]:
    g = load_t(f'model.layers.{L}.mlp.gate_proj.weight')  # [8960, 1536]
    # Project byte_input → gate space
    byte_gate = byte_input @ g.T  # [256, 8960]
    # SiLU(x) = x * sigmoid(x)
    silu = byte_gate / (1 + np.exp(-byte_gate))
    # Sample stats: mean activation per byte
    gate_signatures[L] = silu.mean(axis=1)  # [256]
    print(f"  Layer {L}: gate mean per byte range=[{silu.mean(axis=1).min():.4f}, {silu.mean(axis=1).max():.4f}]")

# Plot — gate activation signature por layer
fig, ax = plt.subplots(figsize=(14, 5))
for L, sig in gate_signatures.items():
    ax.plot(sig, label=f'Layer {L}', alpha=0.7)
ax.set_xlabel('Byte (0-255)'); ax.set_ylabel('Mean SiLU gate activation')
ax.set_title('Qwen MLP gate_proj — byte-wise activation signature per layer')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('data/viz/mlp_gate_signature.png', dpi=100)
plt.close()
print("  ✓ Saved data/viz/mlp_gate_signature.png")

# Save layer-21 gate as new byte-level feature
gate_21 = gate_signatures[21]
out_gate = 'data/byte_gate_qwen.bin'
with open(out_gate, 'wb') as f:
    f.write(struct.pack('<I', 256))
    f.write(gate_21.astype(np.float32).tobytes())
print(f"  ✓ Saved {out_gate}")

# ─────────────────────────────────────────────────────────────────
# Cross-layer byte drift visualization
# ─────────────────────────────────────────────────────────────────
print("\n▸ Cross-layer byte trajectory")
# For 'a', track its representation through layers via v_proj
trajectory = {}
for byte in [ord('a'), ord('A'), ord('0'), ord(' '), ord('.')]:
    points = []
    for L in [0, 7, 14, 21, 27]:
        v = load_t(f'model.layers.{L}.self_attn.v_proj.weight')  # [256, 1536]
        proj = byte_input[byte] @ v.T  # [256]
        # PCA to 2 dims using SVD of v
        U, S, Vt = np.linalg.svd(v, full_matrices=False)
        proj_2d = byte_input[byte] @ Vt[:2].T  # [2]
        points.append(proj_2d)
    trajectory[byte] = np.array(points)

fig, ax = plt.subplots(figsize=(10, 8))
for byte, traj in trajectory.items():
    char = chr(byte) if 32 <= byte <= 126 else f'{byte:02x}'
    ax.plot(traj[:, 0], traj[:, 1], marker='o', label=f"'{char}'", linewidth=2, markersize=10)
    for i, L in enumerate([0, 7, 14, 21, 27]):
        ax.annotate(f'L{L}', traj[i], fontsize=8)
ax.set_title('Byte representation TRAJECTORY across layers (V proj 2D)')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('data/viz/byte_trajectory.png', dpi=110)
plt.close()
print("  ✓ Saved data/viz/byte_trajectory.png")

print("\n═══════════════════════════════════════════════════════════")
print("  Sumário extracts novos")
print("═══════════════════════════════════════════════════════════")
print(f"  • Multi-layer attention 5 layers: {os.path.getsize(out_multi)/1024:.0f} KB")
print(f"  • Gate proj signature L21: {os.path.getsize(out_gate)} bytes")
print(f"\n  Plots novos:")
for p in ['attention_multilayer.png', 'mlp_gate_signature.png', 'byte_trajectory.png']:
    if os.path.exists(f'data/viz/{p}'):
        print(f"    data/viz/{p}: {os.path.getsize(f'data/viz/{p}')/1024:.0f} KB")
