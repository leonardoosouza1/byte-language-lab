#!/usr/bin/env python3
"""Extract TENSION + ENTROPY signals do Qwen.
- Tension per byte = norm of diff vector (input → late)
- Entropy per byte = entropy of embedding dim distribution
- Activity per byte = sum of |late activation|
- Velocity per layer = pace of byte movement
Save per-byte signal arrays para brain usar como specialists.
"""
import struct, json, numpy as np, os
MODEL = '/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/model.safetensors'

with open(MODEL, 'rb') as f:
    hsize = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(hsize))
    DOFF = 8 + hsize

def load_t(name):
    info = header[name]; off = info['data_offsets']; shape = info['shape']
    with open(MODEL, 'rb') as f:
        f.seek(DOFF + off[0])
        raw = f.read(off[1] - off[0])
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)

# Load input + diff existentes
def load_emb(path):
    with open(path, 'rb') as f:
        n, d = struct.unpack('<II', f.read(8))
        return np.frombuffer(f.read(n*d*4), dtype=np.float32).reshape(n, d)

input_emb = load_emb('data/byte_semantics_qwen.bin')
late_emb = load_emb('data/byte_semantics_qwen_late.bin')
diff_emb = load_emb('data/byte_semantics_qwen_diff.bin')

# ── TENSION = norm(diff) per byte
tension = np.linalg.norm(diff_emb, axis=1)
tension /= tension.max() + 1e-9  # normalize [0,1]
print(f"▸ Tension per byte: range [{tension.min():.3f}, {tension.max():.3f}]")
print(f"  Top 5 tense bytes: {sorted(range(256), key=lambda b: -tension[b])[:5]}")
print(f"  Top 5 chill bytes: {sorted(range(256), key=lambda b: tension[b])[:5]}")

# ── ENTROPY per byte: shannon entropy of |embedding_input| distribution
import math
def entropy_of_vec(v):
    p = np.abs(v) + 1e-9
    p = p / p.sum()
    return -(p * np.log2(p)).sum()
entropy = np.array([entropy_of_vec(input_emb[b]) for b in range(256)])
entropy /= entropy.max() + 1e-9
print(f"\n▸ Entropy per byte: range [{entropy.min():.3f}, {entropy.max():.3f}]")
hi_ent = sorted(range(256), key=lambda b: -entropy[b])[:5]
print(f"  Top 5 high-entropy bytes: {hi_ent}")
print(f"    {[chr(b) if 32<=b<=126 else f'0x{b:02x}' for b in hi_ent]}")

# ── ACTIVITY per byte = norm(late_emb)
activity = np.linalg.norm(late_emb, axis=1)
activity /= activity.max() + 1e-9

# Save consolidated signal vector [256, 3]: tension, entropy, activity
out = np.column_stack([tension, entropy, activity]).astype(np.float32)
out_path = 'data/byte_tension_entropy_qwen.bin'
with open(out_path, 'wb') as f:
    f.write(struct.pack('<II', 256, 3))
    f.write(out.tobytes())
print(f"\n  ✓ Saved {out_path} ({os.path.getsize(out_path)} bytes)")

# Top correlated: tension vs entropy
import scipy.stats
r, _ = scipy.stats.pearsonr(tension, entropy)
print(f"\n  Pearson(tension, entropy): {r:+.3f}")
r2, _ = scipy.stats.pearsonr(tension, activity)
print(f"  Pearson(tension, activity): {r2:+.3f}")

# ── Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
ax1.bar(range(256), tension, color='red', alpha=0.7)
ax1.set_ylabel('Tension')
ax1.set_title('Qwen: Tension per byte (norm of input→late diff)')
ax2.bar(range(256), entropy, color='purple', alpha=0.7)
ax2.set_ylabel('Entropy')
ax2.set_title('Entropy per byte (Shannon of embed |x|)')
ax3.bar(range(256), activity, color='green', alpha=0.7)
ax3.set_ylabel('Activity')
ax3.set_title('Activity per byte (norm of late embedding)')
ax3.set_xlabel('Byte (0-255)')
plt.tight_layout()
plt.savefig('data/viz/tension_entropy_activity.png', dpi=110)
plt.close()
print(f"  ✓ Saved data/viz/tension_entropy_activity.png")
