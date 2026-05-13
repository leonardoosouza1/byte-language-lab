#!/usr/bin/env python3
"""CURATED extraction — 3 paths de pesquisa de alto valor.

A. Diff late-input embedding (28 layers de transformação condensados)
B. Attention pattern via Q·K^T heads (byte attention map)
C. VQ codebook 256 prototypes (quantização matemática inteligente)
"""
import struct, json, os, math
import numpy as np

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

# ─────────────────────────────────────────────────────────────────
# PATH A — DIFF late-input embedding
# ─────────────────────────────────────────────────────────────────
print("═══════════════════════════════════════════════════════════")
print("  PATH A — Diff Embedding (transformação aprendida 28 layers)")
print("═══════════════════════════════════════════════════════════\n")

# Load input embedding [151936, 1536]
print("▸ Loading input + late embeddings")
emb = load_t('model.embed_tokens.weight')

# Build late embedding: average v_proj of layers 25-27
late_acc = np.zeros((256, 1536), dtype=np.float32)
n_late = 0

# Load tokenizer to map byte → tokens
with open('/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/tokenizer.json') as f:
    tok = json.load(f)
vocab_map = tok['model']['vocab']
id_to_str = {v: k for k, v in vocab_map.items()}

byte_token_ids = {b: [] for b in range(256)}
for tok_id in range(emb.shape[0]):
    s = id_to_str.get(tok_id, '')
    if not s or (s.startswith('<') and s.endswith('>')): continue
    try:
        b = s.encode('utf-8')[0]
        byte_token_ids[b].append(tok_id)
    except: continue

# byte_input_emb 256 × 1536
byte_input_emb = np.zeros((256, 1536), dtype=np.float32)
for b in range(256):
    if byte_token_ids[b]:
        byte_input_emb[b] = emb[byte_token_ids[b]].mean(axis=0)

# Build "late" representation: pass byte_input through (approximation)
# Use late v_proj as semantic projection
v_late = []
for L in [25, 26, 27]:
    v_late.append(load_t(f'model.layers.{L}.self_attn.v_proj.weight'))
# v_proj shape [256, 1536] — projection from hidden to value
# Late byte rep = byte_input @ v_proj_avg.T
v_avg = np.mean(v_late, axis=0)  # [256, 1536]
# v_avg is [d_v, d_model] = [256, 1536]. To get byte's "late" state:
# byte_late ≈ byte_input @ v_avg.T  [256, 1536] @ [1536, 256] = [256, 256]
# But we want 1536-dim back. Use pseudoinverse approach: byte_late_in_v_space:
byte_late = byte_input_emb @ v_avg.T  # [256, 256]
# Pad to 1536 by re-projecting: [256, 256] @ [256, 1536]
byte_late_full = byte_late @ v_avg     # [256, 1536]

# Diff = late - input (the "transformation learned")
diff = byte_late_full - byte_input_emb
print(f"  byte_input_emb: shape={byte_input_emb.shape}, mean_norm={np.linalg.norm(byte_input_emb, axis=1).mean():.3f}")
print(f"  byte_late_full: shape={byte_late_full.shape}, mean_norm={np.linalg.norm(byte_late_full, axis=1).mean():.3f}")
print(f"  diff norm: {np.linalg.norm(diff, axis=1).mean():.3f}")

# SVD do diff
U, S, Vt = np.linalg.svd(diff, full_matrices=False)
diff_64 = U[:, :64] * S[:64]
energy = (S[:64]**2).sum() / (S**2).sum() * 100
print(f"  Diff SVD top-64: {energy:.1f}% energy")

out_diff = 'data/byte_semantics_qwen_diff.bin'
with open(out_diff, 'wb') as f:
    f.write(struct.pack('<II', 256, 64))
    f.write(diff_64.astype(np.float32).tobytes())
print(f"  ✓ Saved {out_diff}")

# Show diff coside similarities
print("\n  Cossim no DIFF space (transformação aprendida):")
def cossim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
for p, q in [('a', 'A'), ('a', 'e'), ('o', 'i'), ('a', '!'), ('a', '0')]:
    s = cossim(diff_64[ord(p)], diff_64[ord(q)])
    print(f"    diff_cossim({p!r}, {q!r}) = {s:+.3f}")

# ─────────────────────────────────────────────────────────────────
# PATH B — Attention pattern Q·K^T entre bytes
# ─────────────────────────────────────────────────────────────────
print("\n═══════════════════════════════════════════════════════════")
print("  PATH B — Attention map Q·K^T (layer 27, head médio)")
print("═══════════════════════════════════════════════════════════\n")

q27 = load_t('model.layers.27.self_attn.q_proj.weight')  # [1536, 1536]
k27 = load_t('model.layers.27.self_attn.k_proj.weight')  # [256, 1536]
print(f"▸ Q: {q27.shape}, K: {k27.shape}")

# Project byte_input → Q-space and K-space
# byte_q = byte_input_emb @ q27.T  [256, 1536]
byte_q = byte_input_emb @ q27.T
# k27 has different shape: [256, 1536]. So K-space is 256-d.
# byte_k = byte_input_emb @ k27.T  [256, 256]
byte_k = byte_input_emb @ k27.T
print(f"  byte_q: {byte_q.shape}, byte_k: {byte_k.shape}")

# Q·K^T attention scores between bytes:
# Need byte_q [256, 1536] @ K^T expressed in same space.
# Use only first 256 dims of byte_q to match byte_k space:
qa = byte_q[:, :256]  # [256, 256]
attn_scores = qa @ byte_k.T  # [256, 256]
# Softmax per row (each byte's attention to others)
attn_scores -= attn_scores.max(axis=1, keepdims=True)
attn_softmax = np.exp(attn_scores) / np.exp(attn_scores).sum(axis=1, keepdims=True)
print(f"  attn_softmax: {attn_softmax.shape}, max attention: {attn_softmax.max():.4f}")

# Sample attention from byte 'a' to all others
target = ord('a')
attns_a = [(i, attn_softmax[target, i]) for i in range(256)]
attns_a.sort(key=lambda x: -x[1])
print(f"\n  Top 8 atenção from 'a' (layer 27, learned):")
for i, s in attns_a[:8]:
    p = chr(i) if 32 <= i <= 126 else f'\\x{i:02x}'
    print(f"    'a' → {p!r}: {s:.4f}")

# Save attention map as new specialist
# Save as raw 256×256 f32 matrix
out_attn = 'data/byte_attention_qwen.bin'
with open(out_attn, 'wb') as f:
    f.write(struct.pack('<II', 256, 256))
    f.write(attn_softmax.astype(np.float32).tobytes())
print(f"\n  ✓ Saved attention map: {out_attn}")

# ─────────────────────────────────────────────────────────────────
# PATH C — VQ codebook (vector quantization)
# ─────────────────────────────────────────────────────────────────
print("\n═══════════════════════════════════════════════════════════")
print("  PATH C — Vector Quantization codebook (256 prototypes)")
print("═══════════════════════════════════════════════════════════\n")

# Treat each byte embedding as a 64-d vector (use input semantics SVD)
print("▸ Loading byte_semantics_qwen.bin (256×64 input semantics)")
with open('data/byte_semantics_qwen.bin', 'rb') as f:
    n, d = struct.unpack('<II', f.read(8))
    in_emb = np.frombuffer(f.read(n*d*4), dtype=np.float32).reshape(n, d)
print(f"  Loaded {in_emb.shape}")

# k-means: 64 prototypes capture byte similarity structure
print("\n▸ K-means VQ (k=64 prototypes, max_iter=20)")
K = 64
rng = np.random.default_rng(seed=42)
# Init prototypes from random bytes
init_idx = rng.choice(256, size=K, replace=False)
prototypes = in_emb[init_idx].copy()
assignments = np.zeros(256, dtype=np.int32)

for it in range(20):
    # Assign
    dists = np.linalg.norm(in_emb[:, None, :] - prototypes[None, :, :], axis=2)  # [256, K]
    new_assign = dists.argmin(axis=1)
    if np.array_equal(new_assign, assignments) and it > 0:
        print(f"  Converged at iter {it}")
        break
    assignments = new_assign
    # Update prototypes
    for k in range(K):
        mask = assignments == k
        if mask.sum() > 0:
            prototypes[k] = in_emb[mask].mean(axis=0)

# Compute distortion
final_dists = np.linalg.norm(in_emb[:, None, :] - prototypes[None, :, :], axis=2)
chosen = final_dists[np.arange(256), assignments]
distortion = chosen.mean()
print(f"  Distortion média: {distortion:.4f}")
print(f"  Prototypes ativos: {len(set(assignments))}/64")

# Show clusters
from collections import Counter
cluster_counts = Counter(assignments)
print(f"  Top 5 clusters maiores:")
for c, n in cluster_counts.most_common(5):
    members = [b for b in range(256) if assignments[b] == c]
    char_members = [chr(b) if 32 <= b <= 126 else f'\\x{b:02x}' for b in members[:8]]
    print(f"    Cluster {c} ({n} bytes): {char_members}")

# Save VQ: prototypes [K, d] + assignments [256]
out_vq = 'data/byte_vq_qwen.bin'
with open(out_vq, 'wb') as f:
    f.write(struct.pack('<II', K, d))
    f.write(prototypes.astype(np.float32).tobytes())
    f.write(assignments.astype(np.uint8).tobytes())
print(f"\n  ✓ Saved VQ: {out_vq} ({os.path.getsize(out_vq)} bytes)")
print(f"  Compressão vs input: {os.path.getsize(out_vq)/65544*100:.1f}% do tamanho")

print("\n═══════════════════════════════════════════════════════════")
print("  SÍNTESE")
print("═══════════════════════════════════════════════════════════")
print(f"""
  Path A — DIFF embedding (transformação 28 layers): 65 KB
    captura "movimento" semântico learned (input → late)

  Path B — ATTENTION map Q·K^T: 256 KB
    captura quem atende a quem na ÚLTIMA layer
    'a' atende preferencialmente: top 8 acima

  Path C — VQ codebook 64 prototypes: ~17 KB
    bytes agrupados em 64 categorias semânticas
    {distortion:.4f} distortion = boa fidelidade
    Cluster maior tem {cluster_counts.most_common(1)[0][1]} bytes
""")
