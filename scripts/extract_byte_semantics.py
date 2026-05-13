#!/usr/bin/env python3
"""EXTRACT — Qwen token embeddings → byte semantic vectors.
1. Load full embedding matrix [151936, 1536] bfloat16
2. For each byte X (0-255): collect tokens whose first byte = X
3. Average their 1536-dim embeddings → 256 byte vectors
4. SVD reduce to 256 × 64
5. Save as data/byte_semantics_qwen.bin
This gives brain access to Qwen's LEARNED semantic geometry per byte."""
import struct, json, os, sys
import numpy as np

MODEL = '/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/model.safetensors'
TOKENIZER = '/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/tokenizer.json'

print("▸ Loading safetensors header")
with open(MODEL, 'rb') as f:
    header_size = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_size))
    data_offset = 8 + header_size

emb_info = header['model.embed_tokens.weight']
shape = emb_info['shape']
offsets = emb_info['data_offsets']
vocab, hidden = shape
print(f"  Embedding: {vocab} × {hidden} bfloat16")

print("▸ Loading embedding matrix (~467 MB)...")
with open(MODEL, 'rb') as f:
    f.seek(data_offset + offsets[0])
    raw = f.read(offsets[1] - offsets[0])
# bfloat16 → float32
# Each 2-byte little-endian bfloat16
print("  Converting bfloat16 → float32...")
u16 = np.frombuffer(raw, dtype=np.uint16)
u32 = u16.astype(np.uint32) << 16
emb = u32.view(np.float32).reshape(vocab, hidden)
print(f"  Loaded: shape={emb.shape}, range=[{emb.min():.3f}, {emb.max():.3f}]")

print("\n▸ Loading tokenizer")
with open(TOKENIZER) as f:
    tok = json.load(f)
vocab_map = tok['model']['vocab']  # str → id
# Reverse: id → str
id_to_str = {v: k for k, v in vocab_map.items()}
print(f"  Vocab: {len(id_to_str)} tokens")

print("\n▸ Mapping byte → tokens (find tokens whose first byte = X)")
byte_token_ids = {b: [] for b in range(256)}
# Tokens são strings UTF-8 ou GPT-2 byte-tokens (Ġ etc)
# Simple approach: convert token string to bytes and look at first byte
for tok_id in range(vocab):
    s = id_to_str.get(tok_id, '')
    if not s: continue
    # GPT-2 byte tokens have Ġ for space. Skip special tokens.
    if s.startswith('<') and s.endswith('>'): continue
    # Take first byte of UTF-8 encoded token (treating Ġ etc as their literal bytes)
    try:
        b = s.encode('utf-8')[0]
    except:
        continue
    byte_token_ids[b].append(tok_id)

# Stats
present = sum(1 for b in range(256) if byte_token_ids[b])
print(f"  Bytes com tokens: {present}/256")

print("\n▸ Averaging embeddings per byte")
byte_emb = np.zeros((256, hidden), dtype=np.float32)
for b in range(256):
    ids = byte_token_ids[b]
    if ids:
        # Average embeddings of all tokens starting with byte b
        byte_emb[b] = emb[ids].mean(axis=0)

print(f"  byte_emb: {byte_emb.shape}, range=[{byte_emb.min():.3f}, {byte_emb.max():.3f}]")
# Norma média
norms = np.linalg.norm(byte_emb, axis=1)
print(f"  Mean norm: {norms.mean():.3f}, std: {norms.std():.3f}")

print("\n▸ SVD reduce: 256×1536 → 256×64")
U, S, Vt = np.linalg.svd(byte_emb, full_matrices=False)
# Keep top 64 components
top_k = 64
byte_emb_64 = U[:, :top_k] * S[:top_k]
print(f"  Reduced: {byte_emb_64.shape}")
print(f"  Energia retida (top {top_k}/{min(256, hidden)}): {(S[:top_k]**2).sum() / (S**2).sum() * 100:.1f}%")

# Show top byte similarities — sanity check
print("\n▸ Similaridade entre bytes via cossine (espaço Qwen):")
def cossim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

pairs_to_check = [('a', 'A'), ('a', 'e'), ('o', 'i'), ('1', '2'), ('!', '?'), ('a', '!')]
for p, q in pairs_to_check:
    pb = ord(p)
    qb = ord(q)
    s = cossim(byte_emb_64[pb], byte_emb_64[qb])
    print(f"  cossim({p!r}, {q!r}) = {s:+.3f}")

# Save reduced byte embeddings
out_path = 'data/byte_semantics_qwen.bin'
# Format: header u32 (256) | u32 (64) | f32 data [256*64]
with open(out_path, 'wb') as f:
    f.write(struct.pack('<II', 256, top_k))
    f.write(byte_emb_64.astype(np.float32).tobytes())
print(f"\n  ✓ Saved: {out_path} ({os.path.getsize(out_path)} bytes)")

# Compute pairwise distance matrix for top bytes
print("\n▸ TOP 10 bytes mais similares ao 'a':")
target = ord('a')
sims = [(b, cossim(byte_emb_64[target], byte_emb_64[b])) for b in range(256)]
sims.sort(key=lambda x: -x[1])
for b, s in sims[1:11]:
    pb = chr(b) if 32 <= b <= 126 else f'\\x{b:02x}'
    print(f"  {pb!r} (0x{b:02x}): {s:+.3f}")
