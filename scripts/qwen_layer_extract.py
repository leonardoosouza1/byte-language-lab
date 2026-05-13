#!/usr/bin/env python3
"""EXTRACT layer-wise — capture attention + MLP per layer + lm_head.
Goal: get multi-level semantic signatures que brain pode usar."""
import struct, json, os, numpy as np

MODEL = '/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/model.safetensors'

with open(MODEL, 'rb') as f:
    hsize = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(hsize))
    DOFF = 8 + hsize

def load_tensor(name):
    """Load tensor as float32 numpy array."""
    info = header[name]
    off = info['data_offsets']
    shape = info['shape']
    with open(MODEL, 'rb') as f:
        f.seek(DOFF + off[0])
        raw = f.read(off[1] - off[0])
    if info['dtype'] == 'BF16':
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return u32.view(np.float32).reshape(shape)
    elif info['dtype'] == 'F32':
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    return None

print("▸ Carregando lm_head (output projection)")
# lm_head tied to embed_tokens em Qwen2.5 — vamos verificar
all_names = [k for k in header.keys() if k != '__metadata__']
print("  Tensores top-level (sem .layers.):")
for n in all_names:
    if '.layers.' not in n:
        info = header[n]
        shape = info['shape']
        sz = info['data_offsets'][1] - info['data_offsets'][0]
        print(f"    {n}  shape={shape}  size={sz/1e6:.1f}MB")

# Verifica se lm_head existe ou está tied
has_lm_head = any('lm_head' in n for n in all_names)
print(f"\n  has lm_head explícito: {has_lm_head}")

# Carregar token embedding (input)
print("\n▸ Token embedding (input)")
emb = load_tensor('model.embed_tokens.weight')
print(f"  Shape: {emb.shape}")

# Norma por token — quais tokens são "mais importantes"?
norms = np.linalg.norm(emb, axis=1)
top_norm_idx = np.argsort(-norms)[:20]
bot_norm_idx = np.argsort(norms)[:20]

# Load tokenizer pra ver quais tokens
with open('/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/tokenizer.json') as f:
    tok = json.load(f)
vocab_map = tok['model']['vocab']
id_to_str = {v: k for k, v in vocab_map.items()}

print("  Top-20 tokens com maior norma (mais 'energy'):")
for i in top_norm_idx[:20]:
    s = id_to_str.get(int(i), '?')
    print(f"    [{int(i):>6}] norm={norms[i]:.3f}  {s!r}")

# ─────────────────────────────────────────────────────────────────
# Camadas internas: norma média por dim do hidden state
# ─────────────────────────────────────────────────────────────────
print("\n▸ Análise por camada — norma e estatísticas")
print("  Layer | input_norm | mlp_up_mean | mlp_down_norm")
print("  ──────┼────────────┼─────────────┼───────────────")
for L in [0, 5, 10, 15, 20, 25, 27]:
    try:
        ln = load_tensor(f'model.layers.{L}.input_layernorm.weight')
        up = load_tensor(f'model.layers.{L}.mlp.up_proj.weight')
        down = load_tensor(f'model.layers.{L}.mlp.down_proj.weight')
        print(f"  {L:>5} | {np.linalg.norm(ln):.3f}      | {up.mean():+.6f}    | {np.linalg.norm(down):.2f}")
    except Exception as e:
        print(f"  {L}: erro {e}")

# ─────────────────────────────────────────────────────────────────
# Late layer semantics: ÚLTIMA layer's output projection (lm_head ou final layer)
# ─────────────────────────────────────────────────────────────────
print("\n▸ Atenção do layer 27 (último) — Q, K, V")
q27 = load_tensor('model.layers.27.self_attn.q_proj.weight')
k27 = load_tensor('model.layers.27.self_attn.k_proj.weight')
v27 = load_tensor('model.layers.27.self_attn.v_proj.weight')
print(f"  Q: {q27.shape}, mean abs: {np.abs(q27).mean():.5f}")
print(f"  K: {k27.shape}, mean abs: {np.abs(k27).mean():.5f}")
print(f"  V: {v27.shape}, mean abs: {np.abs(v27).mean():.5f}")

# SVD de Q matrix to extract heads pattern
print("\n▸ SVD da Q layer 27")
U, S, Vt = np.linalg.svd(q27[:1536, :512], full_matrices=False)
energy_top10 = (S[:10]**2).sum() / (S**2).sum()
print(f"  Top-10 singular values capturam {energy_top10*100:.1f}% da energia")
print(f"  S[:10]: {S[:10]}")

# ─────────────────────────────────────────────────────────────────
# Extract second-level semantics: averaged late-layer attention contribution
# ─────────────────────────────────────────────────────────────────
print("\n▸ Late layers (20-27) semantic compress")
# Take output projections from late layers, average, SVD
late_v = []
for L in [25, 26, 27]:
    try:
        v = load_tensor(f'model.layers.{L}.self_attn.v_proj.weight')
        late_v.append(v.flatten()[:256*1536].reshape(256, 1536))
    except: pass

if late_v:
    avg_late = np.mean(late_v, axis=0)
    U, S, Vt = np.linalg.svd(avg_late, full_matrices=False)
    avg_late_64 = U[:, :64] * S[:64]
    out_path = 'data/byte_semantics_qwen_late.bin'
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<II', 256, 64))
        f.write(avg_late_64.astype(np.float32).tobytes())
    print(f"  ✓ Late semantics extraída → {out_path}")
    print(f"  Energy capturado top-64: {(S[:64]**2).sum() / (S**2).sum() * 100:.1f}%")

# ─────────────────────────────────────────────────────────────────
# Compare: input embed vs late v_proj — same byte gets different vectors
# ─────────────────────────────────────────────────────────────────
print("\n▸ Compare 'a' (byte 0x61) representação em layers diferentes")
# Input embedding for 'a' — first 1000 tokens starting with 'a'
byte_a_tokens = [i for i in range(emb.shape[0]) if id_to_str.get(i, '').encode('utf-8', errors='ignore')[:1] == b'a']
print(f"  Tokens começando com 'a': {len(byte_a_tokens)}")
if byte_a_tokens:
    a_emb = emb[byte_a_tokens].mean(axis=0)
    print(f"  Input embedding 'a' (1536d): mean={a_emb.mean():+.5f}  std={a_emb.std():.5f}  norm={np.linalg.norm(a_emb):.3f}")

print("\n═══════════════════════════════════════════════════════════")
print("  CONCLUSÕES")
print("═══════════════════════════════════════════════════════════")
print("""
  - Qwen tem 28 layers, todas com estrutura similar mas pesos distintos
  - Input embedding [151k×1536] = 234M params só pra entrada
  - V_proj (attention) capture 'memória semântica' por layer
  - Late layers (25-27) têm output semântico mais 'final'
  - Extraí late embeddings → SVD 256×64 → byte_semantics_qwen_late.bin
  - Brain agora tem 2 visões: input (byte_semantics_qwen.bin) + output (late)
""")
