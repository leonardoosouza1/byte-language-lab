#!/usr/bin/env python3
"""Test multi-layer Qwen forward + layer averaging.
Experimento:
  A. 3 layers cascade (21, 22, 23)
  B. Average de 4 layers (20-23)
  C. Comparar quality output
"""
import struct, json, numpy as np, sys, os

MODEL = '/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/model.safetensors'
TOKENIZER = '/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/tokenizer.json'

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
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape).copy()

# Constants
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
    cos = np.cos(angles)[:, None, :]
    sin = np.sin(angles)[:, None, :]
    return np.concatenate([x[..., :half] * cos - x[..., half:] * sin,
                            x[..., :half] * sin + x[..., half:] * cos], axis=-1)

def silu(x):
    return x / (1.0 + np.exp(-x))

# Load full embed
print("▸ Loading embedding (467 MB)")
embed = load_t('model.embed_tokens.weight')
final_norm = load_t('model.norm.weight')

# Load N layers
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

def forward_layer(h, layer):
    """Single transformer layer forward."""
    seq = h.shape[0]
    h_in = rmsnorm(h, layer['in_ln'])
    q = h_in @ layer['q'].T + layer['q_b']
    k = h_in @ layer['k'].T + layer['k_b']
    v = h_in @ layer['v'].T + layer['v_b']
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
    h_attn = attn_out @ layer['o'].T
    h = h + h_attn

    h_mlp_in = rmsnorm(h, layer['post_ln'])
    mlp_inter = silu(h_mlp_in @ layer['gate'].T) * (h_mlp_in @ layer['up'].T)
    h = h + mlp_inter @ layer['down'].T
    return h

# Tokenizer
print("▸ Loading tokenizer")
with open(TOKENIZER) as f:
    tokdata = json.load(f)
vocab = tokdata['model']['vocab']
id_to_str = {v: k for k, v in vocab.items()}

# BPE merges
merges_raw = tokdata['model'].get('merges', [])
merges = {}
for i, m in enumerate(merges_raw):
    if isinstance(m, list):
        merges[tuple(m)] = i
    else:
        a, b = m.split(' ', 1)
        merges[(a, b)] = i

# Byte mapping
def b2u_map():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
b2u = b2u_map()

def encode(text):
    tokens = list(''.join(b2u[b] for b in text.encode('utf-8')))
    while len(tokens) > 1:
        best_rank = None; best_idx = -1
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            if pair in merges:
                if best_rank is None or merges[pair] < best_rank:
                    best_rank = merges[pair]; best_idx = i
        if best_idx < 0: break
        tokens = tokens[:best_idx] + [tokens[best_idx] + tokens[best_idx+1]] + tokens[best_idx+2:]
    return [vocab[t] for t in tokens if t in vocab]

def decode(ids):
    text = ''.join(id_to_str.get(i, '') for i in ids)
    u2b = {v: k for k, v in b2u.items()}
    return bytes([u2b.get(c, 0x20) for c in text]).decode('utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────
# EXP A: N layers em cascade
# ─────────────────────────────────────────────────────────────────
def gen_with_layers(prompt, layer_ids, n_tokens=15, label=""):
    print(f"\n  ▸ {label}: layers {layer_ids}")
    print(f"    Loading {len(layer_ids)} layers ({len(layer_ids)*650:.0f} MB)...")
    layers = [load_layer(L) for L in layer_ids]
    ids = encode(prompt)
    print(f"    Tokens: {ids}")
    for step in range(n_tokens):
        h = embed[ids].copy()
        for L in layers:
            h = forward_layer(h, L)
        h = rmsnorm(h, final_norm)
        logits = h[-1] @ embed.T
        nxt = int(np.argmax(logits))
        ids.append(nxt)
        if nxt in (151643, 151645): break
    print(f"    Output: {decode(ids)!r}")
    del layers  # free memory
    return ids

print("\n═══════════════════════════════════════════════════════════")
print("  EXPERIMENTOS")
print("═══════════════════════════════════════════════════════════")

PROMPT = "Olá, "

# Exp 1: layer 21 só (baseline)
gen_with_layers(PROMPT, [21], n_tokens=10, label="EXP1: 1 layer só")

# Exp 2: 3 layers consecutivas
gen_with_layers(PROMPT, [20, 21, 22], n_tokens=10, label="EXP2: 3 layers")

# Exp 3: 5 layers
gen_with_layers(PROMPT, [18, 20, 22, 24, 26], n_tokens=10, label="EXP3: 5 layers sparse")

# Exp 4: 7 layers spaced
gen_with_layers(PROMPT, [4, 9, 14, 18, 22, 25, 27], n_tokens=10, label="EXP4: 7 layers ladder")

# Exp 5: ALL 28 layers (= Qwen full)
print(f"\n  ▸ EXP5: ALL 28 layers (= Qwen completo)")
print(f"    Loading 28 layers (~18 GB peak — may swap)...")
try:
    layers = [load_layer(L) for L in range(28)]
    ids = encode(PROMPT)
    for step in range(10):
        h = embed[ids].copy()
        for L in layers:
            h = forward_layer(h, L)
        h = rmsnorm(h, final_norm)
        logits = h[-1] @ embed.T
        nxt = int(np.argmax(logits))
        ids.append(nxt)
        if nxt in (151643, 151645): break
    print(f"    Output: {decode(ids)!r}")
except Exception as e:
    print(f"    ERROR: {e}")

print("\n═══════════════════════════════════════════════════════════")
print("  CONCLUSÃO")
print("═══════════════════════════════════════════════════════════")
print("""
  Cada experimento mostra quanto layer count importa.
  Hipótese: quality emerge gradualmente com mais layers.
  Quebra prevista: muito menos layers = lixo total
                    todas 28 = Qwen real
""")
