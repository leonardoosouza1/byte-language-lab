#!/usr/bin/env python3
"""PROTOTYPE: rodar layer 21 isolada do Qwen + ver se gera algo coerente.
Hipótese: 1 layer só vai produzir lixo (28 layers fazem hierarquia).
Mas vamos validar isso experimentalmente.
"""
import struct, json, numpy as np, sys, os

LAYER_BIN = 'data/qwen_layer21.bin'
TOKENIZER = '/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/tokenizer.json'

print("▸ Loading layer 21 binary")
with open(LAYER_BIN, 'rb') as f:
    magic = f.read(4)
    assert magic == b'QW21'
    hlen = struct.unpack('<I', f.read(4))[0]
    meta = json.loads(f.read(hlen))

# Read tensors em ordem (mesmo do extract script)
tensor_names = ['embed', 'final_norm', 'in_ln', 'post_ln',
                'q', 'k', 'v', 'o', 'q_bias', 'k_bias', 'v_bias',
                'gate', 'up', 'down']
tensors = {}
with open(LAYER_BIN, 'rb') as f:
    f.seek(8 + hlen)
    for name in tensor_names:
        if meta.get(name) is None: continue
        shape = meta[name]
        n_elem = 1
        for d in shape: n_elem *= d
        raw = f.read(n_elem * 4)
        tensors[name] = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        print(f"  {name:>12s}: {shape}")

# Constants Qwen2.5-1.5b
HIDDEN = 1536
N_HEAD = 12
N_KV_HEAD = 2
HEAD_DIM = HIDDEN // N_HEAD  # 128
ROPE_THETA = 1_000_000.0
EPS = 1e-6
INTER = 8960  # MLP intermediate

print(f"\nConfig: hidden={HIDDEN}, heads={N_HEAD}, kv_heads={N_KV_HEAD}, head_dim={HEAD_DIM}")

# RMSNorm
def rmsnorm(x, w, eps=EPS):
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return (x / rms) * w

# Rotary embeddings
def rope(x, pos):
    # x: [seq, n_head, head_dim]
    # Apply RoPE
    half = HEAD_DIM // 2
    freqs = 1.0 / (ROPE_THETA ** (np.arange(0, half, dtype=np.float32) * 2.0 / HEAD_DIM))
    t = np.arange(pos, pos + x.shape[0], dtype=np.float32)
    angles = np.outer(t, freqs)  # [seq, half]
    cos = np.cos(angles)[:, None, :]  # [seq, 1, half]
    sin = np.sin(angles)[:, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

# SiLU
def silu(x):
    return x / (1.0 + np.exp(-x))

# Load tokenizer
print("\n▸ Loading tokenizer (BPE)")
with open(TOKENIZER) as f:
    tokdata = json.load(f)
vocab = tokdata['model']['vocab']
merges_raw = tokdata['model'].get('merges', [])
# Handle merges as list of [a, b] pairs or "a b" strings
merges = {}
for i, m in enumerate(merges_raw):
    if isinstance(m, list):
        merges[tuple(m)] = i
    else:
        a, b = m.split(' ', 1)
        merges[(a, b)] = i
id_to_str = {v: k for k, v in vocab.items()}

# Simple BPE encode (GPT-2 style with byte-level pre-tokenize)
def bytes_to_unicode():
    # Standard GPT-2 byte-level mapping
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

b2u = bytes_to_unicode()

def encode_text(text):
    # Convert text to byte-level chars
    encoded = ''.join(b2u[b] for b in text.encode('utf-8'))
    # BPE: start with chars, merge greedily
    tokens = list(encoded)
    while len(tokens) > 1:
        best_rank = None
        best_idx = -1
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            if pair in merges:
                if best_rank is None or merges[pair] < best_rank:
                    best_rank = merges[pair]
                    best_idx = i
        if best_idx < 0: break
        tokens = tokens[:best_idx] + [tokens[best_idx] + tokens[best_idx+1]] + tokens[best_idx+2:]
    return [vocab[t] for t in tokens if t in vocab]

def decode_tokens(ids):
    text = ''.join(id_to_str.get(i, '') for i in ids)
    # Reverse byte-level mapping
    u2b = {v: k for k, v in b2u.items()}
    bytes_out = bytes([u2b.get(c, 0x20) for c in text])
    return bytes_out.decode('utf-8', errors='replace')

# Simple test
print("\n▸ Tokenizer test")
tids = encode_text("Olá mundo")
print(f"  'Olá mundo' → {tids}")
print(f"  Decoded back: {decode_tokens(tids)!r}")

# ─────────────────────────────────────────────────────────────────
# Forward 1 layer
# ─────────────────────────────────────────────────────────────────
print("\n▸ Forward 1 layer pass")

def forward(token_ids):
    """Run single-layer forward. Returns logits for last token."""
    # Embedding
    h = tensors['embed'][token_ids]  # [seq, hidden]
    seq = h.shape[0]

    # RMSNorm before attention
    h_in = rmsnorm(h, tensors['in_ln'])

    # QKV projections
    q = h_in @ tensors['q'].T + tensors.get('q_bias', np.zeros(HIDDEN))
    k = h_in @ tensors['k'].T + tensors.get('k_bias', np.zeros(256))
    v = h_in @ tensors['v'].T + tensors.get('v_bias', np.zeros(256))

    q = q.reshape(seq, N_HEAD, HEAD_DIM)
    k = k.reshape(seq, N_KV_HEAD, HEAD_DIM)
    v = v.reshape(seq, N_KV_HEAD, HEAD_DIM)

    # RoPE
    q = rope(q, 0)
    k = rope(k, 0)

    # GQA: expand K, V to N_HEAD
    repeat_factor = N_HEAD // N_KV_HEAD  # 6
    k = np.repeat(k, repeat_factor, axis=1)  # [seq, N_HEAD, head_dim]
    v = np.repeat(v, repeat_factor, axis=1)

    # Attention scores
    # q: [seq, n_head, head_dim]
    # k: [seq, n_head, head_dim]
    # scores: [n_head, seq_q, seq_k]
    q_t = q.transpose(1, 0, 2)  # [n_head, seq, head_dim]
    k_t = k.transpose(1, 0, 2)
    v_t = v.transpose(1, 0, 2)
    scores = q_t @ k_t.transpose(0, 2, 1) / np.sqrt(HEAD_DIM)
    # Causal mask
    mask = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)
    scores = scores + mask[None]
    # Softmax
    scores = scores - scores.max(axis=-1, keepdims=True)
    scores = np.exp(scores)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    # Attended
    attn_out = scores @ v_t  # [n_head, seq, head_dim]
    attn_out = attn_out.transpose(1, 0, 2).reshape(seq, HIDDEN)
    # Output projection
    h_attn = attn_out @ tensors['o'].T
    h = h + h_attn  # residual

    # MLP
    h_mlp_in = rmsnorm(h, tensors['post_ln'])
    gate = silu(h_mlp_in @ tensors['gate'].T)
    up = h_mlp_in @ tensors['up'].T
    mlp_inter = gate * up  # [seq, 8960]
    h_mlp = mlp_inter @ tensors['down'].T
    h = h + h_mlp

    # Final norm
    h = rmsnorm(h, tensors['final_norm'])

    # Last token logits via tied lm_head = embed
    last = h[-1]
    logits = last @ tensors['embed'].T  # [vocab]
    return logits

# Test geração
PROMPT = "Olá, "
print(f"\n  Prompt: {PROMPT!r}")
ids = encode_text(PROMPT)
print(f"  Token IDs: {ids}")

# Greedy generation 10 tokens
generated = list(ids)
for step in range(10):
    logits = forward(np.array(generated, dtype=np.int64))
    # Top-3 candidates
    top3 = np.argsort(-logits)[:3]
    top3_str = ', '.join(f"{int(i)}={id_to_str.get(int(i), '?')!r}({logits[i]:.2f})" for i in top3)
    print(f"  Step {step}: top3 = {top3_str}")
    # Greedy
    next_tok = int(np.argmax(logits))
    generated.append(next_tok)
    if next_tok in (151643, 151645):  # eos tokens
        break

output = decode_tokens(generated)
print(f"\n  Generated: {output!r}")
print("\n═══════════════════════════════════════════════════════════")
print("  ANALYSIS")
print("═══════════════════════════════════════════════════════════")
print(f"""
  Single-layer Qwen geração testada.
  Esperado: incoerente (1 layer só)
  Real: vide acima

  Conclusão: layer 21 isolada NÃO é Qwen funcional.
  Mas weights estão disponíveis pra experimentos.
""")
