#!/usr/bin/env python3
"""Extract Qwen layer 21 (mais harmônica) pesos para usar em Rust inference.
- attn.q_proj, k_proj, v_proj, o_proj
- mlp.gate_proj, up_proj, down_proj
- input_layernorm, post_attention_layernorm
- embed_tokens (tied com lm_head)
- final model.norm
Saves binário para Rust loader.
"""
import struct, json, numpy as np, os

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
    arr = (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    return arr, shape

# Config Qwen2.5-1.5b
with open('/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/config.json') as f:
    cfg = json.load(f)
print("Config:", json.dumps(cfg, indent=2)[:600])

LAYER = 21
print(f"\n▸ Extracting layer {LAYER}")

# Tensors needed
tensors = {
    'embed': 'model.embed_tokens.weight',
    'final_norm': 'model.norm.weight',
    'in_ln': f'model.layers.{LAYER}.input_layernorm.weight',
    'post_ln': f'model.layers.{LAYER}.post_attention_layernorm.weight',
    'q': f'model.layers.{LAYER}.self_attn.q_proj.weight',
    'k': f'model.layers.{LAYER}.self_attn.k_proj.weight',
    'v': f'model.layers.{LAYER}.self_attn.v_proj.weight',
    'o': f'model.layers.{LAYER}.self_attn.o_proj.weight',
    'q_bias': f'model.layers.{LAYER}.self_attn.q_proj.bias',
    'k_bias': f'model.layers.{LAYER}.self_attn.k_proj.bias',
    'v_bias': f'model.layers.{LAYER}.self_attn.v_proj.bias',
    'gate': f'model.layers.{LAYER}.mlp.gate_proj.weight',
    'up': f'model.layers.{LAYER}.mlp.up_proj.weight',
    'down': f'model.layers.{LAYER}.mlp.down_proj.weight',
}

# Save header + tensors em arquivo único
out_path = f'data/qwen_layer{LAYER}.bin'
total_bytes = 0
metadata = {}
data_blocks = []

for name, tname in tensors.items():
    if tname not in header:
        print(f"  SKIP {name}: not in header")
        metadata[name] = None
        continue
    arr, shape = load_t(tname)
    metadata[name] = list(shape)
    data_blocks.append((name, arr))
    sz = arr.nbytes
    total_bytes += sz
    print(f"  {name:>15s}: {str(shape):>20s}  {sz/1e6:.2f} MB")

print(f"\n  Total: {total_bytes/1e6:.1f} MB")

# Write file: [magic 4][header_json_len 4][header_json][data blocks...]
import gzip
header_json = json.dumps(metadata).encode()
with open(out_path, 'wb') as f:
    f.write(b'QW21')
    f.write(struct.pack('<I', len(header_json)))
    f.write(header_json)
    for name, arr in data_blocks:
        f.write(arr.astype(np.float32).tobytes())

size = os.path.getsize(out_path)
print(f"\n  ✓ Saved {out_path} ({size/1e6:.1f} MB)")

# Also save tokenizer info essentials
tokpath = '/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b/tokenizer.json'
with open(tokpath) as f:
    tok = json.load(f)

# Save simplified BPE info: vocab + merges
vocab = tok['model']['vocab']
merges = tok['model'].get('merges', [])

# Save vocab as id → bytes
out_tok = 'data/qwen_tokens.txt'
with open(out_tok, 'w') as f:
    for s, idx in sorted(vocab.items(), key=lambda x: x[1]):
        f.write(f"{idx}\t{s}\n")
print(f"  ✓ Tokenizer dump: {out_tok}")

print("""
═══════════════════════════════════════════════════════════
  PRONTO PRA RUST INFERENCE
═══════════════════════════════════════════════════════════
  Layer 21 (mais harmônico) pesos extraídos.
  Próximo passo: Rust runtime que:
   1. Carrega data/qwen_layer21.bin
   2. Tokeniza input (BPE simplificado)
   3. Embed → norm → attention → MLP → norm → unembed → sample
   4. Sample greedy/temperature

  Note: layer single pode dar texto incoerente
        (28 layers fazem hierarquia profunda).
        Mas é prova de conceito de "desmontar Qwen".
""")
