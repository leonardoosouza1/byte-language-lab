#!/usr/bin/env python3
"""LAYER MAPPING — passar 20 prompts diversos pelo Qwen, capturar layer trace.
Objetivo: descobrir onde hidden state CONVERGE.
Se layer 18 já está perto de layer 28, podemos early-exit em 18 = 36% speedup.
"""
import json, urllib.request, sys, time
import numpy as np

URL = "http://localhost:3000/api/trace"

PROMPTS = [
    # Factual
    "qual a capital do Brasil",
    "quem foi Einstein",
    "o que é fotossíntese",
    "explique buracos negros",
    "como funciona DNA",
    # Conversational
    "olá tudo bem",
    "muito obrigado",
    "estou cansado hoje",
    # Math
    "quanto é 2 mais 2",
    "raiz quadrada de 144",
    # Code
    "como inverter string em Python",
    "o que é REST API",
    # Cultural
    "quem foi Pelé",
    "o que é samba",
    # Random
    "açúcar é doce",
    "ontem chovia muito",
    "Brasil ganhou Copa do Mundo",
    "tecnologia é avanço",
    "música clássica europeia",
    "café da manhã saudável",
]

def call_trace(prompt, token_index=9999):
    body = json.dumps({"prompt": prompt, "token_index": token_index}, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=30).read()
        dt = time.time() - t0
        return json.loads(resp), dt
    except Exception as e:
        return None, str(e)

print("═══════════════════════════════════════════════════════")
print("  LAYER MAPPING — 20 prompts × 28 layers")
print("═══════════════════════════════════════════════════════\n")

# Collect per-prompt × per-layer signals
deltas_matrix = []  # [prompt, layer] = residual_delta
norms_matrix = []
mlp_act = []
prompts_done = []

for i, p in enumerate(PROMPTS):
    sys.stdout.write(f"\r  [{i+1}/{len(PROMPTS)}] {p[:40]:<40}")
    sys.stdout.flush()
    trace, dt = call_trace(p)
    if trace is None:
        print(f"\n    ERR: {dt}")
        continue
    if isinstance(dt, str): continue
    layers = trace.get('layers', [])
    if not layers: continue
    deltas = [l['residual_delta'] for l in layers]
    norms = [l['hidden_norm'] for l in layers]
    mlp = [l['mlp_activation_rate'] for l in layers]
    deltas_matrix.append(deltas)
    norms_matrix.append(norms)
    mlp_act.append(mlp)
    prompts_done.append(p)

print(f"\n\n  Captured {len(deltas_matrix)} prompts × {len(deltas_matrix[0]) if deltas_matrix else 0} layers")

if not deltas_matrix:
    print("ERROR: no traces captured")
    sys.exit(1)

D = np.array(deltas_matrix)
N = np.array(norms_matrix)
M = np.array(mlp_act)
print(f"\n  Deltas shape: {D.shape}")

# ─────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────
print("\n▸ Residual delta per layer (avg across prompts):")
avg_delta = D.mean(axis=0)
max_delta = D.max(axis=0)
min_delta = D.min(axis=0)

for L in range(D.shape[1]):
    bar = '█' * int(avg_delta[L] / avg_delta.max() * 40)
    print(f"  L{L:>2}: avg={avg_delta[L]:6.2f}  range=[{min_delta[L]:6.2f}, {max_delta[L]:6.2f}]  {bar}")

# ─────────────────────────────────────────────────────────────────
# Convergence detection
# ─────────────────────────────────────────────────────────────────
print("\n▸ Convergence analysis")
# Convergence = layer onde delta cai abaixo de X% do max
threshold = 0.3 * avg_delta.max()
conv_layer = None
for L in range(len(avg_delta)):
    if avg_delta[L] < threshold:
        # Need consecutive layers below to confirm
        if L > 0 and avg_delta[L-1] < threshold:
            conv_layer = L
            break

if conv_layer:
    speedup = (28 - conv_layer) / 28 * 100
    print(f"  Convergence layer: ~{conv_layer} (delta < {threshold:.1f})")
    print(f"  Potential early-exit speedup: ~{speedup:.0f}%")
else:
    print(f"  No clear convergence — all layers contribute")

# Per-prompt convergence
print("\n▸ Per-prompt earliest convergence layer:")
for i, p in enumerate(prompts_done[:10]):
    deltas = D[i]
    p_max = deltas.max()
    # Find earliest layer where delta < 30% of max for 3+ consecutive
    earliest = None
    for L in range(len(deltas)-2):
        if all(deltas[L+k] < 0.3 * p_max for k in range(3)):
            earliest = L
            break
    print(f"  {p[:35]:<35}: convergence at L{earliest if earliest else 'never'}")

# ─────────────────────────────────────────────────────────────────
# Top layers per signal
# ─────────────────────────────────────────────────────────────────
print("\n▸ Top 5 layers com MAIOR delta médio (mais importantes):")
top_layers = sorted(range(28), key=lambda L: -avg_delta[L])[:5]
for L in top_layers:
    print(f"  L{L}: avg_delta={avg_delta[L]:.2f}")

print("\n▸ Top 5 layers com MENOR delta (candidatas a SKIP):")
low_layers = sorted(range(28), key=lambda L: avg_delta[L])[:5]
for L in low_layers:
    print(f"  L{L}: avg_delta={avg_delta[L]:.2f}")

# Save
np.save('data/qwen_layer_deltas_real.npy', D)
np.save('data/qwen_layer_norms.npy', N)
np.save('data/qwen_layer_mlp.npy', M)
print(f"\n  ✓ Saved data/qwen_layer_deltas_real.npy + norms + mlp")

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

# Heatmap delta
im1 = ax1.imshow(D, aspect='auto', cmap='hot')
ax1.set_title('Residual delta per layer per prompt')
ax1.set_xlabel('Layer'); ax1.set_ylabel('Prompt')
plt.colorbar(im1, ax=ax1)

# Avg delta + std
ax2.errorbar(range(28), avg_delta, yerr=D.std(axis=0), marker='o')
ax2.axhline(threshold, color='red', linestyle='--', label=f'30% threshold')
if conv_layer: ax2.axvline(conv_layer, color='green', linestyle=':', label=f'convergence L{conv_layer}')
ax2.set_xlabel('Layer'); ax2.set_ylabel('residual_delta')
ax2.set_title('Avg residual delta (with std)')
ax2.legend(); ax2.grid(alpha=0.3)

# Hidden norm per layer
avg_norm = N.mean(axis=0)
ax3.plot(avg_norm, marker='o', color='purple')
ax3.set_xlabel('Layer'); ax3.set_ylabel('hidden_norm')
ax3.set_title('Hidden state norm across depth')
ax3.grid(alpha=0.3)

# MLP activation
avg_mlp = M.mean(axis=0)
ax4.plot(avg_mlp, marker='s', color='teal')
ax4.set_xlabel('Layer'); ax4.set_ylabel('MLP activation rate %')
ax4.set_title('MLP activation per layer')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/viz/qwen_layer_map.png', dpi=110)
plt.close()
print(f"  ✓ Plot: data/viz/qwen_layer_map.png")

print("""
═══════════════════════════════════════════════════════
  CONCLUSÕES
═══════════════════════════════════════════════════════
  - Mapping completo de 28 layers via real Qwen forward
  - Identifica layers de alta vs baixa contribuição
  - Early-exit possible se delta converge cedo
  - Brain pode adicionar lógica:
    confidence baseado em delta drop pattern
    Qwen request com max_layer hint
""")
