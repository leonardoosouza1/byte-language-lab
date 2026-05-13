#!/usr/bin/env python3
"""BATTERY TEST — adaptive layer budget validation.
Test same prompts at 3 budgets (fast/medium/slow), measure speed + quality.
Quality heuristic: bigram coherence + token diversity + length.
"""
import json, urllib.request, time, sys
import re

URL = "http://localhost:3000/api/generate"

PROMPTS = [
    "qual a capital do Brasil",
    "explique fotossíntese",
    "como funciona DNA",
    "o que é REST API",
    "quem foi Einstein",
    "qual a velocidade da luz",
    "diferença entre lua e sol",
    "como aprender programação",
    "o que é tecnologia",
    "Brasil é um país grande",
]

def call(prompt, budget=None, max_tokens=80, temp=0.5):
    body_dict = {"model": "qwen2.5-1.5b", "prompt": prompt, "max_tokens": max_tokens, "temperature": temp}
    if budget: body_dict["budget"] = budget
    body = json.dumps(body_dict, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json; charset=utf-8"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=120).read()
        dt = time.time() - t0
        d = json.loads(resp)
        return d.get("response", ""), dt, d.get("tokens_per_second", 0)
    except Exception as e:
        return f"ERR: {e}", time.time() - t0, 0

def quality_score(text):
    """Heuristic: real PT-BR words count + char diversity + no garbage."""
    if not text or text.startswith("ERR"): return 0
    words = re.findall(r"\b[\wáéíóúâêôãõçÁÉÍÓÚ]+\b", text)
    if not words: return 0
    unique = len(set(words))
    # Pt char ratio
    valid_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in '.,!?;:áéíóúâêôãõçÁÉÍÓÚ')
    pt_ratio = valid_chars / max(len(text), 1)
    # Average word length (PT ~5)
    avg_word_len = sum(len(w) for w in words) / len(words)
    word_score = 1.0 - abs(avg_word_len - 5) / 5
    diversity = unique / max(len(words), 1)
    return pt_ratio * 0.4 + diversity * 0.3 + word_score * 0.3

print("═══════════════════════════════════════════════════════")
print("  BUDGET BATTERY — Qwen 28 layers vs 21 vs 7")
print("═══════════════════════════════════════════════════════\n")

results = {"fast": [], "medium": [], "slow": [], "none": []}

print(f"{'Prompt':<35} | {'fast(7L)':>20} | {'med(21L)':>20} | {'slow(28L)':>20}")
print("-" * 105)
for p in PROMPTS:
    row = [f"{p[:33]:<35}"]
    for budget in ["fast", "medium", "slow"]:
        text, dt, tps = call(p, budget=budget, max_tokens=60)
        q = quality_score(text)
        results[budget].append({"prompt": p, "text": text, "dt": dt, "tps": tps, "quality": q})
        row.append(f"{dt:5.2f}s q={q:.2f}".rjust(20))
    print(" | ".join(row))

print()
print("═══ AGREGADO ═══")
for budget in ["fast", "medium", "slow"]:
    rs = results[budget]
    avg_dt = sum(r['dt'] for r in rs) / len(rs)
    avg_q = sum(r['quality'] for r in rs) / len(rs)
    avg_tps = sum(r['tps'] for r in rs) / len(rs)
    print(f"  {budget:>6}: avg_time={avg_dt:.2f}s  quality={avg_q:.3f}  tok/s={avg_tps:.1f}")

# Compute speedup vs slow
slow_avg = sum(r['dt'] for r in results['slow']) / len(results['slow'])
for budget in ["fast", "medium"]:
    avg_dt = sum(r['dt'] for r in results[budget]) / len(results[budget])
    speedup = slow_avg / avg_dt if avg_dt > 0 else 0
    quality_drop = (1 - sum(r['quality'] for r in results[budget]) / sum(r['quality'] for r in results['slow'])) * 100
    print(f"\n  {budget} vs slow: {speedup:.2f}× faster, quality {quality_drop:+.1f}%")

# Sample outputs
print("\n═══ SAMPLE outputs ═══")
sample_idx = [0, 4, 7]  # diverse prompts
for idx in sample_idx:
    p = results['slow'][idx]['prompt']
    print(f"\n  Prompt: {p}")
    for budget in ["fast", "medium", "slow"]:
        r = results[budget][idx]
        text = r['text'][:150].replace('\n', ' ')
        print(f"    [{budget:>6}] ({r['dt']:.2f}s): {text}")
