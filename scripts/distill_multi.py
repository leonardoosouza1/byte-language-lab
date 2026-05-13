#!/usr/bin/env python3
"""Multi-teacher distillation (PLAN-02 Sprint 2).

Combine multiple Qwen variants (1.5b, 3-4b se disponível) como ensemble teachers.
Each teacher contributes a temperature-varied response per prompt.

Math:
- Ensemble p(y|x) ≈ Σ_k w_k · p_k(y|x)
- Pesos baseados em modelo size + acuracidade empírica
- Output diversity > single-model — student learns broader distribution

Usage:
  python3 distill_multi.py <corpus> [n_prompts=50]
"""
import json, sys, time, struct, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

QWEN_URL = "http://127.0.0.1:3000/api/generate"
WAL_PATH = "/home/leonardo/projects/LLM/byte-language-lab/data/auto_learn.wal"
WORKERS = 2  # Lower than single since multiple models compete for GPU
MAX_TOKENS = 60

TEACHERS = [
    {"name": "qwen2.5-1.5b-q4km", "weight": 0.6, "temps": [0.5, 0.9]},
    # Qwen3-4b é opcional — só se loaded. Will skip if not available.
    {"name": "qwen3-4b",          "weight": 0.4, "temps": [0.5]},
]

TYPE_CURIOSITY = 2
FLAG_FROM_QWEN = 1 << 0
FLAG_HIGH_QUALITY = 1 << 2

def wal_append(user, model, flags=FLAG_FROM_QWEN | FLAG_HIGH_QUALITY):
    p = Path(WAL_PATH)
    user_b = user.encode("utf-8"); model_b = model.encode("utf-8")
    rec = bytearray()
    rec.append(TYPE_CURIOSITY); rec.append(flags)
    rec += struct.pack("<Q", int(time.time()))
    rec += struct.pack("<f", 1.0)
    rec += struct.pack("<I", len(user_b)); rec += struct.pack("<I", len(model_b))
    rec += user_b; rec += model_b
    with open(p, "ab") as f:
        if p.stat().st_size == 0:
            f.write(b"IWAL" + struct.pack("<II", 1, 0))
        f.write(rec)

def call_teacher(model, prompt, temp=0.5):
    body = json.dumps({
        "model": model, "prompt": f"USER: {prompt}\nMODEL: ",
        "max_tokens": MAX_TOKENS, "temperature": temp,
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(QWEN_URL, data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        r = urllib.request.urlopen(req, timeout=60).read()
        d = json.loads(r)
        return d.get("response", "").strip()
    except Exception as e:
        return f"ERR:{type(e).__name__}"

def check_model_available(model_name):
    """Try a tiny generation to check if model is currently loaded."""
    try:
        body = json.dumps({"model": model_name, "prompt": "test", "max_tokens": 2,
                           "temperature": 0.5}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(QWEN_URL, data=body,
            headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=5).read()
        d = json.loads(r)
        return "error" not in d and len(d.get("response", "")) > 0
    except Exception:
        return False

def distill_prompt_multi(prompt, available_teachers):
    """Run each available teacher × each temp. Append all to WAL."""
    ingested = 0
    for teacher in available_teachers:
        for temp in teacher["temps"]:
            resp = call_teacher(teacher["name"], prompt, temp=temp)
            if resp and not resp.startswith("ERR") and len(resp) > 15:
                wal_append(prompt, resp)
                ingested += 1
    return ingested

def main():
    if len(sys.argv) < 2:
        print("usage: distill_multi.py <corpus> [n=50]")
        sys.exit(1)
    corpus = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    print(f"═══ MULTI-TEACHER DISTILLATION ═══")
    print(f"Probing available teachers...")
    # Get currently loaded model
    try:
        r = urllib.request.urlopen("http://127.0.0.1:3000/health", timeout=3).read()
        d = json.loads(r)
        loaded = d.get("model", "none")
        print(f"  Currently loaded: {loaded}")
    except Exception:
        loaded = "none"

    # Filter teachers to only those matching current loaded
    available = [t for t in TEACHERS if t["name"].startswith(loaded.split("-")[0]) or loaded.startswith(t["name"].split("-")[0])]
    if not available:
        # Default: use loaded model directly
        available = [{"name": loaded, "weight": 1.0, "temps": [0.5, 0.9]}]
    print(f"  Active teachers: {[t['name'] for t in available]}")

    with open(corpus) as f:
        prompts = [l.strip() for l in f if l.strip()][:n]
    print(f"  Corpus: {corpus} | n: {len(prompts)}")
    print()

    start = time.time()
    total_ingested = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(distill_prompt_multi, p, available) for p in prompts]
        for i, fut in enumerate(as_completed(futures)):
            ingested = fut.result()
            total_ingested += ingested
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                print(f"[{i+1}/{len(prompts)}] {rate:.2f} p/s | ingested: {total_ingested}")

    elapsed = time.time() - start
    print(f"\n═══ FINAL ═══")
    print(f"  Wall: {elapsed:.1f}s ({len(prompts)/elapsed:.2f} p/s)")
    print(f"  Total WAL records: {total_ingested}")
    print(f"  Average per prompt: {total_ingested/len(prompts):.1f}")

if __name__ == "__main__":
    main()
