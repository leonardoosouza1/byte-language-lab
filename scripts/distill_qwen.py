#!/usr/bin/env python3
"""Distillation pipeline Qwen → brain via WAL.

Strategy (PLAN-02 Sprint 1 MVP):
1. Pick N target prompts (from corpus, lists, or domain queries)
2. For each prompt, call Qwen with temperature 0.5 + temperature 1.0 (diverse)
3. Combined response gets ingested via WAL
4. Daemon absorbs → brain learns from soft signal

Math:
- Each prompt generates 2 responses (low+high temp)
- WAL records get HIGH_QUALITY flag for distillation
- Multiple temp runs simulate "soft target" ensemble
- Future: capture full top-K logits via /api/trace + custom WAL record

Usage:
  python3 distill_qwen.py <corpus_file> [n_prompts=100] [temps=0.5,0.9]
"""
import json, sys, time, struct, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

QWEN_URL = "http://127.0.0.1:3000/api/generate"
WAL_PATH = "/home/leonardo/projects/LLM/byte-language-lab/data/auto_learn.wal"
WORKERS = 4
MAX_TOKENS = 80

# WAL record format (mirror Rust src/wal.rs)
TYPE_DIALOG = 1
TYPE_CURIOSITY = 2
FLAG_FROM_QWEN = 1 << 0
FLAG_FROM_CURIOSITY = 1 << 1
FLAG_HIGH_QUALITY = 1 << 2

def wal_append(user: str, model: str, flags: int):
    """Append record to brain WAL."""
    p = Path(WAL_PATH)
    user_b = user.encode("utf-8")
    model_b = model.encode("utf-8")
    rec = bytearray()
    rec.append(TYPE_CURIOSITY)
    rec.append(flags)
    rec += struct.pack("<Q", int(time.time()))
    rec += struct.pack("<f", 1.0)
    rec += struct.pack("<I", len(user_b))
    rec += struct.pack("<I", len(model_b))
    rec += user_b
    rec += model_b
    with open(p, "ab") as f:
        if p.stat().st_size == 0:
            f.write(b"IWAL" + struct.pack("<II", 1, 0))
        f.write(rec)

def call_qwen(prompt: str, temp: float = 0.5, max_tokens: int = MAX_TOKENS):
    """Call Qwen via iara-engine."""
    body = json.dumps({
        "model": "qwen2.5-1.5b-q4km",
        "prompt": f"USER: {prompt}\nMODEL: ",
        "max_tokens": max_tokens,
        "temperature": temp,
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(QWEN_URL, data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        r = urllib.request.urlopen(req, timeout=30).read()
        d = json.loads(r)
        return d.get("response", "").strip()
    except Exception as e:
        return f"ERR:{type(e).__name__}"

def distill_prompt(prompt: str, temps=(0.5, 0.9)):
    """Distill a prompt by collecting multiple temperature responses."""
    ingested = 0
    for temp in temps:
        resp = call_qwen(prompt, temp=temp)
        if resp and not resp.startswith("ERR") and len(resp) > 20:
            wal_append(prompt, resp, FLAG_FROM_QWEN | FLAG_HIGH_QUALITY)
            ingested += 1
    return ingested

def main():
    if len(sys.argv) < 2:
        print("usage: distill_qwen.py <corpus_file> [n_prompts=100] [temps=0.5,0.9]")
        sys.exit(1)
    corpus = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    temps_str = sys.argv[3] if len(sys.argv) > 3 else "0.5,0.9"
    temps = tuple(float(t) for t in temps_str.split(","))

    with open(corpus) as f:
        prompts = [l.strip() for l in f if l.strip()][:n]

    print(f"═══ DISTILLATION ═══")
    print(f"Corpus: {corpus} | n_prompts: {len(prompts)} | temps: {temps}")
    print(f"Expected WAL records: {len(prompts) * len(temps)}")
    print()

    start = time.time()
    total_ingested = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(distill_prompt, p, temps): p for p in prompts}
        for i, fut in enumerate(as_completed(futures)):
            try:
                ingested = fut.result()
                total_ingested += ingested
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed
                    eta = (len(prompts) - i - 1) / rate
                    print(f"[{i+1}/{len(prompts)}] {rate:.1f} prompts/s | total ingested: {total_ingested} | eta: {eta:.0f}s")
            except Exception as e:
                print(f"  ERR: {e}")

    elapsed = time.time() - start
    print(f"\n═══ FINAL ═══")
    print(f"  Wall: {elapsed:.1f}s ({len(prompts)/elapsed:.2f} prompts/s)")
    print(f"  Total ingested: {total_ingested}")
    print(f"  WAL location: {WAL_PATH}")
    print(f"  → Brain daemon (20s cycle) will absorb automatically")

if __name__ == "__main__":
    main()
