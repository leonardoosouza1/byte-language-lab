#!/usr/bin/env python3
"""Battery test paralelo: dispara N perguntas simultâneas no brain :3012.
Brain decide local (brain-hi) ou Qwen (qwen-rag/primary). WAL absorve passivo.

Usage:
  python3 battery_parallel.py <lista.txt> [workers=8] [limit=N]
"""
import sys, json, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

LIST = sys.argv[1] if len(sys.argv) > 1 else "perguntas-1k-manus.txt"
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 99_999_999

BRAIN_URL = "http://127.0.0.1:3012/chat"
TIMEOUT = 180

def ask(prompt: str, idx: int):
    body = json.dumps({"prompt": prompt, "max": 60, "temp": 0.5},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BRAIN_URL, data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT).read()
        dt = time.time() - t0
        d = json.loads(resp)
        return idx, prompt, d.get("source", "?"), dt, len(d.get("response", ""))
    except Exception as e:
        return idx, prompt, f"ERR:{type(e).__name__}", time.time() - t0, 0

def main():
    with open(LIST) as f:
        prompts = [l.strip() for l in f if l.strip()][:LIMIT]
    print(f"Lista: {LIST} = {len(prompts)} perguntas | workers: {WORKERS}", flush=True)

    counters = {"brain": 0, "qwen-rag": 0, "qwen-primary": 0,
                "markov": 0, "err": 0, "other": 0}
    start = time.time()
    lock = threading.Lock()
    done = 0
    total_dt = 0.0
    total_chars = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(ask, p, i) for i, p in enumerate(prompts)]
        for fut in as_completed(futures):
            idx, prompt, src, dt, chars = fut.result()
            with lock:
                done += 1
                total_dt += dt
                total_chars += chars
                if src.startswith("brain"):
                    counters["brain"] += 1
                elif src == "qwen-rag":
                    counters["qwen-rag"] += 1
                elif src == "qwen-primary":
                    counters["qwen-primary"] += 1
                elif src == "markov":
                    counters["markov"] += 1
                elif src.startswith("ERR"):
                    counters["err"] += 1
                else:
                    counters["other"] += 1
                if done % 50 == 0 or done == len(prompts):
                    elapsed = time.time() - start
                    rate = done / elapsed
                    eta = (len(prompts) - done) / rate if rate > 0 else 0
                    brain_pct = counters["brain"] / done * 100
                    qwen_pct = (counters["qwen-rag"] + counters["qwen-primary"]) / done * 100
                    print(f"[{done}/{len(prompts)}] {rate:.2f} q/s | "
                          f"brain={brain_pct:.0f}% qwen={qwen_pct:.0f}% | "
                          f"err={counters['err']} | "
                          f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                          flush=True)

    print("\n═══ FINAL ═══")
    elapsed = time.time() - start
    for k, v in counters.items():
        pct = v / len(prompts) * 100
        print(f"  {k:>15}: {v:>5} ({pct:.1f}%)")
    print(f"  Wall time:    {elapsed:.1f}s ({len(prompts)/elapsed:.1f} q/s)")
    print(f"  Avg latency:  {total_dt/done*1000:.0f}ms")
    print(f"  Total chars:  {total_chars}")

if __name__ == "__main__":
    main()
