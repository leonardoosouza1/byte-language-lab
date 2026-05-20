#!/usr/bin/env python3
"""heldout_code_eval.py — measure brain-hi cache hit rate on code requests.

Three tiers in heldout_code.json:
  - DIRECT  exact corpus phrasing  → should hit cache
  - NEAR    reordered/synonym      → tests HDC robustness
  - NOVEL   not in corpus          → expected to escape

Metric: source field returned by /chat. brain-hi == cache hit, anything
else means the brain escaped to Qwen. We don't grade correctness — just
where the answer came from. Fast (<1 min for 30 requests).
"""
import json, os, time, urllib.request
from collections import Counter

BRAIN = "http://127.0.0.1:3012"
HERE = os.path.dirname(os.path.abspath(__file__))


def post(url, payload, timeout=120):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(q):
    try:
        d = post(f"{BRAIN}/chat", {"prompt": q, "max": 250})
        return d.get("source", "?"), d.get("response", "")
    except Exception as e:
        return f"<err>", str(e)


def main():
    items = json.load(open(os.path.join(HERE, "heldout_code.json")))["questions"]
    print(f"\nrunning {len(items)} code requests against brain :3012\n")

    by_tier = {"direct": [], "near": [], "novel": []}
    t0 = time.time()
    for it in items:
        src, _ = ask(it["q"])
        by_tier[it["tier"]].append(src)
        hit = "✓" if src.startswith("brain") else "✗"
        print(f"  {hit} [{it['tier']:<6}] [{src:<20}] {it['q'][:54]}")

    print(f"\n╔══ CODE HELD-OUT ({len(items)} requests, {time.time()-t0:.0f}s) ══╗")
    for tier, sources in by_tier.items():
        if not sources:
            continue
        hits = sum(1 for s in sources if s.startswith("brain"))
        print(f"  {tier.upper():<7} {hits}/{len(sources)}  ({100*hits/len(sources):.0f}%)  "
              f"sources={dict(Counter(sources))}")
    total = [s for srcs in by_tier.values() for s in srcs]
    total_hits = sum(1 for s in total if s.startswith("brain"))
    print(f"  TOTAL   {total_hits}/{len(total)} brain-hi ({100*total_hits/len(total):.0f}%)")


if __name__ == "__main__":
    main()
