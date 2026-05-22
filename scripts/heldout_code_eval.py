#!/usr/bin/env python3
"""heldout_code_eval.py — cache hit rate + Qwen3 code judge.

For each request in heldout_code.json (3 tiers: direct/near/novel):
  1. Ask the brain — record source field (brain-hi vs qwen-* escape).
  2. Ask Qwen3 — does the returned code actually solve the request?

This catches cache collisions: brain-hi but wrong. Pure brain-hi rate
overstates real performance because HDC can confidently return the
closest cached code even when it's irrelevant.

Two axes:
              SOLVE OK    SOLVE WRONG
  brain-hi    cache win   cache collision  ← the worry
  escape      qwen win    full fail

Usage:  python3 scripts/heldout_code_eval.py
"""
import json, os, re, time, urllib.request
from collections import Counter

BRAIN = "http://127.0.0.1:3012"
ENGINE = "http://127.0.0.1:3000"
MODEL = "qwen3-4b"
HERE = os.path.dirname(os.path.abspath(__file__))


def post(url, payload, timeout=120):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask_brain(q):
    try:
        d = post(f"{BRAIN}/chat", {"prompt": q, "max": 350})
        return d.get("source", "?"), d.get("response", "")
    except Exception as e:
        return "<err>", str(e)


def qwen_judge(request, code):
    """Returns 'OK', 'WRONG', or 'AMBIG'. First alphabetic word parsed."""
    if not code or len(code) < 8:
        return "WRONG"
    snippet = code[:1200]  # cap to keep prompt short
    prompt = (
        f'Solicitação: "{request}"\n\n'
        f"Código retornado:\n```\n{snippet}\n```\n\n"
        f"Esse código atende a solicitação? Comece sua resposta com SIM ou NAO."
    )
    chat = (f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
    try:
        d = post(f"{ENGINE}/api/generate", {
            "model": MODEL, "prompt": chat, "max_tokens": 16, "temperature": 0.0,
        })
        raw = d.get("response", "").upper()
    except Exception:
        return "AMBIG"
    m = re.search(r"[A-ZÀ-Ý]+", raw)
    first = m.group(0) if m else ""
    if first in ("SIM", "OK", "CORRETO", "CERTO"):
        return "OK"
    if first in ("NAO", "NÃO", "NO", "ERRADO", "INCORRETO"):
        return "WRONG"
    return "AMBIG"


def main():
    items = json.load(open(os.path.join(HERE, "heldout_code.json")))["questions"]
    print(f"\nrunning {len(items)} code requests (brain + Qwen judge)\n")
    t0 = time.time()

    rows = []
    for i, it in enumerate(items, 1):
        src, code = ask_brain(it["q"])
        verdict = qwen_judge(it["q"], code) if code else "WRONG"
        rows.append({"q": it["q"], "tier": it["tier"], "src": src, "verdict": verdict,
                     "code": code[:200]})
        cache = "✓" if src.startswith("brain") else "·"
        right = {"OK": "✓", "WRONG": "✗", "AMBIG": "?"}[verdict]
        print(f"  [{i:>2}/{len(items)}] cache:{cache} ok:{right}  "
              f"[{it['tier']:<6}] [{src:<22}] {it['q'][:48]}")

    print(f"\n╔══ CODE HELD-OUT — judge included ({time.time()-t0:.0f}s) ══╗\n")
    print(f"{'TIER':<8} {'HIT':<10} {'OK':<10} {'CACHE-RIGHT':<14} {'CACHE-COLLIDE':<14}")
    for tier in ("direct", "near", "novel"):
        ts = [r for r in rows if r["tier"] == tier]
        if not ts: continue
        hit = sum(1 for r in ts if r["src"].startswith("brain"))
        ok = sum(1 for r in ts if r["verdict"] == "OK")
        cr = sum(1 for r in ts if r["src"].startswith("brain") and r["verdict"] == "OK")
        cc = sum(1 for r in ts if r["src"].startswith("brain") and r["verdict"] == "WRONG")
        n = len(ts)
        print(f"{tier.upper():<8} {hit}/{n} ({100*hit//n}%)  "
              f"{ok}/{n} ({100*ok//n}%)  "
              f"{cr}/{n} ({100*cr//n}%)         "
              f"{cc}/{n} ({100*cc//n}%)")

    total_hit = sum(1 for r in rows if r["src"].startswith("brain"))
    total_ok = sum(1 for r in rows if r["verdict"] == "OK")
    total_cr = sum(1 for r in rows if r["src"].startswith("brain") and r["verdict"] == "OK")
    total_cc = sum(1 for r in rows if r["src"].startswith("brain") and r["verdict"] == "WRONG")
    n = len(rows)
    print(f"\nTOTAL    {total_hit}/{n} ({100*total_hit//n}%)  "
          f"{total_ok}/{n} ({100*total_ok//n}%)  "
          f"{total_cr}/{n} ({100*total_cr//n}%)         "
          f"{total_cc}/{n} ({100*total_cc//n}%)")

    print("\nINTERP:")
    print("  HIT          = brain-hi (cache served, no Qwen escape)")
    print("  OK           = code actually solves the request (Qwen judge)")
    print("  CACHE-RIGHT  = brain-hi AND correct (real cache win)")
    print("  CACHE-COLLIDE= brain-hi BUT wrong (HDC false confidence)")

    out = "data/heldout_code_eval.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nlog → {out}")


if __name__ == "__main__":
    main()
