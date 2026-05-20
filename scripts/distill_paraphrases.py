#!/usr/bin/env python3
"""distill_paraphrases.py — for every cached (request, code) pair, generate
2 PT paraphrases of the request. Cheap way to boost brain-hi on NEAR-tier
queries: same code, multiple wordings → HDC matches more incoming phrasings.

Reads:  data/code_corpus_history_*.txt (latest)  → 227 pairs
Writes: data/code_corpus.txt with ORIGINAL + 2 paraphrases per pair (~681)

SERIAL — engine not thread-safe.
"""
import argparse, glob, json, re, time, urllib.request

ENGINE = "http://127.0.0.1:3000"
MODEL = "qwen3-4b"


def post(url, payload, timeout=120):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def qwen(prompt, max_tokens=200, temp=0.5):
    chat = (f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
    try:
        d = post(f"{ENGINE}/api/generate", {
            "model": MODEL, "prompt": chat,
            "max_tokens": max_tokens, "temperature": temp,
        })
        return d.get("response", "").strip()
    except Exception as e:
        return f"<err {e}>"


def parse_pairs(text):
    out = []
    for chunk in text.split("===PAIR==="):
        chunk = chunk.strip()
        if "---CODE---" not in chunk:
            continue
        req, code = chunk.split("---CODE---", 1)
        req, code = req.strip(), code.strip()
        if req and code:
            out.append((req, code))
    return out


def paraphrases_for(req):
    """Get 2 PT paraphrases — same intent, different wording, ideally with
    Portuguese synonyms ("pilha" for Stack, "janela modal" for modal, etc)."""
    instr = (
        f"Reescreva esta solicitação técnica de duas formas alternativas em "
        f"português, preservando totalmente o sentido. Use sinônimos PT quando "
        f"natural (ex: 'Stack' → 'pilha', 'modal' → 'janela modal', 'queue' → "
        f"'fila'). Responda APENAS as 2 reformulações, uma por linha, "
        f"numeradas 1. e 2. Sem explicação.\n\n"
        f"Original: {req}"
    )
    raw = qwen(instr, max_tokens=160)
    if raw.startswith("<err"):
        return []
    out = []
    for line in raw.splitlines():
        m = re.match(r"^\s*\d+[\.\)\-]\s*(.+)$", line.strip())
        if m:
            text = m.group(1).strip().strip("\"'`")
            if 10 < len(text) < 200 and text.lower() != req.lower():
                out.append(text)
    return out[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", default="data/code_corpus.txt")
    args = ap.parse_args()

    if args.inp:
        src = args.inp
    else:
        cands = sorted(glob.glob("data/code_corpus_history_*.txt"))
        if not cands:
            print("no code_corpus_history_*.txt found"); return
        src = cands[-1]
    print(f"source corpus: {src}")
    text = open(src).read()
    pairs = parse_pairs(text)
    print(f"original pairs: {len(pairs)}")

    t0 = time.time()
    expanded = []
    for i, (req, code) in enumerate(pairs, 1):
        expanded.append((req, code))           # keep original
        paras = paraphrases_for(req)
        for p in paras:
            expanded.append((p, code))
        if i % 10 == 0:
            elapsed = time.time() - t0
            eta = (len(pairs) - i) * elapsed / i / 60
            print(f"  [{i}/{len(pairs)}] +{len(paras)} paras  eta {eta:.1f}min",
                  flush=True)

    with open(args.out, "w") as f:
        for req, code in expanded:
            f.write("===PAIR===\n")
            f.write(req + "\n")
            f.write("---CODE---\n")
            f.write(code + "\n")

    print(f"\n╔══ PARAPHRASES DONE ══╗")
    print(f"  total pairs: {len(expanded)}  (original {len(pairs)} × ~3)")
    print(f"  elapsed: {(time.time()-t0)/60:.1f}min   → {args.out}")
    print(f"  ingest: curl -X POST http://localhost:3012/learn_code")


if __name__ == "__main__":
    main()
