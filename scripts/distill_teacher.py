#!/usr/bin/env python3
"""distill_teacher.py — build a training-ready dataset for a future code
student. Architecture-agnostic (feeds SMG, byte-transformer, or the cache).

Two record types, written to a JSONL:

  FIM  (fill-in-the-middle) — from REAL local code (corpus_local.txt). Free,
       huge: for each file we cut prefix | middle | suffix windows. This is
       the bulk of the data and needs NO teacher. Teaches edit/complete.

  SFT  (intent -> code) — from a teacher LLM (Qwen3 now; Gemma/DeepSeek
       later via --engine/--model). Teaches following requests. Slower
       (one LLM call each) so it's opt-in and capped.

Each record carries a `tech` tag (js/ts/tsx/rust/python/...) so training can
be curriculum-ordered or filtered by stack.

Usage:
  python3 scripts/distill_teacher.py --fim 50000              # FIM only, fast
  python3 scripts/distill_teacher.py --fim 50000 --sft 200    # + teacher SFT
  python3 scripts/distill_teacher.py --engine http://127.0.0.1:3000 --model qwen3-4b
"""
import argparse, json, os, random, re, time, urllib.request

CORPUS = "data/corpus_local.txt"
OUT = "data/train_dataset.jsonl"
CTX = 480          # bytes of context each side of the middle
MID_MIN, MID_MAX = 24, 180   # middle span to predict


def post(url, payload, timeout=120):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def split_files(text):
    """corpus_local.txt is framed with `// FILE: <path> (<lang>)` headers."""
    files = []
    cur_lang, cur_lines = None, []
    for line in text.splitlines(keepends=True):
        m = re.match(r"// FILE: .+ \((\w+)\)\s*$", line)
        if m:
            if cur_lang and cur_lines:
                files.append((cur_lang, "".join(cur_lines)))
            cur_lang, cur_lines = m.group(1), []
        else:
            cur_lines.append(line)
    if cur_lang and cur_lines:
        files.append((cur_lang, "".join(cur_lines)))
    return files


def build_fim(files, n_target, seed=7):
    """Cut prefix|middle|suffix windows from real code. The middle is what a
    model learns to predict given both sides — the FIM objective."""
    rng = random.Random(seed)
    recs = []
    big = [(lang, code) for lang, code in files if len(code) > CTX * 2 + MID_MAX]
    if not big:
        return recs
    attempts = 0
    while len(recs) < n_target and attempts < n_target * 4:
        attempts += 1
        lang, code = rng.choice(big)
        mid_len = rng.randint(MID_MIN, MID_MAX)
        lo = CTX
        hi = len(code) - CTX - mid_len
        if hi <= lo:
            continue
        cut = rng.randint(lo, hi)
        prefix = code[cut - CTX:cut]
        middle = code[cut:cut + mid_len]
        suffix = code[cut + mid_len:cut + mid_len + CTX]
        if not middle.strip():
            continue
        recs.append({"type": "fim", "tech": lang,
                     "prefix": prefix, "middle": middle, "suffix": suffix})
    return recs


SFT_INTENTS = None  # loaded lazily from distill_code_v3 pattern lists


def load_sft_intents():
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dcv3", os.path.join(os.path.dirname(__file__), "distill_code_v3.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        groups = (m.DATA_STRUCT + m.SORTS_SEARCH + m.STRINGS + m.MATH + m.VALIDATORS
                  + m.DATES + m.ARRAYS + m.OBJECTS + m.DOM + m.ASYNC_HTTP
                  + m.ERRORS_STORAGE + m.REACT_COMP + m.REACT_HOOKS
                  + m.EXPRESS_NODE + m.TYPESCRIPT + m.HTML_TEMPLATES + m.CSS)
        return [f"crie {g}" for g in groups]
    except Exception as e:
        print(f"  (sft intents load failed: {e})")
        return []


def teacher_code(intent, engine, model):
    instr = (f"Escreva {intent[5:] if intent.startswith('crie ') else intent}. "
             f"Responda APENAS o código dentro de um bloco ``` . Sem prosa.")
    chat = (f"<|im_start|>user\n{instr}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
    try:
        d = post(f"{engine}/api/generate", {
            "model": model, "prompt": chat, "max_tokens": 1000, "temperature": 0.2})
        raw = d.get("response", "")
    except Exception:
        return None
    blocks = re.findall(r"```[a-zA-Z]*\n?(.*?)```", raw, re.DOTALL)
    code = "\n\n".join(b.strip() for b in blocks if b.strip()) if blocks else raw.strip()
    return code if 15 < len(code) < 7800 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fim", type=int, default=50000, help="FIM examples from real code")
    ap.add_argument("--sft", type=int, default=0, help="teacher SFT pairs (0 = skip)")
    ap.add_argument("--engine", default="http://127.0.0.1:3000")
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    t0 = time.time()
    text = open(CORPUS, encoding="utf-8", errors="replace").read()
    files = split_files(text)
    print(f"corpus: {len(files)} files, {len(text)/1e6:.1f} MB")

    fim = build_fim(files, args.fim)
    print(f"FIM examples: {len(fim)}")

    sft = []
    if args.sft > 0:
        intents = load_sft_intents()
        random.Random(3).shuffle(intents)
        intents = intents[:args.sft]
        print(f"SFT via teacher {args.model}: {len(intents)} intents (serial)...")
        for i, intent in enumerate(intents, 1):
            code = teacher_code(intent, args.engine, args.model)
            if code:
                tech = "tsx" if "react" in intent.lower() or "component" in intent.lower() \
                    else "css" if intent.lower().startswith("crie código css") \
                    else "html" if "html" in intent.lower() else "js"
                sft.append({"type": "sft", "tech": tech, "intent": intent, "code": code})
            if i % 20 == 0:
                print(f"  sft {i}/{len(intents)}", flush=True)

    # tech histogram
    from collections import Counter
    techs = Counter(r["tech"] for r in fim + sft)

    with open(args.out, "w") as f:
        for r in fim + sft:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_bytes = sum(len(r.get("prefix", "")) + len(r.get("middle", ""))
                      + len(r.get("suffix", "")) + len(r.get("code", "")) for r in fim + sft)
    print(f"\n╔══ DATASET DONE ══╗")
    print(f"  records: {len(fim)} FIM + {len(sft)} SFT = {len(fim)+len(sft)}")
    print(f"  approx tokens(bytes): {total_bytes/1e6:.1f}M")
    print(f"  by tech: {dict(techs.most_common(12))}")
    print(f"  elapsed: {(time.time()-t0)/60:.1f}min   → {args.out}")


if __name__ == "__main__":
    main()
