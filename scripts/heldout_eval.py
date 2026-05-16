#!/usr/bin/env python3
"""heldout_eval.py — fixed held-out battery to measure REAL brain progress.

The 50 questions in heldout.json are FROZEN and NEVER fed back as training.
Running this every cycle gives a stable progress curve — unlike auto_eval.py,
whose questions are freshly generated each cycle (so a score change there can
be question luck, not learning).

Qwen3 judges each brain answer against a reference fact. Judging fans out
across the parallel engines. Brain answers are serialized (brain holds a Mutex).

Usage:
  python3 heldout_eval.py [--log path] [--quiet]
"""
import argparse, itertools, json, os, re, threading, time, unicodedata, urllib.request
from concurrent.futures import ThreadPoolExecutor

BRAIN = "http://127.0.0.1:3012"
ENGINES = ["http://127.0.0.1:3000", "http://127.0.0.1:3002"]
MODEL = "qwen3-4b"
HERE = os.path.dirname(os.path.abspath(__file__))

_eng_lock = threading.Lock()
_eng_cycle = itertools.cycle(range(len(ENGINES)))
def next_engine():
    with _eng_lock:
        return ENGINES[next(_eng_cycle)]


def post(url, payload, timeout=90):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def qwen(prompt, max_tokens=80, temp=0.1):
    chat = (f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
    try:
        d = post(f"{next_engine()}/api/generate", {
            "model": MODEL, "prompt": chat,
            "max_tokens": max_tokens, "temperature": temp,
        })
        return d.get("response", "").strip()
    except Exception as e:
        return f"<qwen error: {e}>"


def ask_brain(q):
    try:
        d = post(f"{BRAIN}/chat", {"prompt": q}, timeout=160)
        return d.get("response", "").strip(), d.get("source", "?")
    except Exception as e:
        return f"<brain error: {e}>", "error"


_NUMWORD = {"0": "zero", "1": "um", "2": "dois", "3": "tres", "4": "quatro",
            "5": "cinco", "6": "seis", "7": "sete", "8": "oito", "9": "nove",
            "10": "dez", "100": "cem"}


def _norm(s):
    """Lowercase, strip accents — for accent-insensitive substring match."""
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _edit_le2(a, b):
    """True if Levenshtein(a,b) ≤ 2 — catches spelling/romanization variants."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 2:
        return False
    # banded DP, width 2
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        lo, hi = max(1, i - 2), min(lb, i + 2)
        for j in range(1, lb + 1):
            if j < lo or j > hi:
                cur[j] = 99
                continue
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb] <= 2


# Poison markers — a brain answer carrying these is broken, not just wrong.
_POISON = ("u00e7", "u00e3", "u00ed", "u00e1", "u00f5", "u00e9", "u00f3",
           "the user", "it is the first time", "okay, ", "let me think")


def _terms(text):
    """Discriminative tokens of a string: alpha words ≥4 chars + numbers."""
    n = _norm(text)
    words = {t for t in re.split(r"[^a-z]+", n) if len(t) >= 4}
    nums = set(re.findall(r"\d+", n))
    for d in list(nums):                       # number ↔ word equivalence
        if d in _NUMWORD:
            words.add(_NUMWORD[d])
    return words | nums


def judge(question, reference, brain_answer):
    """Deterministic grader — fast, reproducible, gate-safe.

    A flaky LLM verdict must NEVER drive the train_v2 rollback gate, so this
    uses NO Qwen call. Logic:
      1. poison markers / meta-comment dodge → ERRADO
      2. a DISCRIMINATIVE reference term (one not already in the question,
         numbers and number-words equated) present in the answer → CERTO
      3. otherwise → ERRADO
    Strict but consistent: real improvement still shows as a higher score.
    """
    if not brain_answer or "<brain error" in brain_answer:
        return "ERRADO"
    a_norm = _norm(brain_answer)
    if any(p in a_norm for p in _POISON):
        return "ERRADO"
    if a_norm.lstrip().startswith("a pergunta") and "?" in brain_answer:
        return "ERRADO"  # meta-comment dodge ("A pergunta ... parece ...")

    q_terms = _terms(question)
    # discriminative = in the reference but NOT handed to it by the question
    ref_terms = _terms(reference) - q_terms
    if not ref_terms:
        # short ref ("Ohm", "Rio") — _terms dropped it; keep ≥3-char words
        ref_terms = {w for w in re.split(r"[^a-z]+", _norm(reference))
                     if len(w) >= 3} - q_terms
    a_terms = _terms(brain_answer)
    for t in ref_terms:
        # direct substring (tolerates flexion: "Martes"⊃"marte")
        if t in a_norm:
            return "CERTO"
        # fuzzy: spelling/romanization variants ("Mendeleev"≈"Mendeleiev",
        # "Andromede"≈"Andrômeda") — edit distance ≤2 on words ≥5 chars
        if len(t) >= 5:
            for w in a_terms:
                if len(w) >= 5 and _edit_le2(t, w):
                    return "CERTO"
    return "ERRADO"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/tmp/heldout_eval.jsonl")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--file", default="heldout.json",
                    help="question set: heldout.json (default) or heldout_hard.json")
    args = ap.parse_args()

    with open(os.path.join(HERE, args.file)) as f:
        items = json.load(f)["questions"]

    t0 = time.time()

    def eval_one(item):
        ans, src = ask_brain(item["q"])
        verdict = judge(item["q"], item["ref"], ans)
        return {"q": item["q"], "ref": item["ref"], "answer": ans,
                "src": src, "verdict": verdict}

    workers = max(2, len(ENGINES) * 2)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(eval_one, items))

    grand = {"CERTO": 0, "PARCIAL": 0, "ERRADO": 0}
    solo_ok = 0
    with open(args.log, "w") as logf:
        for r in results:
            grand[r["verdict"]] += 1
            if r["verdict"] == "CERTO" and r["src"].startswith("brain"):
                solo_ok += 1
            if not args.quiet:
                mark = {"CERTO": "✓", "PARCIAL": "~", "ERRADO": "✗"}[r["verdict"]]
                print(f"  {mark} [{r['src']:<22}] {r['q'][:52]}")
            logf.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(grand.values())
    score = grand["CERTO"] + 0.5 * grand["PARCIAL"]
    pct = 100 * score / total if total else 0
    solo_pct = 100 * solo_ok / total if total else 0
    print(f"\n╔══ HELD-OUT ({total} fixed questions) ══╗")
    print(f"  ✓ {grand['CERTO']}  ~ {grand['PARCIAL']}  ✗ {grand['ERRADO']}   elapsed {time.time()-t0:.0f}s")
    print(f"  SCORE: {score:.1f}/{total} ({pct:.0f}%)")
    print(f"  SOLO (correct + no Qwen delegation): {solo_ok}/{total} ({solo_pct:.0f}%)")
    print(f"  HELDOUT_SCORE={pct:.1f} HELDOUT_SOLO={solo_pct:.1f}")


if __name__ == "__main__":
    main()
