#!/usr/bin/env python3
"""auto_eval.py — model-tests-model validation battery.

Qwen3-4B plays two roles:
  1. EXAMINER — generates hard factual questions per domain.
  2. JUDGE    — grades the brain's answer (correct / partial / wrong + reason).

The byte-level brain (:3012) is the subject under test. This removes the
brittle keyword-matching from brain_vs_qwen.py: Qwen3 judges semantically.

Output: per-domain scores, a list of brain failures with the judge's reason,
and a JSONL log for replay. Failures can be fed back into auto-learn.

Usage:
  python3 auto_eval.py [--domains a,b,c] [--per N] [--rounds R] [--log path]
"""
import argparse, itertools, json, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BRAIN = "http://127.0.0.1:3012"
# Parallel Qwen engines — examiner + judge work spread across these.
ENGINES = ["http://127.0.0.1:3000", "http://127.0.0.1:3002"]
MODEL = "qwen3-4b"

# Round-robin engine picker (thread-safe).
_eng_lock = threading.Lock()
_eng_cycle = itertools.cycle(range(len(ENGINES)))
def next_engine():
    with _eng_lock:
        return ENGINES[next(_eng_cycle)]

DOMAINS = [
    "história do Brasil", "ciência (física, química, biologia)",
    "geografia mundial", "matemática básica", "tecnologia e computação",
    "literatura e arte", "economia", "filosofia", "saúde e medicina",
    "atualidades e cultura geral",
]


def post(url, payload, timeout=90):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def qwen(prompt, max_tokens=200, temp=0.5, engine=None):
    """Call Qwen3 with chat template, reasoning disabled.
    engine=None → round-robin across ENGINES (parallel load balancing)."""
    chat = (f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
    eng = engine or next_engine()
    try:
        d = post(f"{eng}/api/generate", {
            "model": MODEL, "prompt": chat,
            "max_tokens": max_tokens, "temperature": temp,
        })
        return d.get("response", "").strip()
    except Exception as e:
        return f"<qwen error: {e}>"


def ask_brain(q):
    try:
        d = post(f"{BRAIN}/chat", {"prompt": q})
        return d.get("response", "").strip(), d.get("source", "?")
    except Exception as e:
        return f"<brain error: {e}>", "error"


def gen_questions(domain, n):
    """Qwen3 as examiner — produce n hard factual questions for a domain."""
    prompt = (
        f"Gere exatamente {n} perguntas factuais de dificuldade média-alta sobre "
        f"{domain}. Cada pergunta deve ter resposta objetiva e verificável. "
        f"Responda APENAS as perguntas, uma por linha, numeradas 1 a {n}. "
        f"Sem explicação, sem cabeçalho."
    )
    raw = qwen(prompt, max_tokens=400, temp=0.7)
    questions = []
    for line in raw.splitlines():
        line = line.strip()
        m = re.match(r"^\d+[\.\)\-]\s*(.+)$", line)
        if m:
            questions.append(m.group(1).strip())
    return questions[:n]


def judge(question, brain_answer):
    """Qwen3 as judge — grade the brain's answer. Returns (verdict, reason).

    Robust parsing: Qwen3 won't reliably emit one exact token, so scan
    tolerantly and check WRONG markers before RIGHT ones ('incorreta'
    must not be misread as 'correta')."""
    prompt = (
        f"Você é um avaliador rigoroso. Pergunta: \"{question}\"\n"
        f"Resposta dada: \"{brain_answer}\"\n\n"
        f"A resposta está factualmente correta e responde a pergunta? "
        f"Responda no formato exato:\n"
        f"VEREDITO: <CERTO|PARCIAL|ERRADO>\n"
        f"RAZAO: <uma frase curta>"
    )
    raw = qwen(prompt, max_tokens=80, temp=0.1)
    verdict = "ERRADO"
    reason = ""
    for line in raw.splitlines():
        lu = line.upper()
        if "VEREDITO" in lu:
            v = lu.split(":", 1)[-1]
            if "PARCIAL" in v:
                verdict = "PARCIAL"
            elif "INCORRET" in v or "ERRAD" in v:
                verdict = "ERRADO"
            elif "CERT" in v or "CORRET" in v:
                verdict = "CERTO"
            else:
                verdict = "ERRADO"
        elif "RAZAO" in lu or "RAZÃO" in lu:
            reason = line.split(":", 1)[-1].strip()
    return verdict, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="", help="comma-separated subset")
    ap.add_argument("--per", type=int, default=5, help="questions per domain")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--log", default="/tmp/auto_eval.jsonl")
    args = ap.parse_args()

    domains = DOMAINS
    if args.domains:
        want = [d.strip() for d in args.domains.split(",")]
        domains = [d for d in DOMAINS if any(w in d for w in want)] or DOMAINS

    logf = open(args.log, "w")
    grand = {"CERTO": 0, "PARCIAL": 0, "ERRADO": 0}
    failures = []
    t0 = time.time()

    workers = max(2, len(ENGINES) * 2)
    for rnd in range(1, args.rounds + 1):
        print(f"\n╔══ ROUND {rnd}/{args.rounds} ══╗")
        # Phase 1: generate all questions, in parallel across engines.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            gen_futs = {ex.submit(gen_questions, d, args.per): d for d in domains}
            dom_qs = {}
            for fut, d in gen_futs.items():
                try:
                    dom_qs[d] = fut.result(timeout=180)
                except Exception:
                    dom_qs[d] = []
        all_q = [(d, q) for d in domains for q in dom_qs.get(d, [])]
        print(f"  generated {len(all_q)} questions across {len(domains)} domains")

        # Phase 2: brain answers (serialized — brain holds a Mutex) but the
        # judge calls fan out across engines.
        def eval_one(item):
            domain, q = item
            ans, src = ask_brain(q)
            verdict, reason = judge(q, ans)  # judge() → qwen() round-robins engines
            return {"round": rnd, "domain": domain, "q": q, "answer": ans,
                    "src": src, "verdict": verdict, "reason": reason}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(eval_one, all_q))

        per_dom = {}
        for r in results:
            d = r["domain"]
            per_dom.setdefault(d, {"CERTO": 0, "PARCIAL": 0, "ERRADO": 0})
            per_dom[d][r["verdict"]] += 1
            grand[r["verdict"]] += 1
            mark = {"CERTO": "✓", "PARCIAL": "~", "ERRADO": "✗"}[r["verdict"]]
            print(f"    {mark} [{r['src']:<20}] {r['q'][:58]}")
            if r["verdict"] != "CERTO":
                failures.append(r)
            logf.write(json.dumps(r, ensure_ascii=False) + "\n")
        logf.flush()
        for d, c in per_dom.items():
            tot = sum(c.values())
            sc = c["CERTO"] + 0.5 * c["PARCIAL"]
            print(f"  ▸ {d}: {sc:.1f}/{tot} (✓{c['CERTO']} ~{c['PARCIAL']} ✗{c['ERRADO']})")

    logf.close()
    total = sum(grand.values())
    score = grand["CERTO"] + 0.5 * grand["PARCIAL"]
    print(f"\n╔══ FINAL ══╗")
    print(f"  questions: {total}   elapsed: {time.time()-t0:.0f}s")
    print(f"  ✓ CERTO   {grand['CERTO']}")
    print(f"  ~ PARCIAL {grand['PARCIAL']}")
    print(f"  ✗ ERRADO  {grand['ERRADO']}")
    if total:
        print(f"  weighted score: {score:.1f}/{total} ({100*score/total:.0f}%)")
    print(f"  failures logged: {len(failures)}  →  {args.log}")


if __name__ == "__main__":
    main()
