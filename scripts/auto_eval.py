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
import argparse, json, re, sys, time, urllib.request

BRAIN = "http://127.0.0.1:3012"
ENGINE = "http://127.0.0.1:3000"
MODEL = "qwen3-4b"

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


def qwen(prompt, max_tokens=200, temp=0.5):
    """Call Qwen3 with chat template, reasoning disabled."""
    chat = (f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
    try:
        d = post(f"{ENGINE}/api/generate", {
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
    """Qwen3 as judge — grade the brain's answer. Returns (verdict, reason)."""
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
            if "CERTO" in lu:
                verdict = "CERTO"
            elif "PARCIAL" in lu:
                verdict = "PARCIAL"
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

    for rnd in range(1, args.rounds + 1):
        print(f"\n╔══ ROUND {rnd}/{args.rounds} ══╗")
        for domain in domains:
            qs = gen_questions(domain, args.per)
            if not qs:
                print(f"  [{domain}] — examiner produced 0 questions, skip")
                continue
            dom = {"CERTO": 0, "PARCIAL": 0, "ERRADO": 0}
            print(f"\n  ▸ {domain}  ({len(qs)} questions)")
            for q in qs:
                ans, src = ask_brain(q)
                verdict, reason = judge(q, ans)
                dom[verdict] += 1
                grand[verdict] += 1
                mark = {"CERTO": "✓", "PARCIAL": "~", "ERRADO": "✗"}[verdict]
                print(f"    {mark} [{src:<12}] {q[:62]}")
                if verdict != "CERTO":
                    print(f"        → {ans[:80]}")
                    print(f"        judge: {reason[:80]}")
                    failures.append({"domain": domain, "q": q, "answer": ans,
                                     "src": src, "verdict": verdict, "reason": reason})
                logf.write(json.dumps({
                    "round": rnd, "domain": domain, "q": q, "answer": ans,
                    "src": src, "verdict": verdict, "reason": reason,
                }, ensure_ascii=False) + "\n")
                logf.flush()
            tot = sum(dom.values())
            sc = (dom["CERTO"] + 0.5 * dom["PARCIAL"])
            print(f"    domain score: {sc:.1f}/{tot}  "
                  f"(✓{dom['CERTO']} ~{dom['PARCIAL']} ✗{dom['ERRADO']})")

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
