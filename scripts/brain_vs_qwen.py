#!/usr/bin/env python3
"""brain_vs_qwen.py — measure the IARA brain's intelligence vs Qwen3.

Asks the SAME questions to both the byte-level brain (:3012) and Qwen3-4B
(engine :3000), scores each answer by keyword presence, and reports the
ratio brain_score / qwen_score — the "% as smart as Qwen3" metric.

Questions are factual PT-BR, broad domains. Not the brain's training probes.

Usage:
  python3 brain_vs_qwen.py [--n N]   # N = how many questions (default all)
"""
import argparse, json, sys, time, urllib.request

BRAIN = "http://127.0.0.1:3012"
ENGINE = "http://127.0.0.1:3000"

# (question, [accepted keywords — any-of match])
QUESTIONS = [
    ("o que é a fotossíntese",            ["luz", "planta", "oxigênio", "clorofila", "energia"]),
    ("qual a capital da França",          ["paris"]),
    ("quem pintou a Mona Lisa",           ["vinci", "leonardo"]),
    ("o que é gravidade",                 ["força", "massa", "atra", "newton"]),
    ("quantos planetas tem o sistema solar", ["oito", "8"]),
    ("o que é o DNA",                     ["genét", "ácido", "célula", "heredit"]),
    ("quem escreveu Dom Casmurro",        ["machado", "assis"]),
    ("o que causa as estações do ano",    ["inclina", "eixo", "terra", "órbita", "translação"]),
    ("o que é inflação",                  ["preço", "moeda", "poder de compra", "aumento"]),
    ("qual o maior oceano",               ["pacífico"]),
    ("o que é um buraco negro",           ["gravidade", "luz", "estrela", "massa", "colapso"]),
    ("quem foi Albert Einstein",          ["físico", "relatividade", "ciência"]),
    ("o que é a célula",                  ["unidade", "vida", "organismo", "membrana"]),
    ("qual a fórmula da água",            ["h2o", "h₂o", "hidrogênio", "oxigênio"]),
    ("o que é democracia",                ["povo", "voto", "governo", "elei"]),
    ("o que é a Amazônia",                ["floresta", "brasil", "rio", "bioma"]),
    ("quem descobriu o Brasil",           ["cabral", "1500", "portugu"]),
    ("o que é energia solar",             ["sol", "luz", "renovável", "painel"]),
    ("o que é a internet",                ["rede", "computador", "conex", "global"]),
    ("o que significa CPU",               ["process", "central", "unidade"]),
]


def post_json(url, payload, timeout=60):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask_brain(q):
    """Returns (response, source). source distinguishes brain-hi cache from
    qwen-* delegation — measures how much the brain answers on its own."""
    try:
        d = post_json(f"{BRAIN}/chat", {"prompt": q})
        return d.get("response", ""), d.get("source", "?")
    except Exception as e:
        return f"<brain error: {e}>", "error"


def ask_qwen(q):
    chat = (f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
    try:
        d = post_json(f"{ENGINE}/api/generate", {
            "model": "qwen3-4b", "prompt": chat,
            "max_tokens": 80, "temperature": 0.1,
        })
        return d.get("response", "")
    except Exception as e:
        return f"<qwen error: {e}>"


def score(answer, keywords):
    a = answer.lower()
    return any(kw in a for kw in keywords)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(QUESTIONS))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    qs = QUESTIONS[:args.n]
    brain_hits = qwen_hits = 0
    src_count = {}
    brain_solo_hits = 0   # correct AND answered without delegating to Qwen

    for q, kw in qs:
        ba, src = ask_brain(q)
        qa = ask_qwen(q)
        bok = score(ba, kw)
        qok = score(qa, kw)
        brain_hits += bok
        qwen_hits += qok
        src_count[src] = src_count.get(src, 0) + 1
        solo = src.startswith("brain")  # brain-hi / brain-mid / brain-fallback
        if bok and solo:
            brain_solo_hits += 1
        mark = lambda x: "✓" if x else "✗"
        print(f"  brain {mark(bok)} [{src:<20}]  qwen {mark(qok)}  {q}")
        if args.verbose:
            print(f"      brain: {ba[:90]}")
            print(f"      qwen:  {qa[:90]}")

    n = len(qs)
    print()
    print(f"═══ RESULTS ({n} questions) ═══")
    print(f"  Brain: {brain_hits}/{n} ({100*brain_hits/n:.0f}%)")
    print(f"  Qwen3: {qwen_hits}/{n} ({100*qwen_hits/n:.0f}%)")
    print(f"  Brain SOLO (no Qwen delegation): {brain_solo_hits}/{n} ({100*brain_solo_hits/n:.0f}%)")
    print(f"  source breakdown: {dict(sorted(src_count.items()))}")
    if qwen_hits > 0:
        ratio = 100 * brain_hits / qwen_hits
        print(f"  Brain is {ratio:.0f}% as smart as Qwen3 (target: ≥50%)")
    else:
        print("  Qwen3 scored 0 — check engine")


if __name__ == "__main__":
    main()
