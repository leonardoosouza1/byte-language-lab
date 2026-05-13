#!/usr/bin/env python3
"""Cross-domain intelligence/knowledge probe for IARA-Byte Brain.

Queries N probe questions per domain.
Counts brain-hi/mid (cached/learned) vs qwen-* (novel/unknown).
Low brain% in domain = brain needs learning there.

Brain self-reports gaps via /curiosity/topics or weak-words inference.
"""
import json, urllib.request, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

BRAIN = "http://127.0.0.1:3012/chat"
WORKERS = 6

# ── Probe questions per domain ────────────────────────────────────
PROBES = {
    "História BR": [
        "quem proclamou a independência do Brasil",
        "quando foi a Inconfidência Mineira",
        "o que foi a Lei Áurea",
        "quem foi Tiradentes",
        "o que foi a Era Vargas",
        "quando aconteceu o Golpe Militar no Brasil",
        "quem foi Zumbi dos Palmares",
        "o que foi a Guerra do Paraguai",
        "quando foi promulgada a Constituição de 1988",
        "quem foi Dom Pedro II",
    ],
    "História Mundial": [
        "quando começou a Primeira Guerra Mundial",
        "quem foi Napoleão Bonaparte",
        "o que foi a Revolução Francesa",
        "quando caiu o Império Romano",
        "quem foi Júlio César",
        "o que foi o Renascimento",
        "quando foi a Revolução Russa",
        "quem foi Cleópatra",
        "o que aconteceu em Hiroshima em 1945",
        "quem fundou Roma",
    ],
    "Ciências Naturais": [
        "o que é fotossíntese",
        "qual a fórmula da água",
        "o que é gravidade",
        "como funciona o sistema imunológico",
        "o que é DNA",
        "qual a velocidade da luz",
        "o que é entropia",
        "como funciona o ciclo de Krebs",
        "o que é um elétron",
        "o que causa um terremoto",
    ],
    "Matemática": [
        "qual a fórmula de Bhaskara",
        "o que é derivada",
        "o que é número primo",
        "qual o valor de pi",
        "o que é teorema de Pitágoras",
        "o que é logaritmo",
        "o que é integral",
        "o que é matriz",
        "o que é probabilidade condicional",
        "o que é função exponencial",
    ],
    "Tecnologia": [
        "o que é Rust",
        "como funciona uma CPU",
        "o que é blockchain",
        "o que é machine learning",
        "como funciona TCP",
        "o que é Docker",
        "o que é JavaScript",
        "como funciona Git",
        "o que é REST API",
        "o que é kubernetes",
    ],
    "Geografia": [
        "qual a capital da França",
        "qual o maior país do mundo",
        "qual o rio mais longo",
        "onde fica o monte Everest",
        "qual a capital do Japão",
        "quais são os países da península Ibérica",
        "onde fica o deserto do Saara",
        "qual a capital da Austrália",
        "qual o oceano entre Brasil e África",
        "onde fica a Cordilheira dos Andes",
    ],
    "Artes e Cultura": [
        "quem pintou a Mona Lisa",
        "quem escreveu Dom Casmurro",
        "quem compôs a Nona Sinfonia",
        "o que é cubismo",
        "quem foi Pablo Picasso",
        "o que é bossa nova",
        "quem dirigiu Cidadão Kane",
        "o que é arte abstrata",
        "quem foi William Shakespeare",
        "o que é literatura modernista",
    ],
    "Filosofia": [
        "quem foi Sócrates",
        "o que é estoicismo",
        "quem escreveu Crítica da Razão Pura",
        "o que é existencialismo",
        "quem foi Nietzsche",
        "o que é dualismo cartesiano",
        "quem foi Aristóteles",
        "o que é fenomenologia",
        "quem foi Karl Marx",
        "o que é niilismo",
    ],
    "Saúde e Medicina": [
        "o que é diabetes",
        "como funciona uma vacina",
        "o que é hipertensão",
        "o que causa câncer",
        "como funciona o coração",
        "o que é antibiótico",
        "o que é metabolismo",
        "como funciona o cérebro",
        "o que é colesterol",
        "o que é insulina",
    ],
    "Economia": [
        "o que é inflação",
        "o que é PIB",
        "como funciona a bolsa de valores",
        "o que é capitalismo",
        "o que é juros compostos",
        "o que é dívida pública",
        "o que é câmbio",
        "o que é oferta e demanda",
        "o que é macroeconomia",
        "o que é Bitcoin",
    ],
}

def ask(prompt):
    body = json.dumps({"prompt": prompt, "max": 60, "temp": 0.5},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(BRAIN, data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=60).read()
        d = json.loads(resp)
        return d.get("source", "?"), time.time() - t0, len(d.get("response", ""))
    except Exception as e:
        return f"ERR:{type(e).__name__}", time.time() - t0, 0

def probe_domain(name, prompts):
    counters = {"brain": 0, "qwen": 0, "markov": 0, "err": 0}
    total_dt = 0.0
    total_chars = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(ask, p) for p in prompts]
        for fut in as_completed(futures):
            src, dt, chars = fut.result()
            total_dt += dt
            total_chars += chars
            if src.startswith("brain"): counters["brain"] += 1
            elif src.startswith("qwen"): counters["qwen"] += 1
            elif src == "markov": counters["markov"] += 1
            elif src.startswith("ERR"): counters["err"] += 1
    n = len(prompts)
    brain_pct = counters["brain"] / n * 100
    return {
        "domain": name,
        "n": n,
        "brain": counters["brain"],
        "qwen": counters["qwen"],
        "markov": counters["markov"],
        "err": counters["err"],
        "brain_pct": brain_pct,
        "avg_ms": total_dt / n * 1000,
        "avg_chars": total_chars / n,
    }

def main():
    print("═" * 70)
    print("  IARA-BYTE BRAIN — TESTE DE INTELIGÊNCIA CROSS-DOMAIN")
    print("═" * 70)
    print()
    start = time.time()
    results = []
    for domain, prompts in PROBES.items():
        print(f"  ▸ Probando {domain}...", flush=True)
        r = probe_domain(domain, prompts)
        results.append(r)

    print()
    print("═" * 70)
    print(f"{'Domínio':<25} {'Brain%':>8} {'Brain':>6} {'Qwen':>6} {'Mkv':>4} {'Avg ms':>8}")
    print("─" * 70)
    # Sort by brain_pct asc (weakest first)
    results.sort(key=lambda r: r["brain_pct"])
    for r in results:
        bar = "█" * int(r["brain_pct"] / 5)
        print(f"  {r['domain']:<23} {r['brain_pct']:>6.1f}%  {r['brain']:>4}/{r['n']:<3} {r['qwen']:>4}  {r['markov']:>4} {r['avg_ms']:>7.0f}  {bar}")
    print("─" * 70)

    # Overall summary
    total_brain = sum(r["brain"] for r in results)
    total_n = sum(r["n"] for r in results)
    overall = total_brain / total_n * 100
    print(f"\n  COMPOSITE BRAIN COVERAGE: {overall:.1f}% ({total_brain}/{total_n})")

    # Top 3 weakest = priority learning
    print(f"\n  ▸ ÁREAS PRIORITÁRIAS PARA APRENDER:")
    for i, r in enumerate(results[:3], 1):
        gap = 100 - r["brain_pct"]
        print(f"    {i}. {r['domain']:<25} gap={gap:.0f}% (cobre {r['brain_pct']:.0f}%)")

    # Top 3 strongest = brain knows well
    print(f"\n  ▸ ÁREAS DOMINADAS:")
    for i, r in enumerate(sorted(results, key=lambda x: -x["brain_pct"])[:3], 1):
        print(f"    {i}. {r['domain']:<25} {r['brain_pct']:.0f}% coverage")

    # Save report
    report = {
        "timestamp": time.time(),
        "wall_time_s": time.time() - start,
        "domains": results,
        "composite_coverage": overall,
        "weakest_3": [r["domain"] for r in results[:3]],
        "strongest_3": [r["domain"] for r in sorted(results, key=lambda x: -x["brain_pct"])[:3]],
    }
    with open("/tmp/iq_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Relatório salvo: /tmp/iq_report.json")
    print(f"  Wall: {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()
