#!/usr/bin/env python3
"""Hard test suite — varia dificuldade pra ver onde brain quebra.
Categorias:
  - easy: recall direto (cached)
  - medium: composição entre 2 conceitos
  - hard: multi-hop / análise crítica
  - generative: criatividade aberta
  - reasoning: lógica/matemática inferência
  - lateral: pegadinhas, contexto
  - oov: gírias, neologismos, palavras novas
"""
import json, time, urllib.request

BRAIN = "http://127.0.0.1:3012/chat"

TESTS = {
    "easy_recall": [
        "qual a capital do Brasil",
        "quem foi Albert Einstein",
        "o que é fotossíntese",
        "quando começou a Primeira Guerra Mundial",
        "qual a fórmula da água",
        "quem pintou a Mona Lisa",
        "quem foi Sócrates",
        "o que é Rust",
    ],
    "medium_composition": [
        "compare Kant e Nietzsche",
        "qual a diferença entre socialismo e comunismo",
        "como Einstein e Newton viam a gravidade diferentemente",
        "relacione fotossíntese com mudanças climáticas",
        "compare Rust com Python em performance",
        "qual a diferença entre República e Monarquia",
        "como bossa nova evoluiu do samba",
        "relacione DNA e evolução",
    ],
    "hard_multihop": [
        "qual a relação entre Revolução Francesa e Iluminismo",
        "como o estoicismo influenciou o cristianismo primitivo",
        "explique por que Pearl Harbor levou aos EUA na Segunda Guerra",
        "qual o impacto da Revolução Industrial no surgimento de Karl Marx",
        "explique como a Reforma Protestante criou o capitalismo segundo Weber",
        "qual a ligação entre Império Bizantino e Renascimento italiano",
        "como o existencialismo respondeu ao niilismo de Nietzsche",
        "explique a relação entre fotossíntese e ciclo do carbono",
    ],
    "generative_creative": [
        "escreva um poema curto sobre o tempo",
        "crie uma metáfora para explicar inteligência artificial",
        "invente uma curta história sobre uma estrela apaixonada",
        "descreva o céu como um pintor impressionista",
        "crie um aforismo no estilo de Nietzsche sobre redes sociais",
        "imagine um diálogo entre Sócrates e Einstein",
        "escreva uma máxima zen sobre programação",
        "descreva o oceano usando vocabulário científico",
    ],
    "reasoning_logic": [
        "se 2+2=4 e 4×2=8 quanto é 2+2+4+8",
        "se todo A é B e algum B é C então algum A é C? verdadeiro ou falso",
        "três sentidos: visão audição olfato. qual o quarto",
        "complete a sequência: 2 4 8 16 ?",
        "se domingo é o primeiro dia e quinta é o quinto qual é o terceiro dia",
        "quantas patas têm 7 cachorros",
        "se A=1 B=2 quanto vale CASA",
        "quanto é metade de 1/4",
    ],
    "lateral_tricky": [
        "qual a cor do cavalo branco de Napoleão",
        "o que vem antes do 1 mas depois do 3",
        "se um trem vai a 60km/h quanto tempo leva pra chegar",
        "Quem está enterrado no túmulo de Napoleão",
        "o que pesa mais 1kg de chumbo ou 1kg de algodão",
        "como você atravessa um rio sem ponte sem barco e sem nadar",
        "quantos meses têm 28 dias",
        "qual a primeira letra do alfabeto",
    ],
    "oov_slang": [
        "explique o que é cringe",
        "o que significa shippar alguém",
        "defina mood",
        "o que é flopar",
        "fala sobre crush",
        "o que significa stalkear",
        "explique cancelamento na internet",
        "o que é red flag em relacionamento",
    ],
    "memory_self": [
        "qual foi a primeira pergunta que te fiz",
        "lembra do que falamos sobre fotossíntese",
        "quem é Leonardo",
        "qual é o nome do modelo que você usa",
        "você sabe quem eu sou",
        "qual sua função",
        "fale sobre IARA",
        "o que é byte-language",
    ],
}

def ask(prompt):
    body = json.dumps({"prompt": prompt, "max": 100, "temp": 0.7},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(BRAIN, data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=60).read()
        d = json.loads(r)
        return d.get("source","?"), d.get("response",""), time.time() - t0
    except Exception as e:
        return f"ERR:{type(e).__name__}", "", time.time() - t0

def main():
    print("═" * 80)
    print("  IARA-BYTE BRAIN — HARD TEST (mixed difficulty)")
    print("═" * 80)
    print()

    overall = {}
    for cat, questions in TESTS.items():
        print(f"╔═══ {cat.upper()} ({len(questions)} q) ═══╗")
        counters = {"brain":0, "qwen":0, "markov":0, "err":0}
        total_dt = 0
        samples = []
        for q in questions:
            src, resp, dt = ask(q)
            total_dt += dt
            if src.startswith("brain"): counters["brain"] += 1
            elif src.startswith("qwen"): counters["qwen"] += 1
            elif src == "markov": counters["markov"] += 1
            elif src.startswith("ERR"): counters["err"] += 1
            samples.append((q, src, resp[:120].replace('\n',' ')))
        # Sample print
        for q, src, r in samples:
            tag = "✓" if src.startswith("brain") else ("Q" if src.startswith("qwen") else ("?" if src.startswith("ERR") else "M"))
            print(f"  [{tag}] {src:<14} {q[:50]:<50} → {r[:70]}")
        n = len(questions)
        bp = counters['brain']/n*100
        avg_ms = total_dt/n*1000
        overall[cat] = {"brain_pct": bp, "counters": counters, "avg_ms": avg_ms}
        print(f"  brain={counters['brain']}/{n} ({bp:.0f}%) qwen={counters['qwen']} mkv={counters['markov']} err={counters['err']} avg={avg_ms:.0f}ms\n")

    print("═" * 80)
    print(f"{'Categoria':<25} {'Brain%':>8} {'B/Q/M/E':>10} {'avg ms':>8}  Saturação")
    print("─" * 80)
    sorted_cats = sorted(overall.items(), key=lambda x: -x[1]["brain_pct"])
    for cat, info in sorted_cats:
        c = info["counters"]
        bar = "█" * int(info["brain_pct"]/5)
        print(f"  {cat:<23} {info['brain_pct']:>5.0f}%  {c['brain']}/{c['qwen']}/{c['markov']}/{c['err']:<3}  {info['avg_ms']:>6.0f}  {bar}")
    composite_brain = sum(info["counters"]["brain"] for info in overall.values())
    composite_total = sum(sum(info["counters"].values()) for info in overall.values())
    print(f"\n  COMPOSITE BRAIN-ONLY: {composite_brain/composite_total*100:.1f}% ({composite_brain}/{composite_total})")
    # Save
    with open("/tmp/hard_test_report.json", "w") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: /tmp/hard_test_report.json")

if __name__ == "__main__":
    main()
