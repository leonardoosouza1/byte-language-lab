#!/usr/bin/env python3
"""Targeted gap-fill loop. Lê /tmp/iq_report.json, identifica áreas fracas,
gera perguntas direcionadas, ingere via brain (→ Qwen → WAL → daemon),
re-testa, repete até cobertura alvo OU max iterations.

Self-feeding learning. Sem humano.
"""
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BRAIN = "http://127.0.0.1:3012/chat"
AUTOLEARN = "http://127.0.0.1:3012/autolearn"
WORKERS = 6
ITER_MAX = 3
TARGET_COVERAGE = 70  # min brain% per weak domain

# Question templates per domain (seeds → many variants)
SEEDS = {
    "História Mundial": {
        "subjects": [
            "Império Romano", "Idade Média", "Carlos Magno", "Cruzadas",
            "Atila o Huno", "Império Bizantino", "Vikings", "Inquisição",
            "Marco Polo", "Cristóvão Colombo", "Magalhães", "Vasco da Gama",
            "Reforma Protestante", "Lutero", "Henrique VIII", "Elisabeth I",
            "Iluminismo", "Voltaire", "Rousseau", "Revolução Industrial",
            "Napoleão", "Bismarck", "Otto von Bismarck", "Czar Nicolau II",
            "Lenin", "Stalin", "Mao Tsé-Tung", "Gandhi", "Churchill",
            "Hitler", "Mussolini", "Roosevelt", "Pearl Harbor",
            "Guerra Fria", "Muro de Berlim", "JFK", "Crise dos mísseis",
            "Independência dos EUA", "George Washington", "Lincoln",
            "Imperatriz Catarina", "Maria Antonieta", "Robespierre",
        ],
        "templates": [
            "quem foi {}", "o que fez {}", "qual o legado de {}",
            "quando viveu {}", "em que país atuou {}",
            "qual a importância de {}", "fale sobre {}",
        ],
    },
    "Filosofia": {
        "subjects": [
            "Sócrates", "Platão", "Aristóteles", "Descartes", "Kant",
            "Hegel", "Nietzsche", "Heidegger", "Sartre", "Beauvoir",
            "Foucault", "Derrida", "Wittgenstein", "Russell",
            "Confúcio", "Lao-Tsé", "Buda", "São Tomás de Aquino",
            "Maquiavel", "Hobbes", "Locke", "Hume", "Spinoza",
            "Schopenhauer", "Kierkegaard", "Husserl", "Marx",
            "estoicismo", "epicurismo", "ceticismo", "racionalismo",
            "empirismo", "idealismo", "materialismo", "existencialismo",
            "fenomenologia", "hermenêutica", "estruturalismo",
            "pós-modernismo", "filosofia analítica", "ética",
            "metafísica", "epistemologia", "lógica", "estética",
        ],
        "templates": [
            "quem foi {}", "o que pensava {}", "qual a tese de {}",
            "explique {}", "qual o conceito central de {}",
            "fale sobre {}", "o que defendia {}",
        ],
    },
    "Artes e Cultura": {
        "subjects": [
            "Leonardo da Vinci", "Michelangelo", "Rafael", "Caravaggio",
            "Rembrandt", "Vermeer", "Van Gogh", "Picasso", "Dalí",
            "Frida Kahlo", "Tarsila do Amaral", "Cândido Portinari",
            "Romantismo", "Barroco", "Impressionismo", "Surrealismo",
            "Mona Lisa", "A Última Ceia", "Guernica", "O Grito",
            "Beethoven", "Mozart", "Bach", "Vivaldi", "Tchaikovsky",
            "Chopin", "Wagner", "Verdi", "Puccini", "Stravinsky",
            "Tom Jobim", "Vinícius de Moraes", "Chico Buarque", "Caetano",
            "samba", "bossa nova", "MPB", "fado", "tango",
            "Shakespeare", "Cervantes", "Goethe", "Tolstoi", "Dostoiévski",
            "Machado de Assis", "Jorge Amado", "Guimarães Rosa",
            "Clarice Lispector", "Cecília Meireles", "Drummond",
            "Hitchcock", "Kubrick", "Fellini", "Bergman", "Glauber Rocha",
        ],
        "templates": [
            "quem foi {}", "qual a obra de {}", "fale sobre {}",
            "quando viveu {}", "qual o estilo de {}", "o que é {}",
        ],
    },
    "Economia": {
        "subjects": [
            "Adam Smith", "Karl Marx", "Keynes", "Milton Friedman",
            "Hayek", "Schumpeter", "Mahatma Bandyopadhyay",
            "capitalismo", "socialismo", "comunismo", "liberalismo",
            "neoliberalismo", "intervencionismo", "monetarismo",
            "inflação", "deflação", "estagflação", "depressão",
            "PIB", "PNB", "IDH", "Selic", "CDI", "IPCA",
            "Banco Central", "FMI", "Banco Mundial", "OMC",
            "bolsa de valores", "ações", "renda fixa", "derivativos",
            "câmbio", "balança comercial", "balança de pagamentos",
            "Bitcoin", "Ethereum", "blockchain econômico",
            "crise de 2008", "crise de 1929", "Grande Depressão",
            "Plano Real", "Plano Cruzado", "hiperinflação",
            "tributação", "impostos", "PIS", "Cofins", "IRPF",
        ],
        "templates": [
            "o que é {}", "explique {}", "como funciona {}",
            "qual a importância de {}", "fale sobre {}",
        ],
    },
    "Geografia": {
        "subjects": [
            "Amazônia", "Pantanal", "Caatinga", "Cerrado", "Mata Atlântica",
            "Pampas", "Andes", "Himalaia", "Alpes", "Rocky Mountains",
            "rio Nilo", "rio Amazonas", "rio Mississippi", "rio Yangtsé",
            "deserto do Saara", "deserto do Atacama", "deserto de Gobi",
            "Oceano Pacífico", "Oceano Atlântico", "Oceano Índico",
            "China", "Índia", "Estados Unidos", "Rússia", "Canadá",
            "Austrália", "África do Sul", "Egito", "Japão", "Coreia",
            "Brasília", "São Paulo", "Rio de Janeiro", "Manaus",
            "Tóquio", "Pequim", "Mumbai", "Cairo", "Lagos",
            "Patagônia", "Mediterrâneo", "Báltico", "Cáspio",
            "Aconcágua", "Everest", "Kilimanjaro", "Fuji",
            "geografia política", "geopolítica", "fronteiras",
            "tropicais", "polos", "equador", "trópicos",
        ],
        "templates": [
            "onde fica {}", "o que é {}", "qual a capital de {}",
            "fale sobre {}", "como é {}",
        ],
    },
}

def ask(prompt):
    body = json.dumps({"prompt": prompt, "max": 80, "temp": 0.5},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(BRAIN, data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=60).read()
        d = json.loads(resp)
        return d.get("source", "?"), time.time() - t0
    except Exception:
        return "ERR", time.time() - t0

def generate_questions(domain):
    """Cross product subjects × templates → uniquequestions."""
    seeds = SEEDS.get(domain, {})
    subjects = seeds.get("subjects", [])
    templates = seeds.get("templates", [])
    questions = []
    for s in subjects:
        for t in templates:
            questions.append(t.format(s))
    return questions

def feed_domain(domain, max_questions=120):
    """Feed questions for domain. Returns counters."""
    questions = generate_questions(domain)[:max_questions]
    counters = {"brain": 0, "qwen": 0, "err": 0}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(ask, q) for q in questions]
        done = 0
        for fut in as_completed(futures):
            src, dt = fut.result()
            done += 1
            if src.startswith("brain"): counters["brain"] += 1
            elif src.startswith("qwen"): counters["qwen"] += 1
            else: counters["err"] += 1
            if done % 30 == 0:
                print(f"    [{done}/{len(questions)}] brain={counters['brain']} qwen={counters['qwen']}")
    return counters, len(questions)

def get_brain_stats():
    try:
        r = json.loads(urllib.request.urlopen(AUTOLEARN, timeout=3).read())
        return r
    except: return {}

def main():
    # Read latest IQ report
    with open("/tmp/iq_report.json") as f:
        report = json.load(f)
    weakest = report["weakest_3"]
    print(f"\n═══ GAP-FILL LOOP ═══")
    print(f"  Áreas fracas: {weakest}")
    print(f"  Coverage atual: {report['composite_coverage']:.1f}%")
    print()

    for iteration in range(1, ITER_MAX + 1):
        print(f"╔═══ ITERATION {iteration}/{ITER_MAX} ═══╗")
        for domain in weakest:
            print(f"  ▸ Feeding domain: {domain}")
            t0 = time.time()
            cnt, n = feed_domain(domain, max_questions=80)
            dt = time.time() - t0
            print(f"    Done: brain={cnt['brain']} qwen={cnt['qwen']} err={cnt['err']} ({dt:.0f}s)")
        # Wait for daemon to ingest
        print(f"  ⏳ Aguardando daemon ingest 45s...")
        time.sleep(45)
        stats = get_brain_stats()
        print(f"  📊 Brain state: cycles={stats.get('cycles',0)} dialogs={stats.get('total_dialogs_learned',0)} pending={stats.get('pending_pairs',0)}")
        print()

    # Final re-probe
    print(f"╔═══ RE-PROBE FINAL ═══╗")
    import subprocess
    subprocess.run(["python3", "/home/leonardo/projects/LLM/byte-language-lab/scripts/iq_test.py"])

if __name__ == "__main__":
    main()
