#!/usr/bin/env python3
"""distill_kg.py — distill a knowledge graph from Qwen3 into the brain.

Qwen3-4B emits factual triples; we parse, batch-validate (Qwen flags wrong
lines), dedup, and write data/kg_triples.txt for the brain's /learn_triples
endpoint to ingest.

The relations are chosen to match the brain's compose_answer templates:
  - "definicao"   → answers "o que é X"
  - "biografia"   → answers "quem foi X"
  - "autor_inv"   → answers "quem escreveu X"  (entity=obra, value=autor)
  - "localizacao" → answers "onde fica X"
  - any noun rel  → answers "qual a/o <rel> de <entity>"

Runs the generator + validator in parallel across both engines.

Usage:
  python3 distill_kg.py [--per N] [--out data/kg_triples.txt]
"""
import argparse, itertools, json, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

# Single engine — 2 iara-engine processes on one RDNA2 GPU crashed under
# concurrent load (HIP error 700). One engine is stable.
ENGINES = ["http://127.0.0.1:3000"]
MODEL = "qwen3-4b"
_eng_lock = threading.Lock()
_eng_cycle = itertools.cycle(range(len(ENGINES)))
def next_engine():
    with _eng_lock:
        return ENGINES[next(_eng_cycle)]


def post(url, payload, timeout=120):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def qwen(prompt, max_tokens=700, temp=0.2):
    chat = (f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
    try:
        d = post(f"{next_engine()}/api/generate", {
            "model": MODEL, "prompt": chat,
            "max_tokens": max_tokens, "temperature": temp,
        })
        return d.get("response", "").strip()
    except Exception as e:
        return f"<err {e}>"


# (relation, instruction). The instruction must make Qwen emit "A | B" lines
# where A is the entity and B the value for that relation.
RELATION_SPECS = [
    ("definicao", "Liste {n} conceitos/termos importantes de {domain}. Para cada um, "
                  "dê o termo e uma definição factual de UMA frase. Formato EXATO por "
                  "linha: Termo | definição. Sem numeração, sem markdown."),
    ("biografia", "Liste {n} personalidades históricas importantes de {domain}. Para "
                  "cada uma: o nome e UMA frase factual sobre quem foi e o que fez. "
                  "Formato EXATO por linha: Nome | descrição. Sem numeração."),
]
# domain-specific relation tasks (entity | value)
DOMAIN_TASKS = [
    ("capital", "geografia", "Liste {n} países e suas capitais. Formato EXATO: País | capital. Sem numeração."),
    ("moeda", "geografia", "Liste {n} países e suas moedas oficiais. Formato EXATO: País | moeda. Sem numeração."),
    ("idioma", "geografia", "Liste {n} países e seus idiomas oficiais. Formato EXATO: País | idioma. Sem numeração."),
    ("continente", "geografia", "Liste {n} países e seus continentes. Formato EXATO: País | continente. Sem numeração."),
    ("autor_inv", "literatura", "Liste {n} obras literárias célebres e seus autores. Formato EXATO: Obra | autor. Sem numeração."),
    ("inventor_inv", "tecnologia e ciência", "Liste {n} invenções/descobertas e quem as fez. Formato EXATO: Invenção | inventor. Sem numeração."),
    ("pintor_inv", "artes", "Liste {n} obras de arte célebres e seus pintores. Formato EXATO: Obra | pintor. Sem numeração."),
    ("simbolo", "química", "Liste {n} elementos químicos e seus símbolos. Formato EXATO: Elemento | símbolo. Sem numeração."),
    ("formula", "química", "Liste {n} substâncias comuns e suas fórmulas químicas. Formato EXATO: Substância | fórmula. Sem numeração."),
    ("ano", "história", "Liste {n} eventos históricos marcantes e o ano em que ocorreram. Formato EXATO: Evento | ano. Sem numeração."),
    ("funcao", "biologia", "Liste {n} órgãos/estruturas do corpo humano e suas funções (1 frase). Formato EXATO: Órgão | função. Sem numeração."),
    ("planeta", "astronomia", "Liste {n} características de planetas e corpos celestes (1 frase). Formato EXATO: Corpo celeste | característica. Sem numeração."),
]
# domains for the generic definicao/biografia specs
DOMAINS = [
    "ciência (física, química, biologia)", "história mundial", "história do Brasil",
    "geografia", "matemática", "tecnologia e computação", "literatura",
    "filosofia", "economia", "saúde e medicina", "artes e música", "astronomia",
]

_NOISE = ("u00", "the user", "```", "</think>", "<think>")


def parse_lines(raw, relation):
    """Parse 'A | B' lines into (entity, relation, value) triples."""
    out = []
    for line in raw.splitlines():
        line = line.strip().lstrip("0123456789.)- *#").strip()
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        ent, val = parts[0], parts[1]
        if not ent or len(val) < 2 or len(val) > 240:
            continue
        if any(n in ent.lower() or n in val.lower() for n in _NOISE):
            continue
        if len(ent) > 80:
            continue
        out.append((ent, relation, val))
    return out


def validate_batch(triples):
    """One Qwen call flags factually wrong lines; drop them. Cheap guard
    against Qwen3-Q4 hallucinations without per-triple cost."""
    if not triples:
        return []
    listing = "\n".join(f"{i+1}. {e} | {r} | {v}"
                        for i, (e, r, v) in enumerate(triples))
    raw = qwen(
        f"Abaixo há fatos no formato 'entidade | relação | valor'. "
        f"Identifique os NÚMEROS das linhas que estão factualmente ERRADAS "
        f"ou duvidosas. Responda só os números separados por vírgula, ou "
        f"'nenhuma' se todas corretas.\n\n{listing}",
        max_tokens=120, temp=0.0)
    bad = set()
    for m in re.findall(r"\d+", raw):
        bad.add(int(m))
    return [t for i, t in enumerate(triples) if (i + 1) not in bad]


def run_task(spec, per):
    relation, domain, instr = spec
    raw = qwen(instr.format(n=per, domain=domain), max_tokens=per * 22 + 200)
    if raw.startswith("<err"):
        return []
    triples = parse_lines(raw, relation)
    kept = validate_batch(triples)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=30, help="items per task")
    ap.add_argument("--out", default="data/kg_triples.txt")
    args = ap.parse_args()

    # build the task list: generic specs × domains + domain-specific tasks
    tasks = []
    for relation, instr in RELATION_SPECS:
        for domain in DOMAINS:
            tasks.append((relation, domain, instr))
    tasks.extend(DOMAIN_TASKS)

    # SERIAL — the iara-engine GPU backend is not thread-safe; concurrent
    # /api/generate races on shared GPU buffers → HIP error 700. One request
    # at a time is slow but stable. This is a one-time offline seeding job.
    print(f"distilling KG: {len(tasks)} tasks × ~{args.per} items, SERIAL")
    t0 = time.time()
    all_triples = []
    for done, t in enumerate(tasks, 1):
        try:
            res = run_task(t, args.per)
        except Exception:
            res = []
        all_triples.extend(res)
        if done % 5 == 0:
            print(f"  {done}/{len(tasks)} tasks  ({len(all_triples)} triples so far)",
                  flush=True)

    # dedup by (entity-lower, relation)
    seen = set()
    uniq = []
    for e, r, v in all_triples:
        key = (e.lower(), r)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((e, r, v))

    with open(args.out, "w") as f:
        f.write("# distilled KG — entity|relation|value\n")
        for e, r, v in uniq:
            f.write(f"{e}|{r}|{v}\n")

    print(f"\n╔══ KG DISTILL DONE ══╗")
    print(f"  tasks: {len(tasks)}   raw triples: {len(all_triples)}")
    print(f"  unique triples: {len(uniq)}   elapsed: {time.time()-t0:.0f}s")
    print(f"  → {args.out}")
    print(f"  ingest:  curl -X POST http://localhost:3012/learn_triples")


if __name__ == "__main__":
    main()
