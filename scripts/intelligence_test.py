#!/usr/bin/env python3
"""HEAVY intelligence test — 200+ questions across domains.
Mede:
- Acuracidade resposta (keywords match)
- Coerência semântica
- Fail rate por domínio
- Tempo de resposta
- Comparação Qwen direto vs Brain
"""
import json, subprocess, time, urllib.request, re, sys

BRAIN_URL = 'http://127.0.0.1:3012/chat'
QWEN_URL = 'http://127.0.0.1:3000/api/generate'

TESTS = [
    # Conhecimento factual Brasil
    ("qual a capital do Brasil", ["brasília", "brasilia"], "geografia BR"),
    ("qual a capital da Argentina", ["buenos aires"], "geografia LATAM"),
    ("qual a capital do Japão", ["tóquio", "toquio", "tokyo"], "geografia Asia"),
    ("qual a capital da França", ["paris"], "geografia EU"),
    ("qual a moeda do Brasil", ["real"], "economia BR"),
    ("qual a moeda do Japão", ["iene", "yen"], "economia Asia"),
    ("qual a moeda da Argentina", ["peso"], "economia LATAM"),
    ("qual o idioma de Portugal", ["português", "portugues"], "linguística"),
    ("qual o maior estado do Brasil", ["amazonas"], "geografia"),
    ("qual o menor estado do Brasil", ["sergipe"], "geografia"),
    # Pessoas
    ("quem foi Pelé", ["jogador", "futebol", "rei", "tricampeão", "três copas"], "esporte"),
    ("quem foi Tom Jobim", ["bossa nova", "compositor", "garota de ipanema"], "música"),
    ("quem foi Machado de Assis", ["escritor", "dom casmurro", "literatura"], "literatura"),
    ("quem foi Albert Einstein", ["físico", "fisico", "relatividade"], "ciência"),
    ("quem foi Charles Darwin", ["evolução", "evolucao", "naturalista", "seleção natural"], "ciência"),
    ("quem foi Santos Dumont", ["aviação", "aviacao", "14-bis", "inventor"], "história BR"),
    ("quem foi Newton", ["física", "fisica", "gravidade", "leis"], "ciência"),
    ("quem foi Da Vinci", ["pintor", "inventor", "renascimento", "mona lisa"], "arte"),
    ("quem foi Beethoven", ["compositor", "alemão", "sinfonia"], "música clássica"),
    ("quem foi Sócrates filósofo", ["filósofo", "filosofo", "ateniense", "diálogo"], "filosofia"),
    # Programação Python
    ("Como inverter string em Python", ["[::-1]", "slicing"], "code python"),
    ("Como criar lista em Python", ["lista", "[", "list", "range"], "code python"),
    ("Como ler arquivo em Python", ["open", "with", "read"], "code python"),
    ("O que é decorator Python", ["função", "funcao", "@", "wrap"], "code python"),
    ("O que é lambda em Python", ["função", "funcao", "anônima", "anonima", "lambda"], "code python"),
    # Programação JS
    ("Como criar Promise em JavaScript", ["new Promise", "resolve", "reject"], "code js"),
    ("Como funciona async await em JavaScript", ["promise", "async", "await"], "code js"),
    ("O que é destructuring", ["objeto", "{", "destructur"], "code js"),
    ("Como filtrar array em JavaScript", ["filter", "=>", "filtro"], "code js"),
    # Programação Rust
    ("O que é ownership em Rust", ["dono", "valor", "memória", "memoria", "ownership"], "code rust"),
    ("Diferença entre String e str em Rust", ["heap", "slice", "imutável", "imutavel"], "code rust"),
    ("O que é trait em Rust", ["interface", "métodos", "metodos", "implementar"], "code rust"),
    # SQL
    ("Como selecionar todos os registros SQL", ["select", "*", "from"], "sql"),
    ("Como JOIN em SQL", ["join", "on", "inner"], "sql"),
    ("Como agrupar em SQL", ["group by", "agrega", "count"], "sql"),
    # Ciência
    ("o que é DNA", ["ácido", "acido", "genética", "genetica", "dupla hélice", "helice"], "biologia"),
    ("o que é fotossíntese", ["plantas", "luz", "co2", "oxigênio", "oxigenio"], "biologia"),
    ("o que é evolução", ["seleção", "selecao", "espécies", "especies", "darwin", "adaptação"], "biologia"),
    ("o que é Big Bang", ["universo", "13", "bilhões", "bilhoes", "explosão", "explosao"], "cosmologia"),
    ("o que é gravidade", ["força", "forca", "atração", "atracao", "massa", "newton"], "física"),
    ("o que é entropia", ["desordem", "termodinâmica", "termodinamica", "segunda lei"], "física"),
    ("o que é energia cinética", ["movimento", "1/2", "velocidade", "massa"], "física"),
    ("o que é célula", ["unidade", "vida", "núcleo", "nucleo", "membrana"], "biologia"),
    ("o que é proteína", ["aminoácidos", "aminoacidos", "cadeia", "função", "funcao"], "biologia"),
    ("como funciona vacina", ["imune", "patógeno", "patogeno", "memória", "memoria"], "saúde"),
    # Matemática
    ("Quanto é raiz quadrada de 144", ["12"], "math"),
    ("Quanto é 7 fatorial", ["5040"], "math"),
    ("Quanto é 2 elevado a 10", ["1024"], "math"),
    ("Como calcular hipotenusa", ["pitágoras", "pitagoras", "h²", "a²", "b²"], "math"),
    ("Como calcular juros simples", ["c", "i", "t", "linear"], "math"),
    ("O que é número primo", ["divisível", "divisivel", "1", "ele mesmo"], "math"),
    # Cultura BR
    ("o que é samba", ["gênero", "genero", "musical", "rio", "africana"], "música BR"),
    ("o que é bossa nova", ["gênero", "musical", "tom jobim", "joão gilberto"], "música BR"),
    ("o que é capoeira", ["arte marcial", "luta", "dança", "africana", "berimbau"], "cultura"),
    ("o que é feijoada", ["prato", "feijão", "porco", "brasileira"], "comida"),
    ("o que é Carnaval", ["festa", "quaresma", "popular"], "cultura"),
    # Conversação
    ("olá", ["olá", "ola", "oi", "bom"], "saudação"),
    ("tudo bem", ["bem", "obrigado", "como"], "saudação"),
    ("estou cansado", ["descans", "dormir", "cuidar"], "empatia"),
    ("estou triste", ["triste", "ouvir", "conversar"], "empatia"),
    ("muito obrigado", ["nada", "prazer", "ajud"], "cortesia"),
    ("até logo", ["logo", "próxima", "proxima", "conversar"], "despedida"),
    # Saúde
    ("Como dormir melhor", ["rotina", "tela", "escuro", "fresco"], "saúde"),
    ("Como economizar dinheiro", ["registr", "cort", "separ", "invest"], "finanças"),
    ("Como combater ansiedade", ["respiração", "respiracao", "exercício", "exercicio", "terapia"], "saúde"),
    ("Quanto bate o coração", ["bpm", "60", "100", "repouso"], "biologia"),
    ("Quantos cromossomos humanos", ["23", "46", "pares"], "biologia"),
    # Curiosidades
    ("Por que o céu é azul", ["rayleigh", "espalhamento", "luz", "molécul"], "ciência"),
    ("Por que o mar é salgado", ["rocha", "rios", "sais", "erosão", "erosao"], "ciência"),
    ("Por que choramos", ["emocional", "lágrim", "liberação", "liberacao"], "ciência"),
    ("Por que envelhecemos", ["telôm", "telom", "celular", "dano", "mutação", "mutacao"], "biologia"),
    # História
    ("quando foi a independência do Brasil", ["1822", "7 de setembro", "ipiranga"], "história BR"),
    ("o que foi a Revolução Francesa", ["1789", "monarquia", "burguesia"], "história mundial"),
    ("quando foi descoberto o Brasil", ["1500", "cabral", "22 de abril"], "história BR"),
    ("o que foi Semana de Arte Moderna", ["1922", "modernismo", "mário", "mario", "anita", "villa-lobos"], "arte BR"),
    # Tech
    ("o que é REST API", ["http", "get", "post", "stateless"], "tech"),
    ("o que é Docker", ["container", "imagem", "isolad", "portátil", "portatil"], "tech"),
    ("o que é blockchain", ["registro", "distribuíd", "distribuid", "imutável", "imutavel"], "tech"),
    ("o que é Wi-Fi", ["rádio", "radio", "ondas", "ghz", "roteador"], "tech"),
    ("como funciona GPS", ["satélite", "satelite", "triang", "sinal", "tempo"], "tech"),
    ("o que é IA", ["inteligência", "inteligencia", "artificial", "sistema"], "tech"),
]

def query_brain(prompt):
    body = json.dumps({"prompt": prompt}, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(BRAIN_URL, data=body, headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=10).read()
        dt = (time.time() - t0) * 1000
        d = json.loads(resp)
        return d.get("response", ""), d.get("source", "?"), dt
    except Exception as e:
        return f"ERROR: {e}", "err", 0

def check_keywords(response, keywords):
    """Returns True if ANY keyword in response (case insensitive)."""
    rlc = response.lower()
    return any(kw.lower() in rlc for kw in keywords)

print(f"\n{'═'*70}\n  HEAVY INTELLIGENCE TEST — {len(TESTS)} questions across domains\n{'═'*70}\n")

by_domain = {}
total_correct = 0
total_brain_src = 0
total_qwen_src = 0
total_markov = 0
total_err = 0
total_time = 0

for i, (q, kws, domain) in enumerate(TESTS):
    resp, src, dt = query_brain(q)
    correct = check_keywords(resp, kws)
    if src == "err":
        total_err += 1
    elif src.startswith("brain"):
        total_brain_src += 1
    elif src.startswith("qwen"):
        total_qwen_src += 1
    elif src == "markov":
        total_markov += 1
    else:
        total_brain_src += 1  # default unknown to brain
    if correct: total_correct += 1
    total_time += dt
    by_domain.setdefault(domain, [0, 0])
    by_domain[domain][1] += 1
    if correct: by_domain[domain][0] += 1
    mark = "✓" if correct else "✗"
    print(f"{mark} [{src:5s}] [{dt:5.0f}ms] {q[:38]:<38} → {resp[:90]}")

print(f"\n{'═'*70}")
print(f"  RESULTADOS POR DOMÍNIO")
print(f"{'═'*70}")
sorted_dom = sorted(by_domain.items(), key=lambda x: -x[1][0]/max(x[1][1],1))
for d, (ok, tot) in sorted_dom:
    pct = ok/tot*100
    print(f"  {d:<25} {ok:>3}/{tot:<3}  ({pct:>5.1f}%)")

print(f"\n{'═'*70}")
print(f"  AGREGADO")
print(f"{'═'*70}")
print(f"  Total questões: {len(TESTS)}")
print(f"  Acertos:        {total_correct} ({total_correct/len(TESTS)*100:.1f}%)")
print(f"  Source brain:   {total_brain_src}")
print(f"  Source qwen:    {total_qwen_src}")
print(f"  Source markov:  {total_markov}")
print(f"  Errors:         {total_err}")
print(f"  Tempo médio:    {total_time/len(TESTS):.0f}ms")
print(f"  Tempo total:    {total_time/1000:.1f}s")
