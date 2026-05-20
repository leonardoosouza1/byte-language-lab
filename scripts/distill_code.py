#!/usr/bin/env python3
"""distill_code.py — distill a code corpus from Qwen3 into the brain.

Qwen3 writes code for a fixed list of common requests (HTML/CSS templates,
JS functions, CSS snippets). Output is data/code_corpus.txt in the format the
brain's /learn_code endpoint ingests:

    ===PAIR===
    <natural-language request>
    ---CODE---
    <code>

SERIAL — the iara-engine GPU backend is not thread-safe (HIP 700 under
concurrency). One request at a time. One-time offline job.

Usage:  python3 distill_code.py [--out data/code_corpus.txt]
"""
import argparse, json, re, time, urllib.request

ENGINE = "http://127.0.0.1:3000"
MODEL = "qwen3-4b"

# Each request becomes a brain dialog: USER=request, MODEL=code. Phrased the
# way a user would actually ask, so HDC retrieval matches real queries.
TEMPLATES = [
    "crie um template de landing page completa em HTML e CSS",
    "crie um site básico de uma página com HTML CSS e JavaScript",
    "crie uma página de portfólio simples em HTML e CSS",
    "crie uma barra de navegação responsiva em HTML e CSS",
    "crie um rodapé de site em HTML e CSS",
    "crie um formulário de contato estilizado em HTML e CSS",
    "crie uma seção hero com título e botão em HTML e CSS",
    "crie um card de produto em HTML e CSS",
    "crie uma grade de cards responsiva em HTML e CSS",
    "crie um menu hambúrguer responsivo com HTML CSS e JavaScript",
    "crie uma página de login em HTML e CSS",
    "crie uma tabela de preços em HTML e CSS",
    "crie um modal popup com HTML CSS e JavaScript",
    "crie um carrossel de imagens simples com HTML CSS e JavaScript",
    "crie uma página em branco responsiva com HTML5 boilerplate",
]
JS_FUNCS = [
    "uma função JavaScript que soma dois números",
    "uma função JavaScript que subtrai dois números",
    "uma função JavaScript que multiplica dois números",
    "uma função JavaScript que divide dois números com proteção contra divisão por zero",
    "uma função JavaScript que valida um endereço de email",
    "uma função JavaScript que valida um CPF",
    "uma função JavaScript que ordena um array de números",
    "uma função JavaScript que inverte uma string",
    "uma função JavaScript que calcula o fatorial de um número",
    "uma função JavaScript que gera a sequência de Fibonacci",
    "uma função JavaScript que verifica se um número é par",
    "uma função JavaScript que encontra o maior valor de um array",
    "uma função JavaScript que faz uma requisição fetch a uma API",
    "uma função JavaScript de debounce",
    "uma função JavaScript de throttle",
    "uma função JavaScript que formata uma data no padrão brasileiro",
    "uma função JavaScript que formata um número como moeda em reais",
    "uma função JavaScript que capitaliza a primeira letra de uma string",
    "uma função JavaScript que conta as palavras de um texto",
    "uma função JavaScript que remove itens duplicados de um array",
    "uma função JavaScript que gera um número aleatório entre dois valores",
    "uma função JavaScript que verifica se uma string é um palíndromo",
    "uma função JavaScript que converte Celsius para Fahrenheit",
    "uma função JavaScript que calcula o IMC dado peso e altura",
    "uma função JavaScript que adiciona um event listener de clique a um botão",
    "uma função JavaScript que alterna o modo escuro da página",
    "uma função JavaScript que faz scroll suave até um elemento",
    "uma função JavaScript que copia um texto para a área de transferência",
    "uma função JavaScript que valida se um campo de formulário está vazio",
    "uma função JavaScript que conta caracteres de um input em tempo real",
    "uma função JavaScript que filtra um array por uma condição",
    "uma função JavaScript que agrupa itens de um array por uma chave",
    "uma função JavaScript que calcula a média de um array de números",
    "uma função JavaScript que verifica se um ano é bissexto",
    "uma função JavaScript que mascara um número de telefone",
    "uma função JavaScript assíncrona que aguarda um tempo (sleep)",
    "uma função JavaScript que detecta se o dispositivo é mobile",
    "uma função JavaScript que salva e lê dados do localStorage",
    "uma função JavaScript que valida a força de uma senha",
    "uma função JavaScript que transforma um array em uma lista HTML",
]
CSS_SNIPPETS = [
    "código CSS para centralizar um elemento com flexbox",
    "código CSS para uma grade responsiva com grid",
    "código CSS para um botão moderno com hover",
    "código CSS para um card com sombra",
    "código CSS para uma animação de fade in",
    "código CSS para um fundo com gradiente",
    "código CSS para texto responsivo com clamp",
    "código CSS para um efeito de hover que aumenta a escala",
    "código CSS para uma barra de navegação fixa no topo",
    "código CSS reset básico para um projeto",
]
# v2 — classes, React, hooks, API (templates de código completos)
CLASSES = [
    "uma classe JavaScript Pessoa com nome, idade e um método apresentar",
    "uma classe JavaScript Retangulo com largura, altura e métodos de área e perímetro",
    "uma classe JavaScript ContaBancaria com saldo, depositar e sacar",
    "uma classe JavaScript Pilha (stack) com push, pop e peek",
    "uma classe JavaScript Fila (queue) com enqueue e dequeue",
    "uma classe JavaScript Animal e uma classe Cachorro que herda dela",
    "uma classe JavaScript Carrinho de compras com adicionar item e total",
    "uma classe JavaScript Temporizador com start, stop e reset",
    "uma classe JavaScript EventEmitter simples com on e emit",
    "uma classe JavaScript Validador com métodos de validar email e senha",
    "uma classe JavaScript LinkedList com append e toArray",
    "uma classe JavaScript Usuario com construtor e getters/setters",
]
REACT = [
    "um componente React funcional que exibe um título",
    "um componente React de contador com useState",
    "um componente React de lista que recebe itens via props",
    "um componente React de formulário de login controlado",
    "um componente React que busca dados de uma API com useEffect",
    "um componente React de botão reutilizável com props",
    "um componente React de card de produto",
    "um componente React de barra de navegação",
    "uma página React simples com header, conteúdo e footer",
    "um componente React de modal com abrir e fechar",
    "um componente React de toggle de tema claro/escuro",
    "um componente React de campo de busca com filtro",
    "um componente React de tabela que renderiza um array de objetos",
    "um componente React de paginação",
    "uma landing page em React com seção hero e call-to-action",
]
HOOKS = [
    "um hook React useState de exemplo comentado",
    "um hook React customizado useLocalStorage",
    "um hook React customizado useFetch para requisições",
    "um hook React customizado useToggle",
    "um hook React customizado useDebounce",
    "um hook React customizado useWindowSize",
    "um exemplo de useEffect com cleanup",
    "um hook React customizado usePrevious",
]
API = [
    "uma rota Express GET que retorna uma lista em JSON",
    "uma rota Express POST que recebe e valida um corpo JSON",
    "um servidor Express básico com uma rota de health check",
    "um wrapper de fetch para chamadas a uma API REST",
    "uma função Node.js que lê um arquivo JSON e retorna o conteúdo",
    "um middleware Express de log de requisições",
    "um middleware Express de tratamento de erros",
    "uma função que faz POST a uma API com async/await e trata erro",
    "um CRUD em memória de tarefas com Express",
    "uma função que conecta e consulta um banco com async/await",
]


def post(url, payload, timeout=120):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def qwen(prompt, max_tokens=800, temp=0.2):
    chat = (f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
    try:
        d = post(f"{ENGINE}/api/generate", {
            "model": MODEL, "prompt": chat,
            "max_tokens": max_tokens, "temperature": temp,
        })
        return d.get("response", "").strip()
    except Exception as e:
        return f"<err {e}>"


def extract_code(raw):
    """Pull code out of Qwen's answer — concatenate all ``` fenced blocks.
    A multi-file answer (HTML+CSS+JS) has several blocks; keep them all."""
    blocks = re.findall(r"```[a-zA-Z]*\n?(.*?)```", raw, re.DOTALL)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks if b.strip())
    # no fence — strip prose lines, keep the rest
    return raw.strip()


def gen_pair(request, kind):
    if kind == "template":
        instr = (f"Escreva {request}. Responda APENAS o código, completo e "
                 f"funcional, dentro de blocos ``` . Sem explicação antes ou depois.")
    elif kind == "js":
        instr = (f"Escreva {request}. Responda APENAS o código JavaScript "
                 f"dentro de um bloco ``` . Inclua um comentário curto e um "
                 f"exemplo de uso. Sem explicação em prosa.")
    else:  # css
        instr = (f"Escreva {request}. Responda APENAS o código dentro de um "
                 f"bloco ``` . Sem explicação em prosa.")
    raw = qwen(instr, max_tokens=900)
    if raw.startswith("<err"):
        return None
    code = extract_code(raw)
    if len(code) < 15 or len(code) > 7800:
        return None
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/code_corpus.txt")
    args = ap.parse_args()

    tasks = ([(f"crie {c}", "js") for c in CLASSES]
             + [(f"crie {r}", "js") for r in REACT]
             + [(f"escreva {h}", "js") for h in HOOKS]
             + [(f"crie {a}", "js") for a in API]
             + [(t, "template") for t in TEMPLATES]
             + [(f"crie {f}", "js") for f in JS_FUNCS]
             + [(c, "css") for c in CSS_SNIPPETS])

    print(f"distilling code: {len(tasks)} requests, SERIAL")
    t0 = time.time()
    pairs = []
    for i, (request, kind) in enumerate(tasks, 1):
        code = gen_pair(request, kind)
        ok = "ok" if code else "SKIP"
        if code:
            pairs.append((request, code))
        print(f"  [{i}/{len(tasks)}] {ok}  {request[:54]}", flush=True)

    with open(args.out, "w") as f:
        for request, code in pairs:
            f.write("===PAIR===\n")
            f.write(request + "\n")
            f.write("---CODE---\n")
            f.write(code + "\n")

    print(f"\n╔══ CODE DISTILL DONE ══╗")
    print(f"  requests: {len(tasks)}   pairs written: {len(pairs)}")
    print(f"  elapsed: {time.time()-t0:.0f}s   → {args.out}")
    print(f"  ingest:  curl -X POST http://localhost:3012/learn_code")


if __name__ == "__main__":
    main()
