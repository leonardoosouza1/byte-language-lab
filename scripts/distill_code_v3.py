#!/usr/bin/env python3
"""distill_code_v3.py — distill ~300 code patterns into the brain.

Wider coverage than v2: data structures, sorts, searches, strings, math,
validators, dates, arrays, objects, DOM, async, HTTP, errors, storage,
React patterns/hooks, Express/Node, HTML templates, CSS, TypeScript.

Goal: more brain-hi cache hits on real-world code requests.

SERIAL — iara-engine GPU backend not thread-safe.
Usage:  python3 scripts/distill_code_v3.py [--out data/code_corpus.txt]
"""
import argparse, json, re, time, urllib.request

ENGINE = "http://127.0.0.1:3000"
MODEL = "qwen3-4b"


def post(url, payload, timeout=120):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def qwen(prompt, max_tokens=900, temp=0.2):
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
    blocks = re.findall(r"```[a-zA-Z]*\n?(.*?)```", raw, re.DOTALL)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks if b.strip())
    return raw.strip()


# ── Pattern library, ~310 entries ──
DATA_STRUCT = [
    "uma classe JavaScript Stack com push, pop, peek e size",
    "uma classe JavaScript Queue com enqueue, dequeue e front",
    "uma classe JavaScript LinkedList com append, prepend, find e remove",
    "uma classe JavaScript Set simples (sem duplicatas) com add, has, remove",
    "uma classe JavaScript HashMap com set, get, has, delete e size",
    "uma classe JavaScript MinHeap com insert e extractMin",
    "uma classe JavaScript MaxHeap com insert e extractMax",
    "uma classe JavaScript BinaryTree com insert e busca",
    "uma classe JavaScript BST (binary search tree) com insert, contains e inOrder",
    "uma classe JavaScript Trie para autocomplete com insert e startsWith",
    "uma classe JavaScript Graph com addNode, addEdge e neighbors",
    "uma classe JavaScript CircularBuffer de tamanho fixo",
    "uma classe JavaScript PriorityQueue (fila de prioridade) com insert e dequeue",
    "uma classe JavaScript DoublyLinkedList com prepend, append e iterar bidirecional",
    "uma classe JavaScript LRUCache com put, get e capacidade máxima",
]
SORTS_SEARCH = [
    "uma função JavaScript que faz quicksort em um array",
    "uma função JavaScript que faz mergesort em um array",
    "uma função JavaScript que faz bubble sort em um array",
    "uma função JavaScript que faz insertion sort em um array",
    "uma função JavaScript que faz selection sort em um array",
    "uma função JavaScript que faz busca binária em um array ordenado",
    "uma função JavaScript que faz busca linear em um array",
    "uma função JavaScript que faz BFS em um grafo a partir de um nó",
    "uma função JavaScript que faz DFS recursiva em um grafo",
    "uma função JavaScript que faz DFS iterativa com pilha em um grafo",
    "uma função JavaScript que faz busca em largura em uma matriz",
    "uma função JavaScript que ordena um array de strings alfabeticamente",
    "uma função JavaScript que ordena um array de objetos por uma chave",
]
STRINGS = [
    "uma função JavaScript que inverte uma string",
    "uma função JavaScript que verifica se uma string é palíndromo",
    "uma função JavaScript que verifica se duas strings são anagramas",
    "uma função JavaScript que conta a frequência de cada caractere",
    "uma função JavaScript que conta as vogais de uma string",
    "uma função JavaScript que capitaliza a primeira letra de cada palavra",
    "uma função JavaScript que transforma uma string em camelCase",
    "uma função JavaScript que transforma uma string em snake_case",
    "uma função JavaScript que transforma uma string em kebab-case",
    "uma função JavaScript que remove acentos de uma string",
    "uma função JavaScript que tronca uma string com reticências",
    "uma função JavaScript que gera um slug a partir de um título",
    "uma função JavaScript que escapa HTML em uma string",
    "uma função JavaScript que conta as palavras de um texto",
    "uma função JavaScript que substring com maior frequência de caracteres únicos",
]
MATH = [
    "uma função JavaScript que calcula o MDC de dois números",
    "uma função JavaScript que calcula o MMC de dois números",
    "uma função JavaScript que verifica se um número é primo",
    "uma função JavaScript que retorna todos os primos até N (crivo de Eratóstenes)",
    "uma função JavaScript que calcula o fatorial iterativo de um número",
    "uma função JavaScript que calcula o fatorial recursivo de um número",
    "uma função JavaScript que gera Fibonacci até N com memoization",
    "uma função JavaScript que converte decimal para binário",
    "uma função JavaScript que converte binário para decimal",
    "uma função JavaScript que arredonda um número para N casas decimais",
    "uma função JavaScript que calcula a potência de um número (sem Math.pow)",
    "uma função JavaScript que calcula raiz quadrada com método de Newton",
    "uma função JavaScript que verifica se um número é perfeito",
    "uma função JavaScript que retorna a soma dos dígitos de um número",
]
VALIDATORS = [
    "uma função JavaScript que valida um email com regex",
    "uma função JavaScript que valida um CPF com dígitos verificadores",
    "uma função JavaScript que valida um CNPJ",
    "uma função JavaScript que valida um número de telefone brasileiro",
    "uma função JavaScript que valida um CEP brasileiro",
    "uma função JavaScript que valida uma URL",
    "uma função JavaScript que valida um cartão de crédito com algoritmo de Luhn",
    "uma função JavaScript que valida uma data no formato dd/mm/aaaa",
    "uma função JavaScript que valida força de senha (8+, maiúscula, número, símbolo)",
    "uma função JavaScript que valida um IPv4",
    "uma função JavaScript que valida um UUID v4",
]
DATES = [
    "uma função JavaScript que formata uma data no padrão brasileiro dd/mm/aaaa",
    "uma função JavaScript que retorna a diferença em dias entre duas datas",
    "uma função JavaScript que calcula a idade a partir da data de nascimento",
    "uma função JavaScript que verifica se uma data é fim de semana",
    "uma função JavaScript que verifica se um ano é bissexto",
    "uma função JavaScript que adiciona dias a uma data",
    "uma função JavaScript que retorna o início e o fim do mês atual",
    "uma função JavaScript que formata duração em hh:mm:ss a partir de segundos",
    "uma função JavaScript que converte timestamp Unix para data legível",
    "uma função JavaScript que retorna o nome do dia da semana em português",
]
ARRAYS = [
    "uma função JavaScript que divide um array em chunks de tamanho N",
    "uma função JavaScript que achata um array de arrays (deep flatten)",
    "uma função JavaScript que remove duplicados de um array",
    "uma função JavaScript que retorna a interseção de dois arrays",
    "uma função JavaScript que retorna a união de dois arrays",
    "uma função JavaScript que retorna a diferença entre dois arrays",
    "uma função JavaScript que faz partição de um array por predicado",
    "uma função JavaScript que faz zip de dois arrays",
    "uma função JavaScript que embaralha um array (Fisher-Yates)",
    "uma função JavaScript que retorna a moda de um array de números",
    "uma função JavaScript que retorna a mediana de um array de números",
    "uma função JavaScript que rotaciona um array N posições para a esquerda",
    "uma função JavaScript que encontra o elemento que mais aparece em um array",
    "uma função JavaScript que cria um array com N elementos preenchidos por uma função",
]
OBJECTS = [
    "uma função JavaScript que faz deep clone de um objeto",
    "uma função JavaScript que faz merge profundo de dois objetos",
    "uma função JavaScript que retorna apenas as chaves especificadas de um objeto (pick)",
    "uma função JavaScript que remove chaves especificadas de um objeto (omit)",
    "uma função JavaScript que inverte chaves e valores de um objeto",
    "uma função JavaScript que verifica se dois objetos são iguais (deep equal)",
    "uma função JavaScript que retorna as chaves aninhadas de um objeto em notação de ponto",
    "uma função JavaScript que aplica uma transformação em todos os valores de um objeto",
]
DOM = [
    "uma função JavaScript que adiciona um listener com debounce em um input",
    "uma função JavaScript que aplica throttle em uma função",
    "uma função JavaScript que cria um elemento com classes e texto e adiciona ao body",
    "uma função JavaScript que faz scroll suave até um elemento por id",
    "uma função JavaScript que copia texto para a área de transferência",
    "uma função JavaScript que detecta clique fora de um elemento",
    "uma função JavaScript que alterna uma classe em um elemento ao clicar",
    "uma função JavaScript que pega o valor de todos os inputs de um formulário",
]
ASYNC_HTTP = [
    "uma função JavaScript async que faz fetch GET e retorna JSON",
    "uma função JavaScript async que faz fetch POST com corpo JSON",
    "uma função JavaScript async que faz upload de arquivo via FormData",
    "uma função JavaScript que adiciona header Authorization Bearer no fetch",
    "uma função JavaScript async sleep com Promise e setTimeout",
    "uma função JavaScript de retry com backoff exponencial",
    "uma função JavaScript que faz Promise.all com limite de concorrência",
    "uma função JavaScript que faz race com timeout",
    "uma função JavaScript que executa promises em sequência",
    "uma função JavaScript que cria um EventEmitter simples",
]
ERRORS_STORAGE = [
    "uma classe JavaScript ValidationError que estende Error",
    "uma função JavaScript que envolve um async em try/catch e retorna [err, data]",
    "um wrapper JavaScript para localStorage com get e set tipados",
    "um wrapper JavaScript para sessionStorage com expiração",
    "uma função JavaScript que lê um cookie pelo nome",
    "uma função JavaScript que define um cookie com expiração em dias",
]
REACT_COMP = [
    "um componente React funcional que renderiza um título e props",
    "um componente React de contador com useState e botões +/-",
    "um componente React que renderiza uma lista de itens com key",
    "um componente React de formulário controlado com vários inputs",
    "um componente React de formulário de login com validação",
    "um componente React que faz fetch a uma API em useEffect e mostra loading",
    "um componente React de card com props image, title, description",
    "um componente React de modal com abrir e fechar via prop e overlay",
    "um componente React de toggle de tema claro/escuro com Context",
    "um componente React de tabela responsiva a partir de um array de objetos",
    "um componente React de barra de navegação responsiva com menu hambúrguer",
    "um componente React de pagination com props page e onChange",
    "um componente React de busca com filtro instantâneo em uma lista",
    "um componente React de Tabs com children",
    "um componente React de Tooltip que aparece no hover",
    "um componente React Lazy Image com placeholder",
    "um componente React ErrorBoundary com componentDidCatch",
    "um componente React de drag and drop de itens em uma lista",
    "um componente React de upload de arquivo com preview",
    "um componente React Carousel de imagens com setas",
    "uma página React de landing com hero, features e CTA",
    "uma página React de signup com formulário e validação",
    "uma página React de dashboard com sidebar e área de conteúdo",
    "uma página React de listagem com paginação e busca",
]
REACT_HOOKS = [
    "um hook React customizado useToggle",
    "um hook React customizado useDebounce de valor",
    "um hook React customizado useLocalStorage tipado",
    "um hook React customizado useFetch que retorna data, loading e error",
    "um hook React customizado useWindowSize",
    "um hook React customizado usePrevious",
    "um hook React customizado useOnClickOutside",
    "um hook React customizado useInterval com cleanup",
    "um hook React customizado useMediaQuery",
    "um hook React customizado useOnlineStatus",
    "um hook React customizado useEventListener",
    "um hook React customizado useKeyPress",
    "um hook React customizado useCopyToClipboard",
    "um hook React customizado useScrollPosition",
    "um exemplo de useReducer para gerenciar formulário",
    "um exemplo de useContext para tema global",
    "um exemplo de useMemo para cálculo caro",
    "um exemplo de useCallback para evitar re-render de filho",
]
EXPRESS_NODE = [
    "um servidor Express básico ouvindo na porta 3000 com rota /health",
    "uma rota Express GET /users que retorna uma lista em JSON",
    "uma rota Express POST /users que valida o corpo e retorna 201",
    "uma rota Express GET /users/:id com parâmetro de URL",
    "uma rota Express com query string ?page=1&limit=10",
    "um middleware Express de log de requisições com método, URL e tempo",
    "um middleware Express de tratamento de erros centralizado",
    "um middleware Express de autenticação via Bearer token",
    "um servidor Express servindo arquivos estáticos da pasta public",
    "um servidor Express com CORS habilitado",
    "uma rota Express de upload de arquivo com multer",
    "um middleware Express de rate limiting simples em memória",
    "um CRUD em memória de tarefas com Express (GET/POST/PUT/DELETE)",
    "uma função Node.js que lê um arquivo JSON e retorna o conteúdo",
    "uma função Node.js que escreve um JSON em disco com pretty-print",
    "uma função Node.js que lê linhas de um arquivo grande com stream",
    "uma função Node.js que executa um comando shell e retorna stdout",
    "um cliente Node.js que faz uma requisição HTTP com axios",
    "uma conexão simples a Postgres com pg e uma query SELECT",
    "uma conexão simples a MongoDB com mongoose e um modelo User",
]
HTML_TEMPLATES = [
    "um template HTML5 básico com meta viewport e charset",
    "uma página HTML de landing page de restaurante",
    "uma página HTML de portfolio simples",
    "uma página HTML de página em construção",
    "um formulário HTML de contato com nome, email e mensagem",
    "um formulário HTML de signup com validação HTML5",
    "uma tabela HTML responsiva com cabeçalho fixo",
    "um card HTML com imagem, título e botão",
    "uma seção hero HTML com título grande e CTA",
    "uma navbar HTML responsiva com logo e links",
    "um footer HTML com copyright e redes sociais",
    "uma página HTML de pricing com 3 planos lado a lado",
    "uma página HTML de blog post com sidebar",
    "uma página HTML de galeria de imagens responsiva",
]
CSS = [
    "código CSS para centralizar um elemento com flexbox",
    "código CSS para centralizar um elemento com grid",
    "código CSS para uma grade responsiva de cards",
    "código CSS para um botão moderno com hover e estados",
    "código CSS para um card com sombra e hover lift",
    "código CSS para uma animação de fade in suave",
    "código CSS para uma animação de slide in da esquerda",
    "código CSS para uma animação de bounce em loop",
    "código CSS para texto responsivo com clamp",
    "código CSS para uma barra de navegação fixa no topo",
    "código CSS reset básico para um projeto",
    "código CSS para um spinner de loading só com CSS",
    "código CSS para uma toggle switch só com CSS",
    "código CSS para skeleton loading com shimmer",
    "código CSS para scrollbar customizada",
    "código CSS para dark mode com prefers-color-scheme",
    "código CSS de media queries para mobile, tablet e desktop",
    "código CSS para um overlay modal com backdrop blur",
    "código CSS para gradiente de fundo animado",
]
TYPESCRIPT = [
    "uma interface TypeScript User com campos id, name, email e opcionais",
    "um tipo TypeScript que faz Partial e Omit ao mesmo tempo",
    "um type guard TypeScript que verifica se um valor é uma string",
    "uma função TypeScript genérica que recebe T e retorna T[]",
    "um enum TypeScript de cores e uso em uma função",
    "um tipo TypeScript Readonly recursivo (DeepReadonly)",
    "uma classe TypeScript com generics para um Repositório<T>",
    "um decorator TypeScript de log que envolve um método",
]


def gen_pair(request, kind="js"):
    if kind == "html":
        instr = (f"Escreva {request}. Responda APENAS o código HTML/CSS dentro de "
                 f"um bloco ``` . Sem explicação em prosa.")
    elif kind == "css":
        instr = (f"Escreva {request}. Responda APENAS o código CSS dentro de "
                 f"um bloco ``` . Sem explicação em prosa.")
    else:
        instr = (f"Escreva {request}. Responda APENAS o código dentro de um "
                 f"bloco ``` . Inclua um comentário curto e um exemplo de uso "
                 f"quando apropriado. Sem explicação em prosa.")
    raw = qwen(instr, max_tokens=1000)
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

    groups = [
        ("js", DATA_STRUCT + SORTS_SEARCH + STRINGS + MATH + VALIDATORS + DATES
                + ARRAYS + OBJECTS + DOM + ASYNC_HTTP + ERRORS_STORAGE
                + REACT_COMP + REACT_HOOKS + EXPRESS_NODE + TYPESCRIPT),
        ("html", HTML_TEMPLATES),
        ("css", CSS),
    ]
    tasks = [(f"crie {r}", k) for k, group in groups for r in group]
    print(f"distill v3: {len(tasks)} patterns, SERIAL\n")

    t0 = time.time()
    pairs = []
    for i, (request, kind) in enumerate(tasks, 1):
        code = gen_pair(request, kind)
        ok = "ok" if code else "SKIP"
        if code:
            pairs.append((request, code))
        elapsed = time.time() - t0
        rate = i / max(elapsed, 0.1)
        eta = (len(tasks) - i) / max(rate, 0.001)
        print(f"  [{i:>3}/{len(tasks)}] {ok}  {request[:48]:<48s}  eta {eta/60:.1f}min",
              flush=True)

    with open(args.out, "w") as f:
        for request, code in pairs:
            f.write("===PAIR===\n")
            f.write(request + "\n")
            f.write("---CODE---\n")
            f.write(code + "\n")

    print(f"\n╔══ CODE DISTILL v3 DONE ══╗")
    print(f"  patterns: {len(tasks)}   pairs: {len(pairs)}")
    print(f"  elapsed: {(time.time()-t0)/60:.1f}min   → {args.out}")
    print(f"  ingest:  curl -X POST http://localhost:3012/learn_code")


if __name__ == "__main__":
    main()
