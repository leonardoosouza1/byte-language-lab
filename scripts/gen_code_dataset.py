#!/usr/bin/env python3
"""Generate multi-language code Q&A pairs.
Languages: Python, JavaScript, Rust, SQL, Bash, HTML
Output: data/code_dataset.txt em formato USER:/MODEL: pairs
"""
import random

PYTHON_QA = [
    ("Como inverter uma string em Python?",
     "Use slicing: s[::-1]. Exemplo: 'hello'[::-1] retorna 'olleh'."),
    ("Como ler um arquivo em Python?",
     "Use with open('file.txt') as f: content = f.read(). O with garante fechamento."),
    ("Como verificar se um número é par em Python?",
     "Use n % 2 == 0. Retorna True para pares, False para ímpares."),
    ("Como criar uma lista de números de 0 a 9 em Python?",
     "Use list(range(10)) ou [i for i in range(10)]."),
    ("Como converter string para inteiro em Python?",
     "Use int(s). Se a string não for numérica, lança ValueError."),
    ("Como ordenar uma lista em Python?",
     "Use sorted(lst) para nova lista ou lst.sort() in-place."),
    ("Como contar ocorrências em uma lista Python?",
     "Use lst.count(item) ou collections.Counter(lst) para todas frequências."),
    ("Como abrir URL com requests Python?",
     "import requests; r = requests.get(url); print(r.text). Instale via pip install requests."),
    ("Como criar um dicionário em Python?",
     "Use chaves: d = {'a': 1, 'b': 2}. Ou dict() construtor: dict(a=1, b=2)."),
    ("Como iterar dict em Python?",
     "for k, v in d.items(): print(k, v). Para só chaves: for k in d."),
    ("Como fazer list comprehension Python?",
     "[expr for item in iter if cond]. Exemplo: [x*2 for x in range(5) if x%2==0]."),
    ("Como tratar exceção em Python?",
     "try: ... except ValueError as e: print(e). Use finally para cleanup."),
    ("Como funciona lambda em Python?",
     "lambda args: expr. Função anônima. Ex: f = lambda x: x*2. f(3) == 6."),
    ("Como instalar pacote Python?",
     "pip install nome_pacote. Para projeto: pip install -r requirements.txt."),
    ("O que é decorator Python?",
     "Função que recebe outra função e retorna nova. Sintaxe @decorator antes da def."),
]

JS_QA = [
    ("Como inverter string em JavaScript?",
     "Use s.split('').reverse().join(''). Cria array, inverte, junta de volta."),
    ("Como verificar se array tem item em JavaScript?",
     "arr.includes(item) retorna true/false. Ou arr.indexOf(item) >= 0."),
    ("O que é async/await em JavaScript?",
     "Sintaxe para Promises. async marca função, await pausa até Promise resolver."),
    ("Como filtrar array em JavaScript?",
     "arr.filter(x => condicao). Retorna novo array com itens que passam."),
    ("Como criar Promise em JavaScript?",
     "new Promise((resolve, reject) => { ... }). Resolve sucesso, reject erro."),
    ("Como funciona this em JavaScript?",
     "Refere ao contexto de execução. Em arrow function herda do escopo léxico."),
    ("Como debounce em JavaScript?",
     "function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }"),
    ("O que é destructuring JavaScript?",
     "const {a, b} = obj. Extrai propriedades em variáveis. Funciona com arrays também."),
    ("Como evitar callback hell?",
     "Use Promises com .then() encadeado ou async/await para código sequencial e claro."),
    ("Diferença entre let const var?",
     "var hoisted function-scope, let block-scope mutável, const block-scope imutável."),
    ("Como funciona spread operator?",
     "... expande iteráveis. Ex: [...arr1, ...arr2] junta arrays. f(...args) chama com args."),
    ("Como fazer fetch em JavaScript?",
     "fetch(url).then(r => r.json()).then(data => console.log(data)). Async retorna Promise."),
    ("O que é closure JavaScript?",
     "Função que lembra escopo onde foi criada. Acessa variáveis externas mesmo após retorno."),
]

RUST_QA = [
    ("O que é ownership em Rust?",
     "Cada valor tem um dono. Quando dono sai de escopo valor é dropado. Garante memória segura sem GC."),
    ("Como evitar borrow checker erros em Rust?",
     "Use referências & em vez de mover. Clone explicitamente quando precisar. Use Rc/Arc para múltiplos donos."),
    ("Diferença entre String e &str em Rust?",
     "String é heap-allocated owned. &str é slice imutável referência. Use String para mutar, &str para parametros."),
    ("O que é trait em Rust?",
     "Interface compartilhada. Define métodos que tipos podem implementar. Equivalente a interface/typeclass."),
    ("Como tratar erro em Rust?",
     "Result<T, E>. Use ? para propagar. match para branches. Box<dyn Error> para erros genéricos."),
    ("O que é lifetime em Rust?",
     "Anotação que indica escopo de referência. 'a no parâmetro garante que referência viva por pelo menos 'a."),
    ("Como criar trait object em Rust?",
     "Box<dyn Trait> ou &dyn Trait. Dispatch dinâmico em runtime, custo de vtable lookup."),
    ("O que é cargo em Rust?",
     "Gerenciador de pacotes e build system oficial. cargo build/run/test/add. Usa Cargo.toml."),
    ("Como evitar clone excessivo em Rust?",
     "Use borrows & sempre que possível. Move quando precisar consumir. Cow<T> para clone-on-write."),
    ("O que é macro em Rust?",
     "Metaprogramação. macro_rules! ou proc-macros. Expansão em compile-time, sem custo runtime."),
]

SQL_QA = [
    ("Como selecionar todos os registros?",
     "SELECT * FROM tabela. Retorna todas as colunas e linhas."),
    ("Como filtrar registros em SQL?",
     "Use WHERE: SELECT * FROM users WHERE age > 18. Combina com AND/OR."),
    ("Como ordenar resultados SQL?",
     "ORDER BY coluna ASC ou DESC. Ex: SELECT * FROM users ORDER BY name ASC."),
    ("Como fazer JOIN em SQL?",
     "SELECT a.*, b.* FROM a INNER JOIN b ON a.id = b.a_id. Outros: LEFT, RIGHT, FULL."),
    ("Como agrupar em SQL?",
     "GROUP BY coluna com agregações: COUNT, SUM, AVG, MAX, MIN. Ex: SELECT cat, COUNT(*) FROM products GROUP BY cat."),
    ("Como criar tabela SQL?",
     "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100), age INT). Define schema."),
    ("Como atualizar registro SQL?",
     "UPDATE tabela SET coluna = valor WHERE condicao. Sem WHERE atualiza todas linhas."),
    ("Diferença entre INNER e LEFT JOIN?",
     "INNER só retorna matches em ambos lados. LEFT retorna todos do esquerdo + matches do direito (NULL se sem match)."),
]

BASH_QA = [
    ("Como listar arquivos em bash?",
     "ls -la mostra todos arquivos com detalhes. ls -lh para tamanhos humanos."),
    ("Como buscar texto em arquivos?",
     "grep 'padrao' arquivo. Recursivo: grep -r 'padrao' diretorio. Use -i para case insensitive."),
    ("Como redirecionar saída em bash?",
     "comando > arquivo sobrescreve. >> anexa. 2> redireciona stderr. 2>&1 junta stderr com stdout."),
    ("Como executar comando em loop bash?",
     "for i in 1 2 3; do echo $i; done. Ou while [ cond ]; do ...; done."),
    ("Como criar alias em bash?",
     "alias nome='comando'. Adicione em ~/.bashrc para persistir. Recarregue com source ~/.bashrc."),
    ("Como ver processos em bash?",
     "ps aux para todos. ps -ef alternativa. top ou htop para monitor live."),
    ("Como matar processo em bash?",
     "kill PID. kill -9 PID força. pkill -f padrao mata por nome."),
]

HTML_QA = [
    ("Como criar link em HTML?",
     "<a href='url'>texto</a>. Atributo target='_blank' abre em nova aba."),
    ("Como inserir imagem em HTML?",
     "<img src='caminho.jpg' alt='descricao'>. Alt é importante para acessibilidade."),
    ("Como criar formulário HTML?",
     "<form action='url' method='post'><input type='text' name='campo'><button type='submit'>OK</button></form>"),
    ("Como criar lista HTML?",
     "<ul><li>item</li></ul> não ordenada. <ol>...</ol> ordenada. <li> para cada item."),
    ("Diferença entre div e span?",
     "div é bloco, ocupa linha inteira. span é inline, fica no fluxo do texto."),
    ("Como criar tabela HTML?",
     "<table><tr><th>Header</th></tr><tr><td>Celula</td></tr></table>. tr=linha, th=header, td=dado."),
]

MATH_QA = [
    ("Quanto é 2 elevado a 10?", "2^10 = 1024."),
    ("Quanto é raiz quadrada de 144?", "√144 = 12."),
    ("Quanto é 7 fatorial?", "7! = 5040."),
    ("Quanto é PI aproximado?", "π ≈ 3.14159, irracional não-periódico."),
    ("Como calcular área do círculo?", "A = π * r². r é o raio."),
    ("Como calcular hipotenusa?", "h² = a² + b². Teorema de Pitágoras."),
    ("Quanto é e (Euler)?", "e ≈ 2.71828, base do logaritmo natural."),
    ("Como derivar x²?", "Derivada de x² é 2x. Regra da potência: nx^(n-1)."),
    ("Quanto é log base 10 de 1000?", "log₁₀(1000) = 3, pois 10³ = 1000."),
    ("O que é número primo?", "Inteiro maior que 1 divisível apenas por 1 e ele mesmo. Ex: 2, 3, 5, 7, 11."),
]

GEO_QA = [
    ("Qual a capital da França?", "Paris é a capital da França, banhada pelo rio Sena."),
    ("Qual a capital da Inglaterra?", "Londres é a capital, atravessada pelo rio Tâmisa."),
    ("Qual a capital da Itália?", "Roma, conhecida como Cidade Eterna, é a capital italiana."),
    ("Qual a capital da Alemanha?", "Berlim é a capital alemã desde a reunificação em 1990."),
    ("Qual a capital da Rússia?", "Moscou é a capital, com a Praça Vermelha e o Kremlin."),
    ("Qual a capital dos Estados Unidos?", "Washington D.C. é a capital, sede dos três poderes."),
    ("Qual a capital da China?", "Pequim (Beijing) é a capital chinesa."),
    ("Qual a capital do Egito?", "Cairo é a capital, próxima às pirâmides de Gizé."),
    ("Qual o maior país do mundo?", "Rússia, com cerca de 17 milhões de km²."),
    ("Qual o menor país do mundo?", "Vaticano, com apenas 0.44 km²."),
]

SCIENCE_QA = [
    ("O que é energia cinética?", "Energia do movimento. K = ½mv². m=massa, v=velocidade."),
    ("O que é entropia?", "Medida de desordem em sistema. Segunda lei termodinâmica: entropia tende a aumentar."),
    ("O que é mitose?", "Divisão celular que gera duas células-filhas idênticas à mãe."),
    ("O que é meiose?", "Divisão celular reducional que produz gametas com metade dos cromossomos."),
    ("O que é evolução por seleção natural?", "Indivíduos mais adaptados sobrevivem e transmitem genes. Proposto por Darwin em 1859."),
    ("O que é átomo?", "Unidade básica da matéria, composto por prótons, nêutrons e elétrons."),
    ("O que é molécula?", "Conjunto de átomos ligados por ligações químicas. Ex: H2O é molécula de água."),
    ("O que é eletricidade?", "Fluxo de elétrons através de condutor. Medida em ampères."),
    ("O que é radiação?", "Emissão de energia em forma de ondas ou partículas. Pode ser eletromagnética ou nuclear."),
    ("O que é campo magnético?", "Região do espaço onde força magnética atua. Gerado por corrente elétrica ou ímãs."),
]

CULTURE_QA = [
    ("Quem foi Pablo Picasso?", "Pintor espanhol, co-fundador do cubismo. Obras: Guernica, Les Demoiselles d'Avignon."),
    ("Quem foi Leonardo da Vinci?", "Pintor, inventor, cientista italiano renascentista. Obras: Mona Lisa, Última Ceia."),
    ("Quem foi Michelangelo?", "Escultor e pintor italiano. Capela Sistina, Davi, Pietà."),
    ("Quem foi Vincent van Gogh?", "Pintor pós-impressionista holandês. Noite Estrelada, Girassóis. Vendeu apenas uma obra em vida."),
    ("Quem foi Frida Kahlo?", "Pintora mexicana, símbolo do surrealismo e feminismo. Autoretratos icônicos."),
    ("Quem foi Tarsila do Amaral?", "Pintora brasileira modernista. Abaporu inaugurou o Movimento Antropofágico."),
    ("Quem foi Cândido Portinari?", "Pintor brasileiro. Painéis Guerra e Paz na ONU, série Os Retirantes."),
    ("O que é Tropicália?", "Movimento cultural brasileiro dos anos 60. Caetano, Gil, Tom Zé, Mutantes."),
    ("O que é Bossa Nova?", "Gênero musical brasileiro dos anos 50. João Gilberto, Tom Jobim. Garota de Ipanema mundial."),
    ("O que é Carnaval?", "Festa popular antes da Quaresma. Brasil, especialmente Rio e Salvador, é referência mundial."),
]

ALL = []
for cat in [PYTHON_QA, JS_QA, RUST_QA, SQL_QA, BASH_QA, HTML_QA, MATH_QA, GEO_QA, SCIENCE_QA, CULTURE_QA]:
    ALL.extend(cat)

random.seed(7)
random.shuffle(ALL)

out_path = 'data/code_dataset.txt'
with open(out_path, 'w', encoding='utf-8') as f:
    for u, m in ALL:
        f.write(f'USER: {u}\nMODEL: {m}\n\n')

import os
print(f'Generated {len(ALL)} Q&A pairs across 10 domains')
print(f'Domains: Python({len(PYTHON_QA)}) JS({len(JS_QA)}) Rust({len(RUST_QA)}) SQL({len(SQL_QA)}) Bash({len(BASH_QA)}) HTML({len(HTML_QA)}) Math({len(MATH_QA)}) Geo({len(GEO_QA)}) Science({len(SCIENCE_QA)}) Culture({len(CULTURE_QA)})')
print(f'Output: {out_path} ({os.path.getsize(out_path)} bytes)')
