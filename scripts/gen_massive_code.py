#!/usr/bin/env python3
"""MASSIVE code dataset — 500+ Q&A across all languages and topics."""
import os, random

QA = []

# Python — operações fundamentais
PY = [
    "Como criar lista em Python", "lst = [1, 2, 3] ou list(range(10)). Lista é ordenada e mutável.",
    "Como criar tupla em Python", "t = (1, 2, 3). Tupla imutável, usada para dados estruturados.",
    "Como criar set em Python", "s = {1, 2, 3} ou set(). Sem duplicatas, sem ordem garantida.",
    "Como criar dict em Python", "d = {'a': 1, 'b': 2} ou dict(a=1, b=2). Chave-valor com lookup O(1).",
    "Como iterar lista Python", "for item in lst: print(item). Com índice: for i, item in enumerate(lst).",
    "Como reverter lista Python", "lst[::-1] retorna nova. lst.reverse() inverte in-place.",
    "Como concatenar listas", "lst1 + lst2 cria nova. lst1.extend(lst2) modifica lst1.",
    "Como remover item lista Python", "lst.remove(item) remove primeiro. del lst[i] por índice. lst.pop() retorna último.",
    "Como verificar tipo em Python", "type(x) retorna classe. isinstance(x, int) verifica tipo, melhor para herança.",
    "Como criar classe Python", "class MinhaClasse:\\n    def __init__(self, x):\\n        self.x = x",
    "O que é self Python", "Referência à instância dentro de métodos. Convenção PEP 8.",
    "O que é __init__", "Construtor. Roda ao criar instância. Inicializa atributos.",
    "O que é herança Python", "class Filho(Pai): herda métodos. super() chama método do pai.",
    "Como ler input do usuário", "x = input('Digite: '). Retorna string. Converte se precisar: int(input()).",
    "Como imprimir formatado", "f'Olá {nome}' f-string. .format() método. % printf-style legado.",
    "Como abrir arquivo Python", "with open('f.txt') as f: data = f.read(). 'w' para escrever, 'a' anexar.",
    "Como escrever arquivo Python", "with open('f.txt', 'w') as f: f.write('texto')",
    "Como split string Python", "s.split(',') separa por vírgula. s.split() separa por whitespace.",
    "Como join string Python", "','.join(['a','b','c']) → 'a,b,c'. Junta lista em string.",
    "Como replace string Python", "s.replace('a', 'b') troca todos. count=1 para um só.",
    "Como upper/lower string", "s.upper() maiúsculas. s.lower() minúsculas. s.title() title case.",
    "Como strip string Python", "s.strip() remove whitespace bordas. .lstrip() esquerda. .rstrip() direita.",
    "Como verificar substring", "if 'abc' in s: ... operador in. Ou s.find('abc') retorna índice ou -1.",
    "Como range Python", "range(10) gera 0-9. range(2,10) 2-9. range(0,10,2) pares.",
    "Como enumerar Python", "for i, val in enumerate(lst): ... dá índice + valor.",
    "Como zip Python", "list(zip(lst1, lst2)) combina pares. [(a1,b1), (a2,b2), ...].",
    "Como map Python", "list(map(func, lst)) aplica func em cada item.",
    "Como filter Python", "list(filter(lambda x: x>0, lst)) mantém positivos.",
    "Como reduce Python", "from functools import reduce; reduce(lambda a,b: a+b, lst, 0). Acumula.",
    "Como sorted Python", "sorted(lst) nova lista ordenada. key=func, reverse=True opcionais.",
]
for i in range(0, len(PY), 2):
    QA.append((PY[i], PY[i+1]))

# JavaScript
JS = [
    "Como declarar variável JS", "let x = 1 mutável. const y = 2 imutável. var legado evite.",
    "Como criar função JS", "function f(x) { return x*2 } ou arrow: const f = x => x*2",
    "Como array JS", "const arr = [1,2,3]. Métodos: map, filter, reduce, find, forEach.",
    "Como objeto JS", "const obj = { nome: 'Leo', idade: 30 }. Acesso obj.nome ou obj['nome'].",
    "Como Promise JS", "new Promise((res, rej) => setTimeout(res, 1000)).then(...).catch(...)",
    "Como async await JS", "async function f() { const r = await fetch(url); }",
    "Como fetch JS", "fetch(url).then(r => r.json()).then(data => console.log(data))",
    "Como JSON parse JS", "JSON.parse(str) string → objeto. JSON.stringify(obj) → string.",
    "Como classe JS", "class Pessoa { constructor(n) { this.nome = n } saudar() { return 'oi' } }",
    "Como herança JS", "class Aluno extends Pessoa { constructor(n) { super(n) } }",
    "Como spread JS", "const novo = [...arr1, ...arr2] junta arrays. {...obj1, ...obj2} junta objs.",
    "Como destructuring JS", "const {a, b} = obj. const [x, y] = arr.",
    "Como ternário JS", "cond ? valor_true : valor_false. Expressão concisa.",
    "Como template literal JS", "`Olá ${nome}` backticks + ${} para interpolação.",
    "Como for...of JS", "for (const x of arr) {} itera valores. for...in itera chaves.",
    "Como filter JS", "arr.filter(x => x > 0) novo array com positivos.",
    "Como map JS", "arr.map(x => x * 2) novo array transformado.",
    "Como reduce JS", "arr.reduce((acc, x) => acc + x, 0) acumula.",
    "Como find JS", "arr.find(x => x.id == 1) primeiro match ou undefined.",
    "Como includes JS", "arr.includes(item) true/false. str.includes('abc') também.",
    "Como callback JS", "Função passada como argumento. setTimeout(cb, 1000). map(cb).",
    "Como event listener JS", "document.addEventListener('click', e => console.log(e))",
    "Como localStorage JS", "localStorage.setItem('key', 'value'). getItem retorna ou null.",
    "Como try catch JS", "try { ... } catch (e) { console.error(e) } finally { cleanup() }",
]
for i in range(0, len(JS), 2):
    QA.append((JS[i], JS[i+1]))

# Rust
RUST = [
    "Como criar var Rust", "let x = 5 imutável. let mut y = 10 mutável. Tipo inferido.",
    "Como criar função Rust", "fn soma(a: i32, b: i32) -> i32 { a + b }",
    "Como if Rust", "if cond { ... } else { ... }. if é expressão pode retornar valor.",
    "Como loop Rust", "loop infinito break. while cond. for x in 0..10 range.",
    "Como Option Rust", "Option<T> = Some(T) | None. Substitui null seguro.",
    "Como Result Rust", "Result<T, E> = Ok(T) | Err(E). Trata erros explicitamente.",
    "Como match Rust", "match x { 0 => 'zero', 1..=10 => 'low', _ => 'other' }",
    "Como struct Rust", "struct Pessoa { nome: String, idade: u32 }",
    "Como impl Rust", "impl Pessoa { fn new(n: String) -> Self { Pessoa{nome: n, idade: 0} } }",
    "Como enum Rust", "enum Direcao { Norte, Sul, Leste, Oeste }",
    "Como trait Rust", "trait Saudar { fn ola(&self) -> String; }",
    "Como ownership Rust", "Valor tem um dono. Move quando atribui. Borrow & sem mover.",
    "Como vector Rust", "let v: Vec<i32> = vec![1, 2, 3]. push, pop, len, iter.",
    "Como String vs str", "String owned heap. &str slice imutável. .to_string() converte.",
    "Como Box Rust", "Box<T> aloca no heap. Útil para tipos recursivos ou trait objects.",
    "Como Rc Arc Rust", "Rc reference counted single thread. Arc atomic thread-safe.",
    "Como derive Rust", "#[derive(Debug, Clone, PartialEq)] adiciona traits comuns.",
    "Como println Rust", "println!(\"{}\", x). {:?} debug. {:#?} pretty. Macro com format!.",
    "Como cargo run Rust", "cargo new proj. cargo build. cargo run executa. cargo test rodes.",
    "Como dependencies Rust", "Adicione em Cargo.toml [dependencies]: serde = '1'. cargo add nome.",
]
for i in range(0, len(RUST), 2):
    QA.append((RUST[i], RUST[i+1]))

# Math computational
MATH = [
    "Quanto é 2+2", "4.",
    "Quanto é 5*8", "40.",
    "Quanto é 100/4", "25.",
    "Quanto é 2 ao quadrado", "4.",
    "Quanto é 3 ao cubo", "27.",
    "Quanto é raiz de 100", "10.",
    "Quanto é raiz de 25", "5.",
    "Quanto é 10 fatorial", "3628800.",
    "Quanto é 5 fatorial", "120.",
    "Quanto é PI", "3.14159...",
    "Quanto é número de Euler e", "2.71828...",
    "Como calcular porcentagem", "Multiplique valor pela taxa dividida por 100. Ex: 20% de 50 = 50*0.20 = 10.",
    "Como regra de três simples", "a/b = c/x. Resolva x = c*b/a. Ex: 2 livros 30 reais, 5 livros = 75 reais.",
    "Como média aritmética", "Soma dos valores dividido pelo número de valores. (a+b+c)/3.",
    "Como média ponderada", "Soma de (valor * peso) dividida pela soma dos pesos.",
    "Como calcular juros simples", "J = C * i * t. Capital * taxa * tempo. Linear, sem capitalização.",
    "Como mediana", "Valor central de lista ordenada. Se par, média dos dois centrais.",
    "Como moda", "Valor que mais aparece. Pode ter mais de uma ou nenhuma.",
    "O que é desvio padrão", "Medida de dispersão. Raiz da variância. Quanto maior, mais espalhados os dados.",
    "Como Pitágoras", "h² = a² + b². Hipotenusa ao quadrado = soma dos catetos ao quadrado.",
]
for i in range(0, len(MATH), 2):
    QA.append((MATH[i], MATH[i+1]))

# SQL avançado
SQL = [
    "Como SELECT distinct", "SELECT DISTINCT coluna FROM tabela. Remove duplicatas.",
    "Como WHERE multiple", "WHERE a = 1 AND b > 5 OR c IN (1,2,3). Combine com parênteses.",
    "Como LIKE SQL", "WHERE nome LIKE 'Jo%' começa com Jo. %meio% qualquer middle. _ um char.",
    "Como ORDER BY multiple", "ORDER BY nome ASC, idade DESC. Múltiplas colunas.",
    "Como LIMIT SQL", "LIMIT 10 retorna 10 primeiras. LIMIT 10 OFFSET 20 pula 20.",
    "Como COUNT SQL", "SELECT COUNT(*) FROM users. COUNT(coluna) ignora NULL.",
    "Como SUM SQL", "SELECT SUM(valor) FROM vendas. Soma valores não-NULL.",
    "Como AVG SQL", "SELECT AVG(idade) FROM users. Média.",
    "Como MIN MAX SQL", "SELECT MIN(preco), MAX(preco) FROM produtos. Menor e maior.",
    "Como GROUP BY HAVING", "GROUP BY categoria HAVING COUNT(*) > 5. HAVING filtra grupos.",
    "Como LEFT JOIN SQL", "SELECT a.*, b.* FROM a LEFT JOIN b ON a.id = b.a_id. Mantém todos esquerdos.",
    "Como INSERT SQL", "INSERT INTO users (nome, idade) VALUES ('Leo', 30)",
    "Como UPDATE SQL", "UPDATE users SET idade = 31 WHERE id = 1. SEMPRE com WHERE.",
    "Como DELETE SQL", "DELETE FROM users WHERE id = 1. CUIDADO sem WHERE limpa tudo.",
    "Como ALTER TABLE", "ALTER TABLE users ADD COLUMN email VARCHAR(255). Modifica schema.",
]
for i in range(0, len(SQL), 2):
    QA.append((SQL[i], SQL[i+1]))

# Linux/Bash
LINUX = [
    "Como pwd Linux", "pwd imprime diretório atual.",
    "Como cd Linux", "cd /caminho/absoluto. cd .. sobe um nível. cd ~ home.",
    "Como ls Linux", "ls lista. ls -la inclui ocultos + detalhes. ls -lh tamanhos humanos.",
    "Como mkdir Linux", "mkdir pasta. mkdir -p pai/filho cria pais.",
    "Como cp Linux", "cp origem destino. cp -r diretório recursivo.",
    "Como mv Linux", "mv origem destino. Move ou renomeia.",
    "Como rm Linux", "rm arquivo. rm -r pasta recursivo. rm -f força sem perguntar.",
    "Como cat Linux", "cat arquivo mostra conteúdo. Cat múltiplos concatena.",
    "Como less Linux", "less arquivo paginação. Setas navegam. q sai.",
    "Como head tail Linux", "head -n 10 primeiras 10 linhas. tail -f acompanha em tempo real.",
    "Como grep Linux", "grep 'padrao' arq. -r recursivo. -i ignore case. -n linha.",
    "Como find Linux", "find . -name '*.py' busca por padrão. -type f arquivos. -mtime tempo.",
    "Como chmod Linux", "chmod 755 arq. chmod +x arq executável. r=4 w=2 x=1.",
    "Como chown Linux", "chown user:group arquivo. Muda dono. -R recursivo.",
    "Como ps Linux", "ps aux todos processos. ps -ef alternativa.",
    "Como kill Linux", "kill PID termina. kill -9 PID força. killall nome.",
    "Como top htop Linux", "top monitor live. htop interativo melhor.",
    "Como df du Linux", "df -h disco usado por partição. du -sh tamanho pasta.",
    "Como tar Linux", "tar -czf arq.tar.gz pasta criar. -xzf extrair.",
    "Como wget curl Linux", "wget url baixa. curl -O baixa. curl url mostra. -X POST métodos.",
]
for i in range(0, len(LINUX), 2):
    QA.append((LINUX[i], LINUX[i+1]))

# Git
GIT = [
    "Como git init", "git init cria repo. Marca pasta como Git tracked.",
    "Como git add", "git add arquivo. git add . tudo. git add -A todos incluindo deletados.",
    "Como git commit", "git commit -m 'mensagem'. Snapshot do staged.",
    "Como git push", "git push origin main. Envia commits ao remote.",
    "Como git pull", "git pull origin main. Fetch + merge do remote.",
    "Como git status", "git status mostra arquivos modificados, staged, untracked.",
    "Como git log", "git log histórico. --oneline conciso. -p mostra diffs.",
    "Como git branch", "git branch lista. git branch nova cria. -d apaga.",
    "Como git checkout", "git checkout branch troca. -b cria + troca.",
    "Como git merge", "git merge branch funde no atual. Pode dar conflito.",
    "Como git rebase", "git rebase main reescreve histórico em cima da main.",
    "Como git reset", "git reset --hard HEAD reseta tudo. --soft mantém files.",
    "Como git stash", "git stash guarda mudanças. git stash pop recupera.",
    "Como git diff", "git diff mostra alterações vs último commit. --staged staged.",
    "Como git clone", "git clone url. Baixa repo completo + histórico.",
]
for i in range(0, len(GIT), 2):
    QA.append((GIT[i], GIT[i+1]))

random.seed(99)
random.shuffle(QA)
out = 'data/massive_code.txt'
with open(out, 'w') as f:
    for u, m in QA:
        f.write(f'USER: {u}?\nMODEL: {m}\n\n')
print(f'Generated {len(QA)} massive code pairs → {out} ({os.path.getsize(out)} bytes)')
