#!/usr/bin/env python3
"""Wave 2: deeper programming patterns + algoritmos."""
import os
QA = [
    # Algoritmos
    ("Como funciona busca binária?", "Array ordenado. Pega meio. Se menor, vai esquerda; maior, direita. O(log n)."),
    ("Como funciona quicksort?", "Escolhe pivô, particiona menores à esquerda e maiores à direita, recursão nas partes. O(n log n) médio."),
    ("Como funciona mergesort?", "Divide array ao meio recursivamente, ordena partes, merge ordenado. O(n log n) garantido."),
    ("Como funciona BFS?", "Breadth-First Search. Usa fila. Visita vizinhos antes de descer. Encontra menor caminho não ponderado."),
    ("Como funciona DFS?", "Depth-First Search. Usa pilha (ou recursão). Vai profundo antes de voltar. Boa para detectar ciclos."),
    ("O que é dynamic programming?", "Quebra problema em subproblemas, armazena resultados (memoization), evita recomputação."),
    ("Como funciona Dijkstra?", "Menor caminho em grafo com pesos não-negativos. Min-heap dos custos, relaxa arestas vizinhas."),
    ("Como funciona hash table?", "Hash mapeia chave para índice. Colisões via chaining ou open addressing. O(1) médio."),

    # Estruturas de dados
    ("O que é linked list?", "Lista encadeada. Cada nó aponta para próximo. Insert/delete O(1), busca O(n)."),
    ("Diferença array e lista?", "Array memória contígua, acesso O(1). Lista nodes ligados, insert middle O(1) se referência."),
    ("O que é árvore binária?", "Cada nó tem até 2 filhos. BST mantém ordem para busca O(log n)."),
    ("O que é heap?", "Árvore com propriedade pai ≥ filhos (max-heap). Usada em priority queues, O(log n) insert."),
    ("O que é trie?", "Árvore para strings. Cada nó letra, caminho = palavra. Autocomplete, prefixos rápidos."),
    ("O que é graph?", "Conjunto de nós e arestas. Dirigido ou não, ponderado ou não. Modela redes."),

    # Padrões de design
    ("O que é Singleton?", "Padrão que garante uma única instância de classe. Cuidado: dificulta testes, estado global."),
    ("O que é Observer?", "Sujeito notifica observers sobre mudanças. Base de event-driven."),
    ("O que é Factory?", "Cria objetos via método em vez de construtor direto. Permite trocar implementação."),
    ("O que é Dependency Injection?", "Dependências passadas externamente em vez de criadas dentro. Facilita testes e flexibilidade."),
    ("O que é MVC?", "Model-View-Controller. Separa dados, UI e lógica. Padrão clássico para apps."),

    # Performance / debugging
    ("Como medir performance Python?", "cProfile mede tempo por função. timeit para snippets. Use line_profiler para linhas específicas."),
    ("Como otimizar Python?", "Use estruturas adequadas (set vs list), evite loops Python, use numpy para matemática, Cython se crítico."),
    ("Como evitar memory leak Node.js?", "Watch heap snapshot, evite closures pesadas em globais, limpe event listeners, use --inspect."),
    ("Como debugar Rust?", "println! básico, dbg!() conveniente, rust-gdb para passos. Cargo test --features debug."),

    # Linguagens snippets
    ("Hello world em Python", "print('Hello, World!')"),
    ("Hello world em JavaScript", "console.log('Hello, World!');"),
    ("Hello world em Rust", "fn main() { println!(\"Hello, World!\"); }"),
    ("Hello world em Go", "package main\\nimport \"fmt\"\\nfunc main() { fmt.Println(\"Hello, World!\") }"),
    ("Hello world em Java", "public class Main { public static void main(String[] args) { System.out.println(\"Hello, World!\"); } }"),
    ("Hello world em C", "#include <stdio.h>\\nint main() { printf(\"Hello, World!\\\\n\"); return 0; }"),

    # Web dev
    ("Como criar servidor HTTP Node.js?", "const http=require('http'); http.createServer((req,res)=>res.end('hi')).listen(3000);"),
    ("Como fetch JSON em JS?", "fetch(url).then(r=>r.json()).then(d=>console.log(d))"),
    ("Como criar app Flask?", "from flask import Flask; app=Flask(__name__); @app.route('/'); def home(): return 'hi'"),
    ("Como criar app Express?", "const express=require('express'); const app=express(); app.get('/', (req,res)=>res.send('hi'));"),
    ("O que é JSX?", "Sintaxe React que mistura HTML com JS. Babel transpila para createElement calls."),
    ("O que é Tailwind?", "Framework CSS utility-first. Classes como bg-blue-500, p-4. Composable e rápido."),

    # Math algo
    ("Como gerar Fibonacci em Python?", "def fib(n): a,b=0,1\\n  for _ in range(n): a,b=b,a+b\\n  return a"),
    ("Como verificar primo?", "def is_prime(n):\\n  if n<2: return False\\n  for i in range(2, int(n**0.5)+1):\\n    if n%i==0: return False\\n  return True"),
    ("Como calcular MDC?", "def gcd(a,b): return a if b==0 else gcd(b, a%b)"),
    ("Como fatorial?", "def fact(n): return 1 if n==0 else n*fact(n-1)"),
]
out = 'data/code_v2_dataset.txt'
with open(out, 'w') as f:
    for u, m in QA:
        f.write(f'USER: {u}\nMODEL: {m}\n\n')
print(f'Generated {len(QA)} code v2 → {out} ({os.path.getsize(out)} bytes)')
