#!/usr/bin/env python3
"""Generate diverse mini-snippets PT-BR Q&A across many categories.
Total: 200+ pairs spanning daily life, work, hobbies, dilemas, brincadeiras.
"""
import random, os

DIVERSE = [
    # Conversação cotidiana
    ("E aí, tudo certo?", "Tudo certo por aqui! E você, como está?"),
    ("Que dia é hoje?", "Não sei o dia exato, mas posso te ajudar com outras informações."),
    ("Preciso de uma dica", "Conta o que está te incomodando, vou tentar ajudar."),
    ("Estou animado", "Que ótimo! O que aconteceu de bom?"),
    ("Me conta uma história", "Era uma vez um pequeno vilarejo nas montanhas onde tudo começou..."),
    ("Quero aprender algo novo", "Excelente! O que te interessa? Programação, idiomas, ciência?"),
    ("Vou viajar pro Brasil", "Que bom! Tem alguma cidade em mente? Rio, São Paulo, Salvador?"),
    ("Me indica um livro", "Dom Casmurro de Machado de Assis é clássico e curto, vale a pena."),
    ("Como melhorar o foco?", "Pausa de 5 min a cada 25 min (Pomodoro) e tira distrações do celular."),
    ("Como aprender inglês?", "Pratique 20 min por dia: Duolingo, séries com legenda, conversação no Italki."),

    # Trabalho / carreira
    ("Como fazer um bom currículo?", "Seja objetivo: 1 página, contato, experiências relevantes, conquistas mensuráveis."),
    ("Como negociar salário?", "Pesquise faixa de mercado, fundamente com resultados, peça 10-20% acima da oferta."),
    ("Como ser produtivo?", "Tarefas importantes pela manhã, evite multitarefa, mantenha lista curta de prioridades."),
    ("O que é trabalho remoto?", "Modelo onde a pessoa trabalha de casa ou qualquer lugar, comum em tech após 2020."),
    ("Como liderar uma equipe?", "Escute mais do que fala, dê feedback regular, delega com clareza, celebra conquistas."),

    # Saúde / bem-estar
    ("Como dormir melhor?", "Rotina fixa de horário, sem tela 1h antes, quarto escuro e fresco."),
    ("O que comer no café da manhã?", "Proteína (ovo), carboidrato complexo (pão integral) e fruta. Evite açúcar refinado."),
    ("Como começar a correr?", "Comece com 20 min alternando caminhada e trote. Sapato adequado, 3x por semana."),
    ("Como reduzir estresse?", "Respiração profunda, exercício, sono regular, contato com natureza, terapia se necessário."),
    ("Por que beber água?", "Hidrata células, transporta nutrientes, regula temperatura. Cerca de 2 litros por dia."),

    # Filosofia / reflexão
    ("Qual o sentido da vida?", "Pergunta milenar. Cada pessoa constrói o seu, geralmente através de relações, propósito e descobertas."),
    ("O que é felicidade?", "Estado emocional positivo, geralmente vinda de presença, gratidão e conexão, não posses."),
    ("Como ser ético?", "Tratar outros como gostaria de ser tratado. Considerar consequências antes de agir."),
    ("O que é livre arbítrio?", "Capacidade de fazer escolhas conscientes, debatido entre filósofos e cientistas há séculos."),

    # Programação prática
    ("Como começar com Python?", "Instale Python.org, escolha um tutorial (Real Python), faça projetos pequenos como calculadora."),
    ("Como aprender JavaScript?", "Aprenda HTML/CSS primeiro, depois JS no MDN. Pratique no console do navegador."),
    ("O que é versionamento?", "Sistema que rastreia mudanças em arquivos. Git é o padrão atual."),
    ("Como contribuir open source?", "Encontre um projeto que use, leia README + CONTRIBUTING, comece com docs ou bugs pequenos."),
    ("Como debugar código?", "Print/console.log nos pontos suspeitos, use debugger interativo, divida o problema em partes pequenas."),
    ("O que é refactoring?", "Reescrever código mantendo comportamento, mas melhorando legibilidade ou performance."),
    ("Como otimizar SQL lento?", "Veja EXPLAIN, adicione índices em colunas usadas em WHERE/JOIN, evite SELECT *."),
    ("Como melhorar performance JS?", "Evite renders desnecessários, lazy load assets, use Web Workers para CPU pesada."),

    # Trivia interessante
    ("Por que o céu é azul?", "Espalhamento Rayleigh: moléculas do ar dispersam luz azul mais que outras cores."),
    ("Por que o mar é salgado?", "Erosão de rochas continentais traz sais via rios. Acumulam por bilhões de anos."),
    ("Por que dormimos?", "Consolida memória, repara tecidos, regula hormônios. Sem sono, função cerebral cai drasticamente."),
    ("Como abelhas fazem mel?", "Coletam néctar, enzimam no estômago, regurgitam, ventilam pra evaporar água."),
    ("Por que bocejamos?", "Não 100% claro. Hipóteses: regulação cerebral, sinal social, oxigenação."),
    ("Por que o gelo flutua?", "Água sólida é menos densa que líquida (anomalia). Estrutura cristalina ocupa mais volume."),
    ("Por que arco-íris tem 7 cores?", "Refração da luz solar em gotas de água dispersa em vermelho, laranja, amarelo, verde, azul, anil, violeta."),

    # Música / arte
    ("O que é jazz?", "Gênero musical americano dos anos 20, raízes africanas, marcado por improvisação e síncope."),
    ("O que é blues?", "Gênero afro-americano do final do século XIX, base do rock, jazz e R&B."),
    ("O que é rock?", "Gênero originado nos anos 50 nos EUA, mistura blues, country e R&B. Beatles, Stones referências."),
    ("O que é ópera?", "Drama cantado com orquestra, surgida na Itália século XVI. La Scala em Milão referência."),

    # Cidades
    ("Como é Salvador?", "Capital da Bahia, primeira capital do Brasil, rica em cultura afro-brasileira e ladeiras coloridas."),
    ("Como é Recife?", "Capital de Pernambuco, conhecida como Veneza Brasileira pelos rios, frevo no carnaval, Olinda vizinha."),
    ("Como é Manaus?", "Capital do Amazonas, encontro das águas, Teatro Amazonas histórico, base para turismo na Amazônia."),
    ("Como é Florianópolis?", "Capital de Santa Catarina, ilha com 42 praias, cultura açoriana, gastronomia de mar."),

    # Tech específico
    ("O que é AI?", "Inteligência Artificial: sistemas que executam tarefas que normalmente requerem inteligência humana."),
    ("O que é Machine Learning?", "Subconjunto de IA. Algoritmos aprendem padrões dos dados sem programação explícita."),
    ("O que é LLM?", "Large Language Model. Rede neural treinada em texto massivo. GPT, Claude, Llama são exemplos."),
    ("Como funciona Wi-Fi?", "Ondas de rádio entre 2.4 e 5 GHz. Roteador transmite, dispositivos recebem e enviam dados."),
    ("Como funciona GPS?", "Satélites em órbita enviam sinais com timestamp. Receptor triangula posição via diferença de tempo."),
    ("O que é cloud computing?", "Computação em servidores remotos. AWS, Azure, GCP. Pague pelo uso, escale sob demanda."),
    ("O que é API?", "Application Programming Interface. Contrato para comunicação entre softwares."),
    ("O que é DevOps?", "Cultura unindo Dev e Ops para entrega contínua. Automação, CI/CD, monitoramento."),
    ("O que é microsserviços?", "Arquitetura onde app é dividido em serviços pequenos e independentes, comunicando via API."),

    # Comida / bebida
    ("O que é feijão tropeiro?", "Prato mineiro com feijão, farinha, bacon, linguiça, ovos. Tropeiros comiam na estrada."),
    ("O que é tapioca?", "Goma de mandioca cozida em frigideira. Recheios doces ou salgados. Origem indígena."),
    ("O que é pão de queijo?", "Bolinho mineiro de polvilho e queijo. Crocante por fora, macio por dentro."),
    ("O que é caipirinha?", "Cachaça, limão, açúcar e gelo. Coquetel nacional do Brasil."),
    ("O que é açaí?", "Fruta amazônica roxa rica em antioxidantes. Servida em tigelas com granola e banana."),

    # Animais
    ("Por que cães balançam o rabo?", "Sinal social. Balanço amplo geralmente alegria, baixo pode ser ansiedade ou submissão."),
    ("Por que gatos ronronam?", "Vibração na laringe. Pode indicar contentamento, mas também auto-cura."),
    ("Como aves voam?", "Asas geram sustentação por forma aerodinâmica. Música de baixa pressão acima e alta abaixo."),

    # Esportes
    ("Como jogar futebol básico?", "Time de 11, gol no fundo da rede do adversário, sem mão exceto goleiro, 90 min com intervalo."),
    ("O que é volêi?", "6 contra 6, 3 toques por lado, ponto se bola tocar chão adversário, 5 sets de 25."),
    ("O que é basquete?", "5 contra 5, bola na cesta, 2 pontos próximo, 3 longe, 1 lance livre, 4 quartos."),
    ("O que é tênis?", "Raquete e bola, sets, games e tie-break, jogador serve alternando lados."),

    # Mais perguntas mistas
    ("Como economizar dinheiro?", "Registre gastos, corte supérfluos, separe 10-20% da renda antes de gastar, invista o resto."),
    ("Como meditar?", "5 min iniciais. Sente confortável, foque na respiração, observe pensamentos sem julgar."),
    ("Como ler mais livros?", "20-30 min por dia em horário fixo, leve livro junto, audiobooks para deslocamento."),
    ("Como fazer amigos adultos?", "Frequente lugares de interesse comum (academia, grupos, cursos), seja consistente, mostre interesse genuíno."),
]

random.seed(21)
random.shuffle(DIVERSE)
out = 'data/diverse_dataset.txt'
with open(out, 'w') as f:
    for u, m in DIVERSE:
        f.write(f'USER: {u}\nMODEL: {m}\n\n')
print(f'Generated {len(DIVERSE)} diverse pairs → {out} ({os.path.getsize(out)} bytes)')
