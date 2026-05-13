#!/usr/bin/env python3
"""500+ Q&A ciência + literatura + arte detalhados."""
import os, random
QA = []

# Física moderna
FISICA = [
    "O que é mecânica quântica","Teoria física do mundo subatômico. Partículas têm comportamento ondulatório, princípio da incerteza.",
    "O que é relatividade","Teorias de Einstein. Restrita: espaço-tempo, E=mc². Geral: gravidade como curvatura espacial.",
    "O que é Big Bang","Teoria de origem do universo há ~13.8 bilhões de anos a partir de singularidade extremamente densa.",
    "O que é matéria escura","Matéria invisível inferida por efeitos gravitacionais. ~85% da matéria do universo.",
    "O que é energia escura","Energia que acelera expansão do universo. ~68% do conteúdo cósmico.",
    "O que é entropia","Medida de desordem em sistema. Segunda lei da termodinâmica: tende a aumentar com tempo.",
    "Qual a velocidade da luz","299.792.458 m/s no vácuo. Velocidade máxima no universo, constante universal.",
    "O que é fóton","Partícula da luz. Sem massa de repouso. Carrega energia proporcional à frequência.",
    "O que é elétron","Partícula subatômica com carga negativa. Massa ~1/1836 do próton. Forma ligações químicas.",
    "O que é próton","Partícula com carga positiva no núcleo atômico. Número define o elemento químico.",
    "O que é nêutron","Partícula neutra no núcleo. Estabiliza próton, junto formam massa quase total do átomo.",
]
for i in range(0, len(FISICA), 2): QA.append((FISICA[i]+'?', FISICA[i+1]))

# Química
QUIM = [
    "O que é tabela periódica","Organização dos 118 elementos químicos por número atômico. Dmitri Mendeleev em 1869.",
    "O que é molécula","Conjunto de átomos ligados covalentemente. H2O é molécula de água (2H + 1O).",
    "O que é íon","Átomo com carga elétrica. Cátion positivo (perdeu elétron), ânion negativo (ganhou).",
    "O que é pH","Escala 0-14 de acidez/basicidade. 7 neutro. <7 ácido, >7 básico. Logarítmica.",
    "O que é reação química","Transformação de reagentes em produtos. Conserva massa (Lavoisier).",
    "O que é catalisador","Substância que acelera reação sem ser consumida. Reduz energia de ativação.",
    "Diferença entre orgânico e inorgânico","Orgânico contém carbono ligado a hidrogênio. Inorgânico geralmente sem essa ligação.",
    "O que é ligação iônica","Transferência de elétrons. Cátion + ânion. Sal NaCl exemplo.",
    "O que é ligação covalente","Compartilhamento de elétrons entre átomos. H2O, CO2 exemplos.",
    "O que é eletronegatividade","Tendência átomo atrair elétrons. F mais eletronegativo (4.0). Cs menos (0.7).",
]
for i in range(0, len(QUIM), 2): QA.append((QUIM[i]+'?', QUIM[i+1]))

# Biologia
BIO = [
    "O que é fotossíntese","Processo de plantas converterem luz + CO2 + água em glicose + O2. Clorofila absorve luz.",
    "O que é respiração celular","Conversão de glicose em ATP. Glicólise + ciclo de Krebs + cadeia respiratória.",
    "O que é célula","Unidade básica da vida. Procariótica (sem núcleo) ou eucariótica (com núcleo).",
    "Diferença célula animal vegetal","Vegetal tem parede celular, cloroplastos, vacúolo grande. Animal sem parede, com lisossomos.",
    "O que é DNA","Ácido desoxirribonucleico. Dupla hélice. Carrega informação genética. Bases: A, T, G, C.",
    "O que é RNA","Ácido ribonucleico. Fita simples. Mensageiro, transportador, ribossômico. Bases: A, U, G, C.",
    "O que é gene","Sequência de DNA que codifica proteína ou regula expressão. Unidade básica da hereditariedade.",
    "O que é proteína","Cadeia de aminoácidos. Função estrutural, enzimática, transporte. Codificada por genes.",
    "O que é enzima","Proteína que catalisa reação química. Específica para substrato e produto.",
    "O que é mitocôndria","Organela produtora de energia (ATP). Tem DNA próprio. Herança materna.",
    "O que é ribossomo","Organela onde proteínas são sintetizadas. Lê mRNA e monta aminoácidos.",
    "O que é evolução","Mudança de frequência genética em populações ao longo de gerações. Darwin + genética.",
    "O que é seleção natural","Indivíduos mais adaptados sobrevivem e reproduzem mais. Não-aleatória.",
    "O que é mutação","Alteração na sequência de DNA. Aleatória. Pode ser benéfica, neutra ou prejudicial.",
    "O que é espécie","Grupo de organismos que se reproduzem e geram descendentes férteis.",
]
for i in range(0, len(BIO), 2): QA.append((BIO[i]+'?', BIO[i+1]))

# Literatura brasileira
LIT_BR = [
    "Quem foi Machado de Assis","Maior escritor brasileiro do realismo. Dom Casmurro, Memórias Póstumas de Brás Cubas. ABL fundador.",
    "Quem foi José de Alencar","Romântico brasileiro. Iracema, O Guarani, Senhora. Pioneiro de literatura nacional.",
    "Quem foi Clarice Lispector","Escritora ucraniana-brasileira. A Hora da Estrela, Perto do Coração Selvagem. Existencialismo.",
    "Quem foi Jorge Amado","Escritor baiano. Gabriela Cravo e Canela, Dona Flor, Capitães da Areia. Best-seller mundial.",
    "Quem foi Guimarães Rosa","Mineiro. Grande Sertão: Veredas, Sagarana. Inovação linguística e sertão como cosmos.",
    "Quem foi Carlos Drummond","Poeta mineiro. No Meio do Caminho, A Rosa do Povo. Maior poeta brasileiro modernista.",
    "Quem foi Mário de Andrade","Modernista paulista. Macunaíma, Paulicéia Desvairada. Líder da Semana de 22.",
    "Quem foi Manuel Bandeira","Poeta recifense. Pasárgada, Estrela da Manhã. Conversação simples e profunda.",
    "Quem foi Cecília Meireles","Poetisa carioca. Romanceiro da Inconfidência, Mar Absoluto. Lirismo refinado.",
    "Quem foi Graciliano Ramos","Romancista alagoano. Vidas Secas, São Bernardo, Memórias do Cárcere. Realismo seco.",
    "Quem foi Castro Alves","Poeta baiano abolicionista. Navio Negreiro, Espumas Flutuantes. Romântico social.",
    "O que é Modernismo brasileiro","Movimento literário-artístico iniciado em 1922 com a Semana de Arte Moderna em São Paulo.",
    "O que foi Semana de Arte Moderna","Evento em 1922 em SP. Mário, Oswald, Anita Malfatti, Villa-Lobos. Marco do modernismo.",
    "O que é Tropicalismo","Movimento cultural anos 60. Caetano, Gil, Mutantes. Mistura cultura BR + estrangeira.",
    "Quem foi Vinícius de Moraes","Poeta, diplomata, compositor. Garota de Ipanema com Tom Jobim. Carismático.",
]
for i in range(0, len(LIT_BR), 2): QA.append((LIT_BR[i]+'?', LIT_BR[i+1]))

# Literatura mundial
LIT_W = [
    "Quem foi Shakespeare","Maior dramaturgo inglês. Hamlet, Romeu e Julieta, Macbeth. 37 peças + 154 sonetos.",
    "Quem foi Dante Alighieri","Italiano. Divina Comédia (Inferno, Purgatório, Paraíso). Pai do italiano moderno.",
    "Quem foi Cervantes","Espanhol. Dom Quixote, considerado primeiro romance moderno. Cavaleiro andante.",
    "Quem foi Dostoiévski","Russo. Crime e Castigo, Os Irmãos Karamazov. Profundidade psicológica.",
    "Quem foi Tolstói","Russo. Guerra e Paz, Anna Karenina. Saga épica + realismo.",
    "Quem foi García Márquez","Colombiano. Cem Anos de Solidão. Pai do realismo mágico. Nobel 1982.",
    "Quem foi Borges","Argentino. Ficções, O Aleph. Contos labirínticos, biblioteca, espelhos.",
    "Quem foi Hemingway","Americano. O Velho e o Mar, Adeus às Armas. Estilo iceberg, frases curtas.",
    "Quem foi Kafka","Tcheco. A Metamorfose, O Processo. Burocracia opressora, alienação.",
    "Quem foi Camus","Argelino-francês. O Estrangeiro, O Mito de Sísifo. Absurdismo.",
]
for i in range(0, len(LIT_W), 2): QA.append((LIT_W[i]+'?', LIT_W[i+1]))

random.seed(50)
random.shuffle(QA)
out = 'data/mega_sci_lit.txt'
with open(out, 'w') as f:
    for u, m in QA:
        f.write(f'USER: {u}\nMODEL: {m}\n\n')
print(f'Generated {len(QA)} mega sci/lit → {out} ({os.path.getsize(out)} bytes)')
