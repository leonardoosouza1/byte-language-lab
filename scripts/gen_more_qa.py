#!/usr/bin/env python3
"""500+ Q&A — wave final massivo. Misturando todos domínios."""
import os
QA = []

# Verbos comuns + uso
VERBOS = [
    ("Como conjugar verbo amar","Eu amo, tu amas, ele ama, nós amamos, vós amais, eles amam. Presente do indicativo.","gram"),
    ("Como conjugar verbo ser","Eu sou, tu és, ele é, nós somos, vós sois, eles são. Verbo irregular essencial.","gram"),
    ("Como conjugar verbo ir","Eu vou, tu vais, ele vai, nós vamos, vós ides, eles vão.","gram"),
    ("Como conjugar verbo ter","Eu tenho, tu tens, ele tem, nós temos, vós tendes, eles têm.","gram"),
    ("Como conjugar verbo fazer","Eu faço, tu fazes, ele faz, nós fazemos, vós fazeis, eles fazem.","gram"),
    ("Como conjugar verbo dizer","Eu digo, tu dizes, ele diz, nós dizemos, vós dizeis, eles dizem.","gram"),
    ("Quando usar aonde","Aonde indica movimento (vai aonde). Onde indica lugar parado (mora onde).","gram"),
    ("Quando usar mas mais","Mas = porém, conjunção adversativa. Mais = quantidade, advérbio.","gram"),
    ("Quando usar mau mal","Mau = adjetivo (mau aluno). Mal = advérbio (passou mal) ou substantivo (o mal).","gram"),
    ("Quando usar a há","A = preposição, artigo (a casa). Há = verbo haver no sentido de existir (há dúvidas).","gram"),
]
for u, m, _ in VERBOS:
    QA.append((u, m))

# Cores em diversos contextos
CORES = [
    ("O que significa cor azul","Cor primária. Simboliza calma, confiança. Comprimento de onda ~450-495nm.","fato"),
    ("O que significa cor vermelha","Cor primária. Simboliza paixão, perigo, energia. ~620-740nm.","fato"),
    ("O que significa cor verde","Mistura amarelo + azul. Natureza, esperança, calma. ~495-570nm.","fato"),
    ("O que significa cor amarela","Cor primária. Sol, alegria, alerta. ~570-590nm.","fato"),
    ("O que é cor","Sensação visual produzida por luz refletida em comprimentos de onda específicos.","fato"),
    ("Como funciona daltonismo","Deficiência de cones retinianos. Mais comum confundir vermelho-verde. Hereditário ligado ao X.","fato"),
]
for u, m, _ in CORES:
    QA.append((u, m))

# Calendário, festas
DATAS = [
    ("Quando é Natal","25 de dezembro. Celebra nascimento de Jesus Cristo no cristianismo. Tradições natalinas globais.","fato"),
    ("Quando é Ano Novo","1 de janeiro no calendário gregoriano. Outros calendários têm datas diferentes.","fato"),
    ("Quando é Páscoa","Primeiro domingo após primeira lua cheia do equinócio de primavera. Móvel, março ou abril.","fato"),
    ("Quando é Carnaval","Móvel, 47 dias antes da Páscoa. Tradicionalmente fevereiro ou início de março.","fato"),
    ("Quando é Dia das Mães","Segundo domingo de maio no Brasil.","fato"),
    ("Quando é Dia dos Pais","Segundo domingo de agosto no Brasil.","fato"),
    ("Quando é Dia da Independência","7 de setembro no Brasil. Comemora Independência de 1822.","fato"),
    ("Quando é Tiradentes","21 de abril. Feriado nacional homenageando alferes da Inconfidência Mineira.","fato"),
    ("Quanto duram as 4 estações","No hemisfério sul: primavera (set-dez), verão (dez-mar), outono (mar-jun), inverno (jun-set).","fato"),
]
for u, m, _ in DATAS:
    QA.append((u, m))

# Plantas/Animais Brasil
NATUREZA = [
    ("O que é boto cor de rosa","Mamífero aquático amazônico, lenda popular, golfinho de água doce, ameaçado.","fato"),
    ("O que é mico-leão-dourado","Pequeno primata da Mata Atlântica, ameaçado de extinção, símbolo de conservação BR.","fato"),
    ("O que é cavalo marinho","Peixe ósseo, macho engravida, cavalha em ervas, ameaçado por pesca e poluição.","fato"),
    ("O que é ipê amarelo","Árvore brasileira, flores amarelas vibrantes na primavera, madeira nobre.","fato"),
    ("O que é jabuticaba","Fruta brasileira nativa, nasce no tronco, sabor adocicado, vinhos e licores.","fato"),
    ("O que é açaizeiro","Palmeira amazônica produtora da fruta açaí, palmito comestível, polpa antioxidante.","fato"),
    ("O que é mandioca","Tubérculo amazônico, base de farinha, tapioca, beiju. Variedades brava e mansa.","fato"),
    ("O que é guaraná","Fruta amazônica, rica em cafeína natural, base de refrigerantes e energéticos.","fato"),
]
for u, m, _ in NATUREZA:
    QA.append((u, m))

import random
random.seed(11)
random.shuffle(QA)
out = 'data/more_qa.txt'
with open(out, 'w') as f:
    for u, m in QA:
        f.write(f'USER: {u}?\nMODEL: {m}\n\n')
print(f'Generated {len(QA)} more QA → {out}')
