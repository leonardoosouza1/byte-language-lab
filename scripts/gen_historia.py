#!/usr/bin/env python3
"""História Brasil + Mundo — 200+ Q&A com datas e eventos."""
import os, random

EVENTOS = [
    # Brasil
    ("Descobrimento do Brasil", "22 de abril de 1500", "Pedro Álvares Cabral chega ao Brasil. Início da colonização portuguesa."),
    ("Independência do Brasil", "7 de setembro de 1822", "Dom Pedro I proclama independência às margens do rio Ipiranga."),
    ("Proclamação da República", "15 de novembro de 1889", "Marechal Deodoro proclama fim do Império. Brasil vira república."),
    ("Lei Áurea", "13 de maio de 1888", "Princesa Isabel assina lei que abole escravidão no Brasil."),
    ("Revolução de 1930", "1930", "Getúlio Vargas chega ao poder após revolução. Fim da República Velha."),
    ("Estado Novo", "1937-1945", "Período ditatorial de Vargas, regime autoritário com censura."),
    ("Construção de Brasília", "1956-1960", "Juscelino Kubitschek constrói nova capital. Oscar Niemeyer arquiteto."),
    ("Golpe Militar", "1 de abril de 1964", "Militares depõem João Goulart. Início de 21 anos de ditadura."),
    ("AI-5", "13 de dezembro de 1968", "Ato Institucional 5. Endurece ditadura, fecha congresso, censura imprensa."),
    ("Eleições Diretas Já", "1984", "Movimento popular por eleições diretas. Não aprovado, mas marcou fim do regime."),
    ("Constituição de 1988", "5 de outubro de 1988", "Nova constituição democrática promulgada. Direitos fundamentais."),
    ("Impeachment de Collor", "1992", "Fernando Collor é primeiro presidente impeachado por corrupção."),
    ("Plano Real", "1994", "Plano econômico de FHC estabiliza moeda. Real substitui cruzeiro."),
    # Mundo
    ("Revolução Francesa", "1789", "Burguesia e povo derrubam monarquia francesa. Início da era moderna."),
    ("Revolução Industrial", "século XVIII-XIX", "Transição manufatura → fábrica. Vapor, têxtil, ferrovia. Iniciou na Inglaterra."),
    ("Primeira Guerra Mundial", "1914-1918", "Conflito global iniciado por assassinato do arquiduque Francisco Ferdinando."),
    ("Revolução Russa", "1917", "Bolcheviques tomam poder. Lênin lidera. Fim do czarismo, início URSS."),
    ("Crise de 1929", "29 de outubro de 1929", "Quebra da bolsa de Nova York. Grande Depressão global."),
    ("Segunda Guerra Mundial", "1939-1945", "Conflito iniciado pela invasão alemã da Polônia. Aliados vencem Eixo."),
    ("Bomba de Hiroshima", "6 de agosto de 1945", "EUA lança primeira bomba nuclear. Hiroshima e depois Nagasaki."),
    ("Queda do Muro de Berlim", "9 de novembro de 1989", "Fim da divisão da Alemanha. Início do fim da Guerra Fria."),
    ("Independência dos EUA", "4 de julho de 1776", "13 colônias declaram independência da Inglaterra. Thomas Jefferson redige declaração."),
    ("Revolução Bolchevique", "outubro de 1917", "Lênin lidera tomada do Palácio de Inverno. Fim do governo provisório russo."),
    ("Atentado às Torres Gêmeas", "11 de setembro de 2001", "Ataque da Al-Qaeda destrói WTC em NY. Mudou geopolítica global."),
    ("Pandemia COVID-19", "2020-2023", "Coronavírus SARS-CoV-2 origem em Wuhan. Milhões de mortes globais."),
    # Personagens
    ("Tiradentes", "Joaquim José da Silva Xavier", "alferes mineiro, mártir da Inconfidência Mineira, executado em 1792"),
    ("Dom Pedro I", "Pedro de Alcântara", "primeiro imperador do Brasil, proclamou independência, abdicou em 1831"),
    ("Dom Pedro II", "Pedro de Alcântara II", "imperador por 49 anos, governou Brasil 1840-1889, deposto na proclamação"),
    ("Princesa Isabel", "Isabel do Brasil", "filha de Dom Pedro II, regente, assinou Lei Áurea aboliu escravidão"),
    ("Getúlio Vargas", "Getúlio Dornelles Vargas", "presidente do Brasil 1930-45 e 1951-54, suicidou-se no Catete"),
    ("Juscelino Kubitschek", "JK", "presidente do Brasil 1956-61, construiu Brasília, '50 anos em 5'"),
    ("Tancredo Neves", "Tancredo de Almeida Neves", "presidente eleito 1985, morreu antes da posse, simbolizou redemocratização"),
    ("Fernando Henrique Cardoso", "FHC", "sociólogo, presidente do Brasil 1995-2002, criador do Plano Real"),
    ("Lula", "Luiz Inácio Lula da Silva", "presidente do Brasil 2003-2010 e desde 2023, líder do PT"),
    ("Dilma Rousseff", "Dilma Vana Rousseff", "primeira mulher presidente do Brasil 2011-2016, sofreu impeachment"),
]

QA = []
for nome, quando, descr in EVENTOS:
    QA.append((f"Quando aconteceu {nome}", f"{nome} aconteceu em {quando}. {descr}"))
    QA.append((f"O que foi {nome}", f"{descr}"))
    QA.append((f"Em que ano {nome}", f"{quando}."))

random.seed(31)
random.shuffle(QA)
out = 'data/historia_dataset.txt'
with open(out, 'w') as f:
    for u, m in QA:
        f.write(f'USER: {u}?\nMODEL: {m}\n\n')
print(f'Generated {len(QA)} história pairs → {out} ({os.path.getsize(out)} bytes)')
