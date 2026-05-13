#!/usr/bin/env python3
"""300+ Q&A combinando muitas categorias."""
import os, random
QA = []

# Esportes detalhado
QA += [
    ("Quem foi Pelé","Edson Arantes do Nascimento, único tricampeão mundial 1958/62/70, considerado o Rei do Futebol.","fato"),
    ("Quem foi Garrincha","Manuel Francisco dos Santos, bicampeão mundial 58/62, símbolo da alegria, dribles impossíveis.","fato"),
    ("Quem foi Zico","Arthur Antunes Coimbra, ídolo do Flamengo e seleção 70s-80s, Galinho de Quintino.","fato"),
    ("Quem foi Tostão","Eduardo Gonçalves de Andrade, campeão mundial 1970, oftalmologista, cronista esportivo.","fato"),
    ("Quem foi Rivellino","Roberto Rivelino, campeão mundial 1970, autor do gol do bigodão.","fato"),
    ("Quem foi Romário","Atacante artilheiro, campeão mundial 1994, ex-jogador do Vasco, Barcelona, PSV.","fato"),
    ("Quem foi Ronaldo Fenômeno","Atacante brasileiro 2x Bola de Ouro, campeão mundial 1994 e 2002, gols icônicos.","fato"),
    ("Quem foi Ronaldinho Gaúcho","Meia brasileiro, Bola de Ouro 2005, campeão mundial 2002, mago dos dribles.","fato"),
    ("Quem é Marta","Atacante brasileira 6x melhor do mundo, ícone do futebol feminino, jogou EUA e Itália.","fato"),
    # Vinhos / bebidas
    ("O que é cachaça","Destilado brasileiro da cana-de-açúcar. Base da caipirinha. Variedades regionais.","fato"),
    ("O que é vinho tinto","Vinho fermentado das cascas e sementes da uva. Aromas complexos, taninos.","fato"),
    ("Diferença vinho tinto branco","Tinto usa uvas tintas com casca. Branco usa polpa, mais leve e ácido.","fato"),
    # Brasil regiões
    ("Quais estados Sudeste","SP, RJ, MG, ES. Maior PIB, maior população, motor econômico do Brasil.","fato"),
    ("Quais estados Nordeste","BA, SE, AL, PE, PB, RN, CE, PI, MA. 9 estados, cultura rica, praias paradisíacas.","fato"),
    ("Quais estados Norte","AM, PA, AC, RO, RR, AP, TO. 7 estados, Amazônia, populações indígenas, calor úmido.","fato"),
    ("Quais estados Sul","RS, SC, PR. 3 estados, clima temperado, influência europeia, gaúchos, catarinenses, paranaenses.","fato"),
    ("Quais estados Centro-Oeste","GO, MT, MS, DF. 4 unidades, cerrado, agronegócio, Brasília, Pantanal.","fato"),
    # Cidades famosas
    ("O que é Rio de Janeiro","Cidade maravilhosa, capital do estado RJ, Cristo Redentor, Pão de Açúcar, praias Copacabana e Ipanema.","fato"),
    ("O que é São Paulo","Maior cidade do Brasil, capital financeira, 12M habitantes, cinza pelas construções, gastronomia diversa.","fato"),
    ("O que é Brasília","Capital federal, planejada por Niemeyer e Lucio Costa, inaugurada em 1960, formato de avião.","fato"),
    ("O que é Salvador","Capital BA, primeira capital do Brasil, Pelourinho, axé, dendê, afro-cultural.","fato"),
    ("O que é Belo Horizonte","Capital MG, planejada, cidade jardim, sede de empresas, gastronomia mineira.","fato"),
    # Esporte detalhado
    ("O que é Vasco da Gama","Clube carioca fundado em 1898, 4x campeão brasileiro, integrou os 12 grandes.","fato"),
    ("O que é Flamengo","Clube carioca, maior torcida do Brasil, fundado 1895, 8 brasileiros, 3 Libertadores.","fato"),
    ("O que é Palmeiras","Clube paulista (fundado 1914), 12 brasileiros, 4 Libertadores, mascote periquito.","fato"),
    ("O que é Corinthians","Clube paulista, fundado 1910, Maracanazo do Mundial 2012, mascote mosqueteiro.","fato"),
    ("O que é São Paulo FC","Clube paulista, fundado 1930, 6 brasileiros, 3 Libertadores, 3 mundiais.","fato"),
    ("O que é Santos","Clube paulista, fundado 1912, equipe de Pelé, 8 brasileiros, 3 Libertadores.","fato"),
    # Animais
    ("O que é capivara","Maior roedor do mundo, nativo da América do Sul, semiaquático, peso até 80kg.","fato"),
    ("O que é arara","Pássaro tropical colorido, família dos psitacídeos, longevidade até 80 anos.","fato"),
    ("O que é jacaré","Réptil amazônico, predador de água doce, parente do crocodilo, focinho mais arredondado.","fato"),
    ("O que é onça-pintada","Maior felino das Américas, no Pantanal e Amazônia, mandíbula mais forte por kg.","fato"),
    ("O que é tatu","Mamífero com carapaça óssea, hábitos noturnos, escava buracos. Vários espécies no Brasil.","fato"),
]
out = 'data/bigshot_dataset.txt'
with open(out, 'w') as f:
    for u, m, _ in QA:
        f.write(f'USER: {u}?\nMODEL: {m}\n\n')
print(f'Generated {len(QA)} bigshot → {out}')
