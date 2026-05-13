#!/usr/bin/env python3
import os, random
QA = []
TOPICS = [
    # Esportes
    ("Quem é Cristiano Ronaldo","Atacante português 5x Bola de Ouro, recordista da Champions League e seleções."),
    ("Quem é Messi","Argentino 8x Bola de Ouro, campeão mundial 2022, ídolo do Barcelona e Inter Miami."),
    ("Quem é Neymar","Atacante brasileiro, ex-Santos e Barcelona, recordista de gols na seleção, Al-Hilal e Santos."),
    ("Quem é Tite","Treinador brasileiro que comandou seleção 2016-2022, 4x campeão estadual com Corinthians."),
    ("O que é Champions League","Principal competição europeia de clubes de futebol. Real Madrid maior campeão."),
    ("O que é Copa Libertadores","Maior competição de clubes da América do Sul. Independiente maior campeão."),
    # Cinema/series
    ("O que é Netflix","Plataforma de streaming americana. House of Cards primeira série original. Bilhões de assinantes."),
    ("Que é HBO","Canal americano. Game of Thrones, Sopranos, Westworld. Foco em prestígio."),
    ("Quem é Steven Spielberg","Diretor americano. ET, Jurassic Park, Schindler's List. Maior bilheteria histórica."),
    # Ciência/tecnologia
    ("O que é ChatGPT","Modelo de linguagem da OpenAI baseado em transformers. Lançado em 2022, popularizou IA generativa."),
    ("O que é OpenAI","Empresa de IA fundada em 2015, criou GPT, DALL-E, ChatGPT. Sam Altman CEO."),
    ("O que é Google","Empresa americana de tecnologia. Mecanismo de busca, Android, YouTube, Gmail, Cloud."),
    ("O que é Microsoft","Empresa americana. Windows, Office, Azure, GitHub. Bill Gates fundador."),
    ("O que é Apple","Empresa americana. iPhone, Mac, iPad. Steve Jobs co-fundador. HQ em Cupertino."),
    ("O que é Tesla","Fabricante americana de carros elétricos. Elon Musk CEO. Pioneira em EVs."),
    ("O que é SpaceX","Empresa de Elon Musk de foguetes reutilizáveis. Falcon 9, Starship, internet Starlink."),
    # Filosofia
    ("Quem foi Sócrates filósofo","Filósofo ateniense, mestre do diálogo, executado em 399 a.C. por corromper jovens."),
    ("Quem foi Platão","Discípulo de Sócrates, fundador da Academia. Diálogos, Teoria das Formas, A República."),
    ("Quem foi Aristóteles","Discípulo de Platão, tutor de Alexandre. Lógica, ética, política, biologia. Liceu de Atenas."),
    ("Quem foi Kant","Alemão, ética dever, Crítica da Razão Pura, imperativo categórico. Iluminismo tardio."),
    ("Quem foi Nietzsche","Alemão, crítica moral cristã, super-homem, eterno retorno. Influência existencialismo."),
    # Música clássica
    ("Quem foi Bach","Compositor alemão barroco. Variações Goldberg, Toccata e Fuga, Paixão Segundo Mateus."),
    ("Quem foi Mozart","Austríaco prodígio. Sinfonia 40, Don Giovanni, Réquiem. Morreu jovem aos 35."),
    ("Quem foi Beethoven","Alemão, ponte clássico-romântico. 9 sinfonias, Ode à Alegria. Surdez nos últimos anos."),
    ("Quem foi Chopin","Polonês, piano romântico. Noturnos, Mazurcas. Morreu jovem de tuberculose."),
    ("Quem foi Tchaikovsky","Russo. Lago dos Cisnes, Quebra-Nozes, 1812. Romantismo expressivo."),
    # Religião
    ("Quem foi Buda","Príncipe Siddhartha (~500 a.C.) que iluminou-se. Fundou o budismo. Quatro Nobres Verdades."),
    ("Quem foi Jesus","Pregador judeu (~5 a.C - 30 d.C.) fundador do cristianismo. Crucificado em Jerusalém."),
    ("Quem foi Maomé","Profeta árabe (~570-632 d.C.) fundador do islamismo. Revelou Corão. Hégira em 622."),
    ("O que é Bíblia","Livro sagrado do cristianismo. Velho + Novo Testamento. ~66 livros canônicos."),
    ("O que é Corão","Livro sagrado do islamismo. 114 suras reveladas a Maomé. Texto em árabe.",),
    # Astronomia
    ("Quanto dura ano terrestre","365.25 dias. Compensados por ano bissexto a cada 4 anos."),
    ("Quanto dura dia terrestre","24 horas exatamente em média. Rotação sidérea ~23h56min."),
    ("Distância da Terra à Lua","~384.400 km. A Lua orbita em ~27 dias. Sempre a mesma face para nós."),
    ("Qual o planeta mais quente","Vênus, ~470°C superficial. Efeito estufa massivo apesar de mais distante que Mercúrio."),
    ("Qual o satélite maior do sistema solar","Ganimedes, maior que Mercúrio. Orbita Júpiter. Tem oceano subterrâneo."),
]
for u, m in TOPICS:
    QA.append((u, m))
random.seed(77)
random.shuffle(QA)
out = 'data/extras_dataset.txt'
with open(out, 'w') as f:
    for u, m in QA:
        f.write(f'USER: {u}?\nMODEL: {m}\n\n')
print(f'Generated {len(QA)} extras → {out}')
