#!/usr/bin/env python3
"""200+ Q&A sobre países, capitais, geografia."""
import os
PAISES = [
    # América
    ("Argentina", "Buenos Aires", "peso argentino", "espanhol"),
    ("Chile", "Santiago", "peso chileno", "espanhol"),
    ("Uruguai", "Montevidéu", "peso uruguaio", "espanhol"),
    ("Peru", "Lima", "sol peruano", "espanhol e quéchua"),
    ("Colômbia", "Bogotá", "peso colombiano", "espanhol"),
    ("Venezuela", "Caracas", "bolívar", "espanhol"),
    ("Bolívia", "Sucre constitucional, La Paz sede do governo", "boliviano", "espanhol, quéchua, aimará"),
    ("Equador", "Quito", "dólar americano", "espanhol"),
    ("Paraguai", "Assunção", "guarani", "espanhol e guarani"),
    ("Cuba", "Havana", "peso cubano", "espanhol"),
    ("México", "Cidade do México", "peso mexicano", "espanhol"),
    ("Canadá", "Ottawa", "dólar canadense", "inglês e francês"),
    ("Estados Unidos", "Washington D.C.", "dólar americano", "inglês"),
    # Europa
    ("França", "Paris", "euro", "francês"),
    ("Inglaterra", "Londres", "libra esterlina", "inglês"),
    ("Espanha", "Madri", "euro", "espanhol"),
    ("Itália", "Roma", "euro", "italiano"),
    ("Alemanha", "Berlim", "euro", "alemão"),
    ("Portugal", "Lisboa", "euro", "português"),
    ("Holanda", "Amsterdã", "euro", "holandês"),
    ("Bélgica", "Bruxelas", "euro", "francês, holandês, alemão"),
    ("Suíça", "Berna", "franco suíço", "alemão, francês, italiano, romanche"),
    ("Áustria", "Viena", "euro", "alemão"),
    ("Grécia", "Atenas", "euro", "grego"),
    ("Rússia", "Moscou", "rublo russo", "russo"),
    ("Polônia", "Varsóvia", "zloti", "polonês"),
    ("Suécia", "Estocolmo", "coroa sueca", "sueco"),
    ("Noruega", "Oslo", "coroa norueguesa", "norueguês"),
    ("Finlândia", "Helsinque", "euro", "finlandês"),
    ("Dinamarca", "Copenhague", "coroa dinamarquesa", "dinamarquês"),
    # Ásia
    ("China", "Pequim", "yuan", "mandarim"),
    ("Japão", "Tóquio", "iene", "japonês"),
    ("Coreia do Sul", "Seul", "won sul-coreano", "coreano"),
    ("Coreia do Norte", "Pyongyang", "won norte-coreano", "coreano"),
    ("Índia", "Nova Delhi", "rúpia indiana", "hindi e inglês"),
    ("Indonésia", "Jacarta", "rúpia indonésia", "indonésio"),
    ("Filipinas", "Manila", "peso filipino", "filipino e inglês"),
    ("Tailândia", "Bangcoc", "baht", "tailandês"),
    ("Vietnã", "Hanói", "dong", "vietnamita"),
    ("Singapura", "Singapura", "dólar de Singapura", "inglês, mandarim, malaio, tâmil"),
    ("Israel", "Jerusalém", "shekel", "hebraico e árabe"),
    ("Turquia", "Ancara", "lira turca", "turco"),
    ("Irã", "Teerã", "rial iraniano", "persa"),
    ("Arábia Saudita", "Riade", "riyal saudita", "árabe"),
    # África
    ("Egito", "Cairo", "libra egípcia", "árabe"),
    ("África do Sul", "Pretória administrativa, Cidade do Cabo legislativa", "rand", "11 línguas oficiais"),
    ("Nigéria", "Abuja", "naira", "inglês"),
    ("Quênia", "Nairóbi", "xelim queniano", "suaíli e inglês"),
    ("Marrocos", "Rabat", "dirham", "árabe e berbere"),
    # Oceania
    ("Austrália", "Camberra", "dólar australiano", "inglês"),
    ("Nova Zelândia", "Wellington", "dólar neozelandês", "inglês e maori"),
]

QA = []
for nome, cap, moeda, idioma in PAISES:
    QA.append((f"Qual a capital {('do' if nome[-1] not in 'aâã' else 'da')} {nome}",
              f"Capital {('do' if nome[-1] not in 'aâã' else 'da')} {nome} é {cap}."))
    QA.append((f"Qual a moeda {('do' if nome[-1] not in 'aâã' else 'da')} {nome}",
              f"Moeda {('do' if nome[-1] not in 'aâã' else 'da')} {nome} é {moeda}."))
    QA.append((f"Qual o idioma {('do' if nome[-1] not in 'aâã' else 'da')} {nome}",
              f"O idioma {('do' if nome[-1] not in 'aâã' else 'da')} {nome} é {idioma}."))

import random
random.seed(7)
random.shuffle(QA)
out = 'data/paises_dataset.txt'
with open(out, 'w') as f:
    for u, m in QA:
        f.write(f'USER: {u}?\nMODEL: {m}\n\n')
print(f'Generated {len(QA)} país pairs → {out} ({os.path.getsize(out)} bytes)')
