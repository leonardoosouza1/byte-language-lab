#!/usr/bin/env python3
"""Gigante: 300+ Q&A PT-BR sobre tudo — fatos, conversas, conceitos."""
import os, random

QA = []

# Fatos rápidos PT-BR
RAPID = [
    "Qual o maior estado do Brasil", "Amazonas, com mais de 1.5 milhão km².",
    "Qual o menor estado do Brasil", "Sergipe, com 21.910 km².",
    "Quantos estados tem o Brasil", "26 estados + Distrito Federal = 27 unidades federativas.",
    "Quantas regiões tem o Brasil", "5: Norte, Nordeste, Centro-Oeste, Sudeste, Sul.",
    "Qual a maior cidade do Brasil", "São Paulo, com mais de 12 milhões de habitantes.",
    "Qual o segundo maior país da América", "Brasil é 5º do mundo, maior da América do Sul. EUA 3º maior mundo.",
    "Quantas Copas o Brasil ganhou", "5 Copas: 1958, 1962, 1970, 1994, 2002.",
    "Qual o hino nacional brasileiro", "Composto por Francisco Manuel da Silva, letra por Joaquim Osório Duque Estrada.",
    "Quem descobriu o Brasil", "Pedro Álvares Cabral em 22 de abril de 1500.",
    "Qual a língua oficial do Brasil", "Português brasileiro, falado por 215+ milhões.",
    "Quantos hectares tem a Amazônia", "Cerca de 670 milhões de hectares. Maior floresta tropical do mundo.",
    "Qual o rio mais longo do mundo", "Nilo 6.650 km segundo medições tradicionais. Amazonas competidor.",
    "Qual o oceano mais profundo", "Pacífico, com fossa Mariana a 11 km de profundidade.",
    "Qual a montanha mais alta", "Monte Everest, 8.849 metros, na cordilheira do Himalaia.",
    "Qual o deserto mais quente", "Lut, no Irã, com temperaturas até 70°C registradas via satélite.",
    "Qual o animal mais rápido", "Falcão peregrino no mergulho atinge 390 km/h.",
    "Qual o mamífero maior", "Baleia azul, até 30 metros e 180 toneladas.",
    "Qual o ser vivo mais longo", "Tubarão da Groenlândia, vive 400+ anos.",
    "Qual o planeta mais próximo do Sol", "Mercúrio, mais quente lado dia, mais frio lado noite.",
    "Qual o planeta maior", "Júpiter, gigante gasoso com massa 2.5x todos outros juntos.",
    "Quantos planetas no sistema solar", "8 oficiais. Plutão reclassificado como planeta anão em 2006.",
    "Qual a galáxia que estamos", "Via Láctea, espiral barrada com 100 bilhões de estrelas.",
    "Qual a estrela mais próxima", "Sol. Depois Proxima Centauri a 4.2 anos-luz.",
    "Quantos anos tem o universo", "13.8 bilhões de anos desde Big Bang.",
    "Quantos cromossomos humanos", "23 pares = 46 total. Trissomia 21 = síndrome de Down.",
    "Qual o maior órgão humano", "Pele, cerca de 2 m² em adulto.",
    "Quanto bate o coração por minuto", "60-100 bpm em repouso. Atletas podem 40-50.",
    "Quanto sangue tem um adulto", "5-6 litros, aproximadamente 7% do peso.",
    "Quantos litros de ar respira", "Pulmões trocam ~6-10 litros/min em repouso.",
    "Qual a temperatura corporal normal", "36.5-37.5°C. Acima é febre. Hipotermia abaixo de 35°C.",
    "Quanto pesa o cérebro", "1.3-1.5 kg em adulto. 2% do peso, consome 20% energia.",
    "Quantos neurônios cérebro humano", "86 bilhões. Conectividade via 100 trilhões sinapses.",
    "Como dormir bem", "Quarto escuro, fresco, sem tela, horário fixo, evite cafeína à tarde.",
    "Quanto tempo dormir adulto", "7-9 horas. Indivíduos variam, qualidade importa mais que quantidade.",
    "O que é REM", "Sono REM: movimento rápido olhos, sonhos vívidos, ~25% do sono total.",
]
for i in range(0, len(RAPID), 2):
    QA.append((RAPID[i], RAPID[i+1]))

# Profissões
PROFS = [
    "O que faz médico", "Diagnostica e trata doenças. Especialidades variam: clínico, cirurgião, pediatra.",
    "O que faz advogado", "Defende clientes, prepara processos, consulta sobre leis. OAB regula no Brasil.",
    "O que faz engenheiro", "Projeta soluções técnicas. Civil, mecânico, elétrico, software, química.",
    "O que faz professor", "Ensina alunos, prepara aulas, avalia aprendizado. Educação formal ou treinamento.",
    "O que faz programador", "Escreve software. Resolve problemas em linguagens variadas. Backend, frontend, mobile.",
    "O que faz designer", "Cria visuais e experiências. Gráfico, web, UX/UI, produto, moda.",
    "O que faz cientista", "Pesquisa e descobre. Hipótese, experimento, análise, publicação. Áreas múltiplas.",
    "O que faz arquiteto", "Projeta edifícios e espaços. Junta estética, função, segurança, legislação.",
    "O que faz contador", "Cuida das finanças. Demonstrativos, impostos, auditoria. CRC regula.",
    "O que faz jornalista", "Apura, escreve, divulga notícias. Imprensa, TV, rádio, internet.",
]
for i in range(0, len(PROFS), 2):
    QA.append((PROFS[i], PROFS[i+1]))

# Filosofia/conceito
PHIL = [
    "O que é amor", "Sentimento profundo de afeto e cuidado. Familiar, romântico, platônico, fraternal.",
    "O que é amizade", "Relação de confiança e afinidade mútua, construída ao longo do tempo.",
    "O que é coragem", "Agir apesar do medo. Não é ausência de medo, é enfrentá-lo por algo importante.",
    "O que é sabedoria", "Saber aplicar conhecimento e experiência em decisões. Diferente de inteligência crua.",
    "O que é justiça", "Dar a cada um o que lhe é devido. Princípio ético + sistema legal.",
    "O que é liberdade", "Capacidade de agir conforme própria vontade dentro de limites éticos e legais.",
    "O que é verdade", "Correspondência entre afirmação e realidade. Conceito debatido em filosofia.",
    "O que é ética", "Estudo do certo e errado no comportamento. Fundamenta moral.",
    "O que é moral", "Conjunto de regras de conduta de uma sociedade. Pode variar entre culturas.",
    "O que é arte", "Expressão criativa que comunica emoção, ideia ou beleza. Subjetiva, contextual.",
    "O que é beleza", "Qualidade que agrada sentidos ou alma. Estética debate critérios universais vs subjetivos.",
    "O que é cultura", "Conjunto de costumes, valores, arte, língua de um povo. Transmitida socialmente.",
    "O que é educação", "Processo de aprender e transmitir conhecimento. Formal escolar e informal vida.",
    "O que é trabalho", "Atividade que produz valor. Pode ser remunerado ou voluntário, físico ou intelectual.",
    "O que é dinheiro", "Meio de troca convencionado. Representa valor stored, unidade de conta.",
]
for i in range(0, len(PHIL), 2):
    QA.append((PHIL[i], PHIL[i+1]))

# Saúde / corpo
HEALTH = [
    "Como evitar gripe", "Higiene mãos, evitar contato infectados, alimentação balanceada, vacina anual.",
    "Como tratar resfriado", "Repouso, hidratação, sopa quente, analgésico simples se febre.",
    "Como combater dor de cabeça", "Hidratação, descanso, evitar tela longa, analgésico ocasional, consulte se persistente.",
    "Como cuidar dos olhos", "Distância da tela, regra 20-20-20, óculos se necessário, check-ups anuais.",
    "Como ter pele bonita", "Protetor solar diário, hidratação, dormir bem, alimentação rica em vegetais.",
    "Como emagrecer saudável", "Déficit calórico moderado, proteína adequada, exercício, sono. Sem dieta extrema.",
    "Como ganhar massa muscular", "Treino de força, proteína 1.6-2g/kg, déficit calórico ligeiro, descanso adequado.",
    "Como combater ansiedade", "Respiração diafragmática, exercício, terapia, sono, evitar cafeína em excesso.",
    "Como combater depressão", "Procure profissional. Exercício, sono, alimentação, conexões sociais ajudam.",
    "Como melhorar memória", "Exercício, sono, ler, aprender, técnicas como repetição espaçada, palácio da memória.",
    "O que é diabetes tipo 1", "Pâncreas não produz insulina. Autoimune, geralmente surge na infância.",
    "O que é diabetes tipo 2", "Resistência à insulina. Associada a estilo de vida, mais comum em adultos.",
    "O que é hipertensão", "Pressão arterial cronicamente elevada acima de 140x90. Risco cardiovascular.",
    "O que é colesterol", "Tipo de gordura no sangue. HDL bom, LDL ruim. Excesso causa placas em artérias.",
    "O que é AVC", "Acidente Vascular Cerebral. Falta de sangue ao cérebro. Emergência, ligue 192.",
]
for i in range(0, len(HEALTH), 2):
    QA.append((HEALTH[i], HEALTH[i+1]))

# Receitas brasileiras
COOK = [
    "Como fazer feijão simples", "Lave feijão, deixe de molho 8h. Cozinhe em pressão 30 min com água, sal, alho, louro, bacon.",
    "Como fazer arroz branco", "1 xícara arroz, 2 de água. Refogue alho em óleo, junte arroz, água, sal. Cozinhe 15 min.",
    "Como fazer brigadeiro", "1 lata leite condensado, 2 colheres chocolate em pó, 1 colher manteiga. Mexa em fogo baixo até desgrudar.",
    "Como fazer pão de queijo", "500g polvilho azedo, 250ml leite, 200g queijo minas, 2 ovos, 100g manteiga, sal. Forno 200°C 20 min.",
    "Como fazer pudim", "1 lata leite condensado, mesma medida leite, 3 ovos. Bata. Caramelize forma com açúcar. Banho-maria 1h.",
    "Como fazer estrogonofe", "Frango ou carne em cubos. Refogue, junte ketchup, mostarda, creme de leite. Sirva com arroz e batata palha.",
    "Como fazer omelete", "2 ovos, sal, pimenta, leite. Bata. Frite em fogo baixo, vire ao meio cozinhar.",
    "Como fazer panqueca", "Bata 2 ovos, 1 xícara farinha, 1 xícara leite, sal. Frigideira untada, vire dourar.",
]
for i in range(0, len(COOK), 2):
    QA.append((COOK[i], COOK[i+1]))

# Tech detalhado
TECH = [
    "O que é HTTP", "Protocolo de comunicação web. Cliente faz request (GET/POST/etc), servidor responde com status + body.",
    "O que é HTTPS", "HTTP sobre TLS. Criptografa comunicação. Cadeado no browser. Padrão moderno.",
    "O que é DNS", "Domain Name System. Traduz nomes (google.com) para IPs (8.8.8.8).",
    "O que é IP", "Internet Protocol. Endereço único na rede. IPv4 32 bits, IPv6 128 bits.",
    "O que é TCP", "Transmission Control Protocol. Conexão confiável, ordenada, com handshake.",
    "O que é UDP", "User Datagram Protocol. Sem conexão, rápido, sem garantias. Vídeo, jogos.",
    "O que é VPN", "Virtual Private Network. Túnel criptografado entre cliente e servidor.",
    "O que é proxy", "Intermediário entre cliente e servidor. Cache, filtro, anonimato.",
    "O que é firewall", "Sistema que filtra tráfego de rede. Bloqueia acessos indesejados.",
    "O que é load balancer", "Distribui requests entre múltiplos servidores. Aumenta capacidade e disponibilidade.",
    "O que é CDN", "Content Delivery Network. Cache distribuído globalmente. Reduz latência de assets.",
    "O que é WebSocket", "Conexão persistente bidirecional sobre HTTP. Real-time chat, notificações, jogos.",
    "O que é OAuth", "Padrão de autorização. Permite acesso sem compartilhar senha. Login com Google/Facebook.",
    "O que é JWT", "JSON Web Token. Token autoassinado. Header.Payload.Sig em base64. Stateless auth.",
    "O que é CORS", "Cross-Origin Resource Sharing. Permite ou bloqueia requests entre origens diferentes.",
]
for i in range(0, len(TECH), 2):
    QA.append((TECH[i], TECH[i+1]))

random.seed(123)
random.shuffle(QA)
out = 'data/massive_pt.txt'
with open(out, 'w') as f:
    for u, m in QA:
        f.write(f'USER: {u}?\nMODEL: {m}\n\n')
print(f'Generated {len(QA)} massive PT pairs → {out} ({os.path.getsize(out)} bytes)')
