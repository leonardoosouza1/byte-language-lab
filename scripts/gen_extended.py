#!/usr/bin/env python3
"""Extended Q&A: programming details, daily dialog, math problems, history facts."""
import random, os

EXTENDED = [
    # Programming deeper
    ("Como funciona garbage collector?", "GC libera memória de objetos sem referência. Algoritmos: mark-sweep, generational, reference counting."),
    ("O que é Big O notation?", "Notação de complexidade assintótica. O(1) constante, O(n) linear, O(n²) quadrático, O(log n) logarítmico."),
    ("O que é REST API?", "Representational State Transfer. Arquitetura usando HTTP: GET, POST, PUT, DELETE. Stateless, resource-based."),
    ("O que é GraphQL?", "Linguagem de query para APIs. Cliente pede exatamente os dados necessários. Single endpoint."),
    ("Diferença entre TCP e UDP?", "TCP confiável, ordenado, com handshake. UDP rápido, sem conexão, pode perder pacotes."),
    ("O que é Docker?", "Plataforma de containers. Empacota app + dependências em imagem isolada e portável."),
    ("O que é Kubernetes?", "Orquestrador de containers. Auto-scaling, load balancing, self-healing, rolling updates."),
    ("O que é git rebase?", "Reescreve histórico movendo commits. Diferente de merge que cria commit de junção."),
    ("Diferença entre SQL e NoSQL?", "SQL relacional com schema fixo. NoSQL flexível, document/key-value/graph. Escolha conforme padrão de acesso."),
    ("O que é JWT?", "JSON Web Token. Token assinado para autenticação stateless. Header.Payload.Signature em base64."),
    ("Como funciona CORS?", "Cross-Origin Resource Sharing. Servidor permite ou nega request de origem diferente via headers."),
    ("O que é WebSocket?", "Protocolo bidirectional sobre HTTP. Conexão persistente para comunicação real-time."),
    ("O que é Server-Sent Events?", "SSE permite servidor enviar updates ao cliente via HTTP. Unidirecional, mais simples que WebSocket."),
    ("O que é hash function?", "Função que mapeia dado de tamanho variável para fixo. SHA-256, MD5. Usada em assinaturas, hash tables."),
    ("O que é índice em banco?", "Estrutura que acelera busca. B-tree comum em SQL. Trade-off: faster reads, slower writes."),

    # Daily dialog
    ("Bom dia, como você está?", "Bom dia! Estou ótimo, obrigado por perguntar. Como posso ajudar você hoje?"),
    ("Estou triste hoje", "Sinto muito ouvir isso. Quer conversar sobre o que está te incomodando?"),
    ("Estou cansado", "Descansar é importante. Tenta dormir cedo e cuidar de você."),
    ("Que horas são?", "Não tenho acesso ao relógio em tempo real, mas posso ajudar com outras questões."),
    ("Você é uma IA?", "Sou um modelo de linguagem, treinado em corpus PT-BR para responder perguntas e conversar."),
    ("Pode me ajudar?", "Claro! Em que posso ajudar você?"),
    ("Muito obrigado", "De nada! Foi um prazer ajudar."),
    ("Até logo", "Até mais! Foi bom conversar com você."),
    ("Como aprender programação?", "Comece com Python ou JavaScript. Pratique projetos pequenos. Site freeCodeCamp e Codecademy ajudam."),
    ("Qual a melhor linguagem de programação?", "Depende do objetivo. Python para dados/IA, JS para web, Rust para performance, Go para servidores."),

    # Math problems
    ("Quanto é 15 vezes 12?", "15 × 12 = 180."),
    ("Quanto é 100 dividido por 7?", "100 ÷ 7 ≈ 14.286, ou 14 com resto 2."),
    ("Qual a soma dos números de 1 a 100?", "Soma é 5050. Fórmula de Gauss: n(n+1)/2."),
    ("Quanto é 2^16?", "2^16 = 65536."),
    ("Como calcular juros simples?", "J = C × i × t. Capital × taxa × tempo. Sem capitalização periódica."),
    ("Como calcular juros compostos?", "M = C × (1 + i)^t. Reinveste juros sobre juros. Crescimento exponencial."),

    # PT-BR cultural
    ("Quem foi Tom Jobim?", "Antonio Carlos Jobim, compositor brasileiro pioneiro da Bossa Nova. Garota de Ipanema, Águas de Março."),
    ("Quem foi Vinícius de Moraes?", "Poeta, diplomata e compositor. Co-autor de Garota de Ipanema. Letrista da Bossa Nova."),
    ("Quem foi Chico Buarque?", "Cantor, compositor e escritor brasileiro. Construção, Apesar de Você, ícone da MPB."),
    ("Quem foi Caetano Veloso?", "Cantor e compositor baiano, líder do Tropicalismo nos anos 60. Sampa, Alegria Alegria."),
    ("O que é capoeira?", "Arte marcial afro-brasileira que combina luta, dança e música. Berimbau marca o ritmo."),
    ("O que é axé music?", "Gênero musical baiano nascido nos anos 80. Trios elétricos, carnaval de Salvador, Daniela Mercury, Ivete Sangalo."),
    ("O que é funk carioca?", "Gênero brasileiro do Rio. Bailes funk nas comunidades. Anitta, Ludmilla referências."),
    ("Quem foi Heitor Villa-Lobos?", "Compositor erudito brasileiro. Bachianas Brasileiras fundiram Bach com folclore nacional."),

    # Science deeper
    ("Como funciona um neurônio?", "Recebe sinais via dendritos, integra no corpo celular, transmite via axônio. Sinapses comunicam neurônios."),
    ("O que é cérebro humano?", "Órgão central do sistema nervoso, ~86 bilhões de neurônios, controla pensamento, emoção, movimento."),
    ("O que é homeostase?", "Capacidade do corpo manter equilíbrio interno (temperatura, pH, glicose) apesar de mudanças externas."),
    ("Como funciona vacina?", "Apresenta versão fraca ou parte do patógeno ao sistema imune. Cria memória para combate futuro."),
    ("O que é antibiótico?", "Substância que mata ou inibe bactérias. Sem efeito contra vírus. Penicillina foi a primeira."),
    ("Como funciona o sol?", "Reação de fusão nuclear: hidrogênio vira hélio liberando energia. Produz luz e calor há 4.6 bilhões de anos."),
    ("O que é buraco negro?", "Região do espaço com gravidade tão forte que nem luz escapa. Formado por colapso estelar massivo."),
    ("O que é Big Bang?", "Teoria de origem do universo há ~13.8 bilhões de anos a partir de singularidade que se expandiu."),
]

random.seed(13)
random.shuffle(EXTENDED)
out = 'data/extended_dataset.txt'
with open(out, 'w') as f:
    for u, m in EXTENDED:
        f.write(f'USER: {u}\nMODEL: {m}\n\n')
print(f'Generated {len(EXTENDED)} pairs → {out} ({os.path.getsize(out)} bytes)')
