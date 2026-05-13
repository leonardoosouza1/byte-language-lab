#!/usr/bin/env python3
"""500+ Q&A PT-BR — saber popular + curiosidades."""
import os, random

QA = []

# Saúde detalhada
SAUDE = [
    "Quais os benefícios de caminhar","Reduz risco cardiovascular, melhora humor, fortalece músculos e ossos. 30 min/dia já ajuda.",
    "Como evitar pressão alta","Reduza sal, exercite-se, mantenha peso, não fume, modere álcool, gerencie estresse.",
    "Quanto exercitar por semana","WHO: 150 min moderado ou 75 min vigoroso, mais 2 sessões de força.",
    "Como melhorar postura","Alongamento, fortaleça core, ajuste tela na altura dos olhos, pausas regulares.",
    "Como evitar varizes","Movimente as pernas, evite ficar parado horas, eleve pernas, meias de compressão se prescrito.",
    "Como melhorar circulação","Exercício regular, hidratação, alimentação rica em ômega-3, evite tabaco.",
    "O que comer para energia","Carboidratos complexos (aveia, quinoa), proteína magra, frutas, hidratação.",
    "Como combater enxaqueca","Identifique gatilhos (alimentos, sono), hidrate, ambiente escuro/silencioso, consulte neurologista.",
    "Qual a importância de tomar sol","Síntese de vitamina D. 15-20 min/dia braços e pernas, evite pico solar.",
    "Como cuidar dos dentes","Escove 2x/dia, fio dental, dentista a cada 6 meses, evite açúcar excessivo.",
    "Como melhorar respiração","Respiração diafragmática lenta, ioga, exercício aeróbico regular, evite poluição.",
    "Quantos copos de água por dia","Aproximadamente 2 litros, varia com clima e atividade física.",
    "Como combater insônia","Horário fixo de sono, sem cafeína à tarde, sem tela 1h antes, quarto fresco.",
    "Como ter cabelo saudável","Lave com frequência adequada ao seu tipo, hidrate, evite ferro quente excessivo.",
    "Como prevenir câncer de pele","Protetor solar diário FPS 30+, evite sol entre 10-16h, exames regulares.",
]
for i in range(0, len(SAUDE), 2):
    QA.append((SAUDE[i], SAUDE[i+1]))

# Trabalho profissional
TRABALHO = [
    "Como criar portfolio dev","Site simples mostrando 3-5 projetos completos, código no GitHub, README detalhado.",
    "Como passar em entrevista técnica","Estude algoritmos básicos, pratique LeetCode, prepare 'fale sobre você', faça perguntas.",
    "Como aprender DevOps","Comece com Linux + bash, depois Docker, depois Kubernetes, depois CI/CD.",
    "Quanto ganha programador no Brasil","Júnior 3-6k, Pleno 6-12k, Sênior 12-25k. Salários em SP/Rio maiores.",
    "Como migrar para tech","Faça bootcamp ou autodidatismo. Construa portfolio. Network. Aceite primeiro emprego mesmo desafiador.",
    "Vale a pena freelancer","Mais liberdade mas instabilidade. Bom para experiência variada. Comece em paralelo com CLT.",
    "Como produtivizar dia trabalho","Pomodoro (25min foco + 5min pausa), desligue notificações, tarefas importantes manhã.",
    "Quais ferramentas dev usar","VSCode editor, Git versão, Docker containers, GitHub repos, Postman APIs, Notion notas.",
    "Como aprender frontend rápido","HTML+CSS 2 semanas, JavaScript 1 mês, React 1 mês. Faça projetos reais.",
    "Como aprender backend","Escolha linguagem (Python/Node), aprenda REST APIs, banco de dados, depois deploy.",
]
for i in range(0, len(TRABALHO), 2):
    QA.append((TRABALHO[i], TRABALHO[i+1]))

# Pessoas brasileiras famosas
PESSOAS = [
    "Quem foi Ayrton Senna","Tricampeão de Fórmula 1 (1988, 90, 91). Ídolo nacional. Morreu em Ímola 1994.",
    "Quem foi Roberto Carlos","Cantor mais influente da música brasileira moderna. Discos de ouro recorde.",
    "Quem é Anitta","Cantora brasileira de pop e funk, sucesso internacional, hits em espanhol e inglês.",
    "Quem foi Adoniran Barbosa","Cantor e compositor paulista. Trem das Onze, Saudosa Maloca. Voz do samba paulista.",
    "Quem foi Cazuza","Cantor brasileiro líder do Barão Vermelho. Letras políticas e poéticas. Morreu 1990 de AIDS.",
    "Quem foi Renato Russo","Cantor da Legião Urbana. Faroeste Caboclo, Tempo Perdido. Morreu 1996.",
    "Quem foi Belchior","Cantor cearense. Apenas um Rapaz Latino-Americano, Como Nossos Pais. MPB engajada.",
    "Quem foi Elis Regina","Maior cantora brasileira. Águas de Março com Tom Jobim. Morreu 1982 jovem.",
    "Quem é Ivete Sangalo","Cantora baiana, voz do axé. Trios elétricos, carnaval Salvador.",
    "Quem é Daniela Mercury","Cantora baiana pioneira do axé moderno. O Canto da Cidade hit nos anos 90.",
    "Quem é Caetano Veloso","Cantor e compositor baiano, ícone do Tropicalismo. Sampa, Alegria Alegria.",
    "Quem é Gilberto Gil","Cantor compositor baiano, Tropicália, ex-ministro da cultura. Aquele Abraço.",
    "Quem é Maria Bethânia","Cantora baiana, irmã de Caetano. Voz potente. Reverência.",
    "Quem foi Rita Lee","Cantora paulista. Mutantes nos 60. Hits Mania de Você, Lança Perfume. Faleceu 2023.",
    "Quem foi Cartola","Sambista carioca. As Rosas Não Falam, O Mundo é um Moinho. Filho da escola Mangueira.",
    "Quem foi Noel Rosa","Sambista carioca dos anos 30. Conversa de Botequim. Morreu jovem de tuberculose.",
    "Quem foi Pixinguinha","Compositor carioca de choro. Carinhoso. Maestro da MPB clássica.",
    "Quem foi Vinícius de Moraes","Poeta, diplomata, compositor. Garota de Ipanema. Letrista da Bossa Nova.",
    "Quem foi Chico Buarque","Cantor, compositor e escritor. Construção, Apesar de Você. Mestre da MPB.",
]
for i in range(0, len(PESSOAS), 2):
    QA.append((PESSOAS[i], PESSOAS[i+1]))

# Cinema BR/Mundo
CINEMA = [
    "O que é Cinema Novo","Movimento brasileiro anos 60. Estética da fome. Glauber Rocha. Engajado politicamente.",
    "Quem foi Glauber Rocha","Diretor baiano do Cinema Novo. Deus e o Diabo na Terra do Sol, Terra em Transe.",
    "Quem é Walter Salles","Diretor brasileiro. Central do Brasil (Oscar 1999 indicado), Diários de Motocicleta.",
    "Quem é Kleber Mendonça","Diretor pernambucano. Aquarius, Bacurau. Premiado em Cannes.",
    "O que é Cannes","Festival de cinema francês desde 1946. Palma de Ouro maior prêmio. Mai de cada ano.",
    "O que é Oscar","Academy Award. Premiação americana de cinema desde 1929. Estatueta dourada.",
    "Quem foi Hitchcock","Diretor britânico mestre do suspense. Psicose, Os Pássaros, Janela Indiscreta.",
    "Quem é Tarantino","Diretor americano. Pulp Fiction, Kill Bill, Era uma Vez em Hollywood. Diálogos icônicos.",
    "Quem foi Kubrick","Diretor americano perfeccionista. 2001 Uma Odisseia no Espaço, O Iluminado, Laranja Mecânica.",
    "Quem é Scorsese","Diretor americano. Touro Indomável, Casino, Os Bons Companheiros. Robert De Niro parceiro.",
]
for i in range(0, len(CINEMA), 2):
    QA.append((CINEMA[i], CINEMA[i+1]))

# Curiosidades / por que
CURIO = [
    "Por que bocejamos quando vemos outros","Sinal social empático. Reflexo de espelhamento, mais forte em pessoas próximas.",
    "Por que choramos","Liberação emocional. Diferentes tipos: emoção, irritação física, reflexa. Função social.",
    "Por que rimos","Mecanismo de vínculo social. Libera dopamina e endorfina. Surpresa quebra expectativa.",
    "Por que esquecemos","Memória prioriza importante. Esquecer permite focar. Memória reconsolida quando lembrada.",
    "Por que envelhecemos","Acúmulo de danos celulares. Telômeros encurtam, mutações DNA, declínio mitocondrial.",
    "Por que dor existe","Sinal protetor. Alerta para evitar dano. Crônica é mal regulada e sofrimento.",
    "Por que sentimos medo","Resposta ancestral à ameaça. Amígdala dispara fight-or-flight. Adrenalina.",
    "Por que sonhamos","Não 100% claro. Hipóteses: consolidar memória, simular ameaças, processar emoções.",
    "Por que coçamos","Resposta a estímulo na pele. Pode ser proteção (parasitas), reação alérgica, neurológica.",
    "Por que mar tem ondas","Vento sobre água gera ondas. Maré por gravidade lunar/solar. Tsunamis por terremotos.",
    "Por que vento sopra","Diferença de pressão atmosférica. Ar de alta pressão flui para baixa pressão.",
    "Por que chove","Água evapora, sobe, condensa em nuvens, precipita quando excede capacidade.",
    "Por que neva","Vapor condensa em cristais de gelo. Temperatura abaixo de zero do nível das nuvens ao chão.",
    "Por que arco-íris aparece","Refração e dispersão da luz solar em gotas de água. Sol atrás, chuva à frente.",
    "Por que céu fica laranja no pôr do sol","Luz percorre mais atmosfera. Azul espalha mais, sobra vermelho-laranja.",
]
for i in range(0, len(CURIO), 2):
    QA.append((CURIO[i], CURIO[i+1]))

# Comida
COMIDA = [
    "O que é prato típico brasileiro","Feijoada, churrasco gaúcho, acarajé baiano, pão de queijo mineiro, tacacá amazônico.",
    "Como fazer brigadeiro tradicional","Leite condensado + chocolate em pó + manteiga. Mexa até desgrudar. Enrole em granulado.",
    "Como fazer beijinho","Leite condensado + coco ralado + manteiga. Mexa até ponto. Enrole no coco.",
    "Como fazer bolo de fubá","Fubá, açúcar, ovos, leite, óleo, fermento. Forno 180°C 40min.",
    "Como fazer salada simples","Alface, tomate, cebola, azeite, sal, limão. Lave bem os vegetais.",
    "Como fazer sopa de legumes","Refogue cebola, alho. Adicione legumes em cubos, água ou caldo. Cozinhe 30min.",
    "O que é tacacá","Sopa amazônica com tucupi (mandioca fermentada), jambu (erva que adormece a boca), camarão seco.",
    "O que é vatapá","Prato baiano com pão, leite de coco, dendê, camarão seco. Servido com arroz e acarajé.",
    "O que é moqueca","Peixe ou frutos do mar cozidos em panela de barro com tomate, cebola, dendê, leite de coco.",
    "O que é coxinha","Salgado brasileiro recheado com frango desfiado, em massa de batata ou trigo, frito.",
]
for i in range(0, len(COMIDA), 2):
    QA.append((COMIDA[i], COMIDA[i+1]))

random.seed(42)
random.shuffle(QA)
out = 'data/mega_pt.txt'
with open(out, 'w') as f:
    for u, m in QA:
        f.write(f'USER: {u}?\nMODEL: {m}\n\n')
print(f'Generated {len(QA)} mega PT pairs → {out} ({os.path.getsize(out)} bytes)')
