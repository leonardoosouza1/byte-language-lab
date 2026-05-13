#!/bin/bash
# Wave 2 — 30 topics paralelo
set -e
NB="77cf63a8-1ff5-483e-b936-bc0484f085ce"
TS=$(date +%s)
OUT="data/mass_v2_${TS}.txt"
cd "$(dirname "$0")/.."

TOPICS=(
  "tecnologia avançada: redes neurais, deep learning, transformers, atenção, embedding, tokenização"
  "linguagens: latim, grego, sânscrito, indo-europeu, etimologia, fonética"
  "astronomia: planetas, estrelas, galáxia, via láctea, sistema solar, eclipse, cometa"
  "química orgânica: hidrocarbonetos, álcool, ácidos graxos, proteínas, aminoácidos"
  "engenharia: civil, elétrica, mecânica, química, software, materiais"
  "economia: PIB, inflação, taxa de juros, mercado, oferta demanda, microeconomia, macro"
  "psicologia: comportamento, cognição, freud, jung, behaviorismo, cognitivismo"
  "neurologia: neurônios, sinapse, neurotransmissor, dopamina, serotonina, plasticidade"
  "medicina: anatomia humana, fisiologia, sistema cardiovascular, respiratório, digestivo"
  "direito brasileiro: constituição, código civil, código penal, STF, jurisprudência"
  "biologia marinha: oceano, peixes, mamíferos marinhos, recife coral, plâncton"
  "geologia: rochas, vulcões, terremotos, placas tectônicas, fósseis, eras geológicas"
  "matemática avançada: cálculo, álgebra linear, topologia, teoria dos números, fractais"
  "estatística: probabilidade, distribuição normal, hipótese, regressão, correlação"
  "literatura mundial: Shakespeare, Dostoiévski, Tolstói, Cervantes, García Márquez"
  "cinema: Hollywood, Cinema Novo brasileiro, gêneros, Oscar, Cannes, Glauber Rocha"
  "moda: estilismo, alta costura, brasileira, jeans, sustentabilidade, Coco Chanel"
  "arquitetura: barroco, modernismo, Niemeyer, Aleijadinho, brasileira colonial"
  "religiões: cristianismo, islamismo, budismo, hinduísmo, judaísmo, candomblé, umbanda"
  "mitologia: grega, romana, nórdica, egípcia, indígena brasileira, africana"
  "história antiga: Egito, Mesopotâmia, Grécia, Roma, China antiga, Inca, Maia, Asteca"
  "história moderna: Iluminismo, Revolução Industrial, colonialismo, descolonização"
  "tecnologia moderna: smartphone, internet das coisas, blockchain, criptomoedas, 5G"
  "ciência da computação: algoritmos, complexidade, máquinas Turing, autômatos"
  "redes computador: TCP/IP, DNS, HTTP, SSL, firewall, roteamento, OSI"
  "sistemas operacionais: Linux, Windows, macOS, kernel, processo, thread, memória virtual"
  "bancos de dados: relacional, NoSQL, MongoDB, Redis, PostgreSQL, índices, transação"
  "segurança: criptografia, hash, AES, RSA, autenticação, autorização, OWASP"
  "DevOps: CI/CD, Kubernetes, Terraform, Ansible, monitoring, observability"
  "PT-BR linguagem: substantivo, verbo, adjetivo, pronome, regência, concordância, ortografia"
)

> "$OUT"
count=0
for t in "${TOPICS[@]}"; do
  count=$((count+1))
  echo "[$count/${#TOPICS[@]}] $(echo "$t" | head -c 60)..."
  Q="Em PT-BR gere 10 Q&A factuais sobre: $t. FORMATO 2 linhas: USER: pergunta?\\nMODEL: resposta 1-2 frases. Sem citações [1], sem intro."
  RAW=$(nlm query notebook "$NB" "$Q" --json 2>/dev/null || echo '{}')
  echo "$RAW" | python3 -c "
import json, re, sys
try:
    d = json.loads(sys.stdin.read())
    text = d.get('value', {}).get('answer', '')
    text = re.sub(r'\s*\[[\d,\s\-]+\]', '', text)
    pairs = re.findall(r'(USER:[^\n]+)\n(MODEL:[^\n]+)', text)
    for u, m in pairs:
        print(u.strip()); print(m.strip()); print()
except: pass
" >> "$OUT"
  PAIRS=$(grep -c '^USER:' "$OUT")
  echo "   Total: $PAIRS pairs"
done
echo "✓ Wave 2 done: $(grep -c '^USER:' "$OUT") pairs → $OUT"
