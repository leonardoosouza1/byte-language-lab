#!/bin/bash
# Massive NLM research — 50 topics across multiple domains
set -e
NB="77cf63a8-1ff5-483e-b936-bc0484f085ce"
TS=$(date +%s)
OUT="data/mass_research_${TS}.txt"
cd "$(dirname "$0")/.."

TOPICS=(
  "ciência: fotossíntese, mitocondria, núcleo celular, DNA, RNA, ribossomos, evolução, seleção natural, cromossomos, proteínas"
  "física: gravidade, energia cinética, energia potencial, força, aceleração, velocidade, atrito, eletromagnetismo, radiação, ondas"
  "química: tabela periódica, elementos, hidrogênio, oxigênio, carbono, ácido, base, pH, sais, ligação iônica"
  "matemática: número primo, números reais, racional, irracional, derivada, integral, função, equação, geometria, trigonometria"
  "história Brasil: descobrimento, colonização, independência, república, Vargas, ditadura militar, redemocratização, Plano Real"
  "história mundial: revolução francesa, revolução industrial, primeira guerra, segunda guerra, guerra fria, queda muro Berlim"
  "literatura Brasil: Machado de Assis, José de Alencar, Clarice Lispector, Jorge Amado, Guimarães Rosa, Drummond"
  "música brasileira: bossa nova, samba, MPB, forró, sertanejo, Tom Jobim, João Gilberto, Caetano Veloso, Gilberto Gil"
  "geografia mundial: continentes, oceanos, países, capitais, montanhas, rios, climas, biomas"
  "tecnologia: internet, computadores, processador, memória, software, hardware, algoritmo, dado, rede, protocolo"
  "programação: variável, função, classe, herança, recursão, loop, condicional, array, hash, stack"
  "biologia: célula, tecido, órgão, sistema, fisiologia, anatomia, evolução, ecologia, genética, microbiologia"
  "filosofia: Sócrates, Platão, Aristóteles, Descartes, Kant, Nietzsche, existencialismo, ética, lógica, epistemologia"
  "esporte: futebol, copa do mundo, olimpíadas, vôlei, basquete, tênis, fórmula 1, atletismo"
  "PT-BR diálogos: como cumprimentar, agradecer, pedir desculpa, despedir, fazer pergunta, dar opinião"
)

> "$OUT"
count=0
for t in "${TOPICS[@]}"; do
  count=$((count+1))
  echo "[$count/15] Researching: ${t:0:80}..."
  Q="Em PT-BR, gere 10 pares Q&A factuais sobre: $t. FORMATO ESTRITO 2 linhas cada par: USER: pergunta?\\nMODEL: resposta 1-2 frases. Sem citações, sem introdução, sem [1]."
  RAW=$(nlm query notebook "$NB" "$Q" --json 2>/dev/null || echo '{}')
  echo "$RAW" | python3 -c "
import json, re, sys
try:
    d = json.loads(sys.stdin.read())
    text = d.get('value', {}).get('answer', '')
    text = re.sub(r'\s*\[[\d,\s\-]+\]', '', text)
    pairs = re.findall(r'(USER:[^\n]+)\n(MODEL:[^\n]+)', text)
    for u, m in pairs:
        print(u.strip())
        print(m.strip())
        print()
except: pass
" >> "$OUT"
  PAIRS=$(grep -c '^USER:' "$OUT")
  echo "   Total pairs so far: $PAIRS"
  sleep 1
done

PAIRS=$(grep -c '^USER:' "$OUT")
BYTES=$(wc -c < "$OUT")
echo
echo "═══════════════════════════════════════════════════"
echo "  ✓ Total: $PAIRS pares, $BYTES bytes → $OUT"
echo "═══════════════════════════════════════════════════"
