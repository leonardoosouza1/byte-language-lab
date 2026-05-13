#!/bin/bash
# Qwen mass distillation — gera Q&A pairs via local Qwen
# Brain absorve cada 100 pairs
set -e
QWEN="http://localhost:3000/api/generate"
TS=$(date +%s)
OUT="data/qwen_distill_${TS}.txt"
cd "$(dirname "$0")/.."
> "$OUT"

# Pool de perguntas-template — Qwen completa
QUESTIONS=(
  "o que é entropia"
  "como funciona um motor a combustão"
  "o que é energia solar"
  "como funciona um foguete"
  "o que é fissão nuclear"
  "o que é fusão nuclear"
  "o que é raio X"
  "como funciona um microondas"
  "como funciona uma geladeira"
  "o que é semicondutor"
  "como funciona um transistor"
  "o que é computador quântico"
  "o que é IA generativa"
  "como funciona uma rede neural"
  "o que é deep learning"
  "o que é reinforcement learning"
  "o que é gradient descent"
  "o que é backpropagation"
  "o que é autoencoder"
  "o que é GAN"
  "como funciona reconhecimento de imagem"
  "como funciona reconhecimento de fala"
  "o que é processamento de linguagem natural"
  "o que é tokenização"
  "o que é embedding"
  "como funciona transformer"
  "o que é atenção em redes neurais"
  "o que é fine-tuning"
  "o que é prompt engineering"
  "o que é retrieval augmented generation"
  "quem foi Newton"
  "quem foi Galileu"
  "quem foi Marie Curie"
  "quem foi Tesla"
  "quem foi Edison"
  "quem foi Da Vinci"
  "quem foi Hawking"
  "quem foi Turing"
  "quem foi Ada Lovelace"
  "quem foi Linus Torvalds"
  "como funciona o coração humano"
  "como funciona o cérebro humano"
  "como funciona o sistema imunológico"
  "como funciona o sistema nervoso"
  "como funciona a digestão"
  "como funciona a respiração"
  "como funciona o sangue"
  "como funcionam os hormônios"
  "o que é diabetes detalhe"
  "o que é hipertensão detalhe"
  "como funciona uma vacina"
  "o que é antibiótico"
  "o que é vírus"
  "o que é bactéria"
  "diferença entre vírus e bactéria"
  "como funciona o DNA"
  "como funciona o RNA"
  "o que é gene"
  "o que é hereditariedade"
  "o que é evolução"
  "o que é mecânica clássica"
  "o que é onda eletromagnética"
  "o que é frequência"
  "o que é amplitude"
  "o que é resonância"
  "o que é interferência"
  "o que é refração"
  "o que é difração"
  "o que é polarização"
  "como funciona som"
  "como funciona luz"
  "o que é arco-íris"
  "como nuvens se formam"
  "como chove"
  "como neva"
  "como funciona vulcão"
  "como funciona terremoto"
  "como funciona maremoto"
  "como funciona furacão"
  "como funciona tornado"
  "o que é El Niño"
  "o que é La Niña"
  "o que é efeito estufa"
  "o que é aquecimento global"
  "o que é camada de ozônio"
  "o que é poluição"
  "o que é sustentabilidade"
  "o que é desenvolvimento sustentável"
  "o que é ESG"
  "o que é economia circular"
  "o que é energia renovável"
  "o que é energia eólica"
  "o que é energia hidrelétrica"
  "o que é biomassa"
  "o que é etanol"
  "o que é hidrogênio verde"
  "o que é carro elétrico"
  "o que é carro autônomo"
  "o que é blockchain detalhe"
  "o que é bitcoin"
  "o que é ethereum"
  "o que é NFT"
  "o que é metaverso"
)

count=0
for q in "${QUESTIONS[@]}"; do
  count=$((count+1))
  RESP=$(curl -s -X POST "$QWEN" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen2.5-1.5b\",\"prompt\":\"USER: $q\nMODEL: \",\"max_tokens\":120,\"temperature\":0.5}" \
    2>/dev/null | python3 -c "
import json, sys, re
try:
    d = json.load(sys.stdin)
    r = d.get('response', '').strip()
    # Stop at next USER: or limit length
    r = r.split('USER:')[0].strip()
    r = re.sub(r'\s+', ' ', r)
    r = r[:280]
    if len(r) >= 20:
        print(r)
except: pass
")
  if [ -n "$RESP" ]; then
    echo "USER: $q" >> "$OUT"
    echo "MODEL: $RESP" >> "$OUT"
    echo "" >> "$OUT"
  fi
  if [ $((count % 10)) -eq 0 ]; then
    P=$(grep -c "^USER:" "$OUT")
    echo "[$count/${#QUESTIONS[@]}] pairs salvos: $P"
  fi
done

P=$(grep -c "^USER:" "$OUT")
B=$(wc -c < "$OUT")
echo "✓ Qwen distill done: $P pairs, $B bytes → $OUT"
echo "Ingerindo..."
./target/release/iara_brain_full ingest "$OUT" 2>&1 | tail -3
