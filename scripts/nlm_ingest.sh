#!/bin/bash
# Usage: ./scripts/nlm_ingest.sh "pergunta NLM"
set -e
NOTEBOOK="77cf63a8-1ff5-483e-b936-bc0484f085ce"
QUERY="${1:-Gere 20 pares USER:/MODEL: sobre conhecimento geral PT-BR}"
TS=$(date +%s)
OUTFILE="data/nlm_${TS}.txt"

cd "$(dirname "$0")/.."

echo "▸ Querying NLM..."
echo "  Q: $QUERY"
nlm query notebook "$NOTEBOOK" "$QUERY" --json > /tmp/nlm_${TS}.json 2>/dev/null

ANSWER=$(python3 -c "
import json
d=json.load(open('/tmp/nlm_${TS}.json'))
print(d['value']['answer'])
")

# Extract USER:/MODEL: pairs only
echo "$ANSWER" | python3 -c "
import sys, re
text = sys.stdin.read()
pairs = re.findall(r'(USER:[^\n]+)\n(MODEL:[^\n]+)', text)
for u, m in pairs:
    print(u)
    print(m)
    print()
" > "$OUTFILE"

LINES=$(wc -l < "$OUTFILE")
echo "  ✓ Salvos $LINES lines → $OUTFILE"
echo
echo "▸ Amostra (primeiros 6 lines):"
head -6 "$OUTFILE"
echo
echo "▸ Ingerindo no brain..."
./target/release/iara_brain_full ingest "$OUTFILE"
