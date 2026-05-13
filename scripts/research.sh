#!/bin/bash
# research <topic> — query NLM, parse, save Q&A pairs, ingest into brain
# Usage:
#   ./scripts/research.sh "Albert Einstein"
#   ./scripts/research.sh "blockchain WebAssembly diabetes"
set -e
NOTEBOOK="77cf63a8-1ff5-483e-b936-bc0484f085ce"  # NorBERTo notebook (default source)
TOPIC="${1:-fatos sobre Brasil}"
TS=$(date +%s)
DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTFILE="$DIR/data/nlm_research_${TS}.txt"

cd "$DIR"

echo "▸ NLM research: $TOPIC"
QUERY="Em PT-BR, gere apenas 10 pares de Q&A diretos sobre: $TOPIC. FORMATO ESTRITO: cada par em 2 linhas exatas:
USER: pergunta?
MODEL: resposta em 1-2 frases.

Use linguagem clara, factual. Não inclua introdução, citações ou marcadores. Apenas os 10 pares no formato acima."

nlm query notebook "$NOTEBOOK" "$QUERY" --json > /tmp/nlm_${TS}.json 2>/dev/null

python3 <<EOF > "$OUTFILE"
import json, re
d = json.load(open('/tmp/nlm_${TS}.json'))
text = d['value']['answer']
# Strip citations like [1, 2] and inline references
text = re.sub(r'\s*\[[\d,\s\-]+\]', '', text)
pairs = re.findall(r'(USER:[^\n]+)\n(MODEL:[^\n]+)', text)
for u, m in pairs:
    print(u.strip())
    print(m.strip())
    print()
EOF

LINES=$(wc -l < "$OUTFILE")
PAIRS=$(grep -c "^USER:" "$OUTFILE" || echo 0)
echo "  ✓ Salvos $PAIRS pares ($LINES lines) → $OUTFILE"
echo
echo "▸ Amostra:"
head -6 "$OUTFILE"
echo
echo "▸ Ingerindo no brain..."
./target/release/iara_brain_full ingest "$OUTFILE"
