#!/usr/bin/env bash
# DEMO.sh — roda demo completo do IARA HDC chat ecosystem.
#
# Pré-requisitos:
#   - cargo build --release feito
#   - iara_chat_server rodando em :3010 (opcional pra HTTP demos)
#   - byte_hvs_qwen.bin existe (auto-gera na primeira run)

set -e
cd "$(dirname "$0")"

GRN='\033[1;32m'
BLU='\033[1;34m'
DIM='\033[2m'
RST='\033[0m'

section() { echo -e "\n${BLU}━━━ $1 ━━━${RST}"; }

section "1. Build tudo"
cargo build --release --bin iara_hdc_chat_v2 --bin iara_chat_rerank \
                     --bin iara_chat_server --bin iara_intelligent_chat \
                     --bin byte_harmony --bin byte_walker \
                     --bin byte_graph_communities \
                     --bin iara_chat_eval --bin iara_typo_eval \
                     --bin adversarial_test --bin stress_test 2>&1 | tail -3

section "2. byte_harmony — geometria bit dos caracteres ASCII"
./target/release/byte_harmony 2>&1 | tail -25

section "3. byte_graph_communities — clusters emergentes do corpus"
./target/release/byte_graph_communities 2>&1 | tail -25

section "4. byte_walker — palavras sintéticas via grafo markoviano"
./target/release/byte_walker 2>&1 | tail -15

section "5. iara_chat_eval — comparação variantes em 40 prompts"
./target/release/iara_chat_eval 2>&1 | tail -5

section "6. Demo chat interativo (iara_chat_rerank)"
echo -e "oi\nO que é Rust?\nfeijoada\nfejoada (typo)\nronaldinho gaucho\n/quit" | \
    timeout 30 ./target/release/iara_chat_rerank 2>&1 | grep -E "MODEL:|match" | head -20

if curl -s --max-time 2 http://127.0.0.1:3010/health >/dev/null 2>&1; then
    section "7. HTTP server stress test (1000 queries)"
    ./target/release/stress_test 2>&1 | tail -15

    section "8. HTTP server adversarial test (~37 queries)"
    ./target/release/adversarial_test 2>&1 | grep -E "★|hello|case|typo-mix|emoji" | head -10
else
    echo "${DIM}⚠ iara_chat_server não está rodando em :3010, pulando HTTP tests.${RST}"
    echo "  Suba com: ./target/release/iara_chat_server &"
fi

section "9. Estado final"
echo "Corpus files:"
ls -la data/*.txt 2>/dev/null | tail -5
echo ""
echo "Binários disponíveis:"
ls -la target/release/iara_* target/release/byte_* target/release/stress_test target/release/adversarial_test 2>/dev/null | awk '{print "  " $NF}' | sort -u | head -20

echo -e "\n${GRN}━━━ DEMO COMPLETO ━━━${RST}\n"
