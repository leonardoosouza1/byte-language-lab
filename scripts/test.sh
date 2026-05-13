#!/bin/bash

# Configurações
ENDPOINT="http://127.0.0.1:3012"
LISTA="perguntas-1500.txt"
LOG_SRV="/tmp/srv.log"

# Verifica se o arquivo existe
if [ ! -f "$LISTA" ]; then
    echo "❌ Erro: Arquivo $LISTA não encontrado!"
    exit 1
fi

# Função para disparar a pergunta
ask_brain() {
    local query="$1"
    curl -s --max-time 10 -X POST "$ENDPOINT/chat" \
        -H 'Content-Type: application/json; charset=utf-8' \
        --data-binary "{\"prompt\":\"$query\"}" | \
        python3 -c "import json,sys; print(json.load(sys.stdin).get('source','?'))" 2>/dev/null
}

TOTAL_LINHAS=$(wc -l < "$LISTA")
CONTADOR=0

echo "═══ 🚀 Iniciando Processamento de $TOTAL_LINHAS perguntas ═══"
echo "Acompanhe o progresso abaixo:"
echo "------------------------------------------------"

# Loop de leitura otimizado
while IFS= read -r pergunta || [ -n "$pergunta" ]; do
    [ -z "$pergunta" ] && continue # Pula linhas vazias
    
    ((CONTADOR++))
    
    # Executa a requisição
    SRC=$(ask_brain "$pergunta")
    
    # Log de progresso a cada 10 perguntas para não poluir o terminal
    if (( CONTADOR % 10 == 0 )) || (( CONTADOR == TOTAL_LINHAS )); then
        echo "Progress: [$CONTADOR/$TOTAL_LINHAS] | Última: $pergunta -> Source: $SRC"
    fi

done < "$LISTA"

echo "------------------------------------------------"
echo "✅ Disparos concluídos. Aguardando 60s para estabilização do daemon..."
sleep 60

echo -e "\n═══ 🧠 Verificando Taxa de Aprendizado (Stats) ═══"
curl -s "$ENDPOINT/stats"

echo -e "\n═══ 📊 Ciclos de Auto-learn detectados no Log: ═══"
grep "Auto-learn cycle" "$LOG_SRV" | tail -n 5

echo -e "\n🚀 Teste finalizado!"
