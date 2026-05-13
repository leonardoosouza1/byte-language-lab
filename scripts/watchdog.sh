#!/usr/bin/env bash
# watchdog.sh — mantém HDC chat ecosystem vivo + rebuild index periódico.
#
# Loop:
#   1. Checa sustained Qwen augment, restart se morreu
#   2. A cada 30min, rebuild corpus_index.bin com dados frescos
#   3. Restart iara_chat_server_indexed
#   4. Log em /tmp/watchdog.log

set -u
cd "$(dirname "$0")"

LOG=/tmp/watchdog.log
INTERVAL_REBUILD=1800   # 30 min
INTERVAL_CHECK=60       # 1 min

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

ensure_sustained() {
    # Não re-inicia se target já atingido (sustained termina cleanly)
    if ! pgrep -af "qwen_augment_sustained.py" >/dev/null; then
        local n=$(grep -c "^USER:" data/qwen_sustained.txt 2>/dev/null || echo 0)
        if [ "$n" -ge 5700 ]; then
            return 0   # Target reached, stay down
        fi
        log "▸ sustained MORTO (n=$n < 5700), reiniciando..."
        nohup python3 data/qwen_augment_sustained.py > /tmp/qwen_sustained.log 2>&1 &
    fi
}

ensure_server() {
    local port="$1" bin="$2"
    if ! curl -s --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
        log "▸ server :$port MORTO, reiniciando..."
        nohup "./target/release/$bin" > "/tmp/server_${port}.log" 2>&1 &
    fi
}

rebuild_index() {
    log "▸ Rebuild corpus index..."
    if ./target/release/build_corpus_index >> "$LOG" 2>&1; then
        log "  ✓ index rebuilt, restarting indexed server"
        pkill -f iara_chat_server_indexed 2>/dev/null
        sleep 2
        nohup ./target/release/iara_chat_server_indexed > /tmp/server_3011.log 2>&1 &
    else
        log "  ✗ rebuild falhou"
    fi
}

log "═══ watchdog iniciado ═══"
last_rebuild=$(date +%s)

while true; do
    ensure_sustained
    ensure_server 3010 iara_chat_server
    ensure_server 3011 iara_chat_server_indexed

    now=$(date +%s)
    elapsed=$((now - last_rebuild))
    if [ $elapsed -ge $INTERVAL_REBUILD ]; then
        rebuild_index
        last_rebuild=$now
    fi

    # corpus size
    if [ -f data/qwen_sustained.txt ]; then
        n=$(grep -c "^USER:" data/qwen_sustained.txt 2>/dev/null || echo 0)
        log "tick. sustained=$n pares | next rebuild em $((INTERVAL_REBUILD - elapsed))s"
    fi

    sleep "$INTERVAL_CHECK"
done
