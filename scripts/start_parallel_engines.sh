#!/bin/bash
# Start N iara-engine instances in parallel for Qwen pool.
# Usage: ./start_parallel_engines.sh [N=3]
#
# Each engine binds HTTP on 3000+i, WS disabled except first.
# Returns after all N engines healthy + Qwen loaded.

set -e
N=${1:-3}
ENGINE_DIR="/home/leonardo/projects/LLM/llm-lab"
ENGINE_BIN="$ENGINE_DIR/target/release/iara-engine"
MODELS_DIR="$ENGINE_DIR/models"
LOG_DIR="/tmp"

# Kill old engines
echo "▸ Killing any existing iara-engine..."
pkill -9 iara-engine 2>/dev/null || true
sleep 3

# ROCm env
export LD_LIBRARY_PATH="$ENGINE_DIR/vendor/llama-cpp/lib:/opt/rocm-7.2.1/lib:$LD_LIBRARY_PATH"
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export ROCR_VISIBLE_DEVICES=0

# Start N engines
echo "▸ Starting $N parallel engines..."
PIDS=()
for i in $(seq 0 $((N-1))); do
    PORT=$((3000 + i))
    WS_PORT=$((8080 + i))
    if [ $i -eq 0 ]; then
        IARA_HTTP_PORT="$PORT" IARA_WS_PORT=$WS_PORT \
            nohup "$ENGINE_BIN" "$MODELS_DIR" > "$LOG_DIR/iara_engine_$PORT.log" 2>&1 &
    else
        IARA_HTTP_PORT="$PORT" IARA_WS_DISABLE=1 \
            nohup "$ENGINE_BIN" "$MODELS_DIR" > "$LOG_DIR/iara_engine_$PORT.log" 2>&1 &
    fi
    PIDS+=($!)
    echo "  engine $i: port $PORT (pid $!)"
done

# Stagger model loads — avoid VRAM thrash
sleep 8
echo "▸ Loading Qwen on each engine..."
for i in $(seq 0 $((N-1))); do
    PORT=$((3000 + i))
    # Wait health
    for try in 1 2 3 4 5 6 7 8 9 10; do
        if curl -s --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    echo "  → loading on $PORT..."
    curl -s -X POST "http://127.0.0.1:$PORT/api/load-model" \
        -H 'Content-Type: application/json' \
        -d '{"name":"qwen2.5-1.5b"}' --max-time 10 >/dev/null
    sleep 6  # stagger to not blast VRAM allocator
done

# Wait all ready
echo "▸ Waiting all models ready..."
for i in $(seq 0 $((N-1))); do
    PORT=$((3000 + i))
    for try in $(seq 1 60); do
        STATUS=$(curl -s --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null \
            | python3 -c "import json,sys; d=json.load(sys.stdin); print('READY' if d.get('model_loading')==False and d.get('model')!='none' else 'LOADING')" 2>/dev/null)
        if [ "$STATUS" = "READY" ]; then
            echo "  port $PORT: READY"
            break
        fi
        sleep 2
    done
done

# Verify each works
echo "▸ Smoke test each engine..."
for i in $(seq 0 $((N-1))); do
    PORT=$((3000 + i))
    R=$(curl -s --max-time 30 -X POST "http://127.0.0.1:$PORT/api/generate" \
        -H 'Content-Type: application/json' \
        -d '{"model":"qwen2.5-1.5b","prompt":"USER: oi\nMODEL: ","max_tokens":15,"temperature":0.5}' 2>&1)
    OK=$(echo "$R" | python3 -c "import json,sys; d=json.load(sys.stdin); print('OK' if len(d.get('response',''))>2 else 'EMPTY')" 2>/dev/null)
    echo "  port $PORT: $OK"
done

# Final VRAM check
echo
echo "▸ Final VRAM:"
rocm-smi --showmeminfo vram 2>/dev/null | grep -E "Used|Total" | head -2

echo
echo "✓ $N engines paralelos rodando. Use IARA_QWEN_PORTS=$(seq -s, 3000 $((3000+N-1)))"
