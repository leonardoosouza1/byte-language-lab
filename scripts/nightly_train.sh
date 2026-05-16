#!/bin/bash
# nightly_train.sh — SAFE autonomous overnight training loop for the IARA brain.
#
# Each cycle: auto_eval (Qwen3 examiner+judge) → analyze+teach → /learn →
# /checkpoint → log. The brain learns from its failures every cycle.
#
# SAFETY (this loop must never freeze the machine):
#   - MAX_CYCLES hard cap — stops itself, never truly infinite
#   - RAM guard: aborts cleanly if available memory drops below MIN_FREE_MB
#   - VRAM guard: restarts engine if GPU free memory collapses
#   - single-instance lock — refuses to run if another copy is live
#   - /tmp cleanup — keeps only the last KEEP_CYCLES cycle artifacts
#   - runs at nice 19 / ionice idle — desktop stays responsive
#   - per-command timeouts — no hung subprocess can stall the loop
#   - consecutive-failure breaker — stops after MAX_FAILS bad cycles
#   - generous inter-cycle sleep — GPU/CPU breather
#
# Usage:  nohup ./scripts/nightly_train.sh > /tmp/nightly.log 2>&1 &
# Stop:   touch /tmp/nightly_train.stop      (graceful, finishes current cycle)
#         or: pkill -f nightly_train.sh

set -u

# ── tunables ────────────────────────────────────────────────────────────────
MAX_CYCLES=25            # hard cap (~6-7h at ~15min/cycle)
MIN_FREE_MB=2048         # abort if available RAM drops below this
MIN_VRAM_GB=1.0          # restart engine if free VRAM below this
KEEP_CYCLES=3            # how many cycles of /tmp artifacts to retain
INTER_CYCLE_SLEEP=90     # seconds between cycles (breather)
MAX_FAILS=3              # consecutive failed cycles → stop
PER_DOMAIN=6             # questions per domain in each battery

LAB="/home/leonardo/projects/LLM/llm-lab"
BLAB="/home/leonardo/projects/LLM/byte-language-lab"
PROGRESS="/tmp/nightly_progress.log"
LOCK="/tmp/nightly_train.lock"
STOPFILE="/tmp/nightly_train.stop"
cd "$BLAB" || exit 1

export LD_LIBRARY_PATH="$LAB/vendor/llama-cpp/lib:/opt/rocm-7.2.1/lib"
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0

log() {
    local m="[$(date +%H:%M:%S)] $*"
    echo "$m"
    echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" >> "$PROGRESS"
}

# ── single-instance lock ────────────────────────────────────────────────────
if [ -f "$LOCK" ]; then
    OLDPID=$(cat "$LOCK" 2>/dev/null)
    if [ -n "$OLDPID" ] && kill -0 "$OLDPID" 2>/dev/null; then
        echo "another nightly_train running (pid $OLDPID) — abort"
        exit 1
    fi
fi
echo $$ > "$LOCK"
rm -f "$STOPFILE"
trap 'rm -f "$LOCK"; log "loop exited (trap)"' EXIT

# ── lower priority so the desktop never stutters ────────────────────────────
renice 19 $$ >/dev/null 2>&1 || true
ionice -c 3 -p $$ >/dev/null 2>&1 || true

free_mb()  { free -m | awk '/^Mem:/{print $7}'; }
vram_free() {
    curl -s --max-time 5 http://localhost:3000/health 2>/dev/null \
        | python3 -c "import sys,json;print(json.load(sys.stdin).get('vram_free_gb',0))" 2>/dev/null \
        || echo 0
}

ensure_engine() {
    local up vfree
    up=$(curl -s --max-time 5 http://localhost:3000/health 2>/dev/null | grep -c '"status"')
    vfree=$(vram_free)
    if [ "$up" = "0" ] || python3 -c "import sys;sys.exit(0 if float('$vfree')<$MIN_VRAM_GB else 1)" 2>/dev/null; then
        log "engine down or low VRAM (${vfree}GB) — restarting"
        pkill -9 -f iara-engine 2>/dev/null; sleep 4
        ( cd "$LAB" && setsid nice -n 10 ./target/release/iara-engine "$LAB/models" \
            >/tmp/nt_engine.log 2>&1 < /dev/null & )
        for _ in $(seq 1 60); do
            curl -s --max-time 3 http://localhost:3000/health 2>/dev/null | grep -q '"status"' && break
            sleep 2
        done
    fi
    local m
    m=$(curl -s --max-time 5 http://localhost:3000/health 2>/dev/null \
        | python3 -c "import sys,json;print(json.load(sys.stdin).get('model','none'))" 2>/dev/null)
    if [ "$m" = "none" ] || [ -z "$m" ]; then
        log "loading qwen3-4b"
        curl -s -X POST http://localhost:3000/api/load-model \
            -H 'Content-Type: application/json' -d '{"name":"qwen3-4b"}' >/dev/null 2>&1
        for _ in $(seq 1 90); do
            [ "$(curl -s http://localhost:3000/health 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin).get("model_loading"))' 2>/dev/null)" = "False" ] && break
            sleep 3
        done
    fi
}

ensure_brain() {
    if ! curl -s --max-time 5 http://localhost:3012/stats 2>/dev/null | grep -q bytes; then
        log "brain down — restarting"
        pkill -9 -f iara_brain_full 2>/dev/null; sleep 3
        ( setsid nice -n 10 ./target/release/iara_brain_full serve \
            >/tmp/nt_brain.log 2>&1 < /dev/null & )
        for _ in $(seq 1 90); do
            curl -s --max-time 3 http://localhost:3012/stats 2>/dev/null | grep -q bytes && break
            sleep 3
        done
    fi
}

cleanup_tmp() {
    # keep only the last KEEP_CYCLES cycles of artifacts
    local cutoff=$((CYCLE - KEEP_CYCLES))
    [ "$cutoff" -lt 1 ] && return
    for c in $(seq 1 "$cutoff"); do
        rm -f "/tmp/nt_eval_${c}.jsonl" "/tmp/nt_eval_${c}.txt" "/tmp/nt_teach_${c}.txt"
    done
}

log "════ nightly_train START (max ${MAX_CYCLES} cycles) ════"
CYCLE=0
FAILS=0

while [ "$CYCLE" -lt "$MAX_CYCLES" ]; do
    CYCLE=$((CYCLE + 1))

    # graceful stop
    if [ -f "$STOPFILE" ]; then
        log "stop file found — finishing"
        break
    fi

    # ── RAM guard ───────────────────────────────────────────────────────────
    FREE=$(free_mb)
    if [ "${FREE:-0}" -lt "$MIN_FREE_MB" ]; then
        log "RAM low (${FREE}MB < ${MIN_FREE_MB}MB) — stopping to protect the machine"
        break
    fi

    log "── cycle $CYCLE/$MAX_CYCLES (free RAM ${FREE}MB) ──"

    ensure_engine
    ensure_brain

    EVAL_JSONL="/tmp/nt_eval_${CYCLE}.jsonl"
    EVAL_TXT="/tmp/nt_eval_${CYCLE}.txt"
    TEACH="$BLAB/data/auto_learn.txt"

    # 1. battery (timeout-guarded — 30min ceiling)
    timeout 1800 python3 scripts/auto_eval.py --per "$PER_DOMAIN" --rounds 1 \
        --log "$EVAL_JSONL" > "$EVAL_TXT" 2>&1
    SCORE=$(grep "weighted score" "$EVAL_TXT" 2>/dev/null | sed 's/.*weighted score: //')

    if [ -z "$SCORE" ] || [ ! -s "$EVAL_JSONL" ]; then
        FAILS=$((FAILS + 1))
        log "cycle $CYCLE eval FAILED (${FAILS}/${MAX_FAILS})"
        if [ "$FAILS" -ge "$MAX_FAILS" ]; then
            log "too many consecutive failures — stopping"
            break
        fi
        sleep 60
        continue
    fi
    FAILS=0
    log "cycle $CYCLE eval: $SCORE"

    # 2. teach file from failures
    timeout 900 python3 scripts/analyze_failures.py "$EVAL_JSONL" --teach "$TEACH" \
        > "/tmp/nt_teach_${CYCLE}.txt" 2>&1
    PAIRS=$(grep -c "^USER:" "$TEACH" 2>/dev/null || echo 0)
    log "cycle $CYCLE teach: $PAIRS pairs"

    # 3. brain learns
    if [ -s "$TEACH" ]; then
        LEARN=$(curl -s -X POST http://localhost:3012/learn --max-time 240 2>/dev/null)
        log "cycle $CYCLE learn: ${LEARN:-no-response}"
    fi

    # 4. checkpoint
    curl -s -X POST http://localhost:3012/checkpoint --max-time 90 >/dev/null 2>&1
    CKPT_MB=$(du -m data/brain_v5.bin 2>/dev/null | cut -f1)
    log "cycle $CYCLE checkpoint saved (brain_v5.bin ${CKPT_MB:-?}MB)"

    cleanup_tmp
    sleep "$INTER_CYCLE_SLEEP"
done

log "════ nightly_train DONE after $CYCLE cycles ════"
