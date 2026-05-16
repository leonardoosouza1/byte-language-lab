#!/bin/bash
# train_v2.sh — SAFE autonomous training loop, v2.
#
# The v1 loop (nightly_train.sh) DEGRADED the brain: it checkpointed blindly
# every cycle, so noisy teaching accumulated and brain SOLO fell 90%→30%.
#
# v2 fixes that with a HELD-OUT GATE + ROLLBACK:
#   - a frozen 50-question held-out set (scripts/heldout.json) measures REAL
#     progress — it is NEVER taught, so the score is comparable across cycles
#   - after each teach cycle the held-out score is remeasured
#   - if a cycle made the brain WORSE, that cycle is rolled back to the best
#     known checkpoint. The brain can only stay equal or improve — never decay.
#   - noise is purged every cycle (/purge_noise: reasoning leaks, dups, junk)
#   - the teacher (Qwen3-Q4) self-validates each answer before it is taught
#   - battery + judging run in parallel across 2 engines (:3000 + :3002)
#
# SAFETY (must never freeze the machine):
#   MAX_CYCLES cap · RAM guard · VRAM guard · single-instance lock ·
#   nice 19 / ionice idle · per-command timeouts · failure breaker ·
#   /tmp cleanup · graceful stop file · inter-cycle breather.
#
# Usage:  nohup ./scripts/train_v2.sh > data/train_v2.log 2>&1 &
# Stop:   touch /tmp/train_v2.stop      (graceful — finishes current cycle)
#         or: pkill -f train_v2.sh

set -u

# ── tunables (all env-overridable: MAX_CYCLES=1 ./scripts/train_v2.sh) ───────
MAX_CYCLES=${MAX_CYCLES:-20}            # hard cap
MIN_FREE_MB=${MIN_FREE_MB:-2048}        # abort if available RAM drops below this
MIN_VRAM_GB=${MIN_VRAM_GB:-0.8}         # restart engine if free VRAM below this
KEEP_CYCLES=${KEEP_CYCLES:-3}           # cycles of /tmp artifacts to retain
INTER_CYCLE_SLEEP=${INTER_CYCLE_SLEEP:-60}   # breather between cycles (s)
MAX_FAILS=${MAX_FAILS:-3}               # consecutive failed cycles → stop
PER_DOMAIN=${PER_DOMAIN:-6}             # battery questions per domain
STALL_LIMIT=${STALL_LIMIT:-5}           # stop if held-out doesn't improve N cycles
REGRESS_TOL=${REGRESS_TOL:-2.0}         # held-out dip allowed before rollback

LAB="/home/leonardo/projects/LLM/llm-lab"
BLAB="/home/leonardo/projects/LLM/byte-language-lab"
PROGRESS="$BLAB/data/train_v2_progress.log"
LOCK="/tmp/train_v2.lock"
STOPFILE="/tmp/train_v2.stop"
BEST_BIN="$BLAB/data/brain_best.bin"
BRAIN_BIN="$BLAB/data/brain_v5.bin"
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
        echo "another train_v2 running (pid $OLDPID) — abort"; exit 1
    fi
fi
echo $$ > "$LOCK"
rm -f "$STOPFILE"
trap 'rm -f "$LOCK"; log "loop exited (trap)"' EXIT

renice 19 $$ >/dev/null 2>&1 || true
ionice -c 3 -p $$ >/dev/null 2>&1 || true

free_mb()  { free -m | awk '/^Mem:/{print $7}'; }
vram_free() {
    curl -s --max-time 5 http://localhost:3000/health 2>/dev/null \
        | python3 -c "import sys,json;print(json.load(sys.stdin).get('vram_free_gb',0))" 2>/dev/null \
        || echo 0
}

# Restart one engine on a given port. WS is disabled for the 2nd to avoid
# the :8080 conflict; only :3000 keeps the websocket.
start_engine() {
    local port=$1 wsflag=$2
    ( cd "$LAB" && IARA_HTTP_PORT=$port $wsflag setsid nice -n 10 \
        ./target/release/iara-engine "$LAB/models" \
        >"/tmp/tv2_engine_${port}.log" 2>&1 < /dev/null & )
    for _ in $(seq 1 60); do
        curl -s --max-time 3 "http://localhost:$port/health" 2>/dev/null \
            | grep -q '"status"' && break
        sleep 2
    done
    local m
    m=$(curl -s --max-time 5 "http://localhost:$port/health" 2>/dev/null \
        | python3 -c "import sys,json;print(json.load(sys.stdin).get('model','none'))" 2>/dev/null)
    if [ "$m" = "none" ] || [ -z "$m" ]; then
        curl -s -X POST "http://localhost:$port/api/load-model" \
            -H 'Content-Type: application/json' -d '{"name":"qwen3-4b"}' >/dev/null 2>&1
        for _ in $(seq 1 90); do
            [ "$(curl -s http://localhost:$port/health 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin).get("model_loading"))' 2>/dev/null)" = "False" ] && break
            sleep 3
        done
    fi
}

ensure_engines() {
    local up vfree
    for port in 3000 3002; do
        up=$(curl -s --max-time 5 "http://localhost:$port/health" 2>/dev/null | grep -c '"status"')
        if [ "$up" = "0" ]; then
            log "engine :$port down — starting"
            if [ "$port" = "3002" ]; then start_engine 3002 "IARA_WS_DISABLE=1"
            else start_engine 3000 ""; fi
        fi
    done
    vfree=$(vram_free)
    if python3 -c "import sys;sys.exit(0 if float('$vfree')<$MIN_VRAM_GB else 1)" 2>/dev/null; then
        log "VRAM low (${vfree}GB) — restarting both engines"
        pkill -9 -f iara-engine 2>/dev/null; sleep 5
        start_engine 3000 ""
        start_engine 3002 "IARA_WS_DISABLE=1"
    fi
}

ensure_brain() {
    if ! curl -s --max-time 5 http://localhost:3012/stats 2>/dev/null | grep -q bytes; then
        log "brain down — restarting"
        pkill -9 -f iara_brain_full 2>/dev/null; sleep 3
        ( setsid nice -n 10 ./target/release/iara_brain_full serve \
            >"$BLAB/data/brain_serve.log" 2>&1 < /dev/null & )
        for _ in $(seq 1 90); do
            curl -s --max-time 3 http://localhost:3012/stats 2>/dev/null | grep -q bytes && break
            sleep 3
        done
    fi
}

# Held-out score → echoes "SCORE SOLO" (floats). 0 0 on failure.
measure_heldout() {
    local out
    out=$(timeout 900 python3 scripts/heldout_eval.py --quiet \
            --log "/tmp/tv2_heldout_${CYCLE}.jsonl" 2>/dev/null)
    echo "$out" | grep -oE 'HELDOUT_SCORE=[0-9.]+ HELDOUT_SOLO=[0-9.]+' \
        | sed 's/HELDOUT_SCORE=//;s/HELDOUT_SOLO=//' | head -1
}

cleanup_tmp() {
    local cutoff=$((CYCLE - KEEP_CYCLES))
    [ "$cutoff" -lt 1 ] && return
    for c in $(seq 1 "$cutoff"); do
        rm -f "/tmp/tv2_eval_${c}.jsonl" "/tmp/tv2_eval_${c}.txt" \
              "/tmp/tv2_teach_${c}.txt" "/tmp/tv2_heldout_${c}.jsonl"
    done
}

# ── boot ────────────────────────────────────────────────────────────────────
log "════ train_v2 START (max ${MAX_CYCLES} cycles) ════"
CYCLE=0
ensure_engines
ensure_brain

log "measuring initial held-out baseline..."
BASE=$(measure_heldout)
BEST_SCORE=$(echo "$BASE" | awk '{print $1}')
BEST_SOLO=$(echo "$BASE" | awk '{print $2}')
[ -z "$BEST_SCORE" ] && BEST_SCORE=0
[ -z "$BEST_SOLO" ] && BEST_SOLO=0
cp -f "$BRAIN_BIN" "$BEST_BIN" 2>/dev/null
log "baseline held-out: score=${BEST_SCORE}% solo=${BEST_SOLO}%  (saved brain_best.bin)"

FAILS=0
STALL=0

while [ "$CYCLE" -lt "$MAX_CYCLES" ]; do
    CYCLE=$((CYCLE + 1))

    [ -f "$STOPFILE" ] && { log "stop file — finishing"; break; }

    FREE=$(free_mb)
    if [ "${FREE:-0}" -lt "$MIN_FREE_MB" ]; then
        log "RAM low (${FREE}MB < ${MIN_FREE_MB}MB) — stopping to protect machine"
        break
    fi

    log "── cycle $CYCLE/$MAX_CYCLES (free RAM ${FREE}MB, best ${BEST_SCORE}%) ──"
    ensure_engines
    ensure_brain

    EVAL_JSONL="/tmp/tv2_eval_${CYCLE}.jsonl"
    EVAL_TXT="/tmp/tv2_eval_${CYCLE}.txt"
    TEACH="$BLAB/data/auto_learn.txt"

    # snapshot before this cycle touches anything
    cp -f "$BRAIN_BIN" "/tmp/tv2_prev_brain.bin" 2>/dev/null

    # 1. purge noise BEFORE teaching (clean slate)
    PURGE=$(curl -s -X POST http://localhost:3012/purge_noise --max-time 120 2>/dev/null)
    log "cycle $CYCLE purge: ${PURGE:-no-response}"

    # 2. training battery (parallel, 2 engines, 30min ceiling)
    timeout 1800 python3 scripts/auto_eval.py --per "$PER_DOMAIN" --rounds 1 \
        --log "$EVAL_JSONL" > "$EVAL_TXT" 2>&1
    SCORE=$(grep "weighted score" "$EVAL_TXT" 2>/dev/null | sed 's/.*weighted score: //')
    if [ -z "$SCORE" ] || [ ! -s "$EVAL_JSONL" ]; then
        FAILS=$((FAILS + 1))
        log "cycle $CYCLE battery FAILED (${FAILS}/${MAX_FAILS})"
        [ "$FAILS" -ge "$MAX_FAILS" ] && { log "too many failures — stopping"; break; }
        sleep 60; continue
    fi
    FAILS=0
    log "cycle $CYCLE battery: $SCORE"

    # 3. validated teach file (teacher self-checks each answer)
    timeout 1200 python3 scripts/analyze_failures.py "$EVAL_JSONL" --teach "$TEACH" \
        > "/tmp/tv2_teach_${CYCLE}.txt" 2>&1
    PAIRS=$(grep -c "^USER:" "$TEACH" 2>/dev/null || echo 0)
    log "cycle $CYCLE teach: $PAIRS validated pairs"

    # 4. brain learns + purge again + checkpoint
    if [ -s "$TEACH" ]; then
        LEARN=$(curl -s -X POST http://localhost:3012/learn --max-time 240 2>/dev/null)
        log "cycle $CYCLE learn: ${LEARN:-no-response}"
        curl -s -X POST http://localhost:3012/purge_noise --max-time 120 >/dev/null 2>&1
    fi
    curl -s -X POST http://localhost:3012/checkpoint --max-time 90 >/dev/null 2>&1

    # 5. HELD-OUT GATE — did this cycle actually help?
    RES=$(measure_heldout)
    CUR_SCORE=$(echo "$RES" | awk '{print $1}')
    CUR_SOLO=$(echo "$RES" | awk '{print $2}')
    [ -z "$CUR_SCORE" ] && CUR_SCORE=0
    [ -z "$CUR_SOLO" ] && CUR_SOLO=0
    log "cycle $CYCLE held-out: score=${CUR_SCORE}% solo=${CUR_SOLO}% (best ${BEST_SCORE}%)"

    IMPROVED=$(python3 -c "print(1 if float('$CUR_SCORE') >= float('$BEST_SCORE') else 0)" 2>/dev/null)
    REGRESSED=$(python3 -c "print(1 if float('$CUR_SCORE') < float('$BEST_SCORE') - $REGRESS_TOL else 0)" 2>/dev/null)

    if [ "$IMPROVED" = "1" ]; then
        BEST_SCORE=$CUR_SCORE
        BEST_SOLO=$CUR_SOLO
        cp -f "$BRAIN_BIN" "$BEST_BIN" 2>/dev/null
        STALL=0
        log "cycle $CYCLE ✓ KEPT — new best ${BEST_SCORE}% (brain_best.bin updated)"
    elif [ "$REGRESSED" = "1" ]; then
        # this cycle hurt the brain — roll back to best known checkpoint
        cp -f "$BEST_BIN" "$BRAIN_BIN" 2>/dev/null
        pkill -9 -f iara_brain_full 2>/dev/null; sleep 3
        ( setsid nice -n 10 ./target/release/iara_brain_full serve \
            >"$BLAB/data/brain_serve.log" 2>&1 < /dev/null & )
        for _ in $(seq 1 90); do
            curl -s --max-time 3 http://localhost:3012/stats 2>/dev/null | grep -q bytes && break
            sleep 3
        done
        STALL=$((STALL + 1))
        log "cycle $CYCLE ✗ ROLLED BACK (${CUR_SCORE}% < ${BEST_SCORE}%) — restored brain_best.bin"
    else
        STALL=$((STALL + 1))
        log "cycle $CYCLE ~ FLAT (${CUR_SCORE}%, within tolerance) — kept, stall ${STALL}/${STALL_LIMIT}"
    fi

    if [ "$STALL" -ge "$STALL_LIMIT" ]; then
        log "held-out stalled ${STALL} cycles — no learning, stopping"
        break
    fi

    cleanup_tmp
    sleep "$INTER_CYCLE_SLEEP"
done

# always leave the brain at its best known state
cp -f "$BEST_BIN" "$BRAIN_BIN" 2>/dev/null
log "════ train_v2 DONE after $CYCLE cycles — final best: score=${BEST_SCORE}% solo=${BEST_SOLO}% ════"
log "restart brain to load best checkpoint: pkill -9 -f iara_brain_full && ./scripts/_start_brain.sh"
