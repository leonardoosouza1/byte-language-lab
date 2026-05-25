#!/bin/bash
# Autonomous training flow: warm-start from the checkpoint, train a chunk, save,
# repeat — until TOTAL seconds elapse. Self-healing: each chunk advances the
# persistent checkpoint, so a kill loses at most the current chunk's tail (and
# the trainer also saves every 120s within a chunk). GPU-bound; nice for CPU so
# it never competes with your work.
#
#   nohup bash scripts/train_loop.sh > /tmp/train_loop.log 2>&1 &
#
# Env overrides: TOTAL, CHUNK, CORPUS, D, L, SEQ, BATCH, FF, LR
set -u
cd "$(dirname "$0")/.."
BIN=./target/release/train_ssm_gpu
CKPT=${CKPT:-data/manjou_ssm.ssmc}
CORPUS=${CORPUS:-data/corpus_code2.txt}
TOTAL=${TOTAL:-7200}      # total wall seconds (~2h)
CHUNK=${CHUNK:-1700}      # seconds per chunk (~28min) → ~4 chunks
D=${D:-256}; L=${L:-6}; SEQ=${SEQ:-256}; BATCH=${BATCH:-8}; FF=${FF:-512}; LR=${LR:-2.5e-3}

start=$(date +%s); round=0
echo "=== train_loop START: total=${TOTAL}s chunk=${CHUNK}s d=$D L=$L seq=$SEQ ff=$FF ckpt=$CKPT ==="
while [ $(( $(date +%s) - start )) -lt "$TOTAL" ]; do
  round=$((round + 1))
  load=""; [ -f "$CKPT" ] && load="$CKPT"
  echo ">>> ROUND $round | elapsed $(( $(date +%s) - start ))s | warm-start='${load:-FRESH}'"
  nice -n 5 "$BIN" "$CORPUS" "$CHUNK" "$D" "$L" "$SEQ" "$LR" "$BATCH" "$FF" "$CKPT" "$load" \
    2>&1 | grep -E "warm-start|held-out:|epochs|greedy|temp |checkpoint"
done
echo "=== train_loop DONE: $round rounds, model at $CKPT ==="
