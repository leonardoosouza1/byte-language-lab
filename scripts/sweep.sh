#!/bin/bash
# Scaling sweep: same compute budget per config (fair efficiency comparison).
# Logs bits/byte reached vs params → thesis curve (2) + efficiency frontier (3).
set -u
cd "$(dirname "$0")/.."
BIN=./target/release/train_transformer
CORPUS=data/corpus_local.txt
CSV=data/sweep_results.csv
BUDGET=${1:-360}     # seconds of GPU per config

echo "d_model,n_layers,params_M,budget_s,tok_s,bits_per_byte,floor,delta_vs_floor" > "$CSV"

run() {
  local d=$1 l=$2 budget=$3
  local log="/tmp/sweep_d${d}_l${l}.log"
  echo ">>> d=$d L=$l budget=${budget}s ..."
  "$BIN" "$CORPUS" "$budget" "$d" "$l" 128 0.002 1.0 adam 8 3.0 1 0.0 \
        "data/sweep_d${d}l${l}.ckpt" > "$log" 2>&1
  local params tps bpb floor delta
  params=$(grep -oE '~[0-9.]+M params' "$log" | head -1 | tr -dc '0-9.')
  tps=$(grep 'throughput:' "$log" | grep -oE '[0-9]+ tok/s' | grep -oE '[0-9]+' | head -1)
  bpb=$(grep '^held-out:' "$log" | grep -oE '[0-9.]+ bits' | grep -oE '[0-9.]+' | head -1)
  floor=$(grep 'unigram floor:' "$log" | grep -oE '[0-9.]+' | head -1)
  delta=$(awk "BEGIN{printf \"%.3f\", $floor-$bpb}")
  echo "$d,$l,$params,$budget,$tps,$bpb,$floor,$delta" >> "$CSV"
  echo "    -> ${bpb} bits/byte (${tps} tok/s, ${params}M params, Δfloor=${delta})"
}

run 128 4 "$BUDGET"
run 192 6 "$BUDGET"
run 256 6 "$BUDGET"
run 384 6 "$BUDGET"
run 512 8 "$((BUDGET + 120))"

echo "=== sweep done ==="
column -t -s, "$CSV"
