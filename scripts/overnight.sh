#!/bin/bash
# Overnight autonomous DATA+TRAIN loop. Each round:
#   1. discover repos (GitHub search API, cycling languages × pages)
#   2. shallow-clone a batch (nice/ionice, to DISK not tmpfs)
#   3. build a corpus (streaming, constant-RAM Rust corpus_builder)
#   4. DELETE the repos (free disk)
#   5. warm-start train one chunk on the corpus (GPU), save checkpoint
#   6. DELETE the corpus
#   → fetch more, forever (until TOTAL).
#
# Bounded disk + constant RAM + nice/ionice → never competes with your work.
# Self-healing: the checkpoint persists; a kill loses at most one chunk's tail.
#
#   nohup bash scripts/overnight.sh > /tmp/overnight.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
CKPT=data/manjou_ssm.ssmc
BUILDER=./target/release/corpus_builder
TRAINER=./target/release/train_ssm_gpu
WORK=data/.oc_work             # disk-backed (NOT /tmp, which may be tmpfs/RAM)
CORPUS=data/.oc_corpus.txt
TOTAL=${TOTAL:-36000}          # ~10h
CHUNK=${CHUNK:-1500}           # ~25min train per round
RPR=${RPR:-6}                  # repos per round
D=256; L=6; SEQ=256; BATCH=8; FF=512; LR=2.5e-3
LANGS=(rust python go typescript javascript c cpp java ruby kotlin)
LO="nice -n 19 ionice -c3"     # low priority for clone/build

start=$(date +%s); round=0
echo "=== overnight START $(date) | total=${TOTAL}s chunk=${CHUNK}s repos/round=$RPR ==="
while [ $(( $(date +%s) - start )) -lt "$TOTAL" ]; do
  round=$((round + 1))
  lang=${LANGS[$(( (round - 1) % ${#LANGS[@]} ))]}
  page=$(( (round - 1) / ${#LANGS[@]} + 1 ))
  echo ">>> ROUND $round | $(date +%H:%M) | lang=$lang page=$page | elapsed $(( ($(date +%s) - start) / 60 ))min"

  # 1. discover repos via GitHub API (unauth ~60/h; we do ~2/h)
  mapfile -t repos < <(curl -s --max-time 30 \
    "https://api.github.com/search/repositories?q=language:$lang+stars:%3E500&sort=stars&order=desc&per_page=$RPR&page=$page" \
    | grep '"full_name"' | sed -E 's/.*"full_name": *"([^"]+)".*/\1/' | head -"$RPR")
  if [ "${#repos[@]}" -eq 0 ]; then echo "  no repos (API rate?), sleep 300s"; sleep 300; continue; fi

  # 2. shallow-clone the batch to disk
  rm -rf "$WORK"; mkdir -p "$WORK"
  for r in "${repos[@]}"; do
    echo "  clone $r"
    timeout 120 $LO git clone --depth 1 -q "https://github.com/$r" "$WORK/$(basename "$r")" 2>/dev/null || echo "    (skip $r)"
  done

  # 3. build corpus (streaming, constant RAM, 1-byte separator)
  $LO "$BUILDER" "$CORPUS" rs,py,go,ts,tsx,js,c,cpp,h,java,rb,kt 400 "$WORK" 2>&1 | grep -E "size:|floor:|unique"

  # 4. DELETE the cloned repos (free disk immediately)
  rm -rf "$WORK"

  # 5. warm-start train one chunk on this corpus (GPU)
  if [ -s "$CORPUS" ]; then
    load=""; [ -f "$CKPT" ] && load="$CKPT"
    nice -n 5 "$TRAINER" "$CORPUS" "$CHUNK" "$D" "$L" "$SEQ" "$LR" "$BATCH" "$FF" "$CKPT" "$load" 2>&1 \
      | grep -E "held-out:|epochs|greedy"
  else
    echo "  empty corpus, skipping train"
  fi

  # 6. DELETE the corpus (free disk; next round fetches fresh code)
  rm -f "$CORPUS"
done
echo "=== overnight DONE $(date): $round rounds, model at $CKPT ==="
